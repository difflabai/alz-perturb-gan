"""review_failures_state.py - learning curves and final.pt state, recomputed independently (CPU, map_location='cpu').

For every hvg2000 fit of the six arms:
  curves: mean over updates 15000-20000 of particle_covariance_trace, prior_regularization_unweighted,
          critic_gap, D_penalty; whole-curve non-finite count; max critic gap.
  final.pt: particle covariance trace of prior z [512,8]; generator first donor layer donor.0.weight [128, 8+7]:
          latent-driven pre-activation variance  tr(Wz Cov(z) Wz^T)  vs  condition-driven
          tr(Wc Cov(c_train) Wc^T)  where c_train are the arm's training condition vectors of the fold's
          train donors (arms.build_conditions semantics, recomputed here with numpy).
  offset arms: stage_offset (final) vs arm_state offset_init: RMS per stage, drift RMS, corr(drift, init),
          singular values of init/final/drift, and correlation of final rows with the UNSHRUNK donor-averaged
          stage log-ratio r_sg recomputed from train-donor counts.
Output: runs/.../screen_analysis/review_failures_state.json
"""
import json
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, '/ml2/alz/analysis/improve_20261109/screen_analysis')
import numpy as np
import torch

from review_failures_common import ARMS, CACHES, FITS, OUTDIR, load_real

torch.set_num_threads(6)


def curve(arm, k, s):
    R = [json.loads(l) for l in (FITS / arm / 'hvg2000' / f'fold{k}' / f'seed{s}' / 'learning_curve.jsonl').read_text().splitlines() if l.strip()]
    st = np.array([r['step'] for r in R])
    late = st >= 15000
    g = lambda f: np.array([np.nan if r.get(f) is None else float(r[f]) for r in R])
    pct = np.array([r['particle_usage']['particle_covariance_trace'] for r in R])
    gap = g('critic_gap')
    return {'n_rows': len(R), 'last_step': int(st.max()),
            'pcov_trace_late': float(pct[late].mean()), 'pcov_trace_first': float(pct[0]),
            'prior_reg_late': float(np.nanmean(g('prior_regularization_unweighted')[late])),
            'critic_gap_late': float(np.nanmean(gap[late])), 'D_penalty_late': float(np.nanmean(g('D_penalty')[late])),
            'G_adv_late': float(np.nanmean(g('G_adversarial')[late])),
            'nonfinite': int((~np.isfinite(gap)).sum() + (~np.isfinite(g('D_adversarial'))).sum()),
            'gap_max': float(np.nanmax(gap))}


def conds(raw, flags, am, asd):
    c = np.zeros_like(raw)
    c[:, 3] = raw[:, 3]
    if 'covars' in flags:
        c[:, 4] = raw[:, 4]; c[:, 5] = raw[:, 5]; c[:, 6] = (raw[:, 6] - am) / asd
    return c


def r_sg(R):
    tr = ~R['held']
    C = R['pb'][tr]; st = R['stage'][tr]
    N = C.sum(1); G = C.shape[1]
    logq = np.log(C.sum(0) + 1) - np.log(N.sum() + G)
    r = np.log(C + 1) - np.log(N[:, None] + G) - logq
    m = r.mean(0)
    return np.stack([r[st == s].mean(0) - m if (st == s).any() else np.zeros(G) for s in range(4)])


def main():
    out = {}
    for k in range(3):
        R = load_real(k)
        tr = ~R['held']
        raw = R['raw'][tr]
        age_m, age_sd = raw[:, 6].mean(), raw[:, 6].std()
        rs = r_sg(R)
        for arm in ARMS:
            flags = arm.split(',')
            ct = conds(raw, flags, age_m, age_sd)
            Cc = np.cov(ct.T)
            for s in range(2):
                e = curve(arm, k, s)
                q = torch.load(FITS / arm / 'hvg2000' / f'fold{k}' / f'seed{s}' / 'final.pt', map_location='cpu', weights_only=False)
                m = q['policy']['models']
                z = m['prior']['z'].double().numpy()
                Cz = np.cov(z.T)
                W = m['generator']['donor.0.weight'].double().numpy()
                Wz, Wc = W[:, :8], W[:, 8:]
                e['pcov_trace_final'] = float(np.trace(Cz))
                e['latent_preact_var'] = float(np.trace(Wz @ Cz @ Wz.T))
                e['cond_preact_var'] = float(np.trace(Wc @ Cc @ Wc.T))
                e['cond_share'] = e['cond_preact_var'] / (e['cond_preact_var'] + e['latent_preact_var'])
                e['Wz_fro'] = float(np.linalg.norm(Wz)); e['Wc_fro'] = float(np.linalg.norm(Wc))
                if 'offset' in flags:
                    fin = m['generator']['stage_offset'].double().numpy()
                    ini = q['arm_state']['offset_init'].double().numpy()
                    d = fin - ini
                    pres = [s_ for s_ in range(4) if R['stage'][tr].tolist().count(s_)]
                    e['offset'] = {'tau2': q['arm_state']['offset_info']['tau2'],
                                   'init_rms_stage': np.sqrt((ini ** 2).mean(1)).tolist(),
                                   'final_rms_stage': np.sqrt((fin ** 2).mean(1)).tolist(),
                                   'drift_rms_stage': np.sqrt((d ** 2).mean(1)).tolist(),
                                   'drift_over_init_rms': float(np.sqrt((d ** 2).mean()) / max(np.sqrt((ini ** 2).mean()), 1e-12)),
                                   'corr_drift_init': float(np.corrcoef(d.ravel(), ini.ravel())[0, 1]) if ini.std() > 0 else None,
                                   'sv_init': np.linalg.svd(ini, compute_uv=False).tolist(),
                                   'sv_final': np.linalg.svd(fin, compute_uv=False).tolist(),
                                   'sv_drift': np.linalg.svd(d, compute_uv=False).tolist(),
                                   'corr_final_row_vs_raw_rsg': [float(np.corrcoef(fin[s_] - fin[s_].mean(), rs[s_])[0, 1]) for s_ in pres],
                                   'corr_init_row_vs_raw_rsg': [float(np.corrcoef(ini[s_], rs[s_])[0, 1]) if ini[s_].std() > 0 else None for s_ in pres],
                                   'raw_rsg_rms': np.sqrt((rs ** 2).mean(1)).tolist(),
                                   'row_mean_final': fin.mean(1).tolist()}
                out[f'{arm}|{k}|{s}'] = e
                print(arm, k, s, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in e.items() if a != 'offset'}, flush=True)
                if 'offset' in e:
                    print('   offset', json.dumps({a: (np.round(b, 4).tolist() if isinstance(b, list) else b) for a, b in e['offset'].items()}), flush=True)
    summ = {}
    for arm in ARMS:
        summ[arm] = {}
        for key in ['pcov_trace_late', 'pcov_trace_final', 'prior_reg_late', 'critic_gap_late', 'D_penalty_late', 'G_adv_late',
                    'latent_preact_var', 'cond_preact_var', 'cond_share', 'nonfinite', 'gap_max']:
            v = [out[f'{arm}|{k}|{s}'][key] for k in range(3) for s in range(2)]
            summ[arm][key] = {'mean': float(np.mean(v)), 'min': float(np.min(v)), 'max': float(np.max(v))}
        print(arm.ljust(24), {a: round(b['mean'], 3) for a, b in summ[arm].items()})
    (OUTDIR / 'review_failures_state.json').write_text(json.dumps({'per_fit': out, 'summary': summ}, indent=1))


if __name__ == '__main__':
    main()
