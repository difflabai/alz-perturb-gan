"""Summaries for the declared comparison (PROTOCOL.md).

synthetic: paired adapter-native differences per width with seed bootstrap.
alz:       frozen architecture_score rows + paired held-out donor bootstrap of
           SD/combined coverage and interval score, references, residual traces.
"""
import os
os.environ.update(OMP_NUM_THREADS='3', OPENBLAS_NUM_THREADS='3', MKL_NUM_THREADS='3')
import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path('/ml2/alz')
OUT = ROOT / 'runs/research_20261120/cond-residual'
R5 = ROOT / 'analysis/improve_20261109/r5'
METRICS = ('mean_rmse', 'sd_rmse', 'logsd_rmse', 'logsd_bias', 'coverage80', 'width80', 'interval_score')
HIST = ('pb_rmse', 'meanlog_rmse_gene', 'nonzero_rmse_gene', 'logvar_rmse_gene', 'pc_energy_rep', 'cell_energy_rep')


def boot_mean(x, n=2000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    b = x[rng.integers(len(x), size=(n, len(x)))].mean(1)
    return [float(np.quantile(b, .025)), float(np.quantile(b, .975))]


def synthetic():
    out = {}
    for w in (2150, 128):
        rows = {}
        for arm in ('native', 'adapter'):
            for s in range(4):
                p = OUT / f'synthetic/w{w}_{arm}_s{s}.json'
                if p.exists():
                    rows[(arm, s)] = json.loads(p.read_text())
        seeds = sorted(s for s in range(4) if ('native', s) in rows and ('adapter', s) in rows)
        if not seeds:
            continue
        res = {'seeds': seeds, 'metrics': {}}
        for m in METRICS:
            n = np.array([rows[('native', s)]['score']['informative'][m] for s in seeds])
            a = np.array([rows[('adapter', s)]['score']['informative'][m] for s in seeds])
            d = a - n
            res['metrics'][m] = dict(native=n.tolist(), adapter=a.tolist(), native_mean=float(n.mean()),
                                     adapter_mean=float(a.mean()), diff_mean=float(d.mean()),
                                     rel_diff=float(d.mean() / n.mean()) if n.mean() else None,
                                     seed_bootstrap_95=boot_mean(d), n_lower=int((d < 0).sum()))
        n_cov = np.array([rows[('native', s)]['score']['informative']['coverage80'] for s in seeds])
        a_cov = np.array([rows[('adapter', s)]['score']['informative']['coverage80'] for s in seeds])
        res['metrics']['abs_coverage_gap'] = dict(native_mean=float(np.abs(n_cov - .8).mean()),
                                                  adapter_mean=float(np.abs(a_cov - .8).mean()))
        res['final_ema_residual'] = {arm: [rows[(arm, s)]['curve'][-1]['ema_residual'] for s in seeds]
                                     for arm in ('native', 'adapter')}
        res['per_group_logsd_bias'] = {arm: np.mean([rows[(arm, s)]['score']['informative']['per_group_logsd_bias']
                                                     for s in seeds], 0).tolist() for arm in ('native', 'adapter')}
        ctl = OUT / f'synthetic/controls_w{w}.json'
        if ctl.exists():
            res['controls'] = {k: {m: v['informative'][m] for m in METRICS}
                               for k, v in json.loads(ctl.read_text()).items()}
        if w == 2150 and len(seeds) == 4:
            ls = res['metrics']['logsd_rmse']; isc = res['metrics']['interval_score']
            res['helps_gate'] = dict(logsd_rel=ls['rel_diff'], seeds_improved=ls['n_lower'],
                                     interval_rel=isc['rel_diff'],
                                     passed=bool(ls['rel_diff'] <= -.05 and ls['n_lower'] >= 3 and isc['rel_diff'] <= .01))
        out[f'width{w}'] = res
    return out


def _alz_env():
    sys.path.insert(0, str(R5))
    import evaluate as E
    import architecture_score as A
    from interval_metrics import bounds, interval_score
    return E, A, bounds, interval_score


def per_donor(path, cache, E, A, bounds, interval_score):
    real = E.load_real(cache)
    gen = E.align_generated(E.load_generated(path), real)
    Y = E.MX.pc_bag_summary(gen['pcs'].astype(np.float32))
    actual = A.observed_summaries(real, gen['pcs'].shape[2])
    lo, hi = bounds(Y, 'percentile')
    mask = real.is_heldout
    out = {}
    for name, sl in (('sd', slice(50, 100)), ('combined', slice(0, 100))):
        y = actual[:, mask, sl]; l, h = lo[mask, sl], hi[mask, sl]
        out[name] = dict(coverage=((y >= l) & (y <= h)).mean(axis=(0, 2)),
                         interval_score=interval_score(l, h, y).mean(axis=(0, 2)),
                         width=np.broadcast_to(h - l, y.shape).mean(axis=(0, 2)))
    return np.asarray(real.donors)[mask], out


def residual_trace(lane, fold):
    p = OUT / f'fits/{lane}/pbulk/hvg2000/fold{fold}/seed0/learning_curve.jsonl'
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return [(r['step'], r['penalty_stats']['ema_residual']) for r in rows if r.get('penalty_stats')]


def alz(folds):
    E, A, bounds, isc = _alz_env()
    out = {}
    for k in folds:
        cache = ROOT / f'runs/improve_20261109/caches/hvg2000/fold{k}'
        rows = {lane: json.loads((OUT / f'results/{lane}_fold{k}_seed0.json').read_text()) for lane in ('native', 'adapter')}
        donors = {}
        per = {}
        for lane in ('native', 'adapter'):
            donors[lane], per[lane] = per_donor(OUT / f'generated/{lane}_fold{k}_seed0.npz', cache, E, A, bounds, isc)
        assert list(donors['native']) == list(donors['adapter'])
        res = {'heldout_donors': len(donors['native']), 'scores': {}, 'historical': {}, 'bootstrap': {}}
        for lane, r in rows.items():
            res['scores'][lane] = {k2: r['intervals']['heldout']['percentile'][k2] for k2 in ('sd', 'combined', 'mean')}
            res['historical'][lane] = {m: r['historical'][m] for m in HIST}
        for name in ('sd', 'combined'):
            for m in ('coverage', 'interval_score', 'width'):
                d = per['adapter'][name][m] - per['native'][name][m]
                res['bootstrap'][f'{name}_{m}'] = dict(diff=float(d.mean()), donor_bootstrap_95=boot_mean(d, seed=k),
                                                       donors_adapter_higher=int((d > 0).sum()))
        res['historical_rel_diff'] = {m: (res['historical']['adapter'][m] - res['historical']['native'][m]) /
                                      res['historical']['native'][m] for m in HIST}
        emp = ROOT / f'runs/improve_20261109/results_r5/empirical_challenge_20261114/empirical32/fold{k}/seed0.json'
        if emp.exists():
            v = json.loads(emp.read_text())
            res['reference_stage_blind_empirical'] = dict(path=str(emp), heldout={k2: v['intervals']['heldout']['percentile'][k2]
                                                          for k2 in ('sd', 'combined')}, historical={m: v['historical'][m] for m in HIST})
        if k == 0:
            pil = ROOT / 'runs/empf_comparison_20261117/results/native.json'
            if pil.exists():
                v = json.loads(pil.read_text())
                res['reference_native_20k_pilot'] = dict(path=str(pil), heldout={k2: v['intervals']['heldout']['percentile'][k2]
                                                         for k2 in ('sd', 'combined')}, historical={m: v['historical'][m] for m in HIST})
        tr = {lane: residual_trace(lane, k) for lane in ('native', 'adapter')}
        res['ema_residual'] = {lane: dict(min=min(v for _, v in t), max=max(v for _, v in t),
                                          last=t[-1][1], mean_after_1000=float(np.mean([v for s, v in t if s > 1000])))
                               for lane, t in tr.items()}
        res['ema_residual']['gauge_coefficient_mean_after_1000'] = {
            lane: .05 + .10 * (1 - min(1., res['ema_residual'][lane]['mean_after_1000'])) for lane in tr}
        res['ema_residual']['settle_engaged_any'] = {lane: any(v < .16 for _, v in t) for lane, t in tr.items()}
        cov = res['bootstrap']['sd_coverage']
        sdis = res['bootstrap']['sd_interval_score']
        res['supports_blind_spot_driving_sd_undercoverage'] = bool(cov['diff'] >= .02 and cov['donor_bootstrap_95'][0] > 0
                                                                     and sdis['diff'] <= 0)
        out[f'fold{k}'] = res
    return out


def determinism(fold=0, upto=5000):
    pil = ROOT / f'runs/empf_comparison_20261117/fits/native/pbulk/hvg2000/fold{fold}/seed0/learning_curve.jsonl'
    new = OUT / f'fits/native/pbulk/hvg2000/fold{fold}/seed0/learning_curve.jsonl'
    keys = ('D_adversarial', 'D_penalty', 'G_adversarial', 'critic_gap')
    a = {r['step']: r for r in map(json.loads, pil.read_text().splitlines()) if r['step'] <= upto}
    b = {r['step']: r for r in map(json.loads, new.read_text().splitlines()) if r['step'] <= upto}
    common = sorted(set(a) & set(b))
    diff = max(abs(a[s][k] - b[s][k]) for s in common for k in keys)
    return dict(steps_compared=len(common), max_abs_diff=diff, identical=diff == 0)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('part', choices=('synthetic', 'alz', 'determinism'))
    p.add_argument('--folds', default='0')
    p.add_argument('--out')
    a = p.parse_args()
    v = synthetic() if a.part == 'synthetic' else determinism() if a.part == 'determinism' else \
        alz([int(x) for x in a.folds.split(',')])
    text = json.dumps(v, indent=1, allow_nan=False)
    if a.out:
        Path(a.out).write_text(text + '\n')
    print(text)
