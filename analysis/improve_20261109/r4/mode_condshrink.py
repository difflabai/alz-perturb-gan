"""Stage-path diagnosis and generation-only stage shrinkage on round-1 fits (F2/F3; no training).

Subcommands (CPU, 3 threads; reads only the 37 development donors' caches)::

    python3 -B r2/mode_condshrink.py measure
    python3 -B r2/mode_condshrink.py gen --fit FIT_DIR --cache CACHE_DIR --out OUT.npz
    python3 -B r2/mode_condshrink.py report

Outputs go to /ml2/alz/runs/improve_20261109/posthoc_r2/stage/.

measure (generated.npz of the complete round-1 base / genemean hvg2000 fits, seeds 0 and 1, folds 0-2)
    Per fold, arm, seed, with P_t = mean over the M generated donors of
    log1p CP10k(pb_counts[t,m]) (the evaluator's pseudobulk prediction) and
    O_t = log1p CP10k of donor t's observed pseudobulk:
      D_k[s] = mean_{train t of stage s} P_t - mean_{train t} P_t   (model, seed k)
      R[s]   = the same for O over train donors                     (real train)
      H[s]   = the same for O over held-out donors                  (real held-out)
    and, weighting stages by their train-donor share, the mean over genes of
    D_k^2, R^2, the seed-shared part <D_0 D_1> (what survives averaging over
    infinitely many seeds), the seed-specific part |D_0 - D_1|^2 / 2, the
    Monte Carlo part (var_m / M / n_s), the part of the shared component
    along R, the least-squares scale <D,R>/<D,D>, the sampling noise of R
    (pooled within-stage variance x (1/n_s - 1/n)), correlations with R and H;
    all also per quartile of train grand-mean expression.

gen (one fit; final state; CPU regeneration with the generation streams of r2/generate.py)
    Generates, for every target, nine plans through r2/generate.generate_arrays:
    ``stage0..stage3`` (every target conditioned on stage s; ADNC replaced in
    raw_c, then ctx.target_conditions, so for a target's own stage this is
    bitwise the CPU default generation), ``condshrink@{0.25,0.5,0.75,1}`` and
    ``condmix@0.5`` (the arm_condshrink.py modes via generate.mode_plans).
    Saves only P (pred, float32 [V,T,G]) and the M-averaged mean_log
    (float32 [V,T,G]) plus metadata (about 5 MB per fit).

report
    Held-out and train pb_rmse (RMSE over the panel's genes per donor, mean
    over donors; identical to evaluate.py 'all' subset pseudobulk rmse) of:
    default (CPU regeneration), condshrink@lam (lam grid), condmix@lam
    (analytic mixture of the stage grid: (1-lam) P_own + lam sum_s pi_s P_s,
    lam = 0..1 step 0.05; and the generated condmix@0.5 as a check), each per
    fit and as the two-seed ensemble (mean of the seeds' P).  lam is either
    fixed in advance (0.5, 1) or chosen per fold and fit from TRAIN targets
    only (argmin of train-target pb_rmse; in-sample, so it tends to
    under-shrink).  Held-out donors are never used to choose anything.
    Reference: evaluator baselines from results_screen48 (grand_mean,
    shrunk_stage_mean, ridge; read-only) and the round-1 evaluator values of
    the same fits (GPU generation).  Secondary (labelled posthoc): the same
    predictions after a train-only per-gene level recalibration
    (P - mean over train targets of (P - O)), to separate the stage part
    from the level bias.  Also repeats the measure decomposition on the
    stage grid (same targets and libraries at every stage), which removes
    target-composition effects from the stage deviations.
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True
if __name__ == '__main__':
    for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[_k] = '3'
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
    _here = os.path.dirname(os.path.abspath(__file__))
    if _here in sys.path:
        sys.path.remove(_here)
    sys.path.insert(0, _here)

import argparse
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
RUNS = Path('/ml2/alz/runs/improve_20261109')
OUT = RUNS / 'posthoc_r2' / 'stage'
FITS = RUNS / 'fits'
CACHES = RUNS / 'caches'
BASELINES = RUNS / 'results_screen48' / 'baselines'
R1_RESULTS = RUNS / 'results_screen48' / 'fits'
FOLDS_JSON = HERE.parent / 'folds.json'
ARMS = ('base', 'genemean')
SEEDS = (0, 1)
SHRINK_GRID = (0.25, 0.5, 0.75, 1.0)
MIX_GRID = tuple(np.round(np.arange(0, 1.0001, 0.05), 2))
VARIANTS = ('stage0', 'stage1', 'stage2', 'stage3') + tuple(f'condshrink@{l:g}' for l in SHRINK_GRID) + ('condmix@0.5',)


def log_cp10k(x):
    x = np.asarray(x, dtype=np.float64)
    return np.log1p(x * (1e4 / np.maximum(x.sum(-1, keepdims=True), 1)))


def dump(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name('.' + p.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1, allow_nan=False) + '\n')
    tmp.replace(p)


# --------------------------------------------------------------------------- real data
def load_obs(cache):
    """Observed pseudobulk per donor of a fold cache: (donors, is_heldout, stage, O [T,G] log1p CP10k)."""
    q = json.loads(FOLDS_JSON.read_text())
    dev, sealed = set(q['development_donors']), set(q['sealed_evaluation_donors_fold0'])
    man = json.loads((Path(cache) / 'manifest.json').read_text())
    ids = list(man['train_donors']) + list(man.get('heldout_donors', []))
    if not set(ids) <= dev or set(ids) & sealed:
        raise PermissionError('cache has donors outside the 37 development donors')
    donors, held, stage, pbs = [], [], [], []
    for split in ('train', 'heldout'):
        with np.load(Path(cache) / split / 'metadata.npz', allow_pickle=False) as m:
            cell = m['donor'].astype(str); dl = m['donors'].astype(str); raw = m['raw_c'].astype(np.float64)
        if not set(dl) <= dev or set(dl) & sealed or not set(cell) <= dev:
            raise PermissionError('split has donors outside the development donors')
        C = sp.load_npz(Path(cache) / split / 'counts.npz').tocsr()
        for j, d in enumerate(dl):
            rows = np.flatnonzero(cell == d)
            pbs.append(np.asarray(C[rows].sum(0)).ravel().astype(np.float64))
            donors.append(d); held.append(split == 'heldout'); stage.append(int(round(raw[j, 3] * 3)))
    return np.array(donors), np.array(held), np.array(stage), log_cp10k(np.stack(pbs))


def gen_pred(path, donors):
    with np.load(path, allow_pickle=False) as f:
        tg = f['targets'].astype(str)
        order = np.array([list(tg).index(d) for d in donors])
        pb = f['pb_counts'][order].astype(np.float64)
    P = np.stack([log_cp10k(pb[t]).mean(0) for t in range(len(pb))])
    V = np.stack([log_cp10k(pb[t]).var(0) / pb.shape[1] for t in range(len(pb))])
    return P, V


# --------------------------------------------------------------------------- decomposition
def stage_devs(P, mask, stage):
    c = P[mask].mean(0)
    return {s: P[mask & (stage == s)].mean(0) - c for s in range(4) if (mask & (stage == s)).any()}


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else float('nan')


def decompose(D0, D1, R, H, wts, noiseR, mc, quart):
    """Stage-weighted decomposition; D0/D1/R/H: {s: [G]}; wts: {s: n_s/n}; noiseR/mc: {s: [G]}."""
    out = {}
    groups = {'all': np.ones_like(quart, bool)}
    groups.update({f'q{q + 1}': quart == q for q in range(4)})
    for gname, gm in groups.items():
        acc = {k: 0.0 for k in ('ms_model', 'ms_real_train', 'noise_real_train', 'shared', 'seed_specific', 'mc',
                                 'shared_along_real', 'cross_model_real')}
        for s, w in wts.items():
            d0, d1, r = D0[s][gm], D1[s][gm], R[s][gm]
            rr = float((r * r).mean())
            acc['ms_model'] += w * 0.5 * float((d0 * d0).mean() + (d1 * d1).mean())
            acc['ms_real_train'] += w * rr
            acc['noise_real_train'] += w * float(noiseR[s][gm].mean())
            acc['shared'] += w * float((d0 * d1).mean())
            acc['seed_specific'] += w * 0.5 * float(((d0 - d1) ** 2).mean())
            acc['mc'] += w * float(mc[s][gm].mean())
            a0, a1 = float((d0 * r).mean()) / rr, float((d1 * r).mean()) / rr
            acc['shared_along_real'] += w * a0 * a1 * rr
            acc['cross_model_real'] += w * 0.5 * float((d0 * r).mean() + (d1 * r).mean())
        acc['inflation_single_seed'] = acc['ms_model'] / acc['ms_real_train']
        acc['inflation_infinite_seeds'] = acc['shared'] / acc['ms_real_train']
        acc['fraction_seed_specific'] = acc['seed_specific'] / acc['ms_model']
        acc['fraction_shared'] = acc['shared'] / acc['ms_model']
        acc['fraction_shared_orthogonal_to_real'] = (acc['shared'] - acc['shared_along_real']) / acc['ms_model']
        acc['ls_scale_single_seed'] = acc['cross_model_real'] / acc['ms_model']
        acc['ls_scale_infinite_seeds'] = acc['cross_model_real'] / acc['shared'] if acc['shared'] > 0 else float('nan')
        acc['real_train_signal'] = acc['ms_real_train'] - acc['noise_real_train']
        out[gname] = acc
    per_stage = {}
    for s in wts:
        dm = 0.5 * (D0[s] + D1[s])
        per_stage[str(s)] = {'weight': wts[s], 'corr_seed0_seed1': corr(D0[s], D1[s]),
                             'corr_model_real_train': 0.5 * (corr(D0[s], R[s]) + corr(D1[s], R[s])),
                             'corr_seedmean_real_train': corr(dm, R[s]),
                             'corr_seedmean_real_heldout': corr(dm, H[s]) if s in H else None,
                             'corr_real_train_real_heldout': corr(R[s], H[s]) if s in H else None,
                             'var_ratio_single': 0.5 * float((D0[s] ** 2).mean() + (D1[s] ** 2).mean()) / float((R[s] ** 2).mean()),
                             'ls_scale_single': 0.5 * (float((D0[s] * R[s]).sum() / (D0[s] ** 2).sum()) + float((D1[s] * R[s]).sum() / (D1[s] ** 2).sum()))}
    out['per_stage'] = per_stage
    return out


def real_noise(O, mask, stage):
    st = stage[mask]; X = O[mask]; n = len(X)
    present = [s for s in range(4) if (st == s).any()]
    ss = sum(((X[st == s] - X[st == s].mean(0)) ** 2).sum(0) for s in present)
    v = ss / (n - len(present))
    return {s: v * (1.0 / (st == s).sum() - 1.0 / n) for s in present}


def fold_measure(panel, fold, preds):
    """preds: {(arm, seed): (P, V)}; returns decomposition per arm."""
    donors, held, stage, O = load_obs(CACHES / panel / f'fold{fold}')
    tr = ~held
    Lbar = O[tr].mean(0)
    quart = np.searchsorted(np.quantile(Lbar, [.25, .5, .75]), Lbar)
    R = stage_devs(O, tr, stage); H = stage_devs(O, held, stage)
    wts = {s: float((stage[tr] == s).mean()) for s in R}
    nR = real_noise(O, tr, stage)
    res = {}
    for arm in ARMS:
        if (arm, 0) not in preds or (arm, 1) not in preds:
            continue
        (P0, V0), (P1, V1) = preds[(arm, 0)], preds[(arm, 1)]
        D0, D1 = stage_devs(P0, tr, stage), stage_devs(P1, tr, stage)
        mc = {s: 0.5 * (V0[tr & (stage == s)].sum(0) + V1[tr & (stage == s)].sum(0)) / (tr & (stage == s)).sum() ** 2 for s in R}
        res[arm] = decompose(D0, D1, R, H, wts, nR, mc, quart)
    res['n_train_per_stage'] = {str(s): int((stage[tr] == s).sum()) for s in range(4)}
    res['n_heldout_per_stage'] = {str(s): int((stage[held] == s).sum()) for s in range(4)}
    return res, (donors, held, stage, O)


def cmd_measure(args):
    out = {}
    for fold in range(3):
        donors = load_obs(CACHES / 'hvg2000' / f'fold{fold}')[0]
        preds = {}
        for arm in ARMS:
            for s in SEEDS:
                p = FITS / arm / 'hvg2000' / f'fold{fold}' / f'seed{s}' / 'generated.npz'
                if p.exists() and (p.parent / 'final.pt').exists():
                    preds[(arm, s)] = gen_pred(p, donors)
        out[f'hvg2000/fold{fold}'] = fold_measure('hvg2000', fold, preds)[0]
        print('measured fold', fold, flush=True)
    out['_doc'] = ('Round-1 GPU generated.npz; D = model stage deviation over train targets, R = real train, H = real held-out; '
                   'stage-weighted mean over genes; see r2/mode_condshrink.py docstring')
    dump(OUT / 'measure.json', out)
    summarize_measure(out)


def summarize_measure(out):
    for k, v in out.items():
        if not k.startswith('hvg'):
            continue
        for arm in ARMS:
            if arm not in v:
                continue
            a = v[arm]['all']
            print(f"{k} {arm}: ms_model {a['ms_model']:.4f} real {a['ms_real_train']:.4f} (noise {a['noise_real_train']:.4f}) "
                  f"infl1 {a['inflation_single_seed']:.2f} inflInf {a['inflation_infinite_seeds']:.2f} seedspec {a['fraction_seed_specific']:.2f} "
                  f"shared_orth {a['fraction_shared_orthogonal_to_real']:.2f} mc {a['mc']:.5f} ls1 {a['ls_scale_single_seed']:.2f} lsInf {a['ls_scale_infinite_seeds']:.2f}")
            print('   quartiles infl1', [round(v[arm][f'q{q}']['inflation_single_seed'], 2) for q in range(1, 5)],
                  'inflInf', [round(v[arm][f'q{q}']['inflation_infinite_seeds'], 2) for q in range(1, 5)],
                  'seedspec', [round(v[arm][f'q{q}']['fraction_seed_specific'], 2) for q in range(1, 5)])


# --------------------------------------------------------------------------- generation
def cmd_gen(args):
    import torch
    import trainer
    import generate
    import arm_condshrink  # noqa: F401  (registers the modes)
    torch.set_num_threads(3)
    fit = Path(args.fit)
    out = Path(args.out)
    if FITS.resolve() in out.resolve().parents or fit.resolve() in out.resolve().parents:
        raise ValueError('output must be outside the round-1 fit tree')
    config = json.loads((fit / 'config.json').read_text())
    ctx = trainer.Context(args.cache, 'cpu', tuple(config['flags']), allow_nonprotocol_split=not config['protocol_split'])
    if trainer.sha(ctx.cache / 'manifest.json') != config['cache_manifest_SHA256']:
        raise ValueError('cache differs from the fit')
    align_genemean_scale(ctx, fit)
    T = generate.targets(ctx)
    raws = np.stack([t[2] for t in T]); heldm = np.array([t[1] for t in T], bool)
    models = generate.load_models(ctx, fit, config, ['final'])
    res = {'pred': [], 'ml': [], 'lib': []}
    t0 = time.monotonic()
    for v in VARIANTS:
        if v.startswith('stage'):
            s = int(v[5:])
            r = raws.copy(); r[:, 3] = s / 3.0
            conds = ctx.target_conditions(r, heldm)
            plans = [(np.repeat(conds[i][None], generate.M, 0), ['final'] * generate.M) for i in range(len(T))]
        else:
            plans, _, _ = generate.mode_plans(ctx, v, T, ('final',))
        arr, _ = generate.generate_arrays(ctx, models, plans, T=T)
        pb = arr['pb_counts'].astype(np.float64)
        res['pred'].append(np.stack([log_cp10k(pb[t]).mean(0) for t in range(len(T))]).astype(np.float32))
        res['ml'].append(arr['mean_log'].astype(np.float64).mean(1).astype(np.float32))
        res['lib'].append(arr['lib'].mean(1))
        print(f'[gen] {fit} {v} {time.monotonic() - t0:.0f}s', flush=True)
        del arr, pb
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name('.' + out.stem + '.partial.npz')
    np.savez_compressed(tmp, variants=np.array(VARIANTS), targets=np.array([t[0] for t in T]), is_heldout=heldm,
                        stage=arms_stage(raws), genes=np.asarray(ctx.genes).astype(str), cbar=np.float32(raws[~heldm, 3].mean()),
                        pred=np.stack(res['pred']), ml=np.stack(res['ml']), lib=np.stack(res['lib']),
                        fit=str(fit), seconds=time.monotonic() - t0)
    tmp.replace(out)
    print('[gen] wrote', out, flush=True)


GENEMEAN_SCALE_RTOL = 1e-5


def align_genemean_scale(ctx, fit):
    """Work-around for a core defect (reported): the round-1 genemean critic scale s_gm is a float32 sum
    computed on the training device; recomputed on CPU it differs in the last bits from the GPU value
    stored in final.pt, so generate.check_final refuses every round-1 GPU genemean fit on CPU.  s_gm only
    scales a critic input block, which generation never uses; if the CPU value agrees within
    GENEMEAN_SCALE_RTOL the stored value is adopted, otherwise the exact check still refuses."""
    import torch
    feats = ctx.features
    if feats.kind != 'genemean':
        return
    q = torch.load(Path(fit) / 'final.pt', map_location='cpu', weights_only=False)
    want = float(q['arm_state']['features']['genemean_scale'])
    del q
    mult = feats.info.get('genemean_multiplier', 1.0)
    if abs(feats.scale - want) <= GENEMEAN_SCALE_RTOL * abs(want):
        print(f'[gen] genemean scale CPU {feats.scale!r} vs stored {want!r}: adopting stored value', flush=True)
        feats.scale = want
        feats.info['genemean_scale'] = want / mult


def arms_stage(raws):
    return np.rint(raws[:, 3] * 3).astype(int)


# --------------------------------------------------------------------------- report
def pb_rmse(P, O, mask):
    return float(np.mean(np.sqrt(np.mean((P[mask] - O[mask]) ** 2, axis=1))))


def load_gen(path, donors):
    with np.load(path, allow_pickle=False) as f:
        tg = f['targets'].astype(str)
        order = np.array([list(tg).index(d) for d in donors])
        return {'variants': list(f['variants'].astype(str)), 'pred': f['pred'][:, order].astype(np.float64),
                'ml': f['ml'][:, order].astype(np.float64), 'stage': f['stage'][order]}


def predictions(g, pi):
    """Named predictions [T,G] from a gen file: default, condshrink grid, analytic condmix grid, generated condmix@0.5."""
    V = {v: g['pred'][i] for i, v in enumerate(g['variants'])}
    S = np.stack([V[f'stage{s}'] for s in range(4)])          # [4,T,G]
    st = g['stage']
    own = S[st, np.arange(len(st))]
    marg = np.tensordot(pi, S, axes=(0, 0))
    out = {'default': own}
    for l in SHRINK_GRID:
        out[f'condshrink@{l:g}'] = V[f'condshrink@{l:g}']
    out['condshrink@0'] = own
    for l in MIX_GRID:
        out[f'condmix@{l:g}'] = (1 - l) * own + l * marg
    out['condmix@0.5_generated'] = V['condmix@0.5']
    return out, S


def recal(P, O, tr):
    return P - (P[tr] - O[tr]).mean(0)


def cmd_report(args):
    rep = {'_doc': __doc__.split('report\n')[1].split('\n\n')[0].strip()}
    rows = {}
    grid_measure = {}
    for fold in range(3):
        donors, held, stage, O = load_obs(CACHES / 'hvg2000' / f'fold{fold}')
        tr = ~held
        pi = np.array([(stage[tr] == s).mean() for s in range(4)])
        base_ref = {}
        for b in ('grand_mean', 'shrunk_stage_mean', 'ridge', 'stage_mean'):
            d = json.loads((BASELINES / 'hvg2000' / f'fold{fold}' / f'{b}.json').read_text())
            base_ref[b] = {'heldout': d['subsets']['all']['heldout']['pseudobulk']['rmse'],
                           'train': d['subsets']['all']['train']['pseudobulk']['rmse']}
        Lbar = O[tr].mean(0)
        base_ref['train_grand_mean_profile_noisefree'] = {'heldout': pb_rmse(np.repeat(Lbar[None], len(O), 0), O, held),
                                                          'train': pb_rmse(np.repeat(Lbar[None], len(O), 0), O, tr)}
        fr = {'baselines': base_ref, 'train_stage_proportions': pi.tolist(), 'fits': {}}
        for arm in ARMS:
            preds = {}
            for s in SEEDS:
                p = OUT / 'gen' / f'{arm}_hvg2000_fold{fold}_seed{s}.npz'
                if not p.exists():
                    continue
                g = load_gen(p, donors)
                if not np.array_equal(g['stage'], stage):
                    raise ValueError('stage mismatch')
                preds[s], S = predictions(g, pi)
                # stage-grid decomposition material
                grid_measure.setdefault((fold, arm), {})[s] = S
            if not preds:
                continue
            units = {f'seed{s}': preds[s] for s in preds}
            if len(preds) == 2:
                units['ens2'] = {k: 0.5 * (preds[0][k] + preds[1][k]) for k in preds[0]}
            for uname, P in units.items():
                r = {}
                for k, v in P.items():
                    r[k] = {'heldout': pb_rmse(v, O, held), 'train': pb_rmse(v, O, tr),
                            'heldout_recal': pb_rmse(recal(v, O, tr), O, held), 'train_recal': pb_rmse(recal(v, O, tr), O, tr)}
                # train-only choices of lambda
                for fam, grid in (('condmix', MIX_GRID), ('condshrink', (0,) + SHRINK_GRID)):
                    for which in ('', '_recal'):
                        keys = [f'{fam}@{l:g}' for l in grid]
                        best = min(keys, key=lambda k: (r[k]['train' + which], k))
                        r[f'{fam}_trainchosen{which}'] = {'chosen': best, 'heldout' + which: r[best]['heldout' + which],
                                                         'train' + which: r[best]['train' + which]}
                r1 = R1_RESULTS / arm / 'hvg2000' / f'fold{fold}' / f'{uname}.json'
                if r1.exists():
                    d = json.loads(r1.read_text())['subsets']['all']
                    r['round1_evaluator_gpu'] = {'heldout': d['heldout']['pseudobulk']['rmse'], 'train': d['train']['pseudobulk']['rmse']}
                fr['fits'][f'{arm}/{uname}'] = r
        rows[f'hvg2000/fold{fold}'] = fr
        # stage-grid version of the decomposition (same targets at every stage)
        for arm in ARMS:
            gm = grid_measure.get((fold, arm), {})
            if len(gm) == 2:
                devs = {}
                for s_, S in gm.items():
                    marg = np.tensordot(pi, S[:, tr], axes=(0, 0)).mean(0)
                    devs[s_] = {s: S[s][tr].mean(0) - marg for s in range(4) if pi[s] > 0}
                R = stage_devs(O, tr, stage); H = stage_devs(O, held, stage)
                wts = {s: float(pi[s]) for s in R}
                quart = np.searchsorted(np.quantile(Lbar, [.25, .5, .75]), Lbar)
                zero = {s: np.zeros(O.shape[1]) for s in R}
                rows[f'hvg2000/fold{fold}'].setdefault('stage_grid_decomposition', {})[arm] = decompose(
                    devs[0], devs[1], R, H, wts, real_noise(O, tr, stage), zero, quart)
    rep['folds'] = rows
    rep['summary'] = summarize_report(rows)
    dump(OUT / 'report.json', rep)
    print(json.dumps(rep['summary'], indent=1))


def summarize_report(rows):
    """Mean over the three folds of held-out (and train) pb_rmse per arm/unit/prediction."""
    acc = {}
    for fk, fr in rows.items():
        for b, v in fr['baselines'].items():
            acc.setdefault(('baseline', b), []).append((v['heldout'], v['train'], None, None))
        for unit, r in fr['fits'].items():
            for k, v in r.items():
                if k.endswith('trainchosen') or k.endswith('trainchosen_recal'):
                    continue
                acc.setdefault((unit, k), []).append((v['heldout'], v['train'], v.get('heldout_recal'), v.get('train_recal')))
            for k in ('condmix_trainchosen', 'condshrink_trainchosen', 'condmix_trainchosen_recal', 'condshrink_trainchosen_recal'):
                v = r[k]
                w = '_recal' if k.endswith('_recal') else ''
                acc.setdefault((unit, k), []).append((v['heldout' + w], v['train' + w], v['chosen'], None))
    out = {}
    for (unit, k), vals in sorted(acc.items()):
        if len(vals) != 3:
            continue
        keep = k in ('default', 'round1_evaluator_gpu', 'condmix@0.5', 'condmix@1', 'condmix@0.5_generated', 'condshrink@0.25',
                     'condshrink@0.5', 'condshrink@0.75', 'condshrink@1') or 'trainchosen' in k or unit == 'baseline'
        if not keep:
            continue
        e = {'heldout_mean3folds': round(float(np.mean([v[0] for v in vals])), 4),
             'train_mean3folds': round(float(np.mean([v[1] for v in vals])), 4)}
        if 'trainchosen' in k:
            e['chosen_per_fold'] = [v[2] for v in vals]
        elif vals[0][2] is not None:
            e['heldout_recal_mean3folds'] = round(float(np.mean([v[2] for v in vals])), 4)
        out[f'{unit} | {k}'] = e
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description='stage-path diagnosis / generation-only stage shrinkage on round-1 fits')
    sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('measure')
    g = sub.add_parser('gen')
    g.add_argument('--fit', required=True)
    g.add_argument('--cache', required=True)
    g.add_argument('--out', required=True)
    sub.add_parser('report')
    a = p.parse_args(argv)
    {'measure': cmd_measure, 'gen': cmd_gen, 'report': cmd_report}[a.cmd](a)


if __name__ == '__main__':
    main()
