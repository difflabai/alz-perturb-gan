"""failures_covars.py - why does the covars arm (and covars,genemean,offset) fail on held-out donors?

For every hvg2000 fit of covars, covars,genemean,offset and base (3 folds x 2 seeds), regenerate on
CPU (failures_common.gen_pb, M=32 x K=256, depth-matched libraries from the target's own cells,
common random numbers across conditions) the 37 targets under counterfactual covariates:

  own        the target's own covariates (what generate.py does; reproduces evaluator pb_rmse)
  mix        each of the M generated donors gets the covariates (Male, APOE4, age z) of a random
             TRAIN donor of the target's stage, excluding the target itself (covariates marginalised
             over the training distribution: a stage-only prediction from the covariate model)
  trainmean  Male, APOE4 = train means, age z = 0 (an "average" covariate vector never seen in training)
  nn         covariates of the nearest other same-stage train donor in (Male, APOE4, age z) space

and records per target: pb_rmse (evaluator definition), mean gene bias (pred - obs), and the
"identity": which train donor's real profile (deviation from train grand mean, Pearson) the predicted
profile is closest to.  base is run with 'own' only (stage-only condition) as the reference.

Writes runs/.../screen_analysis/failures_covars.json and prints a summary.
Run: OMP_NUM_THREADS=6 python3 -B failures_covars.py
"""
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np

import failures_common as fc
import arms

OUT = Path('/ml2/alz/runs/improve_20261109/screen_analysis/failures_covars.json')
COV_ARMS = ['covars', 'covars,genemean,offset']


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def run_fit(arm, k, s, rf, obs, ctx_cache):
    ctx, G, D, P, q, cfg = fc.load_fit(arm, k, s, ctx=ctx_cache.get(arm))
    ctx_cache[arm] = ctx
    flags = ctx.flags
    raw = rf.raw_c
    tr = np.where(~rf.is_heldout)[0]
    grand = obs[tr].mean(0)
    own = arms.build_conditions(raw, flags, ctx.age_mean, ctx.age_sd)
    cov_space = arms.build_conditions(raw, ('covars',), ctx.age_mean, ctx.age_sd)[:, 4:7]  # Male, APOE, age z
    res = []
    for i in range(len(rf.donors)):
        seed = 7_000_000 + 1000 * k + i  # CRN: same across arms, seeds and conditions
        pool = rf.libs[i]
        same = [j for j in tr if rf.stage[j] == rf.stage[i] and j != i]
        conds = {'own': own[i]}
        if 'covars' in flags:
            rng = np.random.default_rng(seed)
            pick = rng.choice(same, size=32, replace=True)
            mix = np.repeat(own[i][None], 32, 0)
            mix[:, 4:7] = cov_space[pick]
            conds['mix'] = mix
            tm = own[i].copy(); tm[4] = cov_space[tr, 0].mean(); tm[5] = cov_space[tr, 1].mean(); tm[6] = 0.0
            conds['trainmean'] = tm
            dist = np.sqrt(((cov_space[same] - cov_space[i]) ** 2).sum(1))
            nn = same[int(np.argmin(dist))]
            c = own[i].copy(); c[4:7] = cov_space[nn]
            conds['nn'] = c
            nn_dist = float(dist.min())
        else:
            nn = None; nn_dist = None
        row = {'donor': str(rf.donors[i]), 'heldout': bool(rf.is_heldout[i]), 'stage': int(rf.stage[i]),
               'cov': cov_space[i].tolist(), 'nn_train_donor': None if nn is None else str(rf.donors[nn]),
               'nn_cov_dist': nn_dist, 'cond': {}}
        for name, c in conds.items():
            pb = fc.gen_pb(G, P, c, pool, seed)
            p = fc.pred_profile(pb)
            dev = p - grand
            cors = np.array([corr(dev, obs[j] - grand) for j in tr])
            best = tr[int(np.argmax(cors))]
            row['cond'][name] = {'pb_rmse': float(np.sqrt(np.mean((p - obs[i]) ** 2))),
                                 'bias': float(np.mean(p - obs[i])),
                                 'dev_corr_own': corr(dev, obs[i] - grand),
                                 'closest_train_donor': str(rf.donors[best]), 'closest_corr': float(cors.max()),
                                 'gen_donor_spread': float(np.sqrt(((fc.log_cp10k(pb) - p) ** 2).sum(1).mean()))}
        res.append(row)
    return res


def summarise(allres):
    print('\nfold-balanced mean pb_rmse (seeds averaged within fold), hvg2000')
    summ = {}
    for arm in ['base'] + COV_ARMS:
        for split in ['train', 'heldout']:
            for cond in ['own', 'mix', 'trainmean', 'nn']:
                vals = []
                for k in range(3):
                    v = [np.mean([r['cond'][cond]['pb_rmse'] for r in allres[f'{arm}|{k}|{s}'] if r['heldout'] == (split == 'heldout')])
                         for s in range(2) if cond in allres[f'{arm}|{k}|{s}'][0]['cond']]
                    if v:
                        vals.append(np.mean(v))
                if vals:
                    summ[f'{arm}|{split}|{cond}'] = float(np.mean(vals))
                    print(f'  {arm:24s} {split:8s} {cond:10s} pb_rmse {np.mean(vals):.3f}')
    print('\nidentity: fraction of TRAIN targets whose predicted profile is closest to their OWN real profile (own covariates)')
    for arm in ['base'] + COV_ARMS:
        f = [np.mean([r['cond']['own']['closest_train_donor'] == r['donor'] for r in allres[f'{arm}|{k}|{s}'] if not r['heldout']])
             for k in range(3) for s in range(2)]
        summ[f'{arm}|train_self_identity'] = float(np.mean(f))
        print(f'  {arm:24s} {np.mean(f):.3f} per fit {np.round(f, 2).tolist()}  (chance ~1/24)')
    print('\nheld-out targets: fraction whose prediction is closest to the nearest-covariate same-stage train donor')
    for arm in COV_ARMS:
        f = [np.mean([r['cond']['own']['closest_train_donor'] == r['nn_train_donor'] for r in allres[f'{arm}|{k}|{s}'] if r['heldout']])
             for k in range(3) for s in range(2)]
        g = [np.mean([r['cond']['own']['closest_corr'] for r in allres[f'{arm}|{k}|{s}'] if r['heldout']]) for k in range(3) for s in range(2)]
        h = [np.mean([r['cond']['own']['dev_corr_own'] for r in allres[f'{arm}|{k}|{s}'] if r['heldout']]) for k in range(3) for s in range(2)]
        summ[f'{arm}|heldout_nn_identity'] = float(np.mean(f))
        summ[f'{arm}|heldout_closest_train_corr'] = float(np.mean(g))
        summ[f'{arm}|heldout_dev_corr_own'] = float(np.mean(h))
        print(f'  {arm:24s} nn-identity {np.mean(f):.3f}; mean corr to closest train donor {np.mean(g):.3f}; dev corr with own real {np.mean(h):.3f}')
    print('\nheld-out per-donor: covars(own) - base(own) pb_rmse vs nearest-covariate distance and |age z| (Spearman, pooled 37 donors x 2 seeds)')
    from scipy.stats import spearmanr
    for arm in COV_ARMS:
        dd, nd, az = [], [], []
        for k in range(3):
            for s in range(2):
                b = {r['donor']: r for r in allres[f'base|{k}|{s}']}
                for r in allres[f'{arm}|{k}|{s}']:
                    if r['heldout']:
                        dd.append(r['cond']['own']['pb_rmse'] - b[r['donor']]['cond']['own']['pb_rmse'])
                        nd.append(r['nn_cov_dist']); az.append(abs(r['cov'][2]))
        r1 = spearmanr(dd, nd)[0]; r2 = spearmanr(dd, az)[0]
        summ[f'{arm}|delta_vs_nndist_spearman'] = float(r1); summ[f'{arm}|delta_vs_absagez_spearman'] = float(r2)
        summ[f'{arm}|delta_mean'] = float(np.mean(dd)); summ[f'{arm}|delta_frac_worse'] = float(np.mean(np.array(dd) > 0))
        print(f'  {arm:24s} mean delta {np.mean(dd):+.3f}, worse in {np.mean(np.array(dd) > 0):.2f}; rho(delta, nn dist) {r1:+.2f}; rho(delta, |age z|) {r2:+.2f}')
    print('\nwithin-target spread of generated pseudobulks (RMS distance to their mean, log CP10k), held-out, own')
    for arm in ['base'] + COV_ARMS:
        v = np.mean([np.mean([r['cond']['own']['gen_donor_spread'] for r in allres[f'{arm}|{k}|{s}'] if r['heldout']]) for k in range(3) for s in range(2)])
        w = np.mean([np.mean([r['cond']['mix']['gen_donor_spread'] for r in allres[f'{arm}|{k}|{s}'] if r['heldout']]) for k in range(3) for s in range(2)]) if arm != 'base' else float('nan')
        summ[f'{arm}|spread_own'] = float(v); summ[f'{arm}|spread_mix'] = float(w)
        print(f'  {arm:24s} own {v:.3f}  mix {w:.3f}')
    return summ


def main():
    allres = json.loads(OUT.read_text())['per_fit'] if OUT.exists() else {}
    for k in range(3):
        rf = fc.real_fold(k)
        obs = fc.log_cp10k(rf.pb)
        ctx_cache = {}
        for arm in ['base'] + COV_ARMS:
            for s in range(2):
                key = f'{arm}|{k}|{s}'
                if key in allres:
                    continue
                allres[key] = run_fit(arm, k, s, rf, obs, ctx_cache)
                print('done', key, flush=True)
                OUT.write_text(json.dumps({'per_fit': allres}))
        del rf
    summ = summarise(allres)
    OUT.write_text(json.dumps({'per_fit': allres, 'summary': summ}))
    print('wrote', OUT)


if __name__ == '__main__':
    main()
