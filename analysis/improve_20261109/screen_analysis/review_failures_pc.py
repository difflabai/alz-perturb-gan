"""review_failures_pc.py - independent PC-space coverage80 / within-condition spread / energy (claims 2, 3, 7).

Real donor summary: bag mean and SD (ddof=0) of fold-space PCs ((log1p CP10k - mean)/sd @ W / pc_sd) over a
random 64-cell subsample (own RNG; evaluator uses a different deterministic subsample).  Generated: same from
generated.npz pcs [T,M,64,50].  coverage80 = fraction of (target, dim) with real inside generated 10-90%;
wc_spread = sqrt(mean_t sum_d var_m(Y) / sum_d var_targets(y)); energy score per target, averaged.
Output: runs/.../screen_analysis/review_failures_pc.json
"""
import json
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, '/ml2/alz/analysis/improve_20261109/screen_analysis')
import numpy as np
from scipy.spatial.distance import cdist

from review_failures_common import CACHES, FITS, OUTDIR, fb, logcp, load_real

ARMS = ['base', 'donorpc', 'covars', 'covars,genemean,offset']


def summ(P):
    return np.concatenate([P.mean(-2), P.std(-2)], -1)


def main():
    per = {}
    for k in range(3):
        R = load_real(k)
        sp_ = np.load(CACHES / 'hvg2000' / f'fold{k}' / 'space.npz')
        W = sp_['W'] / sp_['pc_sd']
        rng = np.random.default_rng(31 + k)
        y = []
        for X in R['cells']:
            sub = X[rng.choice(len(X), min(64, len(X)), replace=False)]
            y.append(summ(((logcp(sub) - sp_['mean']) / sp_['sd']) @ W))
        y = np.array(y)
        for arm in ARMS:
            for s in range(2):
                f = np.load(FITS / arm / 'hvg2000' / f'fold{k}' / f'seed{s}' / 'generated.npz')
                order = [list(f['targets'].astype(str)).index(d) for d in R['donors']]
                Y = summ(f['pcs'][order].astype(np.float64))  # [T,M,100]
                e = {}
                for nm, msk in (('train', ~R['held']), ('heldout', R['held'])):
                    Ym, ym = Y[msk], y[msk]
                    lo, hi = np.percentile(Ym, [10, 90], axis=1)
                    e[f'cov80_{nm}'] = float(((ym >= lo) & (ym <= hi)).mean())
                    e[f'wc_spread_{nm}'] = float(np.sqrt(Ym.var(1, ddof=1).sum(1).mean() / ym.var(0, ddof=1).sum()))
                    es = [cdist(Ym[i], ym[i][None]).mean() - .5 * cdist(Ym[i], Ym[i]).mean() for i in range(len(ym))]
                    e[f'energy_{nm}'] = float(np.mean(es))
                per[(arm, k, s)] = e
    out = {arm: {key: fb({(k, s): per[(arm, k, s)][key] for k in range(3) for s in range(2)}) for key in per[(arm, 0, 0)]} for arm in ARMS}
    for arm in ARMS:
        print(arm.ljust(24), {a: round(b, 3) for a, b in out[arm].items()})
    (OUTDIR / 'review_failures_pc.json').write_text(json.dumps({'fold_balanced': out, 'per_fit': {f'{a}|{k}|{s}': v for (a, k, s), v in per.items()}}, indent=1))


if __name__ == '__main__':
    main()
