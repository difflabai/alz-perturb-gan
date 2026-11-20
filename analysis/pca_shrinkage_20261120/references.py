"""Stage-blind depth-qualified resampling with a configurable source-donor pool.

The cell-drawing loop is the frozen ``r5/baselines.generate('resample')``
code path, reproduced here only so the source pool can change. With the TRAIN
pool it must equal ``baselines.generate('resample_blind')`` bit for bit (see
the tests). Only the arrays this study needs are produced.
"""
from __future__ import annotations

import numpy as np


def resample_pool(real, targets, pool_of, rng_of, MX, M=32, K=256, kpc=64):
    """Generate M bags of K depth-qualified cells per target from pool_of(t).

    Returns dict with pcs [n,M,kpc,D] float16, mean_log [n,M,G] float32,
    sq_log [n,M,G] float16, pools {target: [donor ids]}.
    """
    n, G, D = len(targets), len(real.genes), real.W.shape[1]
    out = dict(pcs=np.zeros((n, M, kpc, D), np.float16), mean_log=np.zeros((n, M, G), np.float32),
               sq_log=np.zeros((n, M, G), np.float16), pools={})
    for j, t in enumerate(targets):
        rng = rng_of(real, t)
        libs = rng.choice(real.libs[t], (M, K), replace=True)
        pool = list(pool_of(t))
        if not pool or t in pool:
            raise ValueError('pool must be nonempty and exclude the target')
        out['pools'][str(real.donors[t])] = [str(real.donors[i]) for i in pool]
        perm = rng.permutation(np.asarray(pool))
        src = np.array([perm[m % len(perm)] for m in range(M)])
        for m in range(M):
            cells = real.cells[src[m]]
            slib = real.libs[src[m]]
            order = np.argsort(slib, kind='stable'); sorted_lib = slib[order]
            first = np.searchsorted(sorted_lib, libs[m], side='left')
            n_ok = len(slib) - first
            pick = np.where(n_ok > 0, first + (rng.random(K) * np.maximum(n_ok, 1)).astype(int), len(slib) - 1)
            rows = order[np.minimum(pick, len(slib) - 1)]
            X = cells[rows].toarray().astype(np.float64)
            sl = X.sum(1)
            p = np.minimum(1.0, libs[m] / np.maximum(sl, 1))
            nzr, nzc = np.nonzero(X)
            X[nzr, nzc] = rng.binomial(X[nzr, nzc].astype(np.int64), p[nzr])
            L = MX.log_cp10k(X)
            out['pcs'][j, m] = ((L[:kpc] - real.mean) / real.sd) @ real.W / real.pc_sd
            out['mean_log'][j, m] = L.mean(0)
            out['sq_log'][j, m] = (L ** 2).mean(0)
    return out
