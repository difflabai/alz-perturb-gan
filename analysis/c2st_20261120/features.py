"""Item builders for the microglia C2ST (real held-out donors vs a generator).

Operates on a RealFold-like object (``donors, is_heldout, stage, cells,
pcs, mean, sd, W``) and an aligned generated dict (``pcs [T,M,k,50]``,
``mean_log [T,M,G]``).  Real items depend only on the real data and fixed
seeds, so every generator is compared against identical real items.

Levels
  cell_pc     one cell = 50 TRAIN-fitted PCs (the coordinates the critic's
              bag summaries are built from).  Generated cells are stored only
              in PC space, so there is no gene-space cell level.
  bag_gene    one donor bag = mean per-cell log1p(panel CP10k), 2000 genes,
              K=256 cells (real: cells resampled with replacement from one
              held-out donor; generated: one generated donor's bag).
  bag_pcsum   one bag = PC mean and SD over 64 cells (interval-metric space).
  bag_inspan  bag_gene standardized and projected onto the 50 PC directions.
  bag_offspan bag_gene standardized residual orthogonal to those directions
              (invisible to anything computed from the 50 PCs).
"""
from __future__ import annotations

import hashlib
import numpy as np

LEVELS = ('cell_pc', 'bag_gene', 'bag_pcsum', 'bag_inspan', 'bag_offspan')


def _seed(*parts):
    h = hashlib.sha256('|'.join(map(str, parts)).encode()).digest()
    return int.from_bytes(h[:8], 'little')


def log_cp10k(X):
    X = np.asarray(X, np.float64)
    return np.log1p(X * (1e4 / np.maximum(X.sum(-1, keepdims=True), 1)))


def heldout_index(real):
    return np.flatnonzero(real.is_heldout)


def cells_per_donor(real, cap=256):
    return {int(i): int(min(len(real.pcs[i]), cap)) for i in heldout_index(real)}


def real_items(real, level, cap=256, n_bags=32, bag_k=256, pc_k=64):
    """Return (X, donor_index) for the real held-out items of one level."""
    X, g = [], []
    for i in heldout_index(real):
        d = str(real.donors[i])
        if level == 'cell_pc':
            n = min(len(real.pcs[i]), cap)
            rng = np.random.default_rng(_seed('c2st-real-cell', d))
            pick = rng.choice(len(real.pcs[i]), n, replace=False)
            # generated PCs are stored as float16; pass real PCs through the same precision
            X.append(real.pcs[i][pick].astype(np.float16).astype(np.float64))
            g += [i] * n
        elif level in ('bag_gene', 'bag_inspan', 'bag_offspan'):
            rng = np.random.default_rng(_seed('c2st-real-bag', d))
            C = real.cells[i]
            rows = rng.integers(0, C.shape[0], (n_bags, bag_k))
            bags = np.stack([log_cp10k(C[r].toarray()).mean(0) for r in rows])
            X.append(bags.astype(np.float32).astype(np.float64)); g += [i] * n_bags
        elif level == 'bag_pcsum':
            rng = np.random.default_rng(_seed('c2st-real-pcbag', d))
            P = real.pcs[i].astype(np.float16).astype(np.float64)
            bags = []
            for _ in range(n_bags):
                rows = rng.choice(len(P), pc_k, replace=len(P) < pc_k)
                bags.append(np.concatenate([P[rows].mean(0), P[rows].std(0)]))
            X.append(np.stack(bags)); g += [i] * n_bags
        else:
            raise ValueError(level)
    X = np.concatenate(X)
    if level in ('bag_inspan', 'bag_offspan'):
        X = span_split(X, real)[level == 'bag_offspan']
    return X, np.asarray(g)


def generated_items(gen, real, level, cap=256):
    """Generated items for the held-out targets, same counts as real_items."""
    X, g = [], []
    for i in heldout_index(real):
        d = str(real.donors[i])
        if level == 'cell_pc':
            P = gen['pcs'][i].reshape(-1, gen['pcs'].shape[-1]).astype(np.float64)
            n = min(len(real.pcs[i]), cap)
            rng = np.random.default_rng(_seed('c2st-gen-cell', d))
            X.append(P[rng.choice(len(P), n, replace=False)]); g += [i] * n
        elif level in ('bag_gene', 'bag_inspan', 'bag_offspan'):
            X.append(gen['mean_log'][i].astype(np.float64)); g += [i] * gen['mean_log'].shape[1]
        elif level == 'bag_pcsum':
            P = gen['pcs'][i].astype(np.float64)
            X.append(np.concatenate([P.mean(1), P.std(1)], -1)); g += [i] * P.shape[0]
        else:
            raise ValueError(level)
    X = np.concatenate(X)
    if level in ('bag_inspan', 'bag_offspan'):
        X = span_split(X, real)[level == 'bag_offspan']
    return X, np.asarray(g)


def span_split(mean_log, real):
    """Standardize gene-space bag means with the TRAIN PCA scaling and split
    into PC-span coordinates [n,50] and the orthogonal residual [n,G]."""
    Z = (np.asarray(mean_log, np.float64) - real.mean) / real.sd
    Q, _ = np.linalg.qr(real.W)          # orthonormal basis of the PC column space
    coords = Z @ Q
    return coords, Z - coords @ Q.T


def build(real, gen, level, real_cache=None, cap=256):
    """Stack real (y=0) and generated (y=1) items; ``real_cache`` dict reuses
    real items across generators."""
    if real_cache is not None and level in real_cache:
        Xr, gr = real_cache[level]
    else:
        Xr, gr = real_items(real, level, cap)
        if real_cache is not None:
            real_cache[level] = (Xr, gr)
    Xg, gg = generated_items(gen, real, level, cap)
    X = np.concatenate([Xr, Xg]); y = np.r_[np.zeros(len(Xr), int), np.ones(len(Xg), int)]
    return X, y, np.r_[gr, gg]
