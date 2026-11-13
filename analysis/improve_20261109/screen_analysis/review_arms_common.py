"""Adversarial review of ANALYSIS_arms (arm effects vs base): shared code.

Written from scratch; does NOT import evaluate.py, metrics.py or arms_*.py.
Reads only development-donor caches and generated.npz files (read-only).
Metric definitions are re-derived from PROTOCOL Interface 3 and checked
against the evaluator JSONs; random subsamples (64-cell PC summaries,
512-cell energy distance) use this script's own RNG, so PC/cell metrics are
expected to agree only up to subsampling noise.
"""
from __future__ import annotations

import os
for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[_v] = "6"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys
sys.dont_write_bytecode = True

import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.spatial.distance import cdist
from scipy.stats import rankdata

RUNS = Path("/ml2/alz/runs/improve_20261109")
OUT = RUNS / "screen_analysis"
STUDY = Path("/ml2/alz/analysis/improve_20261109")
ARMS = {"hvg2000": ["base", "offset", "genemean", "donorpc", "covars", "covars,genemean,offset"],
        "wide": ["base", "genemean"]}
DEV = set(json.loads((STUDY / "folds.json").read_text())["development_donors"])


def lcp(x):
    x = np.asarray(x, dtype=np.float64)
    return np.log1p(x * 1e4 / np.maximum(x.sum(-1, keepdims=True), 1))


class Real:
    """Per-donor real summaries of one cache (train donors then held-out)."""

    def __init__(self, panel, fold):
        d = RUNS / "caches" / panel / f"fold{fold}"
        s = np.load(d / "space.npz")
        self.genes = s["genes"].astype(str)
        mu, sd, W, psd = s["mean"].astype(float), s["sd"].astype(float), s["W"].astype(float), s["pc_sd"].astype(float)
        donors, held, stage, raw, pb, ml, nz, pcs = [], [], [], [], [], [], [], []
        for split in ["train", "heldout"]:
            m = np.load(d / split / "metadata.npz")
            dl = m["donors"].astype(str); cd = m["donor"].astype(str)
            assert set(dl) <= DEV and set(cd) <= DEV, "non-development donor in cache"
            C = sp.load_npz(d / split / "counts.npz").tocsr()
            for j, dn in enumerate(dl):
                X = C[np.where(cd == dn)[0]].toarray().astype(np.float64)
                L = lcp(X)
                donors.append(dn); held.append(split == "heldout")
                raw.append(m["raw_c"][j]); stage.append(int(round(m["raw_c"][j][3] * 3)))
                pb.append(X.sum(0)); ml.append(L.mean(0)); nz.append((X > 0).mean(0))
                pcs.append(((L - mu) / sd) @ W / psd)
        self.donors = np.array(donors); self.held = np.array(held); self.stage = np.array(stage)
        self.raw = np.array(raw); self.pb = np.array(pb); self.ml = np.array(ml); self.nz = np.array(nz)
        self.pcs = pcs
        self.obs = lcp(self.pb)


def load_gen(arm, panel, fold, seed, real):
    g = np.load(RUNS / "fits" / arm / panel / f"fold{fold}" / f"seed{seed}" / "generated.npz")
    t = g["targets"].astype(str)
    assert set(t) <= DEV
    assert np.array_equal(g["genes"].astype(str), real.genes)
    order = np.array([list(t).index(x) for x in real.donors])
    out = {k: g[k][order] for k in ["pb_counts", "mean_log", "nonzero", "pcs", "lib", "is_heldout", "stage"]}
    assert np.array_equal(out["is_heldout"].astype(bool), real.held)
    assert np.array_equal(out["stage"].astype(int), real.stage)
    return out


def concat(gs, sel=None):
    """Pool generated donors across seeds; sel = list of index arrays along M per seed."""
    out = {}
    for k in ["pb_counts", "mean_log", "nonzero", "pcs", "lib"]:
        parts = [g[k] if sel is None else g[k][:, s] for g, s in zip(gs, sel or [None] * len(gs))]
        out[k] = np.concatenate(parts, axis=1)
    return out


def spearman(a, b):
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def contrast(P, stage):
    if not ((stage == 0).any() and (stage == 3).any()):
        return None
    return P[stage == 3].mean(0) - P[stage == 0].mean(0)


def energy_score(Y, y):
    M = len(Y)
    return float(np.linalg.norm(Y - y, axis=1).mean() - cdist(Y, Y).sum() / (2 * M * (M - 1)))


def edist(A, B):
    return float(2 * cdist(A, B).mean() - cdist(A, A).mean() - cdist(B, B).mean())


def pred_profiles(pb, idx=None):
    """[T,M,G] counts -> [T,G] mean over M of log1p CP10k (optionally renormalised on idx)."""
    if idx is not None:
        pb = pb[:, :, idx]
    return np.stack([lcp(pb[t]).mean(0) for t in range(len(pb))])


def score(gen, real, split, rng_seed=0, idx=None, pc=True):
    """Own implementation of the primary metrics for one split ('heldout'/'train')."""
    mask = real.held if split == "heldout" else ~real.held
    P = pred_profiles(gen["pb_counts"], idx)
    obs = real.obs if idx is None else lcp(real.pb[:, idx])
    r = {}
    r["pb_rmse"] = float(np.mean(np.sqrt(((P[mask] - obs[mask]) ** 2).mean(1))))
    if idx is None:
        e = (gen["mean_log"].astype(float).mean(1) - real.ml)[mask].mean(0)
        r["meanlog_rmse_gene"] = float(np.sqrt((e ** 2).mean()))
        e = (gen["nonzero"].astype(float).mean(1) - real.nz)[mask].mean(0)
        r["nonzero_rmse_gene"] = float(np.sqrt((e ** 2).mean()))
    mc = contrast(P[mask], real.stage[mask])
    for ref, rm in [("heldout", real.held), ("train", ~real.held)]:
        rc = contrast(obs[rm], real.stage[rm])
        r[f"hn_spearman_vs_{ref}"] = spearman(mc, rc) if (mc is not None and rc is not None) else None
    if pc:
        rng = np.random.default_rng(rng_seed)
        pcs = gen["pcs"].astype(np.float64)
        Y = np.concatenate([pcs.mean(2), pcs.std(2)], -1)            # [T,M,100]
        k = pcs.shape[2]
        es, inside, ced = [], [], []
        for i in np.where(mask)[0]:
            rp = real.pcs[i]
            sub = rp[rng.choice(len(rp), k, replace=False)] if len(rp) >= k else rp
            y = np.concatenate([sub.mean(0), sub.std(0)])
            es.append(energy_score(Y[i], y))
            lo, hi = np.percentile(Y[i], [10, 90], axis=0)
            inside.append(((y >= lo) & (y <= hi)).mean())
            gc = pcs[i].reshape(-1, pcs.shape[3])
            gs = gc[rng.choice(len(gc), min(512, len(gc)), replace=False)]
            rs = rp[rng.choice(len(rp), min(512, len(rp)), replace=False)]
            ced.append(edist(gs, rs))
        r["pc_energy"] = float(np.mean(es)); r["pc_coverage80"] = float(np.mean(inside))
        r["cell_energy"] = float(np.mean(ced))
    return r, P


LOWER = {"pb_rmse": True, "meanlog_rmse_gene": True, "nonzero_rmse_gene": True, "pc_energy": True,
         "cell_energy": True, "hn_spearman_vs_heldout": False, "hn_spearman_vs_train": False,
         "pc_coverage80": "cov80"}
