"""Shared helpers for the stage-signal analysis (stage_*.py).

Reads ONLY the study caches (/ml2/alz/runs/improve_20261109/caches), which hold
the 37 development donors; every donor list is checked against folds.json
before any count file is opened. Read-only on caches/fits/results.

Profiles are exactly the evaluator's observed profiles: per donor, panel
counts summed over all cache cells, then log1p(CP10k) with the panel (or gene
subset) total as denominator (metrics.log_cp10k).
"""
import os, sys, json
sys.dont_write_bytecode = True
for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(_v, "6")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from scipy.stats import rankdata

STUDY = Path("/ml2/alz/analysis/improve_20261109")
RUNS = Path("/ml2/alz/runs/improve_20261109")
CACHES = RUNS / "caches"
FITS = RUNS / "fits"
OUT = RUNS / "screen_analysis" / "stage"
OUT.mkdir(parents=True, exist_ok=True)
FOLDS = json.loads((STUDY / "folds.json").read_text())
DEV = set(FOLDS["development_donors"])
SEALED = set(FOLDS["sealed_evaluation_donors_fold0"])
RISK_GENES = ["APOE", "TREM2", "INPP5D", "ACSL1", "SORL1", "SPI1", "DGAT1", "DGAT2", "ABCA1", "BIN1",
              "CD33", "TYROBP", "NLRP3", "PLCG2", "ABI3", "MS4A6A", "MS4A4A", "CD2AP", "PICALM", "CLU",
              "CR1", "ABCA7", "GRN", "TMEM119", "P2RY12", "CX3CR1", "CSF1R", "SPP1", "GPNMB", "LPL",
              "CD74", "HLA-DRA"]
DAM_GENES = ["APOE", "TREM2", "SPP1", "GPNMB", "LPL", "CD74", "HLA-DRA", "TYROBP", "CST7", "ITGAX",
             "CD9", "LGALS3", "FTL", "CTSD", "CTSB", "CTSL", "CD68", "AXL", "CLEC7A", "ACSL1", "CHI3L1",
             "C1QA", "C1QB", "C1QC", "TMEM119", "P2RY12", "CX3CR1", "CSF1R"]


def guard(donors, where):
    bad = sorted(set(donors) - DEV) + sorted(set(donors) & SEALED)
    if bad:
        raise PermissionError(f"{where}: non-development donors {bad[:5]}")


def log_cp10k(X):
    X = np.asarray(X, dtype=np.float64)
    return np.log1p(X * (1e4 / np.maximum(X.sum(-1, keepdims=True), 1)))


def load_cache(panel, fold, keep_cells=False):
    """Returns dict: genes, donors, is_heldout, stage, raw_c, pb [n,G] summed counts,
    ncells [n], libmean [n], and optionally cells (list of CSR per donor)."""
    d = CACHES / panel / f"fold{fold}"
    man = json.loads((d / "manifest.json").read_text())
    guard(man["train_donors"] + man["heldout_donors"], d)
    with np.load(d / "space.npz", allow_pickle=False) as f:
        genes = np.asarray(f["genes"]).astype(str)
    out = {k: [] for k in ["donors", "is_heldout", "stage", "raw_c", "pb", "ncells", "libmean", "cells"]}
    for split in ["train", "heldout"]:
        with np.load(d / split / "metadata.npz", allow_pickle=False) as m:
            meta = {k: m[k] for k in ["donor", "donors", "raw_c"]}
        cd = meta["donor"].astype(str)
        guard(meta["donors"].astype(str).tolist() + np.unique(cd).tolist(), d / split)
        C = sp.load_npz(d / split / "counts.npz").tocsr()
        for j, dn in enumerate(meta["donors"].astype(str)):
            rows = np.where(cd == dn)[0]
            Cd = C[rows]
            pb = np.asarray(Cd.sum(0)).ravel().astype(np.float64)
            out["donors"].append(dn); out["is_heldout"].append(split == "heldout")
            rc = meta["raw_c"][j].astype(np.float64)
            out["raw_c"].append(rc); out["stage"].append(int(round(rc[3] * 3)))
            out["pb"].append(pb); out["ncells"].append(len(rows))
            out["libmean"].append(float(np.asarray(Cd.sum(1)).mean()))
            if keep_cells:
                out["cells"].append(Cd)
    res = {"genes": genes, "donors": np.array(out["donors"]), "is_heldout": np.array(out["is_heldout"]),
           "stage": np.array(out["stage"]), "raw_c": np.stack(out["raw_c"]), "pb": np.stack(out["pb"]),
           "ncells": np.array(out["ncells"]), "libmean": np.array(out["libmean"])}
    if keep_cells:
        res["cells"] = out["cells"]
    return res


def common_genes(panel):
    sets = []
    for k in range(3):
        with np.load(CACHES / panel / f"fold{k}" / "space.npz", allow_pickle=False) as f:
            sets.append(set(np.asarray(f["genes"]).astype(str).tolist()))
    return sorted(set.intersection(*sets))


def all37(panel, keep_cells=False):
    """All 37 dev donors from the fold-0 cache, restricted to genes common to the
    three fold caches of the panel (CP10k renormalised within those genes)."""
    c = load_cache(panel, 0, keep_cells=keep_cells)
    g = common_genes(panel)
    pos = {x: i for i, x in enumerate(c["genes"])}
    idx = np.array([pos[x] for x in g])
    c["genes"] = np.array(g); c["pb"] = c["pb"][:, idx]
    if keep_cells:
        c["cells"] = [C[:, idx] for C in c["cells"]]
    assert len(c["donors"]) == 37 and set(c["donors"]) == DEV
    return c


# ---------------------------------------------------------------- contrasts
def contrasts(P, stage):
    """High-minus-NotAD, High-minus-(NotAD+Low) and ordinal OLS slope, [G] each."""
    stage = np.asarray(stage)
    o = {}
    o["hn"] = P[stage == 3].mean(0) - P[stage == 0].mean(0) if (stage == 0).any() and (stage == 3).any() else None
    lo = stage <= 1
    o["hnl"] = P[stage == 3].mean(0) - P[lo].mean(0) if lo.any() and (stage == 3).any() else None
    s = stage - stage.mean()
    o["slope"] = (s[:, None] * (P - P.mean(0))).sum(0) / (s ** 2).sum() if (s ** 2).sum() > 0 else None
    return o


def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 3:
        return np.nan
    ra, rb = rankdata(a), rankdata(b)
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def pearson(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def agree(pred, real, top_frac=.1):
    """pred = train (or model) contrast, real = held-out contrast.
    top10_sign_by_pred: sign agreement on the top decile of |pred| (predictive view);
    top10_sign_by_real: evaluator convention (top decile of |real|);
    slope_real_on_pred: calibration slope of held-out on train (1 = no shrinkage);
    slope_pred_on_real: evaluator convention."""
    k = max(1, int(np.ceil(top_frac * len(real))))
    tp = np.argsort(-np.abs(pred), kind="stable")[:k]
    tr = np.argsort(-np.abs(real), kind="stable")[:k]
    vp, vr = pred.var(), real.var()
    cv = ((pred - pred.mean()) * (real - real.mean())).mean()
    return {"spearman": spearman(pred, real), "pearson": pearson(pred, real),
            "top10_sign_by_pred": float((np.sign(pred[tp]) == np.sign(real[tp])).mean()),
            "top10_sign_by_real": float((np.sign(pred[tr]) == np.sign(real[tr])).mean()),
            "slope_real_on_pred": float(cv / vp) if vp > 0 else np.nan,
            "slope_pred_on_real": float(cv / vr) if vr > 0 else np.nan}


def fmt(x, d=3):
    return "nan" if x is None or not np.isfinite(x) else f"{x:.{d}f}"
