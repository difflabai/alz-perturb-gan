"""Shared helpers for the arms_* screen analysis (arm effects versus base).

Read-only access to results/ (evaluator JSONs), fits/ (generated.npz) and
caches/ (development donors only). Writes nothing by itself.
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

RUNS = Path("/ml2/alz/runs/improve_20261109")
RESULTS = RUNS / "results"
FITS = RUNS / "fits"
CACHES = RUNS / "caches"
OUT = RUNS / "screen_analysis"
STUDY = Path("/ml2/alz/analysis/improve_20261109")

ARMS = {"hvg2000": ["base", "offset", "genemean", "donorpc", "covars", "covars,genemean,offset"],
        "wide": ["base", "genemean"]}
FOLDS = [0, 1, 2]
SEEDS = [0, 1]

# metric -> (subset-relative path in a split block or ("pc", key)), lower_is_better, kind
# kind: 'low' lower better, 'high' higher better, 'cov80' closer to .8, 'abs1' closer to 1
METRICS = {
    "pb_rmse": (("pseudobulk", "rmse"), "low"),
    "pb_dev_pearson": (("pseudobulk", "dev_pearson"), "high"),
    "meanlog_rmse_gene": (("mean_log", "rmse_gene"), "low"),
    "nonzero_rmse_gene": (("nonzero", "rmse_gene"), "low"),
    "hn_spearman_vs_heldout": (("contrast", "high_notad_vs_heldout", "spearman"), "high"),
    "hn_spearman_vs_train": (("contrast", "high_notad_vs_train", "spearman"), "high"),
    "pbdist_energy": (("distribution_pb", "energy_score"), "low"),
    "pc_energy": (("pc", "energy_score"), "low"),
    "pc_coverage80": (("pc", "coverage80"), "cov80"),
    "pc_spread": (("pc", "spread_ratio"), "abs1"),
    "cell_energy": (("pc", "cell_energy_distance"), "low"),
}


def goodness(kind, v):
    if v is None:
        return None
    if kind == "low":
        return -v
    if kind == "high":
        return v
    if kind == "cov80":
        return -abs(v - 0.8)
    if kind == "abs1":
        return -abs(np.log(v))
    raise ValueError(kind)


def dig(d, path):
    for p in path:
        if not isinstance(d, dict) or p not in d or d[p] is None:
            return None
        d = d[p]
    return d


def metric_value(res, metric, split, subset="all"):
    path, _ = METRICS[metric]
    if path[0] == "pc":
        return dig(res, ("pc_space", split, path[1]))
    return dig(res, ("subsets", subset, split) + tuple(path))


def load_fit_json(arm, panel, fold, seed):
    return json.loads((RESULTS / "fits" / arm / panel / f"fold{fold}" / f"seed{seed}.json").read_text())


def load_ens_json(arm, panel, fold):
    return json.loads((RESULTS / "ensembles" / arm / panel / f"fold{fold}.json").read_text())


def load_baseline_json(name, panel, fold):
    return json.loads((RESULTS / "baselines" / panel / f"fold{fold}" / f"{name}.json").read_text())


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    return f"{x:+.{nd}f}" if isinstance(x, float) and x < 0 else f"{x:.{nd}f}"
