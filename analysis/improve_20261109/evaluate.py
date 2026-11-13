#!/usr/bin/env python3
"""Interface 3 evaluator for improve_20261109.

Reads caches (Interface 1) and fit outputs (Interface 2, generated.npz) and
scores gene-level and distribution-level accuracy on HELD-OUT inner-fold
donors (primary) and on train donors (fit quality). Baselines (baselines.py)
produce in-memory arrays with the generated.npz layout and are scored by the
same `score_generated` function.

Usage:
  evaluate.py --fits-root /ml2/alz/runs/improve_20261109/fits \
              --caches-root /ml2/alz/runs/improve_20261109/caches \
              --out /ml2/alz/runs/improve_20261109/results

Outputs (all under --out):
  real/{panel}/fold{k}.json                       real-data descriptors
  fits/{arm}/{panel}/fold{k}/seed{s}.json (+.npz)  per-fit metrics
  ensembles/{arm}/{panel}/fold{k}.json (+.npz)    seed-ensemble + seed stability
  baselines/{panel}/fold{k}/{name}.json (+.npz)   baselines + noise ceiling
  pooled/...                                       pooled-across-folds contrasts
  aggregate.json, aggregate.csv, summary.md
Re-runnable: per-fit/ensemble/baseline JSONs are reused when their recorded
inputs (file size+mtime+sha256 of generated.npz, cache manifest sha256,
evaluator version) are unchanged. Missing or incomplete fits are skipped and
listed in aggregate.json.
"""
from __future__ import annotations

import os
for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(_v, "6")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse
import csv
import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import metrics as MX  # noqa: E402

EVALUATOR_VERSION = "improve_20261109-evaluator-v2"
PANELS = ["hvg2000", "wide"]
STAGES = ["Not AD", "Low", "Intermediate", "High"]
RISK_GENES = ["APOE", "TREM2", "INPP5D", "ACSL1", "SORL1", "SPI1", "DGAT1", "DGAT2", "ABCA1", "BIN1",
              "CD33", "TYROBP", "NLRP3", "PLCG2", "ABI3", "MS4A6A", "MS4A4A", "CD2AP", "PICALM", "CLU",
              "CR1", "ABCA7", "GRN", "TMEM119", "P2RY12", "CX3CR1", "CSF1R", "SPP1", "GPNMB", "LPL",
              "CD74", "HLA-DRA"]
N_WELL_EXPRESSED = 256          # pseudobulk summary space: top genes by train grand mean
CELL_ED_PER_TARGET = 512        # cells per side for per-target cell energy distance
CELL_ED_POOLED = 1500           # cells per side for pooled cell energy (as benchmark)
MEANLOG_THRESHOLDS = [0.05, 0.1, 0.25]
NONZERO_THRESHOLDS = [0.02, 0.05, 0.1]
FOLDS_JSON = HERE / "folds.json"
GEN_KEYS = ["targets", "is_heldout", "stage", "genes", "pb_counts", "mean_log", "nonzero", "pcs", "lib"]


def set_torch_threads():
    try:
        import torch
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "6")))
    except Exception:
        pass


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def _clean(v):
    """Recursively make a value strict-JSON safe (non-finite floats -> None,
    numpy scalars/arrays -> python)."""
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, np.ndarray):
        return _clean(v.tolist())
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return float(v) if np.isfinite(v) else None
    return v


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(_clean(value), indent=1, allow_nan=False) + "\n")
    tmp.replace(path)


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".partial.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


# ==========================================================================
# Real data from a cache (Interface 1)
# ==========================================================================
@dataclass
class RealFold:
    panel: str
    fold: int
    cache_dir: Path
    genes: np.ndarray
    mean: np.ndarray
    sd: np.ndarray
    W: np.ndarray
    pc_sd: np.ndarray
    age_mean: float
    age_sd: float
    donors: np.ndarray            # train donors then heldout donors (cache order)
    is_heldout: np.ndarray
    stage: np.ndarray
    raw_c: np.ndarray
    pb: np.ndarray                # [T,G] summed counts over all cells
    mean_log: np.ndarray          # [T,G] mean per-cell log1p(panel CP10k)
    nonzero: np.ndarray           # [T,G]
    cells: list = field(default_factory=list)   # per donor CSR counts (cells x G)
    pcs: list = field(default_factory=list)     # per donor [n_cells,50] float32
    libs: list = field(default_factory=list)    # per donor panel library sizes
    manifest_sha256: str | None = None

    @property
    def index(self):
        return {d: i for i, d in enumerate(self.donors)}

    def project(self, counts):
        X = np.asarray(counts.toarray() if sp.issparse(counts) else counts, dtype=np.float64)
        L = MX.log_cp10k(X)
        return (((L - self.mean) / self.sd) @ self.W / self.pc_sd).astype(np.float32)


def check_development_donors(donors, where):
    """Raise unless every donor is one of the 37 development donors and none is sealed (PROTOCOL hard rule)."""
    q = json.loads(FOLDS_JSON.read_text())
    dev, sealed = set(q["development_donors"]), set(q["sealed_evaluation_donors_fold0"])
    bad = sorted(set(donors) - dev) + sorted(set(donors) & sealed)
    if bad:
        raise PermissionError(f"{where}: donors outside the 37 development donors {bad[:5]}; refusing to read")


def load_real(cache_dir, panel=None, fold=None) -> RealFold:
    cache_dir = Path(cache_dir)
    # Guard BEFORE any count file is opened: manifest donor lists, then each split's donor list.
    if (cache_dir / "manifest.json").exists():
        man = json.loads((cache_dir / "manifest.json").read_text())
        check_development_donors(list(man.get("train_donors", [])) + list(man.get("heldout_donors", [])), cache_dir)
    for split in ["train", "heldout"]:
        with np.load(cache_dir / split / "metadata.npz", allow_pickle=False) as m:
            check_development_donors(np.asarray(m["donors"]).astype(str).tolist() + np.unique(np.asarray(m["donor"]).astype(str)).tolist(),
                                     cache_dir / split)
    with np.load(cache_dir / "space.npz", allow_pickle=False) as f:
        space = {k: f[k] for k in f.files}
    genes = np.asarray(space["genes"]).astype(str)
    parts = []
    for split in ["train", "heldout"]:
        C = sp.load_npz(cache_dir / split / "counts.npz").tocsr()
        with np.load(cache_dir / split / "metadata.npz", allow_pickle=False) as m:
            meta = {k: m[k] for k in m.files}
        if "genes" in meta and not np.array_equal(np.asarray(meta["genes"]).astype(str), genes):
            raise ValueError(f"{cache_dir}/{split}: gene axis differs from space.npz")
        if C.shape[1] != len(genes):
            raise ValueError(f"{cache_dir}/{split}: counts have {C.shape[1]} genes, space has {len(genes)}")
        parts.append((split, C, meta))
    donors, held, stage, raw, pbs, mls, nzs, cells, pcs, libs = [], [], [], [], [], [], [], [], [], []
    rf = RealFold(panel or cache_dir.parent.name, int(fold if fold is not None else str(cache_dir.name).removeprefix("fold")),
                  cache_dir, genes, space["mean"].astype(np.float64), space["sd"].astype(np.float64),
                  space["W"].astype(np.float64), space["pc_sd"].astype(np.float64),
                  float(space["age_mean"]), float(space["age_sd"]),
                  None, None, None, None, None, None, None)
    for split, C, meta in parts:
        cell_donor = np.asarray(meta["donor"]).astype(str)
        for j, d in enumerate(np.asarray(meta["donors"]).astype(str)):
            rows = np.where(cell_donor == d)[0]
            if len(rows) == 0:
                raise ValueError(f"{cache_dir}/{split}: donor {d} has no cells")
            Cd = C[rows]
            X = Cd.toarray().astype(np.float64)
            lib = X.sum(1)
            L = MX.log_cp10k(X)
            donors.append(d); held.append(split == "heldout")
            rc = np.asarray(meta["raw_c"][j], dtype=np.float64)
            raw.append(rc); stage.append(int(round(rc[3] * 3)))
            pbs.append(X.sum(0)); mls.append(L.mean(0)); nzs.append((X > 0).mean(0))
            cells.append(Cd.astype(np.float32)); libs.append(lib)
            pcs.append((((L - rf.mean) / rf.sd) @ rf.W / rf.pc_sd).astype(np.float32))
    rf.donors = np.array(donors); rf.is_heldout = np.array(held); rf.stage = np.array(stage)
    rf.raw_c = np.stack(raw); rf.pb = np.stack(pbs); rf.mean_log = np.stack(mls); rf.nonzero = np.stack(nzs)
    rf.cells, rf.pcs, rf.libs = cells, pcs, libs
    if (cache_dir / "manifest.json").exists():
        rf.manifest_sha256 = sha256(cache_dir / "manifest.json")
    return rf


# ==========================================================================
# Generated data (Interface 2)
# ==========================================================================
def load_generated(path):
    with np.load(path, allow_pickle=False) as f:
        missing = [k for k in GEN_KEYS if k not in f.files]
        if missing:
            raise ValueError(f"{path}: generated.npz lacks {missing}")
        g = {k: f[k] for k in GEN_KEYS}
    g["targets"] = g["targets"].astype(str); g["genes"] = g["genes"].astype(str)
    return g


def align_generated(gen, real: RealFold):
    """Validate gen against the cache and reorder its targets to real.donors."""
    T, M, G = gen["pb_counts"].shape
    if not np.array_equal(gen["genes"], real.genes):
        raise ValueError("generated genes differ from cache genes")
    if sorted(gen["targets"].tolist()) != sorted(real.donors.tolist()):
        raise ValueError("generated targets differ from cache train+heldout donors")
    for k, shape in [("mean_log", (T, M, G)), ("nonzero", (T, M, G)), ("lib", (T, M))]:
        if gen[k].shape != shape:
            raise ValueError(f"generated {k} shape {gen[k].shape} != {shape}")
    if gen["pcs"].shape[:2] != (T, M) or gen["pcs"].shape[3] != real.W.shape[1]:
        raise ValueError(f"generated pcs shape {gen['pcs'].shape}")
    order = np.array([list(gen["targets"]).index(d) for d in real.donors])
    out = {k: (gen[k][order] if k not in ("genes",) else gen[k]) for k in GEN_KEYS}
    if not np.array_equal(out["is_heldout"].astype(bool), real.is_heldout):
        raise ValueError("generated is_heldout disagrees with cache split")
    if not np.array_equal(out["stage"].astype(int), real.stage):
        raise ValueError("generated stage disagrees with cache metadata")
    for k in ["pb_counts", "mean_log", "nonzero", "lib"]:
        if not np.isfinite(out[k]).all():
            raise ValueError(f"generated {k} has non-finite values")
    if not np.isfinite(out["pcs"].astype(np.float32)).all():
        raise ValueError("generated pcs non-finite")
    return out


def concat_generated(gens):
    """Seed ensemble: pool generated donors across seeds along the M axis."""
    out = dict(gens[0])
    for k in ["pb_counts", "mean_log", "nonzero", "pcs", "lib"]:
        out[k] = np.concatenate([g[k] for g in gens], axis=1)
    return out


# ==========================================================================
# Gene subsets
# ==========================================================================
def gene_subsets(real: RealFold, caches_root=None):
    """{name: (index array, renormalise)}. 'all' always; 'risk' (forced risk
    genes present); for the genes shared with the same-fold cache of the other
    panel: 'shared_<other>_renorm' (log1p CP10k renormalised within the shared
    genes: the ONLY like-for-like cross-panel comparison) and
    'shared_<other>_withinpanel' (each panel's own CP10k denominator, so its
    pseudobulk, contrast, distribution and mean_log values are NOT comparable
    across panels; only its nonzero metrics are, since a count > 0 does not
    depend on the denominator).  Fixer revision: the within-panel subset was
    called 'shared_<other>' before."""
    genes = real.genes
    subsets = {"all": (np.arange(len(genes)), False)}
    risk = np.array([i for i, g in enumerate(genes) if g in set(RISK_GENES)], dtype=int)
    if len(risk):
        subsets["risk"] = (risk, False)
    if caches_root is not None:
        for other in PANELS:
            if other == real.panel:
                continue
            p = Path(caches_root) / other / f"fold{real.fold}" / "space.npz"
            if p.exists():
                with np.load(p, allow_pickle=False) as f:
                    og = set(np.asarray(f["genes"]).astype(str).tolist())
                idx = np.array([i for i, g in enumerate(genes) if g in og], dtype=int)
                if len(idx):
                    subsets[f"shared_{other}_withinpanel"] = (idx, False)
                    subsets[f"shared_{other}_renorm"] = (idx, True)
    return subsets


def pooled_gene_set(real: RealFold, caches_root):
    """Genes present in this panel's cache for all three inner folds (sorted)."""
    if caches_root is None:
        return None
    sets = []
    for k in range(3):
        p = Path(caches_root) / real.panel / f"fold{k}" / "space.npz"
        if not p.exists():
            return None
        with np.load(p, allow_pickle=False) as f:
            sets.append(set(np.asarray(f["genes"]).astype(str).tolist()))
    return np.array(sorted(set.intersection(*sets)))


# ==========================================================================
# Scoring (shared by fits, ensembles and baselines)
# ==========================================================================
def real_pc_summaries(real: RealFold, k):
    out = []
    for i, d in enumerate(real.donors):
        sub = MX.det_subsample(len(real.pcs[i]), k, MX.stable_seed("pc-summary", d))
        out.append(MX.pc_bag_summary(real.pcs[i][sub]))
    return np.stack(out)


def real_pb_subsample(real: RealFold, K):
    """Real pseudobulk over K cells, matching the depth of the generated K-cell
    pseudobulks: a deterministic subsample without replacement when the donor
    has >= K cells, else K cells drawn WITH replacement from its n < K cells
    (fixer revision: before, all n < K cells were summed, so those donors'
    real summaries had lower depth than the generated K-cell donors)."""
    out = []
    for i, d in enumerate(real.donors):
        n = real.cells[i].shape[0]
        if n >= K:
            sub = MX.det_subsample(n, K, MX.stable_seed("pb-summary", d))
        else:
            sub = np.sort(np.random.default_rng(MX.stable_seed("pb-summary-replace", d)).integers(0, n, K))
        out.append(np.asarray(real.cells[i][sub].sum(0)).ravel())
    return np.stack(out).astype(np.float64)


def score_generated(gen, real: RealFold, subsets, pooled_genes=None, train_only_meta=None):
    """Score an aligned generated set (fit, ensemble or baseline).

    Returns (result dict, arrays dict for the small per-fit npz)."""
    t0 = time.monotonic()
    T, M, G = gen["pb_counts"].shape
    held = real.is_heldout; train = ~held
    sets = {"heldout": held, "train": train}
    pb = gen["pb_counts"].astype(np.float64)
    K_est = float(np.median(pb.sum(-1) / np.maximum(gen["lib"], 1e-9)))
    K = int(round(K_est)) if np.isfinite(K_est) and K_est > 0 else 256
    gen_log_all = np.empty((T, M, G), dtype=np.float32)
    for t in range(T):
        gen_log_all[t] = MX.log_cp10k(pb[t])
    obs_log_all = MX.log_cp10k(real.pb)
    real_pb_K = real_pb_subsample(real, K)
    gen_ml = gen["mean_log"].astype(np.float64).mean(1)
    gen_nz = gen["nonzero"].astype(np.float64).mean(1)
    res = {"evaluator_version": EVALUATOR_VERSION, "T": int(T), "M": int(M), "G": int(G),
           "K_cells_per_generated_donor_estimated": K_est,
           "real_pb_summary_cells": {"K": K, "donors_with_fewer_cells_resampled_with_replacement":
                                     [str(d) for d, c in zip(real.donors, real.cells) if c.shape[0] < K]},
           "n_heldout": int(held.sum()), "n_train": int(train.sum()),
           "heldout_donors": real.donors[held].tolist(), "train_donors": real.donors[train].tolist(),
           "generated_library_vs_target": {
               "median_ratio_generated_lib_to_target_mean_lib": MX._f(np.median(
                   gen["lib"].mean(1) / np.array([l.mean() for l in real.libs])))},
           "subsets": {}}
    for sname, (idx, renorm) in subsets.items():
        if renorm:
            gl = np.empty((T, M, len(idx)), dtype=np.float32)
            for t in range(T):
                gl[t] = MX.log_cp10k(pb[t][:, idx])
            ol = MX.log_cp10k(real.pb[:, idx])
            rK = MX.log_cp10k(real_pb_K[:, idx])
        else:
            gl = gen_log_all[:, :, idx]
            ol = obs_log_all[:, idx]
            rK = MX.log_cp10k(real_pb_K)[:, idx]
        pred = gl.astype(np.float64).mean(1)
        grand = ol[train].mean(0)       # train grand-mean profile (donor-weighted)
        block = {"n_genes": int(len(idx)), "renormalised_within_subset": bool(renorm),
                 "cross_panel_comparable": ("all metrics" if renorm else
                                            "nonzero only (CP10k uses this panel's denominator)" if sname.endswith("_withinpanel")
                                            else "not applicable")}
        real_contrasts = {"heldout": MX.stage_contrasts(ol[held], real.stage[held]),
                          "train": MX.stage_contrasts(ol[train], real.stage[train])}
        for ename, mask in sets.items():
            e = {"pseudobulk": MX.pseudobulk_metrics(pred[mask], ol[mask], grand, real.stage[mask], real.donors[mask])}
            if not renorm:
                ref_ml = real.mean_log[train][:, idx].mean(0)
                ref_nz = real.nonzero[train][:, idx].mean(0)
                e["mean_log"] = MX.gene_error_metrics(gen_ml[mask][:, idx], real.mean_log[mask][:, idx], ref_ml, MEANLOG_THRESHOLDS)
                e["nonzero"] = MX.gene_error_metrics(gen_nz[mask][:, idx], real.nonzero[mask][:, idx], ref_ml, NONZERO_THRESHOLDS)
            cb, _ = MX.contrast_block(pred[mask], real.stage[mask], real_contrasts)
            e["contrast"] = cb
            # distribution in pseudobulk space on fixed well-expressed genes
            well = np.argsort(-grand, kind="stable")[:min(N_WELL_EXPRESSED, len(idx))]
            Ytr_ref = MX.within_stage_var_trace(rK[train][:, well], real.stage[train])
            e["distribution_pb"] = MX.distribution_metrics(gl[mask][:, :, well], rK[mask][:, well], real.stage[mask], Ytr_ref)
            e["distribution_pb"]["n_well_expressed_genes"] = int(len(well))
            block[ename] = e
        res["subsets"][sname] = block
    # PC-space distribution and cell energy (subset independent)
    kpc = gen["pcs"].shape[2]
    Ypc = MX.pc_bag_summary(gen["pcs"].astype(np.float32))      # [T,M,100]
    ypc = real_pc_summaries(real, kpc)
    tr_ref = MX.within_stage_var_trace(ypc[train], real.stage[train])
    res["pc_space"] = {"summary": f"bag mean and sd (ddof=0) of {gen['pcs'].shape[3]} fold PCs over {kpc} cells; "
                                  f"real donors use a deterministic {kpc}-cell subsample"}
    for ename, mask in sets.items():
        d = MX.distribution_metrics(Ypc[mask], ypc[mask], real.stage[mask], tr_ref)
        # cell energy per target and pooled
        eds, gpool, rpool = [], [], []
        for i in np.where(mask)[0]:
            donor = real.donors[i]
            gc = gen["pcs"][i].reshape(-1, gen["pcs"].shape[3]).astype(np.float32)
            gs = gc[MX.det_subsample(len(gc), CELL_ED_PER_TARGET, MX.stable_seed("ed-gen", donor))]
            rs = real.pcs[i][MX.det_subsample(len(real.pcs[i]), CELL_ED_PER_TARGET, MX.stable_seed("ed-real", donor))]
            eds.append(MX.cell_energy(gs, rs)); gpool.append(gc); rpool.append(real.pcs[i])
        gp = np.concatenate(gpool); rp = np.concatenate(rpool)
        gp = gp[MX.det_subsample(len(gp), CELL_ED_POOLED, MX.stable_seed("edpool-gen", ename))]
        rp = rp[MX.det_subsample(len(rp), CELL_ED_POOLED, MX.stable_seed("edpool-real", ename))]
        d["per_donor_cell_energy_distance"] = eds
        d["cell_energy_distance"] = MX._f(np.mean(eds))
        d["pooled_cell_energy_distance"] = MX.cell_energy(gp, rp)
        res["pc_space"][ename] = d
    arrays = {"donors": real.donors, "is_heldout": real.is_heldout, "stage": real.stage,
              "pred_profile_heldout": gen_log_all[held].astype(np.float64).mean(1).astype(np.float32),
              "genes": real.genes}
    # Pooled-across-folds material: heldout predictions renormalised over the
    # genes shared by all three folds of the panel.
    if pooled_genes is not None and len(pooled_genes):
        pos = {g: i for i, g in enumerate(real.genes)}
        pidx = np.array([pos[g] for g in pooled_genes])
        hp = np.empty((int(held.sum()), len(pidx)))
        for j, t in enumerate(np.where(held)[0]):
            hp[j] = MX.log_cp10k(pb[t][:, pidx]).mean(0)
        arrays.update(pooled_genes=pooled_genes, pooled_pred_heldout=hp.astype(np.float32),
                      pooled_obs_heldout=MX.log_cp10k(real.pb[held][:, pidx]).astype(np.float32))
    res["score_seconds"] = time.monotonic() - t0
    return res, arrays


# ==========================================================================
# Learning curve
# ==========================================================================
def learning_curve_summary(path, window=5000):
    path = Path(path)
    if not path.exists():
        return {"available": False}
    recs = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    recs = [r for r in recs if isinstance(r, dict) and "step" in r]
    if not recs:
        return {"available": False}
    last = max(r["step"] for r in recs)
    tail = [r for r in recs if r["step"] > last - window]
    keys = ["critic_gap", "D_adversarial", "D_penalty", "D_total", "G_adversarial", "G_total",
            "prior_regularization_unweighted", "D_gradient_norm", "G_gradient_norm"]
    out = {"available": True, "last_step": int(last), "window_updates": window, "n_records": len(tail)}
    for k in keys:
        v = [float(r[k]) for r in tail if isinstance(r.get(k), (int, float)) and np.isfinite(r[k])]
        out[f"mean_{k}"] = float(np.mean(v)) if v else None
    return out


# ==========================================================================
# Headline extraction (flat metrics for aggregation)
# ==========================================================================
# name -> (path, higher_is_better | 'abs1' (closer to 1) | 'cov80')
HEADLINE = {
    "pb_rmse": (("pseudobulk", "rmse"), False),
    "pb_dev_pearson": (("pseudobulk", "dev_pearson"), True),
    "pb_dev_spearman": (("pseudobulk", "dev_spearman"), True),
    "meanlog_rmse_gene": (("mean_log", "rmse_gene"), False),
    "meanlog_bias": (("mean_log", "bias"), "abs0"),
    "nonzero_rmse_gene": (("nonzero", "rmse_gene"), False),
    "nonzero_bias": (("nonzero", "bias"), "abs0"),
    "hn_spearman_vs_heldout": (("contrast", "high_notad_vs_heldout", "spearman"), True),
    "hn_spearman_vs_train": (("contrast", "high_notad_vs_train", "spearman"), True),
    "hn_top10_sign_vs_heldout": (("contrast", "high_notad_vs_heldout", "top_decile_sign_agreement"), True),
    "hn_top10_sign_vs_train": (("contrast", "high_notad_vs_train", "top_decile_sign_agreement"), True),
    "hn_slope_vs_heldout": (("contrast", "high_notad_vs_heldout", "regression_slope"), "slope1"),
    "hn_slope_vs_train": (("contrast", "high_notad_vs_train", "regression_slope"), "slope1"),
    "ord_spearman_vs_heldout": (("contrast", "slope_vs_heldout", "spearman"), True),
    "ord_spearman_vs_train": (("contrast", "slope_vs_train", "spearman"), True),
    "pbdist_energy": (("distribution_pb", "energy_score"), False),
    "pbdist_coverage80": (("distribution_pb", "coverage80"), "cov80"),
    "pbdist_spread": (("distribution_pb", "spread_ratio"), "abs1"),
    "pbdist_within_stage_spread": (("distribution_pb", "within_stage_spread_ratio"), "abs1"),
}
PC_HEADLINE = {
    "pc_energy": ("energy_score", False),
    "pc_coverage80": ("coverage80", "cov80"),
    "pc_spread": ("spread_ratio", "abs1"),
    "pc_within_stage_spread": ("within_stage_spread_ratio", "abs1"),
    "cell_energy": ("cell_energy_distance", False),
    "cell_energy_pooled": ("pooled_cell_energy_distance", False),
}
LC_HEADLINE = {"lc_critic_gap": "mean_critic_gap", "lc_D_total": "mean_D_total", "lc_G_total": "mean_G_total"}


def direction(name):
    base = name.split("__")[-1]
    for table in (HEADLINE, PC_HEADLINE):
        if base in table:
            return table[base][1]
    return None


def _dig(d, path):
    for p in path:
        if not isinstance(d, dict) or p not in d or d[p] is None:
            return None
        d = d[p]
    return d


def headline(result, which="heldout"):
    """Flat metric dict for one evaluated fit/baseline; subset metrics are
    prefixed '<subset>__' (no prefix for 'all')."""
    flat = {}
    for sname, block in result.get("subsets", {}).items():
        pre = "" if sname == "all" else f"{sname}__"
        e = block.get(which, {})
        for name, (path, _) in HEADLINE.items():
            v = _dig(e, path)
            if v is not None or sname == "all":
                flat[pre + name] = v
    pc = result.get("pc_space", {}).get(which, {})
    for name, (key, _) in PC_HEADLINE.items():
        flat[name] = pc.get(key)
    lc = result.get("learning_curve") or {}
    for name, key in LC_HEADLINE.items():
        if lc.get(key) is not None:
            flat[name] = lc[key]
    return flat


def goodness(name, v):
    """Map a metric to a 'higher is better' scalar for paired comparison."""
    d = direction(name)
    if v is None or d is None:
        return None
    if d is True:
        return v
    if d is False:
        return -v
    if d == "abs1":     # positive ratios (spread): symmetric on the log scale
        return -abs(np.log(v)) if v > 0 else None
    if d == "slope1":   # regression slopes: defined for every real value, so a wrong-sign slope counts as worse
        return -abs(v - 1)
    if d == "abs0":
        return -abs(v)
    if d == "cov80":
        return -abs(v - .8)
    return None


# ==========================================================================
# Directory scanning and orchestration
# ==========================================================================
def scan_fits(fits_root):
    fits_root = Path(fits_root)
    found, incomplete = [], []
    if not fits_root.exists():
        return found, incomplete
    for gp in sorted(fits_root.glob("*/*/fold*/seed*")):
        if not gp.is_dir():
            continue
        arm, panel, fold, seed = gp.parts[-4], gp.parts[-3], gp.parts[-2], gp.parts[-1]
        try:
            k = int(fold.removeprefix("fold")); s = int(seed.removeprefix("seed"))
        except ValueError:
            continue
        rec = {"arm": arm, "panel": panel, "fold": k, "seed": s, "dir": str(gp)}
        if (gp / "generated.npz").exists() and (gp / "fit_receipt.json").exists():
            found.append(rec)
        else:
            rec["missing"] = [x for x in ["generated.npz", "fit_receipt.json"] if not (gp / x).exists()]
            incomplete.append(rec)
    return found, incomplete


class RealCache:
    def __init__(self, caches_root):
        self.root = Path(caches_root); self.store = {}

    def get(self, panel, fold):
        key = (panel, fold)
        if key not in self.store:
            d = self.root / panel / f"fold{fold}"
            if not (d / "space.npz").exists():
                self.store[key] = None
            else:
                real = load_real(d, panel, fold)
                self.store[key] = (real, gene_subsets(real, self.root), pooled_gene_set(real, self.root))
        return self.store[key]


def file_key(path):
    st = Path(path).stat()
    return {"size": st.st_size, "mtime_ns": st.st_mtime_ns}


def evaluate_fit(rec, rc: RealCache, out_root, log):
    out_json = Path(out_root) / "fits" / rec["arm"] / rec["panel"] / f"fold{rec['fold']}" / f"seed{rec['seed']}.json"
    gpath = Path(rec["dir"]) / "generated.npz"
    fk = file_key(gpath)
    loaded = rc.get(rec["panel"], rec["fold"])
    if loaded is None:
        return None, f"cache missing for {rec['panel']}/fold{rec['fold']}"
    real, subsets, pooled = loaded
    if out_json.exists():
        old = json.loads(out_json.read_text())
        if (old.get("evaluator_version") == EVALUATOR_VERSION and old.get("generated_file") == fk
                and old.get("cache_manifest_sha256") == real.manifest_sha256
                and old.get("subset_names") == sorted(subsets)):
            return old, "cached"
    gen = align_generated(load_generated(gpath), real)
    res, arrays = score_generated(gen, real, subsets, pooled)
    res.update(kind="fit", arm=rec["arm"], panel=rec["panel"], fold=rec["fold"], seed=rec["seed"],
               fit_dir=rec["dir"], generated_file=fk, generated_sha256=sha256(gpath),
               cache_manifest_sha256=real.manifest_sha256, subset_names=sorted(subsets))
    for name in ["config.json", "fit_receipt.json"]:
        p = Path(rec["dir"]) / name
        try:
            res[name.removesuffix(".json")] = json.loads(p.read_text()) if p.exists() else None
        except json.JSONDecodeError:
            res[name.removesuffix(".json")] = "unparseable"
    res["learning_curve"] = learning_curve_summary(Path(rec["dir"]) / "learning_curve.jsonl")
    save_npz(out_json.with_suffix(".npz"), **arrays)
    write_json(out_json, res)
    return res, "evaluated"


def seed_stability(gens, real: RealFold):
    """Correlation of model High-minus-NotAD / ordinal-slope contrasts between
    seeds (from per-target predicted profiles; heldout targets and all targets)."""
    out = {}
    for ename, mask in [("heldout", real.is_heldout), ("all_targets", np.ones(len(real.donors), bool))]:
        cons = []
        for g in gens:
            pred = MX.predicted_profiles(g["pb_counts"][mask])
            cons.append(MX.stage_contrasts(pred, real.stage[mask]))
        block = {}
        for key in ["high_notad", "slope"]:
            ps, ss = [], []
            for i in range(len(cons)):
                for j in range(i + 1, len(cons)):
                    if cons[i][key] is not None and cons[j][key] is not None:
                        ps.append(MX.pearson(cons[i][key], cons[j][key]))
                        ss.append(MX.spearman(cons[i][key], cons[j][key]))
            block[key] = {"n_pairs": len(ps), "mean_pearson": MX.nanmean(ps), "mean_spearman": MX.nanmean(ss)}
        out[ename] = block
    return out


def evaluate_ensemble(arm, panel, fold, recs, rc: RealCache, out_root, log):
    recs = sorted(recs, key=lambda r: r["seed"])
    out_json = Path(out_root) / "ensembles" / arm / panel / f"fold{fold}.json"
    loaded = rc.get(panel, fold)
    if loaded is None:
        return None
    real, subsets, pooled = loaded
    keys = [{"seed": r["seed"], **file_key(Path(r["dir"]) / "generated.npz")} for r in recs]
    if out_json.exists():
        old = json.loads(out_json.read_text())
        if (old.get("evaluator_version") == EVALUATOR_VERSION and old.get("inputs") == keys
                and old.get("cache_manifest_sha256") == real.manifest_sha256
                and old.get("subset_names") == sorted(subsets)):
            return old
    gens = [align_generated(load_generated(Path(r["dir"]) / "generated.npz"), real) for r in recs]
    stab = seed_stability(gens, real)
    res, arrays = score_generated(concat_generated(gens), real, subsets, pooled)
    del gens
    lcs = [learning_curve_summary(Path(r["dir"]) / "learning_curve.jsonl") for r in recs]
    lc = {"available": any(x.get("available") for x in lcs)}
    for k in LC_HEADLINE.values():
        lc[k] = MX.nanmean([x.get(k) for x in lcs])
    res.update(kind="seed_ensemble", arm=arm, panel=panel, fold=fold, seeds=[r["seed"] for r in recs],
               inputs=keys, cache_manifest_sha256=real.manifest_sha256, subset_names=sorted(subsets),
               seed_stability=stab, learning_curve=lc)
    save_npz(out_json.with_suffix(".npz"), **arrays)
    write_json(out_json, res)
    log(f"ensemble {arm}/{panel}/fold{fold} seeds={res['seeds']}")
    return res


def evaluate_baselines(panel, fold, rc: RealCache, out_root, log):
    import baselines as BL
    loaded = rc.get(panel, fold)
    if loaded is None:
        return {}
    real, subsets, pooled = loaded
    outdir = Path(out_root) / "baselines" / panel / f"fold{fold}"
    results = {}
    for name in BL.BASELINES:
        out_json = outdir / f"{name}.json"
        if out_json.exists():
            old = json.loads(out_json.read_text())
            if (old.get("evaluator_version") == EVALUATOR_VERSION and old.get("baseline_version") == BL.BASELINE_VERSION
                    and old.get("cache_manifest_sha256") == real.manifest_sha256 and old.get("subset_names") == sorted(subsets)):
                results[name] = old
                continue
        t0 = time.monotonic()
        gen, info = BL.generate(name, real)
        res, arrays = score_generated(align_generated(gen, real), real, subsets, pooled)
        res.update(kind="baseline", arm=f"baseline:{name}", baseline=name, panel=panel, fold=fold, seed=None,
                   baseline_version=BL.BASELINE_VERSION, baseline_info=info,
                   meaningful_metrics=BL.MEANINGFUL[name], cache_manifest_sha256=real.manifest_sha256,
                   subset_names=sorted(subsets), generation_seconds=time.monotonic() - t0)
        save_npz(out_json.with_suffix(".npz"), **arrays)
        write_json(out_json, res)
        results[name] = res
        log(f"baseline {name} {panel}/fold{fold} {time.monotonic() - t0:.1f}s")
    nc_json = outdir / "noise_ceiling.json"
    ok = False
    if nc_json.exists():
        old = json.loads(nc_json.read_text())
        ok = (old.get("evaluator_version") == EVALUATOR_VERSION and old.get("baseline_version") == BL.BASELINE_VERSION
              and old.get("cache_manifest_sha256") == real.manifest_sha256 and old.get("subset_names") == sorted(subsets))
        if ok:
            results["noise_ceiling"] = old
    if not ok:
        nc = BL.noise_ceiling(real, subsets)
        nc.update(evaluator_version=EVALUATOR_VERSION, baseline_version=BL.BASELINE_VERSION, panel=panel, fold=fold,
                  cache_manifest_sha256=real.manifest_sha256, subset_names=sorted(subsets))
        write_json(nc_json, nc)
        results["noise_ceiling"] = nc
    return results


# ==========================================================================
# Pooled-across-folds contrast
# ==========================================================================
POOLED_WARNING = ("NOT a held-out estimate: every held-out donor of fold k is a training donor of the other two folds' "
                  "models, so the pooled model contrast (an average of train-stage contrasts over 2/3 of the 37 donors) "
                  "shares data with the pooled real contrast. The stage_mean baseline reaches Spearman ~0.93 here by "
                  "construction while independent halves agree at ~0.1 (split-half) and per-fold held-out Spearman is ~0.1. "
                  "Read it as a fit-to-all-development-donors quantity, never as generalisation; use the per-fold "
                  "hn_*_vs_heldout metrics for held-out claims (see NOTES_integration.md).")


def pooled_contrast(npz_paths):
    """Combine held-out donors of the three folds (each predicted by the model
    that did not train on it) before forming the contrast."""
    parts = []
    for p in npz_paths:
        with np.load(p, allow_pickle=False) as f:
            if "pooled_genes" not in f.files:
                return None
            parts.append({k: f[k] for k in ["pooled_genes", "pooled_pred_heldout", "pooled_obs_heldout", "stage", "is_heldout", "donors"]})
    g0 = parts[0]["pooled_genes"].astype(str)
    if any(not np.array_equal(p["pooled_genes"].astype(str), g0) for p in parts):
        return None
    pred = np.concatenate([p["pooled_pred_heldout"] for p in parts]).astype(np.float64)
    obs = np.concatenate([p["pooled_obs_heldout"] for p in parts]).astype(np.float64)
    stage = np.concatenate([p["stage"][p["is_heldout"]] for p in parts])
    donors = np.concatenate([p["donors"][p["is_heldout"]] for p in parts]).astype(str)
    if len(set(donors.tolist())) != len(donors):
        return None
    m = MX.stage_contrasts(pred, stage); r = MX.stage_contrasts(obs, stage)
    return {"n_donors": int(len(donors)), "n_genes": int(len(g0)), "n_by_stage": r["n_by_stage"],
            "independent_of_training": False,
            "warning": POOLED_WARNING,
            "normalisation": "log1p CP10k renormalised within genes present in all three folds of the panel",
            "high_notad": MX.agreement(m["high_notad"], r["high_notad"]),
            "slope": MX.agreement(m["slope"], r["slope"]),
            "real_split_half": MX.split_half_reliability(obs, stage)}


# ==========================================================================
# Aggregation
# ==========================================================================
def _mean_se(vals):
    v = [x for x in vals if x is not None and np.isfinite(x)]
    if not v:
        return None, None, 0
    return float(np.mean(v)), (float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else None), len(v)


# MEANINGFUL keys (baselines.py) -> headline metric names they govern
MEANINGFUL_GROUPS = {
    "pseudobulk": ["pb_rmse", "pb_dev_pearson", "pb_dev_spearman"],
    "mean_log/nonzero": ["meanlog_rmse_gene", "meanlog_bias", "nonzero_rmse_gene", "nonzero_bias"],
    "contrast": [n for n in HEADLINE if n.startswith(("hn_", "ord_"))],
    "distribution_pb": [n for n in HEADLINE if n.startswith("pbdist_")],
    "pc_space/cell_energy": list(PC_HEADLINE),
}
REFERENCE_BASELINES = ["stage_mean", "stage_mean_log", "shrunk_stage_mean", "ridge"]


def mask_not_meaningful(flat, meaningful):
    """Set to None every metric whose baselines.MEANINGFUL entry says 'NOT meaningful'."""
    masked = []
    for key, text in (meaningful or {}).items():
        if not str(text).startswith("NOT"):
            continue
        for name in list(flat):
            if name.split("__")[-1] in MEANINGFUL_GROUPS.get(key, []):
                flat[name] = None
                masked.append(name)
    return flat, sorted(set(masked))


def aggregate(out_root, fit_results, ens_results, base_results, incomplete, errors, log):
    out_root = Path(out_root)
    units = {}   # (row_label, panel) -> list of (fold, seed, flat, result)
    masked_by_label = {}
    for r in fit_results:
        units.setdefault((r["arm"], r["panel"]), []).append((r["fold"], r["seed"], headline(r), r))
    for r in ens_results:
        units.setdefault((f"{r['arm']} [seed-ensemble]", r["panel"]), []).append((r["fold"], "ens", headline(r), r))
    for (panel, fold), bl in base_results.items():
        for name, r in bl.items():
            if name == "noise_ceiling":
                continue
            flat, masked = mask_not_meaningful(headline(r), r.get("meaningful_metrics"))
            masked_by_label[f"baseline:{name}"] = masked
            units.setdefault((f"baseline:{name}", panel), []).append((fold, None, flat, r))
    rows = []
    agg = {"evaluator_version": EVALUATOR_VERSION, "rows": [], "paired": {}, "pooled_across_folds": {},
           "noise_ceiling": {}, "incomplete_fits": incomplete, "errors": errors,
           "notes": ["Primary metrics are on held-out inner-fold donors. n=3 folds: paired results are "
                     "descriptive (mean difference, sign count), not significance tests.",
                     "Row values are FOLD-BALANCED means (seeds averaged within fold, then the mean over folds); "
                     "__se is the SE over fold means (the fold is the independent unit: seeds of one fold share "
                     "held-out donors and training data); __unit_mean/__unit_se treat (fold, seed) units as "
                     "independent and are kept for reference only.",
                     "The three inner folds are not independent either: any two inner-fold training sets share "
                     "12-13 of their 24-25 donors (the third fold's held-out donors), so fold SEs understate the "
                     "uncertainty of a fold mean.",
                     "Paired comparisons count signs PER FOLD (seed-paired goodness differences averaged within fold); "
                     "a baseline reference is fitted once per fold, so it is never counted twice.",
                     "Reference baselines: stage_mean (log of the mean train proportion, the protocol's comparator) is a "
                     "weak bar; an arm should also be compared with stage_mean_log, shrunk_stage_mean and ridge "
                     "(paired blocks vs_<name>_baseline).",
                     "Baseline metrics marked 'NOT meaningful' in baselines.MEANINGFUL (PC-space and cell energy for "
                     "the profile baselines) are set to None in rows and paired comparisons (masked_not_meaningful).",
                     "shared_<other>_withinpanel subsets use each panel's own CP10k denominator: only their nonzero "
                     "metrics are comparable across panels; use shared_<other>_renorm for cross-panel pseudobulk claims.",
                     "noise_ceiling is a PEER-VARIABILITY REFERENCE, not a ceiling: it predicts a held-out donor from "
                     "full-depth observed peer profiles, and simple baselines can beat it.",
                     "pooled_across_folds: " + POOLED_WARNING]}
    for (label, panel), us in sorted(units.items()):
        names = sorted(set().union(*[set(u[2]) for u in us]))
        row = {"arm": label, "panel": panel, "n_units": len(us), "folds": sorted(set(u[0] for u in us))}
        if label in masked_by_label:
            row["masked_not_meaningful"] = masked_by_label[label]
        for n in names:
            um, use, k = _mean_se([u[2].get(n) for u in us])
            fold_means = [MX.nanmean([u[2].get(n) for u in us if u[0] == f]) for f in row["folds"]]
            fm, fse, nf = _mean_se(fold_means)
            row[n] = fm; row[n + "__se"] = fse; row[n + "__n_folds"] = nf
            row[n + "__unit_mean"] = um; row[n + "__unit_se"] = use; row[n + "__n"] = k
        if label.endswith("[seed-ensemble]"):
            st = [u[3].get("seed_stability", {}).get("heldout", {}).get("high_notad", {}).get("mean_spearman") for u in us]
            row["seed_stability_hn_spearman"] = MX.nanmean(st)
            st = [u[3].get("seed_stability", {}).get("all_targets", {}).get("high_notad", {}).get("mean_spearman") for u in us]
            row["seed_stability_hn_spearman_all_targets"] = MX.nanmean(st)
        rows.append(row)
    agg["rows"] = rows
    # paired comparisons (fold is the unit)
    def unit_map(label, panel):
        return {(u[0], u[1]): u[2] for u in units.get((label, panel), [])}
    for (label, panel), us in sorted(units.items()):
        if label.startswith("baseline:"):
            continue
        base_label = "base [seed-ensemble]" if label.endswith("[seed-ensemble]") else "base"
        names = sorted(set().union(*[set(u[2]) for u in us]))
        mine = unit_map(label, panel)
        comps = {}

        def compare(keys, ref_of):
            block = {}
            folds = sorted(set(k[0] for k in keys))
            for n in names:
                if direction(n) is None:
                    continue
                a_f, b_f, raw_f = [], [], []
                for f in folds:
                    pairs = [(goodness(n, mine[k].get(n)), goodness(n, ref_of(k).get(n)), mine[k].get(n), ref_of(k).get(n))
                             for k in keys if k[0] == f]
                    pairs = [p for p in pairs if p[0] is not None and p[1] is not None]
                    if pairs:
                        a_f.append(float(np.mean([p[0] for p in pairs]))); b_f.append(float(np.mean([p[1] for p in pairs])))
                        raw_f.append(float(np.mean([p[2] - p[3] for p in pairs])))
                s = MX.paired_summary(a_f, b_f, True)
                s["raw_mean_difference"] = MX.nanmean(raw_f)
                block[n] = s
            return block
        if label != base_label:
            ref = unit_map(base_label, panel)
            keys = sorted(k for k in mine if k in ref)
            comps["vs_base"] = {"paired_on": "fold (seed-paired differences averaged within fold)",
                                "units": [list(k) for k in keys], "metrics": compare(keys, lambda k: ref[k])}
        for bname in REFERENCE_BASELINES:
            sm = {k[0]: v for k, v in unit_map(f"baseline:{bname}", panel).items()}
            if not sm:
                continue
            keys = sorted(k for k in mine if k[0] in sm)
            comps[f"vs_{bname}_baseline"] = {"paired_on": "fold (arm seeds averaged within fold; baseline fitted once per fold)",
                                             "units": [list(k) for k in keys],
                                             "metrics": compare(keys, lambda k, sm=sm: sm[k[0]])}
        agg["paired"][f"{label}|{panel}"] = comps
    agg["paired_note"] = ("Paired 'mean_difference' and sign counts are over FOLDS on a goodness scale (higher is better: "
                          "negated for lower-is-better metrics, -|log ratio| for spread ratios, -|slope-1| for contrast "
                          "regression slopes (defined for wrong-sign slopes), -|cov-0.8| for coverage, -|bias| for biases); "
                          "raw_mean_difference is arm minus reference on the raw scale, averaged over folds.")
    # pooled across folds
    for (label, panel), us in sorted(units.items()):
        groups = {}
        for f, s, _, r in us:
            groups.setdefault(s, {})[f] = r
        for s, byfold in groups.items():
            if sorted(byfold) != [0, 1, 2]:
                continue
            paths = []
            for f in [0, 1, 2]:
                r = byfold[f]
                if r["kind"] == "fit":
                    paths.append(out_root / "fits" / r["arm"] / panel / f"fold{f}" / f"seed{r['seed']}.npz")
                elif r["kind"] == "seed_ensemble":
                    paths.append(out_root / "ensembles" / r["arm"] / panel / f"fold{f}.npz")
                else:
                    paths.append(out_root / "baselines" / panel / f"fold{f}" / f"{r['baseline']}.npz")
            if all(p.exists() for p in paths):
                pc = pooled_contrast(paths)
                if pc is not None:
                    agg["pooled_across_folds"][f"{label}|{panel}|seed={s}"] = pc
    for (panel, fold), bl in base_results.items():
        if "noise_ceiling" in bl:
            agg["noise_ceiling"][f"{panel}|fold{fold}"] = bl["noise_ceiling"]
    write_json(out_root / "aggregate.json", agg)
    # CSV
    cols = ["arm", "panel", "n_units"]
    allkeys = sorted(set().union(*[set(r) for r in rows]) - set(cols) - {"folds"}) if rows else []
    with open(out_root / "aggregate.csv.tmp", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols + allkeys)
        for r in rows:
            w.writerow([r.get(c) for c in cols] + [r.get(k) for k in allkeys])
    (out_root / "aggregate.csv.tmp").replace(out_root / "aggregate.csv")
    write_summary_md(out_root, agg)
    return agg


SUMMARY_COLS = ["pb_rmse", "pb_dev_pearson", "meanlog_rmse_gene", "nonzero_rmse_gene",
                "hn_spearman_vs_heldout", "hn_spearman_vs_train", "hn_top10_sign_vs_heldout", "hn_slope_vs_train",
                "pbdist_energy", "pc_energy", "pc_coverage80", "pc_spread", "cell_energy", "lc_critic_gap"]


def _fmt(v, se=None):
    if v is None:
        return "-"
    s = f"{v:.3f}"
    return s + (f" ± {se:.3f}" if se is not None else "")


def write_summary_md(out_root, agg):
    lines = ["# improve_20261109 evaluation summary (auto-generated)", "",
             f"Evaluator {EVALUATOR_VERSION}. Held-out inner-fold donors. Values are fold-balanced means ± SE over "
             "fold means (seeds averaged within fold); the inner folds' training sets overlap by about half, so "
             "even fold SEs understate uncertainty. Development study only: 3 inner folds, no significance claims.", "",
             "`-` for a baseline PC/cell column means the metric is not meaningful for that baseline (masked). "
             "stage_mean (log of mean proportions) is a weak comparator; also read stage_mean_log, "
             "shrunk_stage_mean and ridge. The noise_ceiling in aggregate.json is a peer-variability reference, "
             "not a ceiling.", ""]
    for panel in PANELS:
        prow = [r for r in agg["rows"] if r["panel"] == panel]
        if not prow:
            continue
        lines += [f"## Panel {panel}", "", "| arm | n units (folds) | " + " | ".join(SUMMARY_COLS) + " |",
                  "|" + "---|" * (len(SUMMARY_COLS) + 2)]
        for r in prow:
            lines.append(f"| {r['arm']} | {r['n_units']} ({len(r['folds'])}) | "
                         + " | ".join(_fmt(r.get(c), r.get(c + '__se')) for c in SUMMARY_COLS) + " |")
        lines.append("")
        keyp = ["pb_rmse", "pb_dev_pearson", "meanlog_rmse_gene", "hn_spearman_vs_heldout", "hn_slope_vs_heldout",
                "pbdist_energy", "pc_energy"]
        pl = [(k, v) for k, v in agg["paired"].items() if k.endswith("|" + panel)]
        if pl:
            lines += ["Paired (goodness scale; better/n counted over folds; `-` = no comparable fold):", "",
                      "| arm | vs | " + " | ".join(keyp) + " |", "|" + "---|" * (len(keyp) + 2)]
            for k, comps in pl:
                for cname, c in comps.items():
                    cells = []
                    for m in keyp:
                        s = c["metrics"].get(m) or {}
                        cells.append("-" if not s.get("n") else f"{s['mean_difference']:+.3f} ({s['n_better']}/{s['n']})")
                    lines.append(f"| {k.split('|')[0]} | {cname} | " + " | ".join(cells) + " |")
            lines.append("")
    pooled = agg.get("pooled_across_folds", {})
    if pooled:
        lines += ["## Pooled-across-folds High-minus-NotAD contrast (held-out donors of all folds; LEAKY, not held-out)", "",
                  "Warning: " + POOLED_WARNING, "",
                  "| arm | panel | seed | n donors | Spearman | top-decile sign | slope | real split-half Spearman |", "|---|---|---|---|---|---|---|---|"]
        for k, v in pooled.items():
            a, p, s = k.split("|")
            h = v.get("high_notad") or {}
            sh = v.get("real_split_half", {})
            lines.append(f"| {a} | {p} | {s.removeprefix('seed=')} | {v['n_donors']} | {_fmt(h.get('spearman'))} | "
                         f"{_fmt(h.get('top_decile_sign_agreement'))} | {_fmt(h.get('regression_slope'))} | {_fmt(sh.get('spearman'))} |")
        lines.append("")
    if agg.get("incomplete_fits"):
        lines += [f"Incomplete fits skipped: {len(agg['incomplete_fits'])}", ""]
    if agg.get("errors"):
        lines += [f"Fits that failed evaluation: {len(agg['errors'])} (see aggregate.json)", ""]
    (Path(out_root) / "summary.md").write_text("\n".join(lines) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fits-root", default="/ml2/alz/runs/improve_20261109/fits")
    ap.add_argument("--caches-root", default="/ml2/alz/runs/improve_20261109/caches")
    ap.add_argument("--out", default="/ml2/alz/runs/improve_20261109/results")
    ap.add_argument("--no-baselines", action="store_true")
    ap.add_argument("--no-ensembles", action="store_true")
    ap.add_argument("--panels", default=",".join(PANELS))
    ap.add_argument("--folds", default="0,1,2")
    args = ap.parse_args(argv)
    set_torch_threads()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(json.dumps({"t": time.strftime("%H:%M:%S"), "msg": msg}), flush=True)

    rc = RealCache(args.caches_root)
    panels = args.panels.split(","); folds = [int(x) for x in args.folds.split(",")]
    found, incomplete = scan_fits(args.fits_root)
    found = [f for f in found if f["panel"] in panels and f["fold"] in folds]
    fit_results, errors = [], []
    for rec in found:
        try:
            res, status = evaluate_fit(rec, rc, out, log)
            if res is None:
                errors.append({**rec, "error": status}); continue
            fit_results.append(res)
            log(f"{status} {rec['arm']}/{rec['panel']}/fold{rec['fold']}/seed{rec['seed']}")
        except Exception as e:  # tolerate broken/partial fits
            errors.append({**rec, "error": f"{type(e).__name__}: {e}"})
            log(f"ERROR {rec['dir']}: {e}")
    ens_results = []
    if not args.no_ensembles:
        groups = {}
        for r in fit_results:
            groups.setdefault((r["arm"], r["panel"], r["fold"]), []).append(r)
        for (arm, panel, fold), rs in sorted(groups.items()):
            if len(rs) < 2:
                continue
            recs = [next(f for f in found if f["arm"] == arm and f["panel"] == panel and f["fold"] == fold and f["seed"] == r["seed"]) for r in rs]
            try:
                e = evaluate_ensemble(arm, panel, fold, recs, rc, out, log)
                if e is not None:
                    ens_results.append(e)
            except Exception as ex:
                errors.append({"arm": arm, "panel": panel, "fold": fold, "ensemble": True, "error": f"{type(ex).__name__}: {ex}"})
    base_results = {}
    if not args.no_baselines:
        for panel in panels:
            for fold in folds:
                try:
                    b = evaluate_baselines(panel, fold, rc, out, log)
                    if b:
                        base_results[(panel, fold)] = b
                except Exception as ex:
                    errors.append({"panel": panel, "fold": fold, "baselines": True, "error": f"{type(ex).__name__}: {ex}"})
                    log(f"ERROR baselines {panel}/fold{fold}: {ex}")
    agg = aggregate(out, fit_results, ens_results, base_results, incomplete, errors, log)
    log(f"done: {len(fit_results)} fits, {len(ens_results)} ensembles, {sum(len(v) for v in base_results.values())} baseline records, "
        f"{len(incomplete)} incomplete, {len(errors)} errors")
    return agg


if __name__ == "__main__":
    main()
