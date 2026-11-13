#!/usr/bin/env python3
"""External-donor evaluator for improve_20261109 round 4 (r4/PROTOCOL_EXT.md).

A thin layer over the FROZEN round-3 evaluator (``../r3/evaluate.py``,
``../r3/baselines.py``, ``../r3/metrics.py``), imported read-only from r3/
(r4 had no copies of them when this module was written).  Every SEA-AD score
goes through ``r3.evaluate.score_generated`` with the ``hvg2000`` fold-k
RealFold loaded by ``r3.evaluate.load_real`` (same held-out donors, genes,
space, metrics, seeds and common random numbers), so a fit or baseline that
used external donors is directly comparable with rounds 1-3.

What this module adds
---------------------
1. Model units (hvg2000x fits).  r4/generate.py (GR4-1) generates ONLY the 37
   SEA-AD targets (train + held-out) of an ``hvg2000x`` fit, so with the
   current generator the external view and the pooled recalibration below are
   None for model units (they are computed automatically if a generated file
   ever also covers external train targets).  The file is split by target:
   * SEA-AD view: the 37 SEA-AD targets only, aligned to the hvg2000 fold-k
     RealFold and scored by ``score_generated``.  So every held-out metric uses
     only SEA-AD held-out donors; the ``train`` block is SEA-AD train donors
     only; every ``*_recal`` metric and the xfit/DAM model contrast use SEA-AD
     train targets only (the default: cohorts differ in level).
   * External view (``ext.ext_view``): the same generated file scored on the
     external targets, with SEA-AD train donors as the reference set (the
     ``heldout`` block of a RealFold whose ``heldout`` donors are the external
     donors): external-train fit quality reported separately from SEA-AD train.
     Its ``*_recal`` numbers apply the SEA-AD-train calibration to external
     donors (they measure the cohort level offset).
   * Pooled recalibration (``ext.pooled_recal``): the additive per-gene
     correction fitted on SEA-AD train + external targets, applied to SEA-AD
     held-out donors (``pb_rmse_recal_pooled`` etc.).  Reported next to the
     SEA-AD-only default, never instead of it.
   * ``biascal`` modes change the generator; this evaluator cannot refit them.
     It reads the mode receipt's ``n_train_targets`` and labels the unit
     ``biascal_scope`` = 'seaad_train' / 'pooled' / 'unknown'.  r4/mode_biascal
     fits on the SEA-AD (cohort-0) train targets only, so hvg2000x units read
     'seaad_train' (the label 'pooled' would only appear for a generator that
     calibrates on external targets too).
   Targets are guarded: SEA-AD targets must be development donors (never
   sealed), all other targets must be donors of the external store
   (``ext/GSE268599/donors.json``); anything else raises PermissionError
   before any array is used.

2. Baselines with external donors (``run`` sub-command).  For each r3
   baseline the training set is SEA-AD train + external donors (external
   donors are train donors only) and the generated SEA-AD targets are scored by
   the same code path (``r3.baselines.generate`` on a pooled RealFold, then
   ``score_generated`` on the SEA-AD RealFold; the per-target random stream is
   seeded by (panel, fold, donor), so SEA-AD targets keep the r3 library
   draws).  Variants (unit labels):
     baseline:NAME            SEA-AD only (reproduces results_r3 to 1e-8; ``reproduce``)
     baseline:NAME+ext        pooled, no cohort correction
     baseline:NAME+extoff     pooled, external counts corrected by a per-gene
                              cohort offset (below)
     baseline:ridge_cohort+ext  r3 ridge with an extra cohort-indicator column
                              (SEA-AD targets predicted with cohort 0)
   NAME in grand_mean, stage_mean, stage_mean_log, shrunk_stage_mean, ridge,
   resample (same-stage donors of both cohorts), resample_blind.
   Per-gene cohort offset (fitted on train donors only): stage-matched log
   ratio of mean donor pseudobulk proportions, external minus SEA-AD,
     o_g = sum_s w_s [log(Pbar_ext,s,g + eps) - log(Pbar_sea,s,g + eps)] / sum_s w_s,
     w_s = n_ext,s n_sea,s / (n_ext,s + n_sea,s), stages present in both
     cohorts' train donors, eps = 1e-6, clipped to +-3, then centred so the
     pooled external proportions keep their total (c = log sum_g Pbar_ext,g e^-o_g).
   Correction: every external cell's counts are multiplied by f_g = e^-(o_g - c)
   and stochastically rounded (seed (panel, fold, donor)): expected counts are
   corrected, integers kept.  All derived arrays (pseudobulk, mean_log, PCs,
   libraries) are recomputed from the corrected cells; profile, dispersion and
   resampling baselines then run unchanged.  External-view and pooled-recal
   blocks always compare against the UNcorrected external counts.

3. Stage metrics: per (unit, seed) over the three folds of repeat 0:
   * ``xfit_*``: r3 ``xfit_from_npz`` (same seeds as r3), from the SEA-AD view npz.
   * ``hoc_*`` held-out stage-contrast agreement (new): per fold k the model
     contrast c_k = delta_k(High) - delta_k(Not AD) (the model's predicted
     stage deviations over fold-k SEA-AD TRAIN targets; npz ``xf_stage_dev``)
     against the real held-out contrast r_k = mean over held-out High donors -
     mean over held-out Not AD donors of y_d - gbar_k (npz ``xf_obs_dev``);
     statistic = mean over folds of Spearman (and Pearson) over genes.  Null:
     stage labels permuted among each fold's held-out donors (counts kept),
     2000 draws seeded by (panel, repeat) only, so every unit shares them;
     z and one-sided p.  Not leaky (c_k never sees a fold-k held-out donor).
     With 1-2 held-out Not AD donors per fold its null SD is large; it is the
     pooled version of r3's descriptive ``hn_spearman_vs_heldout``.
   * ``dam_*``: the frozen disease-gene direction score of r3 (SEA-AD targets
     of the fold); on hvg2000 only 3-5 listed genes are present, all 'up', so
     it is descriptive only (r2 repair S2).

4. Comparison (``aggregate``): for every hvg2000x unit the paired difference
   to the matching hvg2000 unit (same arm without cohort flags, same mode,
   fold and seed; searched in fits_r4, fits_r3, fits_r2, fits) on pb_rmse,
   meanlog_rmse_gene, nonzero_rmse_gene, logvar_rmse_gene, pc_energy_rep,
   cell_energy_rep, pc_coverage80_rep (difference of |cov - 0.8|) and
   xfit_score, with r3's seed-noise SE (``seed_noise_se``: reference pooled
   within-fold seed variance, else the round-2 prior for pb_rmse).
   Flags stripped for the hvg2000 reference: every flag starting with
   ``cohort`` plus ``extperm`` (the review-F1 control arm).
   Within-hvg2000x pairings (review finding F1; ``comparison_within_x``):
   * every cohort-aware unit against the NAIVE POOLED hvg2000x unit (same arm
     without cohort/extperm flags, same mode, fold, seed): the effect of the
     cohort handling given the same extra donors;
   * every unit without ``extperm`` against its ``extperm`` control (same
     flags plus extperm, same mode, fold, seed): the PRE-REGISTERED test of
     whether a gain comes from the MADRC stage labels rather than from more
     donors/cells.  A stage gain (hoc, xfit, pb_rmse of stagehalf) is credited
     to MADRC stage information only if this difference is favourable by more
     than 2 seed-noise SE.

External data source: ``caches/hvg2000x/fold{k}`` when it exists (genes and
space must equal hvg2000 fold k; cohort per donor from metadata), else built
directly from the store: hvg2000 fold-k gene index (``space.npz`` ``hvg``)
applied to ``ext/GSE268599/counts/{donor}.npz`` (sha256 checked against
donors.json), cells with zero panel counts dropped (cache rule), at most
``--ext-cap`` cells per donor (deterministic subsample, seed by donor;
default 1000 ~ the SEA-AD maximum of 984, because the store has 257-6,777
cells per donor and pooled-count baselines are cell-weighted).  raw_c =
[Braak/6, Thal/5, CERAD/3, ADNC/3, male, APOE4 dose, age]; Thal = midpoint of
``thal_range_from_A`` (Thal is not published for MADRC); a '90+' age is the
mean age of the fold's SEA-AD TRAIN donors aged >= 90.

Outputs (``--out``; only results_r4/ or smoke_r4/eval/ inside runs/improve_20261109):
  baselines/hvg2000/fold{k}/{label}.json/.npz   SEA-AD view (r3 layout) + 'ext' block
  fits/{label}/hvg2000/fold{k}/seed{s}.json/.npz
  ref/...                                         matching hvg2000 units (r3 evaluate_fit)
  ext_data/fold{k}.json                           external data / offset receipt
  aggregate_ext.json, summary_ext.txt, comparison_ext.csv, baselines_ext.csv

5. Confirmation extension (r4/PROTOCOL_CONFIRM.md, 2026-11-12).  Nothing above changed; additions:
   * Inner folds 3-8.  ``run --folds K`` works for K = 0..8 (hvg2000 fold-k RealFold through the r3 path;
     external donors from ``caches/hvg2000x/fold{k}`` built by build_cache_ext.py).  ``reproduce --folds
     3,...,8`` compares the SEA-AD-only baselines with results_r3 (which has folds 0-8).
   * ``--ref-arm ARM`` (repeatable): also score every complete hvg2000 unit (default generation and every
     mode) of ARM found in the ``--ref-root`` roots (default REF_ROOTS) for the run's folds, whether or not
     an hvg2000x fit exists (``ref/fits/{arm}~{tag}[#mode]/...``, r3 ``evaluate_fit``).  Without it,
     hvg2000 units are scored only as references of existing hvg2000x fits (unchanged).
   * Seed ensembles (``--ensemble LABEL`` repeatable, ``--ensemble-preset confirm|x_lead``,
     ``--ensemble-seeds 0,1``).  LABEL = ``{arm}~{tag}[#{mode}]``.  Tag ``x``: hvg2000x fits of the
     ``--fits-root`` roots; any other tag: the hvg2000 fits of the ``--ref-root`` root with that tag.  For
     every fold of the run where ALL listed seeds of the unit are complete, the generated donors of the
     seeds are pooled along the M axis exactly as ``r3.evaluate.concat_generated`` (n x 32 generated donors
     per target) and scored by ``score_generated`` on the hvg2000 fold-k RealFold.  hvg2000x members are
     first restricted to the 37 SEA-AD targets (``subset_generated``) and aligned (``align_generated``),
     as in ``evaluate_x_fit``; hvg2000 references go through ``r3.evaluate.evaluate_ensemble`` itself
     (pooled and M-matched, as results_r3).  Outputs ``ensembles/{unit}/hvg2000/fold{k}_n{n}.json/.npz``
     (x) and ``ref/ensembles/{unit}/hvg2000/fold{k}_n{n}[_M32].json/.npz`` (references); unit label
     ``{unit} [seed-ensemble n={n}]`` (r3 ``ensemble_label``).  Ensembles are collected by ``aggregate``
     (rows; x ensembles paired with the matching reference ensemble in ``comparison_ensembles_vs_hvg2000``);
     they never enter ``missing_references`` or ``comparison_within_x``.  Preset ``x_lead`` = the study
     lead's smoke_r4/ens/x_ensembles.py list (pbulk and g.dispz,pbulk,t.kappa@0.25 on hvg2000x; default,
     snapens, stagehalf, stageblind, biascal@stagehalf), which this feature replaces and reproduces
     (test_evaluate_ext.py ens_lead).  Preset ``confirm`` = pbulk and g.dispz,pbulk, tags x and r3, modes
     snapens, stagehalf, biascal@stagehalf (PROTOCOL_CONFIRM secondary read-outs).
   * ``aggregate --by-repeat``: one aggregation per repeat with data (folds 0-2, 3-5, 6-8; r3
     ``folds_of_repeat``) written to ``by_repeat/rep{r}/`` (aggregate_ext.json, summary_ext.txt,
     baselines_ext.csv, comparison_ext.csv).  Stage metrics (xfit, hoc) are defined within one repeat
     only.  Without the flag ``aggregate`` is unchanged (``--agg-folds``, default 0,1,2, into ``--out``).

Usage (from r4/):
  python3 -B evaluate_ext.py run --folds 0          # baselines (all variants) [+ fits] for one fold
  python3 -B evaluate_ext.py reproduce --folds 0    # SEA-AD baselines vs results_r3 (1e-8)
  python3 -B evaluate_ext.py aggregate              # stage metrics, comparison, tables
  python3 -B evaluate_ext.py run --folds 3 --no-baselines --fits-root r4=FITS_R4 --ref-root r3=FITS_R3 \
      --ref-arm pbulk --ref-arm g.dispz,pbulk --ensemble-preset confirm      # confirmation units, one fold
  python3 -B evaluate_ext.py aggregate --by-repeat
"""
from __future__ import annotations

import os
for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    try:
        _n = int(os.environ.get(_v, "3"))
    except ValueError:
        _n = 3
    os.environ[_v] = str(max(1, min(_n, 3)))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse
import contextlib
import csv
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent             # .../improve_20261109/r4
R3 = HERE.parent / "r3"


def _load_r3_evaluator():
    """Import ../r3/evaluate.py read-only (it puts r3/ first on sys.path, so its
    metrics and baselines modules are the r3 ones; both are path-checked)."""
    if "evaluate_r3_frozen" in sys.modules:
        return sys.modules["evaluate_r3_frozen"]
    spec = importlib.util.spec_from_file_location("evaluate_r3_frozen", R3 / "evaluate.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["evaluate_r3_frozen"] = mod
    spec.loader.exec_module(mod)
    return mod


EV = _load_r3_evaluator()
MX = EV.MX
import baselines as BL  # noqa: E402  (r3/, first on sys.path after EV import)

for _m, _name in [(MX, "metrics"), (BL, "baselines")]:
    if Path(_m.__file__).resolve().parent != R3.resolve():
        raise ImportError(f"evaluate_ext imported the wrong {_name} module: {_m.__file__}")

EXT_EVAL_VERSION = "improve_20261109-evaluate-ext-r4-v2"   # v2: cache v2 (cap, raw_c), extperm, within-x pairs
RUNS = Path("/ml2/alz/runs/improve_20261109")
EXT_STORE = RUNS / "ext" / "GSE268599"
PANEL = "hvg2000"            # SEA-AD scoring panel
XPANEL = "hvg2000x"          # training panel with external donors
WRITABLE = ["results_r4", "smoke_r4/eval"]
DEFAULT_EXT_CAP = 1000
OFFSET_EPS = 1e-6
OFFSET_CLIP = 3.0
HOC_PERMUTATIONS = 2000
COHORT_FLAG_PREFIXES = ("cohort", "extperm")
REF_ROOTS = ["r4=" + str(RUNS / "fits_r4"), "r3=" + str(RUNS / "fits_r3"), "r2=" + str(RUNS / "fits_r2"),
             "r1=" + str(RUNS / "fits")]
VARIANTS = {"sea": "", "ext": "+ext", "extoff": "+extoff"}
BASELINE_NAMES = list(BL.BASELINES)
COMPARE_METRICS = ["pb_rmse", "meanlog_rmse_gene", "nonzero_rmse_gene", "logvar_rmse_gene", "pc_energy_rep",
                   "cell_energy_rep", "pc_coverage80_rep", "xfit_score", "hoc_spearman"]
ROW_METRICS = ["pb_rmse", "pb_rmse_recal", "pb_rmse_recal_pooled", "pb_dev_pearson", "meanlog_rmse_gene",
               "meanlog_rmse_gene_recal", "meanlog_rmse_gene_recal_pooled", "nonzero_rmse_gene", "logvar_rmse_gene",
               "pc_energy_rep", "cell_energy_rep", "pc_coverage80_rep", "train_seaad_pb_rmse", "train_ext_pb_rmse",
               "train_ext_pb_rmse_recal_seaad"]
STAGE_ROW_METRICS = ["xfit_score", "xfit_score_z", "xfit_p", "hoc_spearman", "hoc_spearman_z", "hoc_spearman_p",
                     "hoc_pearson", "dam_sign_frac", "dam_signed_perm_z", "dam_n_present"]


# ==========================================================================
# Guards
# ==========================================================================
def check_out_dir(out):
    o = Path(out).resolve()
    runs = RUNS.resolve()
    if not any(o == (runs / w) or (runs / w) in o.parents for w in WRITABLE):
        raise PermissionError(f"refusing to write {o}: evaluate_ext writes only under {[str(runs / w) for w in WRITABLE]}")


def ext_donor_table(store=EXT_STORE):
    return json.loads((Path(store) / "donors.json").read_text())


def check_targets(donors, where, store=EXT_STORE):
    """Every donor must be a non-sealed SEA-AD development donor or a donor of the
    external store; returns (sea_list, ext_list) in input order."""
    q = json.loads(EV.FOLDS_JSON_R1.read_text())
    dev, sealed = set(q["development_donors"]), set(q["sealed_evaluation_donors_fold0"])
    ext = set(ext_donor_table(store)["donors"])
    donors = [str(d) for d in donors]
    bad = [d for d in donors if d in sealed or (d not in dev and d not in ext)]
    if bad:
        raise PermissionError(f"{where}: targets outside the 37 development donors and the external store {bad[:5]}; refusing")
    return [d for d in donors if d in dev], [d for d in donors if d in ext]


# ==========================================================================
# External donors in the hvg2000 fold-k space
# ==========================================================================
def _donor_arrays(rf, Cd):
    """Per-donor arrays exactly as r3 load_real computes them."""
    Cd = Cd.tocsr()
    X = Cd.toarray().astype(np.float64)
    lib = X.sum(1)
    L = MX.log_cp10k(X)
    return {"pb": X.sum(0), "ml": L.mean(0), "nz": (X > 0).mean(0),
            "lv": L.var(0, ddof=1) if len(L) > 1 else np.zeros(L.shape[1]),
            "cells": Cd.astype(np.float32), "libs": lib,
            "pcs": (((L - rf.mean) / rf.sd) @ rf.W / rf.pc_sd).astype(np.float32)}


def _sea_entries(real):
    out = []
    for i, d in enumerate(real.donors):
        out.append({"donor": str(d), "held": bool(real.is_heldout[i]), "raw_c": real.raw_c[i], "stage": int(real.stage[i]),
                    "cohort": 0, "pb": real.pb[i], "ml": real.mean_log[i], "nz": real.nonzero[i], "lv": real.log_var[i],
                    "cells": real.cells[i], "libs": real.libs[i], "pcs": real.pcs[i]})
    return out


def ext_raw_c(rec, sea_train_rawc):
    thal = float(np.mean(rec["thal_range_from_A"]))
    age = float(rec["age"])
    if rec.get("age_90plus_censored"):
        a = sea_train_rawc[:, 6]
        a = a[a >= 90]
        if len(a):
            age = float(a.mean())
    return np.array([rec["Braak_index"] / 6.0, thal / 5.0, rec["CERAD_index"] / 3.0, rec["ADNC_index"] / 3.0,
                     float(rec["male"]), float(rec["APOE4_dose"]), age], dtype=np.float64)


def load_ext_entries(real, caches_root=None, store=EXT_STORE, cap=DEFAULT_EXT_CAP, verify_sha=True):
    """External donors as train entries in the fold-k space of ``real``.
    Returns (entries, receipt)."""
    store = Path(store)
    table = ext_donor_table(store)
    xdir = Path(caches_root) / XPANEL / f"fold{real.fold}" if caches_root is not None else None
    sea_tr = real.raw_c[~real.is_heldout]
    entries, per = [], {}
    if xdir is not None and (xdir / "space.npz").exists():
        with np.load(xdir / "space.npz", allow_pickle=False) as a, np.load(real.cache_dir / "space.npz", allow_pickle=False) as b:
            for k in ["genes", "hvg", "mean", "sd", "W", "pc_sd"]:
                if k in b.files and not np.array_equal(np.asarray(a[k]), np.asarray(b[k])):
                    raise ValueError(f"{xdir}/space.npz '{k}' differs from {PANEL} fold{real.fold}")
        with np.load(xdir / "train" / "metadata.npz", allow_pickle=False) as m:
            meta = {k: m[k] for k in m.files}
        donors = np.asarray(meta["donors"]).astype(str)
        if "cohort" not in meta:
            raise ValueError(f"{xdir}/train/metadata.npz has no 'cohort'")
        coh = np.asarray(meta["cohort"]).astype(int)
        ext_d = [d for d, c in zip(donors, coh) if c != 0]
        check_targets(ext_d, xdir)
        if any(d not in table["donors"] for d in ext_d):
            raise PermissionError(f"{xdir}: cohort!=0 donors not in the external store")
        C = sp.load_npz(xdir / "train" / "counts.npz").tocsr()
        cd = np.asarray(meta["donor"]).astype(str)
        for j, d in enumerate(donors):
            if coh[j] == 0:
                continue
            rows = np.where(cd == d)[0]
            n_all = len(rows)
            if cap and len(rows) > cap:
                rows = rows[MX.det_subsample(len(rows), int(cap), MX.stable_seed("ext-cap", d))]
            arr = _donor_arrays(real, C[rows])
            rc = np.asarray(meta["raw_c"][j], dtype=np.float64)
            entries.append({"donor": d, "held": False, "raw_c": rc, "stage": int(round(rc[3] * 3)), "cohort": int(coh[j]), **arr})
            per[d] = {"cells_cache": int(n_all), "cells_used": int(len(rows)), "stage": int(round(rc[3] * 3)), "raw_c": rc.tolist()}
        receipt = {"source": "cache", "cache_dir": str(xdir), "manifest_sha256": EV.sha256(xdir / "manifest.json")
                   if (xdir / "manifest.json").exists() else None, "cap": cap, "per_donor": per,
                   "raw_c_rule": "cache metadata raw_c (build_cache_ext.py v2: published Braak/6, imputed mid-A Thal/5, "
                                 "published CERAD/3; cache already capped at 1000 cells per donor with this module's rule)"}
        return entries, receipt
    with np.load(real.cache_dir / "space.npz", allow_pickle=False) as f:
        hvg = np.asarray(f["hvg"]).astype(int)
    sgenes = np.load(store / "counts" / "genes.npy", allow_pickle=False).astype(str)
    if not np.array_equal(sgenes[hvg], real.genes):
        raise ValueError("external store genes[hvg] differ from the fold's panel genes")
    for d in sorted(table["donors"]):
        rec = table["donors"][d]
        p = store / rec["counts_file"]
        if verify_sha and EV.sha256(p) != rec["counts_sha256"]:
            raise ValueError(f"{p}: sha256 differs from donors.json")
        C = sp.load_npz(p).tocsr()[:, hvg].tocsr()
        n_raw = C.shape[0]
        keep = np.where(np.asarray(C.sum(1)).ravel() > 0)[0]
        n_zero = n_raw - len(keep)
        if cap and len(keep) > cap:
            keep = keep[MX.det_subsample(len(keep), int(cap), MX.stable_seed("ext-cap", d))]
        arr = _donor_arrays(real, C[keep].astype(np.float32))
        rc = ext_raw_c(rec, sea_tr)
        entries.append({"donor": d, "held": False, "raw_c": rc, "stage": int(rec["ADNC_index"]), "cohort": int(rec["cohort_code"]), **arr})
        per[d] = {"cells_store": int(n_raw), "cells_dropped_zero_panel": int(n_zero), "cells_used": int(len(keep)),
                  "stage": int(rec["ADNC_index"]), "raw_c": rc.tolist()}
    receipt = {"source": "store", "store": str(store), "donors_json_sha256": EV.sha256(store / "donors.json"),
               "selection_version": table.get("selection_version"), "cap": cap, "per_donor": per,
               "raw_c_rule": "[Braak/6, mid(thal_range_from_A)/5, CERAD/3, ADNC/3, male, APOE4 dose, age]; '90+' -> mean age of "
                             "the fold's SEA-AD train donors aged >= 90"}
    return entries, receipt


def build_realfold(template, entries, tag):
    rf = EV.RealFold(template.panel, template.fold, template.cache_dir, template.genes, template.mean, template.sd,
                     template.W, template.pc_sd, template.age_mean, template.age_sd, None, None, None, None, None, None, None)
    rf.donors = np.array([e["donor"] for e in entries])
    rf.is_heldout = np.array([e["held"] for e in entries], dtype=bool)
    rf.stage = np.array([e["stage"] for e in entries], dtype=int)
    rf.raw_c = np.stack([np.asarray(e["raw_c"], np.float64) for e in entries])
    rf.pb = np.stack([e["pb"] for e in entries]); rf.mean_log = np.stack([e["ml"] for e in entries])
    rf.nonzero = np.stack([e["nz"] for e in entries]); rf.log_var = np.stack([e["lv"] for e in entries])
    rf.cells = [e["cells"] for e in entries]; rf.libs = [e["libs"] for e in entries]; rf.pcs = [e["pcs"] for e in entries]
    rf.cohort = np.array([e["cohort"] for e in entries], dtype=int)
    rf.manifest_sha256 = hashlib.sha256(f"{template.manifest_sha256}|{tag}".encode()).hexdigest()
    return rf


def pool_entries(real, ext):
    sea = _sea_entries(real)
    return [e for e in sea if not e["held"]] + list(ext) + [e for e in sea if e["held"]]


def extview_entries(real, ext):
    """SEA-AD train donors as 'train', external donors as 'heldout' (external-train fit view)."""
    return [e for e in _sea_entries(real) if not e["held"]] + [{**e, "held": True} for e in ext]


# ==========================================================================
# Cohort offset (train donors only)
# ==========================================================================
def cohort_offset(rf, eps=OFFSET_EPS, clip=OFFSET_CLIP):
    """Per-gene stage-matched log ratio (external minus SEA-AD) of mean donor
    pseudobulk proportions over TRAIN donors (module docstring)."""
    tr = ~rf.is_heldout
    P = rf.pb / np.maximum(rf.pb.sum(1, keepdims=True), 1)
    num = np.zeros(P.shape[1]); wsum = 0.0; per = {}
    for s in range(4):
        a = tr & (rf.cohort == 0) & (rf.stage == s)
        b = tr & (rf.cohort != 0) & (rf.stage == s)
        if a.any() and b.any():
            r = np.log(P[b].mean(0) + eps) - np.log(P[a].mean(0) + eps)
            w = a.sum() * b.sum() / (a.sum() + b.sum())
            num += w * r; wsum += w
            per[str(s)] = {"n_seaad": int(a.sum()), "n_ext": int(b.sum()), "weight": float(w), "r": r}
    if wsum == 0:
        raise ValueError("cohort offset: no stage has train donors of both cohorts")
    o = np.clip(num / wsum, -clip, clip)
    pe = P[tr & (rf.cohort != 0)].mean(0)
    c = float(np.log(np.sum(pe * np.exp(-o))))
    rs = [v["r"] for v in per.values()]
    cons = [MX.pearson(rs[i], rs[j]) for i in range(len(rs)) for j in range(i + 1, len(rs))]
    info = {"definition": "o_g = weighted mean over stages present in both cohorts' train donors of log(Pbar_ext+eps) - "
                          "log(Pbar_sea+eps), w_s = n_ext n_sea/(n_ext+n_sea); clipped; counts x exp(-(o-c))",
            "eps": eps, "clip": clip, "c": c, "stages": {s: {k: v for k, v in d.items() if k != "r"} for s, d in per.items()},
            "rms": float(np.sqrt(np.mean(o ** 2))), "frac_clipped": float((np.abs(num / wsum) >= clip).mean()),
            "quantiles": {str(q): float(np.quantile(o, q)) for q in (0.01, 0.1, 0.5, 0.9, 0.99)},
            "between_stage_estimate_pearson": MX.nanmean(cons), "n_stage_pairs": len(cons)}
    return o, c, info


def apply_offset(rf, o, c):
    """Entries of rf with every cohort!=0 donor's counts scaled by exp(-(o - c)) (stochastic rounding)."""
    f = np.exp(-(o - c))
    out = []
    for i, d in enumerate(rf.donors):
        e = {"donor": str(d), "held": bool(rf.is_heldout[i]), "raw_c": rf.raw_c[i], "stage": int(rf.stage[i]),
             "cohort": int(rf.cohort[i]), "pb": rf.pb[i], "ml": rf.mean_log[i], "nz": rf.nonzero[i], "lv": rf.log_var[i],
             "cells": rf.cells[i], "libs": rf.libs[i], "pcs": rf.pcs[i]}
        if rf.cohort[i] != 0:
            C = rf.cells[i].tocsr().astype(np.float64)
            v = C.data * f[C.indices]
            rng = np.random.default_rng(MX.stable_seed("ext-offset", rf.panel, rf.fold, str(d)))
            fl = np.floor(v)
            v = fl + (rng.random(len(v)) < (v - fl))
            C2 = sp.csr_matrix((v, C.indices.copy(), C.indptr.copy()), shape=C.shape)
            C2.eliminate_zeros()
            keep = np.where(np.asarray(C2.sum(1)).ravel() > 0)[0]
            e.update(_donor_arrays(rf, C2[keep].astype(np.float32)))
        out.append(e)
    return out


@contextlib.contextmanager
def ridge_with_cohort():
    """r3 ridge with an extra cohort-indicator design column (SEA-AD targets: 0)."""
    orig = BL.ridge_design

    def design(real):
        X = orig(real)
        return np.column_stack([X, np.asarray(getattr(real, "cohort", np.zeros(len(real.donors))), np.float64)])
    BL.ridge_design = design
    try:
        yield
    finally:
        BL.ridge_design = orig


# ==========================================================================
# Generated-set helpers
# ==========================================================================
def load_generated_ext(path, store=EXT_STORE):
    """r3 load_generated with the round-4 target guard (SEA-AD dev donors or external store donors)."""
    with np.load(path, allow_pickle=False) as f:
        missing = [k for k in EV.GEN_KEYS if k not in f.files]
        if missing:
            raise ValueError(f"{path}: generated.npz lacks {missing}")
        check_targets(np.asarray(f["targets"]).astype(str).tolist(), path, store)
        g = {k: f[k] for k in EV.GEN_KEYS}
        if "sq_log" in f.files:
            g["sq_log"] = f["sq_log"]
        if "cohort" in f.files:
            g["cohort"] = f["cohort"]
    g["targets"] = g["targets"].astype(str); g["genes"] = g["genes"].astype(str)
    return g


def subset_generated(gen, donors, is_heldout=None):
    """Targets ``donors`` (in that order) of a generated dict; optionally overriding is_heldout."""
    pos = {d: i for i, d in enumerate(np.asarray(gen["targets"]).astype(str))}
    miss = [d for d in donors if d not in pos]
    if miss:
        raise ValueError(f"generated set lacks targets {miss[:5]}")
    idx = np.array([pos[d] for d in donors], dtype=int)
    out = {}
    for k, v in gen.items():
        if k in ("genes", "mode"):
            out[k] = v
        elif isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == len(pos):
            out[k] = v[idx]
        else:
            out[k] = v
    if is_heldout is not None:
        out["is_heldout"] = np.asarray(is_heldout, dtype=bool)
    return out


def pooled_recal_block(gen, real, ext_rf):
    """SEA-AD-only vs pooled (SEA-AD train + external targets) additive recalibration applied to SEA-AD held-out donors."""
    T_all = np.asarray(gen["targets"]).astype(str)
    have_ext = [d for d in ext_rf.donors if d in set(T_all)]
    if not have_ext:
        return None
    sea_tr = real.donors[~real.is_heldout].tolist(); sea_h = real.donors[real.is_heldout].tolist()
    g = subset_generated(gen, sea_tr + have_ext + sea_h)
    pred = MX.predicted_profiles(g["pb_counts"].astype(np.float64))
    gml = g["mean_log"].astype(np.float64).mean(1); gnz = g["nonzero"].astype(np.float64).mean(1)
    epos = {d: i for i, d in enumerate(ext_rf.donors)}
    ei = [epos[d] for d in have_ext]
    ntr, ne = len(sea_tr), len(have_ext)
    ri = {d: i for i, d in enumerate(real.donors)}
    rtr = [ri[d] for d in sea_tr]; rh = [ri[d] for d in sea_h]
    obs = np.concatenate([MX.log_cp10k(real.pb[rtr]), MX.log_cp10k(ext_rf.pb[ei]), MX.log_cp10k(real.pb[rh])])
    oml = np.concatenate([real.mean_log[rtr], ext_rf.mean_log[ei], real.mean_log[rh]])
    onz = np.concatenate([real.nonzero[rtr], ext_rf.nonzero[ei], real.nonzero[rh]])
    s_tr = np.arange(ntr); s_pool = np.arange(ntr + ne); s_h = np.arange(ntr + ne, len(obs))
    grand = obs[s_tr].mean(0)
    stage_h = real.stage[rh]; ref_ml = oml[s_tr].mean(0)
    out = {"n_seaad_train": ntr, "n_ext": ne, "ext_targets": have_ext,
           "definition": "a_g = mean over the listed train targets of (observed - predicted); seaad = SEA-AD train only "
                         "(the default, = score_generated's *_recal); pooled = SEA-AD train + external targets"}
    for nm, sel in [("seaad", s_tr), ("pooled", s_pool)]:
        a_pb = MX.additive_recalibration(pred[sel], obs[sel])
        a_ml = MX.additive_recalibration(gml[sel], oml[sel])
        a_nz = MX.additive_recalibration(gnz[sel], onz[sel])
        pbm = MX.pseudobulk_metrics(pred[s_h] + a_pb, obs[s_h], grand, stage_h, np.array(sea_h))
        out[f"pb_rmse_recal_{nm}"] = pbm["rmse"]; out[f"pb_dev_pearson_recal_{nm}"] = pbm["dev_pearson"]
        out[f"meanlog_rmse_gene_recal_{nm}"] = MX.gene_error_metrics(gml[s_h] + a_ml, oml[s_h], ref_ml, EV.MEANLOG_THRESHOLDS)["rmse_gene"]
        out[f"nonzero_rmse_gene_recal_{nm}"] = MX.gene_error_metrics(gnz[s_h] + a_nz, onz[s_h], ref_ml, EV.NONZERO_THRESHOLDS)["rmse_gene"]
        out[f"shift_{nm}"] = a_pb
    out["shift_rms_pooled_minus_seaad"] = float(np.sqrt(np.mean((out["shift_pooled"] - out["shift_seaad"]) ** 2)))
    out["shift_seaad"] = None; out["shift_pooled"] = None
    return out


def ext_view_block(gen, real, ext_entries, pc_repeats, cell_repeats):
    """External-train fit quality (r3 score_generated on SEA-AD train + external, external as 'heldout')."""
    T_all = set(np.asarray(gen["targets"]).astype(str))
    ents = [e for e in ext_entries if e["donor"] in T_all]
    if not ents:
        return None
    ev = build_realfold(real, extview_entries(real, ents), "extview")
    g = subset_generated(gen, ev.donors.tolist(), is_heldout=ev.is_heldout)
    g = EV.align_generated(g, ev)
    subsets = {"all": (np.arange(len(real.genes)), False)}
    res, _ = EV.score_generated(g, ev, subsets, None, settings=EV.ScoreSettings(pc_repeats, cell_repeats, None))
    per = res["subsets"]["all"]["heldout"]["pseudobulk"]["per_donor"]
    return {"definition": "r3 score_generated on SEA-AD train donors ('train') + external donors ('heldout' slot = external "
                          "TRAIN donors); *_recal applies the SEA-AD-train calibration to external donors",
            "external": EV.headline(res, "heldout"), "seaad_train": EV.headline(res, "train"),
            "external_per_donor_pb_rmse": {p["donor"]: p["rmse"] for p in per},
            "pc_repeats": pc_repeats, "cell_repeats": cell_repeats}


def biascal_scope(mode_receipt, n_sea_train, n_ext):
    if not isinstance(mode_receipt, dict):
        return "unknown"
    s = json.dumps(mode_receipt)
    if "biascal" not in s:
        return None
    n = None
    for key in ("n_train_targets",):
        stack = [mode_receipt]
        while stack and n is None:
            x = stack.pop()
            if isinstance(x, dict):
                if key in x and isinstance(x[key], int):
                    n = x[key]
                stack += list(x.values())
            elif isinstance(x, list):
                stack += x
    if n == n_sea_train:
        return "seaad_train"
    if n == n_sea_train + n_ext:
        return "pooled"
    return "unknown"


# ==========================================================================
# Per-fold context
# ==========================================================================
class FoldCtx:
    def __init__(self, caches_root, fold, opts, store=EXT_STORE):
        self.caches_root = Path(caches_root); self.fold = int(fold); self.opts = opts; self.store = Path(store)
        self.rc = EV.RealCache(self.caches_root, getattr(opts, "dam_draws", EV.DAM_NULL_DRAWS))
        loaded = self.rc.get(PANEL, self.fold)
        if loaded is None:
            raise FileNotFoundError(f"{self.caches_root}/{PANEL}/fold{fold}")
        self.real, self.subsets, self.pooled, self.dam = loaded
        self._ext = None; self._pool = None; self._pool_off = None

    def settings(self):
        return EV.ScoreSettings(self.opts.pc_repeats, self.opts.cell_repeats, self.dam)

    @property
    def ext(self):
        if self._ext is None:
            self._ext, self.ext_receipt = load_ext_entries(self.real, self.caches_root, self.store, self.opts.ext_cap)
            self.ext_rf = build_realfold(self.real, self._ext, "ext-only")
            self.ext_rf.is_heldout[:] = False
        return self._ext

    @property
    def pool(self):
        if self._pool is None:
            self._pool = build_realfold(self.real, pool_entries(self.real, self.ext), "pool:" + json.dumps(self.ext_sig()))
        return self._pool

    @property
    def pool_off(self):
        if self._pool_off is None:
            o, c, info = cohort_offset(self.pool)
            self.offset, self.offset_info = o, info
            self._pool_off = build_realfold(self.real, apply_offset(self.pool, o, c), "pooloff:" + json.dumps(self.ext_sig()))
        return self._pool_off

    def ext_sig(self):
        _ = self.ext
        r = self.ext_receipt
        return {"source": r["source"], "cap": r.get("cap"), "donors_json_sha256": r.get("donors_json_sha256"),
                "manifest_sha256": r.get("manifest_sha256")}

    def write_receipt(self, out_root):
        _ = self.pool_off
        rec = {"ext_eval_version": EXT_EVAL_VERSION, "panel": PANEL, "fold": self.fold, "ext": self.ext_receipt,
               "n_ext": len(self.ext), "ext_stage_counts": {str(s): int((self.ext_rf.stage == s).sum()) for s in range(4)},
               "seaad_train_stage_counts": {str(s): int(((self.real.stage == s) & ~self.real.is_heldout).sum()) for s in range(4)},
               "offset": self.offset_info, "offset_gene_top": _top_genes(self.real.genes, self.offset, 15)}
        EV.write_json(Path(out_root) / "ext_data" / f"fold{self.fold}.json", rec)
        return rec


def _top_genes(genes, v, k):
    o = np.argsort(-v)
    return {"ext_higher": [(str(genes[i]), float(v[i])) for i in o[:k]],
            "seaad_higher": [(str(genes[i]), float(v[i])) for i in o[::-1][:k]]}


# ==========================================================================
# Baselines
# ==========================================================================
def baseline_label(name, variant):
    if name == "ridge_cohort":
        return "baseline:ridge_cohort+ext"
    return f"baseline:{name}{VARIANTS[variant]}"


def baseline_jobs(variants):
    jobs = []
    for v in variants:
        for n in BASELINE_NAMES:
            jobs.append((n, v))
        if v == "ext":
            jobs.append(("ridge_cohort", "ext"))
    return jobs


def evaluate_baseline(ctx: FoldCtx, name, variant, out_root, log):
    label = baseline_label(name, variant)
    fname = label.removeprefix("baseline:")
    out_json = Path(out_root) / "baselines" / PANEL / f"fold{ctx.fold}" / f"{fname}.json"
    settings = ctx.settings()
    sig = None if variant == "sea" else ctx.ext_sig()
    if out_json.exists() and out_json.with_suffix(".npz").exists():
        old = json.loads(out_json.read_text())
        if EV._fresh(old, ctx.real, ctx.subsets, settings, baseline_version=BL.BASELINE_VERSION) and \
                old.get("ext_eval_version") == EXT_EVAL_VERSION and old.get("ext_signature") == sig:
            return old, "cached"
    t0 = time.monotonic()
    src = {"sea": lambda: ctx.real, "ext": lambda: ctx.pool, "extoff": lambda: ctx.pool_off}[variant]()
    if name == "ridge_cohort":
        with ridge_with_cohort():
            gen, info = BL.generate("ridge", src)
        info["cohort_column"] = "extra design column: cohort indicator (SEA-AD 0, external 1); SEA-AD targets predicted with 0"
        meaningful = BL.MEANINGFUL["ridge"]
    else:
        gen, info = BL.generate(name, src)
        meaningful = BL.MEANINGFUL[name]
    gen_seconds = time.monotonic() - t0
    g_sea = subset_generated(gen, ctx.real.donors.tolist())
    res, arrays = EV.score_generated(EV.align_generated(g_sea, ctx.real), ctx.real, ctx.subsets, ctx.pooled, settings=settings)
    res.update(kind="baseline", unit=label, arm=label, baseline=name if variant == "sea" else label.removeprefix("baseline:"),
               panel=PANEL, fold=ctx.fold, repeat=EV.repeat_of(ctx.fold), seed=None, baseline_version=BL.BASELINE_VERSION,
               baseline_info=info, meaningful_metrics=meaningful, cache_manifest_sha256=ctx.real.manifest_sha256,
               subset_names=sorted(ctx.subsets), generation_seconds=gen_seconds)
    res["ext_eval_version"] = EXT_EVAL_VERSION; res["ext_signature"] = sig; res["variant"] = variant
    if variant != "sea":
        _ = ctx.ext
        res["ext"] = {"training_donors": {"seaad_train": int((~ctx.real.is_heldout).sum()), "external": len(ctx.ext)},
                      "pooled_recal": pooled_recal_block(gen, ctx.real, ctx.ext_rf),
                      "ext_view": ext_view_block(gen, ctx.real, ctx.ext, ctx.opts.ext_pc_repeats, ctx.opts.ext_cell_repeats)}
        if variant == "extoff":
            res["ext"]["offset"] = ctx.offset_info
    EV.save_npz(out_json.with_suffix(".npz"), **arrays)
    EV.write_json(out_json, res)
    log(f"baseline {label} fold{ctx.fold} gen {gen_seconds:.0f}s total {time.monotonic() - t0:.0f}s "
        f"pb_rmse {res['subsets']['all']['heldout']['pseudobulk']['rmse']:.4f}")
    return res, "evaluated"


# ==========================================================================
# Fits
# ==========================================================================
def strip_cohort(arm):
    flags = [f for f in arm.split(",") if f and f != "base"]
    keep = [f for f in flags if not f.split("@")[0].startswith(COHORT_FLAG_PREFIXES)]
    return ",".join(sorted(keep)) if keep else "base"


def x_label(rec):
    return f"{rec['arm']}~x" + (f"#{rec['mode']}" if rec["mode"] else "")


def scan_x_fits(roots, modes_roots=(), folds=None):
    found, incomplete, shadowed = EV.scan_fits(roots, modes_roots, include_modes=True)
    return [r for r in found if r["panel"] == XPANEL and (folds is None or r["fold"] in folds)], incomplete


def evaluate_x_fit(rec, ctx: FoldCtx, out_root, log):
    label = x_label(rec)
    out_json = Path(out_root) / "fits" / label / PANEL / f"fold{rec['fold']}" / f"seed{rec['seed']}.json"
    gpath = Path(rec["gen"]); fk = EV.file_key(gpath)
    settings = ctx.settings()
    if out_json.exists() and out_json.with_suffix(".npz").exists():
        old = json.loads(out_json.read_text())
        if EV._fresh(old, ctx.real, ctx.subsets, settings, generated_file=fk) and old.get("ext_eval_version") == EXT_EVAL_VERSION \
                and old.get("ext_signature") == ctx.ext_sig():
            return old, "cached"
    gen = load_generated_ext(gpath, ctx.store)
    if not np.array_equal(gen["genes"], ctx.real.genes):
        raise ValueError(f"{gpath}: genes differ from {PANEL} fold{ctx.fold}")
    sea_t, ext_t = check_targets(gen["targets"].tolist(), gpath, ctx.store)
    if sorted(sea_t) != sorted(ctx.real.donors.tolist()):
        raise ValueError(f"{gpath}: SEA-AD targets differ from the fold's 37 donors")
    if "is_heldout" in gen:
        h = dict(zip(gen["targets"].tolist(), gen["is_heldout"].astype(bool).tolist()))
        if any(h[d] for d in ext_t):
            raise ValueError(f"{gpath}: an external target is marked held-out (external donors are training donors only)")
    g_sea = subset_generated(gen, ctx.real.donors.tolist())
    res, arrays = EV.score_generated(EV.align_generated(g_sea, ctx.real), ctx.real, ctx.subsets, ctx.pooled, settings=settings)
    res.update(kind="fit", unit=label, arm=rec["arm"], mode=rec["mode"], panel=PANEL, train_panel=XPANEL, fold=rec["fold"],
               repeat=EV.repeat_of(rec["fold"]), seed=rec["seed"], fit_dir=rec["dir"], root=rec["root"],
               generated_path=str(gpath), generated_file=fk, generated_sha256=EV.sha256(gpath),
               cache_manifest_sha256=ctx.real.manifest_sha256, subset_names=sorted(ctx.subsets))
    for name in ["config.json", "fit_receipt.json"]:
        p = Path(rec["dir"]) / name
        try:
            res[name.removesuffix(".json")] = json.loads(p.read_text()) if p.exists() else None
        except json.JSONDecodeError:
            res[name.removesuffix(".json")] = "unparseable"
    mr = None
    if rec["mode"] is not None:
        rp = gpath.with_name(gpath.stem + ".receipt.json")
        try:
            mr = json.loads(rp.read_text()) if rp.exists() else None
        except json.JSONDecodeError:
            mr = "unparseable"
        res["mode_receipt"] = mr
    res["learning_curve"] = EV.learning_curve_summary(Path(rec["dir"]) / "learning_curve.jsonl")
    _ = ctx.ext
    res["ext_eval_version"] = EXT_EVAL_VERSION; res["ext_signature"] = ctx.ext_sig()
    res["ext"] = {"n_ext_targets": len(ext_t), "n_seaad_targets": len(sea_t),
                  "generated_cohort_of_seaad_targets": (np.unique(subset_generated(gen, sea_t)["cohort"]).tolist()
                                                        if "cohort" in gen else "not recorded"),
                  "biascal_scope": biascal_scope(mr, int((~ctx.real.is_heldout).sum()), len(ext_t))
                  if rec["mode"] and "biascal" in rec["mode"] else None,
                  "pooled_recal": pooled_recal_block(gen, ctx.real, ctx.ext_rf) if ext_t else None,
                  "ext_view": ext_view_block(gen, ctx.real, ctx.ext, ctx.opts.ext_pc_repeats, ctx.opts.ext_cell_repeats)
                  if ext_t else None}
    EV.save_npz(out_json.with_suffix(".npz"), **arrays)
    EV.write_json(out_json, res)
    log(f"fit {label} fold{rec['fold']} seed{rec['seed']} pb_rmse {res['subsets']['all']['heldout']['pseudobulk']['rmse']:.4f}")
    return res, "evaluated"


def find_reference(rec, ref_roots):
    """The hvg2000 unit matching an hvg2000x unit: (ref_label, [ref records over all its seeds in this fold]) or None."""
    ref_arm = strip_cohort(rec["arm"])
    for spec in ref_roots:
        tag, root = spec.split("=", 1) if "=" in spec else ("ref", spec)
        fd = Path(root) / ref_arm / PANEL / f"fold{rec['fold']}"
        if not (fd / f"seed{rec['seed']}").is_dir():
            continue
        recs = []
        for sd in sorted(fd.glob("seed*")):
            if rec["mode"] is None:
                g = sd / "generated.npz"; ok = g.exists() and (sd / "fit_receipt.json").exists()
            else:
                g = sd / f"generated_{rec['mode']}.npz"; ok = g.exists() and (sd / f"generated_{rec['mode']}.receipt.json").exists()
            if ok:
                recs.append({"unit": f"{ref_arm}~{tag}" + (f"#{rec['mode']}" if rec["mode"] else ""), "arm": ref_arm,
                             "mode": rec["mode"], "panel": PANEL, "fold": rec["fold"], "seed": int(sd.name.removeprefix("seed")),
                             "dir": str(sd), "gen": str(g), "root": root})
        if any(r["seed"] == rec["seed"] for r in recs):
            return recs[0]["unit"], recs
    return None


def evaluate_ref(ref_rec, ctx: FoldCtx, out_root, log):
    ctx.rc.store[(PANEL, ctx.fold)] = (ctx.real, ctx.subsets, ctx.pooled, ctx.dam)
    res, status = EV.evaluate_fit(ref_rec, ctx.rc, Path(out_root) / "ref", log, ctx.opts)
    if res is None:
        raise RuntimeError(status)
    log(f"ref {ref_rec['unit']} fold{ref_rec['fold']} seed{ref_rec['seed']} {status}")
    return res


# ==========================================================================
# Confirmation extension: hvg2000 reference arms and seed ensembles
# ==========================================================================
ENSEMBLE_PRESETS = {
    "x_lead": [f"{a}~x" + (f"#{m}" if m else "") for a in ("pbulk", "g.dispz,pbulk,t.kappa@0.25")
               for m in ("", "snapens", "stagehalf", "stageblind", "biascal@stagehalf")],
    "confirm": [f"{a}~{t}#{m}" for t in ("x", "r3") for a in ("pbulk", "g.dispz,pbulk")
                for m in ("snapens", "stagehalf", "biascal@stagehalf")],
}
ENS_TAG = " [seed-ensemble"


def is_ensemble_label(label):
    return ENS_TAG in label


def _root_specs(specs):
    out = []
    for spec in specs:
        tag, root = spec.split("=", 1) if "=" in spec else ("ref", spec)
        out.append((tag, Path(root)))
    return out


def scan_ref_arm_fits(ref_roots, arms, folds):
    """Complete hvg2000 units (default + modes) of the given arms in the reference roots; labels '{arm}~{tag}[#mode]'.
    A (unit, fold, seed) found in an earlier root shadows later roots (as find_reference)."""
    want = {strip_flags_canon(a) for a in arms}
    found, seen = [], set()
    for tag, root in _root_specs(ref_roots):
        recs, _, _ = EV.scan_fits([f"{tag}={root}"], (), include_modes=True)
        for r in recs:
            if r["panel"] != PANEL or r["fold"] not in folds or strip_flags_canon(r["arm"]) not in want:
                continue
            key = (r["arm"], r["mode"], r["fold"], r["seed"])
            if key in seen:
                continue
            seen.add(key)
            found.append(r)
    return found


def strip_flags_canon(arm):
    flags = [f for f in arm.split(",") if f and f != "base"]
    return ",".join(sorted(flags)) if flags else "base"


def parse_unit_label(label):
    """'{arm}~{tag}[#{mode}]' -> (arm, tag, mode or None)."""
    unit, _, mode = label.partition("#")
    arm, sep, tag = unit.rpartition("~")
    if not sep or not arm or not tag:
        raise ValueError(f"ensemble label {label!r}: want ARM~TAG[#MODE]")
    return strip_flags_canon(arm), tag, (mode or None)


def ensemble_members(label, fold, fits_roots, ref_roots, seeds):
    """Records (one per seed, sorted) of the unit in this fold, or (None, missing seeds)."""
    arm, tag, mode = parse_unit_label(label)
    if tag == "x":
        roots, panel = _root_specs(fits_roots), XPANEL
    else:
        roots, panel = [(t, r) for t, r in _root_specs(ref_roots) if t == tag], PANEL
        if not roots:
            raise ValueError(f"ensemble label {label!r}: no --ref-root with tag {tag!r}")
    recs, missing = [], []
    for s in seeds:
        rec = None
        for t, root in roots:
            sd = root / arm / panel / f"fold{fold}" / f"seed{s}"
            if mode is None:
                g, ok = sd / "generated.npz", (sd / "generated.npz").exists() and (sd / "fit_receipt.json").exists()
            else:
                g = sd / f"generated_{mode}.npz"
                ok = g.exists() and (sd / f"generated_{mode}.receipt.json").exists()
            if ok:
                unit = f"{arm}~x" if tag == "x" else f"{arm}~{t}"
                rec = {"unit": unit + (f"#{mode}" if mode else ""), "arm": arm, "mode": mode, "panel": panel, "fold": fold,
                       "seed": int(s), "dir": str(sd), "gen": str(g), "root": str(root)}
                break
        if rec is None:
            missing.append(int(s))
        else:
            recs.append(rec)
    if missing:
        return None, missing
    return sorted(recs, key=lambda r: r["seed"]), []


def x_ensemble_generated(recs, ctx: FoldCtx):
    """Pooled seed ensemble of hvg2000x members: SEA-AD targets, aligned to the hvg2000 RealFold, concatenated
    along M exactly as r3 concat_generated."""
    gens = []
    for r in recs:
        g = load_generated_ext(Path(r["gen"]), ctx.store)
        if not np.array_equal(g["genes"], ctx.real.genes):
            raise ValueError(f"{r['gen']}: genes differ from {PANEL} fold{ctx.fold}")
        gens.append(EV.align_generated(subset_generated(g, ctx.real.donors.tolist()), ctx.real))
    return EV.concat_generated(gens), gens


def evaluate_x_ensemble(label, recs, ctx: FoldCtx, out_root, log):
    n = len(recs)
    unit = recs[0]["unit"]
    out_json = Path(out_root) / "ensembles" / unit / PANEL / f"fold{ctx.fold}_n{n}.json"
    settings = ctx.settings()
    keys = [{"seed": r["seed"], **EV.file_key(r["gen"])} for r in recs]
    if out_json.exists() and out_json.with_suffix(".npz").exists():
        old = json.loads(out_json.read_text())
        if EV._fresh(old, ctx.real, ctx.subsets, settings, inputs=keys) and old.get("ext_eval_version") == EXT_EVAL_VERSION \
                and old.get("ext_signature") == ctx.ext_sig():
            return old, "cached"
    ens, gens = x_ensemble_generated(recs, ctx)
    res, arrays = EV.score_generated(ens, ctx.real, ctx.subsets, ctx.pooled, settings=settings)
    lcs = [EV.learning_curve_summary(Path(r["dir"]) / "learning_curve.jsonl") for r in recs]
    lc = {"available": any(x.get("available") for x in lcs)}
    for k in EV.LC_HEADLINE.values():
        lc[k] = MX.nanmean([x.get(k) for x in lcs])
    res.update(kind="seed_ensemble", unit=unit, arm=recs[0]["arm"], mode=recs[0]["mode"], panel=PANEL, train_panel=XPANEL,
               fold=ctx.fold, repeat=EV.repeat_of(ctx.fold), seeds=[r["seed"] for r in recs],
               seed_key="+".join(str(r["seed"]) for r in recs), n_seeds=n, m_matched=False,
               label=EV.ensemble_label(unit, n, False), inputs=keys, cache_manifest_sha256=ctx.real.manifest_sha256,
               subset_names=sorted(ctx.subsets), learning_curve=lc, members=[r["gen"] for r in recs],
               ensemble_rule="all n*32 generated donors pooled (r3 concat_generated) over the SEA-AD targets")
    res["seed_stability"] = EV.seed_stability(gens, ctx.real)
    res["ext_eval_version"] = EXT_EVAL_VERSION; res["ext_signature"] = ctx.ext_sig()
    EV.save_npz(out_json.with_suffix(".npz"), **arrays)
    EV.write_json(out_json, res)
    log(f"ensemble {res['label']} fold{ctx.fold} seeds={res['seeds']} pb_rmse {res['subsets']['all']['heldout']['pseudobulk']['rmse']:.4f}")
    return res, "evaluated"


def evaluate_ref_ensemble(recs, ctx: FoldCtx, out_root, log, opts):
    ctx.rc.store[(PANEL, ctx.fold)] = (ctx.real, ctx.subsets, ctx.pooled, ctx.dam)
    return EV.evaluate_ensemble(recs[0]["unit"], PANEL, ctx.fold, recs, ctx.rc, Path(out_root) / "ref", log, opts)


def run_ensembles(labels, ctx: FoldCtx, out_root, log, opts, fits_roots, ref_roots, seeds):
    """Evaluate every listed ensemble for this fold; returns (done, missing) lists."""
    done, missing = [], []
    for label in labels:
        recs, miss = ensemble_members(label, ctx.fold, fits_roots, ref_roots, seeds)
        if recs is None:
            missing.append({"label": label, "fold": ctx.fold, "missing_seeds": miss})
            continue
        if parse_unit_label(label)[1] == "x":
            res, status = evaluate_x_ensemble(label, recs, ctx, out_root, log)
        else:
            outs = evaluate_ref_ensemble(recs, ctx, out_root, log, opts)
            res, status = (outs[0] if outs else None), "r3"
        done.append({"label": label, "fold": ctx.fold, "status": status,
                     "pb_rmse": (res or {}).get("subsets", {}).get("all", {}).get("heldout", {}).get("pseudobulk", {}).get("rmse")})
    return done, missing


# ==========================================================================
# Stage metrics: held-out contrast agreement
# ==========================================================================
def _rank(x):
    from scipy.stats import rankdata
    return rankdata(x)


def heldout_contrast_agreement(npz_paths, n_perm=HOC_PERMUTATIONS, seed=0):
    """Pooled held-out stage-contrast agreement over the folds of one repeat (module docstring)."""
    folds = []
    for p in npz_paths:
        with np.load(p, allow_pickle=False) as f:
            if "xf_stage_dev" not in f.files:
                return None
            held = f["is_heldout"].astype(bool)
            st = f["stage"][held].astype(int)
            obs = f["xf_obs_dev"].astype(np.float64)
            sdev = f["xf_stage_dev"].astype(np.float64)
        if not (np.isfinite(sdev[0]).all() and np.isfinite(sdev[3]).all()):
            continue
        mc = sdev[3] - sdev[0]
        if mc.std() == 0:
            return None
        folds.append({"mc": mc, "mr": _rank(mc), "obs": obs, "st": st})

    def stat(f, st):
        if not ((st == 0).any() and (st == 3).any()):
            return None, None
        rc = f["obs"][st == 3].mean(0) - f["obs"][st == 0].mean(0)
        return MX.pearson(f["mr"], _rank(rc)), MX.pearson(f["mc"], rc)

    obs_s, obs_p, per = [], [], []
    for f in folds:
        s, pr = stat(f, f["st"])
        per.append({"spearman": s, "pearson": pr, "n_heldout_by_stage": {str(k): int((f["st"] == k).sum()) for k in range(4)}})
        if s is not None:
            obs_s.append(s); obs_p.append(pr)
    if not obs_s:
        return None
    rng = np.random.default_rng(seed)
    null_s, null_p = [], []
    for _ in range(n_perm):
        ss, pp = [], []
        for f in folds:
            s, pr = stat(f, f["st"][rng.permutation(len(f["st"]))])
            if s is not None:
                ss.append(s); pp.append(pr)
        null_s.append(np.mean(ss) if ss else np.nan); null_p.append(np.mean(pp) if pp else np.nan)

    def blk(v, null):
        null = np.asarray(null, np.float64); null = null[np.isfinite(null)]
        sd = null.std(ddof=1) if len(null) > 1 else np.nan
        return {"value": float(v), "null_mean": float(null.mean()), "null_sd": float(sd),
                "z": float((v - null.mean()) / sd) if sd > 0 else None,
                "p_one_sided": float((np.sum(null >= v) + 1) / (len(null) + 1))}
    return {"n_folds": len(obs_s), "n_perm": int(n_perm), "per_fold": per,
            "spearman": blk(float(np.mean(obs_s)), null_s), "pearson": blk(float(np.mean(obs_p)), null_p),
            "definition": "mean over folds of Spearman/Pearson over genes between the model's High-minus-NotAD deviation "
                          "(fold-k SEA-AD train targets) and the real held-out High-minus-NotAD contrast; null = held-out "
                          "stage labels permuted within folds"}


# ==========================================================================
# Aggregation
# ==========================================================================
def _flat(res):
    flat = EV.headline(res, "heldout")
    tr = EV.headline(res, "train")
    flat["train_seaad_pb_rmse"] = tr.get("pb_rmse")
    ext = res.get("ext") or {}
    pr = ext.get("pooled_recal") or {}
    for k in ["pb_rmse_recal", "meanlog_rmse_gene_recal", "nonzero_rmse_gene_recal", "pb_dev_pearson_recal"]:
        flat[k + "_pooled"] = pr.get(k + "_pooled")
        flat[k + "_seaad_check"] = pr.get(k + "_seaad")
    evb = (ext.get("ext_view") or {}).get("external") or {}
    flat["train_ext_pb_rmse"] = evb.get("pb_rmse")
    flat["train_ext_pb_rmse_recal_seaad"] = evb.get("pb_rmse_recal")
    flat["train_ext_meanlog_rmse_gene"] = evb.get("meanlog_rmse_gene")
    return flat


def collect(out_root):
    """units[(label)] -> list of (fold, seed, flat, res, npz path)"""
    out_root = Path(out_root)
    units = {}
    for p in sorted((out_root / "baselines" / PANEL).glob("fold*/*.json")):
        r = json.loads(p.read_text())
        if r.get("kind") != "baseline":
            continue
        flat, _ = EV.mask_not_meaningful(_flat(r), r.get("meaningful_metrics"))
        units.setdefault(r["unit"], []).append((int(r["fold"]), None, flat, r, p.with_suffix(".npz")))
    for sub in ["fits", "ref/fits"]:
        for p in sorted((out_root / sub).glob(f"*/{PANEL}/fold*/seed*.json")):
            r = json.loads(p.read_text())
            units.setdefault(r["unit"], []).append((int(r["fold"]), int(r["seed"]), _flat(r), r, p.with_suffix(".npz")))
    for sub in ["ensembles", "ref/ensembles"]:          # confirmation extension: seed ensembles (seed slot None)
        for p in sorted((out_root / sub).glob(f"*/{PANEL}/fold*_n*.json")):
            r = json.loads(p.read_text())
            if r.get("kind") != "seed_ensemble":
                continue
            units.setdefault(r["label"], []).append((int(r["fold"]), None, _flat(r), r, p.with_suffix(".npz")))
    return units


def stage_blind(label):
    unit = label.partition(" [")[0]
    arm, _, mode = unit.partition("#")
    if "nostage" in arm.split("~")[0].split(","):
        return True
    if mode and (EV.stage_blind_mode_r3(mode) == "stage-blind" or mode in EV.STAGE_BLIND_MODES):
        return True
    return label.startswith("baseline:grand_mean") or label.startswith("baseline:resample_blind")


def stage_metrics(units, n_perm_x, n_perm_h, folds=(0, 1, 2)):
    out = {}
    rep = EV.repeat_of(folds[0])
    for label, us in units.items():
        if stage_blind(label) and not label.startswith("baseline:"):
            continue
        byseed = {}
        for f, s, flat, r, npz in us:
            if f in folds:
                byseed.setdefault(s, {})[f] = npz
        for s, d in byseed.items():
            if sorted(d) != sorted(folds) or not all(p.exists() for p in d.values()):
                continue
            paths = [d[f] for f in sorted(d)]
            key = f"{label}|seed={s}"
            blk = {}
            try:
                xf = EV.xfit_from_npz(paths, n_perm_x, MX.stable_seed("xfit", PANEL, rep))
            except ValueError as ex:
                xf = None; blk["xfit_error"] = str(ex)
            if xf is not None:
                sc = xf["stage"]["score"]
                blk.update(xfit_score=sc.get("value"), xfit_score_z=sc.get("z"), xfit_p=sc.get("p_one_sided", sc.get("p")),
                           xfit_pooled=xf["stage"]["pooled_pairs"].get("value"), xfit_target_score=xf["target"]["score"].get("value"))
            hoc = heldout_contrast_agreement(paths, n_perm_h, MX.stable_seed("hoc", PANEL, rep))
            if hoc is not None:
                blk.update(hoc_spearman=hoc["spearman"]["value"], hoc_spearman_z=hoc["spearman"]["z"],
                           hoc_spearman_p=hoc["spearman"]["p_one_sided"], hoc_pearson=hoc["pearson"]["value"],
                           hoc_pearson_z=hoc["pearson"]["z"], hoc_detail=hoc)
            out[key] = blk
    return out


def row(units, label, stage, folds=(0, 1, 2)):
    us = [u for u in units.get(label, []) if u[0] in folds]
    if not us:
        return None
    r = {"label": label, "n_folds": len({u[0] for u in us}), "seeds": sorted({u[1] for u in us if u[1] is not None})}
    for m in ROW_METRICS + ["dam_sign_frac", "dam_signed_perm_z", "dam_n_present"]:
        fm = EV._fold_means({(label, PANEL): [(u[0], u[1], u[2], u[3]) for u in us]}, label, PANEL, m)
        m_, se, _, _, nf, _ = EV.repeat_balanced(fm)
        r[m] = m_; r[m + "__se"] = se
    if stage_blind(label) and not label.startswith("baseline:"):
        return r
    vals = {}
    for key, blk in stage.items():
        lab, _, sd = key.rpartition("|seed=")
        if lab == label:
            for k, v in blk.items():
                if k in STAGE_ROW_METRICS and v is not None:
                    vals.setdefault(k, []).append(v)
    for k in STAGE_ROW_METRICS:
        if k.startswith(("xfit", "hoc")):
            r[k] = float(np.mean(vals[k])) if vals.get(k) else None
    return r


def paired(units, stage, label, ref_label, metric, folds=(0, 1, 2), seed_map=None):
    """Mean over folds (repeat-balanced) of label - ref on common (fold, seed); for coverage the
    difference of |cov - 0.8|; for stage metrics per seed.  seed_map: unit seed -> ref seed (None for baselines)."""
    def val(m, v):
        if v is None:
            return None
        return abs(v - 0.8) if m == "pc_coverage80_rep" else v
    if metric.startswith(("xfit", "hoc")):
        a = {k.rpartition("|seed=")[2]: b.get(metric) for k, b in stage.items() if k.rpartition("|seed=")[0] == label}
        b = {k.rpartition("|seed=")[2]: bb.get(metric) for k, bb in stage.items() if k.rpartition("|seed=")[0] == ref_label}
        common = [s for s in a if s in b and a[s] is not None and b[s] is not None]
        if not common:
            return None
        d = [a[s] - b[s] for s in common]
        bv = [v for v in b.values() if v is not None]
        se = float(np.sqrt(2 * np.var(bv, ddof=1) / len(common))) if len(bv) >= 2 else None
        return {"mean_difference": float(np.mean(d)), "n_pairs": len(common), "seed_noise_se": se,
                "se_source": "reference seed variance" if se is not None else None}
    mine = {(u[0], u[1]): val(metric, u[2].get(metric)) for u in units.get(label, []) if u[0] in folds}
    ref = {(u[0], u[1]): val(metric, u[2].get(metric)) for u in units.get(ref_label, []) if u[0] in folds}
    keys = [k for k in mine if k in ref and mine[k] is not None and ref[k] is not None]
    if not keys:
        return None
    byf = {}
    for k in keys:
        byf.setdefault(k[0], []).append(mine[k] - ref[k])
    fm = {f: float(np.mean(v)) for f, v in byf.items()}
    m, se_f, _, _, nf, _ = EV.repeat_balanced(fm)
    out = {"mean_difference": m, "n_folds": nf, "n_pairs": len(keys), "n_lower": int(sum(v < 0 for v in fm.values())),
           "per_fold": fm, "se_over_folds": se_f}
    if keys[0][1] is not None:
        uu = {(l, PANEL): [(u[0], u[1], {metric: val(metric, u[2].get(metric))}, u[3]) for u in units.get(l, [])]
              for l in (label, ref_label)}
        se, src = EV.seed_noise_se(uu, label, ref_label, PANEL, list(folds), metric)
        out["seed_noise_se"] = se; out["se_source"] = src
    return out


def aggregate(out_root, opts, log, write_root=None, folds=None):
    """Aggregate the units under out_root over ``folds`` (default --agg-folds) into write_root (default out_root)."""
    out_root = Path(out_root)
    write_root = Path(write_root) if write_root is not None else out_root
    write_root.mkdir(parents=True, exist_ok=True)
    units = collect(out_root)
    folds = tuple(folds) if folds is not None else tuple(int(x) for x in opts.agg_folds.split(","))
    stage = stage_metrics(units, opts.xfit_permutations, opts.hoc_permutations, folds)
    rows = [r for r in (row(units, l, stage, folds) for l in sorted(units)) if r]
    # baselines: each external variant against its SEA-AD-only version
    bpaired = {}
    for l in sorted(units):
        if not l.startswith("baseline:") or "+" not in l:
            continue
        nm = l.removeprefix("baseline:").split("+")[0]
        ref = "baseline:" + ("ridge" if nm == "ridge_cohort" else nm)
        bpaired[l] = {m: paired(units, stage, l, ref, m, folds) for m in ["pb_rmse", "pb_rmse_recal", "pb_dev_pearson", "xfit_score",
                                                                           "hoc_spearman", "meanlog_rmse_gene", "pc_energy_rep"]}
        bpaired[l]["reference"] = ref
    # fits: hvg2000x unit vs matching hvg2000 unit
    comp = {}
    refs = {}
    for p in sorted(out_root.glob("ref_map_fold*.json")):
        refs.update(json.loads(p.read_text()))
    for l, ref in sorted(refs.items()):
        comp[l] = {"reference": ref, **{m: paired(units, stage, l, ref, m, folds) for m in COMPARE_METRICS}}
    # review F2: gate -- every hvg2000x (fold, seed) needs its hvg2000 reference before a comparison is read
    missing_refs = {}
    for l in units:
        if "~x" not in l or l.startswith("baseline:") or "#asmadrc" in l or is_ensemble_label(l):   # asmadrc: diagnostic
            continue
        ref = refs.get(l)
        have = {(u[0], u[1]) for u in units.get(ref, [])} if ref else set()
        miss = sorted({(u[0], u[1]) for u in units[l] if u[0] in folds} - have)
        if miss:
            missing_refs[l] = {"reference": ref, "missing_fold_seed": [list(m) for m in miss]}
    # review F1: within-hvg2000x pairings (naive pooled; extperm control)
    xunits = {}
    for l in units:
        if "~x" in l and not l.startswith("baseline:") and not is_ensemble_label(l):
            arm, _, rest = l.partition("~x")
            xunits[l] = (frozenset(f for f in arm.split(",") if f and f != "base"), rest)
    comp_x = {}
    for l, (fl, rest) in sorted(xunits.items()):
        naive = frozenset(f for f in fl if not f.split("@")[0].startswith(COHORT_FLAG_PREFIXES))
        for kind, want in (("vs_naive_pooled", naive if naive != fl else None),
                           ("vs_extperm_control", fl | {"extperm"} if "extperm" not in fl else None)):
            if want is None:
                continue
            ref = [m for m, (f2, r2) in xunits.items() if f2 == want and r2 == rest]
            if ref:
                comp_x.setdefault(l, {})[kind] = {"reference": ref[0],
                                                   **{m: paired(units, stage, l, ref[0], m, folds) for m in COMPARE_METRICS}}
    # confirmation extension: hvg2000x seed ensembles vs the matching hvg2000 seed ensemble (same fold, mode, n)
    comp_ens = {}
    for l in sorted(units):
        if not is_ensemble_label(l) or "~x" not in l.partition(" [")[0]:
            continue
        base, _, suffix = l.partition(" [")
        ref = refs.get(base)
        if ref is None:
            arm, tag, mode = parse_unit_label(base)
            cands = [u for u in units if is_ensemble_label(u) and u.endswith(" [" + suffix)
                     and "~x" not in u.partition(" [")[0]
                     and parse_unit_label(u.partition(" [")[0])[0] == strip_cohort(arm)
                     and parse_unit_label(u.partition(" [")[0])[2] == mode]
            ref_e = sorted(cands)[0] if cands else None
        else:
            ref_e = f"{ref} [{suffix}"
        if ref_e in units:
            comp_ens[l] = {"reference": ref_e, **{m: paired(units, stage, l, ref_e, m, folds) for m in COMPARE_METRICS}}
    agg = {"ext_eval_version": EXT_EVAL_VERSION, "r3_evaluator_version": EV.EVALUATOR_VERSION, "folds": list(folds),
           "rows": rows, "stage": {k: {kk: vv for kk, vv in v.items() if kk != "hoc_detail"} for k, v in stage.items()},
           "stage_detail": {k: v.get("hoc_detail") for k, v in stage.items()},
           "baseline_paired_vs_seaad_only": bpaired, "comparison_vs_hvg2000": comp, "comparison_within_x": comp_x,
           "missing_references": missing_refs, "comparison_ensembles_vs_hvg2000": comp_ens,
           "comparison_gate": ("READ comparison_vs_hvg2000 ONLY when missing_references is empty for the units "
                               "compared (r3 pbulk hvg2000 folds 0-2 x seeds 0-1 complete; review F2)"),
           "notes": ["every metric is on SEA-AD held-out donors of hvg2000 folds (same genes, space, metrics as rounds 1-3)",
                     "*_recal: SEA-AD train calibration (default); *_recal_pooled: SEA-AD train + external targets",
                     "train_ext_*: fit on external TRAIN donors (ext view); train_seaad_*: SEA-AD train donors",
                     "baseline mean_log/nonzero/second-moment/PC metrics masked as in r3 (MEANINGFUL)",
                     "hoc_*: pooled held-out stage-contrast agreement with a within-fold stage-label permutation null",
                     "dam_*: descriptive only on hvg2000 (3-5 listed genes, all up)"]}
    EV.write_json(write_root / "aggregate_ext.json", agg)
    _write_tables(write_root, rows, bpaired, comp)
    if missing_refs:
        with open(write_root / "summary_ext.txt", "a") as fh:
            fh.write("\nGATE (review F2): hvg2000 references missing; do not read these comparisons yet:\n")
            for l, v in sorted(missing_refs.items()):
                fh.write(f"- {l} (ref {v['reference']}): missing fold/seed {v['missing_fold_seed']}\n")
    if comp_ens:
        with open(write_root / "summary_ext.txt", "a") as fh:
            fh.write("\nSeed ensembles: hvg2000x minus matching hvg2000 ensemble (paired by fold; no seed-noise SE)\n\n")
            for l, x in comp_ens.items():
                parts = [f"{m} {x[m]['mean_difference']:+.4f} ({x[m]['n_lower']}/{x[m]['n_folds']} lower)"
                         for m in ("pb_rmse", "pc_energy_rep", "meanlog_rmse_gene") if x.get(m)]
                fh.write(f"- {l} vs {x['reference']}: " + "; ".join(parts) + "\n")
    if comp_x:
        with open(write_root / "summary_ext.txt", "a") as fh:
            fh.write("\nWithin hvg2000x (review F1): unit minus naive pooled / minus its extperm control "
                     "(paired, same fold/seed/mode; +-SE = seed-noise SE)\n\n")
            for l, d in comp_x.items():
                for kind, x in d.items():
                    parts = [f"{m} {x[m]['mean_difference']:+.4f}" + (f"+-{x[m]['seed_noise_se']:.4f}" if x[m].get("seed_noise_se") else "")
                             for m in COMPARE_METRICS if x.get(m)]
                    fh.write(f"- {l} {kind} ({x['reference']}): " + "; ".join(parts) + "\n")
    log(f"aggregate: {len(rows)} rows, {len(bpaired)} baseline pairs, {len(comp)} comparisons, {len(comp_x)} within-x, "
        f"{len(comp_ens)} ensemble comparisons -> {write_root}")
    return agg


def aggregate_by_repeat(out_root, opts, log):
    """One aggregation per repeat with any unit (folds 0-2, 3-5, 6-8) into out_root/by_repeat/rep{r}/."""
    out_root = Path(out_root)
    have = {u[0] for us in collect(out_root).values() for u in us}
    reps = sorted({EV.repeat_of(f) for f in have})
    out = {}
    for r in reps:
        folds = tuple(EV.folds_of_repeat(r))
        out[r] = aggregate(out_root, opts, log, write_root=out_root / "by_repeat" / f"rep{r}", folds=folds)
    return out


def _f(v, nd=4):
    return "" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def _write_tables(out_root, rows, bpaired, comp):
    cols = ["label", "n_folds"] + ROW_METRICS + ["xfit_score", "xfit_score_z", "xfit_p", "hoc_spearman", "hoc_spearman_z",
                                                   "hoc_spearman_p", "hoc_pearson", "dam_sign_frac", "dam_signed_perm_z"]
    with open(out_root / "baselines_ext.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(cols)
        for r in rows:
            w.writerow([_f(r.get(c), 6) for c in cols])
    lines = ["SEA-AD held-out donors, hvg2000 folds; lower is better for rmse/energy; z/p from permutation nulls", ""]
    short = ["pb_rmse", "pb_rmse_recal", "pb_rmse_recal_pooled", "train_ext_pb_rmse", "xfit_score", "xfit_score_z", "xfit_p",
             "hoc_spearman", "hoc_spearman_z", "hoc_spearman_p"]
    lines.append("| unit | " + " | ".join(short) + " |")
    lines.append("|" + "---|" * (len(short) + 1))
    for r in rows:
        lines.append(f"| {r['label']} | " + " | ".join(_f(r.get(c)) for c in short) + " |")
    lines += ["", "Baselines with external donors minus the same baseline on SEA-AD donors only (mean over folds; n lower)", ""]
    for l, d in bpaired.items():
        parts = []
        for m in ["pb_rmse", "pb_rmse_recal", "xfit_score", "hoc_spearman"]:
            x = d.get(m)
            if x:
                parts.append(f"{m} {x['mean_difference']:+.4f}" + (f" ({x['n_lower']}/{x['n_folds']} lower)" if "n_lower" in x else ""))
        lines.append(f"- {l} vs {d['reference']}: " + "; ".join(parts))
    if comp:
        lines += ["", "hvg2000x units minus matching hvg2000 units (paired, same fold/seed/mode; +-SE = r3 seed-noise SE)", ""]
        with open(out_root / "comparison_ext.csv", "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["unit", "reference", "metric", "mean_difference", "seed_noise_se", "n_pairs", "n_lower"])
            for l, d in comp.items():
                parts = []
                for m in COMPARE_METRICS:
                    x = d.get(m)
                    if not x:
                        continue
                    w.writerow([l, d["reference"], m, x["mean_difference"], x.get("seed_noise_se"), x["n_pairs"], x.get("n_lower")])
                    se = x.get("seed_noise_se")
                    parts.append(f"{m} {x['mean_difference']:+.4f}" + (f"+-{se:.4f}" if se else ""))
                lines.append(f"- {l} vs {d['reference']}: " + "; ".join(parts))
    (out_root / "summary_ext.txt").write_text("\n".join(lines) + "\n")


# ==========================================================================
# Reproduction check against results_r3
# ==========================================================================
SKIP_KEYS = {"generation_seconds", "score_seconds", "ext_eval_version", "ext_signature", "variant"}


def compare_json(a, b, path="", tol=1e-8, out=None, skip=SKIP_KEYS):
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in set(a) | set(b):
            if k in skip:
                continue
            if k not in a or k not in b:
                out.append(f"{path}/{k}: missing on one side")
                continue
            compare_json(a[k], b[k], f"{path}/{k}", tol, out, skip)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} vs {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                compare_json(x, y, f"{path}[{i}]", tol, out, skip)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if abs(a - b) > tol * max(1.0, abs(b)):
            out.append(f"{path}: {a} vs {b}")
    elif a != b:
        out.append(f"{path}: {a!r} vs {b!r}")
    return out


def reproduce(out_root, ref_root, folds, log):
    report = {}
    for k in folds:
        for name in BASELINE_NAMES:
            a = Path(out_root) / "baselines" / PANEL / f"fold{k}" / f"{name}.json"
            b = Path(ref_root) / "baselines" / PANEL / f"fold{k}" / f"{name}.json"
            if not (a.exists() and b.exists()):
                report[f"fold{k}/{name}"] = "missing"
                continue
            diffs = compare_json(json.loads(a.read_text()), json.loads(b.read_text()))
            na, nb = np.load(a.with_suffix(".npz")), np.load(b.with_suffix(".npz"))
            for key in nb.files:
                x, y = na[key], nb[key]
                if x.dtype.kind in "fc":
                    if x.shape != y.shape or not np.allclose(x, y, rtol=0, atol=1e-8, equal_nan=True):
                        diffs.append(f"npz {key} differs")
                elif not np.array_equal(x, y):
                    diffs.append(f"npz {key} differs")
            report[f"fold{k}/{name}"] = "identical (1e-8)" if not diffs else diffs[:10]
            log(f"reproduce fold{k}/{name}: {report[f'fold{k}/{name}'] if diffs else 'ok'}")
    return report


# ==========================================================================
# CLI
# ==========================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["run", "reproduce", "aggregate"])
    ap.add_argument("--out", default=str(RUNS / "results_r4"))
    ap.add_argument("--caches-root", default=str(RUNS / "caches"))
    ap.add_argument("--store", default=str(EXT_STORE))
    ap.add_argument("--folds", default="0,1,2")
    ap.add_argument("--agg-folds", default="0,1,2")
    ap.add_argument("--variants", default="sea,ext,extoff")
    ap.add_argument("--baselines", default=None, help="comma list of baseline names (default all r3 baselines + ridge_cohort)")
    ap.add_argument("--no-baselines", action="store_true")
    ap.add_argument("--fits-root", action="append", default=None, help="[TAG=]DIR with hvg2000x fits (default r4=fits_r4)")
    ap.add_argument("--modes-root", action="append", default=[])
    ap.add_argument("--ref-root", action="append", default=None, help="[TAG=]DIR searched in order for hvg2000 references")
    ap.add_argument("--ext-cap", type=int, default=DEFAULT_EXT_CAP, help="max cells per external donor (0 = all)")
    ap.add_argument("--pc-repeats", type=int, default=EV.DEFAULT_PC_REPEATS)
    ap.add_argument("--cell-repeats", type=int, default=EV.DEFAULT_CELL_REPEATS)
    ap.add_argument("--ext-pc-repeats", type=int, default=4)
    ap.add_argument("--ext-cell-repeats", type=int, default=1)
    ap.add_argument("--dam-draws", type=int, default=EV.DAM_NULL_DRAWS)
    ap.add_argument("--xfit-permutations", type=int, default=EV.XFIT_PERMUTATIONS)
    ap.add_argument("--hoc-permutations", type=int, default=HOC_PERMUTATIONS)
    ap.add_argument("--reference-results", default=str(RUNS / "results_r3"))
    ap.add_argument("--ref-arm", action="append", default=[],
                    help="score every complete hvg2000 unit of this arm in the --ref-root roots (confirmation extension)")
    ap.add_argument("--ensemble", action="append", default=[], help="seed-ensemble unit label ARM~TAG[#MODE] (repeatable)")
    ap.add_argument("--ensemble-preset", action="append", default=[], choices=sorted(ENSEMBLE_PRESETS))
    ap.add_argument("--ensemble-seeds", default="0,1", help="seeds pooled in every ensemble (all must be complete)")
    ap.add_argument("--no-fits", action="store_true", help="run: skip single hvg2000x fits (ensembles / --ref-arm only)")
    ap.add_argument("--by-repeat", action="store_true", help="aggregate: one aggregation per repeat into by_repeat/rep{r}/")
    args = ap.parse_args(argv)
    check_out_dir(args.out)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    EV.set_torch_threads()

    def log(msg):
        print(json.dumps({"t": time.strftime("%H:%M:%S"), "msg": msg}), flush=True)

    folds = [int(x) for x in args.folds.split(",")]
    if args.command == "reproduce":
        rep = reproduce(out, args.reference_results, folds, log)
        EV.write_json(out / f"reproduce_r3_folds{''.join(map(str, folds))}.json", rep)
        return rep
    if args.command == "aggregate":
        return aggregate_by_repeat(out, args, log) if args.by_repeat else aggregate(out, args, log)
    variants = [v for v in args.variants.split(",") if v]
    jobs = baseline_jobs(variants)
    if args.baselines:
        want = set(args.baselines.split(","))
        jobs = [j for j in jobs if j[0] in want]
    fits_roots = args.fits_root or ["r4=" + str(RUNS / "fits_r4")]
    xrecs, incomplete = scan_x_fits(fits_roots, args.modes_root, set(folds))
    if args.no_fits:
        xrecs = []
    ref_roots = args.ref_root or REF_ROOTS
    ens_labels = list(dict.fromkeys([l for p in args.ensemble_preset for l in ENSEMBLE_PRESETS[p]] + list(args.ensemble)))
    for l in ens_labels:
        parse_unit_label(l)
    ens_seeds = [int(x) for x in args.ensemble_seeds.split(",") if x != ""]
    arm_recs = scan_ref_arm_fits(ref_roots, args.ref_arm, set(folds)) if args.ref_arm else []
    errors, no_ref, ens_done, ens_missing = [], [], [], []
    for k in folds:
        ref_map = {}
        ctx = FoldCtx(args.caches_root, k, args, args.store)
        if any(v != "sea" for v in variants) or xrecs:
            ctx.write_receipt(out)
        if not args.no_baselines:
            for name, v in jobs:
                try:
                    evaluate_baseline(ctx, name, v, out, log)
                except Exception as ex:      # keep going; reported
                    errors.append({"fold": k, "baseline": name, "variant": v, "error": f"{type(ex).__name__}: {ex}"})
                    log(f"ERROR baseline {name}/{v} fold{k}: {ex}")
        for rec in [r for r in xrecs if r["fold"] == k]:
            try:
                evaluate_x_fit(rec, ctx, out, log)
                ref = find_reference(rec, args.ref_root or REF_ROOTS)
                if ref is None:
                    no_ref.append({"unit": x_label(rec), "fold": k, "seed": rec["seed"], "note": "no matching hvg2000 unit"})
                    continue
                ref_label, ref_recs = ref
                for rr in ref_recs:
                    evaluate_ref(rr, ctx, out, log)
                ref_map[x_label(rec)] = ref_label
            except Exception as ex:
                errors.append({"fit": rec["gen"], "error": f"{type(ex).__name__}: {ex}"})
                log(f"ERROR fit {rec['gen']}: {ex}")
        for rr in [r for r in arm_recs if r["fold"] == k]:
            try:
                evaluate_ref(rr, ctx, out, log)
            except Exception as ex:
                errors.append({"ref_fit": rr["gen"], "error": f"{type(ex).__name__}: {ex}"})
                log(f"ERROR ref {rr['gen']}: {ex}")
        if ens_labels:
            try:
                d, m = run_ensembles(ens_labels, ctx, out, log, args, fits_roots, ref_roots, ens_seeds)
                ens_done += d; ens_missing += m
            except Exception as ex:
                errors.append({"ensembles_fold": k, "error": f"{type(ex).__name__}: {ex}"})
                log(f"ERROR ensembles fold{k}: {ex}")
        if ref_map:
            EV.write_json(out / f"ref_map_fold{k}.json", ref_map)
        del ctx
    rep = {"errors": errors, "no_reference": no_ref, "incomplete_fits": incomplete}
    if ens_labels or args.ref_arm:
        rep.update(ensembles_done=ens_done, ensembles_missing=ens_missing, ref_arm_units=len(arm_recs))
    EV.write_json(out / f"run_errors_folds{''.join(map(str, folds))}.json", rep)
    log(f"done folds {folds}: {len(errors)} errors")
    return errors


if __name__ == "__main__":
    main()
