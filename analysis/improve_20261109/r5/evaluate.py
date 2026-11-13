#!/usr/bin/env python3
"""Round-3 evaluator for improve_20261109 (R7 in r3/PROTOCOL_R3.md).

Round-3 copy of r2/evaluate.py.  The scoring code (score_generated and
everything it calls, baselines except the new ``resample_blind``) is
unchanged, so every round-2 per-unit, row and paired value is reproduced
(test_evaluate_r3.py ``reproduce`` checks results_r2_interim to 1e-8).
Round-3 additions:

* Inputs.  Default fit roots: round-1 ``fits`` (untagged), ``r2=fits_r2``,
  ``r3=fits_r3`` (``TAG=DIR`` names units ``arm~TAG``).  ``--modes-root``
  accepts ``TAG=DIR`` too and otherwise inherits the tag of the
  ``--fits-root`` that contains the receipt's fit (so mode files made with
  ``generate.py --out`` on round-2 fits pair with ``arm~r2``).  Arm strings
  may carry ``t.``/``g.``/``d.`` flags (``genemean,t.anneal@0.1``); they are
  plain text here.  Every generation mode is its own unit, including
  ``raw``, ``ema@D``, ``snapens``/``snapens@K``, ``snap@U``,
  ``snapema@U:D``, ``stagehalf``, ``biascal`` and '+'-composed modes
  (``biascal+snapens``).  Longer runs: the fit's run length is config
  ``updates`` (horizon ``recipe_total_steps``).
* Stage kind (``stage_mode`` per row; extends repair FID-3/S4): the round-2
  name rule applied to every '+' component; else a ``stage_blind: true`` in
  the mode receipt or fit config; else, for mode files that store the
  generated ``conditions``, the OLS slope of the per-target mean generated
  ADNC code on the target's own code (< 0.05 stage-blind, < 0.95
  'stage-shrunk', e.g. ``stagehalf``).  Only stage-blind units lose their
  stage metrics.  ``baseline:resample_blind`` is stage-blind.
* ``vs_control`` paired block: a label with t./g./d. flags is paired with
  the same label without them (same tag, mode and ensemble suffix): the
  round-3 single-factor control (PROTOCOL_R3 addition 5).
* ``curves.json`` (build_curves): score-versus-updates per fit from
  ``snap@U`` / ``snapema@U:D`` / ``raw`` / ``ema@D`` / default units, with
  late slopes, plus fold-balanced aggregate curves and baseline references.
* ``targets.json`` (build_targets): T1 (T1a raw / T1b biascal) against
  ``baseline:grand_mean`` + 0.005 and T2 (6 metrics against
  ``baseline:resample``; count, pass, margin, worst miss) per label and fold
  group (folds 0-2, folds 3-8; common folds with each reference), per seed
  for single-fit labels, T3 per unit, stage-blind units listed separately.
  ``*_recal`` metrics are reported in the rows but never used for T1.
* ``leaderboard.txt`` (write_leaderboard): units ranked by pb_rmse with
  folds, seeds, T1, T2 k/6, stage kind, primary read-out flag
  (primary_readout, PROTOCOL_R3 addition 4) and paired differences to
  round-1 ``base``, the best round-2 unit and the round-3 control; reference
  rows for the baselines and the floors of ``floor.json`` (r3/floor.py;
  ``--floor PATH``, default ``<out>/floor.json``).
* Output guard: inside runs/improve_20261109 only ``results_r3`` and
  ``smoke_r3`` may be written.
* ``--ensemble-arm PATTERN`` (repeatable): seed ensembles only for matching
  units (fnmatch on the unit name); default all, as in round 2.
* ``baseline:resample_blind`` (r3/baselines.py): ideal stage-blind generator
  emulation through the same read-out (M = 32, K = 256).  On folds 0-2 it
  scores pb_rmse 0.474 against grand_mean 0.464: the M = 32 read-out alone
  costs about 0.011 (r3/floor.py route D), so a generator that reproduces
  the real between-donor spread cannot reach T1 (<= 0.469) from 32
  generated donors per target.
* EVALUATOR_VERSION r3-v1 (per-unit JSONs gain ``r3_condition_stage``; no
  score value changes).

Repair after the round-3 adversarial review (2026-11-11; no score value changes)
------------------------------------------------------------------------------
* E2.  dCtrl / ``vs_control`` pairs the SAME read-out, with these matched mappings
  (control_label): ``snap@20000`` -> the control's default (its update-20000
  state); ``snapema@20000:D`` -> the control's ``ema@D``; ``snapens@5`` ->
  the control's ``snapens`` (5 snapshots at 1000-update spacing + final:
  the window-matched read-out of a longer run, TR-2/E4); ``nostage`` arms
  -> the control's STAGE-BLIND read-outs (``snapens`` -> ``stageblind``,
  default -> ``condmix@1``; no pair otherwise, E9b).  NEW column dCtrlP /
  targets entry ``vs_control_primary``: the arm's PRIMARY read-out (addition
  4) against the control's primary (``snapens``; ``stagehalf``;
  ``biascal@stagehalf``; ``stageblind`` for nostage), so annealed and
  native-EMA primaries (one final state) are compared with what the control
  is actually read with, not with the control's single raw state.
* E5.  Seed noise.  dCtrlP is printed with +-2 SE of a paired single-seed
  difference: SE^2 = (1/F^2) sum_f 2 s^2 / k_f, s^2 = the control's
  pooled within-fold seed variance of the metric when the control has >= 2
  seeds in some fold, else the declared round-2 prior
  ``SEED_SD_PRIOR`` (pb_rmse SD of a single snapens read-out between
  seeds, from round-2 seed pairs on folds 0-2: see its comment).  One
  seed on three folds resolves differences of about 2 SE only; stage-1
  nulls are uninformative (README_R3 section 7).
* E9.  fixedstage fits whose offset table is numerically zero carry
  ``stage_blind_generator`` (arm info) and their units are listed
  STAGE-BLIND; targets.json states that a T1a pass through ``stagehalf``
  is half stage-blind (16 of 32 generated donors get the stage mixture).
* E11e.  T2 ``missing`` lists metrics a unit lacks (they count as not
  passed); the leaderboard shows them when there is no numeric miss.
* E12.  targets.json declares that the T1 reference (grand_mean, v4 residual
  centring) was itself chosen in round 1 on held-out scores (v3 gave 0.500,
  v4 0.464): this makes T1 stricter, not looser.
* E8.  Selection history (round 3): see S3 below and README_R3.txt section 7.

Round-2 docstring
-----------------
Round-2 evaluator for improve_20261109 (F8 in r2/PROTOCOL_R2.md).

Round-2 copy of ``../evaluate.py``.  Every round-1 metric is computed by the
round-1 code path unchanged, so on the round-1 fits the round-1 numbers are
reproduced (test_evaluate_r2.py ``reproduce`` checks every per-fit /
ensemble / baseline JSON value of results_screen48 to 1e-8).  Additions:

Inputs
------
* ``--fits-root`` may be repeated (default: runs/improve_20261109/fits and
  fits_r2).  Fit folders are ``{root}/{arm}/{panel}/fold{k}/seed{s}/``; arm
  strings may carry ``@`` parameters; inner folds 0-8 (caches under
  ``caches/{panel}/fold{k}``; folds from ``r2/folds_r2.json`` when present,
  else ``../folds.json``; repeat r = inner folds 3r..3r+2).  The same
  (arm, panel, fold, seed) found in two roots: the first root listed wins and
  the other is listed under ``shadowed`` in aggregate.json.
* Every generation-mode file ``generated_{mode}.npz`` in a fit folder (with
  its ``generated_{mode}.receipt.json``) is its own evaluation unit named
  ``{arm}#{mode}``.  ``--modes-root`` (repeatable) adds mode files written
  elsewhere with ``r2/generate.py --out`` (e.g. on round-1 fits into
  posthoc_r2/): every ``*.receipt.json`` with ``fit`` and ``mode`` keys and a
  sibling ``.npz`` becomes unit ``{arm of the fit}#{mode}``.
* Development-donor guard (PROTOCOL hard rule): before any count file is
  opened, every donor named by a cache manifest / metadata, and every target
  of a generated file, must be one of the 37 development donors and not a
  sealed donor (raises PermissionError otherwise).

Primary metrics (held-out donors of each inner fold)
----------------------------------------------------
Round-1 metrics as before (pb_rmse, meanlog_rmse_gene, nonzero_rmse_gene, ...)
plus:

* Train-only recalibrated metrics (``*_recal``): per-gene additive
  correction a_g = mean over the fold's TRAIN donors of (observed - predicted),
  added to every held-out prediction.  pb: observed = full-depth pseudobulk
  log1p(CP10k), predicted = mean over M of generated-pseudobulk log1p(CP10k)
  (the round-1 pb prediction).  mean_log / nonzero: observed and predicted
  per-cell mean log expression / nonzero fraction (depth-matched).  Same code
  for models and baselines.  On train donors the recalibrated numbers are
  in-sample by construction (reported, never a primary metric).
* Second moment (``logvar_*``; only when the generated file has ``sq_log``,
  i.e. round-2 fits, mode files and r2 baselines; None for round-1 files):
  per gene and per generated donor the per-cell variance of
  log1p(panel CP10k) = (sq_log - mean_log^2) * K/(K-1); averaged over M; the
  real value is the donor's per-cell variance over all its cells (ddof=1).
  Depth-matched because generated libraries are drawn from the target's own
  cells.  Error e_g = mean over held-out donors of (generated - real);
  logvar_rmse_gene = sqrt(mean_g e_g^2), logvar_bias = mean_g e_g.  sq_log is
  float16; rounding error after averaging over M is < 1e-3 for typical genes.
* Repeat-averaged PC / cell metrics (``pc_*_rep``, ``cell_energy_rep``,
  ``cell_energy_pooled_rep``): the round-1 metric re-computed for R
  independent real-cell subsample draws (draw r of donor d uses seed
  stable_seed('pc-summary-rep', d, r); cell energy also redraws the generated
  512-cell subsample) and averaged.  Draws depend only on donor and r, never
  on the arm, so all units share them (common random numbers).  ``*_rep_mcse``
  = SD over draws / sqrt(R) (the remaining redraw error of the average).  The
  round-1 single-subsample values (``pc_energy`` etc.) are still reported.
  Default R (``--pc-repeats``, ``--cell-repeats``) was chosen with
  ``--measure-redraw`` so that the single-fit redraw SD of pc_energy /
  cell_energy falls below 0.01 (see DEFAULT_PC_REPEATS).
* Frozen DAM direction score (``dam_*``), definition in
  r2/PROTOCOL_R2.md; gene lists in metrics.DAM_UP / DAM_DOWN (verbatim).  The
  model contrast c_g = mean over the fold's High targets minus mean over its
  Not AD targets (ALL 37 targets, train + held-out: the model's own stage
  effect; held-out targets contribute only their library sizes / covariates)
  of the model's per-target mean per-cell log expression (``mean_log``
  averaged over M; the pseudobulk-profile version is also in the JSON as
  ``dam.pseudobulk``).  Statistics over the n listed genes present:
  dam_sign_frac (fraction with sign = expected direction), dam_signed_mean
  (mean of direction * c_g), dam_signed_z (z of dam_signed_mean against the
  null), dam_heldout_spearman (Spearman over listed genes of c with the
  held-out donors' real contrast; z in dam_heldout_spearman_z; descriptive
  in the same way as hn_spearman_vs_heldout, since a fold has 1-2 held-out
  Not AD donors).  Null: 2000 expression-matched random gene sets (each
  listed gene replaced by a non-listed gene from the same decile of the
  train-donor mean of real mean_log; draws seeded by (panel, fold) only, so
  shared by all units).  ``dam_n_present`` = listed genes on the panel.
  Within this study the score is descriptive (PROTOCOL_R2.md).
* Pooled cross-fitted donor score (``xfit_*``), computed per (unit, panel,
  repeat, seed) when all three inner folds of the repeat are present:
    For fold k of the repeat: gbar_k = mean over fold-k TRAIN donors of the
    observed full-depth pseudobulk log1p(CP10k) profile; the model of fold k
    gives per-target predictions p_t (round-1 pb prediction); mbar_k = mean of
    p_t over fold-k train targets; the model's predicted stage deviation
    delta_k(s) = mean of p_t over fold-k train targets of stage s - mbar_k.
    For every donor d held out in fold k (stage s_d):
        o_d = y_d - gbar_k,  score_d = Pearson over genes(o_d, delta_k(s_d)).
    xfit_score = mean of score_d over the 37 donors of the repeat (each
    development donor is held out exactly once per repeat);
    xfit_pooled = Pearson over all stacked (donor, gene) pairs.
    Null: stage labels permuted among the held-out donors WITHIN each fold
    (2000 permutations, seeded by (panel, repeat) only); xfit_score_z and
    p in aggregate.json.  xfit_target_score: the same with the donor's own
    target prediction p_d - mbar_k in place of delta_k(s_d) (permutation of
    predictions among the fold's held-out donors).
  Not leaky: o_d uses donor d's counts and fold-k train donors; delta_k uses
  only the model trained on fold-k train donors (d was not among them) and
  its predictions for fold-k train conditions.  Nothing derived from d's
  counts enters the prediction side.  Round 1's pooled contrast instead
  compared an average of the three fold models' contrasts (each trained on
  2/3 of the donors) with the contrast of all 37 donors, so the same donors'
  counts were on both sides; it is still reported (``pooled_across_folds``,
  flagged LEAKY) for continuity only.
* ``hn_*_vs_heldout`` and ``ord_spearman_vs_heldout`` (per-fold contrast vs
  the 12-13 held-out donors) are kept in the per-fit JSON and the rows but
  are DESCRIPTIVE ONLY: their null SD (0.11-0.23) exceeds arm differences,
  so they are excluded from all paired tables (DESCRIPTIVE_ONLY).

Ensembles
---------
For every (unit, panel, fold) with S >= 2 seeds (after --seeds): for n = S
and for every n in --ensemble-sizes with 2 <= n < S, the first n seeds
(sorted) form
  ``{unit} [seed-ensemble n={n}]``       all n*32 generated donors pooled;
  ``{unit} [seed-ensemble n={n} M=32]``  M-matched: generated donor m taken
                                         from seed index m mod n (32 donors,
                                         disjoint m positions across seeds),
                                         so it compares with a single fit at
                                         the same number of generated donors.
Ensembles are paired with base ensembles over the SAME seed set per fold.

Baselines
---------
r2/baselines.py (round-1 baselines + sq_log), per (panel, fold).  In rows
and paired tables the mean_log / nonzero / second-moment metrics of a
baseline are masked (None) unless baselines.MEANINGFUL says 'meaningful'
for that group (only ``resample``); PC/cell metrics of profile baselines are
masked as in round 1.

Aggregation (what the SE means)
-------------------------------
Row value = repeat-balanced mean: seeds averaged within a fold, folds
averaged within a repeat (the three folds of a repeat partition the 37
donors), then repeats averaged.  ``__se`` = SE over repeat means when >= 2
repeats are present, else SE over fold means (round 1); ``__se_kind`` says
which; ``__se_folds`` is always the SE over all fold means.  Repeats re-use
the same 37 donors with different splits, and folds of a repeat share
training donors, so both SEs describe split-to-split variability and
understate the uncertainty of a population claim.  Paired comparisons: per
fold the goodness difference (seed-paired, averaged over seeds in the fold);
n_better / n counted over folds; mean_difference repeat-balanced;
se_difference as for rows.  xfit metrics are per repeat (unit = repeat).

Outputs (--out, default runs/improve_20261109/results_r2)
---------------------------------------------------------
  fits/{unit}/{panel}/fold{k}/seed{s}.json (+.npz)
  ensembles/{unit}/{panel}/fold{k}_n{n}[_M32].json (+.npz)
  baselines/{panel}/fold{k}/{name}.json (+.npz), noise_ceiling.json
  real/{panel}/fold{k}.json   model-free DAM reference, gene counts
  aggregate.json, aggregate.csv, summary.txt (markdown tables: one section
  per panel, a like-for-like shared-gene section with both panels'
  shared-gene renormalised metrics, descriptive contrasts, xfit table).
Re-runnable: per-unit JSONs are reused when their inputs (file size+mtime,
cache manifest sha256, evaluator/baseline version, repeat settings) match.

Repair 2026-11-10 (two adversarial reviews; evaluator version r2-v2)
--------------------------------------------------------------------
* S1 (DAM null).  The expression-matched random-gene null is competitive
  and ignores gene-gene correlation: on real hvg2000 fold-0 donors, 400
  random within-High half-splits (no stage contrast) gave P(p<0.05) = 0.235
  with it.  New self-contained null ``metrics.dam_perm_null``: High / Not AD
  labels permuted among the fold's High + Not AD targets (sizes kept), the
  contrast recomputed on the same listed genes (2000 draws, seeded by
  (panel, fold), shared by all units).  Same check: P(p<0.05) = 0.07 / 0.045
  (High / High+Intermediate), uniform deciles.  Headline stage metric:
  ``dam_signed_perm_z`` (and ``dam_sign_frac_perm_z``); ``dam_signed_z``
  (matched-gene null) is DESCRIPTIVE_ONLY.
* S2 (DAM coverage).  A panel's DAM metrics enter paired tables only when it
  has >= DAM_MIN_PRESENT (8) listed genes including >= 1 expected-down gene.
  hvg2000 has 3-5 listed genes, all up (no P2RY12/TMEM119/CX3CR1/...), so on
  hvg2000 every dam_* metric is descriptive only; use xfit_* as the stage
  metric there (the wide panel has 13-15 genes, 4 down).
* FID-3 / S4 (stage-blind modes).  ``stage_blind_mode``: condmix@lam and
  condunk@lam with floor(lam*32+.5) = 32, condshrink@1 and anchorgain@0 give
  every target the same stage input.  Such units are labelled
  'stage-blind': their stage metrics (STAGE_METRICS: hn_*, ord_*, dam_*,
  within-stage spreads, xfit_*) are set to None (no xfit is computed) and
  their pb_rmse is a calibration result (read it against the grand-mean
  baseline / posthoc direct:train_mean, never as an F3 fix).  Partial
  condmix / condunk units are flagged 'stage-mixture' (donor-level
  distribution metrics describe a stage mixture).
* FID-1 (common random numbers).  Every unit records the device its
  generated file was made on (``gen_device``: mode receipt
  ``generation_device``, else the fit receipt's device for files inside a
  fit folder; 'unknown' otherwise).  Paired blocks carry ``crn``: True only
  when every paired unit and its reference were generated on the same known
  device type (CPU and CUDA torch generators differ, so cross-device pairs
  add Monte Carlo noise; no bias).  The summary shows it per paired row.
* S7 (overlapping repeats).  Paired blocks add ``n_better_by_repeat``
  {repeat: [better, n folds]} and ``repeats_majority_better``; the summary
  shows 'b/n f' and, with >= 2 repeats, 'r/R rep' (repeats where most folds
  are better).  The repeats reuse the same 37 donors: a fold count over
  folds 0-8 is not 9 independent replications.
* S3 (selection history).  Many round-2 choices were informed by round-1
  HELD-OUT results on these same 37 donors: (a) condmix@1 on every screen
  arm (best generation-only held-out value; the train-chosen lam was
  0.65-0.9), (b) condmix preferred over condshrink, (c) genemean as the
  partner of both combinations and the extra-fold arm, (d) pbulk motivated
  partly by held-out pseudobulk error quartiles, (e) donorpc2's share bounded
  below the 0.61 that hurt, and covdrop / covbin / stageanchor / stagedrop
  exist because of held-out failures.  Folds 3-8 re-split the same donors,
  each of which was held out in round 1.  Every round-2 held-out number is
  therefore development evidence conditional on round-1 selection; a
  confirmed fix needs donors used in neither round (sealed / external).
  condmix@1 and condmix@0.5 are both run and reported as pre-specified
  candidates.
  Round 3 (repair E8): the control observation pbulk was chosen from round-2
  HELD-OUT primary read-outs (pbulk~r2#snapens 0.547 vs genemean@0.25 0.555,
  a margin decided by one extra seed on one fold; tied at 0.549 on seed 0)
  and from the stage task's held-out biascal@stagehalf (0.487); the stage-1
  arm list was chosen from round-2 held-out results and train-donor
  diagnostics; stagehalf's 0.5 was declared (PROTOCOL addition 4, 05:31)
  before any stagehalf score but informed by round-1/2 condmix results;
  biascal's update rule was changed (unit steps -> Newton steps) AFTER its
  first held-out scoring (biascal@stagehalf v1 0.4918 / 0.4882, final
  0.4894 / 0.4866 for genemean@0.25 / pbulk; about 0.002 better).  Every
  round-3 number is selection-conditioned development evidence.

Usage:
  evaluate.py [--fits-root DIR ...] [--modes-root DIR ...] [--out DIR]
              [--panels hvg2000,wide] [--folds 0,...,8] [--seeds 0,1]
              [--arm PATTERN ...] [--no-baselines] [--no-ensembles] [--no-modes]
  evaluate.py --measure-redraw FIT_DIR --cache CACHE [--draws 64]
"""
from __future__ import annotations

import os
for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    try:
        _n = int(os.environ.get(_v, "3"))
    except ValueError:
        _n = 3
    os.environ[_v] = str(max(1, min(_n, 3)))      # round-2 rule: at most 3 threads
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse
import csv
import fnmatch
import hashlib
import json
import sys
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent            # .../improve_20261109/r3
if str(HERE) in sys.path:
    sys.path.remove(str(HERE))
sys.path.insert(0, str(HERE))
import metrics as MX  # noqa: E402

if Path(MX.__file__).resolve().parent != HERE:
    raise ImportError(f"r3 evaluate imported the wrong metrics module: {MX.__file__}")

EVALUATOR_VERSION = "improve_20261109-evaluator-r3-v1"   # r3: round-3 outputs; scores identical to r2-v2
RUNS = Path("/ml2/alz/runs/improve_20261109")
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
LOGVAR_THRESHOLDS = [0.02, 0.05, 0.1]
FOLDS_JSON_R1 = HERE.parent / "folds.json"
FOLDS_JSON_R2 = HERE / "folds_r2.json"
GEN_KEYS = ["targets", "is_heldout", "stage", "genes", "pb_counts", "mean_log", "nonzero", "pcs", "lib"]
# Repeat counts for the PC / cell metrics, measured with --measure-redraw (48
# draws) on round-1 fits hvg2000 fold 0 seed 0 (base, genemean): single-draw
# SD of held-out pc_energy 0.035 / 0.035 (train 0.025 / 0.024), pc_spread
# 0.028 / 0.033, pc_coverage80 0.011 / 0.010, cell_energy 0.0074 / 0.0082,
# pooled cell energy 0.0067 / 0.0093.  R_pc = 24 brings pc_energy's redraw SD
# to 0.035/sqrt(24) = 0.007 (< 0.01; R = 16 gave a measured MC SE of 0.011 on
# the subset run, so 24 keeps a margin; PC draws are cheap, ~0.05 s each);
# cell metrics are already below 0.01 for one draw, R_cell = 4 halves that
# (0.004) at a quarter of the cost of 16 (~1 s per draw per fit).
DEFAULT_PC_REPEATS = 24
DEFAULT_CELL_REPEATS = 4
DAM_NULL_DRAWS = 2000
DAM_PERMUTATIONS = 2000      # repair S1: stage-label permutations for dam_*_perm_z
DAM_MIN_PRESENT = 8          # repair S2: fewer listed genes (or no down gene) -> dam_* descriptive only
DAM_MIN_DOWN = 1
XFIT_PERMUTATIONS = 2000
FROZEN_OUTPUTS = ["fits", "results", "results_screen48", "screen_analysis", "caches", "results_stale_prescreen"]
R3_WRITABLE = ["results_r5", "smoke_r5"]      # round 5 (r5 trainer task edit): the only evaluator output trees inside RUNS


def set_torch_threads():
    try:
        import torch
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "3")))
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
# Folds, repeats and the development-donor guard
# ==========================================================================
def load_folds():
    """folds_r2.json when present (folds 0-2, development and sealed sets
    checked equal to round 1), else ../folds.json."""
    r1 = json.loads(FOLDS_JSON_R1.read_text())
    if not FOLDS_JSON_R2.exists():
        return r1
    q = json.loads(FOLDS_JSON_R2.read_text())
    if set(q["development_donors"]) != set(r1["development_donors"]):
        raise ValueError("folds_r2.json development donors differ from folds.json")
    if set(q["sealed_evaluation_donors_fold0"]) != set(r1["sealed_evaluation_donors_fold0"]):
        raise ValueError("folds_r2.json sealed donors differ from folds.json")
    for k in range(len(r1["inner_folds"])):
        a, b = r1["inner_folds"][k], q["inner_folds"][k]
        if set(a["train_donors"]) != set(b["train_donors"]) or set(a["heldout_donors"]) != set(b["heldout_donors"]):
            raise ValueError(f"folds_r2.json inner fold {k} differs from round 1")
    return q


def repeat_map():
    """{inner fold: repeat}.  From folds_r2.json 'repeats' when present, else fold // 3."""
    q = load_folds()
    out = {}
    for rep in q.get("repeats", []):
        for k in rep["inner_folds"]:
            out[int(k)] = int(rep["repeat"])
    for f in q["inner_folds"]:
        out.setdefault(int(f["inner_fold"]), int(f["inner_fold"]) // 3)
    return out


def repeat_of(fold):
    return repeat_map().get(int(fold), int(fold) // 3)


def folds_of_repeat(r):
    return sorted(k for k, v in repeat_map().items() if v == r)


def check_development_donors(donors, where):
    """Raise unless every donor is one of the 37 development donors and none is sealed (PROTOCOL hard rule)."""
    q = json.loads(FOLDS_JSON_R1.read_text())
    dev, sealed = set(q["development_donors"]), set(q["sealed_evaluation_donors_fold0"])
    bad = sorted(set(donors) - dev) + sorted(set(donors) & sealed)
    if bad:
        raise PermissionError(f"{where}: donors outside the 37 development donors {bad[:5]}; refusing to read")


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
    log_var: np.ndarray | None = None           # r2: [T,G] per-cell variance of log1p CP10k (ddof=1)

    @property
    def index(self):
        return {d: i for i, d in enumerate(self.donors)}

    def project(self, counts):
        X = np.asarray(counts.toarray() if sp.issparse(counts) else counts, dtype=np.float64)
        L = MX.log_cp10k(X)
        return (((L - self.mean) / self.sd) @ self.W / self.pc_sd).astype(np.float32)


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
    donors, held, stage, raw, pbs, mls, nzs, cells, pcs, libs, lvs = [], [], [], [], [], [], [], [], [], [], []
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
            lvs.append(L.var(0, ddof=1) if len(L) > 1 else np.zeros(L.shape[1]))
            cells.append(Cd.astype(np.float32)); libs.append(lib)
            pcs.append((((L - rf.mean) / rf.sd) @ rf.W / rf.pc_sd).astype(np.float32))
    rf.donors = np.array(donors); rf.is_heldout = np.array(held); rf.stage = np.array(stage)
    rf.raw_c = np.stack(raw); rf.pb = np.stack(pbs); rf.mean_log = np.stack(mls); rf.nonzero = np.stack(nzs)
    rf.cells, rf.pcs, rf.libs = cells, pcs, libs
    rf.log_var = np.stack(lvs)
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
        check_development_donors(np.asarray(f["targets"]).astype(str).tolist(), path)
        g = {k: f[k] for k in GEN_KEYS}
        if "sq_log" in f.files:
            g["sq_log"] = f["sq_log"]
        if "mode" in f.files:
            g["mode"] = str(f["mode"])
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
    if "sq_log" in gen:
        if gen["sq_log"].shape != (T, M, G):
            raise ValueError(f"generated sq_log shape {gen['sq_log'].shape}")
        out["sq_log"] = gen["sq_log"][order]
        if not np.isfinite(out["sq_log"].astype(np.float32)).all():
            raise ValueError("generated sq_log non-finite")
    return out


ENS_KEYS = ["pb_counts", "mean_log", "nonzero", "pcs", "lib", "sq_log"]


def concat_generated(gens):
    """Seed ensemble: pool generated donors across seeds along the M axis
    (sq_log kept only if every member has it)."""
    out = dict(gens[0])
    for k in ENS_KEYS:
        if all(k in g for g in gens):
            out[k] = np.concatenate([g[k] for g in gens], axis=1)
        else:
            out.pop(k, None)
    return out


def mmatched_generated(gens):
    """M-matched seed ensemble: generated donor m comes from member m mod n."""
    n = len(gens)
    M = gens[0]["pb_counts"].shape[1]
    if any(g["pb_counts"].shape[1] != M for g in gens):
        raise ValueError("M-matched ensemble needs equal M")
    src = np.arange(M) % n
    out = dict(gens[0])
    for k in ENS_KEYS:
        if all(k in g for g in gens):
            arr = np.empty_like(gens[0][k])
            for i, g in enumerate(gens):
                arr[:, src == i] = g[k][:, src == i]
            out[k] = arr
        else:
            out.pop(k, None)
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
    """Genes present in this panel's cache for all three inner folds of the
    fold's repeat (sorted).  Repeat 0 = folds 0-2 = round 1."""
    if caches_root is None:
        return None
    sets = []
    for k in folds_of_repeat(repeat_of(real.fold)):
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


class ScoreSettings:
    """Round-2 knobs passed to score_generated (all have round-2 defaults)."""

    def __init__(self, pc_repeats=DEFAULT_PC_REPEATS, cell_repeats=DEFAULT_CELL_REPEATS, dam=None):
        self.pc_repeats = int(pc_repeats)
        self.cell_repeats = int(cell_repeats)
        self.dam = dam          # dict from dam_context(real) or None

    def key(self):
        return {"pc_repeats": self.pc_repeats, "cell_repeats": self.cell_repeats,
                "dam_null_draws": None if self.dam is None else int(len(self.dam["null"])),
                "dam_permutations": DAM_PERMUTATIONS}


def dam_context(real: RealFold, n_draws=DAM_NULL_DRAWS):
    """Listed genes present, directions and the expression-matched null sets
    (level = train-donor mean of real mean_log; seeded by panel and fold)."""
    idx, sgn, missing = MX.dam_genes(real.genes)
    level = real.mean_log[~real.is_heldout].mean(0)
    null = MX.matched_null_sets(idx, level, n_draws, MX.stable_seed("dam-null", real.panel, real.fold)) if len(idx) else np.zeros((0, 0), int)
    return {"idx": idx, "sgn": sgn, "missing": missing, "null": null, "level_bins": 10,
            "present": [str(real.genes[i]) for i in idx]}


def pc_space_repeated(gen, real: RealFold, n_pc, n_cell):
    """Round-1 PC-space / cell-energy metrics averaged over independent
    real-cell subsample draws (module docstring)."""
    held = real.is_heldout; train = ~held
    sets = {"heldout": held, "train": train}
    kpc = gen["pcs"].shape[2]; D = gen["pcs"].shape[3]
    Ypc = MX.pc_bag_summary(gen["pcs"].astype(np.float32))
    acc = {e: {k: [] for k in ["energy_score", "coverage80", "spread_ratio", "within_stage_spread_ratio",
                               "cell_energy_distance", "pooled_cell_energy_distance"]} for e in sets}
    for r in range(n_pc):
        ypc = np.stack([MX.pc_bag_summary(real.pcs[i][MX.det_subsample(len(real.pcs[i]), kpc, MX.stable_seed("pc-summary-rep", d, r))])
                        for i, d in enumerate(real.donors)])
        tr_ref = MX.within_stage_var_trace(ypc[train], real.stage[train])
        for e, mask in sets.items():
            d = MX.distribution_metrics(Ypc[mask], ypc[mask], real.stage[mask], tr_ref)
            for k in ["energy_score", "coverage80", "spread_ratio", "within_stage_spread_ratio"]:
                acc[e][k].append(d.get(k))
    gcs = [gen["pcs"][i].reshape(-1, D).astype(np.float32) for i in range(len(real.donors))]
    for r in range(n_cell):
        for e, mask in sets.items():
            eds, gpool, rpool = [], [], []
            for i in np.where(mask)[0]:
                donor = real.donors[i]
                gc = gcs[i]
                gs = gc[MX.det_subsample(len(gc), CELL_ED_PER_TARGET, MX.stable_seed("ed-gen-rep", donor, r))]
                rs = real.pcs[i][MX.det_subsample(len(real.pcs[i]), CELL_ED_PER_TARGET, MX.stable_seed("ed-real-rep", donor, r))]
                eds.append(MX.cell_energy(gs, rs)); gpool.append(gc); rpool.append(real.pcs[i])
            gp = np.concatenate(gpool); rp = np.concatenate(rpool)
            gp = gp[MX.det_subsample(len(gp), CELL_ED_POOLED, MX.stable_seed("edpool-gen-rep", e, r))]
            rp = rp[MX.det_subsample(len(rp), CELL_ED_POOLED, MX.stable_seed("edpool-real-rep", e, r))]
            acc[e]["cell_energy_distance"].append(float(np.mean(eds)))
            acc[e]["pooled_cell_energy_distance"].append(MX.cell_energy(gp, rp))
    out = {"n_pc_repeats": int(n_pc), "n_cell_repeats": int(n_cell),
           "definition": "round-1 pc_space metrics averaged over R independent real-cell subsample draws "
                         "(seed stable_seed('pc-summary-rep'|'ed-real-rep'|'ed-gen-rep', donor, r)); "
                         "<metric>_draw_sd = SD over draws (single-draw redraw error); <metric>_mcse = draw_sd/sqrt(R)"}
    for e in sets:
        blk = {}
        for k, v in acc[e].items():
            v = np.array([x for x in v if x is not None], dtype=np.float64)
            blk[k] = MX._f(v.mean()) if len(v) else None
            blk[k + "_draw_sd"] = MX._f(v.std(ddof=1)) if len(v) > 1 else None
            blk[k + "_mcse"] = MX._f(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else None
        out[e] = blk
    return out


def score_generated(gen, real: RealFold, subsets, pooled_genes=None, train_only_meta=None, settings: ScoreSettings | None = None):
    """Score an aligned generated set (fit, ensemble or baseline).

    Returns (result dict, arrays dict for the small per-fit npz).  The
    round-1 body is unchanged; round-2 blocks are added (module docstring)."""
    t0 = time.monotonic()
    settings = settings or ScoreSettings()
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
    has_sq = "sq_log" in gen
    if has_sq:
        Kc = max(K, 2)
        gv = (gen["sq_log"].astype(np.float64) - gen["mean_log"].astype(np.float64) ** 2) * (Kc / (Kc - 1))
        gen_lv = np.maximum(gv, 0.0).mean(1)
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
        # r2: train-only additive recalibration (per gene)
        a_pb = MX.additive_recalibration(pred[train], ol[train])
        if not renorm:
            a_ml = MX.additive_recalibration(gen_ml[train][:, idx], real.mean_log[train][:, idx])
            a_nz = MX.additive_recalibration(gen_nz[train][:, idx], real.nonzero[train][:, idx])
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
            # ---- r2 additions (do not change any round-1 value above) ----
            insample = "in-sample (correction fitted on these donors)" if ename == "train" else "train-only correction applied to held-out donors"
            e["pseudobulk_recal"] = MX.pseudobulk_metrics(pred[mask] + a_pb, ol[mask], grand, real.stage[mask], real.donors[mask])
            e["pseudobulk_recal"]["note"] = insample
            if not renorm:
                e["mean_log_recal"] = MX.gene_error_metrics(gen_ml[mask][:, idx] + a_ml, real.mean_log[mask][:, idx], ref_ml, MEANLOG_THRESHOLDS)
                e["nonzero_recal"] = MX.gene_error_metrics(gen_nz[mask][:, idx] + a_nz, real.nonzero[mask][:, idx], ref_ml, NONZERO_THRESHOLDS)
                e["mean_log_recal"]["note"] = e["nonzero_recal"]["note"] = insample
                if has_sq:
                    e["second_moment"] = MX.gene_error_metrics(gen_lv[mask][:, idx], real.log_var[mask][:, idx], ref_ml, LOGVAR_THRESHOLDS)
                    e["second_moment"]["definition"] = ("per-cell variance of log1p(panel CP10k) per gene: generated "
                                                        "(sq_log - mean_log^2)*K/(K-1) averaged over M vs real ddof=1 over all cells")
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
    # ---- r2: repeat-averaged PC / cell metrics ----
    res["pc_space_rep"] = pc_space_repeated(gen, real, settings.pc_repeats, settings.cell_repeats)
    # ---- r2: frozen DAM direction score ----
    dam = settings.dam if settings.dam is not None else dam_context(real)
    pred_all = gen_log_all.astype(np.float64).mean(1)
    res["dam"] = {"genes_present": dam["present"], "genes_missing": dam["missing"],
                  "model_contrast": "High minus Not AD over all T targets of the fold (train + held-out)"}
    for space, P_model, P_real in [("mean_log", gen_ml, real.mean_log), ("pseudobulk", pred_all, obs_log_all)]:
        mc = MX.stage_contrasts(P_model, real.stage)["high_notad"]
        rc = {"heldout": MX.stage_contrasts(P_real[held], real.stage[held])["high_notad"],
              "train": MX.stage_contrasts(P_real[train], real.stage[train])["high_notad"]}
        res["dam"][space] = MX.dam_score(mc, dam["idx"], dam["sgn"], dam["null"], rc)
        if res["dam"][space].get("defined"):      # repair S1: self-contained stage-permutation null
            res["dam"][space]["perm"] = MX.dam_perm_null(P_model, real.stage, dam["idx"], dam["sgn"], DAM_PERMUTATIONS,
                                                         MX.stable_seed("dam-perm", real.panel, real.fold))
    # ---- r2: material for the pooled cross-fitted donor score ----
    mbar = pred_all[train].mean(0)
    sdev = np.full((4, G), np.nan)
    for s in range(4):
        m_s = train & (real.stage == s)
        if m_s.any():
            sdev[s] = pred_all[m_s].mean(0) - mbar
    arrays.update(xf_stage_dev=sdev.astype(np.float32),
                  xf_target_dev=(pred_all[held] - mbar).astype(np.float32),
                  xf_obs_dev=(obs_log_all[held] - obs_log_all[train].mean(0)).astype(np.float32))
    res["has_sq_log"] = bool(has_sq)
    res["score_settings"] = settings.key()
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
ROUND1_HEADLINE = list(HEADLINE) + list(PC_HEADLINE) + list(LC_HEADLINE)
# ---- r2 headline metrics ----
R2_HEADLINE = {
    "pb_rmse_recal": (("pseudobulk_recal", "rmse"), False),
    "pb_dev_pearson_recal": (("pseudobulk_recal", "dev_pearson"), True),
    "meanlog_rmse_gene_recal": (("mean_log_recal", "rmse_gene"), False),
    "nonzero_rmse_gene_recal": (("nonzero_recal", "rmse_gene"), False),
    "logvar_rmse_gene": (("second_moment", "rmse_gene"), False),
    "logvar_bias": (("second_moment", "bias"), "abs0"),
}
PC_REP_HEADLINE = {f"{k}_rep": (v[0], v[1]) for k, v in PC_HEADLINE.items()}
PC_REP_HEADLINE.update({f"{k}_rep_mcse": (v[0] + "_mcse", None) for k, v in PC_HEADLINE.items() if k in ("pc_energy", "cell_energy")})
# dam name -> (space, statistic path, direction)
DAM_HEADLINE = {
    "dam_n_present": (("n_present",), None),
    "dam_n_down_present": (("n_down_present",), None),
    "dam_sign_frac": (("sign_frac", "value"), True),
    "dam_signed_mean": (("signed_mean", "value"), None),
    "dam_signed_z": (("signed_mean", "z"), True),                 # matched-gene (competitive) null: descriptive only
    "dam_signed_perm_z": (("perm", "signed_mean", "z"), True),     # repair S1: stage-permutation null
    "dam_signed_perm_p": (("perm", "signed_mean", "p_one_sided"), None),
    "dam_sign_frac_perm_z": (("perm", "sign_frac", "z"), True),
    "dam_heldout_spearman": (("spearman_vs_{which}", "value"), True),
    "dam_heldout_spearman_z": (("spearman_vs_{which}", "z"), True),
}
XFIT_HEADLINE = {"xfit_score": True, "xfit_score_z": True, "xfit_pooled": True, "xfit_target_score": True}
# Descriptive only: never in paired tables (per-fold contrast vs 12-13 held-out donors, null SD 0.11-0.23).
DESCRIPTIVE_ONLY = {"hn_spearman_vs_heldout", "hn_top10_sign_vs_heldout", "hn_slope_vs_heldout", "ord_spearman_vs_heldout",
                    "dam_heldout_spearman", "dam_heldout_spearman_z",
                    "dam_signed_z"}   # repair S1: competitive matched-gene null, miscalibrated as a stage test
# Repair FID-3 / S4: metrics that describe the model's stage effect; None for stage-blind units.
STAGE_METRIC_PREFIXES = ("hn_", "ord_", "dam_", "xfit_")
STAGE_METRIC_NAMES = {"pbdist_within_stage_spread", "pc_within_stage_spread", "pc_within_stage_spread_rep"}
STAGE_METRIC_KEEP = {"dam_n_present", "dam_n_down_present"}


def is_stage_metric(name):
    b = name.split("__")[-1]
    return b not in STAGE_METRIC_KEEP and (b.startswith(STAGE_METRIC_PREFIXES) or b in STAGE_METRIC_NAMES)


def label_mode(label):
    """Generation mode of a unit / ensemble label ('arm[~tag]#mode [seed-ensemble ...]'), or None."""
    unit = label.partition(" [")[0]
    return unit.split("#", 1)[1] if "#" in unit else None


def stage_blind_mode(mode, M=32):
    """Repair FID-3 / S4: 'stage-blind' if the mode gives every generated donor
    of every target the same stage input (prediction independent of the
    target's stage), 'stage-mixture' if part of each target's donors get a
    stage mixture / unknown stage, else None."""
    if not mode or "@" not in mode:
        return None
    name, _, v = mode.partition("@")
    try:
        x = float(v)
    except ValueError:
        return None
    if name in ("condmix", "condunk"):
        n = int(np.floor(x * M + 0.5))
        return "stage-blind" if n >= M else ("stage-mixture" if n > 0 else None)
    if name == "condshrink":
        return "stage-blind" if x >= 1 else None
    if name == "anchorgain":
        return "stage-blind" if x == 0 else None
    return None


def gen_device(res):
    """Repair FID-1: device type ('cpu' / 'cuda') the unit's generated file was made on, or 'unknown'."""
    def norm(d):
        d = str(d or "")
        return "cuda" if d.startswith("cuda") else ("cpu" if d.startswith("cpu") else None)
    mr = res.get("mode_receipt")
    if res.get("mode") is not None:
        if isinstance(mr, dict):
            d = norm(mr.get("generation_device")) or norm((mr.get("arm_state_tolerance") or {}).get("generation_device"))
            if d:
                return d
        # a mode file inside an r2 fit folder is written by run_screen/trainer on the fit's device
        fr = res.get("fit_receipt")
        gp = Path(str(res.get("generated_path", "")))
        if isinstance(fr, dict) and str(gp.parent) == str(res.get("fit_dir")):
            d = norm(fr.get("device"))
            if d:
                return d + "?"
        return "unknown"
    fr = res.get("fit_receipt")
    if isinstance(fr, dict):
        d = norm((fr.get("generation") or {}).get("generation_device")) or norm(fr.get("device"))
        if d:
            return d
    if res.get("kind") == "baseline":
        return "baseline"
    return "unknown"


def direction(name):
    base = name.split("__")[-1]
    for table in (HEADLINE, PC_HEADLINE, R2_HEADLINE, PC_REP_HEADLINE, DAM_HEADLINE):
        if base in table:
            return table[base][1]
    if base in XFIT_HEADLINE:
        return XFIT_HEADLINE[base]
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
        for name, (path, _) in R2_HEADLINE.items():
            v = _dig(e, path)
            if v is not None or sname == "all":
                flat[pre + name] = v
    pc = result.get("pc_space", {}).get(which, {})
    for name, (key, _) in PC_HEADLINE.items():
        flat[name] = pc.get(key)
    rep = (result.get("pc_space_rep") or {}).get(which, {})
    for name, (key, _) in PC_REP_HEADLINE.items():
        flat[name] = rep.get(key)
    dam = (result.get("dam") or {}).get("mean_log") or {}
    for name, (path, _) in DAM_HEADLINE.items():
        path = tuple(p.format(which=which) for p in path)
        flat[name] = _dig(dam, path) if dam.get("defined") or name == "dam_n_present" else None
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
# Directory scanning
# ==========================================================================
def _fit_parts(gp):
    gp = Path(gp)
    arm, panel, fold, seed = gp.parts[-4], gp.parts[-3], gp.parts[-2], gp.parts[-1]
    return arm, panel, int(fold.removeprefix("fold")), int(seed.removeprefix("seed"))


def scan_fits(fits_roots, modes_roots=(), include_modes=True):
    """Records {unit, arm, mode, panel, fold, seed, dir, gen, root} for every
    complete default generation and mode file; duplicates of the same
    (unit, panel, fold, seed) keep the first root (others in 'shadowed')."""
    found, incomplete, shadowed, seen = [], [], [], {}
    root_tags = []      # r3: (resolved fits root, '~TAG' or '') for mode-root tag inheritance

    def add(rec):
        key = (rec["unit"], rec["panel"], rec["fold"], rec["seed"])
        if key in seen:
            shadowed.append({**rec, "shadowed_by": seen[key]})
            return
        seen[key] = rec["gen"]
        found.append(rec)

    for root in fits_roots:
        # Integration addition (2026-11-10): 'TAG=DIR' labels every unit of that
        # root '{arm}~TAG' (modes '{arm}~TAG#{mode}'), so an r2 refit of a round-1
        # arm (the snapshot-bearing 'genemean' control) is not shadowed by the
        # round-1 fit of the same (arm, panel, fold, seed) and its mode units pair
        # vs_parent with their own default generation.  vs_base still pairs with
        # the untagged 'base'.  A root without '=' behaves exactly as before.
        tag = ""
        if "=" in str(root):
            tag, root = str(root).split("=", 1)
            if not re.fullmatch(r"[A-Za-z0-9_]+", tag):
                raise ValueError(f"bad --fits-root tag {tag!r} (want [A-Za-z0-9_]+=DIR)")
            tag = "~" + tag
        root = Path(root)
        if not root.exists():
            continue
        root_tags.append((root.resolve(), tag))
        for gp in sorted(root.glob("*/*/fold*/seed*")):
            if not gp.is_dir():
                continue
            try:
                arm, panel, k, s = _fit_parts(gp)
            except ValueError:
                continue
            base = {"arm": arm, "panel": panel, "fold": k, "seed": s, "dir": str(gp), "root": str(root)}
            if (gp / "generated.npz").exists() and (gp / "fit_receipt.json").exists():
                add({**base, "unit": arm + tag, "mode": None, "gen": str(gp / "generated.npz")})
            else:
                incomplete.append({**base, "unit": arm + tag,
                                   "missing": [x for x in ["generated.npz", "fit_receipt.json"] if not (gp / x).exists()]})
            if include_modes:
                for mp in sorted(gp.glob("generated_*.npz")):
                    mode = mp.stem.removeprefix("generated_")
                    if mode.endswith(".partial") or not (gp / f"{mp.stem}.receipt.json").exists():
                        continue
                    add({**base, "unit": f"{arm}{tag}#{mode}", "mode": mode, "gen": str(mp)})
    if include_modes:
        for root in modes_roots:
            mtag = None
            if "=" in str(root):          # r3: explicit TAG=DIR
                mtag, root = str(root).split("=", 1)
                if not re.fullmatch(r"[A-Za-z0-9_]+", mtag):
                    raise ValueError(f"bad --modes-root tag {mtag!r} (want [A-Za-z0-9_]+=DIR)")
                mtag = "~" + mtag
            root = Path(root)
            if not root.exists():
                continue
            for rp in sorted(root.rglob("*.receipt.json")):
                try:
                    q = json.loads(rp.read_text())
                except (json.JSONDecodeError, OSError):
                    continue
                npz = rp.with_name(rp.name.removesuffix(".receipt.json") + ".npz")
                if not isinstance(q, dict) or not q.get("fit") or not q.get("mode") or not npz.exists():
                    continue
                try:
                    arm, panel, k, s = _fit_parts(q["fit"])
                except (ValueError, IndexError):
                    continue
                tag = mtag
                if tag is None:           # r3: inherit the tag of the fits root holding the fit
                    fr = Path(q["fit"]).resolve()
                    tag = next((t for r_, t in root_tags if r_ == fr or r_ in fr.parents), "")
                add({"arm": arm, "panel": panel, "fold": k, "seed": s, "dir": str(q["fit"]), "root": str(root),
                     "unit": f"{arm}{tag}#{q['mode']}", "mode": q["mode"], "gen": str(npz)})
    return found, incomplete, shadowed


class RealCache:
    def __init__(self, caches_root, dam_draws=DAM_NULL_DRAWS):
        self.root = Path(caches_root); self.store = {}; self.dam_draws = dam_draws

    def get(self, panel, fold):
        key = (panel, fold)
        if key not in self.store:
            d = self.root / panel / f"fold{fold}"
            if not (d / "space.npz").exists():
                self.store[key] = None
            else:
                real = load_real(d, panel, fold)
                self.store[key] = (real, gene_subsets(real, self.root), pooled_gene_set(real, self.root),
                                   dam_context(real, self.dam_draws))
        return self.store[key]

    def drop_except(self, keep):
        for k in list(self.store):
            if k not in keep:
                del self.store[k]


def file_key(path):
    st = Path(path).stat()
    return {"size": st.st_size, "mtime_ns": st.st_mtime_ns}


def unit_path(out_root, unit, panel):
    return Path(out_root) / "fits" / unit / panel


def _fresh(old, real, subsets, settings, **extra):
    if (old.get("evaluator_version") != EVALUATOR_VERSION or old.get("cache_manifest_sha256") != real.manifest_sha256
            or old.get("subset_names") != sorted(subsets) or old.get("score_settings") != settings.key()):
        return False
    return all(old.get(k) == v for k, v in extra.items())


def evaluate_fit(rec, rc: RealCache, out_root, log, opts):
    out_json = unit_path(out_root, rec["unit"], rec["panel"]) / f"fold{rec['fold']}" / f"seed{rec['seed']}.json"
    gpath = Path(rec["gen"])
    fk = file_key(gpath)
    loaded = rc.get(rec["panel"], rec["fold"])
    if loaded is None:
        return None, f"cache missing for {rec['panel']}/fold{rec['fold']}"
    real, subsets, pooled, dam = loaded
    settings = ScoreSettings(opts.pc_repeats, opts.cell_repeats, dam)
    if out_json.exists():
        old = json.loads(out_json.read_text())
        if _fresh(old, real, subsets, settings, generated_file=fk) and out_json.with_suffix(".npz").exists():
            return old, "cached"
    gen = align_generated(load_generated(gpath), real)
    res, arrays = score_generated(gen, real, subsets, pooled, settings=settings)
    res.update(kind="fit", unit=rec["unit"], arm=rec["arm"], mode=rec["mode"], panel=rec["panel"], fold=rec["fold"],
               repeat=repeat_of(rec["fold"]), seed=rec["seed"], fit_dir=rec["dir"], root=rec["root"],
               generated_path=str(gpath), generated_file=fk, generated_sha256=sha256(gpath),
               cache_manifest_sha256=real.manifest_sha256, subset_names=sorted(subsets))
    for name in ["config.json", "fit_receipt.json"]:
        p = Path(rec["dir"]) / name
        try:
            res[name.removesuffix(".json")] = json.loads(p.read_text()) if p.exists() else None
        except json.JSONDecodeError:
            res[name.removesuffix(".json")] = "unparseable"
    if rec["mode"] is not None:
        rp = gpath.with_name(gpath.stem + ".receipt.json")
        try:
            res["mode_receipt"] = json.loads(rp.read_text()) if rp.exists() else None
        except json.JSONDecodeError:
            res["mode_receipt"] = "unparseable"
    res["learning_curve"] = learning_curve_summary(Path(rec["dir"]) / "learning_curve.jsonl")
    if rec["mode"] is not None:       # r3: stage dependence of the generated conditions (stage kind)
        res["r3_condition_stage"] = condition_stage_info(gpath)
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


def ensemble_label(unit, n, mmatched):
    return f"{unit} [seed-ensemble n={n}{' M=32' if mmatched else ''}]"


def evaluate_ensemble(unit, panel, fold, recs, rc: RealCache, out_root, log, opts, gens_cache=None):
    """Full and M-matched ensembles of the given (sorted) seed records."""
    recs = sorted(recs, key=lambda r: r["seed"])
    n = len(recs)
    loaded = rc.get(panel, fold)
    if loaded is None:
        return []
    real, subsets, pooled, dam = loaded
    settings = ScoreSettings(opts.pc_repeats, opts.cell_repeats, dam)
    keys = [{"seed": r["seed"], **file_key(r["gen"])} for r in recs]
    seedkey = "+".join(str(r["seed"]) for r in recs)
    outs, gens = [], None
    for mm in (False, True):
        out_json = Path(out_root) / "ensembles" / unit / panel / f"fold{fold}_n{n}{'_M32' if mm else ''}.json"
        if out_json.exists():
            old = json.loads(out_json.read_text())
            if _fresh(old, real, subsets, settings, inputs=keys) and out_json.with_suffix(".npz").exists():
                outs.append(old)
                continue
        if gens is None:
            gens = [align_generated(load_generated(r["gen"]), real) for r in recs]
        res, arrays = score_generated((mmatched_generated if mm else concat_generated)(gens), real, subsets, pooled, settings=settings)
        lcs = [learning_curve_summary(Path(r["dir"]) / "learning_curve.jsonl") for r in recs]
        lc = {"available": any(x.get("available") for x in lcs)}
        for k in LC_HEADLINE.values():
            lc[k] = MX.nanmean([x.get(k) for x in lcs])
        res.update(kind="seed_ensemble", unit=unit, arm=recs[0]["arm"], mode=recs[0]["mode"], panel=panel, fold=fold,
                   repeat=repeat_of(fold), seeds=[r["seed"] for r in recs], seed_key=seedkey, n_seeds=n, m_matched=mm,
                   label=ensemble_label(unit, n, mm), inputs=keys, cache_manifest_sha256=real.manifest_sha256,
                   subset_names=sorted(subsets), learning_curve=lc,
                   ensemble_rule=("generated donor m from seed index m mod n (M = 32)" if mm else "all n*32 generated donors pooled"))
        if not mm:
            res["seed_stability"] = seed_stability(gens, real)
        save_npz(out_json.with_suffix(".npz"), **arrays)
        write_json(out_json, res)
        log(f"ensemble {res['label']}/{panel}/fold{fold} seeds={res['seeds']}")
        outs.append(res)
    return outs


def evaluate_baselines(panel, fold, rc: RealCache, out_root, log, opts):
    import baselines as BL
    if Path(BL.__file__).resolve().parent != HERE:
        raise ImportError(f"r3 evaluate imported the wrong baselines module: {BL.__file__}")
    loaded = rc.get(panel, fold)
    if loaded is None:
        return {}
    real, subsets, pooled, dam = loaded
    settings = ScoreSettings(opts.pc_repeats, opts.cell_repeats, dam)
    outdir = Path(out_root) / "baselines" / panel / f"fold{fold}"
    results = {}
    for name in BL.BASELINES:
        out_json = outdir / f"{name}.json"
        if out_json.exists():
            old = json.loads(out_json.read_text())
            if _fresh(old, real, subsets, settings, baseline_version=BL.BASELINE_VERSION) and out_json.with_suffix(".npz").exists():
                results[name] = old
                continue
        t0 = time.monotonic()
        gen, info = BL.generate(name, real)
        res, arrays = score_generated(align_generated(gen, real), real, subsets, pooled, settings=settings)
        res.update(kind="baseline", unit=f"baseline:{name}", arm=f"baseline:{name}", baseline=name, panel=panel, fold=fold,
                   repeat=repeat_of(fold), seed=None, baseline_version=BL.BASELINE_VERSION, baseline_info=info,
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


def write_real_reference(real, dam, out_root):
    """Model-free per-fold reference: DAM statistics of the real contrasts."""
    held = real.is_heldout; train = ~held
    out = {"evaluator_version": EVALUATOR_VERSION, "panel": real.panel, "fold": real.fold, "repeat": repeat_of(real.fold),
           "cache_manifest_sha256": real.manifest_sha256, "n_genes": int(len(real.genes)),
           "dam_genes_present": dam["present"], "dam_genes_missing": dam["missing"], "dam_n_present": len(dam["present"]),
           "n_by_stage": {"heldout": MX.stage_contrasts(real.mean_log[held], real.stage[held])["n_by_stage"],
                          "train": MX.stage_contrasts(real.mean_log[train], real.stage[train])["n_by_stage"]},
           "dam_real": {}}
    for space, P in [("mean_log", real.mean_log), ("pseudobulk", MX.log_cp10k(real.pb))]:
        out["dam_real"][space] = {
            nm: MX.real_dam_reference(MX.stage_contrasts(P[m], real.stage[m])["high_notad"], dam["idx"], dam["sgn"], dam["null"])
            for nm, m in [("heldout", held), ("train", train), ("all_fold_donors", np.ones_like(held))]}
        out["dam_real_perm"] = out.get("dam_real_perm", {})
        out["dam_real_perm"][space] = {   # repair S1: stage-permutation null on real donors
            nm: MX.dam_perm_null(P[m], real.stage[m], dam["idx"], dam["sgn"], DAM_PERMUTATIONS,
                                 MX.stable_seed("dam-perm-real", real.panel, real.fold, nm))
            for nm, m in [("heldout", held), ("train", train), ("all_fold_donors", np.ones_like(held))]}
    write_json(Path(out_root) / "real" / real.panel / f"fold{real.fold}.json", out)


# ==========================================================================
# Pooled-across-folds contrast (round 1, LEAKY) and cross-fitted donor score
# ==========================================================================
POOLED_WARNING = ("NOT a held-out estimate: every held-out donor of fold k is a training donor of the other two folds' "
                  "models, so the pooled model contrast (an average of train-stage contrasts over 2/3 of the 37 donors) "
                  "shares data with the pooled real contrast. The stage_mean baseline reaches Spearman ~0.93 here by "
                  "construction while independent halves agree at ~0.1 (split-half) and per-fold held-out Spearman is ~0.1. "
                  "Read it as a fit-to-all-development-donors quantity, never as generalisation; use xfit_* "
                  "(cross-fitted donor score) for held-out stage claims.")


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


def xfit_from_npz(npz_paths, n_perm, seed):
    """Pooled cross-fitted donor score of one repeat from the per-unit npz
    files of its folds (definition in the module docstring)."""
    stage_folds, target_folds, donors = [], [], []
    for p in npz_paths:
        with np.load(p, allow_pickle=False) as f:
            if "xf_stage_dev" not in f.files:
                return None
            held = f["is_heldout"].astype(bool)
            st = f["stage"][held].astype(int)
            obs = f["xf_obs_dev"].astype(np.float64)
            sdev = f["xf_stage_dev"].astype(np.float64)
            tdev = f["xf_target_dev"].astype(np.float64)
            donors += f["donors"][held].astype(str).tolist()
        if not np.isfinite(sdev[np.unique(st)]).all():
            raise ValueError(f"{p}: a held-out stage has no train target (stage deviation undefined)")
        stage_folds.append({"obs": obs, "cand": sdev, "assign": st, "group": st})
        target_folds.append({"obs": obs, "cand": tdev, "assign": np.arange(len(st)), "group": st})
    if len(set(donors)) != len(donors):
        raise ValueError("a donor is held out in two folds of one repeat")
    check_development_donors(donors, "xfit")
    out = {"n_donors": len(donors), "donors": donors,
           "stage": MX.xfit_score(stage_folds, n_perm, seed),
           "target": MX.xfit_score(target_folds, n_perm, seed + 1),
           "independent_of_training": True,
           "definition": "score_d = Pearson_g(y_d - gbar_k, delta_k(s_d)) for each donor d held out in fold k; mean over the "
                         "repeat's donors; null = stage labels permuted within folds among held-out donors"}
    return out


# ==========================================================================
# Aggregation
# ==========================================================================
def _mean_se(vals):
    v = [x for x in vals if x is not None and np.isfinite(x)]
    if not v:
        return None, None, 0
    return float(np.mean(v)), (float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else None), len(v)


def repeat_balanced(fold_values):
    """{fold: value} -> (mean over repeats of the mean over folds within the
    repeat, se, se_kind, se_folds, n_folds, n_repeats)."""
    fv = {f: v for f, v in fold_values.items() if v is not None and np.isfinite(v)}
    if not fv:
        return None, None, None, None, 0, 0
    reps = {}
    for f, v in fv.items():
        reps.setdefault(repeat_of(f), []).append(v)
    rmeans = [float(np.mean(v)) for _, v in sorted(reps.items())]
    m = float(np.mean(rmeans))
    _, se_f, nf = _mean_se(list(fv.values()))
    if len(rmeans) >= 2:
        se, kind = float(np.std(rmeans, ddof=1) / np.sqrt(len(rmeans))), "repeats"
    else:
        se, kind = se_f, "folds"
    return m, se, kind, se_f, nf, len(rmeans)


# MEANINGFUL keys (baselines.py) -> headline metric names they govern
MEANINGFUL_GROUPS = {
    "pseudobulk": ["pb_rmse", "pb_dev_pearson", "pb_dev_spearman", "pb_rmse_recal", "pb_dev_pearson_recal"],
    "mean_log/nonzero": ["meanlog_rmse_gene", "meanlog_bias", "nonzero_rmse_gene", "nonzero_bias",
                         "meanlog_rmse_gene_recal", "nonzero_rmse_gene_recal"],
    "second_moment": ["logvar_rmse_gene", "logvar_bias"],
    "contrast": [n for n in HEADLINE if n.startswith(("hn_", "ord_"))] + list(DAM_HEADLINE) + list(XFIT_HEADLINE),
    "distribution_pb": [n for n in HEADLINE if n.startswith("pbdist_")],
    "pc_space/cell_energy": list(PC_HEADLINE) + list(PC_REP_HEADLINE),
}
# Groups masked unless the baseline's MEANINGFUL text starts with 'meaningful' (round-2 rule, F8);
# other groups are masked only when the text starts with 'NOT' (round-1 rule).
STRICT_GROUPS = {"mean_log/nonzero", "second_moment"}
REFERENCE_BASELINES = ["grand_mean", "stage_mean", "stage_mean_log", "shrunk_stage_mean", "ridge", "resample"]


def mask_not_meaningful(flat, meaningful):
    """Set to None every metric whose baselines.MEANINGFUL entry says 'NOT meaningful', and every
    mean_log / nonzero / second-moment metric whose entry does not start with 'meaningful'."""
    masked = []
    for key, text in (meaningful or {}).items():
        text = str(text)
        if not (text.startswith("NOT") or (key in STRICT_GROUPS and not text.startswith("meaningful"))):
            continue
        for name in list(flat):
            if name.split("__")[-1] in MEANINGFUL_GROUPS.get(key, []):
                flat[name] = None
                masked.append(name)
    return flat, sorted(set(masked))


def _base_label(label):
    """Reference label for vs_base: same suffix with unit 'base' (mode stripped)."""
    unit, sep, suffix = label.partition(" [")
    return "base" + (sep + suffix if sep else "")


def _parent_label(label):
    unit, sep, suffix = label.partition(" [")
    if "#" not in unit:
        return None
    return unit.split("#", 1)[0] + (sep + suffix if sep else "")


def aggregate(out_root, fit_results, ens_results, base_results, incomplete, errors, log, shadowed=(), opts=None):
    out_root = Path(out_root)
    n_perm = getattr(opts, "xfit_permutations", XFIT_PERMUTATIONS)
    units = {}   # (row_label, panel) -> list of (fold, seedkey, flat, result)
    masked_by_label = {}
    for r in fit_results:
        units.setdefault((r["unit"], r["panel"]), []).append((r["fold"], r["seed"], headline(r), r))
    for r in ens_results:
        units.setdefault((r["label"], r["panel"]), []).append((r["fold"], r["seed_key"], headline(r), r))
    for (panel, fold), bl in base_results.items():
        for name, r in bl.items():
            if name == "noise_ceiling":
                continue
            flat, masked = mask_not_meaningful(headline(r), r.get("meaningful_metrics"))
            masked_by_label.setdefault((f"baseline:{name}", panel), set()).update(masked)   # r2 fix: per panel, union over folds
            units.setdefault((f"baseline:{name}", panel), []).append((fold, None, flat, r))
    # ---- repair FID-3 / S4: stage-blind units lose their stage metrics ----
    stage_kind = {}
    ukinds = unit_stage_kinds(fit_results)          # r3: name rule per '+' component, receipt/config, conditions
    for (label, panel), us in units.items():
        unit = label.partition(" [")[0]
        if label == "baseline:resample_blind":
            k = "stage-blind"
        elif unit in ukinds:
            k = ukinds[unit]
        else:
            k = stage_blind_mode_r3(label_mode(label))
        stage_kind[label] = k
        if k == "stage-blind":
            for u in us:
                for n in list(u[2]):
                    if is_stage_metric(n):
                        u[2][n] = None
    # ---- repair S2: is the frozen DAM list adequate on this panel? ----
    dam_ok = {}
    for (label, panel), us in units.items():
        for u in us:
            npres, ndown = u[2].get("dam_n_present"), u[2].get("dam_n_down_present")
            if npres is None:
                continue
            ok = npres >= DAM_MIN_PRESENT and (ndown or 0) >= DAM_MIN_DOWN
            dam_ok[panel] = dam_ok.get(panel, True) and ok
            dam_ok.setdefault(("range", panel), [npres, npres, ndown or 0])
            rg = dam_ok[("range", panel)]
            rg[0] = min(rg[0], npres); rg[1] = max(rg[1], npres); rg[2] = max(rg[2], ndown or 0)
    # ---- repair FID-1: generation device per unit (common random numbers only within one device type) ----
    dev = {}
    for (label, panel), us in units.items():
        for u in us:
            if isinstance(u[3], dict) and u[3].get("kind") != "seed_ensemble":
                dev[(label, panel, u[0], u[1])] = gen_device(u[3])
    for (label, panel), us in units.items():       # ensembles inherit their member fits' device
        for u in us:
            if isinstance(u[3], dict) and u[3].get("kind") == "seed_ensemble":
                ds = set(dev.get((u[3]["unit"], panel, u[0], sd), "unknown") for sd in u[3].get("seeds", []))
                dev[(label, panel, u[0], u[1])] = ds.pop() if len(ds) == 1 else "unknown"
    # ---- cross-fitted donor score per (label, panel, repeat, seedkey) ----
    xfit = {}
    xfit_flat = {}      # (label, panel) -> {(repeat, seedkey): flat}
    for (label, panel), us in sorted(units.items()):
        if stage_kind.get(label) == "stage-blind":
            continue        # repair S4: no stage score for a stage-blind predictor
        groups = {}
        for f, s, _, r in us:
            groups.setdefault((repeat_of(f), s), {})[f] = r
        for (rep, s), byfold in sorted(groups.items(), key=lambda x: (x[0][0], str(x[0][1]))):
            if sorted(byfold) != folds_of_repeat(rep):
                continue
            paths = [_unit_npz(out_root, byfold[f]) for f in sorted(byfold)]
            if not all(p.exists() for p in paths):
                continue
            try:
                xf = xfit_from_npz(paths, n_perm, MX.stable_seed("xfit", panel, rep))
            except ValueError as ex:
                errors.append({"xfit": f"{label}|{panel}|repeat={rep}|seed={s}", "error": str(ex)})
                continue
            if xf is None:
                continue
            xfit[f"{label}|{panel}|repeat={rep}|seed={s}"] = xf
            xfit_flat.setdefault((label, panel), {})[(rep, s)] = {
                "xfit_score": xf["stage"]["score"].get("value"), "xfit_score_z": xf["stage"]["score"].get("z"),
                "xfit_pooled": xf["stage"]["pooled_pairs"].get("value"), "xfit_target_score": xf["target"]["score"].get("value")}
    rows = []
    agg = {"evaluator_version": EVALUATOR_VERSION, "rows": [], "paired": {}, "pooled_across_folds": {}, "xfit": xfit,
           "noise_ceiling": {}, "incomplete_fits": incomplete, "errors": errors, "shadowed": list(shadowed),
           "settings": {"pc_repeats": getattr(opts, "pc_repeats", None), "cell_repeats": getattr(opts, "cell_repeats", None),
                        "xfit_permutations": n_perm, "dam_null_draws": getattr(opts, "dam_draws", None)},
           "descriptive_only": sorted(DESCRIPTIVE_ONLY),
           "dam_paired_panels": {p: v for p, v in dam_ok.items() if isinstance(p, str)},
           "dam_present_range": {k[1]: {"min_present": v[0], "max_present": v[1], "max_down": v[2]}
                                 for k, v in dam_ok.items() if isinstance(k, tuple)},
           "stage_blind_units": sorted(l for l, k in stage_kind.items() if k == "stage-blind"),
           "stage_mixture_units": sorted(l for l, k in stage_kind.items() if k == "stage-mixture"),
           "stage_shrunk_units": sorted(l for l, k in stage_kind.items() if k == "stage-shrunk"),
           "notes": ["Primary metrics are on held-out inner-fold donors. Paired results are descriptive (mean difference, "
                     "sign counts over folds), not significance tests.",
                     "Row values are REPEAT-BALANCED means: seeds averaged within fold, folds averaged within a repeat "
                     "(the 3 folds of a repeat partition the 37 donors), then repeats averaged. __se is the SE over repeat "
                     "means when >= 2 repeats are present (__se_kind='repeats'), else the SE over fold means ('folds', as "
                     "round 1); __se_folds is always the SE over all fold means. Repeats reuse the same 37 donors and folds "
                     "share training donors, so these SEs describe split-to-split variability and understate uncertainty.",
                     "__unit_mean/__unit_se treat (fold, seed) units as independent (reference only).",
                     "Paired comparisons count signs PER FOLD (seed-paired goodness differences averaged within fold); "
                     "mean_difference is repeat-balanced; ensembles are paired only over identical seed sets.",
                     "DESCRIPTIVE ONLY (not in paired tables): " + ", ".join(sorted(DESCRIPTIVE_ONLY)) +
                     " - per-fold contrasts against 12-13 held-out donors with 1-2 Not AD donors.",
                     "Baseline masking: mean_log / nonzero / second-moment metrics (incl. recalibrated) are None unless "
                     "baselines.MEANINGFUL says 'meaningful' (resample only); PC/cell metrics of profile baselines are None.",
                     "*_recal: per-gene additive train-only correction (mean over train donors of observed - predicted) added "
                     "to held-out predictions; identical for models and baselines.",
                     "pc_*_rep / cell_energy*_rep: averaged over repeated real-cell subsample draws; pc_energy etc. without "
                     "_rep are the round-1 single-subsample values.",
                     "xfit_*: pooled cross-fitted donor score (not leaky); one value per (repeat, seed); see evaluate.py docstring.",
                     "shared_<other>_renorm subsets are the only like-for-like cross-panel comparison; per-panel tables are "
                     "never comparable across panels.",
                     "noise_ceiling is a PEER-VARIABILITY REFERENCE, not a ceiling.",
                     "pooled_across_folds: " + POOLED_WARNING,
                     "REPAIR S1/S2: dam_signed_perm_z uses a stage-label permutation null (valid as a stage test); "
                     "dam_signed_z (matched random genes) ignores gene-gene correlation and is descriptive only. dam_* enter "
                     f"paired tables only on panels with >= {DAM_MIN_PRESENT} listed genes incl. >= {DAM_MIN_DOWN} down gene "
                     "(dam_paired_panels); on hvg2000 (3-5 up genes only) use xfit_* as the stage metric.",
                     "REPAIR FID-3/S4: stage_blind_units (condmix@1, condunk@1, condshrink@1, anchorgain@0) predict the same "
                     "stage mixture for every target: a generator-made grand-mean predictor. Their stage metrics are None and "
                     "their pb_rmse is a calibration result to be read against the grand-mean baseline, not an F3 fix. "
                     "stage_mixture_units: distribution metrics describe a stage mixture.",
                     "REPAIR FID-1: paired blocks carry crn (True = all paired files generated on the same device type, so "
                     "they share random numbers; False = CPU vs CUDA pair: extra Monte Carlo noise, no bias).",
                     "REPAIR S7: sign counts are over folds; folds 0-8 are 3 repeats of the SAME 37 donors, so counts over "
                     "folds are not independent replications. n_better_by_repeat and repeats_majority_better give the "
                     "per-repeat view.",
                     "REPAIR S3 (selection history): several round-2 choices (condmix@1 on every arm, condmix over condshrink, "
                     "genemean as combination partner, pbulk, donorpc2 share bound, the existence of covdrop/covbin/"
                     "stageanchor/stagedrop) were informed by round-1 HELD-OUT results on these same 37 donors; folds 3-8 "
                     "re-split them. Round-2 held-out results are development evidence conditional on that selection; "
                     "confirmation needs donors used in neither round."]}
    for (label, panel), us in sorted(units.items()):
        names = sorted(set().union(*[set(u[2]) for u in us]))
        folds = sorted(set(u[0] for u in us))
        row = {"arm": label, "panel": panel, "n_units": len(us), "folds": folds,
               "repeats": sorted(set(repeat_of(f) for f in folds)),
               "complete_repeats": sorted(r for r in set(repeat_of(f) for f in folds)
                                          if set(folds_of_repeat(r)) <= set(folds))}
        if (label, panel) in masked_by_label:
            row["masked_not_meaningful"] = sorted(masked_by_label[(label, panel)])
        if stage_kind.get(label):
            row["stage_mode"] = stage_kind[label]
        row["gen_devices"] = sorted(set(dev.get((label, panel, u[0], u[1]), "unknown") for u in us))
        for n in names:
            um, use, k = _mean_se([u[2].get(n) for u in us])
            fold_means = {f: MX.nanmean([u[2].get(n) for u in us if u[0] == f]) for f in folds}
            m, se, kind, se_f, nf, nr = repeat_balanced(fold_means)
            row[n] = m; row[n + "__se"] = se; row[n + "__se_kind"] = kind; row[n + "__se_folds"] = se_f
            row[n + "__n_folds"] = nf; row[n + "__n_repeats"] = nr
            row[n + "__unit_mean"] = um; row[n + "__unit_se"] = use; row[n + "__n"] = k
        xf = xfit_flat.get((label, panel), {})
        for n in XFIT_HEADLINE:
            reps = {}
            for (rep, s), fl in xf.items():
                if fl.get(n) is not None:
                    reps.setdefault(rep, []).append(fl[n])
            rm = [float(np.mean(v)) for _, v in sorted(reps.items())]
            mm, se, k = _mean_se(rm)
            row[n] = mm; row[n + "__se"] = se; row[n + "__se_kind"] = "repeats" if k >= 2 else None
            row[n + "__n_repeats"] = k
        if label.endswith("]") and "seed-ensemble" in label and " M=32" not in label:
            st = [u[3].get("seed_stability", {}).get("heldout", {}).get("high_notad", {}).get("mean_spearman") for u in us]
            row["seed_stability_hn_spearman"] = MX.nanmean(st)
            st = [u[3].get("seed_stability", {}).get("all_targets", {}).get("high_notad", {}).get("mean_spearman") for u in us]
            row["seed_stability_hn_spearman_all_targets"] = MX.nanmean(st)
        rows.append(row)
    agg["rows"] = rows

    # paired comparisons (fold is the unit; xfit: repeat is the unit)
    def unit_map(label, panel):
        return {(u[0], u[1]): u[2] for u in units.get((label, panel), [])}

    for (label, panel), us in sorted(units.items()):
        if label.startswith("baseline:"):
            continue
        names = sorted(set().union(*[set(u[2]) for u in us]))
        mine = unit_map(label, panel)
        comps = {}

        def compare(keys, ref_of, ref_xfit=None):
            block = {}
            folds = sorted(set(k[0] for k in keys))
            for n in names:
                if direction(n) is None or n.split("__")[-1] in DESCRIPTIVE_ONLY:
                    continue
                if n.split("__")[-1].startswith("dam_") and not dam_ok.get(panel, False):
                    continue        # repair S2: listed genes inadequate on this panel
                a_f, b_f, raw_f, fl = [], [], [], []
                for f in folds:
                    pairs = [(goodness(n, mine[k].get(n)), goodness(n, ref_of(k).get(n)), mine[k].get(n), ref_of(k).get(n))
                             for k in keys if k[0] == f]
                    pairs = [p for p in pairs if p[0] is not None and p[1] is not None]
                    if pairs:
                        a_f.append(float(np.mean([p[0] for p in pairs]))); b_f.append(float(np.mean([p[1] for p in pairs])))
                        raw_f.append(float(np.mean([p[2] - p[3] for p in pairs]))); fl.append(f)
                s = MX.paired_summary(a_f, b_f, True)
                if s.get("n"):
                    m, se, kind, se_f, _, nr = repeat_balanced({f: a - b for f, a, b in zip(fl, a_f, b_f)})
                    s["mean_difference"], s["se_difference"], s["se_kind"], s["n_repeats"] = m, se, kind, nr
                    s["raw_mean_difference"] = repeat_balanced(dict(zip(fl, raw_f)))[0]
                    byr = {}        # repair S7: sign counts per repeat (repeats share donors)
                    for f, a, b in zip(fl, a_f, b_f):
                        c = byr.setdefault(repeat_of(f), [0, 0]); c[0] += int(a > b); c[1] += 1
                    s["n_better_by_repeat"] = {str(r): c for r, c in sorted(byr.items())}
                    s["repeats_majority_better"] = int(sum(c[0] * 2 > c[1] for c in byr.values()))
                    s["n_repeats_compared"] = len(byr)
                else:
                    s["raw_mean_difference"] = None
                block[n] = s
            if ref_xfit is not None:
                xm = xfit_flat.get((label, panel), {})
                for n in XFIT_HEADLINE:
                    reps = sorted(set(r for r, _ in xm) & set(r for r, _ in ref_xfit))
                    a_r, b_r = [], []
                    for rep in reps:
                        pr = [(xm[(rep, s)].get(n), ref_xfit(rep, s).get(n)) for (r2, s) in xm if r2 == rep and ref_xfit(rep, s)]
                        pr = [p for p in pr if p[0] is not None and p[1] is not None]
                        if pr:
                            a_r.append(float(np.mean([p[0] for p in pr]))); b_r.append(float(np.mean([p[1] for p in pr])))
                    s = MX.paired_summary(a_r, b_r, True)
                    s["paired_on"] = "repeat"
                    block[n] = s
            return block

        class _XRef:
            def __init__(self, d, any_seed=False):
                self.d, self.any_seed = d, any_seed

            def __iter__(self):
                return iter(self.d)

            def __call__(self, rep, s):
                if self.any_seed:
                    v = [x for (r, _), x in self.d.items() if r == rep]
                    return v[0] if v else {}
                return self.d.get((rep, s), {})

        refs = [("vs_base", _base_label(label))]
        if _parent_label(label):
            refs.append(("vs_parent", _parent_label(label)))
        if control_label(label):          # r3: single-factor control (same label without t./g./d. flags)
            refs.append(("vs_control", control_label(label)))
        for cname, ref_label in refs:
            if ref_label == label:
                continue
            ref = unit_map(ref_label, panel)
            keys = sorted((k for k in mine if k in ref), key=lambda k: (k[0], str(k[1])))
            if not keys:
                continue
            dpairs = sorted(set((dev.get((label, panel, k[0], k[1]), "unknown").rstrip("?"),
                                 dev.get((ref_label, panel, k[0], k[1]), "unknown").rstrip("?")) for k in keys))
            crn = (None if any("unknown" in d for d in dpairs) else all(a == b for a, b in dpairs))
            comps[cname] = {"reference": ref_label, "paired_on": "fold (seed-paired differences averaged within fold)",
                            "crn": crn, "device_pairs": [list(d) for d in dpairs],
                            "units": [list(k) for k in keys],
                            "metrics": compare(keys, lambda k, ref=ref: ref[k], _XRef(xfit_flat.get((ref_label, panel), {})))}
        for bname in REFERENCE_BASELINES:
            sm = {k[0]: v for k, v in unit_map(f"baseline:{bname}", panel).items()}
            if not sm:
                continue
            keys = sorted((k for k in mine if k[0] in sm), key=lambda k: (k[0], str(k[1])))
            comps[f"vs_{bname}_baseline"] = {"reference": f"baseline:{bname}", "crn": None,
                                             "paired_on": "fold (arm seeds averaged within fold; baseline fitted once per fold)",
                                             "units": [list(k) for k in keys],
                                             "metrics": compare(keys, lambda k, sm=sm: sm[k[0]],
                                                                _XRef(xfit_flat.get((f"baseline:{bname}", panel), {}), any_seed=True))}
        agg["paired"][f"{label}|{panel}"] = comps
    agg["paired_note"] = ("Paired 'mean_difference' (repeat-balanced) and sign counts (over FOLDS, which share donors across "
                          "repeats; n_better_by_repeat per repeat; over repeats for xfit_*) on a "
                          "goodness scale (higher is better: negated for lower-is-better metrics, -|log ratio| for spread ratios, "
                          "-|slope-1| for contrast regression slopes, -|cov-0.8| for coverage, -|bias| for biases); "
                          "raw_mean_difference is arm minus reference on the raw scale. Baseline metrics masked as not "
                          "meaningful have no paired value.")
    # pooled across folds (round-1 leaky contrast), per repeat
    for (label, panel), us in sorted(units.items()):
        groups = {}
        for f, s, _, r in us:
            groups.setdefault((repeat_of(f), s), {})[f] = r
        for (rep, s), byfold in groups.items():
            if sorted(byfold) != folds_of_repeat(rep):
                continue
            paths = [_unit_npz(out_root, byfold[f]) for f in sorted(byfold)]
            if all(p.exists() for p in paths):
                pc = pooled_contrast(paths)
                if pc is not None:
                    key = f"{label}|{panel}|seed={s}" if rep == 0 else f"{label}|{panel}|repeat={rep}|seed={s}"
                    agg["pooled_across_folds"][key] = pc
    for (panel, fold), bl in base_results.items():
        if "noise_ceiling" in bl:
            agg["noise_ceiling"][f"{panel}|fold{fold}"] = bl["noise_ceiling"]
    write_json(out_root / "aggregate.json", agg)
    # CSV
    cols = ["arm", "panel", "n_units"]
    allkeys = sorted(set().union(*[set(r) for r in rows]) - set(cols) - {"folds", "repeats", "complete_repeats"}) if rows else []
    with open(out_root / "aggregate.csv.tmp", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols + ["folds", "repeats"] + allkeys)
        for r in rows:
            w.writerow([r.get(c) for c in cols] + [" ".join(map(str, r["folds"])), " ".join(map(str, r["repeats"]))]
                       + [r.get(k) for k in allkeys])
    (out_root / "aggregate.csv.tmp").replace(out_root / "aggregate.csv")
    write_summary(out_root, agg)
    r3_outputs(out_root, units, stage_kind, fit_results, opts)     # r3: curves.json, targets.json, leaderboard.txt
    return agg


def _unit_npz(out_root, r):
    if r["kind"] == "fit":
        return unit_path(out_root, r["unit"], r["panel"]) / f"fold{r['fold']}" / f"seed{r['seed']}.npz"
    if r["kind"] == "seed_ensemble":
        return Path(out_root) / "ensembles" / r["unit"] / r["panel"] / f"fold{r['fold']}_n{r['n_seeds']}{'_M32' if r['m_matched'] else ''}.npz"
    return Path(out_root) / "baselines" / r["panel"] / f"fold{r['fold']}" / f"{r['baseline']}.npz"


SUMMARY_COLS = ["pb_rmse", "pb_rmse_recal", "pb_dev_pearson", "meanlog_rmse_gene", "meanlog_rmse_gene_recal",
                "nonzero_rmse_gene", "logvar_rmse_gene", "pbdist_energy", "pc_energy_rep", "pc_coverage80_rep",
                "cell_energy_rep", "pc_energy", "dam_signed_perm_z", "dam_sign_frac", "xfit_score", "xfit_score_z", "lc_critic_gap"]
PAIRED_COLS = ["pb_rmse", "pb_rmse_recal", "meanlog_rmse_gene", "meanlog_rmse_gene_recal", "logvar_rmse_gene",
               "pc_energy_rep", "cell_energy_rep", "dam_signed_perm_z", "xfit_score"]
SHARED_COLS = ["pb_rmse", "pb_rmse_recal", "pb_dev_pearson", "pbdist_energy"]
DESCRIPTIVE_COLS = ["hn_spearman_vs_heldout", "hn_top10_sign_vs_heldout", "hn_slope_vs_heldout", "ord_spearman_vs_heldout",
                    "dam_heldout_spearman", "dam_heldout_spearman_z", "dam_signed_z", "hn_spearman_vs_train", "hn_slope_vs_train",
                    "dam_n_present", "dam_n_down_present"]


def _fmt(v, se=None):
    if v is None:
        return "-"
    s = f"{v:.3f}"
    return s + (f" ± {se:.3f}" if se is not None else "")


def _table(rows, cols, label_cols):
    lines = ["| " + " | ".join(label_cols + cols) + " |", "|" + "---|" * (len(cols) + len(label_cols))]
    for r in rows:
        tag = {"stage-blind": " [STAGE-BLIND]", "stage-mixture": " [stage-mixture]", "stage-shrunk": " [stage-shrunk]"}.get(r.get("stage_mode"), "")
        lab = [str(r["arm"]) + tag, f"{r['n_units']} ({len(r['folds'])}f/{len(r['repeats'])}r)"] + ([r["panel"]] if "panel" in label_cols else [])
        lines.append("| " + " | ".join(lab + [_fmt(r.get(c), r.get(c + '__se')) for c in cols]) + " |")
    return lines


def write_summary(out_root, agg):
    """summary.txt: markdown-formatted tables (one section per panel, then the
    like-for-like shared-gene section; raw cross-panel columns never side by side)."""
    lines = ["# improve_20261109 round-3 evaluation summary (auto-generated; round-2 tables + stage-shrunk tag)", "",
             f"Evaluator {EVALUATOR_VERSION}. Held-out inner-fold donors. Values are repeat-balanced means ± SE "
             "(SE over repeat means when >= 2 repeats, else over fold means; see aggregate.json notes). "
             "Development study only; no significance claims.", "",
             "`-` for a baseline column means the metric is masked as not meaningful for that baseline. "
             "`_recal` = train-only per-gene additive recalibration. `_rep` = averaged over repeated real-cell subsamples "
             "(pc_energy without _rep = round-1 single subsample). xfit = pooled cross-fitted donor score (per repeat). "
             "Per-fold held-out contrast metrics are descriptive only and appear only in the descriptive table.", "",
             "REPAIR NOTES (2026-11-10). (1) [STAGE-BLIND] units (condmix@1, condunk@1, condshrink@1, anchorgain@0) give "
             "every target the same stage mixture: their pb_rmse is a calibration result (a generator-made grand-mean "
             "predictor; compare with baseline:grand_mean and posthoc direct:train_mean), NOT a fix of stage inflation (F3); "
             "their stage metrics are blank. Judge stage fixes with condmix@0.5 and the trained arms. (2) dam_signed_perm_z "
             "uses a stage-label permutation null; the matched-gene dam_signed_z is miscalibrated (~20% p<0.05 without "
             "stage signal) and only descriptive. (3) Paired cells read 'diff (b/n f[; r/R rep]) [CRN]': b of n FOLDS "
             "better, r of R REPEATS with most folds better; folds of different repeats reuse the same 37 donors, so "
             "fold counts are not independent. 'xdev' = the arm and reference were generated on different devices "
             "(no common random numbers). (4) Selection history: round-2 choices were informed by round-1 held-out "
             "results on these same donors (see evaluate.py docstring, S3); every number here is development evidence "
             "conditional on that selection.", ""]
    for panel in PANELS:
        prow = [r for r in agg["rows"] if r["panel"] == panel]
        if not prow:
            continue
        lines += [f"## Panel {panel} (this panel's genes only; not comparable across panels)", ""]
        lines += _table(prow, SUMMARY_COLS, ["arm", "units (folds/repeats)"]) + [""]
        rg = (agg.get("dam_present_range") or {}).get(panel)
        if rg is not None:
            ok = (agg.get("dam_paired_panels") or {}).get(panel)
            lines += [f"DAM on {panel}: {rg['min_present']}-{rg['max_present']} of 27 listed genes present, at most "
                      f"{rg['max_down']} expected-down gene(s); " + ("dam_* paired." if ok else
                      "TOO FEW / ONE-DIRECTION: every dam_* value on this panel is descriptive only (not paired); "
                      "use xfit_* as the stage metric."), ""]
        pl = [(k, v) for k, v in agg["paired"].items() if k.endswith("|" + panel)]
        if pl:
            lines += ["Paired (goodness scale, higher = arm better; mean difference (better/n folds; repeats for xfit)):", "",
                      "| arm | vs | " + " | ".join(PAIRED_COLS) + " |", "|" + "---|" * (len(PAIRED_COLS) + 2)]
            for k, comps in pl:
                for cname, c in comps.items():
                    cells = []
                    for m in PAIRED_COLS:
                        s = c["metrics"].get(m) or {}
                        if not s.get("n"):
                            cells.append("-"); continue
                        if s.get("paired_on") == "repeat":
                            cells.append(f"{s['mean_difference']:+.3f} ({s['n_better']}/{s['n']} rep)"); continue
                        txt = f"{s['mean_difference']:+.3f} ({s['n_better']}/{s['n']} f"
                        if (s.get("n_repeats_compared") or 0) >= 2:
                            txt += f"; {s['repeats_majority_better']}/{s['n_repeats_compared']} rep"
                        cells.append(txt + ")")
                    crn = {True: "", False: " xdev", None: ""}.get(c.get("crn"), "")
                    lines.append(f"| {k.rsplit('|', 1)[0]} | {cname}{crn} | " + " | ".join(cells) + " |")
            lines.append("")
        lines += [f"Descriptive per-fold contrast metrics, panel {panel} (NOT for ranking arms):", ""]
        lines += _table(prow, DESCRIPTIVE_COLS, ["arm", "units (folds/repeats)"]) + [""]
    # like-for-like shared-gene section
    shared = []
    for r in agg["rows"]:
        other = [p for p in PANELS if p != r["panel"]]
        if not other:
            continue
        pre = f"shared_{other[0]}_renorm__"
        if any(r.get(pre + c) is not None for c in SHARED_COLS):
            rr = {"arm": r["arm"], "panel": r["panel"], "n_units": r["n_units"], "folds": r["folds"], "repeats": r["repeats"]}
            for c in SHARED_COLS:
                rr[c] = r.get(pre + c); rr[c + "__se"] = r.get(pre + c + "__se")
            shared.append(rr)
    if shared:
        lines += ["## Like-for-like: genes shared by both panels (same fold), log1p CP10k renormalised within the shared genes", "",
                  "Only folds where both panels have a cache contribute. These are the only numbers comparable across panels.", ""]
        lines += _table(shared, SHARED_COLS, ["arm", "units (folds/repeats)", "panel"]) + [""]
    xf = agg.get("xfit", {})
    if xf:
        lines += ["## Pooled cross-fitted donor score (not leaky; one value per repeat and seed)", "",
                  "| unit | panel | repeat | seed | n donors | score | null mean | z | p | pooled pairs | target score |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for k, v in xf.items():
            a, p, rep, s = k.split("|")
            sc = v["stage"]["score"]
            lines.append(f"| {a} | {p} | {rep.removeprefix('repeat=')} | {s.removeprefix('seed=')} | {v['n_donors']} | {_fmt(sc.get('value'))} | "
                         f"{_fmt(sc.get('null_mean'))} | {_fmt(sc.get('z'))} | {_fmt(sc.get('p_one_sided'))} | "
                         f"{_fmt(v['stage']['pooled_pairs'].get('value'))} | {_fmt(v['target']['score'].get('value'))} |")
        lines.append("")
    pooled = agg.get("pooled_across_folds", {})
    if pooled:
        lines += ["## Pooled-across-folds High-minus-NotAD contrast (round 1; LEAKY, not held-out)", "",
                  "Warning: " + POOLED_WARNING, "",
                  "| key | n donors | Spearman | top-decile sign | slope | real split-half Spearman |", "|---|---|---|---|---|---|"]
        for k, v in pooled.items():
            h = v.get("high_notad") or {}
            sh = v.get("real_split_half", {})
            lines.append(f"| {k} | {v['n_donors']} | {_fmt(h.get('spearman'))} | "
                         f"{_fmt(h.get('top_decile_sign_agreement'))} | {_fmt(h.get('regression_slope'))} | {_fmt(sh.get('spearman'))} |")
        lines.append("")
    if agg.get("incomplete_fits"):
        lines += [f"Incomplete fits skipped: {len(agg['incomplete_fits'])}", ""]
    if agg.get("shadowed"):
        lines += [f"Units found in more than one root (first root used): {len(agg['shadowed'])}", ""]
    if agg.get("errors"):
        lines += [f"Units that failed evaluation: {len(agg['errors'])} (see aggregate.json)", ""]
    (Path(out_root) / "summary.txt").write_text("\n".join(lines) + "\n")


# ==========================================================================
# Redraw-SD measurement (chooses the default repeat counts)
# ==========================================================================
def measure_redraw(fit_dir, cache_dir, draws, out=None):
    """Single-draw SD of the round-1 pc / cell metrics of one fit over
    independent real-subsample draws, and the R needed for SD/sqrt(R) < 0.01."""
    cache_dir = Path(cache_dir)
    real = load_real(cache_dir, cache_dir.parent.name, int(cache_dir.name.removeprefix("fold")))
    p = Path(fit_dir)
    gen = align_generated(load_generated(p / "generated.npz" if p.is_dir() else p), real)
    t0 = time.monotonic()
    rep = pc_space_repeated(gen, real, draws, draws)
    sec = time.monotonic() - t0
    out_d = {"fit": str(fit_dir), "draws": draws, "seconds_for_draws": sec, "per_set": {}}
    for e in ["heldout", "train"]:
        blk = {}
        for k in ["energy_score", "coverage80", "spread_ratio", "within_stage_spread_ratio", "cell_energy_distance",
                  "pooled_cell_energy_distance"]:
            sd = rep[e].get(k + "_draw_sd")
            blk[k] = {"mean": rep[e].get(k), "single_draw_sd": sd,
                      "R_for_sd_below_0.01": int(np.ceil((sd / 0.01) ** 2)) if sd else None}
        out_d["per_set"][e] = blk
    print(json.dumps(_clean(out_d), indent=1))
    if out:
        write_json(out, out_d)
    return out_d


# ==========================================================================
# Round-3 additions (R7): label parsing, stage kinds, primary read-outs,
# score-versus-updates curves, targets T1/T2/T3, leaderboard
# ==========================================================================
T1_TOLERANCE = 0.005
CALIBRATION_MODES = {"biascal"}          # T1b: generator-side train-only calibration (PROTOCOL_R3 addition 2/3)
STAGE_SHRINK_MODES = {"stagehalf"}       # pre-declared stage shrink at generation (allowed for T1; not stage-blind)
ROUND3_FACTOR_ARMS = {"nostage", "fixedstage"}   # integration: round-3 stage single-factor arms, stripped like t./g./d. for dCtrl
CONTROL_RUN_LENGTH = 20000               # repair E2: the round-3 control's run length (no t.steps flag)
CONTROL_SNAPENS_K = 5                    # the control's snapens = 5 snapshots (15000-19000) + raw final
STAGE_BLIND_ARMS = {"nostage"}           # repair E9b: arms paired with the control's stage-blind read-outs
# repair E5: prior SD of pb_rmse between two seeds of ONE single-fit snapens read-out, used for the dCtrlP
# interval until the round-3 control has >= 2 seeds in a fold.  Value: pooled within-fold SD over the round-2
# seed pairs (seeds 0/1, folds 0-2) of pbulk, genemean, genemean@0.25, genemean@0.5 and gmlite@64 #snapens,
# scored with this evaluator (smoke_r3/repair/eval_seed); see README_R3.txt section 7.  Measured 2026-11-11:
# pooled 0.0156 (15 pairs; pbulk alone 0.026 from 3 pairs).  The interval assumes the arm has the control's
# seed variance: for SINGLE-STATE read-outs (round-2 pbulk default, one raw final state) the seed SD was 0.067,
# so dCtrlP intervals of single-state primaries (t.anneal / t.ema final state) are optimistic.
SEED_SD_PRIOR = {"pb_rmse": 0.0156}
STAGE_BLIND_MODES = {"stageblind"}       # integration: stage task's stage-blind companion (= condmix@1 conditions)
T2_METRICS = [("pc_energy_rep", "lower"), ("cell_energy_rep", "lower"), ("pc_coverage80_rep", "cov80"),
              ("meanlog_rmse_gene", "lower"), ("nonzero_rmse_gene", "lower"), ("logvar_rmse_gene", "lower")]
CURVE_METRICS = ["pb_rmse", "meanlog_rmse_gene", "pc_energy_rep", "nonzero_rmse_gene", "logvar_rmse_gene",
                 "cell_energy_rep", "pc_coverage80_rep", "pb_rmse_recal"]
TARGET_GROUPS = [("folds0-2", lambda f: repeat_of(f) == 0), ("folds3-8", lambda f: repeat_of(f) >= 1)]
STAGE_SLOPE_BLIND = 0.05                 # |slope of mean generated ADNC code on target stage| below this: stage-blind
STAGE_SLOPE_FULL = 0.95                  # above this: stage-conditional; in between: stage-shrunk
_STAGE_TAGS = {"stage-blind": " [STAGE-BLIND]", "stage-mixture": " [stage-mixture]", "stage-shrunk": " [stage-shrunk]"}


def decay_text(d):
    """Canonical decay text ('%g'), as r3/arms.decay_text."""
    return "%g" % float(d)


def label_parts(label):
    """'arm[~tag][#mode][ [seed-ensemble ...]]' -> dict(unit, arm, tag, mode, components, suffix)."""
    unit, sep, suffix = label.partition(" [")
    head, _, mode = unit.partition("#")
    arm, _, tag = head.partition("~")
    mode = mode or None
    return {"unit": unit, "arm": arm, "tag": tag or None, "mode": mode, "suffix": (sep + suffix) if sep else "",
            "components": mode_components(mode), "base_unit": head}


def mode_components(mode):
    """Composite modes are '+'-joined (e.g. 'biascal+snapens'); a plain mode is one component.

    Integration (2026-11-11): the stage task's modes use a nested syntax instead of '+', so they are
    expanded here: ``biascal@INNER`` -> ['biascal'] + components(INNER with its first ':' read as '@')
    (bare ``biascal`` = the default 'final' read-out); ``stagehalf`` / ``stageblind`` without a value ->
    [name, 'primary'] (they generate from the PRIMARY read-out states of PROTOCOL_R3 item 4, see
    mode_stagedefault.primary_states); ``@final`` -> [name]; ``@ema<D>`` -> [name, 'ema@<D>'];
    ``@snapens`` / ``@raw`` -> [name, value]."""
    if not mode:
        return []
    out = []
    for c in str(mode).split("+"):
        if c:
            out += _expand_component(c)
    if PRIMARY_COMPONENT in out and any(c != PRIMARY_COMPONENT and c.partition("@")[0] not in ("stagehalf", "stageblind", "biascal")
                                        for c in out):
        out = [c for c in out if c != PRIMARY_COMPONENT]   # an explicit '+' read-out (e.g. stagehalf+snapens) names the states
    return out


PRIMARY_COMPONENT = "primary"


def _expand_component(c):
    name, at, v = c.partition("@")
    if name == "biascal":
        return ["biascal"] + (_expand_component(v.replace(":", "@", 1)) if at and v else [])
    if name in ("stagehalf", "stageblind"):
        if not at:
            return [name, PRIMARY_COMPONENT]
        if v == "final":
            return [name]
        if v.startswith("ema") and not v.startswith("ema@"):
            return [name, "ema@" + v[3:]]
        return [name, v]
    return [c]


def _comp_names(comps):
    return {c.partition("@")[0] for c in comps}


def stage_blind_mode_r3(mode, M=32):
    """Name rule: round-2 stage_blind_mode on every '+' component (any stage-blind component makes the
    unit stage-blind, else any stage-mixture component makes it a mixture); STAGE_SHRINK_MODES ->
    'stage-shrunk'.  For a plain round-2 mode this is exactly stage_blind_mode."""
    comps = mode_components(mode)
    kinds = [stage_blind_mode(c, M) for c in comps]
    if "stage-blind" in kinds or _comp_names(comps) & STAGE_BLIND_MODES:
        return "stage-blind"
    if "stage-mixture" in kinds:
        return "stage-mixture"
    if _comp_names(comps) & STAGE_SHRINK_MODES:
        return "stage-shrunk"
    return None


def condition_stage_info(path):
    """For a mode file with a 'conditions' array [T,M,7] (r2/r3 generate.py): OLS slope of the per-target
    mean generated ADNC code (column 3) on the target's own ADNC code (stage/3).  1 = conditioned on
    the target's stage, 0 = the same stage input for every target.  None when not available."""
    try:
        with np.load(path, allow_pickle=False) as f:
            if "conditions" not in f.files or "stage" not in f.files:
                return None
            c = f["conditions"].astype(np.float64); st = f["stage"].astype(np.float64) / 3.0
    except Exception:
        return None
    if c.ndim != 3 or c.shape[-1] < 4 or st.std() == 0:
        return None
    a = c[:, :, 3].mean(1)
    slope = float(np.cov(a, st, ddof=0)[0, 1] / st.var())
    return {"adnc_slope_on_target_stage": slope, "definition": condition_stage_info.__doc__.split("For a mode file")[1].strip()}


def unit_stage_kinds(fit_results):
    """{unit: kind} from the name rule, then a 'stage_blind' key in the mode receipt or fit config,
    then the generated conditions (condition_stage_info)."""
    out = {}
    for r in fit_results:
        unit = r["unit"]
        k = stage_blind_mode_r3(r.get("mode"))
        cfg0 = r.get("config") if isinstance(r.get("config"), dict) else {}
        if any(isinstance(v, dict) and isinstance(v.get("info"), dict) and v["info"].get("stage_blind_generator") is True
               for v in (cfg0.get("r2_arms") or {}).values()):
            k = "stage-blind"      # repair E9a: the generator ignores stage whatever the read-out (fixedstage, zero table)
        if k is None:
            mr = r.get("mode_receipt") if isinstance(r.get("mode_receipt"), dict) else {}
            cfg = r.get("config") if isinstance(r.get("config"), dict) else {}
            arm_info = [v.get("info", {}) for v in (cfg.get("r2_arms") or {}).values() if isinstance(v, dict)]
            if mr.get("stage_blind") is True or cfg.get("stage_blind") is True or \
                    any(isinstance(i, dict) and (i.get("stage_blind") is True or i.get("stage_blind_generator") is True)
                        for i in arm_info):   # integration: nostage; repair E9a: fixedstage with a zero offset table
                k = "stage-blind"
            else:
                ci = r.get("r3_condition_stage") or {}
                s = ci.get("adnc_slope_on_target_stage")
                if s is not None and r.get("mode") is not None:
                    if abs(s) < STAGE_SLOPE_BLIND:
                        k = "stage-blind"
                    elif s < STAGE_SLOPE_FULL:
                        k = "stage-shrunk"
        if k is not None:
            prev = out.get(unit)
            out[unit] = k if prev in (None, k) else ("stage-blind" if "stage-blind" in (prev, k) else prev)
        else:
            out.setdefault(unit, None)
    return out


def fit_round_of(res):
    cfg = res.get("config") if isinstance(res.get("config"), dict) else {}
    return int(cfg.get("round", 1)) if cfg else None


def primary_readout(label, fit_round):
    """PROTOCOL_R3 addition 4 (declared before any round-3 score): constant-learning-rate arms are read
    with 'snapens' (fits with snapshots, rounds 2-3; round-1 fits have none, so their default); annealed
    (t.anneal) arms with the final state (default); native-EMA (t.ema) arms with the EMA generator
    (default).  Stage-conditional arms are additionally read with 'stagehalf'.  Returns 'P' (primary),
    'P+stagehalf' or 'P+biascal' / 'P+stagehalf+biascal' (the T1b read-out), or None.  Composition
    syntax assumed: '+'-joined mode components (stage task / integrator to confirm)."""
    p = label_parts(label)
    flags = p["arm"].split(",")
    names = _comp_names(p["components"])
    core = {c for c in p["components"] if c.partition("@")[0] not in (STAGE_SHRINK_MODES | CALIBRATION_MODES)}
    if any(f.startswith(("t.ema@", "t.anneal@")) for f in flags) or not fit_round or fit_round < 2:
        expected = set()
    else:
        expected = {"snapens"}
    if core != expected and core != {PRIMARY_COMPONENT}:   # integration: bare stagehalf / biascal@stagehalf use the primary states
        return None
    tag = "P"
    if names & STAGE_SHRINK_MODES:
        tag += "+stagehalf"
    if names & CALIBRATION_MODES:
        tag += "+biascal"
    return tag


def _control_arm(p):
    flags = p["arm"].split(",")
    keep = [f for f in flags if not re.match(r"^[tgd]\.", f) and f.partition("@")[0] not in ROUND3_FACTOR_ARMS]
    if len(keep) == len(flags):
        return None, flags
    return (",".join(sorted(keep)) if keep else "base"), flags


_NO_PAIR = object()


def control_mode(mode, flags):
    """Repair E2/E9b: the control read-out matched to the arm's read-out ``mode`` (None = default generation;
    _NO_PAIR = the control has no matching read-out)."""
    blind = any(f.partition("@")[0] in STAGE_BLIND_ARMS for f in flags)
    if mode is not None:
        name, _, v = mode.partition("@")
        if name == "snap" and v == str(CONTROL_RUN_LENGTH):
            mode = None                                   # the control's update-20000 state is its default generation
        elif name == "snapema" and v.partition(":")[0] == str(CONTROL_RUN_LENGTH):
            mode = "ema@" + v.partition(":")[2]           # the control's final shadow EMA
        elif mode == f"snapens@{CONTROL_SNAPENS_K}":
            mode = "snapens"                              # window-matched snapshot ensemble
    if blind:                                             # stage-blind arm: only stage-blind control read-outs
        return {None: "condmix@1", "snapens": "stageblind"}.get(mode, _NO_PAIR)
    return mode


def control_label(label):
    """Round-3 single-factor control: the same label with every t./g./d. flag (and nostage / fixedstage)
    removed (same tag and ensemble suffix), read with the matched read-out of control_mode.  None if the
    arm has no such flag or the control has no matching read-out."""
    p = label_parts(label)
    arm, flags = _control_arm(p)
    if arm is None:
        return None
    m = control_mode(p["mode"], flags)
    if m is _NO_PAIR:
        return None
    return arm + (f"~{p['tag']}" if p["tag"] else "") + (f"#{m}" if m else "") + p["suffix"]


_CONTROL_PRIMARY_MODE = {"P": "snapens", "P+stagehalf": "stagehalf", "P+stagehalf+biascal": "biascal@stagehalf",
                         "P+biascal": "biascal@snapens"}


def control_primary_label(label, fit_round):
    """Repair E2: the round-3 control's PRIMARY read-out matching the arm's primary read-out (the control trains
    with constant rates, so its primary is snapens; +stagehalf / +biascal as declared; a stage-blind arm
    (nostage) is paired with the control's ``stageblind``).  None when ``label`` is not a primary read-out."""
    pr = primary_readout(label, fit_round)
    p = label_parts(label)
    arm, flags = _control_arm(p)
    if pr is None or arm is None:
        return None
    if any(f.partition("@")[0] in STAGE_BLIND_ARMS for f in flags):
        m = {"P": "stageblind"}.get(pr)
    else:
        m = _CONTROL_PRIMARY_MODE.get(pr)
    if m is None:
        return None
    return arm + (f"~{p['tag']}" if p["tag"] else "") + f"#{m}" + p["suffix"]


def length_reference_label(label):
    """Repair TR-2/E4: for an arm with t.steps@N (N > 20000), the SAME run's matched read-out at update 20000:
    ``ema@D`` -> ``snapema@20000:D``; default / ``raw`` -> ``snap@20000`` (constant rates: the long run's first
    20000 updates are the 20000-update trajectory, bitwise on GPU).  None otherwise.  This is the training-length
    comparison that needs no separate 20000-update fit (plain base t.steps@60000 has none)."""
    p = label_parts(label)
    steps = [int(f.partition("@")[2]) for f in p["arm"].split(",") if f.startswith("t.steps@")]
    if not steps or steps[0] <= CONTROL_RUN_LENGTH or p["suffix"]:
        return None
    m = p["mode"]
    if m is None or m == "raw":
        ref = f"snap@{CONTROL_RUN_LENGTH}"
    elif m.startswith("ema@"):
        ref = f"snapema@{CONTROL_RUN_LENGTH}:{m.partition('@')[2]}"
    else:
        return None
    return p["base_unit"] + "#" + ref


def seed_noise_se(units, label, ref_label, panel, folds, metric="pb_rmse"):
    """Repair E5: SE of the paired difference label - ref_label (single-seed fits): sqrt(sum_f 2 s^2 / k_f) / F
    with k_f the common seeds in fold f and s^2 the reference's pooled within-fold seed variance (folds with
    >= 2 seeds) or SEED_SD_PRIOR[metric]^2.  Returns (se, source) or (None, None)."""
    mine = {(u[0], u[1]) for u in units.get((label, panel), []) if u[0] in folds}
    ref = {}
    for f, sd_, flat, _ in units.get((ref_label, panel), []):
        v = flat.get(metric)
        if f in folds and v is not None and np.isfinite(v):
            ref.setdefault(f, {})[sd_] = float(v)
    k = {}
    for (f, sd_) in mine:
        if sd_ in ref.get(f, {}):
            k[f] = k.get(f, 0) + 1
    if not k:
        return None, None
    var = [np.var(list(d.values()), ddof=1) for d in ref.values() if len(d) >= 2]
    if var:
        s2, src = float(np.mean(var)), f"control seeds ({sum(len(d) for d in ref.values() if len(d) >= 2)} fits)"
    elif SEED_SD_PRIOR.get(metric):
        s2, src = float(SEED_SD_PRIOR[metric]) ** 2, "round-2 prior"
    else:
        return None, None
    F = len(k)
    return float(np.sqrt(sum(2 * s2 / kf for kf in k.values())) / F), src


def _fold_means(units, label, panel, metric, folds=None, seeds=None):
    out = {}
    for f, s, flat, _ in units.get((label, panel), []):
        if (folds is not None and f not in folds) or (seeds is not None and s not in seeds):
            continue
        v = flat.get(metric)
        if v is not None and np.isfinite(v):
            out.setdefault(f, []).append(float(v))
    return {f: float(np.mean(v)) for f, v in out.items()}


def _rb(d):
    return repeat_balanced(d)[0] if d else None


def paired_delta(units, label, ref_label, panel, folds, metric="pb_rmse"):
    """Unit minus reference on common (fold, seed) keys: per-fold mean difference, repeat-balanced;
    n_better = folds where the unit is lower."""
    mine = {(u[0], u[1]): u[2].get(metric) for u in units.get((label, panel), []) if u[0] in folds}
    ref = {(u[0], u[1]): u[2].get(metric) for u in units.get((ref_label, panel), []) if u[0] in folds}
    keys = [k for k in mine if k in ref and mine[k] is not None and ref[k] is not None]
    if not keys:
        return None
    byf = {}
    for k in keys:
        byf.setdefault(k[0], []).append(mine[k] - ref[k])
    fm = {f: float(np.mean(v)) for f, v in byf.items()}
    return {"reference": ref_label, "mean_difference": _rb(fm), "n_folds": len(fm), "n_pairs": len(keys),
            "n_better": int(sum(v < 0 for v in fm.values()))}


def _target_block(units, label, panel, folds, seeds=None):
    """T1 / T2 of one label over the given folds (common folds with each reference)."""
    out = {}
    u = _fold_means(units, label, panel, "pb_rmse", folds, seeds)
    gm = _fold_means(units, "baseline:grand_mean", panel, "pb_rmse", folds)
    common = sorted(set(u) & set(gm))
    if common:
        val = _rb({f: u[f] for f in common}); ref = _rb({f: gm[f] for f in common})
        out["T1"] = {"pb_rmse": val, "grand_mean": ref, "threshold": ref + T1_TOLERANCE, "pass": bool(val - (ref + T1_TOLERANCE) <= 1e-12),
                     "margin": val - (ref + T1_TOLERANCE), "folds": common}
    else:
        out["T1"] = None
    t2, n_pass, n_av = {}, 0, 0
    for m, kind in T2_METRICS:
        a = _fold_means(units, label, panel, m, folds, seeds)
        r = _fold_means(units, "baseline:resample", panel, m, folds)
        c = sorted(set(a) & set(r))
        if not c:
            t2[m] = {"available": False, "pass": False}
            continue
        va, vr = _rb({f: a[f] for f in c}), _rb({f: r[f] for f in c})
        margin = (abs(va - .8) - abs(vr - .8)) if kind == "cov80" else (va - vr)
        ok = bool(margin <= 1e-12)        # ties pass (1e-12 absorbs float rounding of equal values)
        n_av += 1; n_pass += int(ok)
        t2[m] = {"available": True, "value": va, "resample": vr, "margin": margin, "pass": ok, "folds": c,
                 "rule": "|v-0.8| <= |resample-0.8|" if kind == "cov80" else "v <= resample"}
    out["T2"] = {"n_pass": n_pass, "n_available": n_av, "n_metrics": len(T2_METRICS), "pass": n_pass == len(T2_METRICS),
                 "metrics": t2,
                 "worst_miss": max(((m, d["margin"]) for m, d in t2.items() if d.get("available") and not d["pass"]),
                                   key=lambda x: x[1], default=None),
                 "missing": [m for m, d in t2.items() if not d.get("available")]}   # repair E11e: not passed, no value
    return out


def build_targets(units, stage_kind, unit_round, panel_list):
    """targets.json content: per label and fold group, T1 (T1a / T1b) and T2 (count of 6 and margins),
    per-seed values for single-fit labels, T3 per unit (single seed and seed ensemble, both fold groups)."""
    res = {"definition": {
        "T1": f"held-out pb_rmse (fold-balanced over the unit's folds) <= baseline:grand_mean on the same folds + {T1_TOLERANCE}; "
              "T1a = no calibration, T1b = a 'biascal' generation mode; recalibrated (*_recal) metrics are never used",
        "T2": "each of " + ", ".join(m for m, _ in T2_METRICS) + " at least as good as baseline:resample on the same folds "
              "(coverage: |v-0.8| <= |resample-0.8|); margin > 0 = miss by that much; a metric the unit lacks (e.g. "
              "logvar for round-1 files without sq_log) counts as not passed",
        "T3": "T1 and T2 from a single seed (label without ensemble) AND from a seed ensemble (largest full ensemble), "
              "on folds 0-2 AND on folds 3-8, each with every fold of the group present",
        "stage": "stage-blind units (mode or fit gives every target the same stage input) are listed separately; a "
                 "stage-conditioned generator that reaches T1 only by ignoring stage has not reached it as a stage model. "
                 "A 'stage-shrunk' T1a pass through stagehalf is HALF stage-blind: 16 of the 32 generated donors per target "
                 "get the training stage mixture (repair E9c). fixedstage fits with a numerically zero offset table "
                 "(stage_blind_generator) make their unit STAGE-BLIND (repair E9a)",
        "T1_reference_selection": "the grand_mean baseline's v4 residual centring was kept in round 1 over v3 because v3 "
                                  "scored worse on HELD-OUT donors (0.500 vs 0.464); this makes T1 stricter, not looser. "
                                  "The floor routes of r3/floor.py (A/B, about 0.452-0.455) do not depend on it (repair E12)",
        "vs_length_reference": "t.steps@N arms: read-out at N minus the same run's matched read-out at update 20000 "
                               "(length_reference_label; seed-paired by construction, se = seed_noise_se is conservative)",
        "vs_control_primary": "arm primary read-out minus the round-3 control's primary read-out (control_primary_label) on "
                              "common folds and seeds, with se = seed_noise_se (repair E2/E5)",
        "primary": primary_readout.__doc__},
        "units": {}, "T3": {}, "lists": {}}
    for (label, panel), us in sorted(units.items()):
        if label.startswith("baseline:") or panel not in panel_list:
            continue
        p = label_parts(label)
        kind = stage_kind.get(label)
        rnd = unit_round.get(p["unit"])
        variant = "T1b" if _comp_names(p["components"]) & CALIBRATION_MODES else "T1a"
        entry = {"panel": panel, "stage_kind": kind, "stage_blind": kind == "stage-blind", "T1_variant": variant,
                 "primary_readout": primary_readout(label, rnd), "fit_round": rnd, "ensemble": bool(p["suffix"]),
                 "control": control_label(label), "control_primary": control_primary_label(label, rnd),
                 "length_reference": length_reference_label(label), "groups": {}}
        all_folds = sorted(set(u[0] for u in us))
        for gname, inside in TARGET_GROUPS:
            gf = [f for f in all_folds if inside(f)]
            if not gf:
                continue
            want = [f for f in range(9) if inside(f)]
            blk = _target_block(units, label, panel, gf)
            blk["folds_present"] = gf
            for key, ref in (("vs_control", entry["control"]), ("vs_control_primary", entry["control_primary"]),
                             ("vs_length_reference", entry["length_reference"])):
                d = paired_delta(units, label, ref, panel, gf) if ref else None
                if d:
                    d["se"], d["se_source"] = seed_noise_se(units, label, ref, panel, gf)
                blk[key] = d
            blk["complete"] = bool(set(gf) >= set(want))
            if not p["suffix"]:
                seeds = sorted(set(u[1] for u in us if u[0] in gf))
                blk["per_seed"] = {}
                for s in seeds:
                    b = _target_block(units, label, panel, gf, {s})
                    blk["per_seed"][str(s)] = {"T1_pass": (b["T1"] or {}).get("pass"), "T1_margin": (b["T1"] or {}).get("margin"),
                                               "pb_rmse": (b["T1"] or {}).get("pb_rmse"), "T2_n_pass": b["T2"]["n_pass"],
                                               "folds": (b["T1"] or {}).get("folds")}
                blk["seeds"] = seeds
            entry["groups"][gname] = blk
        res["units"][f"{label}|{panel}"] = entry
    # T3 per unit (single label + its largest full seed ensemble)
    byunit = {}
    for key, e in res["units"].items():
        label, panel = key.rsplit("|", 1)
        p = label_parts(label)
        d = byunit.setdefault((p["unit"], panel), {"single": None, "ensemble": None, "ens_n": 0})
        if not p["suffix"]:
            d["single"] = key
        elif "M=32" not in p["suffix"]:
            n = int(re.search(r"n=(\d+)", p["suffix"]).group(1))
            if n > d["ens_n"]:
                d["ensemble"], d["ens_n"] = key, n
    for (unit, panel), d in sorted(byunit.items()):
        comp = {}
        for which in ("single", "ensemble"):
            for gname, _ in TARGET_GROUPS:
                e = res["units"].get(d[which]) if d[which] else None
                g = (e or {}).get("groups", {}).get(gname)
                if g is None or g.get("T1") is None:
                    comp[f"{which}_{gname}"] = None
                else:
                    comp[f"{which}_{gname}"] = bool(g["T1"]["pass"] and g["T2"]["pass"] and g["complete"])
        res["T3"][f"{unit}|{panel}"] = {"components": comp, "pass": all(v is True for v in comp.values()),
                                        "single": d["single"], "ensemble": d["ensemble"]}
    for gname, _ in TARGET_GROUPS:
        for v in ("T1a", "T1b"):
            for sb in (False, True):
                lst = sorted(k for k, e in res["units"].items() if e["T1_variant"] == v and e["stage_blind"] == sb
                             and ((e["groups"].get(gname) or {}).get("T1") or {}).get("pass"))
                res["lists"][f"{v}_pass_{gname}_{'stage_blind' if sb else 'stage_model'}"] = lst
        res["lists"][f"T2_pass_{gname}"] = sorted(k for k, e in res["units"].items()
                                                  if ((e["groups"].get(gname) or {}).get("T2") or {}).get("pass"))
    res["lists"]["T3_pass"] = sorted(k for k, v in res["T3"].items() if v["pass"])
    return res


def build_curves(fit_results, units, panel_list):
    """Score-versus-updates curves (curves.json).  Per fit (unit without mode, panel, fold, seed):
    series 'raw' (snap@U at U, and the raw final state at the run length: mode 'raw', or the default
    generation when the fit has no native EMA), 'ema@D' (snapema@U:D at U, ema@D at the run length)
    and 'native_ema@D' (default generation of a t.ema fit); other modes are listed as read-outs.
    Per series: best update by pb_rmse, final value, late slope (OLS per 1000 updates over points with
    u >= 0.75 x run length, >= 3 points) and late SD; EMA series also give init_weight_share = D^U per point
    (repair TR-3: share of the untrained initial weights; points above 0.01 are listed).  Aggregated per (unit, panel, series): fold-balanced
    mean per update (seeds averaged within fold, folds within repeat, repeats averaged) with n_folds."""
    groups = {}
    for r in fit_results:
        if r["panel"] not in panel_list:
            continue
        base = r["unit"].split("#", 1)[0]
        g = groups.setdefault((base, r["panel"], r["fold"], r["seed"]), {"points": {}, "config": None})
        if isinstance(r.get("config"), dict) and g["config"] is None:
            g["config"] = r["config"]
        g["points"][r.get("mode")] = headline(r)
    per_fit, agg_pts = {}, {}
    for (base, panel, fold, seed), g in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1], x[0][2], x[0][3])):
        cfg = g["config"] or {}
        U = cfg.get("updates")
        flags = cfg.get("flags") or []
        native = cfg.get("native_ema_decay")
        if not native:   # integration: config records 0.0 (not None) when there is no native EMA
            native = None
            for fl in flags:
                if str(fl).startswith("t.ema@"):
                    native = float(str(fl).split("@", 1)[1])
        series, readouts = {}, {}
        for mode, flat in g["points"].items():
            vals = {m: flat.get(m) for m in CURVE_METRICS}
            if mode is None:
                if U is None:
                    continue
                if native is not None:
                    name, u = f"native_ema@{decay_text(native)}", U
                elif "raw" in g["points"]:
                    readouts["default"] = vals; continue
                else:
                    name, u = "raw", U
            elif mode == "raw":
                name, u = "raw", U
            elif re.fullmatch(r"snap@\d+", mode):
                name, u = "raw", int(mode.split("@")[1])
            elif re.fullmatch(r"snapema@\d+:[0-9.eE+-]+", mode):
                uu, d = mode.split("@")[1].split(":")
                name, u = f"ema@{decay_text(d)}", int(uu)
            elif re.fullmatch(r"ema@[0-9.eE+-]+", mode):
                name, u = f"ema@{decay_text(mode.split('@')[1])}", U
            else:
                readouts[mode] = vals; continue
            if u is None:
                continue
            series.setdefault(name, {})[int(u)] = vals
        if sum(len(v) for v in series.values()) < 2:      # a curve needs two points (any series)
            continue
        entry = {"unit": base, "panel": panel, "fold": fold, "seed": seed, "run_length": U,
                 "recipe_total_steps": cfg.get("recipe_total_steps"), "fit_round": cfg.get("round", 1),
                 "snapshot_updates": cfg.get("snapshot_updates"), "series": {}, "readouts": readouts}
        for name, pts in series.items():
            us = sorted(pts)
            s = {"updates": us, "values": {m: [pts[u].get(m) for u in us] for m in CURVE_METRICS}}
            if "ema@" in name:   # repair TR-3: the EMA starts from the initial weights (no bias correction)
                dd_ = float(name.split("@")[1])
                s["init_weight_share"] = [dd_ ** u for u in us]
                s["init_share_above_0.01"] = [u for u in us if dd_ ** u > 0.01]
            pb = [(u, pts[u].get("pb_rmse")) for u in us if pts[u].get("pb_rmse") is not None]
            if pb:
                bu, bv = min(pb, key=lambda x: x[1])
                s["best_update_pb_rmse"] = bu; s["best_pb_rmse"] = bv
                s["last_update"] = us[-1]; s["last_pb_rmse"] = pts[us[-1]].get("pb_rmse")
            if U:
                late = [u for u in us if 4 * u >= 3 * U]
                s["late_updates"] = late
                for m in ("pb_rmse", "meanlog_rmse_gene", "pc_energy_rep"):
                    xy = [(u, pts[u].get(m)) for u in late if pts[u].get(m) is not None]
                    if len(xy) >= 3:
                        x = np.array([a for a, _ in xy], float) / 1000.0; y = np.array([b for _, b in xy], float)
                        s[f"late_slope_per_1000_{m}"] = float(np.polyfit(x, y, 1)[0])
                        s[f"late_sd_{m}"] = float(y.std(ddof=1))
            entry["series"][name] = s
            for u in us:
                for m in CURVE_METRICS:
                    v = pts[u].get(m)
                    if v is not None:
                        agg_pts.setdefault((base, panel, name), {}).setdefault(int(u), {}).setdefault(m, {}).setdefault(fold, []).append(v)
        per_fit[f"{base}|{panel}|fold{fold}|seed{seed}"] = entry
    agg = {}
    for (base, panel, name), byu in sorted(agg_pts.items()):
        blk = {"updates": sorted(byu), "metrics": {}}
        for m in CURVE_METRICS:
            vals, nf = [], []
            for u in blk["updates"]:
                fm = {f: float(np.mean(v)) for f, v in byu[u].get(m, {}).items()}
                vals.append(_rb(fm)); nf.append(len(fm))
            blk["metrics"][m] = {"value": vals, "n_folds": nf}
        agg[f"{base}|{panel}|{name}"] = blk
    refs = {}
    for panel in panel_list:
        for bname in ("grand_mean", "resample", "resample_blind", "ridge"):
            for m in ("pb_rmse", "meanlog_rmse_gene", "pc_energy_rep"):
                fm = _fold_means(units, f"baseline:{bname}", panel, m)
                if fm:
                    refs[f"{bname}|{panel}|{m}"] = {"per_fold": {str(f): v for f, v in sorted(fm.items())}}
    return {"definition": build_curves.__doc__, "per_fit": per_fit, "aggregate": agg, "references": refs}


def _best_r2(units, stage_kind, panel, folds):
    """Lowest pb_rmse round-2 single-fit unit (tag ~r2, not stage-blind, no calibration) with every fold of
    the group; the leaderboard's 'best round-2 unit' reference."""
    best = None
    for (label, pn), us in units.items():
        p = label_parts(label)
        if pn != panel or p["tag"] != "r2" or p["suffix"] or stage_kind.get(label) == "stage-blind" \
                or _comp_names(p["components"]) & CALIBRATION_MODES:
            continue
        fm = _fold_means(units, label, panel, "pb_rmse", folds)
        if set(fm) != set(folds):
            continue
        v = _rb(fm)
        if best is None or v < best[1]:
            best = (label, v)
    return best


def _fmt3(v, plus=False):
    if v is None:
        return "-"
    return f"{v:+.3f}" if plus else f"{v:.3f}"


def write_leaderboard(out_root, units, stage_kind, targets, unit_round, panel_list, floor=None, top=None):
    """leaderboard.txt: per panel and fold group, units ranked by held-out pb_rmse (units with every fold of
    the group first, partial units after), with reference rows (floors, baselines, the T1 threshold)."""
    lines = ["improve_20261109 round-3 leaderboard (auto-generated by r3/evaluate.py)", "",
             "Held-out pb_rmse, fold-balanced over the unit's folds in the group (seeds averaged within fold). "
             "T1 = pb_rmse <= grand_mean + 0.005 on the same folds (T1a raw, T1b biascal); T2 = k of 6 metrics "
             "at least as good as real-donor resampling (worst miss shown); stage: BLIND = every target gets the same "
             "stage input (reaching T1 this way is not a stage model), shrunk/mixture = partial; P = primary read-out "
             "declared in PROTOCOL_R3 addition 4. dBase / dBestR2 / dCtrl = paired difference (unit - reference) on "
             "common folds and seeds, with (folds better/folds); dCtrl pairs the matched read-out of the round-3 control "
             "(evaluate.control_mode), dCtrlP pairs the unit's PRIMARY read-out with the control's primary read-out, "
             "+- 2 SE of seed noise (seed_noise_se). One seed on 3 folds resolves only differences beyond about 2 SE: "
             "a stage-1 'null' is uninformative. Development evidence only; selection-conditioned.", ""]
    for panel in panel_list:
        for gname, inside in TARGET_GROUPS:
            gall = [f for f in range(9) if inside(f)]
            rows = []
            for (label, pn), us in units.items():
                if pn != panel or label.startswith("baseline:"):
                    continue
                fm = _fold_means(units, label, panel, "pb_rmse", gall)
                if not fm:
                    continue
                rows.append((label, _rb(fm), sorted(fm)))
            if not rows:
                continue
            present = sorted(set(f for _, _, fs in rows for f in fs))
            best = _best_r2(units, stage_kind, panel, present)
            lines += [f"## {panel}, {gname} (folds present: {present})", ""]
            lines.append("References (same folds):")
            for bname in ("grand_mean", "ridge", "resample", "resample_blind", "stage_mean_log"):
                fm = _fold_means(units, f"baseline:{bname}", panel, "pb_rmse", present)
                if fm:
                    lines.append(f"  baseline:{bname:<16} pb_rmse {_rb(fm):.3f}" +
                                 (f"   (T1 threshold {_rb(fm) + T1_TOLERANCE:.3f})" if bname == "grand_mean" else ""))
            if floor:
                pk = "repeat0" if gname == "folds0-2" else "repeats_1_2"
                fl = (floor.get("pooled") or {}).get(pk)
                if fl:
                    for key, txt in [("pb_A_raw_train_mean", "direct train mean (no generator)"),
                                     ("pb_A_grand", "FLOOR oracle population mean (no stage)"),
                                     ("pb_A_stage", "FLOOR oracle stage means"),
                                     ("pb_D_ideal_generator", "ideal stage-blind generator, M=32 read-out")]:
                        if fl.get(key):
                            v = fl[key]; ci = v.get("ci95")
                            lines.append(f"  {txt:<45} {v['value']:.3f}" + (f"  [95% {ci[0]:.3f}-{ci[1]:.3f}]" if ci else ""))
                    lines.append(f"  (floor.json folds {fl.get('folds')}; see r3/floor.py)")
            if best:
                lines.append(f"  best round-2 unit (dBestR2 reference): {best[0]} {best[1]:.3f}")
            lines.append("")
            hdr = f"{'rank':>4}  {'unit':<66} {'pb_rmse':>7} {'folds':>5} {'seeds':<9} {'T1':<14} {'T2':<26} {'stage':<7} {'P':<4} {'dBase':<16} {'dBestR2':<16} {'dCtrl':<16} {'dCtrlP':<24}"
            blind = lambda lab: stage_kind.get(lab) == "stage-blind"
            for title, sel in (("complete units, stage model (not stage-blind)", lambda r: set(r[2]) == set(present) and not blind(r[0])),
                               ("complete units, STAGE-BLIND (listed separately: not a stage model)",
                                lambda r: set(r[2]) == set(present) and blind(r[0])),
                               ("partial units (not every fold of the group)", lambda r: set(r[2]) != set(present))):
                rs = sorted([r for r in rows if sel(r)], key=lambda r: r[1])
                if not rs:
                    continue
                lines += [f"{title} ({len(rs)}):", hdr, "-" * len(hdr)]
                for i, (label, v, fs) in enumerate(rs[:top] if top else rs, 1):
                    p = label_parts(label)
                    e = (targets["units"].get(f"{label}|{panel}") or {})
                    g = (e.get("groups") or {}).get(gname) or {}
                    t1 = g.get("T1")
                    t1s = "-" if not t1 else f"{e.get('T1_variant')} {'PASS' if t1['pass'] else 'miss'} {t1['margin']:+.3f}"
                    t2 = g.get("T2") or {}
                    wm = t2.get("worst_miss")
                    t2s = f"{t2.get('n_pass', 0)}/6" + (f" {wm[0].replace('_rmse_gene', '').replace('_rep', '')} {wm[1]:+.3f}" if wm else
                                                        (" missing " + ",".join(m.replace('_rmse_gene', '').replace('_rep', '') for m in t2.get('missing') or [])
                                                         if t2.get('missing') else ""))
                    if p["suffix"]:
                        seeds = re.search(r"n=(\d+)", p["suffix"]).group(0) + (" M32" if "M=32" in p["suffix"] else "")
                    else:
                        seeds = ",".join(str(s) for s in sorted(set(u[1] for u in units[(label, panel)] if u[0] in fs)))
                    sk = {"stage-blind": "BLIND", "stage-mixture": "mix", "stage-shrunk": "shrunk"}.get(stage_kind.get(label), "-")
                    pr = primary_readout(label, unit_round.get(p["unit"])) or "-"

                    def dd(ref):
                        if not ref or ref == label:
                            return "-"
                        d = paired_delta(units, label, ref, panel, present)
                        return "-" if not d else f"{d['mean_difference']:+.3f} ({d['n_better']}/{d['n_folds']})"
                    dctrl = dd(control_label(label))
                    dbest = dd((best[0] + p["suffix"]) if best else None)
                    cp = control_primary_label(label, unit_round.get(p["unit"]))
                    dctrlp = "-"
                    if cp and cp != label:
                        d = paired_delta(units, label, cp, panel, present)
                        if d:
                            se, _src = seed_noise_se(units, label, cp, panel, present)
                            dctrlp = f"{d['mean_difference']:+.3f}" + (f"+-{2 * se:.3f}" if se else "") + f" ({d['n_better']}/{d['n_folds']})"
                    lines.append(f"{i:>4}  {label[:66]:<66} {v:7.3f} {len(fs):>5} {seeds[:9]:<9} {t1s:<14} {t2s[:26]:<26} {sk:<7} {pr[:4]:<4} "
                                 f"{dd(_base_label(label)):<16} {dbest:<16} {dctrl:<16} {dctrlp:<24}")
                lines.append("")
    (Path(out_root) / "leaderboard.txt").write_text("\n".join(lines) + "\n")
    return lines


def r3_outputs(out_root, units, stage_kind, fit_results, opts=None):
    """Write curves.json, targets.json and leaderboard.txt (round-3 additions)."""
    out_root = Path(out_root)
    panel_list = sorted(set(p for _, p in units))
    unit_round = {}
    for r in fit_results:
        rr = fit_round_of(r)
        if rr is not None:
            unit_round[r["unit"]] = rr
    for (label, _), us in units.items():
        for u in us:
            if isinstance(u[3], dict) and u[3].get("kind") == "seed_ensemble":
                unit_round.setdefault(u[3]["unit"], unit_round.get(u[3]["unit"]))
    curves = build_curves(fit_results, units, panel_list)
    write_json(out_root / "curves.json", {"evaluator_version": EVALUATOR_VERSION, **curves})
    targets = build_targets(units, stage_kind, unit_round, panel_list)
    targets["evaluator_version"] = EVALUATOR_VERSION
    write_json(out_root / "targets.json", targets)
    floor = None
    fp = getattr(opts, "floor", None) or (out_root / "floor.json")
    if fp and Path(fp).exists():
        try:
            floor = json.loads(Path(fp).read_text())
        except json.JSONDecodeError:
            floor = None
    write_leaderboard(out_root, units, stage_kind, targets, unit_round, panel_list, floor)
    return curves, targets


def _check_out_dir(out):
    """Refuse frozen trees; inside runs/improve_20261109 only results_r5/ and smoke_r5/ (round-5 rule; round 3: results_r3/smoke_r3)."""
    o = Path(out).resolve()
    for name in FROZEN_OUTPUTS:
        root = (RUNS / name).resolve()
        if o == root or root in o.parents:
            raise PermissionError(f"refusing to write evaluator output into frozen round-1 tree {root}")
    runs = RUNS.resolve()
    if o == runs or runs in o.parents:
        if not any(o == (runs / w) or (runs / w) in o.parents for w in R3_WRITABLE):
            raise PermissionError(f"refusing to write {o}: inside {runs} round-3 outputs go only to {R3_WRITABLE}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fits-root", action="append", default=None,
                    help="repeatable; [TAG=]DIR (TAG labels units arm~TAG); default fits, r2=fits_r2, r3=fits_r3")
    ap.add_argument("--modes-root", action="append", default=[],
                    help="repeatable; [TAG=]DIR of mode files written with generate.py --out (tag inherited from the fit's root)")
    ap.add_argument("--floor", default=None, help="floor.json of r3/floor.py for the leaderboard (default <out>/floor.json)")
    ap.add_argument("--caches-root", default=str(RUNS / "caches"))
    ap.add_argument("--out", default=str(RUNS / "results_r5"))   # round 5 (was results_r3)
    ap.add_argument("--no-baselines", action="store_true")
    ap.add_argument("--no-ensembles", action="store_true")
    ap.add_argument("--no-modes", action="store_true")
    ap.add_argument("--panels", default=",".join(PANELS))
    ap.add_argument("--folds", default="0,1,2,3,4,5,6,7,8")
    ap.add_argument("--seeds", default=None, help="comma list; default all seeds")
    ap.add_argument("--arm", action="append", default=None, help="fnmatch pattern on the unit name (repeatable), e.g. 'base' 'genemean*'")
    ap.add_argument("--ensemble-sizes", default="2", help="extra first-n-seed ensembles when more seeds exist")
    ap.add_argument("--ensemble-arm", action="append", default=None,
                    help="r3: fnmatch pattern(s) on the unit name; seed ensembles only for matching units (default all)")
    ap.add_argument("--pc-repeats", type=int, default=DEFAULT_PC_REPEATS)
    ap.add_argument("--cell-repeats", type=int, default=DEFAULT_CELL_REPEATS)
    ap.add_argument("--dam-draws", type=int, default=DAM_NULL_DRAWS)
    ap.add_argument("--xfit-permutations", type=int, default=XFIT_PERMUTATIONS)
    ap.add_argument("--measure-redraw", default=None, help="fit dir (or npz): print single-draw SDs and exit")
    ap.add_argument("--cache", default=None, help="cache dir for --measure-redraw")
    ap.add_argument("--draws", type=int, default=64)
    ap.add_argument("--measure-out", default=None)
    args = ap.parse_args(argv)
    set_torch_threads()
    if args.measure_redraw:
        return measure_redraw(args.measure_redraw, args.cache, args.draws, args.measure_out)
    _check_out_dir(args.out)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    t_start = time.monotonic()
    timings = {}

    def log(msg):
        print(json.dumps({"t": time.strftime("%H:%M:%S"), "msg": msg}), flush=True)

    fits_roots = args.fits_root or [str(RUNS / "fits"), "r2=" + str(RUNS / "fits_r2"), "r3=" + str(RUNS / "fits_r3")]
    rc = RealCache(args.caches_root, args.dam_draws)
    panels = args.panels.split(","); folds = [int(x) for x in args.folds.split(",")]
    seeds = None if args.seeds is None else {int(x) for x in args.seeds.split(",")}
    found, incomplete, shadowed = scan_fits(fits_roots, args.modes_root, include_modes=not args.no_modes)
    found = [f for f in found if f["panel"] in panels and f["fold"] in folds and (seeds is None or f["seed"] in seeds)
             and (args.arm is None or any(fnmatch.fnmatchcase(f["unit"], p) for p in args.arm))]
    found.sort(key=lambda r: (r["panel"], r["fold"], r["unit"], r["seed"]))
    fit_results, errors = [], []
    t0 = time.monotonic()
    for rec in found:
        rc.drop_except({(rec["panel"], rec["fold"])})       # keep one fold's real data in memory
        try:
            res, status = evaluate_fit(rec, rc, out, log, args)
            if res is None:
                errors.append({**rec, "error": status}); continue
            fit_results.append(res)
            log(f"{status} {rec['unit']}/{rec['panel']}/fold{rec['fold']}/seed{rec['seed']} "
                f"({res.get('score_seconds', 0):.1f}s score)")
        except Exception as e:  # tolerate broken/partial fits
            errors.append({**rec, "error": f"{type(e).__name__}: {e}"})
            log(f"ERROR {rec['gen']}: {e}")
    timings["fits_seconds"] = time.monotonic() - t0
    ens_results = []
    t0 = time.monotonic()
    if not args.no_ensembles:
        sizes = sorted({int(x) for x in args.ensemble_sizes.split(",") if x.strip()})
        groups = {}
        for r in fit_results:
            groups.setdefault((r["unit"], r["panel"], r["fold"]), []).append(r)
        for (unit, panel, fold), rs in sorted(groups.items(), key=lambda x: (x[0][1], x[0][2], x[0][0])):
            if len(rs) < 2:
                continue
            if args.ensemble_arm is not None and not any(fnmatch.fnmatchcase(unit, p) for p in args.ensemble_arm):
                continue
            rc.drop_except({(panel, fold)})
            recs = sorted([next(f for f in found if f["unit"] == unit and f["panel"] == panel and f["fold"] == fold and f["seed"] == r["seed"])
                           for r in rs], key=lambda x: x["seed"])
            for n in sorted({len(recs)} | {s for s in sizes if 2 <= s < len(recs)}):
                try:
                    ens_results += evaluate_ensemble(unit, panel, fold, recs[:n], rc, out, log, args)
                except Exception as ex:
                    errors.append({"unit": unit, "panel": panel, "fold": fold, "ensemble": n, "error": f"{type(ex).__name__}: {ex}"})
                    log(f"ERROR ensemble {unit}/{panel}/fold{fold} n={n}: {ex}")
    timings["ensembles_seconds"] = time.monotonic() - t0
    base_results = {}
    t0 = time.monotonic()
    for panel in panels:
        for fold in folds:
            rc.drop_except({(panel, fold)})
            loaded = rc.get(panel, fold) if (args.no_baselines is False or any(r["panel"] == panel and r["fold"] == fold for r in fit_results)) else None
            if loaded is not None:
                write_real_reference(loaded[0], loaded[3], out)
            if args.no_baselines:
                continue
            try:
                b = evaluate_baselines(panel, fold, rc, out, log, args)
                if b:
                    base_results[(panel, fold)] = b
            except Exception as ex:
                errors.append({"panel": panel, "fold": fold, "baselines": True, "error": f"{type(ex).__name__}: {ex}"})
                log(f"ERROR baselines {panel}/fold{fold}: {ex}")
            rc.drop_except(set())
    timings["baselines_seconds"] = time.monotonic() - t0
    t0 = time.monotonic()
    agg = aggregate(out, fit_results, ens_results, base_results, incomplete, errors, log, shadowed, args)
    timings["aggregate_seconds"] = time.monotonic() - t0
    timings["total_seconds"] = time.monotonic() - t_start
    write_json(out / "timings.json", {**timings, "n_fit_units": len(fit_results), "n_ensembles": len(ens_results),
                                      "n_baseline_folds": len(base_results), "threads": os.environ.get("OMP_NUM_THREADS")})
    log(f"done: {len(fit_results)} fit units, {len(ens_results)} ensembles, {sum(len(v) for v in base_results.values())} baseline records, "
        f"{len(incomplete)} incomplete, {len(shadowed)} shadowed, {len(errors)} errors; {timings}")
    return agg


if __name__ == "__main__":
    main()
