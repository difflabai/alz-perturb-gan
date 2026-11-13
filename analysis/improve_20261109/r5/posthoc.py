#!/usr/bin/env python3
"""improve_20261109 round 2, F5/F7: post-hoc calibration of predictions (no training).

Library + CLI.  Reads finished fits (round-1 ``fits/`` and round-2 ``fits_r2/``,
``generated.npz`` and generation-mode files ``generated_{mode}.npz``), the
fold caches, and the round-1 baselines (regenerated with the frozen round-1
``baselines.py``), and scores honest post-hoc predictors of a held-out donor's

  * pseudobulk log1p(CP10k) profile           -> ``pb_rmse``
  * per-gene mean per-cell log expression     -> ``meanlog_rmse_gene``

with EXACTLY the round-1 metric definitions (``../evaluate.py::score_generated``
/ ``../metrics.py``; ``verify`` reproduces the stored results_screen48 numbers
for raw predictions to 1e-6 before anything else is reported).

Prediction arrays (per unit = fit, seed ensemble or baseline; T targets x G genes)
--------------------------------------------------------------------------------
P[t]  = mean over the M generated donors of log1p(CP10k(generated pseudobulk))
        (bit-for-bit the ``pred`` of score_generated: per-target float32 logs,
        float64 mean).
ML[t] = mean over M of the generated ``mean_log`` (float64), as score_generated.
O[t]  = log1p(CP10k) of the target's observed full pseudobulk, RML[t] = observed
        mean per-cell log1p(CP10k) (``evaluate.load_real``).
pb_rmse            = mean over held-out donors of sqrt(mean_g (P - O)^2)
meanlog_rmse_gene  = sqrt(mean_g (mean over held-out donors of (ML - RML))^2)
A seed ensemble of S fits is the mean of their P (resp. ML): identical, up to
float rounding, to round-1 ``concat_generated`` (pooling the M axes).

Calibration methods (applied identically to P with target O and to ML with target RML;
every quantity below is fitted on the inner fold's TRAIN targets only)
--------------------------------------------------------------------------------
Decomposition of a unit's prediction (train targets tr, stage s(t)):
  level_P = mean_{t in tr} P[t]                                (stage-pooled mean)
  S_s     = mean_{t in tr, s(t)=s} P[t]   (model stage profile; level_P if no train target in s)
  D[t]    = S_{s(t)} - level_P            (model stage deviation)
  R[t]    = P[t] - S_{s(t)}               (within-stage part: depth / MC noise for stage-only
                                           arms, covariate effects for covars arms)
  b       = mean_{t in tr} (O[t] - P[t])  (per-gene additive recalibration)
Methods (P' for every target, train and held-out):
  raw             P
  recal           P + b                                   (F5: per-gene level recalibration)
  pool            level_P + D            (lambda=1, rho=0: stage-pooled prediction, nothing tuned)
  shrink_g        level_P + lam*D              lam in [0,1]       (stage-deviation shrinkage, global)
  shrink_s        level_P + lam_s(t)*D         lam_0..lam_3 in [0,1]   (per stage)
  shrinkr_g/_s    the same plus rho*R with rho in [0,1] also fitted (keeps part of the
                  within-stage / covariate part of the prediction; DIAGNOSTIC: for covars arms the
                  in-sample rho rewards memorised train donors, see below)
  recal_pool, recal_shrink_g, recal_shrink_s, recal_shrinkr_g, recal_shrinkr_s: the same with
                  level = level_P + b (= mean_{tr} O, the train mean of the observed profiles).
lam/rho are the box-constrained least-squares fit of sum_{t in tr} ||O[t] - level - lam D[t]
- rho R[t]||^2 (exact active-set enumeration, ``box_ls``; with recal or without, the fitted
lam/rho are the same because sum_tr D = sum_tr R = 0).  These are 2 or 5 scalars fitted on
25 donors x G genes; the leave-one-train-donor-out (LOO) score below shows their own
overfitting is negligible.  Per-stage lam_s for a stage with no train target is irrelevant
(D = 0 there).  With lam = rho = 0 every recal_* method becomes ``direct:train_mean`` (the
train mean of observed profiles), so a calibrated generator that ties the grand mean by
fitting lam ~ 0 has been replaced by the grand mean: the reports give lam and rho.

Which variant is used / cross-fitting
--------------------------------------
Primary: in-sample calibration on train targets (the model was trained on those donors).
A cross-fitted bias estimate WITHOUT retraining does not exist in this design: every
fold-k model was trained on all fold-k train donors, and the only models that did not see
a given fold-k train donor are the other folds' models, which were trained on fold k's
held-out donors (using their residuals would leak held-out data into fold k's calibration;
rejected).  What is cross-fitted: the calibration itself, by leave-one-train-donor-out
(``train_loo`` = each train donor scored with b / lam / rho / level / S_s refitted without
it).  The model's in-sample optimism is not removable without retraining; it is MEASURED
(evaluation only) by the train (in-sample) vs held-out scores, and by
``bias_transfer`` = Pearson over genes between the train-fitted b and the held-out
residual mean, both reported per unit.

Seed ensembles (F7(3))
----------------------
For a (unit, panel, fold) with n seeds, every subset of size S = 1..n (all C(n,S) subsets,
deterministically subsampled to at most --max-subsets) is averaged, calibrated (calibration
refitted on the ensemble's train predictions) and scored; metrics are averaged over subsets.
1/S curve: per held-out donor, subset-mean MSE_t(S) = a_t + b_t/S by least squares over
S = 1..n (for raw predictions this form is exact: a_t = ||mean_n - O_t||^2/G - s2_t/(n-1),
b_t = n s2_t/(n-1), s2_t the across-seed variance; checked in the tests), projected
pb_rmse(inf) = mean_t sqrt(max(a_t, 0)); meanlog: rmse_gene(S)^2 = a + b/S -> sqrt(a).
Generation used common random numbers (same per target across seeds), so the part of the
Monte-Carlo noise shared by seeds is not removed by S -> inf (about 2% of the error).

Baselines (F7(5), like-for-like)
--------------------------------
``baseline:<name>``: the six round-1 baselines regenerated with ../baselines.py (deterministic;
held-out predictions and metrics are verified against results_screen48), then the same
methods.  ``direct:*``: count-noise-free profile predictors without any generation:
``direct:train_mean`` (mean of train observed profiles; recal is EXACTLY a no-op: b = 0),
``direct:stage_mean`` (train stage means of observed profiles), ``direct:shrunk_stage_mean``
and ``direct:ridge`` (the profile mu of baselines.fit_profile_baseline; P only, no meanlog).
Recalibrating the round-1 ``grand_mean`` baseline is NOT a no-op: its prediction is a mean of
generated 256-cell pseudobulks, whose logs differ from full-depth donor logs; recal moves it
to (almost) direct:train_mean.  All baselines are fitted on train donors only.

Error decomposition and floors (held-out, MSE over donors x genes)
-------------------------------------------------------------------
For a calibrated prediction with held-out error d_t = P'_t - O_t:
  MSE = shared + stage + donor,  shared = mean_g (mean_t d)^2,
  stage = mean_t mean_g (mean_{s(t)} d - mean_t d)^2, donor = mean_t mean_g (d_t - mean_{s(t)} d)^2.
For any predictor constant within stage, donor = floor_insample = mean_t mean_g (O_t - Obar_{s(t)})^2
(held-out stage means): the donor-specific error that NO stage-level predictor can remove.
``floors`` also gives the dof-corrected pooled within-stage variance (held-out and train) and the
pb_rmse of two held-out ORACLES (evaluation references, never used for fitting): the held-out
stage-mean oracle (in-sample, optimistic) and the held-out grand-mean oracle.

Disclosure: the first version fitted rho inside recal_shrink_g/_s.  After the fold-0 hvg2000
results of all arms had been printed (covars: in-sample rho 0.4-0.55 rewarded memorised train
donors and hurt held-out error), rho was moved into the separate shrinkr variants, so that
shrink_* is the pure stage-deviation shrinkage the task specifies.  Both are reported.

Verdict rule (fixed before looking at results): a calibrated generator "ties" the grand mean
if its fold-balanced held-out pb_rmse is at most 0.005 above ``baseline:grand_mean`` raw
(about 1%), "beats" it if below; reported both against the pre-specified primary
(genemean, all available seeds, recal_shrink_g) and as the best over all generator rows
(best-of-many on held-out = selection-optimistic; flagged).

Repair 2026-11-10 (reviews S5, S6, S3, S4; version posthoc-r2-v2)
-----------------------------------------------------------------
* S5, like-for-like verdict.  The round-1 rule compared a calibrated generator
  with the RAW baseline:grand_mean (a mean of simulated 256-cell pseudobulks,
  which carries a ~0.002 Jensen penalty).  The verdict now also compares every
  generator row with its LIKE-FOR-LIKE reference = min(baseline:grand_mean with
  the SAME method, direct:train_mean raw), over the folds both cover.  Keys:
  ``*_like_for_like`` (primary claim) and ``*_raw_round1_grand_mean`` (the old
  booleans, renamed so they cannot be read as a like-for-like win).
* S6, seed curve.  ``verdict[panel]['seed_curve']``: recal_shrink_g pb_rmse and
  the fitted lambda for S = 1..n per generator unit with >= 3 seeds, next to
  direct:train_mean, with flags ``increases_with_S`` and
  ``above_train_mean_at_every_S``.  The calibrated generators get WORSE as seeds
  are added (lambda grows as seed noise averages out and lets more of the
  generator's wrong stage deviation through): a calibrated single seed is best
  only because it is closest to the train mean.  Seed ensembles here pool S*32
  generated donors (not M-matched); Monte Carlo noise is ~2% of the error.
* The like-for-like best row excludes stage-blind units; their best row is
  reported under ``*_stage_blind_*`` (on round-1 hvg2000 it is genemean#condmix@1
  recal_shrink_g S=2 at 0.46183 vs direct:train_mean 0.46184: a numerical tie
  with the train mean).  ``*_by_more_than_tol`` = below the reference by more
  than the tie tolerance.
* S4, stage-blind units: mode units condmix@lam / condunk@lam with
  floor(32*lam+.5) = 32, condshrink@1 and anchorgain@0 are tagged
  ``stage_mode = 'stage-blind'`` in rows: a generator-made grand-mean
  predictor, a calibration result, not a stage fix.
* S3, selection history: see r2/evaluate.py docstring (repair section).  The
  primary (genemean, recal_shrink_g) was pre-specified, but genemean itself
  was picked from round-1 held-out results on the same 37 donors.
* ``posthoc.py reverdict --out DIR`` recomputes rows, verdict and summary.txt
  from an existing results.json (no data are read).

CLI
---
  posthoc.py reverdict --out DIR                         # rows/verdict/summary from results.json
  posthoc.py baselines --panel hvg2000 --fold 0          # precompute one baseline cache
  posthoc.py run [--fits-root [TAG=]DIR ...] [--gen-file NPZ ...] [--panels ..] [--folds ..]
                 [--modes all|default|m1,m2] [--no-baselines] [--screen48 DIR|none]
                 [--out runs/improve_20261109/posthoc_r2/calib]
--fits-root may be repeated (default: round-1 fits/; add fits_r2/ for round 2; TAG= prefixes
the unit label).  Units: ``arm`` for generated.npz, ``arm#mode`` for generated_{mode}.npz.
--gen-file takes a mode file written with generate.py --out: its *.receipt.json names the
fit, whose config.json gives arm, panel, fold, seed.  Caches: caches/{panel}/fold{k} for any
fold k (0-8).  Outputs (small): results.json, rows.csv, summary.txt, baseline_preds/ (float32
P/ML caches of the regenerated baselines, keyed by baseline version and cache manifest).
Threads: 3 (OMP/OPENBLAS/MKL), CPU only.  Reads only development-donor caches
(evaluate.load_real refuses others).
"""
from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

R2 = Path(__file__).resolve().parent
ROUND1 = R2.parent


def _import_round1():
    """Import the FROZEN round-1 metrics/evaluate/baselines (read-only), never an r2 module of
    the same name (r2/evaluate.py exists): r2/ is removed from sys.path during the import and the
    resolved files are checked."""
    names = ("metrics", "evaluate", "baselines")
    for n in names:
        m = sys.modules.get(n)
        if m is not None and Path(m.__file__).resolve().parent != ROUND1:
            raise ImportError(f"module {n} already imported from {m.__file__}, not round-1 {ROUND1}")
    saved = list(sys.path)
    sys.path[:] = [str(ROUND1)] + [p for p in saved if Path(p or ".").resolve() not in (R2, ROUND1)]
    try:
        import importlib
        mods = [importlib.import_module(n) for n in names]
    finally:
        sys.path[:] = saved
    for m in mods:
        if Path(m.__file__).resolve().parent != ROUND1:
            raise ImportError(f"{m.__name__} resolved to {m.__file__}")
    return mods


MX, E, BL = _import_round1()

VERSION = "improve_20261109-posthoc-r2-v2"   # v2: repair 2026-11-10 (like-for-like verdict, seed curve)
RUNS = Path("/ml2/alz/runs/improve_20261109")
DEFAULT_OUT = RUNS / "posthoc_r2" / "calib"
METHODS = {   # name: (recal, shrink, fit rho)
    "raw": (False, None, False), "recal": (True, None, False),
    "pool": (False, "pool", False), "shrink_g": (False, "global", False), "shrink_s": (False, "stage", False),
    "shrinkr_g": (False, "global", True), "shrinkr_s": (False, "stage", True),
    "recal_pool": (True, "pool", False), "recal_shrink_g": (True, "global", False),
    "recal_shrink_s": (True, "stage", False), "recal_shrinkr_g": (True, "global", True),
    "recal_shrinkr_s": (True, "stage", True),
}
TIE_TOL = 0.005
PRIMARY = ("genemean", "recal_shrink_g")
BASELINE_NAMES = list(BL.BASELINES)
DIRECT_NAMES = ["train_mean", "stage_mean", "shrunk_stage_mean", "ridge"]


# ==========================================================================
# Numerics
# ==========================================================================
def box_ls(H, g, lo=0.0, hi=1.0, tol=1e-12):
    """argmin_x x'Hx - 2g'x subject to lo <= x <= hi (H symmetric PSD, p <= ~6).

    Exact: tries the unconstrained solution, else enumerates every assignment of each
    coordinate to {free, lo, hi} and keeps the best feasible stationary point.  Coordinates
    with a (numerically) zero column (H_jj ~ 0) are fixed at lo."""
    H = np.asarray(H, float); g = np.asarray(g, float); p = len(g)
    scale = max(float(np.max(np.abs(np.diag(H)))) if p else 0.0, 1e-300)
    dead = np.diag(H) <= tol * scale
    live = np.where(~dead)[0]
    x = np.full(p, lo, float)
    if not len(live):
        return x

    def obj(v):
        return float(v @ H @ v - 2 * g @ v)
    Hl, gl = H[np.ix_(live, live)], g[live]
    try:
        u = np.linalg.solve(Hl, gl)
        if np.all(u >= lo - 1e-12) and np.all(u <= hi + 1e-12):
            x[live] = np.clip(u, lo, hi)
            return x
    except np.linalg.LinAlgError:
        pass
    best, bx = np.inf, None
    for assign in itertools.product((0, 1, 2), repeat=len(live)):
        v = np.full(p, lo, float)
        free = [live[i] for i, a in enumerate(assign) if a == 0]
        for i, a in enumerate(assign):
            if a == 2:
                v[live[i]] = hi
        if free:
            fixed = [j for j in range(p) if j not in free]
            rhs = g[free] - H[np.ix_(free, fixed)] @ v[fixed]
            sol = np.linalg.lstsq(H[np.ix_(free, free)], rhs, rcond=None)[0]
            if np.any(sol < lo - 1e-10) or np.any(sol > hi + 1e-10):
                continue
            v[free] = np.clip(sol, lo, hi)
        f = obj(v)
        if f < best - 1e-15:
            best, bx = f, v
    return bx if bx is not None else x


def calibrate(P, O, train, stage, method):
    """Calibrated prediction for EVERY target from train targets only.

    P [T,G] prediction, O [T,G] observed target values (only O[train] is read),
    train bool [T], stage int [T].  Returns (P' [T,G], params dict)."""
    recal, shrink, fit_rho = METHODS[method]
    if not recal and shrink is None:
        return P.copy(), {}
    tr = np.where(train)[0]
    levelP = P[tr].mean(0)
    b = (O[tr] - P[tr]).mean(0) if recal else np.zeros(P.shape[1])
    params = {"b_rms": float(np.sqrt(np.mean(b ** 2)))} if recal else {}
    if shrink is None:
        return P + b, params
    level = levelP + b
    st_tr = stage[tr]
    Sm = {int(s): P[tr[st_tr == s]].mean(0) for s in np.unique(st_tr)}
    D = np.stack([Sm[int(s)] - levelP if int(s) in Sm else np.zeros_like(levelP) for s in stage])
    R = P - levelP - D
    stages = sorted(Sm)
    if shrink == "pool":
        lam = {s: 1.0 for s in range(4)}; rho = 0.0
    else:
        y = O[tr] - level
        if shrink == "global":
            cols = [D[tr]]
        else:
            cols = [D[tr] * (st_tr == s)[:, None] for s in stages]
        if fit_rho:
            cols.append(R[tr])
        p = len(cols)
        H = np.empty((p, p)); gv = np.empty(p)
        for i in range(p):
            gv[i] = float(np.sum(cols[i] * y))
            for j in range(i, p):
                H[i, j] = H[j, i] = float(np.sum(cols[i] * cols[j]))
        th = box_ls(H, gv)
        rho = float(th[-1]) if fit_rho else 0.0
        if shrink == "global":
            lam = {s: float(th[0]) for s in range(4)}
        else:
            lam = {s: 0.0 for s in range(4)}
            lam.update({s: float(th[i]) for i, s in enumerate(stages)})
    lam_t = np.array([lam[int(s)] for s in stage])[:, None]
    out = level + lam_t * D + rho * R
    params.update(lam={str(s): lam[s] for s in stages}, rho=rho)
    dv = np.mean(D[tr] ** 2); rv = np.mean(R[tr] ** 2)
    params.update(model_stage_dev_ms=float(dv), model_within_ms=float(rv))
    return out, params


def pb_rmse(P, O, mask):
    d = P[mask] - O[mask]
    return float(np.sqrt((d ** 2).mean(1)).mean())


def per_donor_mse(P, O, mask):
    d = P[mask] - O[mask]
    return (d ** 2).mean(1)


def meanlog_rmse_gene(ML, RML, mask):
    e = (ML[mask] - RML[mask]).mean(0)
    return float(np.sqrt(np.mean(e ** 2)))


def decompose(P, O, stage, mask):
    """Held-out MSE = shared + stage + donor (see module docstring)."""
    d = P[mask] - O[mask]; st = stage[mask]
    eb = d.mean(0)
    es = {s: d[st == s].mean(0) for s in np.unique(st)}
    stage_part = np.stack([es[s] - eb for s in st])
    donor_part = d - np.stack([es[s] for s in st])
    return {"mse": float((d ** 2).mean()), "shared": float((eb ** 2).mean()),
            "stage": float((stage_part ** 2).mean()), "donor": float((donor_part ** 2).mean())}


def loo_train_scores(P, O, train, stage, method, kind):
    """Leave-one-train-donor-out: calibration refitted without donor i, scored on i."""
    tr = np.where(train)[0]
    preds = np.empty((len(tr), P.shape[1]))
    for j, i in enumerate(tr):
        t2 = train.copy(); t2[i] = False
        preds[j] = calibrate(P, O, t2, stage, method)[0][i]
    if kind == "pb":
        return float(np.sqrt(((preds - O[tr]) ** 2).mean(1)).mean())
    return float(np.sqrt(np.mean((preds - O[tr]).mean(0) ** 2)))


def score_unit(P, ML, O, RML, train, stage, method, loo=False):
    """All metrics of one calibrated prediction (P and/or ML may be None)."""
    held = ~train
    out = {"method": method}
    if P is not None:
        Pc, prm = calibrate(P, O, train, stage, method)
        out.update(pb_rmse=pb_rmse(Pc, O, held), pb_rmse_train=pb_rmse(Pc, O, train),
                   pb_mse_donor=per_donor_mse(Pc, O, held), decomp=decompose(Pc, O, stage, held), pb_params=prm)
        if METHODS[method][0]:
            bh = (O[held] - P[held]).mean(0); bt = (O[train] - P[train]).mean(0)
            out["bias_transfer"] = MX.pearson(bt, bh)
        if loo:
            out["pb_rmse_train_loo"] = loo_train_scores(P, O, train, stage, method, "pb")
    if ML is not None:
        MLc, mprm = calibrate(ML, RML, train, stage, method)
        out.update(meanlog_rmse_gene=meanlog_rmse_gene(MLc, RML, held),
                   meanlog_rmse_gene_train=meanlog_rmse_gene(MLc, RML, train), ml_params=mprm)
        if loo:
            out["meanlog_rmse_gene_train_loo"] = loo_train_scores(ML, RML, train, stage, method, "ml")
    return out


def fit_inv_s(Ss, vals):
    """Least-squares a + b/S over the given S values; returns (a, b)."""
    Ss = np.asarray(Ss, float); vals = np.asarray(vals, float)
    if len(Ss) < 2:
        return float(vals[0]), 0.0
    A = np.column_stack([np.ones_like(Ss), 1.0 / Ss])
    sol = np.linalg.lstsq(A, vals, rcond=None)[0]
    return float(sol[0]), float(sol[1])


def subsets(n, S, cap, key):
    combos = list(itertools.combinations(range(n), S))
    if len(combos) <= cap:
        return combos
    rng = np.random.default_rng(MX.stable_seed("posthoc-subsets", key, n, S))
    pick = np.sort(rng.choice(len(combos), cap, replace=False))
    return [combos[i] for i in pick]


def ensemble_curve(Ps, MLs, O, RML, train, stage, methods, cap=35, key="", loo=True):
    """Ps / MLs: list over seeds of [T,G] (MLs entries may be None).  Returns
    {S: {method: averaged metrics}} plus 1/S fits."""
    n = len(Ps)
    out = {"n_seeds": n, "S": {}}
    for S in range(1, n + 1):
        combos = subsets(n, S, cap, key)
        acc = {m: [] for m in methods}
        for c in combos:
            P = np.mean([Ps[i] for i in c], axis=0) if Ps[0] is not None else None
            ML = np.mean([MLs[i] for i in c], axis=0) if MLs[0] is not None else None
            do_loo = loo and (S == 1 or S == n)
            for m in methods:
                acc[m].append(score_unit(P, ML, O, RML, train, stage, m, loo=do_loo))
        out["S"][S] = {m: average_scores(v) for m, v in acc.items()}
        out["S"][S]["_n_subsets"] = len(combos)
    if n >= 2:
        fits = {}
        Ss = list(range(1, n + 1))
        for m in methods:
            f = {}
            if "pb_mse_donor" in out["S"][1][m]:
                mse = np.stack([out["S"][S][m]["pb_mse_donor"] for S in Ss])     # [nS, n_held]
                ab = [fit_inv_s(Ss, mse[:, j]) for j in range(mse.shape[1])]
                f["pb_rmse_inf"] = float(np.mean([np.sqrt(max(a, 0.0)) for a, _ in ab]))
                f["pb_mse_inf"] = float(np.mean([a for a, _ in ab]))
                f["pb_mse_seed_term"] = float(np.mean([b for _, b in ab]))
            if "meanlog_rmse_gene" in out["S"][1][m]:
                a, b = fit_inv_s(Ss, [out["S"][S][m]["meanlog_rmse_gene"] ** 2 for S in Ss])
                f["meanlog_rmse_gene_inf"] = float(np.sqrt(max(a, 0.0)))
                f["meanlog_sq_seed_term"] = b
            fits[m] = f
        out["inv_s_fit"] = fits
    return out


def average_scores(lst):
    """Mean over subsets of every numeric field (nested dicts averaged key-wise)."""
    out = {}
    keys = set().union(*[set(d) for d in lst])
    for k in keys:
        vals = [d[k] for d in lst if k in d and d[k] is not None]
        if not vals:
            out[k] = None
        elif isinstance(vals[0], dict):
            out[k] = average_scores(vals) if all(isinstance(v, dict) for v in vals) else vals[0]
        elif isinstance(vals[0], np.ndarray):
            out[k] = np.mean(vals, axis=0)
        elif isinstance(vals[0], (int, float, np.floating)) and not isinstance(vals[0], bool):
            out[k] = float(np.mean(vals))
        else:
            out[k] = vals[0]
    return out


# ==========================================================================
# Real data, predictions and floors
# ==========================================================================
class Fold:
    def __init__(self, real):
        self.real = real
        self.O = MX.log_cp10k(real.pb)
        self.RML = real.mean_log.astype(np.float64)
        self.train = ~real.is_heldout
        self.stage = real.stage.astype(int)


def predictions_from_gen(gen):
    """Aligned generated dict -> (P, ML) exactly as evaluate.score_generated."""
    pb = gen["pb_counts"].astype(np.float64)
    T, M, G = pb.shape
    gl = np.empty((T, M, G), dtype=np.float32)
    for t in range(T):
        gl[t] = MX.log_cp10k(pb[t])
    P = gl.astype(np.float64).mean(1)
    ML = gen["mean_log"].astype(np.float64).mean(1)
    return P, ML


def floors(f: Fold):
    O, st, held, tr = f.O, f.stage, ~f.train, f.train

    def within(mask):
        Om, s = O[mask], st[mask]
        ss = np.zeros(O.shape[1]); dof = 0; ins = []
        for v in np.unique(s):
            x = Om[s == v]
            ss += ((x - x.mean(0)) ** 2).sum(0); dof += len(x) - 1
        return ss, dof
    ssh, dofh = within(held); sst, doft = within(tr)
    Oh, sh = O[held], st[held]
    sm = {v: Oh[sh == v].mean(0) for v in np.unique(sh)}
    dev = Oh - np.stack([sm[v] for v in sh])
    out = {"n_heldout": int(held.sum()), "heldout_n_by_stage": {str(v): int((sh == v).sum()) for v in range(4)},
           "floor_insample_mse": float((dev ** 2).mean()),
           "oracle_heldout_stage_mean_pb_rmse": float(np.sqrt((dev ** 2).mean(1)).mean()),
           "oracle_heldout_grand_mean_pb_rmse": float(np.sqrt(((Oh - Oh.mean(0)) ** 2).mean(1)).mean()),
           "within_stage_var_heldout_dofcorr": float(ssh.mean() / dofh) if dofh > 0 else None,
           "within_stage_var_train_dofcorr": float(sst.mean() / doft) if doft > 0 else None}
    # meanlog: expected rmse_gene^2 of a population-stage-mean oracle ~ within-stage var / n_held
    R = f.RML[held]
    ssm = np.zeros(R.shape[1]); dofm = 0
    for v in np.unique(sh):
        x = R[sh == v]; ssm += ((x - x.mean(0)) ** 2).sum(0); dofm += len(x) - 1
    out["meanlog_oracle_rmse_gene_expected"] = float(np.sqrt(ssm.mean() / dofm / held.sum())) if dofm > 0 else None
    return out


# ==========================================================================
# Discovery
# ==========================================================================
def parse_root(text):
    if "=" in text and not text.startswith("/"):
        tag, path = text.split("=", 1)
        return tag, Path(path)
    return "", Path(text)


def scan(roots, modes="all", gen_files=()):
    """-> list of records {unit, arm, mode, panel, fold, seed, path, root}."""
    recs, skipped = [], []
    for tag, root in roots:
        if not root.exists():
            skipped.append({"root": str(root), "why": "missing"}); continue
        for gp in sorted(root.glob("*/*/fold*/seed*")):
            if not gp.is_dir():
                continue
            arm, panel, fold, seed = gp.parts[-4:]
            try:
                k = int(fold.removeprefix("fold")); s = int(seed.removeprefix("seed"))
            except ValueError:
                continue
            for npz in sorted(gp.glob("generated*.npz")):
                if npz.name.startswith("."):
                    continue
                if npz.name == "generated.npz":
                    mode = None
                    if not (gp / "fit_receipt.json").exists():
                        skipped.append({"path": str(npz), "why": "no fit_receipt.json"}); continue
                else:
                    mode = npz.name[len("generated_"):-len(".npz")]
                if modes == "default" and mode is not None:
                    continue
                if modes not in ("all", "default") and mode is not None and mode not in modes.split(","):
                    continue
                unit = (f"{tag}:" if tag else "") + arm + (f"#{mode}" if mode else "")
                recs.append({"unit": unit, "arm": arm, "mode": mode, "panel": panel, "fold": k, "seed": s,
                             "path": str(npz), "root": str(root)})
    for p in gen_files:
        p = Path(p)
        rc = json.loads(p.with_name(p.stem + ".receipt.json").read_text())
        cfg = json.loads((Path(rc["fit"]) / "config.json").read_text())
        mode = rc.get("mode")
        recs.append({"unit": cfg["arm"] + (f"#{mode}" if mode else "#file"), "arm": cfg["arm"], "mode": mode,
                     "panel": cfg["panel"], "fold": int(cfg["fold"]), "seed": int(cfg["seed"]),
                     "path": str(p), "root": "gen-file"})
    seen = set()
    for r in recs:
        key = (r["unit"], r["panel"], r["fold"], r["seed"])
        if key in seen:
            raise ValueError(f"duplicate unit {key}; give the roots distinct TAG= prefixes")
        seen.add(key)
    return recs, skipped


# ==========================================================================
# Baselines (regenerated with the frozen round-1 code, cached as float32)
# ==========================================================================
def baseline_cache_path(out, panel, fold, name):
    return Path(out) / "baseline_preds" / panel / f"fold{fold}" / f"{name}.npz"


def baseline_predictions(real, name, out, screen48=None, log=print):
    """(P, ML, info) of a round-1 baseline; regenerated once and cached."""
    path = baseline_cache_path(out, real.panel, real.fold, name)
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            if str(z["baseline_version"]) == BL.BASELINE_VERSION and str(z["manifest"]) == str(real.manifest_sha256):
                return z["P"].astype(np.float64), z["ML"].astype(np.float64), json.loads(str(z["info"]))
    t0 = time.monotonic()
    gen, info = BL.generate(name, real)
    gen = E.align_generated(gen, real)
    P, ML = predictions_from_gen(gen)
    ver = verify_baseline(real, name, P, ML, screen48)
    info = {"seconds": time.monotonic() - t0, "verification": ver}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".partial.npz")
    np.savez_compressed(tmp, P=P.astype(np.float32), ML=ML.astype(np.float32), baseline_version=BL.BASELINE_VERSION,
                        manifest=str(real.manifest_sha256), info=json.dumps(info))
    tmp.replace(path)
    log(f"baseline {name} {real.panel}/fold{real.fold} {info['seconds']:.0f}s verify={ver}")
    # return the cached float32 values so a first run and a rerun give identical numbers
    return P.astype(np.float32).astype(np.float64), ML.astype(np.float32).astype(np.float64), info


def verify_baseline(real, name, P, ML, screen48):
    if screen48 is None:
        return None
    j = Path(screen48) / "baselines" / real.panel / f"fold{real.fold}" / f"{name}.json"
    if not j.exists():
        return {"available": False}
    ref = json.loads(j.read_text())["subsets"]["all"]
    f = Fold(real); held = ~f.train
    out = {"available": True,
           "pb_rmse_heldout_absdiff": abs(pb_rmse(P, f.O, held) - ref["heldout"]["pseudobulk"]["rmse"]),
           "pb_rmse_train_absdiff": abs(pb_rmse(P, f.O, f.train) - ref["train"]["pseudobulk"]["rmse"]),
           "meanlog_heldout_absdiff": abs(meanlog_rmse_gene(ML, f.RML, held) - ref["heldout"]["mean_log"]["rmse_gene"])}
    z = j.with_suffix(".npz")
    if z.exists():
        with np.load(z, allow_pickle=False) as q:
            out["pred_heldout_maxabsdiff"] = float(np.max(np.abs(q["pred_profile_heldout"].astype(np.float64) - P[held])))
    out["ok"] = all(v <= 1e-6 for k, v in out.items() if k.endswith("absdiff")) and out.get("pred_heldout_maxabsdiff", 0) <= 1e-5
    return out


def direct_predictions(real, f: Fold, name):
    tr, st = f.train, f.stage
    if name == "train_mean":
        return np.tile(f.O[tr].mean(0), (len(st), 1)), np.tile(f.RML[tr].mean(0), (len(st), 1))
    if name == "stage_mean":
        def sm(X):
            g = X[tr].mean(0)
            m = {s: X[tr & (st == s)].mean(0) for s in range(4) if (tr & (st == s)).any()}
            return np.stack([m.get(int(s), g) for s in st])
        return sm(f.O), sm(f.RML)
    mu, _, _ = BL.fit_profile_baseline(name, real)
    return mu, None


# ==========================================================================
# Verification of raw predictions against results_screen48
# ==========================================================================
def verify_fit(rec, f: Fold, P, ML, screen48):
    if screen48 is None or rec["mode"] is not None or rec["root"] != str(RUNS / "fits"):
        return None
    j = Path(screen48) / "fits" / rec["arm"] / rec["panel"] / f"fold{rec['fold']}" / f"seed{rec['seed']}.json"
    if not j.exists():
        return None
    ref = json.loads(j.read_text())
    if ref.get("generated_sha256") is not None and ref.get("generated_file", {}).get("size") != Path(rec["path"]).stat().st_size:
        return {"ok": False, "why": "generated.npz changed since results_screen48"}
    ref = ref["subsets"]["all"]
    held = ~f.train
    d = {"pb_rmse_heldout": abs(pb_rmse(P, f.O, held) - ref["heldout"]["pseudobulk"]["rmse"]),
         "pb_rmse_train": abs(pb_rmse(P, f.O, f.train) - ref["train"]["pseudobulk"]["rmse"]),
         "meanlog_heldout": abs(meanlog_rmse_gene(ML, f.RML, held) - ref["heldout"]["mean_log"]["rmse_gene"]),
         "meanlog_train": abs(meanlog_rmse_gene(ML, f.RML, f.train) - ref["train"]["mean_log"]["rmse_gene"])}
    return {"ok": max(d.values()) <= 1e-6, "max_absdiff": max(d.values()), **d}


def verify_ensemble(arm, panel, fold, seeds, Ps, MLs, f: Fold, screen48):
    if screen48 is None:
        return None
    j = Path(screen48) / "ensembles" / arm / panel / f"fold{fold}.json"
    if not j.exists():
        return None
    ref = json.loads(j.read_text())
    want = ref["seeds"]
    if not all(s in seeds for s in want):
        return None
    idx = [seeds.index(s) for s in want]
    P = np.mean([Ps[i] for i in idx], axis=0); ML = np.mean([MLs[i] for i in idx], axis=0)
    r = ref["subsets"]["all"]; held = ~f.train
    d = {"pb_rmse_heldout": abs(pb_rmse(P, f.O, held) - r["heldout"]["pseudobulk"]["rmse"]),
         "meanlog_heldout": abs(meanlog_rmse_gene(ML, f.RML, held) - r["heldout"]["mean_log"]["rmse_gene"])}
    return {"ok": max(d.values()) <= 1e-6, "seeds": want, "max_absdiff": max(d.values()), **d}


# ==========================================================================
# Orchestration
# ==========================================================================
def _clean(v):
    return E._clean(v)


def run(args, log=print):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    roots = [parse_root(r) for r in (args.fits_root or [str(RUNS / "fits")])]
    panels = args.panels.split(",")
    folds = [int(x) for x in args.folds.split(",")]
    screen48 = None if args.screen48 == "none" else Path(args.screen48)
    recs, skipped = scan(roots, args.modes, args.gen_file or [])
    recs = [r for r in recs if r["panel"] in panels and r["fold"] in folds]
    methods = list(METHODS)
    res = {"version": VERSION, "baseline_version": BL.BASELINE_VERSION, "evaluator_version": E.EVALUATOR_VERSION,
           "roots": [f"{t}={p}" if t else str(p) for t, p in roots], "skipped": skipped, "methods": methods,
           "tie_tolerance": TIE_TOL, "units": {}, "floors": {}, "verification": {"fits": {}, "ensembles": {}, "baselines": {}},
           "inputs": []}
    caches_root = Path(args.caches_root)
    for panel in panels:
        res["units"][panel] = {}; res["floors"][panel] = {}
        for fold in folds:
            cdir = caches_root / panel / f"fold{fold}"
            prs = [r for r in recs if r["panel"] == panel and r["fold"] == fold]
            if not (cdir / "space.npz").exists():
                if prs:
                    log(f"cache missing {cdir}; skipping {len(prs)} fits")
                continue
            if not prs and args.no_baselines:
                continue
            real = E.load_real(cdir, panel, fold)
            f = Fold(real)
            res["floors"][panel][fold] = floors(f)
            units = {}
            for r in prs:
                units.setdefault(r["unit"], []).append(r)
            for unit, rs in sorted(units.items()):
                rs = sorted(rs, key=lambda r: r["seed"])
                Ps, MLs, seeds = [], [], []
                for r in rs:
                    gen = E.align_generated(E.load_generated(r["path"]), real)
                    P, ML = predictions_from_gen(gen)
                    del gen
                    Ps.append(P); MLs.append(ML); seeds.append(r["seed"])
                    v = verify_fit(r, f, P, ML, screen48)
                    if v is not None:
                        res["verification"]["fits"][f"{unit}|{panel}|{fold}|{r['seed']}"] = v
                    st = Path(r["path"]).stat()
                    res["inputs"].append({**r, "size": st.st_size, "mtime_ns": st.st_mtime_ns})
                if "#" not in unit and ":" not in unit:
                    v = verify_ensemble(unit, panel, fold, seeds, Ps, MLs, f, screen48)
                    if v is not None:
                        res["verification"]["ensembles"][f"{unit}|{panel}|{fold}"] = v
                t0 = time.monotonic()
                cur = ensemble_curve(Ps, MLs, f.O, f.RML, f.train, f.stage, methods, cap=args.max_subsets,
                                     key=f"{unit}|{panel}|{fold}")
                cur.update(kind="generator", seeds=seeds)
                res["units"][panel].setdefault(unit, {})[fold] = cur
                log(f"{unit} {panel}/fold{fold} seeds={seeds} {time.monotonic() - t0:.0f}s "
                    f"raw={cur['S'][1]['raw']['pb_rmse']:.4f} recal_shrink_g={cur['S'][len(seeds)]['recal_shrink_g']['pb_rmse']:.4f}"
                    f"@S={len(seeds)}")
            if args.no_baselines:
                continue
            for name in BASELINE_NAMES:
                P, ML, info = baseline_predictions(real, name, out, screen48, log)
                res["verification"]["baselines"][f"{name}|{panel}|{fold}"] = info.get("verification")
                cur = ensemble_curve([P], [ML], f.O, f.RML, f.train, f.stage, methods, key=name)
                cur.update(kind="baseline")
                res["units"][panel].setdefault(f"baseline:{name}", {})[fold] = cur
            for name in DIRECT_NAMES:
                P, ML = direct_predictions(real, f, name)
                cur = ensemble_curve([P], [ML], f.O, f.RML, f.train, f.stage, methods, key=name)
                cur.update(kind="direct")
                res["units"][panel].setdefault(f"direct:{name}", {})[fold] = cur
    res["rows"] = make_rows(res)
    res["verdict"] = make_verdict(res)
    E.write_json(out / "results.json", strip_arrays(res))
    write_rows_csv(out / "rows.csv", res["rows"])
    (out / "summary.txt").write_text(summary_text(res))
    log(f"wrote {out}/results.json, rows.csv, summary.txt")
    return res


def _intkeys(x, depth=0):
    """results.json -> in-memory layout (fold and S keys back to int)."""
    if isinstance(x, dict):
        return {(int(k) if isinstance(k, str) and k.isdigit() else k): _intkeys(v, depth + 1) for k, v in x.items()}
    if isinstance(x, list):
        return [_intkeys(v, depth + 1) for v in x]
    return x


def reverdict(out):
    """Repair: rows, verdict and summary.txt recomputed from results.json (keeps the previous verdict as
    'verdict_before_repair' the first time)."""
    out = Path(out)
    res = json.loads((out / "results.json").read_text())
    old = res.get("verdict")
    res = _intkeys(res)
    res["methods"] = list(METHODS) if not res.get("methods") else res["methods"]
    res["rows"] = make_rows(res)
    res["verdict"] = make_verdict(res)
    if "verdict_before_repair" not in res:
        res["verdict_before_repair"] = old
    res["reverdict_version"] = VERSION
    E.write_json(out / "results.json", strip_arrays(res))
    write_rows_csv(out / "rows.csv", res["rows"])
    (out / "summary.txt").write_text(summary_text(res))
    return res


def strip_arrays(x):
    if isinstance(x, dict):
        return {str(k): strip_arrays(v) for k, v in x.items() if k != "pb_mse_donor"}
    if isinstance(x, list):
        return [strip_arrays(v) for v in x]
    return x


# ==========================================================================
# Aggregation: per fold and fold-balanced rows
# ==========================================================================
ROW_METRICS = ["pb_rmse", "meanlog_rmse_gene", "pb_rmse_train", "pb_rmse_train_loo", "meanlog_rmse_gene_train",
               "meanlog_rmse_gene_train_loo", "bias_transfer"]


def stage_mode_of(unit, M=32):
    """Repair S4: 'stage-blind' / 'stage-mixture' / None for a unit label (same rule as
    r2/evaluate.stage_blind_mode; copied because this module imports round-1 evaluate)."""
    if "#" not in unit:
        return None
    mode = unit.split("#", 1)[1]
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


def make_rows(res):
    rows = []
    for panel, units in res["units"].items():
        for unit, byfold in units.items():
            fl = sorted(byfold)
            nmin = min(byfold[k]["n_seeds"] for k in fl)
            nmax = max(byfold[k]["n_seeds"] for k in fl)
            for S in sorted(set(range(1, nmin + 1)) | ({"all"} if nmax > nmin else set()), key=str):
                for m in res["methods"]:
                    row = {"panel": panel, "unit": unit, "kind": byfold[fl[0]]["kind"], "method": m, "S": S, "folds": fl}
                    if stage_mode_of(unit):
                        row["stage_mode"] = stage_mode_of(unit)
                    for metric in ROW_METRICS + ["mse", "shared", "stage", "donor"]:
                        vals = []
                        for k in fl:
                            Sk = byfold[k]["n_seeds"] if S == "all" else S
                            blk = byfold[k]["S"][Sk][m]
                            v = blk.get("decomp", {}).get(metric) if metric in ("mse", "shared", "stage", "donor") else blk.get(metric)
                            vals.append(v)
                            row[f"{metric}_f{k}"] = v
                        ok = [v for v in vals if v is not None]
                        row[metric] = float(np.mean(ok)) if len(ok) == len(vals) and ok else None
                    lam = []
                    for k in fl:
                        Sk = byfold[k]["n_seeds"] if S == "all" else S
                        prm = byfold[k]["S"][Sk][m].get("pb_params") or {}
                        if "lam" in prm:
                            lam.append((prm["lam"], prm.get("rho")))
                    if lam:
                        row["pb_lam"] = {k: float(np.mean([l[0][k] for l in lam if k in l[0]])) for k in lam[0][0]}
                        row["pb_rho"] = float(np.mean([l[1] for l in lam]))
                    rows.append(row)
            if nmin >= 2:
                for m in res["methods"]:
                    vals = [byfold[k].get("inv_s_fit", {}).get(m, {}) for k in fl]
                    row = {"panel": panel, "unit": unit, "kind": "projection", "method": m, "S": "inf", "folds": fl}
                    for metric, key in [("pb_rmse", "pb_rmse_inf"), ("meanlog_rmse_gene", "meanlog_rmse_gene_inf")]:
                        v = [x.get(key) for x in vals]
                        for k, x in zip(fl, v):
                            row[f"{metric}_f{k}"] = x
                        row[metric] = float(np.mean(v)) if all(x is not None for x in v) else None
                    row["n_seeds_by_fold"] = [byfold[k]["n_seeds"] for k in fl]
                    rows.append(row)
    return rows


def write_rows_csv(path, rows):
    keys = ["panel", "unit", "kind", "method", "S"]
    allk = sorted(set().union(*[set(r) for r in rows]) - set(keys) - {"folds"}) if rows else []
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(keys + allk)
        for r in rows:
            w.writerow([r.get(k) for k in keys] + [json.dumps(r.get(k)) if isinstance(r.get(k), (dict, list)) else r.get(k) for k in allk])


def _row(res, panel, unit, method, S):
    for r in res["rows"]:
        if r["panel"] == panel and r["unit"] == unit and r["method"] == method and r["S"] == S:
            return r
    return None


def _max_common_S(res, panel, unit):
    by = res["units"][panel].get(unit)
    return min(v["n_seeds"] for v in by.values()) if by else None


def _common_folds(row, ref):
    """Folds present in both rows and the reference's pb_rmse averaged over them."""
    ks = [k for k in row["folds"] if k in ref["folds"] and row.get(f"pb_rmse_f{k}") is not None]
    return ks, (float(np.mean([ref[f"pb_rmse_f{k}"] for k in ks])) if ks else None)


def _like_for_like(res, panel, row):
    """Repair S5: (reference name, reference fold-balanced pb_rmse over the row's folds, folds) =
    the better of baseline:grand_mean with the row's method and direct:train_mean raw."""
    best = None
    for unit, m in (("baseline:grand_mean", row["method"]), ("direct:train_mean", "raw")):
        ref = _row(res, panel, unit, m, 1)
        if ref is None:
            continue
        ks, v = _common_folds(row, ref)
        if v is not None and (best is None or v < best[1]):
            best = (f"{unit} {m}", v, ks, ref)
    return best


def seed_curve(res, panel, method="recal_shrink_g", min_seeds=3):
    """Repair S6: per generator unit with >= min_seeds seeds in every fold: pb_rmse and lambda over S."""
    out = {}
    tm = _row(res, panel, "direct:train_mean", "raw", 1)
    for unit, by in res["units"][panel].items():
        if not by or next(iter(by.values())).get("kind") != "generator":
            continue
        n = min(v["n_seeds"] for v in by.values())
        if n < min_seeds:
            continue
        pts = []
        for S in range(1, n + 1):
            r = _row(res, panel, unit, method, S)
            if r is None or r.get("pb_rmse") is None:
                continue
            tmv = _common_folds(r, tm)[1] if tm else None
            lam = r.get("pb_lam") or {}
            pts.append({"S": S, "pb_rmse": r["pb_rmse"], "lam": float(np.mean(list(lam.values()))) if lam else None,
                        "direct_train_mean_same_folds": tmv})
        if not pts:
            continue
        v = [p_["pb_rmse"] for p_ in pts]
        out[unit] = {"method": method, "points": pts,
                     "increases_with_S": bool(all(b >= a for a, b in zip(v, v[1:]))),
                     "above_train_mean_at_every_S": bool(all(p_["direct_train_mean_same_folds"] is not None
                                                            and p_["pb_rmse"] > p_["direct_train_mean_same_folds"] for p_ in pts))}
    return out


def make_verdict(res):
    out = {"rule": f"like-for-like (primary, repair S5): a generator row ties if its fold-balanced held-out pb_rmse <= "
                   f"reference + {TIE_TOL}, beats if below, where reference = min(baseline:grand_mean with the SAME "
                   "calibration method, direct:train_mean raw) over the same folds. *_raw_round1_grand_mean keys keep the "
                   "round-1 rule (raw baseline:grand_mean) for continuity only; it is not like-for-like.",
           "selection_note": "genemean (primary unit) and several round-2 choices were informed by round-1 held-out "
                             "results on the same 37 donors (r2/evaluate.py docstring, S3)."}
    for panel in res["units"]:
        ref = _row(res, panel, "baseline:grand_mean", "raw", 1)
        dref = _row(res, panel, "direct:train_mean", "raw", 1)
        if ref is None:
            continue
        gm = ref["pb_rmse"]; fl = ref["folds"]
        p = {"grand_mean_pb_rmse": gm, "direct_train_mean_pb_rmse": dref["pb_rmse"] if dref else None,
             "grand_mean_meanlog_rmse_gene": ref["meanlog_rmse_gene"]}
        gen_rows = [r for r in res["rows"] if r["panel"] == panel and r["kind"] == "generator" and r["S"] != "all"
                    and r.get("pb_rmse") is not None]
        # Integration fix (2026-11-10): units may cover only some of the reference
        # folds (e.g. smoke fits, or extra folds 3-8 for a few arms).  Prefer rows
        # covering every reference fold, and compare each row with the grand mean
        # over the folds both have (the fold-balanced means are otherwise not
        # comparable; it raised KeyError before).
        full = [r for r in gen_rows if list(r["folds"]) == list(fl)]
        gen_rows = full or gen_rows
        if gen_rows:
            best = min(gen_rows, key=lambda r: r["pb_rmse"])
            ks, gms = _common_folds(best, ref)
            p["best_generator_row"] = {k: best.get(k) for k in ["unit", "method", "S", "pb_rmse", "meanlog_rmse_gene", "pb_lam", "pb_rho"]}
            p["best_generator_row"]["folds_compared"] = ks
            p["best_generator_row"]["grand_mean_same_folds"] = gms
            p["best_generator_row"]["diff_vs_grand_mean"] = best["pb_rmse"] - gms
            p["best_generator_row"]["folds_better_than_grand_mean"] = int(sum(best[f"pb_rmse_f{k}"] < ref[f"pb_rmse_f{k}"] for k in ks))
            p["best_generator_row"]["note"] = "best of many rows chosen on held-out: selection-optimistic"
            p["any_ties_or_beats_raw_round1_grand_mean"] = bool(best["pb_rmse"] <= gms + TIE_TOL)
            p["any_beats_raw_round1_grand_mean"] = bool(best["pb_rmse"] < gms)
            # repair S5: best row by its like-for-like margin.  Stage-blind units (condmix@1 etc.) are a
            # generator-made grand mean: reported separately, never as the generator verdict.
            for tag, rows_ in (("", [r for r in gen_rows if r.get("stage_mode") != "stage-blind"]),
                               ("_stage_blind", [r for r in gen_rows if r.get("stage_mode") == "stage-blind"])):
                lfl = best_lfl = None
                for r in rows_:
                    ref_ = _like_for_like(res, panel, r)
                    if ref_ is None:
                        continue
                    d = r["pb_rmse"] - ref_[1]
                    if lfl is None or d < lfl:
                        lfl, best_lfl = d, (r, ref_)
                if best_lfl is None:
                    continue
                r, ref_ = best_lfl
                p[f"best_generator_row{tag}_like_for_like"] = {
                    **{k: r.get(k) for k in ["unit", "method", "S", "pb_rmse", "pb_lam", "pb_rho", "stage_mode"]},
                    "reference": ref_[0], "reference_pb_rmse_same_folds": ref_[1], "folds_compared": ref_[2],
                    "diff_vs_reference": lfl,
                    "folds_better": int(sum(r[f"pb_rmse_f{k}"] < ref_[3][f"pb_rmse_f{k}"] for k in ref_[2])),
                    "note": "best of many rows chosen on held-out: selection-optimistic"}
                p[f"any{tag}_ties_or_beats_like_for_like"] = bool(lfl <= TIE_TOL)
                p[f"any{tag}_beats_like_for_like"] = bool(lfl < 0)
                p[f"any{tag}_beats_like_for_like_by_more_than_tol"] = bool(lfl < -TIE_TOL)
        S = _max_common_S(res, panel, PRIMARY[0])
        if S:
            r = _row(res, panel, PRIMARY[0], PRIMARY[1], S)
            raw = _row(res, panel, PRIMARY[0], "raw", S)
            ks, gms = _common_folds(r, ref)
            p["primary"] = {"unit": PRIMARY[0], "method": PRIMARY[1], "S": S, "pb_rmse": r["pb_rmse"],
                            "raw_pb_rmse_same_S": raw["pb_rmse"], "diff_vs_grand_mean": r["pb_rmse"] - gms,
                            "folds_compared": ks, "grand_mean_same_folds": gms,
                            "ties_or_beats_raw_round1_grand_mean": bool(r["pb_rmse"] <= gms + TIE_TOL),
                            "beats_raw_round1_grand_mean": bool(r["pb_rmse"] < gms),
                            "folds_better": int(sum(r[f"pb_rmse_f{k}"] < ref[f"pb_rmse_f{k}"] for k in ks)),
                            "pb_lam": r.get("pb_lam"), "pb_rho": r.get("pb_rho"),
                            "meanlog_rmse_gene": r["meanlog_rmse_gene"],
                            "mse_parts": {k: r.get(k) for k in ("mse", "shared", "stage", "donor")}}
            if dref:
                p["primary"]["diff_vs_direct_train_mean"] = r["pb_rmse"] - dref["pb_rmse"]
            ref_ = _like_for_like(res, panel, r)
            if ref_ is not None:
                p["primary"].update(like_for_like_reference=ref_[0], like_for_like_reference_pb_rmse=ref_[1],
                                    diff_vs_like_for_like=r["pb_rmse"] - ref_[1],
                                    ties_or_beats_like_for_like=bool(r["pb_rmse"] <= ref_[1] + TIE_TOL),
                                    beats_like_for_like=bool(r["pb_rmse"] < ref_[1]))
            pj = _row(res, panel, PRIMARY[0], PRIMARY[1], "inf")
            pjr = _row(res, panel, PRIMARY[0], "raw", "inf")
            if pj:
                p["primary"]["projection_inf_pb_rmse"] = pj["pb_rmse"]
                p["primary"]["raw_projection_inf_pb_rmse"] = pjr["pb_rmse"]
                p["primary"]["raw_projection_inf_meanlog_rmse_gene"] = pjr["meanlog_rmse_gene"]
            rr = _row(res, panel, PRIMARY[0], "recal_shrinkr_g", S)
            if rr:
                p["primary"]["with_rho_variant_pb_rmse"] = rr["pb_rmse"]
        fl_ = res["floors"][panel]
        p["floor_insample_mse_mean"] = float(np.mean([fl_[k]["floor_insample_mse"] for k in fl]))
        p["oracle_heldout_stage_mean_pb_rmse"] = float(np.mean([fl_[k]["oracle_heldout_stage_mean_pb_rmse"] for k in fl]))
        p["oracle_heldout_grand_mean_pb_rmse"] = float(np.mean([fl_[k]["oracle_heldout_grand_mean_pb_rmse"] for k in fl]))
        p["within_stage_var_heldout_dofcorr_mean"] = float(np.mean([fl_[k]["within_stage_var_heldout_dofcorr"] for k in fl]))
        p["within_stage_var_train_dofcorr_mean"] = float(np.mean([fl_[k]["within_stage_var_train_dofcorr"] for k in fl]))
        p["floor_note"] = ("floor_insample_mse is optimistic (held-out stage means include the donor; single-donor stages "
                           "count 0); within_stage_var_*_dofcorr estimates the expected MSE of an oracle that knows the "
                           "POPULATION stage means: the donor-specific error no stage-level predictor can remove")
        p["seed_curve"] = seed_curve(res, panel)
        p["grand_mean_mse"] = ref["mse"]
        p["grand_mean_mse_shares"] = {k: ref[k] for k in ("shared", "stage", "donor")}
        out[panel] = p
    return out


def _f(v, w=6):
    return "-" if v is None else f"{v:.{3 if w == 6 else w}f}"


def summary_text(res):
    L = [f"improve_20261109 round-2 post-hoc calibration ({VERSION}). Held-out inner-fold donors; fold-balanced = mean "
         "over folds of (seed/subset-averaged) fold values. Development study only; panels are not comparable.",
         "Methods: see posthoc.py docstring. S = seeds averaged (all subsets of size S). 'inf' = 1/S projection.",
         "MSE columns: held-out MSE = shared + stage + donor; floor = donor error of the held-out stage-mean oracle.", ""]
    ver = res["verification"]
    bad = [k for sec in ver.values() for k, v in sec.items() if isinstance(v, dict) and v.get("ok") is False]
    nchk = sum(1 for sec in ver.values() for v in sec.values() if isinstance(v, dict) and "ok" in v)
    L.append(f"Verification vs results_screen48 (raw, 1e-6): {nchk - len(bad)}/{nchk} ok" + (f"; FAILED: {bad}" if bad else ""))
    L.append("")
    for panel, units in res["units"].items():
        fl = sorted(next(iter(units.values()))) if units else []
        L.append(f"=== panel {panel} (folds {fl}) ===")
        hdr = (f"{'unit':34s} {'method':15s} {'S':>4s} " + " ".join(f"pb_f{k:<4d}" for k in fl) + "  pb_FB  "
               + " ".join(f"ml_f{k:<4d}" for k in fl) + "  ml_FB  pbTrain pbTrLOO  MSE   shared stage  donor  lam/rho")
        L.append(hdr)
        for r in res["rows"]:
            if r["panel"] != panel:
                continue
            if r["kind"] != "projection" and r["S"] not in (1, "all") and r["S"] != _max_common_S(res, panel, r["unit"]):
                continue
            lam = ""
            if r.get("pb_lam"):
                lam = ",".join(f"{v:.2f}" for v in r["pb_lam"].values()) + f"/{r.get('pb_rho', 0):.2f}"
            L.append(f"{(r['unit'] + (' [SB]' if r.get('stage_mode') == 'stage-blind' else ''))[:34]:34s} {r['method']:15s} {str(r['S']):>4s} "
                     + " ".join(f"{_f(r.get(f'pb_rmse_f{k}'), 4):>8s}" for k in fl) + f" {_f(r.get('pb_rmse'), 4):>6s} "
                     + " ".join(f"{_f(r.get(f'meanlog_rmse_gene_f{k}'), 4):>8s}" for k in fl)
                     + f" {_f(r.get('meanlog_rmse_gene'), 4):>6s} {_f(r.get('pb_rmse_train'), 4):>7s} {_f(r.get('pb_rmse_train_loo'), 4):>7s}"
                     + f" {_f(r.get('mse'), 4):>6s} {_f(r.get('shared'), 4):>6s} {_f(r.get('stage'), 4):>6s} {_f(r.get('donor'), 4):>6s}  {lam}")
        fl_ = res["floors"].get(panel, {})
        for k in fl:
            x = fl_.get(k, {})
            L.append(f"floor fold{k}: insample donor MSE {_f(x.get('floor_insample_mse'), 4)}, oracle held-out stage-mean pb_rmse "
                     f"{_f(x.get('oracle_heldout_stage_mean_pb_rmse'), 4)}, oracle held-out grand-mean pb_rmse "
                     f"{_f(x.get('oracle_heldout_grand_mean_pb_rmse'), 4)}, within-stage var held-out/train "
                     f"{_f(x.get('within_stage_var_heldout_dofcorr'), 4)}/{_f(x.get('within_stage_var_train_dofcorr'), 4)}, "
                     f"meanlog oracle {_f(x.get('meanlog_oracle_rmse_gene_expected'), 4)}, n by stage {x.get('heldout_n_by_stage')}")
        L.append("")
    for panel, pv in (res.get("verdict") or {}).items():
        if not isinstance(pv, dict) or not pv.get("seed_curve"):
            continue
        L.append(f"=== seed curve, panel {panel}: recal_shrink_g held-out pb_rmse (fitted lambda) by number of seeds S,"
                 " next to direct:train_mean (repair S6) ===")
        for unit, c in pv["seed_curve"].items():
            pts = "  ".join(f"S={q['S']}: {q['pb_rmse']:.4f} (lam {_f(q['lam'], 2)})"
                            for q in c["points"])
            tm = c["points"][0]["direct_train_mean_same_folds"]
            L.append(f"{unit[:34]:34s} {pts}   train_mean {_f(tm, 4)}  increases_with_S={c['increases_with_S']}"
                     f"  worse_than_train_mean_at_every_S={c['above_train_mean_at_every_S']}")
        if all(c["above_train_mean_at_every_S"] for c in pv["seed_curve"].values()):
            L.append("-> at every S the calibrated generators are worse than direct:train_mean: the generators' stage "
                     "deviations add error; a calibrated single seed is best only because lambda is smallest there.")
        L.append("")
    L.append("Stage-blind units (condmix@1 etc.) are generator-made grand-mean predictors: calibration results, not stage fixes.")
    L.append("VERDICT: " + json.dumps(_clean(res.get("verdict", {})), indent=1))
    return "\n".join(L) + "\n"


# ==========================================================================
# CLI
# ==========================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("baselines", help="precompute the regenerated-baseline cache for one panel/fold")
    b.add_argument("--panel", required=True); b.add_argument("--fold", type=int, required=True)
    b.add_argument("--names", default=",".join(BASELINE_NAMES))
    rv = sub.add_parser("reverdict", help="recompute rows / verdict / summary from an existing results.json")
    rv.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--fits-root", action="append", help="[TAG=]DIR, repeatable (default runs/improve_20261109/fits)")
    r.add_argument("--gen-file", action="append", help="mode file written with generate.py --out (repeatable)")
    r.add_argument("--modes", default="all", help="all | default | comma list of mode names")
    r.add_argument("--max-subsets", type=int, default=35)
    r.add_argument("--no-baselines", action="store_true")
    r.add_argument("--panels", default="hvg2000,wide"); r.add_argument("--folds", default="0,1,2")
    for p in (b, r):
        p.add_argument("--caches-root", default=str(RUNS / "caches"))
        p.add_argument("--out", default=str(DEFAULT_OUT))
        p.add_argument("--screen48", default=str(RUNS / "results_screen48"), help="DIR or 'none'")
    a = ap.parse_args(argv)
    out = Path(a.out).resolve()
    if a.cmd == "reverdict":
        return reverdict(out)
    for frozen in ("fits", "results", "results_screen48", "screen_analysis", "caches"):
        fp = (RUNS / frozen).resolve()
        if out == fp or fp in out.parents:
            raise SystemExit(f"refusing to write into frozen {fp}")

    def log(msg):
        print(json.dumps({"t": time.strftime("%H:%M:%S"), "msg": msg}), flush=True)
    if a.cmd == "baselines":
        real = E.load_real(Path(a.caches_root) / a.panel / f"fold{a.fold}", a.panel, a.fold)
        for name in a.names.split(","):
            baseline_predictions(real, name, out, None if a.screen48 == "none" else Path(a.screen48), log)
        return None
    return run(a, log)


if __name__ == "__main__":
    main()
