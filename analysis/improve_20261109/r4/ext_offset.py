#!/usr/bin/env python3
"""Cohort offset and cross-cohort stage replication (round 4, external donors).

Analysis only: no model is trained.  Reads
  * SEA-AD: counts of the 37 development donors only (folds.json
    ``development_donors``; every count file read is listed in the outputs
    under ``donor_guard``), supertype labels and donor covariates from the
    frozen hvg2000 fold-0 cache metadata (train + heldout = the 37 donors);
  * MADRC ITG (GSE268599 store, 31 donors) and GSE264648 MTG (15 donors,
    offset check only, on its ``genes_present`` genes).
Writes only to /ml2/alz/runs/improve_20261109/ext/analysis_offset/.

Definitions (shared by every part)
----------------------------------
* Donor profile ("pb"): log1p(CP10k) of the donor's summed counts over the
  panel genes, panel total as denominator -- exactly the evaluator's
  observed pseudobulk (metrics.log_cp10k on summed panel counts).
  "meanlog": mean over the donor's cells of per-cell log1p(panel CP10k).
* Panels: ``hvg2000`` fold k = caches/hvg2000/fold{k}/space.npz ``hvg``
  (fitted on that fold's SEA-AD train donors).  ``neutral`` = genes detected
  (count > 0) in at least 10% of pooled cells in BOTH SEA-AD and MADRC
  (global version: all 37 SEA-AD donors -- descriptive; per-fold version:
  that fold's SEA-AD train donors only).
* Stage = NIA-AA ADNC index 0..3 (Not AD, Low, Intermediate, High).
  Contrasts over donor profiles: ``hn`` = mean(High) - mean(Not AD);
  ``hnl`` = mean(High) - mean(Not AD + Low donors); ``slope`` = OLS slope on
  ordinal stage (metrics.stage_contrasts); ``braak_slope`` = OLS slope on
  Braak 0..6.  Adjusted versions (``*_adj``): per-gene OLS within a cohort
  on [1, stage dummies (or ordinal stage / Braak for slopes), standardised
  age, male, APOE4 dose, log median UMIs per cell]; ``hnl_adj`` =
  b_High - n_Low/(n_NotAD+n_Low) * b_Low.  All contrasts are linear in the
  profiles (weight vector over donors), which makes permutation cheap.
* Agreement of the MADRC contrast y with the SEA-AD contrast x over genes:
  Spearman, Pearson, top-decile sign agreement (top 10% of genes by |x|),
  regression slope of y on x.  Nulls: stage labels permuted among the donors
  of ONE cohort (each cohort separately; covariates kept with their donor),
  2,000 draws, one-sided p = (1 + #null >= obs) / (1 + n).
* Split-half reliability: donors of each stage split at random into halves
  (stratified), Spearman between the half contrasts, mean over splits and
  Spearman-Brown corrected value (= estimated reliability of the full-cohort
  contrast).  Its null: the same after permuting stage labels.  Attenuation
  ceiling of a cross-cohort Spearman = sqrt(rel_SEA-AD * rel_MADRC).
* Frozen disease-associated-microglia list: metrics.DAM_UP / DAM_DOWN, a
  verbatim copy of r2/PROTOCOL_R2.md (copied again here as DAM_UP_R2 /
  DAM_DOWN_R2 and checked equal).  Direction score of an hn contrast:
  sign_frac and signed_mean over listed genes present, with the stage-label
  permutation null metrics.dam_perm_null.

Part 1 (cohort offset, MADRC vs SEA-AD, global, all 37 SEA-AD donors)
  offset_g = mean over the 4 stages of (MADRC stage mean - SEA-AD stage mean).
  Noise: se2_g = (1/16) sum_s (v_M/n_Ms + v_S/n_Ss), v = pooled within-stage
  between-donor variance of that cohort; sigma_g = pooled within-(cohort,
  stage) between-donor SD; t = offset/se, dof = n - 8, BH q.
  Noise-corrected RMS offset = sqrt(max(mean(offset^2) - mean(se2), 0)).
  Compared with sigma and with the SEA-AD hn contrast (raw and
  noise-corrected).  Steps, each recomputed from cells:
    S0 raw;
    S1 depth matched: every SEA-AD cell binomially thinned to the MADRC
       per-cell UMI distribution by quantile matching (full 36,601 axis,
       p = min(1, q_M(rank) / umi), seed fixed); MADRC untouched (it is the
       shallower cohort);
    S2 S1 + PVM removed: cells weighted by 1 - P(Micro-PVM_1);
    S3 S1 + supertype mix standardised: a donor profile is the mixture over
       the six SEA-AD supertypes of the donor's per-type (soft-assigned)
       pseudobulk proportions, with the pooled SEA-AD mix as weights;
       supertype probabilities from a multinomial logistic regression on 30
       PCs of the global neutral panel, trained on SEA-AD cells
       (out-of-fold by donor-grouped 3-fold CV for SEA-AD, refitted on all
       SEA-AD cells for MADRC);
    S4 S3 + ambient filter: only cells whose astrocyte AND neuronal marker
       scores (ext_select_microglia MARKERS, mean log1p full-axis CP10k) are
       <= 0.2, in both cohorts.
  Also GSE264648: two-group matched offset (Control vs SEA-AD Not AD+Low,
  AD vs SEA-AD Intermediate+High, averaged) on neutral genes present in
  GSE264648, compared with the MADRC offset defined the same way.

Part 2 (stage replication across cohorts): agreement of MADRC with SEA-AD
  contrasts on the global neutral panel and every hvg2000 fold-0..2 panel,
  raw and adjusted, permutation nulls, split-half reliabilities, DAM.  Per
  fold k (0..8): MADRC contrast vs the contrast of fold-k SEA-AD TRAIN donors
  (MADRC-label permutation null) and, descriptively, SEA-AD train / MADRC /
  combined contrasts vs the fold's held-out contrast (held-out-label
  permutation null).

Part 3 (does adding MADRC donors help predict SEA-AD held-out donors?)
  Per inner fold k (0..8, panel hvg2000 fold k, profiles from full counts
  restricted to the panel = the evaluator's observed pb):
  predictors stage_mean (mean of donor log profiles within stage),
  shrunk (empirical-Bayes stage shrinkage, baselines.py formula), ridge
  (baselines.py design and LOO alpha); training sets
    seaad          SEA-AD fold-k train donors only;
    ext_none       + MADRC profiles unchanged;
    ext_add        + MADRC profiles minus (mean MADRC - mean SEA-AD train);
    ext_stageadj   + MADRC minus the cohort coefficient of a per-gene OLS
                   on [stage one-hot, cohort] over train donors;
    ext_stageadj_cov  same with male, APOE4, standardised age in the OLS;
  and pure-transfer predictors ext_delta / ext_delta_shrunk: SEA-AD train
  mean + MADRC (shrunken) stage deviations re-centred on the SEA-AD train
  stage mix (no SEA-AD stage information).  References: grand_pooled
  (log CP10k of pooled train counts = evaluator grand_mean) and train_mean
  (mean of train donor profiles).  Scores on held-out SEA-AD donors: pb_rmse
  (mean over donors), hn / slope agreement with the held-out contrast
  (descriptive), and the pooled cross-fitted donor score (metrics.xfit_score,
  one value per repeat: folds 0-2, 3-5, 6-8) with its within-fold
  permutation null.  Paired differences against the seaad training set: per
  donor (37 per repeat) and per fold.

Review repair (2026-11-11; findings D2, D3, F3, F4, F5, F6)
-----------------------------------------------------------
* F3 depth.  Binomial thinning (S1) leaves expected pseudobulk proportions
  unchanged, so S1 can only detect the log1p nonlinearity, NOT a depth or
  capture-efficiency effect; it is reported as that check only.  The depth
  evidence is ``depth_regression`` (part 1, per panel): within each cohort a
  per-gene OLS of donor profiles on [1, stage dummies, log median UMI]; the
  offset predicted from the cohort depth gap = slope_g * (mean log depth
  MADRC - mean log depth SEA-AD), with each cohort's slope; RMS and
  correlation with the S0 offset.
* F4 shared covariates.  Part 2 adds ``*_adjpvm``: the ``*_adj`` covariates
  plus the donor PVM fraction (mean P(Micro-PVM_1) of the part-1 supertype
  classifier, out-of-fold for SEA-AD, the same classifier for MADRC).  This,
  not the raw value, is the headline replication number.
* D3 ambient genes.  Part 2 adds panels ``hvg2000_fold{k}_noast`` (minus the
  fixed astrocyte ambient list AST_AMBIENT_GENES) and ``hvg2000_fold{k}_noamb``
  (minus that list, the neuronal markers and the 20% of panel genes with the
  largest |S0 offset|), and ``ambient_stage_trend`` (per-donor detection of
  astrocyte genes vs stage in each cohort, Spearman with a 20,000-draw label
  permutation null).
* D2 library quality.  ``madrc_qc_stage_trend``: per-donor barcodes, QC pass
  fraction, kept cells, kept fraction, median mito of kept cells and the
  per-reason rejection fractions of myeloid-cluster nuclei vs ADNC stage
  (Spearman, 20,000-draw permutation null), from donors.json receipts.
* F5 age.  Part 3 recodes MADRC '90+' ages per fold as the mean age of that
  fold's SEA-AD TRAIN donors aged >= 90 (the store's 94.35 used all 37).
* F6 fold SEs.  Folds 0-8 are three repeats over the same 37 donors, so the
  per-fold summaries report the SE over the 3 folds WITHIN each repeat
  (averaged over repeats) and the repeat means, not sd/sqrt(9).
MADRC profiles use every stored cell (donor pseudobulk; no cell cap is needed
for donor-level profiles, unlike the cell-weighted cache statistics).

Usage: python3 -B ext_offset.py [--quick] [--n-perm 2000]
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "3"

import argparse
import hashlib
import json
import sys
import time
from math import comb
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.stats import rankdata, t as tdist

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import metrics as MX  # noqa: E402  (r4 copy of r3/metrics.py, read only)

STUDY = HERE.parent
REBUILT = Path("/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark/rebuilt")
RUNS = Path("/ml2/alz/runs/improve_20261109")
EXT = RUNS / "ext"
OUT = EXT / "analysis_offset"
CACHE = RUNS / "caches" / "hvg2000"
TYPES = ["Micro-PVM_1", "Micro-PVM_2", "Micro-PVM_2_1-SEAAD", "Micro-PVM_2_3-SEAAD", "Micro-PVM_3-SEAAD",
         "Micro-PVM_4-SEAAD"]
# r2/PROTOCOL_R2.md "Frozen stage-signal gene list (F8)", copied verbatim; must equal metrics.DAM_*.
DAM_UP_R2 = ["APOE", "TREM2", "TYROBP", "SPP1", "GPNMB", "LPL", "CD9", "LGALS3", "CST7", "ITGAX", "CLEC7A", "AXL",
             "CSF1", "CD63", "CTSD", "CTSB", "FTH1", "LILRB4", "MITF", "MYO1E"]
DAM_DOWN_R2 = ["P2RY12", "TMEM119", "CX3CR1", "P2RY13", "SELPLG", "CSF1R", "SALL1"]
assert DAM_UP_R2 == MX.DAM_UP and DAM_DOWN_R2 == MX.DAM_DOWN
# ext_select_microglia.MARKERS (ast, neu), copied verbatim.
AMBIENT_MARKERS = {"ast": ["GFAP", "AQP4", "SLC1A2", "ADGRV1", "GJA1"],
                   "neu": ["RBFOX3", "SNAP25", "SYT1", "GAD1", "SLC17A7"]}
# Review D3: fixed astrocyte ambient list (ext_select_microglia MARKERS['ast'] + ALDH1L1, ATP1A2, SLC1A3).
AST_AMBIENT_GENES = ["GFAP", "AQP4", "SLC1A2", "ADGRV1", "GJA1", "ALDH1L1", "ATP1A2", "SLC1A3"]
NEUTRAL_DET = 0.10
AMBIENT_THR = 0.2
SEED = 20261111
DONOR_GUARD: list[str] = []


# ============================================================================ helpers
def clean(o):
    """JSON-safe copy: numpy scalars to Python, nan/inf to None."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return [clean(v) for v in o.tolist()]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    return o


def f_(x):
    return MX._f(x)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def bh(p):
    p = np.asarray(p, float); n = len(p)
    o = np.argsort(p); q = p[o] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n); out[o] = np.minimum(q, 1)
    return out


def folds_all():
    f1 = json.loads((STUDY / "folds.json").read_text())
    f2 = json.loads((HERE / "folds_r2.json").read_text())
    dev = list(f1["development_donors"])
    for k in range(3):
        assert f2["inner_folds"][k]["train_donors"] == f1["inner_folds"][k]["train_donors"]
    return dev, f2["inner_folds"]


# ============================================================================ data
class Cohort:
    """Per-donor full-axis CSR counts and donor covariates of one cohort."""

    def __init__(self, name):
        self.name = name
        self.donors, self.X, self.stage, self.braak = [], [], [], []
        self.male, self.apoe4, self.age = [], [], []
        self.types = None  # SEA-AD only: list of int arrays per donor
        self.extra = {}

    def finish(self):
        for k in ["stage", "braak", "male", "apoe4", "age"]:
            setattr(self, k, np.asarray(getattr(self, k), float))
        self.stage = self.stage.astype(int)
        self.umi = [np.asarray(x.sum(1)).ravel() for x in self.X]
        self.depth = np.array([np.median(u) for u in self.umi])
        self.ncell = np.array([x.shape[0] for x in self.X])
        return self

    def subset(self, names):
        ix = [self.donors.index(d) for d in names]
        c = Cohort(self.name)
        c.donors = list(names); c.X = [self.X[i] for i in ix]
        for k in ["stage", "braak", "male", "apoe4", "age", "depth", "ncell"]:
            setattr(c, k, getattr(self, k)[ix])
        c.umi = [self.umi[i] for i in ix]
        c.types = None if self.types is None else [self.types[i] for i in ix]
        c.extra = self.extra
        return c


def load_seaad(dev):
    meta = {}
    for part in ["train", "heldout"]:
        m = np.load(CACHE / "fold0" / part / "metadata.npz", allow_pickle=False)
        for i, d in enumerate(m["donors"]):
            sel = m["donor"] == d
            meta[str(d)] = dict(raw_c=m["raw_c"][i], cell=m["cell"][sel], type=m["type"][sel])
    assert sorted(meta) == sorted(dev), "fold-0 cache does not hold exactly the 37 development donors"
    stage_by = json.loads((STUDY / "folds.json").read_text())["stage_by_donor"]
    c = Cohort("SEA-AD")
    for d in sorted(dev):
        if d not in dev:
            raise RuntimeError("non-development donor")
        DONOR_GUARD.append(d)
        X = sp.load_npz(REBUILT / "counts" / f"{d}.npz").tocsr()
        names = np.load(REBUILT / "counts" / f"{d}_names.npy", allow_pickle=False)
        mm = meta[d]
        if not np.array_equal(names, mm["cell"]):
            raise RuntimeError(f"cell order of {d} differs from cache")
        rc = mm["raw_c"]
        st = int(round(rc[3] * 3))
        assert st == int(stage_by[d])
        c.donors.append(d); c.X.append(X); c.stage.append(st); c.braak.append(round(rc[0] * 6))
        c.male.append(rc[4]); c.apoe4.append(rc[5]); c.age.append(rc[6])
        c.types = (c.types or []) + [mm["type"].astype(int)]
    return c.finish()


def load_madrc():
    root = EXT / "GSE268599"
    dj = json.loads((root / "donors.json").read_text())["donors"]
    c = Cohort("MADRC")
    c.extra["pvm_prob_fetch"] = []
    c.extra["ambient_fetch"] = []
    c.extra["age_censored"] = []
    for d in sorted(dj):
        r = dj[d]
        X = sp.load_npz(root / r["counts_file"]).tocsr()
        cm = np.load(root / r["cellmeta_file"])
        assert X.shape[0] == r["cells_kept"] == len(cm["umi"])
        c.donors.append(d); c.X.append(X); c.stage.append(r["ADNC_index"]); c.braak.append(r["Braak_index"])
        c.male.append(r["male"]); c.apoe4.append(r["APOE4_dose"]); c.age.append(r["age"])
        c.extra["pvm_prob_fetch"].append(cm["pvm_prob"])
        c.extra["ambient_fetch"].append(np.stack([cm["score_ast"], cm["score_neu"]], 1))
        c.extra["age_censored"].append(bool(r["age_90plus_censored"]))
    c.extra["age_censored"] = np.array(c.extra["age_censored"])
    genes = np.load(root / "counts" / "genes.npy", allow_pickle=False)
    return c.finish(), genes


def load_imp():
    root = EXT / "GSE264648"
    dj = json.loads((root / "donors.json").read_text())["donors"]
    c = Cohort("GSE264648")
    for d in sorted(dj):
        r = dj[d]
        X = sp.load_npz(root / "counts" / f"{d}.npz").tocsr()
        c.donors.append(d); c.X.append(X)
        c.stage.append(1 if r["diagnosis"] == "AD" else 0)  # dx: 0 Control, 1 AD
        c.braak.append(r["braak"]); c.male.append(1.0 if r["sex"] == "M" else 0.0); c.apoe4.append(np.nan)
        c.age.append(r["age"])
    present = np.load(root / "counts" / "genes_present.npy")
    genes = np.load(root / "counts" / "genes.npy", allow_pickle=False)
    return c.finish(), present, genes


# ============================================================================ profiles
def panel_sums(C: Cohort, gidx, weights=None, Xs=None):
    """[n_donors, G] summed (optionally cell-weighted) panel counts."""
    Xs = C.X if Xs is None else Xs
    out = np.zeros((len(Xs), len(gidx)))
    for i, X in enumerate(Xs):
        Y = X[:, gidx]
        if weights is None:
            out[i] = np.asarray(Y.sum(0)).ravel()
        else:
            out[i] = np.asarray(Y.T @ weights[i]).ravel()
    return out


def pb_profiles(C, gidx, weights=None, Xs=None):
    return MX.log_cp10k(panel_sums(C, gidx, weights, Xs))


def meanlog_profiles(C, gidx, Xs=None):
    Xs = C.X if Xs is None else Xs
    out = np.zeros((len(Xs), len(gidx)))
    for i, X in enumerate(Xs):
        Y = X[:, gidx].astype(np.float64).tocsr()
        s = np.asarray(Y.sum(1)).ravel()
        Y = sp.diags(1e4 / np.maximum(s, 1)) @ Y
        Y.data = np.log1p(Y.data)
        out[i] = np.asarray(Y.mean(0)).ravel()
    return out


def mix_profiles(C, gidx, probs, wref, Xs=None, min_eff=3.0):
    """Supertype-mix-standardised profile: sum_t w_t * normalised(sum_c p_ct x_c)."""
    Xs = C.X if Xs is None else Xs
    out = np.zeros((len(Xs), len(gidx)))
    n_fallback = 0
    for i, X in enumerate(Xs):
        Y = X[:, gidx].astype(np.float64)
        S = np.asarray((Y.T @ probs[i]).T)            # [T, G]
        eff = probs[i].sum(0)
        ok = (eff >= min_eff) & (S.sum(1) > 0)
        n_fallback += int((~ok).sum())
        w = np.where(ok, wref, 0.0); w = w / w.sum()
        q = (w[:, None] * (S / np.maximum(S.sum(1, keepdims=True), 1e-12))).sum(0)
        out[i] = np.log1p(1e4 * q / q.sum())
    return out, n_fallback


def detection(C, donors_ix=None):
    ix = range(len(C.X)) if donors_ix is None else donors_ix
    nnz = np.zeros(C.X[0].shape[1]); n = 0
    for i in ix:
        X = C.X[i]
        nnz += np.bincount(X.indices, minlength=X.shape[1]); n += X.shape[0]
    return nnz / n


def thin_to(C: Cohort, target_umi, seed):
    """Binomially thin every cell of C so the pooled per-cell UMI distribution
    matches target_umi by quantile matching (downward only)."""
    allu = np.concatenate(C.umi)
    ranks = (rankdata(allu, method="ordinal") - 0.5) / len(allu)
    tq = np.quantile(target_umi, ranks)
    p_all = np.minimum(1.0, tq / np.maximum(allu, 1))
    rng = np.random.default_rng(seed)
    out, k = [], 0
    for X in C.X:
        n = X.shape[0]; p = p_all[k:k + n]; k += n
        Y = X.copy().astype(np.int64)
        pr = np.repeat(p, np.diff(Y.indptr))
        Y.data = rng.binomial(Y.data, pr)
        Y.eliminate_zeros()
        out.append(Y.tocsr())
    return out, p_all


def marker_scores(C, genes, Xs=None):
    Xs = C.X if Xs is None else Xs
    ix = {g: i for i, g in enumerate(genes)}
    res = []
    for X in Xs:
        u = np.asarray(X.sum(1)).ravel()
        cols = {}
        for k, gl in AMBIENT_MARKERS.items():
            Y = X[:, [ix[g] for g in gl if g in ix]].astype(np.float64)
            Y = sp.diags(1e4 / np.maximum(u, 1)) @ Y
            Y.data = np.log1p(Y.data)
            cols[k] = np.asarray(Y.mean(1)).ravel()
        res.append(np.stack([cols["ast"], cols["neu"]], 1))
    return res


# ============================================================================ supertype classifier
def supertype_probs(S: Cohort, M: Cohort, gidx, seed=SEED):
    from sklearn.linear_model import LogisticRegression

    def feats(C):
        L = []
        for X in C.X:
            Y = X[:, gidx].astype(np.float64).tocsr()
            s = np.asarray(Y.sum(1)).ravel()
            Y = sp.diags(1e4 / np.maximum(s, 1)) @ Y
            Y.data = np.log1p(Y.data)
            L.append(Y.toarray().astype(np.float32))
        return L
    LS = feats(S)
    yS = np.concatenate(S.types)
    dS = np.concatenate([[i] * len(t) for i, t in enumerate(S.types)])
    AS = np.concatenate(LS)

    def fit(A, y):
        mu = A.mean(0); sd = A.std(0) + 1e-3
        Z = (A - mu) / sd
        sub = np.random.default_rng(seed).choice(len(Z), min(len(Z), 8000), replace=False)
        _, _, Vt = np.linalg.svd(Z[sub] - Z[sub].mean(0), full_matrices=False)
        W = Vt[:30].T
        clf = LogisticRegression(C=1.0, max_iter=3000)
        clf.fit(Z @ W, y)
        return lambda B: clf.predict_proba(((B - mu) / sd) @ W), clf.classes_
    # out-of-fold for SEA-AD (donor-grouped 3-fold, stratified by stage via donor order)
    rng = np.random.default_rng(seed)
    grp = np.zeros(len(S.donors), int)
    for s in range(4):
        ix = rng.permutation(np.where(S.stage == s)[0])
        grp[ix] = np.arange(len(ix)) % 3
    oof = np.zeros((len(AS), len(TYPES)))
    for g in range(3):
        te = np.isin(dS, np.where(grp == g)[0])
        pred, cls = fit(AS[~te], yS[~te])
        P = np.zeros((te.sum(), len(TYPES))); P[:, cls] = pred(AS[te]); oof[te] = P
    pred, cls = fit(AS, yS)
    PM = []
    for i in range(len(M.X)):
        one = Cohort("tmp"); one.X = [M.X[i]]
        A = feats(one)[0]
        P = np.zeros((len(A), len(TYPES))); P[:, cls] = pred(A); PM.append(P)
    PS, k = [], 0
    for t in S.types:
        PS.append(oof[k:k + len(t)]); k += len(t)
    acc = float((oof.argmax(1) == yS).mean())
    per = {TYPES[t]: {"n": int((yS == t).sum()), "recall": f_((oof.argmax(1)[yS == t] == t).mean()) if (yS == t).any() else None}
           for t in range(len(TYPES))}
    mixS = oof.mean(0); mixM = np.concatenate(PM).mean(0)
    true_mix = np.bincount(yS, minlength=len(TYPES)) / len(yS)
    info = {"features": f"30 PCs of standardised log1p CP10k over {len(gidx)} neutral genes", "oof_accuracy": acc,
            "per_type_oof": per, "seaad_true_mix": dict(zip(TYPES, map(float, true_mix))),
            "seaad_oof_mix": dict(zip(TYPES, map(float, mixS))), "madrc_pred_mix": dict(zip(TYPES, map(float, mixM)))}
    return PS, PM, mixS, info


# ============================================================================ contrasts
def _design_dummies(stage):
    return np.stack([(stage == s).astype(float) for s in (1, 2, 3)], 1)


def contrast_weights(kind, stage, braak=None, cov=None):
    """Weight vector w [n] with contrast = w @ P.  cov [n, k] -> adjusted."""
    stage = np.asarray(stage); n = len(stage)
    if cov is None:
        w = np.zeros(n)
        if kind == "hn":
            a, b = stage == 3, stage == 0
        elif kind == "hnl":
            a, b = stage == 3, stage <= 1
        elif kind in ("slope", "braak_slope"):
            x = (stage if kind == "slope" else np.asarray(braak, float)).astype(float)
            x = x - x.mean(); ss = (x ** 2).sum()
            return x / ss if ss > 0 else None
        else:
            raise ValueError(kind)
        if a.sum() == 0 or b.sum() == 0:
            return None
        w[a] = 1 / a.sum(); w[b] = -1 / b.sum()
        return w
    cov = np.asarray(cov, float)
    if kind in ("hn", "hnl"):
        X = np.column_stack([np.ones(n), _design_dummies(stage), cov])
        if np.linalg.matrix_rank(X) < X.shape[1]:
            return None
        Pi = np.linalg.pinv(X)
        if kind == "hn":
            return Pi[3]
        n0, n1 = (stage == 0).sum(), (stage == 1).sum()
        return Pi[3] - n1 / (n0 + n1) * Pi[1]
    x = (stage if kind == "slope" else np.asarray(braak, float)).astype(float)
    X = np.column_stack([np.ones(n), x, cov])
    if np.linalg.matrix_rank(X) < X.shape[1]:
        return None
    return np.linalg.pinv(X)[1]


def cov_matrix(C: Cohort, ix=None, extra=None):
    """[std age, male, (APOE4), log median UMI] (+ extra columns, e.g. PVM fraction; review F4)."""
    ix = np.arange(len(C.donors)) if ix is None else np.asarray(ix)
    age = C.age[ix]; age = (age - age.mean()) / (age.std() + 1e-12)
    cols = [age, C.male[ix], np.log(C.depth[ix])]
    if np.isfinite(C.apoe4[ix]).all() and C.apoe4[ix].std() > 0:
        cols.insert(2, C.apoe4[ix])
    if extra is not None:
        cols.append(np.asarray(extra, float)[ix])
    return np.column_stack(cols)


def stage_trend(values, stage, n_perm=20000, seed=SEED, label=""):
    """Spearman of per-donor values with ordinal stage, one label-permutation null (two-sided p)."""
    v, st = np.asarray(values, float), np.asarray(stage)
    ok = np.isfinite(v)
    v, st = v[ok], st[ok]
    rho = MX.spearman(v, st)
    rng = np.random.default_rng(MX.stable_seed(seed, "trend", label))
    rv, rs = rankdata(v), rankdata(st)
    null = np.array([np.corrcoef(rv, rs[rng.permutation(len(rs))])[0, 1] for _ in range(n_perm)])
    return {"spearman": f_(rho), "p_two_sided": f_((1 + (np.abs(null) >= abs(rho) - 1e-12).sum()) / (1 + n_perm)),
            "by_stage_median": [f_(np.median(v[st == s])) if (st == s).any() else None for s in range(4)], "n": int(len(v))}


def perm_weights(kind, stage, braak, cov, n_perm, rng):
    """Weights under permuted stage (and Braak permuted jointly with stage)."""
    W = []
    n = len(stage)
    tries = 0
    while len(W) < n_perm and tries < 5 * n_perm:
        tries += 1
        p = rng.permutation(n)
        w = contrast_weights(kind, stage[p], None if braak is None else braak[p], cov)
        if w is not None:
            W.append(w)
    return np.array(W)


def _rank_rows(A):
    return rankdata(A, axis=1)


def agree_stats(x, Y, topk_idx=None, X_rows=None):
    """Agreement of rows Y [m,G] (MADRC side) with x [G] (SEA-AD side), or with
    rows X_rows [m,G] pairwise when given.  Returns dict of arrays [m]."""
    Y = np.atleast_2d(np.asarray(Y, float))
    G = Y.shape[1]; k = max(1, int(np.ceil(0.1 * G)))
    fixed = X_rows is None
    if fixed:
        x = np.asarray(x, float)
        rx1 = rankdata(x)[None]; top1 = np.argsort(-np.abs(x), kind="stable")[:k]
    else:
        X_rows = np.atleast_2d(np.asarray(X_rows, float))
        if X_rows.shape[0] == 1 and Y.shape[0] > 1:
            X_rows = np.broadcast_to(X_rows, Y.shape)
        if Y.shape[0] == 1 and X_rows.shape[0] > 1:
            Y = np.broadcast_to(Y, X_rows.shape)
    out = {k_: [] for k_ in ("spearman", "pearson", "top_decile_sign", "slope")}
    for a in range(0, len(Y), 400):
        y = np.array(Y[a:a + 400])
        if fixed:
            xx = np.broadcast_to(x, y.shape); rx = np.broadcast_to(rx1, y.shape)
            sx = np.broadcast_to(np.sign(x[top1]), (len(y), k)); sy = np.sign(y[:, top1])
        else:
            xx = np.array(X_rows[a:a + 400]); rx = _rank_rows(xx)
            top = np.argsort(-np.abs(xx), axis=1, kind="stable")[:, :k]
            sx = np.take_along_axis(np.sign(xx), top, 1); sy = np.take_along_axis(np.sign(y), top, 1)
        ry = _rank_rows(y)
        out["spearman"].append(_rowcorr(rx, ry)); out["pearson"].append(_rowcorr(xx, y))
        out["top_decile_sign"].append((sx == sy).mean(1))
        xc = xx - xx.mean(1, keepdims=True); yc = y - y.mean(1, keepdims=True)
        out["slope"].append((xc * yc).sum(1) / (xc ** 2).sum(1))
    return {k_: np.concatenate(v) for k_, v in out.items()}


def _rowcorr(A, B):
    A = A - A.mean(1, keepdims=True); B = B - B.mean(1, keepdims=True)
    return (A * B).sum(1) / np.sqrt((A ** 2).sum(1) * (B ** 2).sum(1))


def null_block(obs, null):
    null = np.asarray(null, float); null = null[np.isfinite(null)]
    if len(null) < 2 or obs is None or not np.isfinite(obs):
        return {"value": f_(obs), "null_n": int(len(null))}
    sd = null.std(ddof=1)
    return {"value": f_(obs), "null_mean": f_(null.mean()), "null_sd": f_(sd), "null_q95": f_(np.quantile(null, .95)),
            "z": f_((obs - null.mean()) / sd) if sd > 0 else None,
            "p_one_sided": f_((1 + (null >= obs).sum()) / (1 + len(null))), "null_n": int(len(null))}


def replicate(PS, sS, bS, covS, PM, sM, bM, covM, kinds, n_perm, seed, label):
    """Cross-cohort agreement with two one-cohort permutation nulls."""
    out = {}
    rng = np.random.default_rng(MX.stable_seed(seed, label))
    # covS/covM: None, an array (-> '_adj'), or {suffix: array} (review F4: '_adj', '_adjpvm')
    csS = covS if isinstance(covS, dict) else {"_adj": covS}
    csM = covM if isinstance(covM, dict) else {"_adj": covM}
    for kind in kinds:
        for suf in [""] + list(csS):
            name = kind + suf
            cS_ = csS.get(suf) if suf else None
            cM_ = csM.get(suf) if suf else None
            wS = contrast_weights(kind, sS, bS, cS_)
            wM = contrast_weights(kind, sM, bM, cM_)
            if wS is None or wM is None:
                out[name] = {"defined": False}
                continue
            cS, cM = wS @ PS, wM @ PM
            obs = agree_stats(cS, cM)
            WpM = perm_weights(kind, sM, bM, cM_, n_perm, rng)
            WpS = perm_weights(kind, sS, bS, cS_, n_perm, rng)
            nullM = agree_stats(cS, WpM @ PM)
            nullS = agree_stats(None, np.broadcast_to(cM, (len(WpS), len(cM))), X_rows=WpS @ PS)
            res = {"defined": True, "sd_seaad": f_(cS.std()), "sd_madrc": f_(cM.std())}
            for st in obs:
                res[st] = {"value": f_(obs[st][0]),
                           "null_perm_madrc": null_block(obs[st][0], nullM[st]),
                           "null_perm_seaad": null_block(obs[st][0], nullS[st])}
            out[name] = res
    return out


def split_half(P, stage, braak, kind, n_split, rng, permute=False, n_perm_outer=1):
    """Mean split-half Spearman (and Spearman-Brown) of a contrast."""
    vals = []
    n = len(stage)
    for _ in range(n_perm_outer):
        pp = rng.permutation(n) if permute else np.arange(n)
        st = stage[pp]
        bk = None if braak is None else np.asarray(braak)[pp]
        Wa, Wb = [], []
        for _s in range(n_split):
            ha = np.zeros(n, bool)
            for s in np.unique(st):
                ix = rng.permutation(np.where(st == s)[0])
                ha[ix[:len(ix) // 2]] = True
            # odd-sized groups: the extra donor goes to half b; alternate by coin flip
            if rng.random() < .5:
                ha = ~ha
            wa = contrast_weights(kind, st[ha], None if bk is None else bk[ha])
            wb = contrast_weights(kind, st[~ha], None if bk is None else bk[~ha])
            if wa is None or wb is None:
                continue
            fa = np.zeros(n); fa[ha] = wa; fb = np.zeros(n); fb[~ha] = wb
            Wa.append(fa); Wb.append(fb)
        if not Wa:
            continue
        A = np.array(Wa) @ P; B = np.array(Wb) @ P
        vals.append(agree_stats(None, B, X_rows=A)["spearman"])
    if not vals:
        return None
    v = np.concatenate(vals)
    return v


def reliability(P, stage, braak, kinds, seed, label, n_split=300, n_null=60):
    rng = np.random.default_rng(MX.stable_seed(seed, "split", label))
    out = {}
    for kind in kinds:
        v = split_half(P, stage, braak, kind, n_split, rng)
        if v is None:
            out[kind] = {"defined": False}
            continue
        r = float(np.nanmean(v))
        null = split_half(P, stage, braak, kind, 10, rng, permute=True, n_perm_outer=n_null)
        nm = float(np.nanmean(null)) if null is not None else None
        out[kind] = {"defined": True, "n_splits": int(len(v)), "split_half_spearman": r,
                     "spearman_brown": f_(2 * r / (1 + r)) if r > -1 else None,
                     "perm_null_mean": f_(nm), "perm_null_sd_of_split_mean": f_(np.nanstd(null)) if null is not None else None,
                     "perm_null_spearman_brown": f_(2 * nm / (1 + nm)) if nm is not None and nm > -1 else None}
    return out


# ============================================================================ part 1
def depth_regression(PS, sS, dS, PM, sM, dM, off0):
    """Review F3: offset predicted from the depth gap with each cohort's within-cohort depth slope."""
    out = {"log_depth_gap_madrc_minus_seaad": f_(np.log(dM).mean() - np.log(dS).mean())}
    gap = np.log(dM).mean() - np.log(dS).mean()
    for nm, P, st, d in (("seaad_slope", PS, sS, dS), ("madrc_slope", PM, sM, dM)):
        X = np.column_stack([np.ones(len(st)), _design_dummies(st), np.log(d)])
        B = np.linalg.lstsq(X, P, rcond=None)[0]
        pred = B[-1] * gap
        out[nm] = {"rms_predicted_offset": f_(np.sqrt((pred ** 2).mean())),
                   "corr_with_observed_offset": f_(np.corrcoef(pred, off0)[0, 1]),
                   "spearman_with_observed_offset": MX.spearman(pred, off0)}
    out["note"] = ("between-donor depth slopes within a cohort (stage-adjusted) times the cohort depth gap; "
                   "binomial thinning (S1) cannot test depth for pseudobulk proportions, this can (review F3)")
    return out


def offset_stats(PS, sS, PM, sM, genes, top=25):
    stages = range(4)
    mS = np.stack([PS[sS == s].mean(0) for s in stages]); mM = np.stack([PM[sM == s].mean(0) for s in stages])
    nS = np.array([(sS == s).sum() for s in stages]); nM = np.array([(sM == s).sum() for s in stages])
    ssS = sum(((PS[sS == s] - mS[s]) ** 2).sum(0) for s in stages)
    ssM = sum(((PM[sM == s] - mM[s]) ** 2).sum(0) for s in stages)
    vS = ssS / (len(PS) - 4); vM = ssM / (len(PM) - 4)
    d = mM - mS
    off = d.mean(0)
    se2 = (vM[None] / nM[:, None] + vS[None] / nS[:, None]).sum(0) / 16
    sigma = np.sqrt((ssS + ssM) / (len(PS) + len(PM) - 8))
    dof = len(PS) + len(PM) - 8
    tt = off / np.sqrt(np.maximum(se2, 1e-20))
    p = 2 * tdist.sf(np.abs(tt), dof); q = bh(p)
    hnS = mS[3] - mS[0]; hnS_noise = vS * (1 / nS[0] + 1 / nS[3])
    hnM = mM[3] - mM[0]; hnM_noise = vM * (1 / nM[0] + 1 / nM[3])
    ms_off = float((off ** 2).mean()); ms_noise = float(se2.mean())
    corr_ms = max(ms_off - ms_noise, 0.0)
    ms_hnS = float((hnS ** 2).mean()); ms_hnS_c = max(ms_hnS - float(hnS_noise.mean()), 0.0)
    ms_hnM = float((hnM ** 2).mean()); ms_hnM_c = max(ms_hnM - float(hnM_noise.mean()), 0.0)
    sq = np.sort(off ** 2)[::-1]; cs = np.cumsum(sq) / sq.sum()
    G = len(off)
    # cohort x stage interaction: deviations of per-stage offsets from their mean
    inter = d - off
    inter_noise = (vM[None] / nM[:, None] + vS[None] / nS[:, None]).mean(0) * (3 / 4)
    ms_inter = float((inter ** 2).mean()); ms_inter_c = max(ms_inter - float(inter_noise.mean()), 0.0)
    o = np.argsort(-np.abs(tt))[:top]
    res = {
        "n_genes": G, "n_seaad": int(len(PS)), "n_madrc": int(len(PM)),
        "rms_offset": f_(np.sqrt(ms_off)), "rms_offset_noise_expected": f_(np.sqrt(ms_noise)),
        "rms_offset_noise_corrected": f_(np.sqrt(corr_ms)),
        "mean_offset": f_(off.mean()),
        "rms_between_donor_sd_within_cohort_stage": f_(np.sqrt((sigma ** 2).mean())),
        "ratio_offset_to_donor_sd": f_(np.sqrt(corr_ms) / np.sqrt((sigma ** 2).mean())),
        "median_abs_offset_over_sd": f_(np.median(np.abs(off) / np.maximum(sigma, 1e-9))),
        "frac_genes_abs_offset_gt_1sd": f_((np.abs(off) > sigma).mean()),
        "frac_genes_abs_offset_gt_2sd": f_((np.abs(off) > 2 * sigma).mean()),
        "n_genes_fdr05": int((q < .05).sum()), "n_genes_fdr05_and_gt_1sd": int(((q < .05) & (np.abs(off) > sigma)).sum()),
        "n_genes_abs_offset_gt_0.5": int((np.abs(off) > .5).sum()), "n_genes_abs_offset_gt_1.0": int((np.abs(off) > 1).sum()),
        "share_sq_offset_top1pct": f_(cs[max(0, int(np.ceil(.01 * G)) - 1)]),
        "share_sq_offset_top5pct": f_(cs[max(0, int(np.ceil(.05 * G)) - 1)]),
        "share_sq_offset_top10pct": f_(cs[max(0, int(np.ceil(.10 * G)) - 1)]),
        "n_genes_for_50pct_sq_offset": int(np.searchsorted(cs, .5) + 1),
        "n_genes_for_80pct_sq_offset": int(np.searchsorted(cs, .8) + 1),
        "seaad_hn": {"rms": f_(np.sqrt(ms_hnS)), "rms_noise_corrected": f_(np.sqrt(ms_hnS_c)),
                     "rms_noise_expected": f_(np.sqrt(hnS_noise.mean()))},
        "madrc_hn": {"rms": f_(np.sqrt(ms_hnM)), "rms_noise_corrected": f_(np.sqrt(ms_hnM_c)),
                     "rms_noise_expected": f_(np.sqrt(hnM_noise.mean()))},
        "ratio_offset_to_seaad_hn_raw": f_(np.sqrt(ms_off / ms_hnS)),
        "ratio_offset_to_seaad_hn_noise_corrected": f_(np.sqrt(corr_ms / ms_hnS_c)) if ms_hnS_c > 0 else None,
        "ratio_offset_to_madrc_hn_noise_corrected": f_(np.sqrt(corr_ms / ms_hnM_c)) if ms_hnM_c > 0 else None,
        "frac_genes_abs_offset_gt_abs_seaad_hn": f_((np.abs(off) > np.abs(hnS)).mean()),
        "per_stage_offset_rms": {MX_STAGE[s]: f_(np.sqrt((d[s] ** 2).mean())) for s in stages},
        "per_stage_offset_corr_notad_vs_high": f_(np.corrcoef(d[0], d[3])[0, 1]),
        "cohort_x_stage_interaction_rms_noise_corrected": f_(np.sqrt(ms_inter_c)),
        "top_genes": [{"gene": str(genes[i]), "offset": f_(off[i]), "sd": f_(sigma[i]), "t": f_(tt[i]), "q": f_(q[i]),
                       "seaad_hn": f_(hnS[i])} for i in o],
    }
    return res, off, sigma, q


MX_STAGE = ["Not AD", "Low", "Intermediate", "High"]


def two_group_offset(PS, sS, PX, gX):
    """Offset of a cohort with binary groups gX (0 control-like, 1 AD-like)
    against SEA-AD (Not AD+Low vs Intermediate+High), averaged over groups."""
    a = PX[gX == 0].mean(0) - PS[sS <= 1].mean(0)
    b = PX[gX == 1].mean(0) - PS[sS >= 2].mean(0)
    return (a + b) / 2


# ============================================================================ part 3 predictors
RIDGE_ALPHAS = np.logspace(-2, 4, 13)


def ridge_design(stage, male, apoe4, age, age_mean, age_sd):
    onehot = np.stack([(stage == s).astype(float) for s in (1, 2, 3)], 1)
    return np.column_stack([onehot, stage / 3.0, male, apoe4, (age - age_mean) / age_sd])


def ridge_fit(X, Y, alpha):
    xm = X.mean(0); xs = X.std(0); xs = np.where(xs > 0, xs, 1.0)
    Z = (X - xm) / xs
    ym = Y.mean(0)
    B = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (Y - ym))
    return B, ym, xm, xs


def choose_alpha(X, Y):
    n = len(X); loo = []
    for a in RIDGE_ALPHAS:
        err = 0.0
        for i in range(n):
            keep = np.arange(n) != i
            B, b0, xm, xs = ridge_fit(X[keep], Y[keep], a)
            err += float(((((X[i] - xm) / xs) @ B + b0 - Y[i]) ** 2).mean())
        loo.append(err / n)
    return RIDGE_ALPHAS[int(np.argmin(loo))]


def stage_mean_profiles(L, st):
    grand = L.mean(0)
    return {s: (L[st == s].mean(0) if (st == s).any() else grand) for s in range(4)}


def shrunk_profiles(L, st):
    grand = L.mean(0)
    S = [s for s in range(4) if (st == s).any()]
    sm = {s: L[st == s].mean(0) for s in S}
    n_s = {s: int((st == s).sum()) for s in S}
    resid_ss = sum(((L[st == s] - sm[s]) ** 2).sum(0) for s in S)
    sigma2 = resid_ss / max(len(L) - len(S), 1)
    delta = {s: sm[s] - grand for s in S}
    tau2 = np.maximum(0.0, np.mean([delta[s] ** 2 - sigma2 / n_s[s] for s in S], axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        B = {s: np.where(tau2 + sigma2 / n_s[s] > 0, tau2 / (tau2 + sigma2 / n_s[s]), 0.0) for s in S}
    return {s: grand + (B[s] * delta[s] if s in S else 0.0) for s in range(4)}, float(np.mean([B[s].mean() for s in S]))


def cohort_offset(Ytr, sTr, E, sE, covTr=None, covE=None):
    """Per-gene cohort coefficient of OLS on [stage one-hot (4), cohort (+cov)]."""
    st = np.concatenate([sTr, sE]); coh = np.r_[np.zeros(len(Ytr)), np.ones(len(E))]
    X = np.column_stack([np.stack([(st == s).astype(float) for s in range(4)], 1), coh])
    if covTr is not None:
        cv = np.vstack([covTr, covE]); cv = (cv - cv.mean(0)) / np.where(cv.std(0) > 0, cv.std(0), 1)
        X = np.column_stack([X, cv])
    beta = np.linalg.lstsq(X, np.vstack([Ytr, E]), rcond=None)[0]
    return beta[4]


def predict_fold(Ytr, sTr, cTr, Yho, sHo, cHo, E, sE, cE, age_mean, age_sd):
    """Returns {name: (pred_train [n_tr,G], pred_heldout [n_ho,G], info)}.
    c* = (male, apoe4, age) arrays."""
    out = {}
    trainsets = {"seaad": None}
    off_add = E.mean(0) - Ytr.mean(0)
    trainsets["ext_none"] = E
    trainsets["ext_add"] = E - off_add
    off_sa = cohort_offset(Ytr, sTr, E, sE)
    trainsets["ext_stageadj"] = E - off_sa
    covT = np.column_stack(cTr); covE = np.column_stack(cE)
    off_sac = cohort_offset(Ytr, sTr, E, sE, covT, covE)
    trainsets["ext_stageadj_cov"] = E - off_sac
    offsets_info = {"rms_ext_add": f_(np.sqrt((off_add ** 2).mean())), "rms_ext_stageadj": f_(np.sqrt((off_sa ** 2).mean())),
                    "rms_ext_stageadj_cov": f_(np.sqrt((off_sac ** 2).mean())),
                    "corr_add_vs_stageadj": f_(np.corrcoef(off_add, off_sa)[0, 1])}
    for tsname, Eadj in trainsets.items():
        if Eadj is None:
            L, st = Ytr, sTr
            cov = covT
        else:
            L, st = np.vstack([Ytr, Eadj]), np.r_[sTr, sE]
            cov = np.vstack([covT, covE])
        sm = stage_mean_profiles(L, st)
        out[f"stage_mean|{tsname}"] = (np.stack([sm[s] for s in sTr]), np.stack([sm[s] for s in sHo]), {})
        sh, bmean = shrunk_profiles(L, st)
        out[f"shrunk|{tsname}"] = (np.stack([sh[s] for s in sTr]), np.stack([sh[s] for s in sHo]),
                                   {"mean_shrinkage_factor": bmean})
        Xall = ridge_design(st, cov[:, 0], cov[:, 1], cov[:, 2], age_mean, age_sd)
        alpha = choose_alpha(Xall, L)
        B, b0, xm, xs = ridge_fit(Xall, L, alpha)
        Xtr = ridge_design(sTr, *cTr, age_mean, age_sd); Xho = ridge_design(sHo, *cHo, age_mean, age_sd)
        out[f"ridge|{tsname}"] = (((Xtr - xm) / xs) @ B + b0, ((Xho - xm) / xs) @ B + b0, {"alpha": float(alpha)})
        # covariates-only ridge (no stage columns): separates the covariate gain from the stage gain
        cs = slice(4, 7)
        alpha_c = choose_alpha(Xall[:, cs], L)
        Bc, b0c, xmc, xsc = ridge_fit(Xall[:, cs], L, alpha_c)
        out[f"ridgecov|{tsname}"] = (((Xtr[:, cs] - xmc) / xsc) @ Bc + b0c, ((Xho[:, cs] - xmc) / xsc) @ Bc + b0c,
                                     {"alpha": float(alpha_c)})
    # pure transfer: SEA-AD train mean + MADRC stage deviations re-centred on SEA-AD train stage mix
    pi = np.array([(sTr == s).mean() for s in range(4)])
    base = Ytr.mean(0)
    for nm, prof in (("ext_delta", stage_mean_profiles(E, sE)), ("ext_delta_shrunk", shrunk_profiles(E, sE)[0])):
        cen = sum(pi[s] * prof[s] for s in range(4))
        pr = {s: base + prof[s] - cen for s in range(4)}
        out[f"stage_mean|{nm}"] = (np.stack([pr[s] for s in sTr]), np.stack([pr[s] for s in sHo]), {})
    return out, offsets_info


def run_part3(S, M, folds, spaces, n_perm):
    """Part 3 (see module docstring)."""
    # ================================================================ PART 3
    t3 = time.time()
    p3 = {"folds": {}, "notes": "pb_rmse on held-out SEA-AD donors, hvg2000 fold-k panel; every fit uses fold-k SEA-AD train donors "
                                "(+ MADRC); nothing derived from held-out donors enters any prediction"}
    xf_in = {}
    perdonor = {}
    for f in folds:
        k = f["inner_fold"]; rep_ = f.get("repeat", 0)
        gidx = spaces[k]["hvg"]
        tr = [S.donors.index(d) for d in f["train_donors"]]; ho = [S.donors.index(d) for d in f["heldout_donors"]]
        sumS = panel_sums(S, gidx)
        PS_ = MX.log_cp10k(sumS); PM_ = pb_profiles(M, gidx)
        Ytr, Yho = PS_[tr], PS_[ho]; sTr, sHo = S.stage[tr], S.stage[ho]
        cTr = (S.male[tr], S.apoe4[tr], S.age[tr]); cHo = (S.male[ho], S.apoe4[ho], S.age[ho])
        # review F5: MADRC '90+' -> mean age of THIS fold's SEA-AD train donors aged >= 90
        old = S.age[tr][S.age[tr] >= 90]
        ageE = np.where(M.extra["age_censored"], old.mean(), M.age)
        cE = (M.male, M.apoe4, ageE)
        preds, offinfo = predict_fold(Ytr, sTr, cTr, Yho, sHo, cHo, PM_, M.stage, cE, spaces[k]["age_mean"], spaces[k]["age_sd"])
        grand = MX.log_cp10k(sumS[tr].sum(0))
        preds["grand_pooled|ref"] = (np.tile(grand, (len(tr), 1)), np.tile(grand, (len(ho), 1)), {})
        preds["train_mean|ref"] = (np.tile(Ytr.mean(0), (len(tr), 1)), np.tile(Ytr.mean(0), (len(ho), 1)), {})
        gbar = Ytr.mean(0)
        cH = MX.stage_contrasts(Yho, sHo)
        fold_res = {"repeat": rep_, "n_genes": int(len(gidx)), "cohort_offsets": offinfo, "units": {}}
        for nm, (ptr, pho, info) in preds.items():
            rm = np.sqrt(((pho - Yho) ** 2).mean(1))
            u = {"pb_rmse": f_(rm.mean()), "pb_rmse_by_stage": {str(s): f_(rm[sHo == s].mean()) for s in np.unique(sHo)}, **info}
            cm = MX.stage_contrasts(pho, sHo)
            if not nm.endswith("|ref"):
                u["hn_vs_heldout"] = (MX.agreement(cm["high_notad"], cH["high_notad"]) or {}).get("spearman")
                u["slope_vs_heldout"] = (MX.agreement(cm["slope"], cH["slope"]) or {}).get("spearman")
            fold_res["units"][nm] = u
            perdonor.setdefault(nm, {}).setdefault(rep_, {}).update({S.donors[i]: float(r) for i, r in zip(ho, rm)})
            # xfit material
            mbar = ptr.mean(0)
            cand = np.full((4, len(gidx)), np.nan)
            for s in range(4):
                if (sTr == s).any():
                    cand[s] = ptr[sTr == s].mean(0) - mbar
            xf_in.setdefault(nm, {}).setdefault(rep_, []).append(
                {"obs": Yho - gbar, "cand": cand, "assign": sHo, "group": sHo, "donors": [S.donors[i] for i in ho]})
        p3["folds"][str(k)] = fold_res
        print(f"[part3] fold {k}: grand {fold_res['units']['grand_pooled|ref']['pb_rmse']:.4f} "
              f"shrunk seaad {fold_res['units']['shrunk|seaad']['pb_rmse']:.4f} shrunk ext_stageadj {fold_res['units']['shrunk|ext_stageadj']['pb_rmse']:.4f} "
              f"({time.time() - t3:.0f}s)", flush=True)
    units = list(perdonor)
    reps = sorted({f.get("repeat", 0) for f in folds})
    agg = {}
    for nm in units:
        a_ = {}
        fold_rm = [p3["folds"][str(f["inner_fold"])]["units"][nm]["pb_rmse"] for f in folds]
        a_["pb_rmse_mean_over_folds"] = f_(np.mean(fold_rm))
        a_["pb_rmse_folds0_2"] = f_(np.mean(fold_rm[:3]))
        a_["per_fold_pb_rmse"] = fold_rm
        hn = [p3["folds"][str(f["inner_fold"])]["units"][nm].get("hn_vs_heldout") for f in folds]
        a_["hn_vs_heldout_mean"] = MX.nanmean(hn)
        sl = [p3["folds"][str(f["inner_fold"])]["units"][nm].get("slope_vs_heldout") for f in folds]
        a_["slope_vs_heldout_mean"] = MX.nanmean(sl)
        if not nm.endswith("|ref"):
            xs = {}
            for r in reps:
                fl = xf_in[nm][r]
                if len(fl) < 3:
                    continue
                xsr = MX.xfit_score(fl, n_perm=n_perm, seed=MX.stable_seed(SEED, "xfit", r))
                xs[str(r)] = {"score": xsr["score"], "pooled": xsr["pooled_pairs"], "by_stage": xsr["by_group"],
                              "per_donor": dict(zip(sum([x["donors"] for x in fl], []), xsr["per_donor"]))}
            a_["xfit"] = xs
            a_["xfit_score_mean_over_repeats"] = MX.nanmean([v["score"]["value"] for v in xs.values()])
            a_["xfit_z_mean_over_repeats"] = MX.nanmean([v["score"].get("z") for v in xs.values()])
        agg[nm] = a_
    # paired differences vs the SEA-AD-only fit of the same predictor, and vs grand
    paired = {}
    for nm in units:
        pred_name, ts = nm.split("|")
        refs = []
        if ts.startswith("ext"):
            refs.append(f"{'stage_mean' if pred_name == 'stage_mean' else pred_name}|seaad")
        refs += ["grand_pooled|ref", "train_mean|ref"]
        if pred_name == "ridge":
            refs.append(f"ridgecov|{ts}")
        for ref in refs:
            if ref == nm or ref not in perdonor:
                continue
            per_rep = []
            for r in reps:
                d1, d0 = perdonor[nm].get(r, {}), perdonor[ref].get(r, {})
                common = sorted(set(d1) & set(d0))
                dd = np.array([d1[d] - d0[d] for d in common])
                if len(dd):
                    per_rep.append({"repeat": r, "n_donors": len(dd), "mean": float(dd.mean()), "se_donor": float(dd.std(ddof=1) / np.sqrt(len(dd))),
                                    "n_better": int((dd < 0).sum())})
            fd = np.array([p3["folds"][str(f["inner_fold"])]["units"][nm]["pb_rmse"] - p3["folds"][str(f["inner_fold"])]["units"][ref]["pb_rmse"]
                           for f in folds])
            frep = np.array([f.get("repeat", 0) for f in folds])
            e = {"pb_rmse_diff_per_repeat": per_rep,
                 "pb_rmse_diff_mean": f_(np.mean([x["mean"] for x in per_rep])) if per_rep else None,
                 "pb_rmse_diff_se_donor_avg": f_(np.mean([x["se_donor"] for x in per_rep])) if per_rep else None,
                 "per_fold_diff": [f_(x) for x in fd], "folds_better": int((fd < 0).sum()), "n_folds": int(len(fd)),
                 "fold_se_within_repeat": f_(np.mean([fd[frep == r].std(ddof=1) / np.sqrt((frep == r).sum())
                                                      for r in reps if (frep == r).sum() > 1])) if len(fd) > 1 else None,
                 "fold_se_note": "SE over the 3 folds of one repeat, averaged over repeats (folds 0-8 reuse the same 37 donors; "
                                 "review F6)"}
            if "xfit" in agg[nm] and "xfit" in agg.get(ref, {}):
                xd = []
                for r in agg[nm]["xfit"]:
                    if r in agg[ref]["xfit"]:
                        a1 = agg[nm]["xfit"][r]["per_donor"]; a0 = agg[ref]["xfit"][r]["per_donor"]
                        dd = np.array([a1[d] - a0[d] for d in a1 if a1[d] is not None and a0.get(d) is not None])
                        if len(dd):
                            xd.append({"repeat": int(r), "mean": float(dd.mean()), "se_donor": float(dd.std(ddof=1) / np.sqrt(len(dd)))})
                e["xfit_score_diff_per_repeat"] = xd
                e["xfit_score_diff_mean"] = f_(np.mean([x["mean"] for x in xd])) if xd else None
            paired[f"{nm} vs {ref}"] = e
    p3["aggregate"] = agg
    p3["paired"] = paired
    p3["repeats"] = reps
    return p3
    print(f"[part3] done {time.time() - t3:.0f}s", flush=True)



def madrc_qc_trend(M):
    """Review D2: MADRC library quality / yield / rejection terms vs ADNC stage (per donor, from receipts)."""
    dj = json.loads((EXT / "GSE268599" / "donors.json").read_text())["donors"]
    keys = ["n_barcodes", "qc_pass_frac", "cells_kept", "kept_frac_of_barcodes", "median_mito_kept",
            "kept_frac_of_myeloid_cluster", "rej_other_lineage_frac", "rej_hetero_doublet_frac", "rej_t_or_b_frac",
            "rej_mg_low_frac", "ambient_rule_cluster_cells"]
    vals = {k: [] for k in keys}
    for i, d in enumerate(M.donors):
        libs = dj[d]["selection_receipt"]["libraries"]
        nb = sum(l["n_barcodes"] for l in libs); nq = sum(l["n_qc_pass"] for l in libs)
        nm = sum(l["n_myeloid_cluster"] for l in libs); nk = sum(l["n_kept"] for l in libs)
        rej = {r: sum(l["rejected"][r] for l in libs) for r in libs[0]["rejected"]}
        cm = np.load(EXT / "GSE268599" / dj[d]["cellmeta_file"])
        vals["n_barcodes"].append(nb); vals["qc_pass_frac"].append(nq / nb); vals["cells_kept"].append(nk)
        vals["kept_frac_of_barcodes"].append(nk / nb); vals["median_mito_kept"].append(float(np.median(cm["mito"])))
        vals["kept_frac_of_myeloid_cluster"].append(nk / nm)
        vals["rej_other_lineage_frac"].append(rej["other_lineage_dominates"] / nm)
        vals["rej_hetero_doublet_frac"].append(rej["hetero_doublet"] / nm)
        vals["rej_t_or_b_frac"].append(rej["t_or_b"] / nm); vals["rej_mg_low_frac"].append(rej["mg_lt_min"] / nm)
        vals["ambient_rule_cluster_cells"].append(sum(c["n"] for l in libs for c in l["clusters"]
                                                      if c.get("myeloid_rule") == "ambient_ast"))
    return {k: stage_trend(v, M.stage, label="qc" + k) for k, v in vals.items()}


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--quick", action="store_true", help="few permutations, folds 0-2 only (smoke)")
    ap.add_argument("--part3-only", action="store_true", help="only part 3 -> results_part3.json")
    a = ap.parse_args()
    n_perm = 200 if a.quick else a.n_perm
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    dev, folds = folds_all()
    if a.quick:
        folds = folds[:3]
    S = load_seaad(dev)
    M, genes = load_madrc()
    I, present_I, genes_I = load_imp()
    genes_S = np.load(REBUILT / "counts" / "genes.npy", allow_pickle=False)
    assert np.array_equal(genes, genes_S) and np.array_equal(genes_I, genes_S)
    print(f"[load] {time.time() - t0:.0f}s SEA-AD {len(S.donors)} donors {S.ncell.sum()} cells; "
          f"MADRC {len(M.donors)} / {M.ncell.sum()}; GSE264648 {len(I.donors)} / {I.ncell.sum()}", flush=True)

    if a.part3_only:
        spaces = {}
        for f in folds:
            sz = np.load(CACHE / f"fold{f['inner_fold']}" / "space.npz", allow_pickle=False)
            spaces[f["inner_fold"]] = dict(hvg=sz["hvg"], age_mean=float(sz["age_mean"]), age_sd=float(sz["age_sd"]))
        r3 = {"script": str(Path(__file__).resolve()), "script_sha256": sha256(__file__), "n_perm": n_perm,
              "part3_prediction": run_part3(S, M, folds, spaces, n_perm),
              "donor_guard": {"seaad_count_files_read": DONOR_GUARD, "all_development": all(d in dev for d in DONOR_GUARD)}}
        (OUT / "results_part3.json").write_text(json.dumps(clean(r3), indent=1, allow_nan=False))
        print(f"[done] part3 {time.time() - t0:.0f}s", flush=True)
        return
    res = {"script": str(Path(__file__).resolve()), "script_sha256": sha256(__file__), "seed": SEED,
           "n_perm": n_perm, "quick": a.quick, "definitions": __doc__}
    # ---------------------------------------------------------------- cohort description
    allS, allM, allI = np.concatenate(S.umi), np.concatenate(M.umi), np.concatenate(I.umi)
    res["cohorts"] = {
        c.name: {"n_donors": len(c.donors), "n_cells": int(c.ncell.sum()),
                 "stage_counts": {str(s): int((c.stage == s).sum()) for s in np.unique(c.stage)},
                 "umi_per_cell_pct_5_25_50_75_95": [float(x) for x in np.percentile(u, [5, 25, 50, 75, 95])],
                 "donor_median_umi_range": [float(c.depth.min()), float(c.depth.max())],
                 "cells_per_donor_range": [int(c.ncell.min()), int(c.ncell.max())],
                 "braak_by_stage": {str(s): sorted(int(b) for b in c.braak[c.stage == s]) for s in np.unique(c.stage)},
                 "apoe4_by_stage": {str(s): [float(b) for b in c.apoe4[c.stage == s]] for s in np.unique(c.stage)}}
        for c, u in ((S, allS), (M, allM), (I, allI))}
    res["cohorts"]["GSE264648"]["stage_note"] = "stage here = diagnosis (0 Control, 1 AD); no ADNC available"
    res["madrc_qc_stage_trend"] = madrc_qc_trend(M)

    # ---------------------------------------------------------------- panels
    detS, detM = detection(S), detection(M)
    neutral = np.where((detS >= NEUTRAL_DET) & (detM >= NEUTRAL_DET))[0]
    spaces = {}
    for f in folds:
        k = f["inner_fold"]
        sz = np.load(CACHE / f"fold{k}" / "space.npz", allow_pickle=False)
        assert np.array_equal(genes_S[sz["hvg"]], sz["genes"])
        spaces[k] = dict(hvg=sz["hvg"], age_mean=float(sz["age_mean"]), age_sd=float(sz["age_sd"]))
    res["panels"] = {"neutral_rule": f"detected in >= {NEUTRAL_DET:.0%} of pooled cells in both SEA-AD (37 dev donors) and MADRC",
                     "neutral_n_genes": int(len(neutral)),
                     "neutral_n_present_gse264648": int(present_I[neutral].sum()),
                     "hvg2000_in_neutral": {str(k): int(np.isin(v["hvg"], neutral).sum()) for k, v in spaces.items()}}
    print(f"[panels] neutral {len(neutral)} genes", flush=True)

    # ================================================================ PART 1
    t1 = time.time()
    thinS, p_thin = thin_to(S, allM, SEED)
    umi_thin = np.concatenate([np.asarray(x.sum(1)).ravel() for x in thinS])
    PS_ty, PM_ty, mixS, clf_info = supertype_probs(S, M, neutral)
    ambS = marker_scores(S, genes_S); ambM = marker_scores(M, genes_S)
    # check own ambient scores against the fetch task's MADRC cellmeta values
    amb_chk = float(np.max(np.abs(np.concatenate(ambM) - np.concatenate(M.extra["ambient_fetch"]))))
    pvm_chk = float(np.corrcoef(np.concatenate([p[:, 0] for p in PM_ty]), np.concatenate(M.extra["pvm_prob_fetch"]))[0, 1])
    keepS = [((a_[:, 0] <= AMBIENT_THR) & (a_[:, 1] <= AMBIENT_THR)).astype(float) for a_ in ambS]
    keepM = [((a_[:, 0] <= AMBIENT_THR) & (a_[:, 1] <= AMBIENT_THR)).astype(float) for a_ in ambM]
    p1 = {"depth_matching": {"seaad_umi_pct_before": [float(x) for x in np.percentile(allS, [5, 25, 50, 75, 95])],
                             "seaad_umi_pct_after": [float(x) for x in np.percentile(umi_thin, [5, 25, 50, 75, 95])],
                             "madrc_umi_pct": [float(x) for x in np.percentile(allM, [5, 25, 50, 75, 95])],
                             "mean_keep_prob": float(p_thin.mean())},
          "supertype_classifier": clf_info,
          "check_own_ambient_scores_vs_fetch_cellmeta_max_abs_diff": amb_chk,
          "check_pvm_prob_corr_own_vs_fetch": pvm_chk,
          "ambient_filter": {"rule": f"astrocyte and neuronal marker scores <= {AMBIENT_THR}",
                             "frac_cells_kept_seaad": float(np.concatenate(keepS).mean()),
                             "frac_cells_kept_madrc": float(np.concatenate(keepM).mean()),
                             "min_cells_kept_per_donor_seaad": int(min(k_.sum() for k_ in keepS)),
                             "min_cells_kept_per_donor_madrc": int(min(k_.sum() for k_ in keepM)),
                             "median_ast_seaad": float(np.median(np.concatenate(ambS)[:, 0])),
                             "median_ast_madrc": float(np.median(np.concatenate(ambM)[:, 0])),
                             "median_neu_seaad": float(np.median(np.concatenate(ambS)[:, 1])),
                             "median_neu_madrc": float(np.median(np.concatenate(ambM)[:, 1]))},
          "panels": {}}
    wpvmS = [1 - p[:, 0] for p in PS_ty]; wpvmM = [1 - p[:, 0] for p in PM_ty]
    panel_list = {"neutral": neutral}
    for k in (0, 1, 2):
        if k in spaces:
            panel_list[f"hvg2000_fold{k}"] = spaces[k]["hvg"]
    keep_off = {}
    for pname, gidx in panel_list.items():
        steps = {}
        prof = {}
        prof["S0_raw"] = (pb_profiles(S, gidx), pb_profiles(M, gidx))
        prof["S1_depth"] = (pb_profiles(S, gidx, Xs=thinS), prof["S0_raw"][1])
        prof["S2_depth_pvm_removed"] = (pb_profiles(S, gidx, wpvmS, Xs=thinS), pb_profiles(M, gidx, wpvmM))
        mS_, fbS = mix_profiles(S, gidx, PS_ty, mixS, Xs=thinS)
        mM_, fbM = mix_profiles(M, gidx, PM_ty, mixS)
        prof["S3_depth_mix_std"] = (mS_, mM_)
        mS2, fbS2 = mix_profiles(S, gidx, [p * k_[:, None] for p, k_ in zip(PS_ty, keepS)], mixS, Xs=thinS)
        mM2, fbM2 = mix_profiles(M, gidx, [p * k_[:, None] for p, k_ in zip(PM_ty, keepM)], mixS)
        prof["S4_depth_mix_ambient"] = (mS2, mM2)
        prof["meanlog_S0_raw"] = (meanlog_profiles(S, gidx), meanlog_profiles(M, gidx))
        prof["meanlog_S1_depth"] = (meanlog_profiles(S, gidx, Xs=thinS), prof["meanlog_S0_raw"][1])
        offs = {}
        for st, (A, B) in prof.items():
            r, off, sig, q = offset_stats(A, S.stage, B, M.stage, genes_S[gidx])
            steps[st] = r; offs[st] = off
        steps["mix_fallback_type_slots"] = {"S3": [fbS, fbM], "S4": [fbS2, fbM2]}
        steps["S1_note"] = ("binomial thinning keeps expected pseudobulk proportions, so S1 only checks the log1p "
                            "nonlinearity; depth evidence is depth_regression (review F3)")
        steps["depth_regression"] = depth_regression(prof["S0_raw"][0], S.stage, S.depth, prof["S0_raw"][1], M.stage,
                                                     M.depth, offs["S0_raw"])
        base = offs["S0_raw"]
        steps["offset_corr_with_S0"] = {st: f_(np.corrcoef(base, o)[0, 1]) for st, o in offs.items() if not st.startswith("meanlog")}
        ms0 = steps["S0_raw"]["rms_offset_noise_corrected"] ** 2
        steps["share_of_S0_sq_offset_removed"] = {st: f_(1 - steps[st]["rms_offset_noise_corrected"] ** 2 / ms0) if ms0 > 0 else None
                                                  for st in prof if not st.startswith("meanlog")}
        p1["panels"][pname] = steps
        keep_off[pname] = offs
        print(f"[part1] {pname}: rms offset S0 {steps['S0_raw']['rms_offset_noise_corrected']:.3f} "
              f"S1 {steps['S1_depth']['rms_offset_noise_corrected']:.3f} S3 {steps['S3_depth_mix_std']['rms_offset_noise_corrected']:.3f} "
              f"S4 {steps['S4_depth_mix_ambient']['rms_offset_noise_corrected']:.3f} ({time.time() - t1:.0f}s)", flush=True)
    # ambient-gene share of the offset on the neutral panel
    ix_n = {g: i for i, g in enumerate(genes_S[neutral])}
    amb_genes = [g for v in AMBIENT_MARKERS.values() for g in v if g in ix_n]
    off0 = keep_off["neutral"]["S0_raw"]
    p1["panels"]["neutral"]["ambient_marker_genes"] = {g: f_(off0[ix_n[g]]) for g in amb_genes}
    # GSE264648 two-group offsets on neutral & present genes
    gI = neutral[present_I[neutral]]
    PS_I, PM_I, PI_I = pb_profiles(S, gI), pb_profiles(M, gI), pb_profiles(I, gI)
    oM = two_group_offset(PS_I, S.stage, PM_I, (M.stage >= 2).astype(int))
    oI = two_group_offset(PS_I, S.stage, PI_I, I.stage)
    oMI = two_group_offset(PM_I[:, :], (M.stage >= 2).astype(int) * 3, PI_I, I.stage)  # MADRC as reference
    # noise level of a two-group offset: split each cohort's donors in random halves
    p1["gse264648"] = {
        "n_genes": int(len(gI)), "note": "two-group matched offsets: (control-like, AD-like) = SEA-AD (Not AD+Low, Int+High), "
                                         "MADRC same, GSE264648 (Control, AD)",
        "rms_offset_madrc_vs_seaad": f_(np.sqrt((oM ** 2).mean())), "rms_offset_gse264648_vs_seaad": f_(np.sqrt((oI ** 2).mean())),
        "rms_offset_gse264648_vs_madrc": f_(np.sqrt((oMI ** 2).mean())),
        "corr_offsets_madrc_and_gse264648_both_vs_seaad": f_(np.corrcoef(oM, oI)[0, 1]),
        "spearman_offsets_madrc_and_gse264648_both_vs_seaad": MX.spearman(oM, oI),
        "median_umi_per_cell": float(np.median(allI)),
        "interpretation_hint": "a high correlation means SEA-AD differs from both other cohorts in the same genes "
                               "(SEA-AD-specific, e.g. its neuronal ambient RNA or MTG-vs-sorting); a low one means "
                               "the MADRC offset is MADRC-specific"}
    # review D3: ambient astrocyte detection vs stage, per cohort
    ixg = {g: i for i, g in enumerate(genes_S)}
    amb_tr = {}
    for cname, C_ in (("seaad", S), ("madrc", M)):
        blk = {}
        for g in ("SLC1A2", "AQP4", "GFAP", "ALDH1L1", "ATP1A2"):
            j = ixg[g]
            nz = [float((x[:, j].toarray().ravel() > 0).mean()) for x in C_.X]
            blk[f"{g}_nonzero_frac"] = stage_trend(nz, C_.stage, label=f"amb{cname}{g}")
        aj = [ixg[g] for g in AST_AMBIENT_GENES]
        cp = [float(np.log1p(np.asarray(x[:, aj].sum(1)).ravel() / u * 1e4).mean()) for x, u in zip(C_.X, C_.umi)]
        blk["ast_list_log1p_cp10k_mean"] = stage_trend(cp, C_.stage, label=f"amb{cname}list")
        amb_tr[cname] = blk
    res["ambient_stage_trend"] = amb_tr
    res["part1_offset"] = p1
    print(f"[part1] done {time.time() - t1:.0f}s; GSE264648 corr {p1['gse264648']['corr_offsets_madrc_and_gse264648_both_vs_seaad']}", flush=True)

    # ================================================================ PART 2
    t2 = time.time()
    kinds = ["hn", "hnl", "slope", "braak_slope"]
    # review F4: PVM fraction per donor from the part-1 supertype classifier (OOF for SEA-AD)
    pvmS = np.array([p[:, 0].mean() for p in PS_ty]); pvmM = np.array([p[:, 0].mean() for p in PM_ty])
    covS = {"_adj": cov_matrix(S), "_adjpvm": cov_matrix(S, extra=pvmS)}
    covM = {"_adj": cov_matrix(M), "_adjpvm": cov_matrix(M, extra=pvmM)}
    p2 = {"covariates_seaad": "std age, male, APOE4 dose, log median UMI", "covariates_madrc": "std age, male, APOE4 dose, log median UMI",
          "covariates_adjpvm": "the _adj covariates + donor PVM fraction (mean classifier P(Micro-PVM_1))",
          "pvm_fraction_by_stage": {"seaad": [f_(pvmS[S.stage == s].mean()) for s in range(4)],
                                    "madrc": [f_(pvmM[M.stage == s].mean()) for s in range(4)]},
          "apoe4_by_stage": {"seaad": [f_(S.apoe4[S.stage == s].mean()) for s in range(4)],
                             "madrc": [f_(M.apoe4[M.stage == s].mean()) for s in range(4)]},
          "panels": {}}
    # review D3: panels without astrocyte ambient genes / without any ambient-like genes
    panel_list2 = dict(panel_list)
    amb_set = set(AST_AMBIENT_GENES)
    neu_set = set(AMBIENT_MARKERS["neu"])
    for k in (0, 1, 2):
        nm = f"hvg2000_fold{k}"
        if nm not in panel_list:
            continue
        g = panel_list[nm]
        gs = genes_S[g]
        noast = np.array([x not in amb_set for x in gs])
        off0 = np.abs(keep_off[nm]["S0_raw"])
        top20 = off0 >= np.quantile(off0, 0.8)
        panel_list2[nm + "_noast"] = g[noast]
        panel_list2[nm + "_noamb"] = g[noast & ~top20 & np.array([x not in neu_set for x in gs])]
    p2["ambient_panels"] = {k: {"n_genes": int(len(v)), "removed": [str(x) for x in sorted(set(genes_S[panel_list[k.rsplit('_', 1)[0]]]) - set(genes_S[v]))][:60]}
                            for k, v in panel_list2.items() if k.endswith(("_noast", "_noamb"))}
    rel_cache = {}
    for pname, gidx in panel_list2.items():
        PS_, PM_ = pb_profiles(S, gidx), pb_profiles(M, gidx)
        rep = replicate(PS_, S.stage, S.braak, covS, PM_, M.stage, M.braak, covM, kinds, n_perm, SEED, pname)
        relS = reliability(PS_, S.stage, S.braak, kinds, SEED, "S" + pname)
        relM = reliability(PM_, M.stage, M.braak, kinds, SEED, "M" + pname)
        for kind in kinds:
            rs, rm = relS[kind].get("spearman_brown"), relM[kind].get("spearman_brown")
            ceil = np.sqrt(rs * rm) if (rs is not None and rm is not None and rs > 0 and rm > 0) else None
            if rep.get(kind, {}).get("defined"):
                rep[kind]["attenuation_ceiling"] = f_(ceil)
                rep[kind]["disattenuated_spearman"] = f_(rep[kind]["spearman"]["value"] / ceil) if ceil else None
        # DAM
        idx, sgn, missing = MX.dam_genes(genes_S[gidx])
        dam = {"n_present": int(len(idx)), "present": [str(genes_S[gidx][i]) for i in idx], "missing": missing}
        if len(idx):
            for cname, P_, st_ in (("seaad", PS_, S.stage), ("madrc", PM_, M.stage)):
                dam[cname] = MX.dam_perm_null(P_, st_, idx, sgn, n_perm, MX.stable_seed(SEED, "dam", pname, cname))
                c_ = contrast_weights("hn", st_) @ P_
                dam[cname]["per_gene_hn"] = {str(genes_S[gidx][i]): f_(c_[i]) for i in idx}
            if len(idx) >= 4:
                rng = np.random.default_rng(MX.stable_seed(SEED, "damx", pname))
                cS = contrast_weights("hn", S.stage) @ PS_[:, idx]; cM = contrast_weights("hn", M.stage) @ PM_[:, idx]
                obs = MX.spearman(cS, cM)
                nM = agree_stats(cS, perm_weights("hn", M.stage, None, None, n_perm, rng) @ PM_[:, idx])["spearman"]
                nS = agree_stats(None, np.broadcast_to(cM, (n_perm, len(idx))),
                                 X_rows=perm_weights("hn", S.stage, None, None, n_perm, rng) @ PS_[:, idx])["spearman"]
                dam["cross_cohort_hn_spearman"] = {"value": f_(obs), "null_perm_madrc": null_block(obs, nM),
                                                   "null_perm_seaad": null_block(obs, nS)}
                # expression-matched random-gene null (competitive, descriptive)
                lvl = PS_.mean(0)
                ns = MX.matched_null_sets(idx, lvl, 2000, MX.stable_seed(SEED, "damnull", pname))
                dam["madrc_hn_vs_matched_random_genes"] = MX.dam_score(contrast_weights("hn", M.stage) @ PM_, idx, sgn, ns,
                                                                     real={"seaad_hn": contrast_weights("hn", S.stage) @ PS_})
                dam["madrc_hn_vs_matched_random_genes"].pop("per_gene_model_contrast", None)
        p2["panels"][pname] = {"agreement": rep, "reliability_seaad": relS, "reliability_madrc": relM, "dam": dam}
        print(f"[part2] {pname}: hn sp {rep['hn']['spearman']['value']:.3f} p_M {rep['hn']['spearman']['null_perm_madrc'].get('p_one_sided')} "
              f"p_S {rep['hn']['spearman']['null_perm_seaad'].get('p_one_sided')}; slope sp {rep['slope']['spearman']['value']:.3f}; "
              f"relS hn {relS['hn'].get('spearman_brown')} relM hn {relM['hn'].get('spearman_brown')} ({time.time() - t2:.0f}s)", flush=True)
    # GSE264648 diagnosis contrast vs SEA-AD and MADRC (descriptive; diagnosis, not ADNC)
    PS_I_, PM_I_ = PS_I, PM_I
    wI = contrast_weights("hn", I.stage * 3)
    cI = wI @ PI_I
    rngI = np.random.default_rng(MX.stable_seed(SEED, "imp"))
    WpI = perm_weights("hn", I.stage * 3, None, None, n_perm, rngI)
    gi = {}
    for nm, P_, st_ in (("seaad_hn", PS_I_, S.stage), ("madrc_hn", PM_I_, M.stage)):
        cx = contrast_weights("hn", st_) @ P_
        o = MX.spearman(cx, cI)
        nul = agree_stats(cx, WpI @ PI_I)["spearman"]
        gi[nm] = null_block(o, nul)
    p2["gse264648_ad_minus_control"] = {"n_genes": int(len(gI)), "spearman_vs": gi,
                                        "null": "GSE264648 diagnosis labels permuted"}
    # ---------------- per fold
    pf = {}
    for f in folds:
        k = f["inner_fold"]
        tr = [S.donors.index(d) for d in f["train_donors"]]; ho = [S.donors.index(d) for d in f["heldout_donors"]]
        detTr = detection(S, tr)
        neutral_k = np.where((detTr >= NEUTRAL_DET) & (detM >= NEUTRAL_DET))[0]
        fk = {}
        for pname, gidx in (("hvg2000", spaces[k]["hvg"]), ("neutral_train", neutral_k)):
            PS_, PM_ = pb_profiles(S, gidx), pb_profiles(M, gidx)
            Ptr, sTr = PS_[tr], S.stage[tr]; Pho, sHo = PS_[ho], S.stage[ho]
            rng = np.random.default_rng(MX.stable_seed(SEED, "fold", k, pname))
            blk = {"n_genes": int(len(gidx)), "n_train_by_stage": [int((sTr == s).sum()) for s in range(4)],
                   "n_heldout_by_stage": [int((sHo == s).sum()) for s in range(4)]}
            off = cohort_offset(Ptr, sTr, PM_, M.stage)
            for kind in ("hn", "hnl", "slope"):
                wT = contrast_weights(kind, sTr); wM = contrast_weights(kind, M.stage); wH = contrast_weights(kind, sHo)
                if wT is None or wM is None:
                    continue
                cT, cM = wT @ Ptr, wM @ PM_
                obs = agree_stats(cT, cM)
                nM = agree_stats(cT, perm_weights(kind, M.stage, None, None, n_perm, rng) @ PM_)
                e = {"madrc_vs_seaad_train": {st: null_block(obs[st][0], nM[st]) for st in ("spearman", "top_decile_sign", "slope")}}
                if wH is not None:
                    cH = wH @ Pho
                    comb_P = np.vstack([Ptr, PM_ - off]); comb_s = np.r_[sTr, M.stage]
                    cC = contrast_weights(kind, comb_s) @ comb_P
                    WpH = perm_weights(kind, sHo, None, None, n_perm, rng)
                    nullH = WpH @ Pho
                    for nm, cx in (("seaad_train", cT), ("madrc", cM), ("combined_stageadj", cC)):
                        o = agree_stats(cH, cx)
                        nl = agree_stats(None, np.broadcast_to(cx, nullH.shape), X_rows=nullH)
                        e[f"{nm}_vs_heldout"] = {st: null_block(o[st][0], nl[st]) for st in ("spearman", "top_decile_sign")}
                blk[kind] = e
            fk[pname] = blk
        pf[str(k)] = fk
        print(f"[part2] fold {k}: hvg hn madrc-vs-train sp {fk['hvg2000']['hn']['madrc_vs_seaad_train']['spearman']['value']:.3f}", flush=True)
    # summary over folds
    summ = {}
    for pname in ("hvg2000", "neutral_train"):
        for kind in ("hn", "hnl", "slope"):
            for nm in ("madrc_vs_seaad_train", "seaad_train_vs_heldout", "madrc_vs_heldout", "combined_stageadj_vs_heldout"):
                vr = [(f.get("repeat", 0), pf[str(f["inner_fold"])][pname][kind][nm]["spearman"]["value"]) for f in folds
                      if kind in pf[str(f["inner_fold"])][pname] and nm in pf[str(f["inner_fold"])][pname][kind]]
                vr = [(r, x) for r, x in vr if x is not None]
                v = [x for _, x in vr]
                if v:
                    # review F6: folds 0-8 are 3 repeats over the same donors -> SE within a repeat (3 folds)
                    byr = {}
                    for r, x in vr:
                        byr.setdefault(r, []).append(x)
                    sew = [np.std(x, ddof=1) / np.sqrt(len(x)) for x in byr.values() if len(x) > 1]
                    summ[f"{pname}|{kind}|{nm}"] = {"mean": f_(np.mean(v)),
                                                    "se_within_repeat": f_(np.mean(sew)) if sew else None,
                                                    "repeat_means": {str(r): f_(np.mean(x)) for r, x in byr.items()},
                                                    "n_folds": len(v), "per_fold": [f_(x) for x in v],
                                                    "note": "folds overlap across repeats; not independent (review F6)"}
    p2["per_fold"] = pf
    p2["per_fold_summary"] = summ
    res["part2_replication"] = p2

    res["part3_prediction"] = run_part3(S, M, folds, spaces, n_perm)

    res["donor_guard"] = {"seaad_count_files_read": DONOR_GUARD, "n": len(DONOR_GUARD),
                          "all_development": all(d in dev for d in DONOR_GUARD)}
    res["wall_seconds"] = time.time() - t0
    name = "results_quick.json" if a.quick else "results.json"
    (OUT / name).write_text(json.dumps(clean(res), indent=1, allow_nan=False))
    # small per-gene offset table for the next tasks
    np.savez_compressed(OUT / ("offsets_quick.npz" if a.quick else "offsets.npz"),
                        **{f"{p}__{st}": v.astype(np.float32) for p, d_ in keep_off.items() for st, v in d_.items()},
                        **{f"genes__{p}": genes_S[g] for p, g in panel_list.items()})
    print(f"[done] {time.time() - t0:.0f}s -> {OUT / name}", flush=True)


if __name__ == "__main__":
    main()
