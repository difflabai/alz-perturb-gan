"""Pure metric functions for the improve_20261109 evaluator, round 2 (F8).

Round-2 copy of ``../metrics.py``.  Every function above the line "Round-2
additions" is the round-1 code, except ``pearson`` (explicit formula instead
of np.corrcoef: same value to ~1e-15, faster); round-1 metrics reproduce to
1e-8 (test_evaluate_r2.py reproduce).  The round-2 additions are at the end:

  * DAM_UP / DAM_DOWN: the frozen disease-associated-microglia gene lists of
    r2/PROTOCOL_R2.md, copied verbatim (do not edit).
  * expression_bins / matched_null_sets: expression-matched random-gene null.
  * dam_score: signed agreement of a model contrast with the expected
    directions and its correlation with a real contrast, each against the
    matched null.  The matched null is a COMPETITIVE gene-set null (listed
    genes vs random genes); it ignores gene-gene correlation and is NOT a
    test of stage (repair 2026-11-10, review S1: on real hvg2000 fold-0
    donors, random within-High half-splits, i.e. no stage contrast, gave
    P(p<0.05) = 0.19-0.22 against nominal 0.05, because listed genes co-vary
    across donors: mean signed between-donor correlation 0.24 vs 0.06 for
    matched random genes).  Its z / p are descriptive only.
  * dam_perm_null (repair, S1): self-contained null for the same statistics:
    the High / Not AD labels are permuted among the High + Not AD rows
    (group sizes kept) and the contrast is recomputed on the SAME listed
    genes, so gene-gene correlation is carried into the null.  Under no
    stage effect the rows are exchangeable and the permutation p is valid
    (checked in test_evaluate_r2.py units: uniform p on random within-stage
    splits of real donors).  This is the stage test used for ranking
    (``dam_signed_perm_z``).
  * xfit_score: the pooled cross-fitted donor score with a within-fold
    permutation null (definition in evaluate.py's module docstring).
  * additive_recalibration: per-gene train-only additive correction.

The round-1 module docstring follows.

Pure metric functions for the improve_20261109 evaluator (Interface 3).

Everything here is numpy-only and takes in-memory arrays, so fits (read from
generated.npz, Interface 2) and baselines (baselines.py, in-memory arrays with
the same layout) are scored by exactly the same code.

Definitions reused verbatim from the earlier benchmark so numbers mean the
same thing (benchmark/bench_core.py, benchmark/evaluation.py,
core-comparison-20261103/scoring.py):
  * energy_score(Y, y): mean ||Y_m - y|| - 0.5 * mean_{m != m'} ||Y_m - Y_m'||
  * cell_energy(A, B): 2 E||a-b|| - E||a-a'|| - E||b-b'|| with V-statistic
    (diagonal-including) within-set means, exactly as bench_core.cell_energy
  * coverage80: fraction of (donor, dimension) pairs whose real value lies in
    the [10th, 90th] percentile interval of that donor's generated summaries
  * spread_ratio: sqrt(total generated variance trace pooled over targets and
    generated donors / real between-donor variance trace (ddof=1))
  * within_condition_spread_ratio: sqrt(mean over targets of the generated
    variance trace at fixed condition / real between-donor variance trace)
Differences are documented in NOTES_evaluator.md (summary spaces, the extra
within-stage spread ratio, per-target cell energy).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import rankdata

STAGE_NOT_AD = 0
STAGE_HIGH = 3


# --------------------------------------------------------------------------
# Shared definitions (identical to bench_core)
# --------------------------------------------------------------------------
def energy_score(Y, y):
    Y = np.asarray(Y, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    cross = np.linalg.norm(Y - y, axis=1).mean()
    return float(cross - .5 * cdist(Y, Y).sum() / (len(Y) * (len(Y) - 1))) if len(Y) > 1 else float(cross)


def cell_energy(A, B):
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    return float(2 * cdist(A, B).mean() - cdist(A, A).mean() - cdist(B, B).mean())


def log_cp10k(counts):
    """log1p(CP10k) along the last axis (panel total as denominator)."""
    X = np.asarray(counts, dtype=np.float64)
    return np.log1p(X * (1e4 / np.maximum(X.sum(-1, keepdims=True), 1)))


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def _f(x):
    """JSON-safe float (None for nan/inf/None)."""
    if x is None:
        return None
    x = float(x)
    return x if np.isfinite(x) else None


def pearson(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return None
    # r2: explicit formula instead of np.corrcoef (same value to ~1e-15, several
    # times faster on a loaded machine); clipped to [-1, 1] like corrcoef.
    a = a - a.mean(); b = b - b.mean()
    return float(np.clip(np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b)), -1.0, 1.0))


def spearman(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return None
    return pearson(rankdata(a[ok]), rankdata(b[ok]))


def nanmean(values):
    v = [x for x in values if x is not None and np.isfinite(x)]
    return float(np.mean(v)) if v else None


def det_subsample(n, k, seed):
    """Deterministic subsample of min(n, k) indices (sorted)."""
    if n <= k:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, k, replace=False))


def stable_seed(*parts):
    """Deterministic 32-bit seed from strings/ints (independent of PYTHONHASHSEED)."""
    import hashlib
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:4], "little")


# --------------------------------------------------------------------------
# Pseudobulk prediction
# --------------------------------------------------------------------------
def predicted_profiles(pb_counts):
    """Model prediction per target: mean over M generated donors of the
    generated pseudobulk log1p(CP10k). pb_counts [T,M,G] -> [T,G]."""
    T, M, G = pb_counts.shape
    out = np.empty((T, G), dtype=np.float64)
    for t in range(T):
        out[t] = log_cp10k(pb_counts[t]).mean(0)
    return out


def pseudobulk_metrics(pred, obs, grand, stage, donors):
    """pred, obs [n,G] log1p(CP10k); grand [G] train grand-mean profile.
    Per donor: RMSE over genes, Pearson/Spearman of deviations from grand."""
    per = []
    for i in range(len(obs)):
        dp, do = pred[i] - grand, obs[i] - grand
        per.append({"donor": str(donors[i]), "stage": int(stage[i]),
                    "rmse": _f(np.sqrt(np.mean((pred[i] - obs[i]) ** 2))),
                    "dev_pearson": pearson(dp, do), "dev_spearman": spearman(dp, do)})
    out = {"n_donors": len(per), "per_donor": per}
    for k in ["rmse", "dev_pearson", "dev_spearman"]:
        out[k] = nanmean([p[k] for p in per])
        out[k + "_by_stage"] = {str(s): nanmean([p[k] for p in per if p["stage"] == s])
                                for s in sorted(set(int(x) for x in stage))}
    return out


# --------------------------------------------------------------------------
# Per-gene per-cell mean log expression and nonzero fraction (depth-matched)
# --------------------------------------------------------------------------
def gene_error_metrics(gen, real, strata_ref, thresholds):
    """gen, real [n,G]: model value (already averaged over M) and real value
    per target donor. Gene error e_g = mean over donors of (gen - real).
    strata_ref [G]: expression level used for quartile strata (train-only).
    """
    err = gen - real
    e = err.mean(0)
    out = {"rmse_gene": _f(np.sqrt(np.mean(e ** 2))),
           "rmse_donor_gene": _f(np.sqrt(np.mean(err ** 2))),
           "bias": _f(e.mean()), "mae_gene": _f(np.abs(e).mean()),
           "frac_abs_err_above": {str(t): _f((np.abs(e) > t).mean()) for t in thresholds}}
    q = np.quantile(strata_ref, [.25, .5, .75])
    bins = np.digitize(strata_ref, q)
    out["by_expression_quartile"] = {
        f"Q{b + 1}": {"n_genes": int((bins == b).sum()),
                      "rmse_gene": _f(np.sqrt(np.mean(e[bins == b] ** 2))) if (bins == b).any() else None,
                      "bias": _f(e[bins == b].mean()) if (bins == b).any() else None}
        for b in range(4)}
    out["quartile_edges"] = [float(x) for x in q]
    return out


# --------------------------------------------------------------------------
# Stage contrasts
# --------------------------------------------------------------------------
def stage_contrasts(P, stage):
    """P [n,G] donor profiles, stage [n] ints in 0..3.
    Returns High-minus-NotAD (stage-mean difference) and per-gene OLS slope on
    ordinal stage across donors, plus the per-stage donor counts."""
    stage = np.asarray(stage)
    counts = {str(s): int((stage == s).sum()) for s in range(4)}
    out = {"n_by_stage": counts}
    if counts["0"] > 0 and counts["3"] > 0:
        out["high_notad"] = P[stage == STAGE_HIGH].mean(0) - P[stage == STAGE_NOT_AD].mean(0)
    else:
        out["high_notad"] = None
    out["low_n_high_notad"] = bool(min(counts["0"], counts["3"]) < 2)
    if len(set(stage.tolist())) >= 2:
        s = stage - stage.mean()
        out["slope"] = (s[:, None] * (P - P.mean(0))).sum(0) / (s ** 2).sum()
    else:
        out["slope"] = None
    return out


def agreement(model, real, top_frac=.1):
    """Agreement between a model effect vector and a real effect vector over genes."""
    if model is None or real is None:
        return None
    model = np.asarray(model, dtype=np.float64)
    real = np.asarray(real, dtype=np.float64)
    nz = (np.sign(model) != 0) & (np.sign(real) != 0)
    k = max(1, int(np.ceil(top_frac * len(real))))
    top = np.argsort(-np.abs(real), kind="stable")[:k]
    vr = real.var()
    out = {"spearman": spearman(model, real), "pearson": pearson(model, real),
           "sign_agreement": _f((np.sign(model[nz]) == np.sign(real[nz])).mean()) if nz.any() else None,
           "top_decile_sign_agreement": _f((np.sign(model[top]) == np.sign(real[top])).mean()),
           "top_decile_spearman": spearman(model[top], real[top]),
           # Shrinkage: OLS slope of the model effect on the real effect.
           "regression_slope": _f(((model - model.mean()) * (real - real.mean())).mean() / vr) if vr > 0 else None,
           "top_decile_regression_slope": _f(np.polyfit(real[top], model[top], 1)[0]) if k >= 3 and real[top].std() > 0 else None,
           "sd_ratio": _f(model.std() / np.sqrt(vr)) if vr > 0 else None,
           "n_genes": int(len(real)), "n_top": int(k)}
    return out


def contrast_block(pred_E, stage_E, real_sets):
    """Model contrasts from predicted profiles of target set E, compared against
    real contrasts given as {name: stage_contrasts(...)}."""
    m = stage_contrasts(pred_E, stage_E)
    out = {"model_n_by_stage": m["n_by_stage"]}
    for rname, r in real_sets.items():
        out[f"high_notad_vs_{rname}"] = agreement(m["high_notad"], r["high_notad"])
        out[f"slope_vs_{rname}"] = agreement(m["slope"], r["slope"])
        out[f"real_{rname}_n_by_stage"] = r["n_by_stage"]
        out[f"real_{rname}_low_n"] = r["low_n_high_notad"]
    return out, m


def split_half_reliability(P, stage, n_splits=200, seed=20261109):
    """Split-half reliability of the real High-minus-NotAD contrast: per split,
    each stage's donors are randomly halved; Spearman/Pearson between the two
    half contrasts; mean over splits and Spearman-Brown corrected value.
    Requires >= 2 donors in both Not AD and High."""
    stage = np.asarray(stage)
    idx0, idx3 = np.where(stage == 0)[0], np.where(stage == 3)[0]
    if len(idx0) < 2 or len(idx3) < 2:
        return {"defined": False, "n_notad": int(len(idx0)), "n_high": int(len(idx3))}
    rng = np.random.default_rng(seed)
    sp, pe = [], []
    for _ in range(n_splits):
        a0 = rng.permutation(idx0); a3 = rng.permutation(idx3)
        h0a, h0b = a0[:len(a0) // 2], a0[len(a0) // 2:]
        h3a, h3b = a3[:len(a3) // 2], a3[len(a3) // 2:]
        ca = P[h3a].mean(0) - P[h0a].mean(0)
        cb = P[h3b].mean(0) - P[h0b].mean(0)
        sp.append(spearman(ca, cb)); pe.append(pearson(ca, cb))
    r = nanmean(sp)
    return {"defined": True, "n_notad": int(len(idx0)), "n_high": int(len(idx3)), "n_splits": n_splits,
            "spearman": r, "pearson": nanmean(pe),
            "spearman_brown_spearman": _f(2 * r / (1 + r)) if r is not None and r > -1 else None}


# --------------------------------------------------------------------------
# Donor-level distribution metrics
# --------------------------------------------------------------------------
def pc_bag_summary(pcs):
    """[..., K, D] PC coordinates -> [..., 2D] bag mean and bag sd (ddof=0)."""
    pcs = np.asarray(pcs, dtype=np.float64)
    return np.concatenate([pcs.mean(-2), pcs.std(-2)], -1)


def within_stage_var_trace(y, stage):
    """Pooled within-stage variance trace of real summaries y [n,D]."""
    stage = np.asarray(stage)
    ss, dof = 0.0, 0
    for s in set(stage.tolist()):
        ys = y[stage == s]
        if len(ys) > 1:
            ss += float(((ys - ys.mean(0)) ** 2).sum()); dof += len(ys) - 1
    return ss / dof if dof > 0 else None


def distribution_metrics(Y, y, stage, train_within_trace=None):
    """Y [n,M,D] generated summaries per target; y [n,D] real summary per
    target; stage [n]. Definitions follow scoring.score_summary_arrays."""
    Y = np.asarray(Y, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    stage = np.asarray(stage)
    n, M, D = Y.shape
    scores = np.array([energy_score(Y[i], y[i]) for i in range(n)])
    lo, hi = np.percentile(Y, [10, 90], axis=1)
    inside = (y >= lo) & (y <= hi)
    out = {"n_targets": int(n), "M": int(M), "D": int(D),
           "per_donor_energy_score": [float(s) for s in scores],
           "energy_score": _f(scores.mean()),
           "stage_balanced_energy_score": nanmean([scores[stage == s].mean() for s in set(stage.tolist())]),
           "coverage80": _f(inside.mean()),
           "per_donor_coverage80": [float(c) for c in inside.mean(1)]}
    gen_within = float(Y.var(1, ddof=1).mean(0).sum()) if M > 1 else None
    real_var = float(y.var(0, ddof=1).sum()) if n > 1 else 0.0
    if real_var > 0:
        out["spread_ratio"] = _f(np.sqrt(Y.reshape(-1, D).var(0, ddof=1).sum() / real_var))
        out["within_condition_spread_ratio"] = _f(np.sqrt(gen_within / real_var)) if gen_within is not None else None
    else:
        out["spread_ratio"] = out["within_condition_spread_ratio"] = None
    w = within_stage_var_trace(y, stage)
    out["real_within_stage_var_trace"] = _f(w)
    out["within_stage_spread_ratio"] = _f(np.sqrt(gen_within / w)) if (w and gen_within is not None) else None
    if train_within_trace:
        out["within_stage_spread_ratio_train_reference"] = _f(np.sqrt(gen_within / train_within_trace)) if gen_within is not None else None
    return out


# --------------------------------------------------------------------------
# Paired comparison helpers for aggregation
# --------------------------------------------------------------------------
def paired_summary(a, b, higher_is_better):
    """Mean difference a-b and sign count (how many units a is better), no CI
    claims with n=3 folds."""
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if not pairs:
        return {"n": 0}
    d = np.array([x - y for x, y in pairs])
    better = (d > 0) if higher_is_better else (d < 0)
    return {"n": int(len(d)), "mean_difference": _f(d.mean()),
            "se_difference": _f(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else None,
            "n_better": int(better.sum()), "n_worse": int(((d < 0) if higher_is_better else (d > 0)).sum()),
            "n_tied": int((d == 0).sum()), "direction": "higher_is_better" if higher_is_better else "lower_is_better"}


# ==========================================================================
# Round-2 additions (F8).  Nothing above this line differs from round 1.
# ==========================================================================
# Frozen stage-signal gene list: r2/PROTOCOL_R2.md "Frozen stage-signal gene
# list (F8)", copied verbatim.  Do not edit.
DAM_UP = ["APOE", "TREM2", "TYROBP", "SPP1", "GPNMB", "LPL", "CD9", "LGALS3", "CST7", "ITGAX", "CLEC7A", "AXL",
          "CSF1", "CD63", "CTSD", "CTSB", "FTH1", "LILRB4", "MITF", "MYO1E"]
DAM_DOWN = ["P2RY12", "TMEM119", "CX3CR1", "P2RY13", "SELPLG", "CSF1R", "SALL1"]


def dam_genes(genes):
    """(index array, expected direction +1/-1, missing names) of the frozen
    list on a gene axis; order = DAM_UP then DAM_DOWN as listed."""
    pos = {str(g): i for i, g in enumerate(genes)}
    idx, sgn, missing = [], [], []
    for names, s in ((DAM_UP, 1.0), (DAM_DOWN, -1.0)):
        for g in names:
            if g in pos:
                idx.append(pos[g]); sgn.append(s)
            else:
                missing.append(g)
    return np.array(idx, dtype=int), np.array(sgn), missing


def expression_bins(level, n_bins=10):
    """Quantile bin (0..n_bins-1) of every gene by a train-only expression level."""
    level = np.asarray(level, dtype=np.float64)
    edges = np.quantile(level, np.linspace(0, 1, n_bins + 1)[1:-1])
    return np.digitize(level, edges)


def matched_null_sets(listed, level, n_draws, seed, n_bins=10):
    """[n_draws, len(listed)] gene indices: draw d replaces listed gene i by a
    uniformly drawn NON-listed gene of the same expression-level bin (quantile
    bins of ``level`` over all genes); genes within one draw are distinct
    whenever the bin has enough genes."""
    listed = np.asarray(listed, dtype=int)
    bins = expression_bins(level, n_bins)
    rng = np.random.default_rng(seed)
    pool = {b: np.setdiff1d(np.where(bins == b)[0], listed) for b in np.unique(bins[listed])}
    out = np.empty((n_draws, len(listed)), dtype=int)
    for b, cand in pool.items():
        cols = np.where(bins[listed] == b)[0]
        if len(cand) == 0:
            raise ValueError(f"expression bin {b} has no non-listed gene")
        for d in range(n_draws):
            out[d, cols] = rng.choice(cand, len(cols), replace=len(cand) < len(cols))
    return out


def _null_block(obs, null):
    null = np.asarray([x for x in null if x is not None and np.isfinite(x)], dtype=np.float64)
    if obs is None or len(null) < 2:
        return {"value": _f(obs), "null_n": int(len(null))}
    sd = null.std(ddof=1)
    return {"value": _f(obs), "null_mean": _f(null.mean()), "null_sd": _f(sd), "null_q95": _f(np.quantile(null, .95)),
            "z": _f((obs - null.mean()) / sd) if sd > 0 else None,
            "p_one_sided": _f((1 + (null >= obs).sum()) / (1 + len(null))), "null_n": int(len(null))}


def _rowwise_spearman(A, B):
    """Spearman per row of [n, k] arrays (ties averaged)."""
    ra = rankdata(A, axis=1); rb = rankdata(B, axis=1)
    ra = ra - ra.mean(1, keepdims=True); rb = rb - rb.mean(1, keepdims=True)
    den = np.sqrt((ra ** 2).sum(1) * (rb ** 2).sum(1))
    return np.where(den > 0, (ra * rb).sum(1) / np.where(den > 0, den, 1), np.nan)


def dam_score(model_c, idx, sgn, null_sets, real=None):
    """Frozen DAM direction score of one model contrast vector.

    model_c [G]: the model's High-minus-NotAD contrast over the whole panel.
    idx, sgn: dam_genes(); null_sets: matched_null_sets(idx, ...) (gene i of a
    null draw inherits the expected direction sgn[i]).  real: {name: real
    contrast [G] or None}.  Statistics over the listed genes present:
      sign_frac   fraction with sign(c_g) == expected direction (0 counts as wrong)
      signed_mean mean of sgn_g * c_g (log units)
      spearman_vs_<name>  Spearman over listed genes of c with the real contrast
    each with the same statistic on the matched random-gene draws (null_mean,
    null_sd, z = (value - null_mean)/null_sd, one-sided p = (1 + #null >= value)
    / (1 + n_draws))."""
    if model_c is None or len(idx) == 0:
        return {"defined": False, "n_present": int(len(idx))}
    c = np.asarray(model_c, dtype=np.float64)
    cn = c[null_sets]
    out = {"defined": True, "n_present": int(len(idx)), "n_up_present": int((sgn > 0).sum()),
           "n_down_present": int((sgn < 0).sum()), "n_null_draws": int(len(null_sets)),
           "sign_frac": _null_block(float((np.sign(c[idx]) == sgn).mean()), (np.sign(cn) == sgn).mean(1)),
           "signed_mean": _null_block(float((sgn * c[idx]).mean()), (sgn * cn).mean(1)),
           "per_gene_model_contrast": [float(x) for x in c[idx]]}
    for name, r in (real or {}).items():
        if r is None:
            out[f"spearman_vs_{name}"] = None
            continue
        r = np.asarray(r, dtype=np.float64)
        out[f"spearman_vs_{name}"] = _null_block(spearman(c[idx], r[idx]), _rowwise_spearman(cn, r[null_sets]))
    return out


def dam_perm_null(P, stage, idx, sgn, n_perm, seed):
    """Stage-label permutation null for the DAM direction statistics.

    P [n, G] per-row profiles (model targets or real donors), stage [n] in
    0..3.  The observed contrast is mean(High rows) - mean(Not AD rows) on the
    listed genes idx (the dam_score contrast restricted to idx).  Null: the
    labels of the High + Not AD rows are permuted (group sizes kept), n_perm
    draws from default_rng(seed); statistics sign_frac and signed_mean with
    z / one-sided p as in _null_block.  When the number of distinct splits
    C(n_high + n_notad, n_notad) is <= n_perm, every split is enumerated
    instead (exact permutation distribution; the observed split is part of
    it).  Returns {'defined': False} without both groups or listed genes."""
    from itertools import combinations
    from math import comb
    stage = np.asarray(stage)
    hi = np.where(stage == STAGE_HIGH)[0]; lo = np.where(stage == STAGE_NOT_AD)[0]
    if len(idx) == 0 or len(hi) == 0 or len(lo) == 0:
        return {"defined": False, "n_high": int(len(hi)), "n_notad": int(len(lo))}
    X = np.asarray(P, dtype=np.float64)[np.concatenate([hi, lo])][:, idx]
    n, nl = len(X), len(lo)
    sgn = np.asarray(sgn, dtype=np.float64)
    tot = X.sum(0)

    def stats(lo_sets):                      # lo_sets [D, nl] row indices into X
        s_lo = X[lo_sets].sum(1)             # [D, k]
        c = (tot - s_lo) / (n - nl) - s_lo / nl
        return (np.sign(c) == sgn).mean(1), (sgn * c).mean(1)

    obs_sf, obs_sm = stats(np.arange(n - nl, n)[None, :])
    n_splits = comb(n, nl)
    if n_splits <= n_perm:
        sets = np.array(list(combinations(range(n), nl)), dtype=int)
        kind = "exact"
    else:
        rng = np.random.default_rng(seed)
        sets = np.stack([rng.permutation(n)[:nl] for _ in range(n_perm)])
        kind = "random"
    sf, sm = stats(sets)
    return {"defined": True, "n_high": int(len(hi)), "n_notad": int(nl), "n_splits": int(n_splits),
            "null_kind": kind, "sign_frac": _null_block(float(obs_sf[0]), sf),
            "signed_mean": _null_block(float(obs_sm[0]), sm)}


def real_dam_reference(real_c, idx, sgn, null_sets):
    """Model-free reference: the same direction statistics for a real contrast."""
    if real_c is None or len(idx) == 0:
        return None
    r = np.asarray(real_c, dtype=np.float64)
    rn = r[null_sets]
    return {"sign_frac": _null_block(float((np.sign(r[idx]) == sgn).mean()), (np.sign(rn) == sgn).mean(1)),
            "signed_mean": _null_block(float((sgn * r[idx]).mean()), (sgn * rn).mean(1))}


def additive_recalibration(pred_train, obs_train):
    """Per-gene additive correction a_g = mean over train donors of
    (observed - predicted); add it to held-out predictions."""
    return (np.asarray(obs_train, dtype=np.float64) - np.asarray(pred_train, dtype=np.float64)).mean(0)


def _row_pearson(O, P):
    """Pearson per row pair of O [n,G] and P [m,G] -> [n,m] (rows centred)."""
    O = O - O.mean(1, keepdims=True); P = P - P.mean(1, keepdims=True)
    no = np.sqrt((O ** 2).sum(1)); npp = np.sqrt((P ** 2).sum(1))
    C = (O @ P.T) / np.outer(no, npp)
    C[~np.isfinite(C)] = np.nan
    return C


def xfit_score(folds, n_perm=2000, seed=20261109):
    """Pooled cross-fitted donor score with a within-fold permutation null.

    folds: list of dicts, one per inner fold of ONE repeat, with
      obs  [n_k, G_k]  held-out donor deviations o_d = y_d - gbar_k
      cand [J_k, G_k]  candidate predicted deviations (rows may be NaN)
      assign [n_k]     index into cand of each donor's own prediction
      group [n_k]      label used for the permutation (stage)
    score = mean over all donors of Pearson_g(o_d, cand[assign_d])
    pooled_pairs = Pearson over all stacked (donor, gene) pairs.
    Null: within each fold the held-out donors' assignments are permuted
    (stage counts per fold kept); same statistics; one-sided p."""
    stats = []
    for f in folds:
        O = np.asarray(f["obs"], np.float64); P = np.asarray(f["cand"], np.float64)
        C = _row_pearson(O, np.nan_to_num(P))
        C[:, ~np.isfinite(P).all(1)] = np.nan
        so, soo = O.sum(1), (O ** 2).sum(1)
        sp_, spp = np.nan_to_num(P).sum(1), (np.nan_to_num(P) ** 2).sum(1)
        sop = O @ np.nan_to_num(P).T
        stats.append(dict(C=C, so=so, soo=soo, sp=sp_, spp=spp, sop=sop, G=O.shape[1],
                          assign=np.asarray(f["assign"], int), group=np.asarray(f["group"])))

    def compute(assigns):
        cs, tot = [], np.zeros(6)
        for s, a in zip(stats, assigns):
            r = np.arange(len(a))
            cs.append(s["C"][r, a])
            tot += [s["G"] * len(a), s["so"].sum(), s["sp"][a].sum(), s["soo"].sum(), s["spp"][a].sum(), s["sop"][r, a].sum()]
        cs = np.concatenate(cs)
        n, a1, b1, a2, b2, ab = tot
        cov = ab / n - a1 / n * b1 / n
        den = np.sqrt(max(a2 / n - (a1 / n) ** 2, 0) * max(b2 / n - (b1 / n) ** 2, 0))
        return (float(np.nanmean(cs)) if np.isfinite(cs).any() else np.nan), (cov / den if den > 0 else np.nan), cs

    obs_mean, obs_pooled, per = compute([s["assign"] for s in stats])
    rng = np.random.default_rng(seed)
    nm, npool = [], []
    for _ in range(n_perm):
        nm_a = [s["assign"][rng.permutation(len(s["assign"]))] for s in stats]
        m, p, _ = compute(nm_a)
        nm.append(m); npool.append(p)
    groups = np.concatenate([s["group"] for s in stats])
    return {"n_donors": int(len(per)), "n_folds": len(stats), "n_perm": int(n_perm),
            "score": _null_block(_f(obs_mean), nm), "pooled_pairs": _null_block(_f(obs_pooled), npool),
            "per_donor": [_f(x) for x in per],
            "by_group": {str(g): nanmean(per[groups == g].tolist()) for g in sorted(set(groups.tolist()))}}

