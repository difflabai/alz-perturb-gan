"""Pure metric functions for the improve_20261109 evaluator (Interface 3).

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
    return float(np.corrcoef(a, b)[0, 1])


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
