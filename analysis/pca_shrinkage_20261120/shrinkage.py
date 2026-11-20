"""Reusable scoring for separating PCA projection shrinkage from generator dispersion.

NumPy only (scikit-learn is imported lazily by ``refit_space`` for the frozen
cache recipe). Conventions follow the frozen interval metrics:

* predictive draws ``samples`` are ``[targets, draws, coordinates]``;
* real observations ``y`` are ``[repeats, targets, coordinates]``;
* bounds are the 10/90 percentiles over draws (alpha = .2) and the interval
  score is Gneiting-Raftery: width + (2/alpha) * distance outside.

Every aggregate is also returned per target (donor) so donor-bootstrap CIs can
be computed with ``bootstrap``.
"""
from __future__ import annotations

import math

import numpy as np

ALPHA = .2
PC_BLOCKS = {'pc01-10': slice(0, 10), 'pc11-30': slice(10, 30), 'pc31-50': slice(30, 50), 'all': slice(0, 50)}


# ---------------------------------------------------------------------- intervals
def percentile_bounds(samples, alpha=ALPHA):
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 3 or samples.shape[1] < 2 or not np.isfinite(samples).all():
        raise ValueError('need finite [targets, draws, coordinates] with >= 2 draws')
    return tuple(np.quantile(samples, [alpha/2, 1-alpha/2], axis=1))


def interval_score(low, high, y, alpha=ALPHA):
    """Elementwise interval score; infinite bounds give an infinite score."""
    low, high, y = (np.asarray(a, dtype=np.float64) for a in (low, high, y))
    if np.any(high < low):
        raise ValueError('high < low')
    with np.errstate(invalid='ignore'):
        width = high - low
    miss = np.maximum(low - y, 0) + np.maximum(y - high, 0)
    return width + (2/alpha) * miss


def per_target(low, high, y, alpha=ALPHA):
    """Per-target means over repeats and coordinates.

    low/high [n, D]; y [R, n, D]. Returns dict of [n] arrays.
    """
    low, high = np.asarray(low, np.float64), np.asarray(high, np.float64)
    y = np.asarray(y, np.float64)
    if y.ndim == 2:
        y = y[None]
    if low.shape != high.shape or y.shape[1:] != low.shape:
        raise ValueError(f'shape mismatch {low.shape} {high.shape} {y.shape}')
    lo, hi = low[None], high[None]
    out = {'coverage': ((y >= lo) & (y <= hi)).mean((0, 2)),
           'below': (y < lo).mean((0, 2)), 'above': (y > hi).mean((0, 2)),
           'width': np.broadcast_to(hi - lo, y.shape).mean((0, 2)),
           'interval_score': interval_score(lo, hi, y, alpha).mean((0, 2))}
    return out


def summarize_targets(per, mask=None):
    """Equal-weight mean over targets of a ``per_target`` dict."""
    mask = slice(None) if mask is None else np.asarray(mask)
    return {k: float(np.mean(v[mask])) for k, v in per.items()}


def log_sd_ratio(sample_sd, real_sd, eps=0.0):
    """log(median over draws of generated SD / mean over repeats of real SD).

    sample_sd [n, M, D]; real_sd [R, n, D] -> [n, D]. eps is added to both.
    """
    g = np.median(np.asarray(sample_sd, np.float64), axis=1)
    r = np.asarray(real_sd, np.float64).mean(0)
    if eps <= 0 and (np.any(g <= 0) or np.any(r <= 0)):
        raise ValueError('nonpositive SD; pass eps > 0')
    return np.log((g + eps) / (r + eps))


def block_table(samples, y, blocks, mean_offset=None, alpha=ALPHA, sd_eps=0.0):
    """Score [n,M,2D] mean+SD summaries per block of the SD half (and mean half).

    Returns {block: {'mean':per, 'sd':per, 'combined':per, 'log_sd_ratio':[n]}}
    where each per is a dict of [n] arrays.
    """
    samples = np.asarray(samples, np.float64)
    D = samples.shape[-1] // 2 if mean_offset is None else mean_offset
    low, high = percentile_bounds(samples, alpha)
    ratio = log_sd_ratio(samples[..., D:], np.asarray(y)[..., D:], sd_eps)
    out = {}
    for name, sl in blocks.items():
        idx = np.arange(D)[sl]
        m, s = idx, idx + D
        both = np.concatenate([m, s])
        out[name] = {'mean': per_target(low[:, m], high[:, m], y[..., m], alpha),
                     'sd': per_target(low[:, s], high[:, s], y[..., s], alpha),
                     'combined': per_target(low[:, both], high[:, both], y[..., both], alpha),
                     'log_sd_ratio': ratio[:, idx].mean(1)}
    return out


# ---------------------------------------------------------------------- bootstrap
def bootstrap(values, B=2000, seed=20261120, paired=None):
    """Donor-bootstrap percentile CI for the mean of ``values`` [n].

    With ``paired`` (another [n] array), the CI is for mean(values - paired).
    Returns dict(mean, lo, hi, n).
    """
    v = np.asarray(values, np.float64)
    if paired is not None:
        v = v - np.asarray(paired, np.float64)
    n = len(v)
    if n < 2:
        return dict(mean=float(v.mean()) if n else None, lo=None, hi=None, n=n)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, (B, n))
    stats = v[idx].mean(1)
    lo, hi = np.quantile(stats, [.025, .975])
    return dict(mean=float(v.mean()), lo=float(lo), hi=float(hi), n=n)


# ---------------------------------------------------------------------- PC summaries
def bag_summary(pcs):
    """[..., K, D] -> [..., 2D] bag mean then SD (ddof 0), as the frozen metric."""
    pcs = np.asarray(pcs, np.float64)
    return np.concatenate([pcs.mean(-2), pcs.std(-2)], -1)


def fit_whitening(cells):
    """Rotation/whitening of the coordinate space fit on pooled cells [N, D].

    Returns (mu, A) with new = (x - mu) @ A; columns ordered by variance
    (descending), unit variance on the fitting cells, deterministic signs.
    """
    X = np.asarray(cells, np.float64)
    mu = X.mean(0)
    C = np.cov(X - mu, rowvar=False, ddof=0)
    lam, V = np.linalg.eigh(C)
    order = np.argsort(lam)[::-1]
    lam, V = lam[order], V[:, order]
    signs = np.sign(V[np.abs(V).argmax(0), np.arange(V.shape[1])])
    V = V * np.where(signs == 0, 1, signs)
    return mu, V / np.sqrt(np.maximum(lam, 1e-12))


def apply_whitening(x, mu, A):
    return (np.asarray(x, np.float64) - mu) @ A


def shrinkage_scale_sd(samples, factors):
    """Multiply the SD half of [.., 2D] summaries by per-coordinate factors [D]."""
    s = np.array(samples, np.float64, copy=True)
    D = s.shape[-1] // 2
    f = np.asarray(factors, np.float64)
    if f.shape != (D,) or np.any(f <= 0):
        raise ValueError('need positive factors per coordinate')
    s[..., D:] *= f
    return s


# ---------------------------------------------------------------------- gene space
def gene_sd_from_moments(mean_log, sq_log):
    m = np.asarray(mean_log, np.float64)
    q = np.asarray(sq_log, np.float64)
    return np.sqrt(np.maximum(q - m*m, 0.0))


def gene_moments(logcp):
    """[K, G] per-cell log1p CP10k -> (mean_log float32, sq_log float16) like the archives."""
    L = np.asarray(logcp, np.float64)
    return L.mean(0).astype(np.float32), (L*L).mean(0).astype(np.float16)


def decile_groups(score, n=10):
    edges = np.quantile(score, np.linspace(0, 1, n+1))
    g = np.clip(np.searchsorted(edges, score, side='right') - 1, 0, n-1)
    return g, edges


# ---------------------------------------------------------------------- conformal
def conformal_rank(n_cal, alpha=ALPHA):
    return int(math.ceil((n_cal + 1) * (1 - alpha)))


def loo_conformal(low, high, y_cal, donors, groups=None, eligible=None, alpha=ALPHA, lower_clip=None):
    """Leave-one-donor-out split conformal (CQR signed score), optionally Mondrian.

    low/high [n, D] base bounds for n donors; y_cal [n, D] one calibration
    observation per donor; donors [n] identities (must be unique); groups [n]
    Mondrian labels or None (global); eligible [n] bool of donors allowed to
    calibrate (e.g. not GAN TRAIN donors). For each target i the calibration
    set is the other eligible donors in its group. If the required rank exceeds
    the calibration size, the correction is +inf (unbounded interval).

    Returns (lo, hi, info) with info['n_cal'], info['rank'] per target.
    """
    low, high, y_cal = (np.asarray(a, np.float64) for a in (low, high, y_cal))
    donors = np.asarray(donors).astype(str)
    if len(set(donors.tolist())) != len(donors):
        raise ValueError('duplicate donor identities')
    n = len(donors)
    groups = np.zeros(n, int) if groups is None else np.asarray(groups)
    eligible = np.ones(n, bool) if eligible is None else np.asarray(eligible, bool)
    score = np.maximum(low - y_cal, y_cal - high)
    lo, hi = low.copy(), high.copy()
    n_cal, ranks = np.zeros(n, int), np.zeros(n, int)
    for i in range(n):
        cal = np.flatnonzero(eligible & (groups == groups[i]) & (np.arange(n) != i))
        if i in cal:
            raise AssertionError('target in its own calibration set')
        k = conformal_rank(len(cal), alpha)
        n_cal[i], ranks[i] = len(cal), k
        if k > len(cal):
            q = np.full(low.shape[1], np.inf)
        else:
            q = np.sort(score[cal], axis=0)[k-1]
        lo[i], hi[i] = low[i] - q, high[i] + q
        if np.any(hi[i] < lo[i]):           # negative correction cannot cross
            mid = (low[i] + high[i]) / 2
            bad = hi[i] < lo[i]
            lo[i, bad] = hi[i, bad] = mid[bad]
    if lower_clip is not None:
        lo = np.maximum(lo, lower_clip)
        hi = np.maximum(hi, lo)
    return lo, hi, dict(n_cal=n_cal, rank=ranks)


def finite_summary(per, mask=None):
    """Like summarize_targets but reports infinite widths/scores honestly."""
    mask = np.ones(len(per['coverage']), bool) if mask is None else np.asarray(mask)
    out = {}
    for k, v in per.items():
        v = v[mask]
        fin = np.isfinite(v)
        out[k] = float(v.mean()) if fin.all() else None
        if k in ('width', 'interval_score'):
            out[k + '_finite_targets'] = int(fin.sum())
            out[k + '_mean_over_finite'] = float(v[fin].mean()) if fin.any() else None
    return out
