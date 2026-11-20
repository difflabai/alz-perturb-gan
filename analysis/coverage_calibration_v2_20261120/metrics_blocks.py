"""Per-PC-block, per-stage and per-donor interval readouts with donor-bootstrap CIs.

Coordinates are [donor, 100]: 50 PC bag means then 50 PC bag SDs, as in
``analysis/improve_20261109/r5/interval_metrics.py``. Observations are
[repeat, donor, 100]. The interval score is the same alpha=.2 proper score
(width + 10 * miss distance). Aggregates over ``all`` reproduce the pinned
equal-weight readout exactly; blocks and weights only add views of it.
"""
import numpy as np

D = 50
ALPHA = .2
BLOCKS = {'pc01_10': slice(0, 10), 'pc11_30': slice(10, 30), 'pc31_50': slice(30, 50), 'all': slice(0, 50)}
GROUPS = ('mean', 'sd', 'combined')
METRICS = ('coverage', 'below', 'above', 'width', 'interval_score')


def columns(group, block):
    pcs = np.arange(D)[BLOCKS[block]]
    return {'mean': pcs, 'sd': pcs + D, 'combined': np.r_[pcs, pcs + D]}[group]


def elementwise(low, high, observed, alpha=ALPHA):
    """Per [repeat, donor, coordinate] metric arrays."""
    low, high, y = (np.asarray(a, dtype=float) for a in (low, high, observed))
    if low.shape != high.shape or y.shape[1:] != low.shape or low.shape[-1] != 2*D:
        raise ValueError('expected [donor,100] bounds and [repeat,donor,100] observations')
    if np.any(high < low):
        raise ValueError('reversed bounds')
    below, above = y < low, y > high
    width = np.broadcast_to(high - low, y.shape)
    score = width + (2/alpha) * (np.maximum(low - y, 0) + np.maximum(y - high, 0))
    return dict(coverage=~below & ~above, below=below, above=above, width=width, interval_score=score)


def weights(pc_variance, group):
    """Coordinate weights proportional to each PC's TRAIN variance (pc_sd**2)."""
    w = np.asarray(pc_variance, dtype=float)
    if w.shape != (D,) or np.any(w <= 0):
        raise ValueError('need 50 positive PC variances')
    w = w / w.sum()
    return {'mean': w, 'sd': w, 'combined': np.r_[w, w] / 2}[group]


def summarize(low, high, observed, mask, pc_variance=None):
    """{group: {block: metrics}} over donors in ``mask``; adds 'variance_weighted'."""
    E = elementwise(low, high, observed)
    mask = np.asarray(mask, dtype=bool)
    out = {}
    for group in GROUPS:
        out[group] = {block: {m: float(E[m][:, mask][..., columns(group, block)].mean()) for m in METRICS}
                      for block in BLOCKS}
        if pc_variance is not None:
            w = weights(pc_variance, group); cols = columns(group, 'all')
            out[group]['variance_weighted'] = {
                m: float((E[m][:, mask][..., cols].mean((0, 1)) * w).sum()) for m in METRICS}
    return out


def per_donor(low, high, observed, metric, group, block='all'):
    """Per-donor mean of ``metric`` over repeats and the block's coordinates."""
    E = elementwise(low, high, observed)
    return E[metric][..., columns(group, block)].astype(float).mean((0, 2))


def per_stage(low, high, observed, mask, stage, names=('Not AD', 'Low', 'Intermediate', 'High')):
    E = elementwise(low, high, observed)
    stage, mask = np.asarray(stage), np.asarray(mask, dtype=bool)
    out = {}
    for s, name in enumerate(names):
        rows = mask & (stage == s)
        out[name] = dict(donors=int(rows.sum()))
        for group in GROUPS:
            out[name][group] = (None if not rows.any() else
                                {m: float(E[m][:, rows][..., columns(group, 'all')].mean())
                                 for m in ('coverage', 'interval_score')})
    return out


def paired_bootstrap(base, candidate, B=20000, seed=20261120, kind='relative_reduction', level=.95):
    """Paired donor bootstrap of a candidate-vs-base contrast of donor means.

    ``relative_reduction`` = 1 - mean(candidate)/mean(base) (positive = lower
    score, as a fraction); ``difference`` = mean(candidate) - mean(base).
    Donors are resampled with replacement as pairs. Percentile interval.
    """
    a, b = np.asarray(base, dtype=float), np.asarray(candidate, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or len(a) < 2:
        raise ValueError('need paired per-donor vectors')

    def stat(x, y):
        return 1 - y.mean(-1) / x.mean(-1) if kind == 'relative_reduction' else y.mean(-1) - x.mean(-1)
    idx = np.random.default_rng(seed).integers(0, len(a), size=(B, len(a)))
    draws = stat(a[idx], b[idx])
    q = (1 - level) / 2
    lo, hi = np.quantile(draws, [q, 1 - q])
    return dict(kind=kind, estimate=float(stat(a, b)), ci=[float(lo), float(hi)], level=level,
                B=B, seed=seed, n_donors=len(a), donors_candidate_lower=int((b < a).sum()),
                donors_candidate_higher=int((b > a).sum()), donors_tied=int((a == b).sum()))
