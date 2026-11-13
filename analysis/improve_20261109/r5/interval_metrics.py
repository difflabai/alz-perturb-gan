"""Predictive interval diagnostics, kept separate from historical scores.

The rank endpoints assume iid perfect-model draws only for their nominal
rank calculation. They give no calibrated guarantee for a fitted generator.
"""
import numpy as np


def bounds(samples, method='percentile', alpha=.2):
    samples = np.asarray(samples)
    if samples.ndim != 3 or samples.shape[1] < 2 or not 0 < alpha < 1:
        raise ValueError('need finite [targets,draws,coordinates], >=2 draws and alpha in (0,1)')
    if not np.isfinite(samples).all():
        raise ValueError('nonfinite predictive samples')
    if method == 'percentile':
        return tuple(np.quantile(samples, [alpha/2, 1-alpha/2], axis=1))
    if method != 'rank':
        raise ValueError(method)
    n = samples.shape[1]
    lo = max(1, int(np.floor(alpha/2*(n+1))))
    hi = min(n, int(np.ceil((1-alpha/2)*(n+1))))
    ordered = np.sort(samples, axis=1)
    return ordered[:,lo-1], ordered[:,hi-1]


def interval_score(low, high, observed, alpha=.2):
    low, high, observed = map(np.asarray, (low,high,observed))
    if not 0 < alpha < 1 or np.any(high < low):
        raise ValueError('invalid interval or alpha')
    return high-low + (2/alpha)*(np.maximum(low-observed,0)+np.maximum(observed-high,0))


def summarize(samples, observed, mask, method='percentile'):
    low,high = bounds(samples,method)
    out = {}
    for name, sl in [('mean',slice(0,50)),('sd',slice(50,100)),('combined',slice(0,100))]:
        lo,hi = low[mask,sl],high[mask,sl]
        y = observed[:,mask,sl]
        out[name] = {'coverage':float(((y>=lo)&(y<=hi)).mean()),
                     'below':float((y<lo).mean()), 'above':float((y>hi).mean()),
                     'width':float((hi-lo).mean()),
                     'interval_score':float(interval_score(lo,hi,y).mean())}
    return out
