import numpy as np
import pytest
from interval_metrics import bounds, interval_score


def test_score_penalizes_width_and_misses():
    assert interval_score(0.,2.,1.) == 2.
    assert interval_score(0.,2.,-1.) == 12.
    assert interval_score(0.,2.,3.) == 12.
    assert interval_score(-10.,10.,1.) > interval_score(0.,2.,1.)


def test_order_statistics_and_finite_sample_null():
    rng=np.random.default_rng(20261113)
    values=rng.random((30000,32,1))
    observed=rng.random((30000,1))
    lo,hi=bounds(values,'percentile')
    raw=((observed>=lo)&(observed<=hi)).mean()
    lo,hi=bounds(values,'rank')
    rank=((observed>=lo)&(observed<=hi)).mean()
    assert abs(raw-.8*31/33)<.01
    assert abs(rank-27/33)<.01
    ordered=np.sort(values,axis=1)
    assert np.array_equal(lo,ordered[:,2])
    assert np.array_equal(hi,ordered[:,29])


def test_invalid_intervals_fail():
    with pytest.raises(ValueError):bounds(np.ones((2,1,100)))
    with pytest.raises(ValueError):interval_score(2,1,0)
