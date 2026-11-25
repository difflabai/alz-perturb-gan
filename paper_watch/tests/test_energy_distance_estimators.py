"""The unbiased energy distance removes the draw-count offset of the V-statistic."""
import numpy as np
import pytest

from alztest.harness import energy_distance_unbiased


def v_statistic(a, b):
    def mean_distance(x, y):
        return np.sqrt(((x[:, None] - y[None]) ** 2).sum(-1)).mean()
    return 2 * mean_distance(a, b) - mean_distance(a, a) - mean_distance(b, b)


def test_unbiased_estimate_differs_from_v_statistic_by_the_self_pair_terms():
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=(40, 5)), rng.normal(0.3, 1.2, size=(25, 5))

    def within(x):
        return np.sqrt(((x[:, None] - x[None]) ** 2).sum(-1)).sum() / (len(x) * (len(x) - 1))

    expected = v_statistic(a, b) - within(a) / len(a) - within(b) / len(b)
    assert energy_distance_unbiased(a, b) == pytest.approx(expected, rel=1e-10)


def test_unbiased_estimate_removes_the_draw_count_offset():
    rng = np.random.default_rng(2)
    observed = rng.normal(size=(160, 10))
    v_bias, u_bias = [], []
    for _ in range(40):
        generated = rng.normal(size=(256, 10))
        u_bias.append(energy_distance_unbiased(generated, observed))
        v_bias.append(v_statistic(generated, observed))
    # Same distribution: the V-statistic carries a positive offset of about
    # mean|a-a'|/256 + mean|b-b'|/160 (0.045 here); the U-statistic does not.
    offset = np.mean(v_bias) - np.mean(u_bias)
    assert 0.035 < offset < 0.055
    assert abs(np.mean(u_bias)) < offset / 2


def test_unbiased_estimate_needs_two_rows():
    with pytest.raises(ValueError, match="at least two rows"):
        energy_distance_unbiased(np.zeros((1, 2)), np.zeros((3, 2)))
