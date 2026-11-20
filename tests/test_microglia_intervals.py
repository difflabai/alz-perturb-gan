"""Calibration uses independent donors and leaves expression-mean bounds alone."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('sd_calibration', Path(__file__).resolve().parents[1]/
    'analysis/coverage_empf_20261117/calibration.py')
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)


def example():
    low, high = np.ones((6, 100)), np.full((6, 100), 2.)
    y = np.full((6, 100), 1.5)
    y[:, 50:] = np.arange(1, 7)[:, None]
    return low, high, y, [f'cal{i}' for i in range(6)]


def test_six_donors_supply_six_examples_and_sd_support_is_respected():
    low, high, y, ids = example()
    record = C.fit_sd(low, high, y, ids, ['train'])
    assert record['rank'] == record['n_calibration_donors'] == 6
    assert np.all(np.asarray(record['sd_correction']) == 4)
    lo, hi = C.apply_sd(low[:2], high[:2], record, ['test1', 'test2'])
    assert np.array_equal(lo[:, :50], low[:2, :50])
    assert np.array_equal(hi[:, :50], high[:2, :50])
    assert np.all(lo[:, 50:] == 0) and np.all(hi[:, 50:] == 6)


def test_training_or_calibration_donor_cannot_be_used_as_independent_test():
    low, high, y, ids = example()
    with pytest.raises(ValueError, match='GAN training'):
        C.fit_sd(low, high, y, ids, [ids[0]])
    record = C.fit_sd(low, high, y, ids, ['train'])
    with pytest.raises(ValueError, match='overlap'):
        C.apply_sd(low[:1], high[:1], record, [ids[0]])


def test_mean_observations_do_not_fit_sd_corrections():
    low, high, y, ids = example()
    original = C.fit_sd(low, high, y, ids, ['train'])
    y[:, :50] = 1e9
    assert C.fit_sd(low, high, y, ids, ['train']) == original


def test_cells_or_coordinates_cannot_substitute_for_unique_calibration_donors():
    low, high, y, ids = example()
    with pytest.raises(ValueError, match='unique'):
        C.fit_sd(low, high, y, [ids[0]] * 6, ['train'])
    with pytest.raises(ValueError, match='insufficient'):
        C.fit_sd(low[:3], high[:3], y[:3], ids[:3], ['train'])
    with pytest.raises(ValueError, match='one-bag'):
        C.fit_sd(low, high, y[None], ids, ['train'])
