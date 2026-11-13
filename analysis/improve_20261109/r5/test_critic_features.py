"""Verify donor isolation and stage-confound removal in the feature probe."""
import numpy as np
from critic_features import fit_readout, feature_diagnostics
from compare_critic_features import remove_train_stage_effect


def test_heldout_features_targets_cannot_choose_readout_or_change_train_predictions():
    rng = np.random.default_rng(19)
    groups = np.repeat(np.arange(8), 2)
    train = groups < 6
    x = rng.normal(size=(16, 5))
    y = rng.normal(size=(16, 7))
    prediction, info = fit_readout(x, y, groups, train)
    changed_x, changed_y = x.copy(), y.copy()
    changed_x[~train] += 100
    changed_y[~train] -= 100
    changed, changed_info = fit_readout(changed_x, changed_y, groups, train)
    assert info == changed_info
    np.testing.assert_array_equal(prediction[train], changed[train])


def test_stage_only_activation_cannot_create_residual_repeatability():
    stage = np.repeat(np.eye(4)[np.arange(12) % 4], 2, axis=0)
    held = np.repeat(np.arange(12) >= 8, 2)
    x = stage @ np.arange(24).reshape(4, 6) + 1000
    diagnostic = feature_diagnostics(x, stage, ~held, held)
    assert diagnostic['heldout']['split_cell_repeatability'] > .99
    assert abs(diagnostic['heldout']['stage_residual_repeatability']) < 1e-6


def test_stage_removal_cannot_manufacture_cross_model_donor_deviations():
    stage = np.eye(4)[np.arange(12) % 4]
    train = np.arange(12) < 8
    constant = np.broadcast_to(np.arange(7, dtype=float), (12, 7))
    stage_only = stage @ np.arange(28).reshape(4, 7)
    np.testing.assert_allclose(remove_train_stage_effect(constant, stage, train), 0., atol=1e-12)
    np.testing.assert_allclose(remove_train_stage_effect(stage_only, stage, train), 0., atol=1e-12)
