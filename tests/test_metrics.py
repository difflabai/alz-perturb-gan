import math

import torch

from alz_perturb_gan.metrics import (
    des_overlap,
    expression_report,
    pearson_delta,
    perturbation_discrimination,
    phenocopy_rank,
)


def test_pearson_delta_is_one_for_a_scaled_effect():
    control = torch.zeros(4)
    true = torch.tensor([1.0, -2.0, 0.5, 0.0])
    pred = 0.5 * true
    assert abs(pearson_delta(pred, true, control) - 1.0) < 1e-5


def test_no_change_overlap_is_zero():
    control = torch.ones(6)
    true = control + torch.tensor([0.0, 3.0, 0.0, 2.0, 0.0, 1.0])
    assert des_overlap(control, true, control, k=2) == 0.0


def test_discrimination_is_perfect_when_prediction_matches():
    true = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    assert perturbation_discrimination(true, true) == 1.0


def test_phenocopy_tie_is_not_a_top1():
    scores = torch.tensor([0.2, 0.2, 0.2])
    row = phenocopy_rank(scores, 1)
    assert row["top1"] == 0.0
    assert row["rank"] == 2.0
    unique = phenocopy_rank(torch.tensor([0.1, 0.9, 0.2]), 1)
    assert unique["top1"] == 1.0
    assert unique["mrr"] == 1.0


def test_expression_report_shapes():
    control = torch.zeros(4)
    true = torch.eye(3, 4)
    report = expression_report(true, true, control, k=2)
    assert report["mae"] == 0.0
    assert report["pds"] == 1.0
    assert report["n_profiles"] == 3.0
    assert not math.isnan(report["pearson_delta"])
