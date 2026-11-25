"""The held-out learning-rate fit runs only after validation and refinement finish."""
import pytest

from alztest.neuroko_lr import RATES, finer_rates, refined_selection, selected_rate


def receipt(scores, group="Screen1_0"):
    rows = [{"base_rate": rate, "mean_energy_distance": value} for rate, value in sorted(scores.items())]
    return {"stage": "validation", "held_out_group": group, "rates": rows, "selected_rate": selected_rate(rows)}


def coarse():
    return {rate: abs(rate - 0.028) + 1 for rate in RATES}


def test_heldout_refuses_a_coarse_only_validation():
    with pytest.raises(ValueError, match="run the refine stage"):
        refined_selection(receipt(coarse()), "Screen1_0")


def test_heldout_uses_the_refined_selection():
    scores = coarse()
    lower, upper = finer_rates(receipt(scores)["rates"])
    scores.update({lower: 0.5, upper: 2.0})
    assert refined_selection(receipt(scores), "Screen1_0") == lower


def test_heldout_refuses_an_incomplete_coarse_grid_or_another_group():
    scores = coarse()
    del scores[RATES[-1]]
    with pytest.raises(ValueError, match="run the validation stage"):
        refined_selection(receipt(scores), "Screen1_0")
    with pytest.raises(ValueError, match="own validation receipt"):
        refined_selection(receipt(coarse(), "Screen1_1"), "Screen1_0")


def test_heldout_refuses_a_selection_that_disagrees_with_its_scores():
    scores = coarse()
    scores.update(dict.fromkeys(finer_rates(receipt(scores)["rates"]), 3.0))
    tuning = receipt(scores)
    tuning["selected_rate"] = RATES[0]
    with pytest.raises(ValueError, match="differs from its rate scores"):
        refined_selection(tuning, "Screen1_0")
