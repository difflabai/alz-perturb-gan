"""Regression checks for the two-program readout, selection, and control arms."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from paper_watch.alztest.psychad import MIN_TRAIN_DONORS
from paper_watch.alztest.psychad_programs import (
    ARMS, GROUPS, GBA1_ENSEMBL, LIPID, LYSOSOME, METRICS, donor_bootstrap, fit_checkpoints,
    inner_split, marker_expression, program_scores, run, select_configuration, split_half_noise,
)


def synthetic_psychad(donors_per_group=32, seed=0):
    """Return an AnnData-like donor-level extract with the program marker columns."""
    rng = np.random.default_rng(seed)
    markers = [*LIPID, *LYSOSOME]
    codes = [GBA1_ENSEMBL if gene == "GBA1" else f"gene-{i}" for i, gene in enumerate(markers)]
    groups = np.repeat(GROUPS, donors_per_group)
    shift = {group: k for k, group in enumerate(GROUPS)}
    counts = rng.poisson(20 + 5 * np.array([shift[g] for g in groups])[:, None], (len(groups), len(markers)))
    return SimpleNamespace(
        X=sparse.csr_matrix(counts.astype(float)),
        obs={"donor_id": [f"donor-{i:03d}" for i in range(len(groups))], "diagnosis": groups},
        var=pd.DataFrame({"feature_name": markers, "feature_id": codes}),
        shape=counts.shape,
    )


def test_gba_marker_uses_the_ensembl_id():
    markers = [*LIPID, *LYSOSOME, "GBA1"]
    codes = [f"gene-{i}" for i in range(len(markers))]
    gba_index = markers.index("GBA1")
    codes[gba_index] = GBA1_ENSEMBL
    markers[gba_index] = "GBA"
    counts = np.ones((3, len(markers)))
    counts[:, gba_index] = 3
    data = SimpleNamespace(
        X=sparse.csr_matrix(counts),
        obs={"donor_id": ["c", "d", "l"], "diagnosis": GROUPS},
        var=pd.DataFrame({"feature_name": markers, "feature_id": codes}),
    )
    expression, groups = marker_expression(data)
    assert expression.shape == (3, 15)
    np.testing.assert_array_equal(groups, GROUPS)
    np.testing.assert_allclose(expression[:, gba_index], np.log1p(3 * 1e4 / counts.sum(1)))


def test_program_standardization_uses_training_donors():
    expression = np.random.default_rng(0).normal(size=(20, len(LIPID) + len(LYSOSOME)))
    train = np.arange(15)
    scores = program_scores(expression, train)
    expression[15:] += 100
    changed = program_scores(expression, train)

    np.testing.assert_array_equal(scores[train], changed[train])
    assert np.all(changed[15:] > scores[15:])


def test_checkpoint_sampling_preserves_fit_and_audits_regularizer():
    rng = np.random.default_rng(7)
    train = rng.normal(size=(72, 2))
    labels = np.repeat(np.arange(len(GROUPS)), 24)
    observed = {group: rng.normal(size=(8, 2)) for group in GROUPS}

    earlier = fit_checkpoints(train, labels, observed, seed=3, rate=0.01, checkpoints=(2, 4))
    terminal = fit_checkpoints(train, labels, observed, seed=3, rate=0.01, checkpoints=(4,))

    assert earlier["peak_rates"] == {"generator": 0.01, "critic": 0.015, "prior": 0.025}
    assert [row["regularizer_calls"] for row in earlier["checkpoints"]] == [2, 4]
    assert earlier["checkpoints"][-1] == terminal["checkpoints"][-1]


def test_inner_split_keeps_minimum_fitting_donors_and_is_disjoint():
    # 25 Lewy training donors (the real outer split) leave 5 for validation.
    groups = np.array(["control"] * 195 + ["Alzheimer"] * 243 + ["Lewy"] * 25)
    fit, validation = inner_split(groups, 1000)

    assert len(set(fit) & set(validation)) == 0
    assert sorted([*fit, *validation]) == list(range(len(groups)))
    counts = {g: int(np.sum(groups[validation] == g)) for g in GROUPS}
    assert counts == {"control": 48, "Alzheimer": 60, "Lewy": 5}
    assert all(np.sum(groups[fit] == g) >= MIN_TRAIN_DONORS for g in GROUPS)

    with pytest.raises(ValueError, match="inner validation"):
        inner_split(np.array(["control"] * 40 + ["Alzheimer"] * 40 + ["Lewy"] * 22), 0)


def test_selection_uses_metric_then_energy_and_skips_undefined_correlations():
    def row(error, energy):
        return {"mean_absolute_correlation_error": error, "mean_energy_distance": energy}

    grid = {
        "0.005": {"300": row(0.30, 0.10), "1200": row(0.20, 0.09)},
        "0.012": {"300": row(None, 0.01), "1200": row(0.20, 0.05)},
    }
    assert select_configuration(grid) == ("0.012", "1200")
    with pytest.raises(ValueError):
        select_configuration({"0.005": {"300": row(None, 0.1)}})


def bootstrap_splits(seed=0):
    rng = np.random.default_rng(seed)
    groups = np.repeat(GROUPS, 20)
    held = np.concatenate([np.flatnonzero(groups == g)[:8] for g in GROUPS])
    observed = {g: rng.normal(size=(8, 2)) for g in GROUPS}
    generated = {g: rng.normal(size=(64, 2)) for g in GROUPS}
    samples = {arm: generated for arm in ARMS}
    return groups, [{"held": held, "held_groups": groups[held], "observed": observed, "samples": samples}]


def test_donor_bootstrap_intervals_and_identical_arm_differences():
    groups, splits = bootstrap_splits()
    intervals, differences, record = donor_bootstrap(groups, splits, replicates=100, seed=3)
    again = donor_bootstrap(groups, splits, replicates=100, seed=3)

    assert (intervals, differences, record) == again
    assert 50 <= record["used"] <= 100
    for arm in ARMS:
        for metric in METRICS:
            assert intervals[arm][metric]["bootstrap_se"] > 0
            assert "ci95" not in intervals[arm][metric]
    # Every arm here reuses the same generated draws, so paired differences vanish.
    assert all(
        row["ci95"] == [0.0, 0.0] and row["bootstrap_se"] == 0
        for by_metric in differences.values() for row in by_metric.values()
    )


def test_split_half_noise_compares_disjoint_halves_of_observed_donors():
    identical = {g: np.tile([[1.0, 2.0]], (8, 1)) for g in GROUPS}
    floor = split_half_noise([identical], repeats=5)
    assert floor["energy_comparisons"] == 15
    assert floor["mean_energy_distance"] == pytest.approx(0, abs=1e-12)
    assert floor["correlation_comparisons"] == 0
    assert floor["mean_absolute_correlation_error"] is None

    varied = {g: np.random.default_rng(k).normal(size=(9, 2)) for k, g in enumerate(GROUPS)}
    floor = split_half_noise([varied], repeats=5)
    assert floor["mean_energy_distance"] > 0
    assert floor["correlation_comparisons"] == 15


def test_run_selects_on_training_donors_and_reports_control_arms():
    data = synthetic_psychad()
    result = run(data, seeds=(0,), rates=(0.01, 0.02), checkpoints=(2, 4), bootstrap_replicates=100)

    assert result["selected_on"] == "inner_train_split"
    assert set(result["summary"]) == set(ARMS)
    for arm in ARMS:
        assert set(result["summary"][arm]["bootstrap"]) == set(METRICS)
    assert set(result["paired_differences"]) == {f"empf_minus_{arm}" for arm in ARMS[1:]}
    for by_metric in result["paired_differences"].values():
        for row in by_metric.values():
            assert set(row) == {"estimate", "bootstrap_mean", "bootstrap_se", "ci95"}
    assert result["noise_floor"]["kind"] == "observed_split_half"
    split = result["splits"][0]
    selected = split["selection"]["selected"]
    assert selected["learning_rate"] in (0.01, 0.02) and selected["updates"] in (2, 4)
    assert [row["updates"] for row in split["fits"]["empf"]["checkpoints"]] == [selected["updates"]]
    assert split["fits"]["empf"]["peak_rates"]["generator"] == selected["learning_rate"]
    assert split["fits"]["shuffled"]["peak_rates"] == split["fits"]["empf"]["peak_rates"]
    json.dumps(result, allow_nan=False)

    # Changing only the held-out donors' counts cannot change the selection record.
    from paper_watch.alztest.psychad import donor_profiles, split_donors
    _, groups = donor_profiles(data.X, data.obs["donor_id"], data.obs["diagnosis"])
    _, held = split_donors(groups, 0)
    counts = data.X.toarray()
    counts[held] *= np.random.default_rng(1).uniform(0.2, 5.0, size=(len(held), counts.shape[1]))
    changed = run(
        SimpleNamespace(X=sparse.csr_matrix(counts), obs=data.obs, var=data.var, shape=data.shape),
        seeds=(0,), rates=(0.01, 0.02), checkpoints=(2, 4), bootstrap_replicates=100,
    )
    assert changed["splits"][0]["selection"] == split["selection"]
    assert changed["splits"][0]["observed"] != split["observed"]
