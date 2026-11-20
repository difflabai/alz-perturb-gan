"""CPU regression checks for the paper-watch harness."""
from types import SimpleNamespace

import numpy as np
import pytest

from paper_watch.alztest.harness import BATCH, ConditionalPGAN


def test_fit_keeps_condition_ids_and_fixed_batch_size():
    model = ConditionalPGAN(data_dim=2, n_cond=3, seed=7)
    labels = np.repeat([0, 1, 2], [8, 33, BATCH])
    values = np.broadcast_to(labels[:, None], (len(labels), 2))
    seen = []

    def record(batch):
        expected = model.G.cond.argmax().item()
        assert model.D.cond.argmax().item() == expected
        assert batch.shape == (BATCH, 2)
        assert (batch == expected).all()
        seen.append(expected)

    model.trainer = SimpleNamespace(step=record)
    model.fit(values, labels, steps=4)

    assert sorted(seen) == [1, 1, 2, 2]


def test_fit_rejects_an_empty_set_of_supported_conditions():
    model = ConditionalPGAN(data_dim=2, n_cond=2)
    with pytest.raises(ValueError, match="at least 32 rows"):
        model.fit(np.zeros((5, 2)), np.zeros(5, dtype=int), steps=1)


def test_released_empf_trains_and_draws_on_cpu():
    model = ConditionalPGAN(data_dim=2, n_cond=2)
    labels = np.repeat([0, 1], BATCH)
    values = np.random.default_rng(0).normal(size=(len(labels), 2))
    model.fit(values, labels, steps=2)
    draws = model.sample(0, 8)
    assert draws.shape == (8, 2)
    assert np.isfinite(draws).all()


# ------------------------------------------------------------ metrics and inference

from scipy.stats import kstest  # noqa: E402

from paper_watch.alztest.data import adnc_group  # noqa: E402
from paper_watch.alztest.harness import (  # noqa: E402
    donor_bootstrap_ci,
    energy_distance,
    grouped_permutations,
    pca_apply,
    pca_fit,
    pca_fit_transform,
    permutation_pvalue,
    standardize,
    standardize_apply,
    standardize_fit,
)


@pytest.mark.parametrize("m", [25, 80, 600])
def test_unbiased_energy_distance_is_near_zero_for_one_distribution(m):
    rng = np.random.default_rng(m)
    unbiased, biased = [], []
    for rep in range(100 if m < 600 else 40):
        model = rng.normal(size=(512, 30))
        donor = rng.normal(size=(m, 30))
        unbiased.append(energy_distance(model, donor, seed=rep))
        biased.append(energy_distance(model, donor, seed=rep, unbiased=False))
    # The U-statistic has expectation 0 whatever m is; the V-statistic
    # carries roughly E|X - X'| / m (about 7.7 / m in 30-d N(0, I)).
    assert abs(np.mean(unbiased)) < 0.02
    assert np.mean(biased) > 0.9 * 7.7 * (1 / m + 1 / 512) - 0.02


def test_unbiased_energy_distance_does_not_track_donor_depth():
    # The bias the review found: same distribution, different cell counts.
    rng = np.random.default_rng(0)
    depths = np.repeat([25, 40, 80, 200, 600], 30)
    model = rng.normal(size=(512, 10))
    v = [energy_distance(model, rng.normal(size=(m, 10)), unbiased=False) for m in depths]
    u = [energy_distance(model, rng.normal(size=(m, 10))) for m in depths]
    from scipy.stats import spearmanr
    assert spearmanr(v, depths).statistic < -0.8
    assert abs(spearmanr(u, depths).statistic) < 0.3


def test_energy_distance_detects_a_shift_and_handles_max_n():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(300, 5))
    b = rng.normal(size=(300, 5)) + 1.0
    assert energy_distance(a, b) > 0.5
    assert energy_distance(a, b, max_n=None) == pytest.approx(
        energy_distance(a, b, max_n=300))
    with pytest.raises(ValueError, match="at least 2 rows"):
        energy_distance(a, b[:1])


def test_pca_fit_apply_round_trip():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(60, 8)) @ rng.normal(size=(8, 8))
    projected, parts = pca_fit_transform(X, n_comp=8)
    np.testing.assert_allclose(pca_apply(X, parts), projected)
    mean, comps = pca_fit(X, n_comp=8)
    np.testing.assert_allclose(pca_apply(X, (mean, comps)) @ comps + mean, X, atol=1e-10)
    # A basis fit on training rows ignores held-out rows entirely.
    train = np.arange(40)
    held = X.copy()
    held[40:] += 100.0
    np.testing.assert_allclose(pca_fit(held[train], 3)[1], pca_fit(X[train], 3)[1])
    np.testing.assert_allclose(standardize_apply(X, standardize_fit(X)), standardize(X))


def _null_units(rng, n_donors=24):
    """Donor-level label, 2-4 units per donor, strong donor random effect."""
    label = rng.permutation(np.arange(n_donors) % 3 == 0).astype(int)
    effect = rng.normal(size=n_donors) * 2.0
    reps = rng.integers(2, 5, size=n_donors)
    donors = np.repeat(np.arange(n_donors), reps)
    score = effect[donors] + 0.3 * rng.normal(size=len(donors))
    return score, label[donors], donors


def _mean_difference(score):
    def stat(labels):
        return score[labels == 1].mean() - score[labels == 0].mean()
    return stat


def test_grouped_permutation_pvalue_is_uniform_under_the_null():
    rng = np.random.default_rng(3)
    grouped, per_unit = [], []
    for sim in range(500):
        score, labels, donors = _null_units(rng)
        stat = _mean_difference(score)
        grouped.append(permutation_pvalue(stat, labels, donors, n_perm=199, seed=sim)["p"])
        per_unit.append(permutation_pvalue(stat, labels, None, n_perm=199, seed=sim)["p"])
    grouped, per_unit = np.array(grouped), np.array(per_unit)
    assert kstest(grouped, "uniform").pvalue > 0.01
    assert 0.02 < np.mean(grouped <= 0.05) < 0.09
    # Permuting units separately breaks the donor pairing; with a donor
    # random effect the null is too narrow and small p-values are too common.
    assert np.mean(per_unit <= 0.05) > 0.12


def test_grouped_permutation_keeps_donor_records_together():
    labels = np.array([1, 1, 0, 0, 0, 1])
    donors = np.array(["a", "a", "b", "c", "c", "d"])
    for perm in grouped_permutations(labels, donors, n_perm=50, seed=0):
        assert perm[0] == perm[1] and perm[3] == perm[4]
        assert sorted(perm[[0, 2, 3, 5]]) == [0, 0, 1, 1]
    with pytest.raises(ValueError, match="labels differ within a group"):
        next(grouped_permutations(np.arange(6), donors))


def test_grouped_permutation_with_per_region_labels():
    donors = np.array(["a", "a", "b", "b", "c", "d", "e"])
    region = np.array(["x", "y", "x", "y", "x", "x", "y"])
    target = np.array([10, 11, 20, 21, 30, 40, 51])
    seen = set()
    for perm in grouped_permutations(target, donors, within=region, n_perm=200, seed=1):
        # a, b share the {x, y} pattern; c, d share {x}; e is alone.
        assert {tuple(perm[:2]), tuple(perm[2:4])} == {(10, 11), (20, 21)}
        assert set(perm[4:6]) == {30, 40}
        assert perm[6] == 51
        seen.add(tuple(perm))
    assert len(seen) == 4
    with pytest.raises(ValueError, match="identify one unit"):
        next(grouped_permutations(target, donors, within=np.zeros(7)))


def test_permutation_pvalue_detects_a_real_donor_effect():
    rng = np.random.default_rng(4)
    score, labels, donors = _null_units(rng, n_donors=40)
    score = score + 3.0 * labels
    out = permutation_pvalue(_mean_difference(score), labels, donors, n_perm=499)
    assert out["p"] < 0.01
    assert out["observed"] > out["null_q95"]


def test_donor_bootstrap_ci_covers_the_population_mean():
    rng = np.random.default_rng(5)
    covered = 0
    for sim in range(200):
        donors = np.repeat(np.arange(30), 2)
        score = rng.normal(size=30)[donors] + 0.1 * rng.normal(size=60)
        ci = donor_bootstrap_ci(lambda idx: score[idx].mean(), donors,
                                n_boot=400, seed=sim)
        covered += ci["lo"] <= 0.0 <= ci["hi"]
    # Nominal 95 %; percentile intervals at 30 donors run slightly short.
    assert 0.88 <= covered / 200 <= 0.99


def test_donor_bootstrap_drops_single_class_replicates():
    donors = np.array([0, 1, 2])
    labels = np.array([1, 0, 0])

    def stat(idx):
        return None if len(set(labels[idx])) < 2 else float(labels[idx].mean())

    ci = donor_bootstrap_ci(stat, donors, n_boot=200, seed=0)
    assert ci["n_valid"] < 200
    assert 0 < ci["lo"] <= ci["hi"] < 1


def test_adnc_mapping_is_explicit():
    groups = adnc_group(["Not AD", "Low", "Intermediate", "High"])
    assert list(groups) == ["CTRL", "CTRL", "EARLY", "AD"]
    with pytest.raises(ValueError, match="missing"):
        adnc_group(["High", np.nan])
    with pytest.raises(ValueError, match="unexpected ADNC"):
        adnc_group(["High", "Reference"])
