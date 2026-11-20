"""Data-free checks for donor splits, projection, and disease specificity."""
import numpy as np
from scipy import sparse

from paper_watch.alztest.psychad import (
    GROUPS, donor_profiles, project_profiles, score, split_donors,
)


def test_psychad_profiles_and_split_are_donor_unit():
    groups = np.repeat(GROUPS, 28)
    donors = [f"donor-{index:03d}" for index in range(len(groups))]
    counts = sparse.csr_matrix(np.tile([2, 1, 0], (2 * len(donors), 1)))
    profiles, labels = donor_profiles(counts, np.repeat(donors, 2), np.repeat(groups, 2))
    train, held = split_donors(labels, 7)

    assert profiles.shape == (len(donors), 3)
    assert len(set(train) & set(held)) == 0
    assert sorted(np.bincount(np.searchsorted(sorted(GROUPS), labels[held]))) == [7, 7, 7]


def test_psychad_projection_uses_training_donors_only():
    raw = np.random.default_rng(0).uniform(size=(24, 7))
    train = np.arange(18)
    original = project_profiles(sparse.csr_matrix(raw), train, max_genes=4)
    raw[18:] *= 500
    changed = project_profiles(sparse.csr_matrix(raw), train, max_genes=4)
    np.testing.assert_array_equal(original[train], changed[train])
    assert not np.array_equal(original[18:], changed[18:])


def test_psychad_specificity_penalizes_swapped_diseases():
    real = {
        "control": np.zeros((8, 2)),
        "Alzheimer": np.tile([1, 0], (8, 1)),
        "Lewy": np.tile([0, 1], (8, 1)),
    }
    swapped = {**real, "Alzheimer": real["Lewy"], "Lewy": real["Alzheimer"]}
    assert score(real, real)["specificity"] == 2
    assert score(swapped, real)["specificity"] == -2
