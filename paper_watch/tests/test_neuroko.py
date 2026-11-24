"""Regression checks for the NeuroKO evaluation and release metadata mapping."""
from contextlib import ExitStack

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix
import torch

from alztest.neuroko import LAYOUT, evaluate, normalize_counts, open_screen, project_fold, select_targets


def test_test_cells_use_training_coordinates():
    train = csr_matrix(np.array([[0., 0.], [3., 1.], [2., 1.], [1., 0.], [4., 3.]]))
    test = train[[2, 0, 3]]
    fit, projected = project_fold(train, test, n_features=2, n_components=2)
    np.testing.assert_allclose(projected, fit[[2, 0, 3]])
    _, shifted = project_fold(train, test * 10, n_features=2, n_components=2)
    assert not np.allclose(projected, shifted)


def test_normalization_uses_complete_library():
    matrix = csr_matrix([[1, 3], [4, 4]])
    result = normalize_counts(matrix)
    np.testing.assert_allclose(result.toarray(), np.log1p([[2500, 7500], [5000, 5000]]))
    np.testing.assert_array_equal(matrix.toarray(), [[1, 3], [4, 4]])
    with pytest.raises(ValueError, match="integer-valued"):
        normalize_counts(csr_matrix([[0.5, 1.0]]))


@pytest.fixture
def mock_release(tmp_path):
    import anndata as ad

    rng = np.random.default_rng(0)
    for name, cols in LAYOUT.items():
        rows, counts = [], []
        for group in ("g1", "g2"):
            for idx, target in enumerate(("INT", "A", "B")):
                for j in range(40):
                    rows.append((target, f"{target}_{j % 2}", group))
                    levels = np.full(6, 3.0)
                    levels[idx] = 8
                    counts.append(rng.poisson(levels) + 1)
        obs = pd.DataFrame(rows, columns=cols, index=[str(i) for i in range(len(rows))])
        var = pd.DataFrame(index=["A", "B", "C", "D", "E", "F"])
        ad.AnnData(csr_matrix(np.array(counts)), obs=obs, var=var).write_h5ad(tmp_path / f"{name}.h5ad")
    return tmp_path


def test_backed_group_holdouts_and_control_arms(mock_release):
    torch.set_num_threads(2)
    with ExitStack() as stack:
        screens = [stack.enter_context(open_screen(mock_release, name, window_cells=116)) for name in LAYOUT]
        for screen in screens:
            assert screen.inventory()["pilot_intervals"] == [[0, 116], [120, 236]]
            assert screen.inventory()["control_cells_by_group"] == {"g1": 40, "g2": 40}
        assert select_targets(screens, 32, 6, ["B", "A"]) == ["B", "A"]
        with pytest.raises(ValueError, match="requested targets below 38"):
            select_targets(screens, 38, 6, ["A", "B"])
        report = evaluate(screens, min_cells=32, cap=40, max_targets=2, steps=2, seeds=(0,))
    assert report["targets"] == ["A", "B"]
    assert report["transfer"]["common_unique_gene_symbols"] == 6
    assert report["transfer"]["genes_after_excluding_panel_targets"] == 4
    for folds in report["folds"].values():
        assert [fold["held_out_group"] for fold in folds] == ["g1", "g2"]
        assert [row["test_cells"] for row in folds[0]["targets"]] == [40, 36]
        for fold in folds:
            for row in fold["targets"]:
                assert row["train_cells"] == row["test_cells"]
                assert row["expression_matched_target"] != row["target"]
                assert set(row["scores"]) == {
                    "control_only", "mean_shift", "empirical_target", "diagonal_gaussian",
                    "empf", "shuffled_empf", "expression_matched_target",
                }
                assert row["scores"]["control_only"]["shift_cosine"] is None
                assert np.isfinite(row["scores"]["empf"]["energy_distance"])


def test_fetch_promotes_only_verified_bytes(tmp_path, monkeypatch):
    from hashlib import md5

    from paper_watch.scripts import fetch_neuroko

    released = b"abcde"
    screen = "iNeurons"
    monkeypatch.setitem(fetch_neuroko.FILES, screen, (len(released), md5(released).hexdigest()))
    partial = tmp_path / f"{screen}.h5ad.part"
    partial.write_bytes(released[:2])
    downloaded = []

    def resume(command, *, check):
        assert check and command[command.index("--continue-at") + 1] == "-"
        assert command[command.index("--output") + 1] == str(partial)
        assert partial.read_bytes() == released[:2]
        partial.write_bytes(released)
        downloaded.append(command[-1])

    monkeypatch.setattr(fetch_neuroko.subprocess, "run", resume)
    path = fetch_neuroko.fetch(tmp_path, screen)
    assert path.read_bytes() == released
    assert not partial.exists()
    assert fetch_neuroko.fetch(tmp_path, screen) == path
    assert len(downloaded) == 1
    path.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="checksum mismatch"):
        fetch_neuroko.fetch(tmp_path, screen)


def test_rate_selection_keeps_outer_group_out_of_preprocessing_and_validation():
    from types import SimpleNamespace

    from alztest.neuroko_lr import validation_split

    panel = SimpleNamespace(groups=np.array(["a", "b", "c", "d"]))
    fit, validation, description = validation_split(panel, "b", ["INT", "A"])
    np.testing.assert_array_equal(fit, [True, False, False, True])
    np.testing.assert_array_equal(validation, [False, False, True, False])
    assert description == {"validation_group": "c"}


def test_rate_selection_in_one_batch_holds_out_whole_vectors():
    from types import SimpleNamespace

    from alztest.neuroko_lr import validation_split

    panel = SimpleNamespace(
        groups=np.repeat(["a", "b"], 192),
        labels=np.tile(np.repeat([0, 1], 96), 2),
        vectors=np.tile(np.repeat(["i1", "i2", "i3", "A1", "A2", "A3"], 32), 2),
    )
    fit, validation, description = validation_split(panel, "a", ["INT", "A"])
    assert not np.any((fit | validation) & (panel.groups == "a"))
    assert set(panel.vectors[fit]).isdisjoint(panel.vectors[validation])
    assert validation.sum() == 64
    assert fit.sum() == 128
    assert description["validation_group"] == "b"


def test_rate_refinement_brackets_an_upper_bound_winner():
    from alztest.neuroko_lr import RATES, finer_rates

    records = [{"base_rate": rate, "mean_energy_distance": 1 / rate} for rate in RATES]
    assert finer_rates(records) == (0.0792, 0.224)


def test_fixed_rate_fit_reuses_only_a_fully_verified_receipt(tmp_path):
    import json
    from paper_watch.scripts.compare_neuroko_v12 import reusable_fit

    path = tmp_path / "fit.json"
    assert not reusable_fit(path, True)
    cached = {"updates": 1200, "rate_trace": [{"update": 1200, "generator": 0.014}]}
    path.write_text(json.dumps(cached))
    assert reusable_fit(path, False)
    assert not reusable_fit(path, True)
    for count in (1199, 1200):
        path.write_text(json.dumps(dict(cached, constant_lr_verified_updates=count)))
        assert reusable_fit(path, True) == (count == 1200)
