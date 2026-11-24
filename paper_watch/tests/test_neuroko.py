"""Regression checks for the NeuroKO evaluation and release metadata mapping."""
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix
import torch

from alztest.neuroko import LAYOUT, evaluate, normalize_counts, open_screen, project_fold, select_targets


def test_critic_rate_interpolation_checks_the_same_v11_reference(tmp_path, monkeypatch):
    import json

    paper_watch = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(paper_watch / "scripts"))
    from neuroko_rate_noise import calibrated_critic

    baseline = paper_watch / "notes/neuroko_v12_critic_motion.json"
    check = paper_watch / "notes/neuroko_v12_critic_motion_check.json"
    trial, calibration = calibrated_critic(baseline, check)
    assert calibration["lower_critic_lr"] == .0035
    assert trial["lr"] * trial["d_lr_mult"] == calibration["fixed_critic_lr"] == .0046
    different_probe = json.loads(check.read_text())
    different_probe["cells"][0]["probe_indices_sha256"] = "different"
    incompatible = tmp_path / "incompatible.json"
    incompatible.write_text(json.dumps(different_probe))
    with pytest.raises(ValueError, match="identical v11 probes"):
        calibrated_critic(baseline, incompatible)


def test_prior_weight_uses_the_paired_particle_gradient_change(tmp_path, monkeypatch):
    import json

    paper_watch = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(paper_watch / "scripts"))
    from neuroko_rate_noise import NOISY_PRIOR, balanced_prior

    trial = dict(NOISY_PRIOR, name="motion-critic-46", d_lr_mult=.0046 / .014)
    path = paper_watch / "notes/neuroko_v12_particle_balance.json"
    balanced, scale = balanced_prior(path, trial)
    assert balanced["prior_reg"] == round(.1 / scale, 3) == .041
    assert balanced["lr"] * balanced["d_lr_mult"] == .0046
    bad = json.loads(path.read_text())
    bad["cells"][0]["v12_calibrated"]["ratios_by_update"].pop()
    partial = tmp_path / "partial.json"
    partial.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="Incomplete paired particle gradients"):
        balanced_prior(partial, trial)


@pytest.mark.parametrize("wins", (True, False))
def test_individual_role_rate_shortlist_preserves_other_rates(tmp_path, monkeypatch, wins):
    from types import SimpleNamespace

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    import neuroko_rate_noise as study

    def evaluate(args, arms, seeds):
        assert seeds == [22, 23, 24]
        rates = []
        for _, recipe, trial in arms:
            assert recipe == "unified_v12"
            gen = trial["lr"]
            rates.append((gen, gen * trial["d_lr_mult"], gen * trial["prior_lr_mult"]))
            assert (trial["prior_sigma"], trial["prior_reg"]) == (.15, .1)
        assert rates == [(.014, .021, .0406), (.010, .021, .0406), (.020, .021, .0406),
                         (.014, .021, .0287), (.014, .021, .0574)]
        values = [1.0, .9 if wins else 1.0, 1.1, 1.05, 1.02]
        return {study.key(arm): [value] * 9 for arm, value in zip(arms, values, strict=True)}

    monkeypatch.setattr(study, "evaluate", evaluate)
    args = SimpleNamespace(output=tmp_path, shared_screen=None, stage="search")
    chosen = study.individual_rates(args)
    if wins:
        assert chosen["name"] == "noise150-prior100-g010"
        args.stage = "confirm"
        assert study.individual_rates(args) == chosen
    else:
        assert chosen is None
        args.stage = "confirm"
        with pytest.raises(ValueError, match="No individual role rate passed"):
            study.individual_rates(args)


def test_wide_noise_rivals_can_be_measured_without_promoting_failed_candidates(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    import neuroko_rate_noise as study

    calls = []

    def evaluate(args, arms, seeds):
        calls.append((list(seeds), [a[2]["name"] for a in arms]))
        return {study.key(a): [1.] * 9 for a in arms}

    monkeypatch.setattr(study, "evaluate", evaluate)
    args = SimpleNamespace(output=tmp_path, shared_screen=None, second_screen=None, stage="search")
    assert study.individual_rates(args) is None
    study.compare_noise_controls(args)
    first = json.loads((tmp_path / "wide_noise_first_rates.json").read_text())
    assert len(first["differences"]) == 4
    assert all(len(comparators) == 2 for comparators in first["differences"].values())
    assert (tmp_path / "wide_noise_second_controls.json").is_file()
    assert not (tmp_path / "wide_noise_second_rates.json").exists()
    assert calls[-1] == ([251, 252, 253], ["noise150-prior100", "noise250-prior100", "noise250-prior100"])


def test_shared_prior_reuses_only_identical_native_recipes(monkeypatch):
    from importlib.metadata import version

    pytest.importorskip("particlegan")
    if version("particlegan") != "0.12.0":
        pytest.skip("requires ParticleGAN 0.12.0")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from neuroko_matched_rate_replication import TRIAL, comparison, prior_equivalence

    matches, _ = comparison(TRIAL, False)
    checked = prior_equivalence("0.12.0", matches)
    for name in ("unified_v12", "unified_v11"):
        assert checked[name] == dict(native_weight=.02, shared_weight=.02, identical_full_recipe=True)
    assert checked["empf"] == dict(native_weight=.01, shared_weight=.02, identical_full_recipe=False)


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
