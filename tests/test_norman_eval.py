"""Small CPU fixtures; no biological data or torch dependencies."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from alz_perturb_gan.norman_eval import (
    Packet, array_hash, digest, evaluate, file_hash, leakage_audit, load_packet,
    main, make_splits, members, novelty, paired_bootstrap, predict_baselines,
    score_predictions,
)


@pytest.fixture
def packet():
    genes = [f"G{i}" for i in range(10)]
    labels = genes + [f"G{i}+G{j}" for i in range(10) for j in range(i + 1, 10)]
    rng = np.random.default_rng(21)
    effects = rng.normal(size=(10, 12))
    delta = np.stack([sum((effects[genes.index(g)] for g in members(p)),
                         start=np.zeros(12)) for p in labels])
    return Packet(labels, [f"E{i}" for i in range(12)], delta, genes,
                  rng.normal(size=(len(labels), 10)), {"fixture": True}, np.abs(delta))


@pytest.mark.parametrize("protocol", ["perturbation_label", "whole_gene"])
def test_reproducible_row_order_invariant_folds(packet, protocol):
    folds = make_splits(packet.labels, protocol, 31)
    assert folds == make_splits(packet.labels[::-1], protocol, 31)
    assert folds != make_splits(packet.labels, protocol, 32)
    assert len(folds) == 5
    test = [p for f in folds for p in f["test"]]
    assert len(test) == len(set(test))
    for f in folds:
        assert f["sha256"] == digest({k: v for k, v in f.items() if k != "sha256"})
        assert f["audit"]["passed"]
        if protocol == "whole_gene":
            assert not f["audit"]["perturbed_gene_overlap"]
            assert all(f["audit"]["novelty"][p]["n_unseen_genes"] == len(members(p))
                       for p in f["test"])
    if protocol == "perturbation_label":
        assert set(test) == set(packet.labels)
        assert all(sum(len(members(p)) == 1 for p in f["test"]) == 2 for f in folds)
    else:
        assert len(test) == 15  # 10 singles, 5 within-block doubles
        assert all(len(f["excluded"]) == 16 for f in folds)


@pytest.mark.parametrize("labels", [["A+B", "B+A"], ["A+A"], ["control"], ["A+"], [" A"]])
def test_invalid_labels(labels):
    with pytest.raises(ValueError):
        make_splits(labels, "perturbation_label", 1)


def test_audit_fails_on_gene_leak_and_malformed_membership(packet):
    f = make_splits(packet.labels, "whole_gene", 1)[0]
    bad = copy.deepcopy(f)
    bad["train"].append(bad["excluded"].pop())
    with pytest.raises(ValueError, match="leaks"):
        leakage_audit(packet.labels, bad)
    bad = copy.deepcopy(f)
    bad["train"].append(bad["test"][0])
    with pytest.raises(ValueError, match="partition"):
        leakage_audit(packet.labels, bad)


def test_novelty_distinguishes_labels_genes_and_combinations():
    assert novelty("A+B", ["A", "B"])["stratum"] == "combination_0_unseen_genes"
    assert novelty("A+B", ["A"])["stratum"] == "combination_1_unseen_genes"
    assert novelty("A+B", ["C"])["stratum"] == "combination_2_unseen_genes"
    n = novelty("A", ["A+B"])
    assert n["unseen_label"] and n["stratum"] == "single_seen_gene"


def test_baseline_exact_mean_additive_and_zero_fallback():
    y = np.array([[1., 2., 3.], [2., 4., 6.]])
    pred, fit = predict_baselines(["A", "B"], y, y[:, :2], ["A", "B"],
                                 ["A+B", "A+C", "C"])
    np.testing.assert_allclose(pred["mean_effect"], np.tile(y.mean(0), (3, 1)))
    np.testing.assert_allclose(pred["additive"], [y.sum(0), y[0], np.zeros(3)])
    assert fit["additive_missing_single_count"] == {"A+B": 0, "A+C": 1, "C": 1}
    assert fit["missing_target_genes_zero_embedding"] == ["C"]


def test_ridge_matches_historical_formula_on_nonsingular_training_fixture(packet):
    tr = np.arange(40)
    te = np.arange(40, 45)
    labels = [packet.labels[i] for i in tr]
    tests = [packet.labels[i] for i in te]
    pred, fit = predict_baselines(labels, packet.delta[tr], packet.target_delta[tr],
                                 packet.target_genes, tests, components=4)
    y = packet.delta[tr].T
    b = y.mean(1, keepdims=True)
    u, s, vt = np.linalg.svd(y - b, full_matrices=False)
    basis = u[:, :4]
    em = {g: (packet.target_delta[tr, j] - packet.target_delta[tr, j].mean())
          @ vt[:4].T / s[:4] for j, g in enumerate(packet.target_genes)}
    design = lambda ll: np.stack([sum(em[g] for g in members(p)) for p in ll])
    p = design(labels)
    w = np.linalg.solve(basis.T @ basis + .1 * np.eye(4), basis.T @ (y-b) @ p)
    w = w @ np.linalg.inv(p.T @ p + .1 * np.eye(4))
    np.testing.assert_allclose(pred["pca_ridge"], (basis @ w @ design(tests).T + b).T,
                               atol=1e-12)
    assert fit["effective_components"] == 4


def test_holdout_outcome_poison_does_not_change_fit_or_predictions(packet):
    f = make_splits(packet.labels, "whole_gene", 17)[0]
    idx = {p: i for i, p in enumerate(packet.labels)}
    tr = [idx[p] for p in f["train"]]
    te = [idx[p] for p in f["test"] + f["excluded"]]
    def fit(p):
        return predict_baselines(f["train"], p.delta[tr], p.target_delta[tr],
                                 p.target_genes, f["test"])
    before, receipt = fit(packet)
    poisoned = copy.deepcopy(packet)
    poisoned.delta[te] = 1e20
    poisoned.target_delta[te] = -1e20
    poisoned.de_stat[te] = 1e30
    after, after_receipt = fit(poisoned)
    assert receipt == after_receipt
    for arm in before:
        np.testing.assert_array_equal(before[arm], after[arm])


def test_de_stat_is_scoring_only_and_packet_not_mutated(packet):
    original = array_hash(packet.delta), array_hash(packet.target_delta)
    a = evaluate(packet, bootstrap_draws=2)
    packet.de_stat = packet.de_stat[:, ::-1].copy()
    b = evaluate(packet, bootstrap_draws=2)
    assert [r["prediction_sha256"] for r in a["runs"]] == [r["prediction_sha256"] for r in b["runs"]]
    assert original == (array_hash(packet.delta), array_hash(packet.target_delta))


def test_zero_rank_is_finite_and_falls_back_to_mean():
    pred, fit = predict_baselines(["A", "B"], np.ones((2, 4)), np.ones((2, 2)),
                                 ["A", "B"], ["C"])
    assert fit["effective_components"] == 0
    np.testing.assert_array_equal(pred["pca_ridge"], np.ones((1, 4)))


def test_systema_removes_common_shift_and_handles_undefined():
    true = np.array([[2., 0., 1.], [0., 2., 1.]])
    reference = true.mean(0)
    score = score_predictions(true, true, reference)
    assert all(r["systema_train_centroid_pearson"] == pytest.approx(1) for r in score)
    common = np.array([100., 30., 70.])
    shifted = score_predictions(true + common, true + common, reference + common)
    assert [r["systema_train_centroid_pearson"] for r in score] == pytest.approx(
        [r["systema_train_centroid_pearson"] for r in shifted])
    mean = score_predictions(np.tile(reference, (2, 1)), true, reference)
    assert all(r["systema_train_centroid_pearson"] is None for r in mean)
    assert all(r["centroid_accuracy"] == .5 for r in mean)
    assert all(r["centroid_rank"] == .5 for r in mean)


def test_bootstrap_paired_clusters_across_seeds():
    a = [{"seed": s, "fold": i, "label": str(i), "m": float(i + 1)}
         for s in [1, 2, 3] for i in range(3)]
    b = [{**r, "m": 0.} for r in a]
    result = paired_bootstrap(a, b[::-1], "m", draws=100)
    assert result["difference"] == 2.
    assert result["n_labels"] == 3 and result["n_paired_rows"] == 9
    assert result == paired_bootstrap(a[::-1], b, "m", draws=100)
    with pytest.raises(ValueError, match="paired"):
        paired_bootstrap(a, b[:-1], "m")
    with pytest.raises(ValueError, match="paired"):
        paired_bootstrap(a + a[:1], b, "m")
    with pytest.raises(ValueError, match="mismatched"):
        paired_bootstrap([{**r, "split_sha256": "different"} for r in a], b, "m")


def test_bootstrap_exposes_undefined_exclusions():
    a = [{"seed": 1, "fold": 0, "label": "A", "m": None}]
    b = [{**a[0], "m": 1.}]
    result = paired_bootstrap(a, b, "m", draws=10)
    assert result["difference"] is None and result["ci95"] is None
    assert result["n_excluded_undefined"] == 1


def test_three_seeds_full_cpu_receipt_and_strict_json(packet):
    result = evaluate(packet, bootstrap_draws=2)
    assert len(result["runs"]) == 30
    assert len(result["seeds"]) == 3
    assert not result["claims"]["official_rubric"]
    assert not result["claims"]["matched_reference_reproduction"]
    for run in result["runs"]:
        assert run["split"]["audit"]["passed"]
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    assert result == evaluate(packet, bootstrap_draws=2)
    for seeds in ([1, 2], [1, 1, 2]):
        with pytest.raises(ValueError, match="three"):
            evaluate(packet, seeds=seeds)


@pytest.fixture
def manifest(tmp_path, packet):
    bulk, target = tmp_path / "pseudobulk.npz", tmp_path / "target.npz"
    np.savez(bulk, D=packet.delta, T=packet.de_stat, genes=packet.genes, perts=packet.labels)
    np.savez(target, Dt=packet.target_delta, targets=packet.target_genes, perts=packet.labels)
    m = {"schema_version": 1, "status": "frozen", "dataset": "synthetic fixture",
         "normalization": "fixture", "controls": {"fixture": True},
         "source": {"sha256": "fixture"}, "preparation": {"code_sha256": "fixture"},
         "terms": {"status": "synthetic"}, "limitations": ["unit tests only"],
         "outcome_transform_scope": "per_label_only",
         "gene_selection": {"fit_scope": "controls_only"},
         "target_row_labels": packet.labels,
         "pseudobulk": {"path": bulk.name, "sha256": file_hash(bulk)},
         "target_deltas": {"path": target.name, "sha256": file_hash(target)}}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(m))
    return path, m


def test_packet_hash_order_scope_and_shape_guards(manifest, tmp_path):
    path, m = manifest
    packet = load_packet(path)
    assert packet.provenance["target_rows"] == "embedded_labels_verified"
    for field, value, message in [("status", "draft", "frozen"),
                                  ("outcome_transform_scope", "all_labels", "global"),
                                  ("gene_selection", {"fit_scope": "all_labels"}, "gene axis"),
                                  ("target_row_labels", m["target_row_labels"][::-1], "order")]:
        bad = {**m, field: value}
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError, match=message):
            load_packet(path)
    path.write_text(json.dumps(m))
    with (tmp_path / "pseudobulk.npz").open("ab") as f:
        f.write(b"changed")
    with pytest.raises(ValueError, match="hash"):
        load_packet(path)


def test_cli_hash_ledger_and_refuses_overwrite(manifest, tmp_path):
    path, _ = manifest
    output = tmp_path / "out"
    args = ["--manifest", str(path), "--output", str(output), "--bootstrap-draws", "2"]
    main(args)
    result_path = output / "results.json"
    result = json.loads(result_path.read_text())
    content_hash = result.pop("result_content_sha256")
    assert digest(result) == content_hash
    assert (output / "SHA256SUMS").read_text().split()[0] == file_hash(result_path)
    assert result["code_provenance"]["device"] == "cpu"
    with pytest.raises(FileExistsError):
        main(args)


def test_nonfinite_packet_rejected(packet):
    packet.delta[0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        packet.validate()


def test_target_row_embedded_order_and_raw_source_hash_guards(manifest, tmp_path):
    path, m = manifest
    source = tmp_path / "raw.bin"
    source.write_bytes(b"unit fixture")
    m["source"] = {"path": source.name, "sha256": file_hash(source)}
    path.write_text(json.dumps(m))
    assert load_packet(path).provenance["source_verification"] == "sha256_verified"
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="raw source hash"):
        load_packet(path)
    del m["source"]["path"]
    target = tmp_path / "target.npz"
    with np.load(target, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    arrays["perts"] = arrays["perts"][::-1]
    np.savez(target, **arrays)
    m["target_deltas"]["sha256"] = file_hash(target)
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="misaligned"):
        load_packet(path)
