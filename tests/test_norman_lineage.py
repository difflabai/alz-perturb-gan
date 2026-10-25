"""CPU-only aggregate-lineage fixtures, including partial/failed evidence."""
import copy
import json

import pytest

from alz_perturb_gan.norman_eval import digest, file_hash, make_splits
from alz_perturb_gan.norman_lineage import audit, main, read_adapter, read_receipt


@pytest.fixture
def case():
    labels = [f"G{i}" for i in range(10)] + [f"G{i}+G{i+1}" for i in range(0, 10, 2)]
    splits = [s for protocol in ("perturbation_label", "whole_gene")
              for s in make_splits(labels, protocol, 17)]
    rows = [{"protocol": s["protocol"], "seed": s["seed"], "fold": s["fold"],
             "arm": "pca_ridge", "label": p, "split_sha256": s["sha256"],
             "data_sha256": digest("fixture data"), "reference_sha256": digest(s["sha256"])}
            for s in splits for p in s["test"]]
    receipt = {"packet_provenance": {"manifest": {"target_row_labels": labels,
                "source": {"sha256": digest("fixture source")}},
                "artifact_sha256": {"pseudobulk": digest("bulk"), "target_deltas": digest("target")}},
               "data_content_sha256": digest("fixture data"),
               "runs": [{"split": s, "reference_sha256": digest(s["sha256"]),
                         "fit": {"missing_target_genes_zero_embedding": []}} for s in splits],
               "rows": rows}
    observations, groups, memberships = [], {}, []
    for p in labels:
        for role in ("treatment", "control"):
            name = f"{role}:{p}"
            refs = []
            for i in range(2):
                cell_id = f"{name}:{i}"
                observations.append({"cell_id": cell_id, "library_id": "capture1",
                    "label": p if role == "treatment" else "control",
                    "guide_id": name, "gemgroup": p})
                refs.append(["capture1", cell_id])
            groups[name] = refs
        memberships.append({"label": p, "treatment_groups": [f"treatment:{p}"],
                            "control_groups": [f"control:{p}"]})
    adapter = {"observations": observations, "memberships": {"granularity": "aggregate",
                "cell_groups": groups, "rows": memberships},
               "identity_scope": {"namespace": "fixture physical capture", "status": "canonical_physical_namespace",
                                  "guide_namespace": "fixture guide catalog"},
               "provenance_kind": "producer_attested", "manifest_status": "frozen",
               "control_use": ["response_construction"], "evidence": {"missing_bindings": []}}
    return receipt, adapter


def codes(result, category="findings"):
    return {x["code"] for x in result[category]}


def test_many_to_one_lineage_is_not_independently_attested(case):
    receipt, adapter = case
    result = audit(receipt, adapter, policy="group_disjoint")
    assert result["policy_result"] == "unknown"
    assert not result["findings"]
    assert codes(result, "missing_evidence") == {"producer_attestations_not_independently_verified"}
    assert result["evidence_status"] == "partial"
    assert not result["claims"]["producer_attestations_independently_verified"]
    assert len(result["evaluation_row_links"]) == len(receipt["rows"])
    assert len(result["folds"]) == 10
    assert {f["seed"] for f in result["folds"]} == {17}
    assert {f["split_sha256"] for f in result["folds"]} == {r["split"]["sha256"] for r in receipt["runs"]}
    for trace in result["aggregate_trace"].values():
        assert trace["granularity"] == "aggregate"
        assert trace["treatment"]["n_unique_contributors"] == 2
        assert trace["control"]["n_unique_contributors"] == 2
    assert not result["claims"]["biological_independence_certified"]


@pytest.mark.parametrize("control_use", [["response_construction"], ["fixed_reference_only"]])
def test_shared_control_contributors_separate_from_target_leakage(case, control_use):
    receipt, adapter = case
    pool = adapter["memberships"]["cell_groups"]["control:G0"]
    adapter["observations"] = [o for o in adapter["observations"]
        if not o["cell_id"].startswith("control:") or o["cell_id"].startswith("control:G0:")]
    adapter["memberships"]["cell_groups"] = {k:v for k,v in adapter["memberships"]["cell_groups"].items()
        if k.startswith("treatment:")}
    adapter["memberships"]["cell_groups"]["pooled"] = pool
    for row in adapter["memberships"]["rows"]:
        row["control_groups"] = ["pooled"]
    adapter["control_use"] = control_use
    result = audit(receipt, adapter)
    assert result["policy_result"] == "not_enforced" and not result["findings"]
    assert result["control_use"] == control_use
    assert not result["claims"]["shared_reference_invalidates_legacy_score"]
    for fold in result["folds"]:
        assert fold["source_cell_overlap"]["control_to_control"]["n_unique_shared"] == 2
        assert fold["source_cell_overlap"]["treatment_to_treatment"]["n_unique_shared"] == 0
        assert fold["n_eval_affected_by_shared_source"] == fold["n_eval_aggregates"]
        assert fold["within_split_contributor_reuse"]["train"]["control"]["n_unique_reused_contributors"] == 2
    assert audit(receipt, adapter, policy="source_disjoint")["policy_result"] == "fail"
    assert result["disjointness_feasibility"]["n_folds_with_shared_controls"] == 10


def test_group_sharing_is_a_separate_policy_not_target_leakage(case):
    receipt, adapter = case
    for obs in adapter["observations"]:
        obs["gemgroup"] = "shared batch"
    result = audit(receipt, adapter, policy="source_disjoint")
    assert result["policy_result"] == "unknown"
    assert all(f["source_cell_overlap"]["any_source_cell"]["n_unique_shared"] == 0 for f in result["folds"])
    assert all(f["batch_sharing"]["all_contributors"]["n_unique_shared"] == 1 for f in result["folds"])
    assert audit(receipt, adapter, policy="group_disjoint")["policy_result"] == "fail"


def test_library_scopes_cells_and_groups_not_export_hash(case):
    receipt, adapter = case
    for obs in adapter["observations"]:
        old = obs["cell_id"]
        p = old.split(":")[1]
        obs["library_id"] = p
        obs["cell_id"] = old.split(":")[0] + old.split(":")[2]
        obs["gemgroup"] = "1"
    for refs in adapter["memberships"]["cell_groups"].values():
        for ref in refs:
            old = ref[1].split(":")
            ref[:] = [old[1], old[0] + old[2]]
    result = audit(receipt, adapter, policy="group_disjoint")
    assert result["policy_result"] == "unknown" and not result["findings"]


def test_missing_adapter_is_unknown_not_clean(case):
    result = audit(case[0], policy="source_disjoint")
    assert result["policy_result"] == "unknown" and result["evidence_status"] == "partial"
    assert "lineage_adapter_missing" in codes(result, "missing_evidence")
    assert all(f["source_cell_overlap"]["any_source_cell"]["status"] == "unknown_or_partial" for f in result["folds"])


@pytest.mark.parametrize("change,expected", [
    ("partial", "lineage_manifest_not_frozen"),
    ("reconstruction", "aggregation_lineage_not_producer_attested"),
    ("namespace", "physical_cell_namespace_unverified"),
    ("binding", "incomplete_data_bindings"),
    ("control_use", "control_use_unspecified"),
    ("granularity", "aggregate_granularity_unspecified"),
])
def test_missing_canonical_evidence_never_certifies(case, change, expected):
    receipt, adapter = case
    if change == "partial": adapter["manifest_status"] = "partial"
    if change == "reconstruction": adapter["provenance_kind"] = "metadata_reconstruction"
    if change == "namespace": adapter["identity_scope"]["status"] = "export_local_only"
    if change == "binding": adapter["evidence"]["missing_bindings"] = ["source_sha256"]
    if change == "control_use": adapter["control_use"] = []
    if change == "granularity": adapter["memberships"].pop("granularity")
    result = audit(receipt, adapter, policy="source_disjoint")
    assert result["policy_result"] == "unknown"
    assert expected in codes(result, "missing_evidence")


@pytest.mark.parametrize("field", ["label", "guide_id", "gemgroup"])
def test_missing_observation_fields(case, field):
    receipt, adapter = case
    adapter["observations"][0].pop(field)
    result = audit(receipt, adapter, policy="group_disjoint")
    assert result["policy_result"] == "unknown"
    assert "observation_field_missing" in codes(result, "missing_evidence")


@pytest.mark.parametrize("change,expected", [
    ("duplicate_source", "duplicate_source_identity"),
    ("duplicate_group", "duplicate_within_group"),
    ("duplicate_role", "duplicate_within_aggregate_role"),
    ("orphan_contributor", "orphan_contributor"),
    ("orphan_source", "orphan_source_observations"),
    ("orphan_group", "orphan_group_reference"),
    ("orphan_aggregate", "orphan_aggregate_row"),
    ("duplicate_aggregate", "duplicate_aggregate_row"),
    ("cross_role", "cross_role_reuse"),
    ("wrong_label", "contributor_role_or_label_mismatch"),
    ("duplicate_eval", "duplicate_evaluation_row"),
    ("orphan_eval", "orphan_evaluation_row"),
    ("eval_binding", "evaluation_binding_mismatch"),
    ("invalid_exclusion", "invalid_declared_exclusion"),
])
def test_duplicates_orphans_cross_roles_and_row_binding(case, change, expected):
    receipt, adapter = case
    group = adapter["memberships"]["cell_groups"]["treatment:G0"]
    row = adapter["memberships"]["rows"][0]
    if change == "duplicate_source": adapter["observations"].append(copy.deepcopy(adapter["observations"][0]))
    if change == "duplicate_group": group.append(group[0])
    if change == "duplicate_role": row["treatment_groups"] *= 2
    if change == "orphan_contributor": group.append(["capture1", "absent"])
    if change == "orphan_source": adapter["observations"].append({**adapter["observations"][0], "cell_id": "unreferenced"})
    if change == "orphan_group": row["treatment_groups"].append("absent")
    if change == "orphan_aggregate": adapter["memberships"]["rows"].append({"label": "absent"})
    if change == "duplicate_aggregate": adapter["memberships"]["rows"].append(copy.deepcopy(row))
    if change == "cross_role": row["control_groups"].append("treatment:G0")
    if change == "wrong_label": adapter["observations"][0]["label"] = "G9"
    if change == "duplicate_eval": receipt["rows"].append(copy.deepcopy(receipt["rows"][0]))
    if change == "orphan_eval": receipt["rows"][0]["label"] = "absent"
    if change == "eval_binding": receipt["rows"][0]["reference_sha256"] = digest("wrong")
    if change == "invalid_exclusion": adapter["memberships"]["excluded_cells"] = [{"library_id": "capture1", "cell_id": "absent", "reason": "QC"}]
    result = audit(receipt, adapter, policy="source_disjoint")
    assert result["policy_result"] == "fail"
    assert expected in codes(result)


def test_treatment_reuse_within_and_across_exact_splits(case):
    receipt, adapter = case
    adapter["memberships"]["cell_groups"]["treatment:G1"].append(["capture1", "treatment:G0:0"])
    result = audit(receipt, adapter, policy="source_disjoint")
    assert any(f["source_cell_overlap"]["treatment_to_treatment"]["n_unique_shared"] for f in result["folds"])
    assert any(f["within_split_contributor_reuse"]["train"]["treatment"]["n_unique_reused_contributors"] for f in result["folds"])
    assert result["policy_result"] == "fail"


def test_missing_aggregate_and_role_coverage(case):
    receipt, adapter = case
    adapter["memberships"]["rows"].pop()
    result = audit(receipt, adapter)
    assert "aggregate_row_missing" in codes(result, "missing_evidence")
    assert "contributors_missing" in codes(result, "missing_evidence")
    assert result["evidence_status"] == "invalid"  # unreferenced source cells are also present
    assert result["evidence_completeness"] == "partial"


@pytest.mark.parametrize("part", ["observations", "memberships"])
def test_missing_artifact_never_passes(case, part):
    receipt, adapter = case
    adapter[part] = None
    result = audit(receipt, adapter, policy="source_disjoint")
    assert result["policy_result"] != "pass" and result["evidence_completeness"] == "partial"
    expected = "source_observations_missing" if part == "observations" else "aggregate_memberships_missing"
    assert expected in codes(result, "missing_evidence")


def test_incomplete_evaluation_row_coverage(case):
    receipt, adapter = case
    receipt["rows"].pop()
    result = audit(receipt, adapter, policy="source_disjoint")
    assert result["policy_result"] == "unknown"
    assert "evaluation_coverage" in codes(result, "missing_evidence")


def test_duplicate_split_and_invalid_identity_are_reported(case):
    receipt, adapter = case
    receipt["runs"].append(copy.deepcopy(receipt["runs"][0]))
    adapter["observations"][0].pop("library_id")
    adapter["memberships"]["cell_groups"]["treatment:G0"].append(["unscoped"])
    result = audit(receipt, adapter, policy="source_disjoint")
    assert {"duplicate_split", "invalid_source_identity", "invalid_contributor_reference"} <= codes(result)
    assert result["policy_result"] == "fail"


def test_declared_exclusion_is_not_an_orphan(case):
    receipt, adapter = case
    adapter["observations"].append({**adapter["observations"][0], "cell_id": "excluded"})
    adapter["memberships"]["excluded_cells"] = [{"library_id": "capture1", "cell_id": "excluded", "reason": "predeclared QC"}]
    result = audit(receipt, adapter, policy="group_disjoint")
    assert result["policy_result"] == "unknown"
    assert result["coverage"]["n_valid_declared_exclusions"] == 1
    assert result["coverage"]["n_orphan_source_cells"] == 0


@pytest.fixture
def files(case, tmp_path):
    receipt, adapter = case
    receipt["result_content_sha256"] = digest(receipt)
    rp = tmp_path / "evaluation.json"
    rp.write_text(json.dumps(receipt))
    op, mp = tmp_path / "observations.jsonl", tmp_path / "memberships.json"
    op.write_text("\n".join(json.dumps(o) for o in adapter["observations"]) + "\n")
    mp.write_text(json.dumps(adapter["memberships"]))
    manifest = {"schema_version": 1, "status": "frozen", "provenance_kind": "producer_attested",
                "identity_scope": adapter["identity_scope"], "control_use": adapter["control_use"],
                "bindings": {"source_sha256": receipt["packet_provenance"]["manifest"]["source"]["sha256"],
                    "data_content_sha256": receipt["data_content_sha256"],
                    **{f"{k}_sha256":v for k,v in receipt["packet_provenance"]["artifact_sha256"].items()}},
                "observations": {"path": op.name, "sha256": file_hash(op)},
                "memberships": {"path": mp.name, "sha256": file_hash(mp)}}
    path = tmp_path / "lineage.json"
    path.write_text(json.dumps(manifest))
    return rp, path, manifest


def test_hashed_adapter_and_receipt_guards(files, tmp_path):
    rp, path, manifest = files
    receipt, ref = read_receipt(rp)
    assert ref["sha256"] == file_hash(rp)
    result = audit(receipt, read_adapter(path, receipt), policy="group_disjoint")
    assert result["policy_result"] == "unknown"
    assert result["evidence"]["missing_bindings"] == []
    assert result["evidence_status"] == "partial"  # byte integrity is not producer verification
    bad = copy.deepcopy(manifest)
    bad["bindings"]["source_sha256"] = digest("different dataset")
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="differs from evaluated"):
        read_adapter(path, receipt)
    path.write_text(json.dumps(manifest))
    (tmp_path / "observations.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="byte hash"):
        read_adapter(path, receipt)
    rp.write_text(json.dumps({**receipt, "changed": True}))
    with pytest.raises(ValueError, match="content hash"):
        read_receipt(rp)


def test_cli_opt_in_policies_missingness_ledger_and_no_overwrite(files, tmp_path):
    rp, path, _ = files
    args = ["--evaluation-receipt", str(rp), "--lineage-manifest", str(path),
            "--policy", "report", "--output", str(tmp_path / "reported")]
    before = rp.read_bytes()
    assert main(args) == 0
    result_path = tmp_path / "reported/lineage-audit.json"
    result = json.loads(result_path.read_text())
    claim = result.pop("result_content_sha256")
    assert digest(result) == claim
    assert (result_path.parent / "SHA256SUMS").read_text().split()[0] == file_hash(result_path)
    assert rp.read_bytes() == before
    with pytest.raises(FileExistsError): main(args)
    assert main(["--evaluation-receipt", str(rp), "--policy", "source_disjoint",
                 "--output", str(tmp_path / "unknown")]) == 2
    assert main(["--evaluation-receipt", str(rp), "--output", str(tmp_path / "report")]) == 0


def test_split_tampering_rejected(case):
    receipt, adapter = case
    receipt["runs"][0]["split"]["seed"] += 1
    with pytest.raises(ValueError, match="split hash"):
        audit(receipt, adapter)


@pytest.mark.parametrize("field", ["reference", "data"])
def test_missing_evaluation_provenance_refused(case, field):
    receipt, adapter = case
    if field == "reference": receipt["runs"][0].pop("reference_sha256")
    else: receipt["data_content_sha256"] = "abbreviated"
    with pytest.raises(ValueError, match="complete"):
        audit(receipt, adapter)


def test_invalid_evidence_not_masked_by_incompleteness(case):
    receipt, adapter = case
    adapter["manifest_status"] = "partial"
    adapter["observations"].append(copy.deepcopy(adapter["observations"][0]))
    result = audit(receipt, adapter, policy="report")
    assert result["evidence_status"] == result["evidence_validity"] == "invalid"
    assert result["evidence_completeness"] == "partial"
    assert result["policy_result"] == "not_enforced"
    assert not result["claims"]["report_exit_zero_means_evidence_accepted"]
    assert audit(receipt, adapter, policy="source_disjoint")["policy_result"] == "fail"


def test_report_exit_zero_emits_invalid_not_acceptance(files, tmp_path):
    rp, path, manifest = files
    observations = tmp_path / "observations.jsonl"
    first = observations.read_text().splitlines()[0]
    with observations.open("a") as stream: stream.write(first + "\n")
    manifest["observations"]["sha256"] = file_hash(observations)
    manifest["status"] = "partial"
    path.write_text(json.dumps(manifest))
    args = ["--evaluation-receipt", str(rp), "--lineage-manifest", str(path)]
    assert main(args + ["--output", str(tmp_path / "invalid-report")]) == 0
    result = json.loads((tmp_path / "invalid-report/lineage-audit.json").read_text())
    assert result["evidence_status"] == "invalid" and result["evidence_completeness"] == "partial"
    assert result["policy_result"] == "not_enforced"
    assert main(args + ["--policy", "source_disjoint", "--output", str(tmp_path / "invalid-strict")]) == 2


def test_same_guide_across_libraries_and_captures_is_counted(case):
    receipt, adapter = case
    for obs in adapter["observations"]:
        label = obs["cell_id"].split(":")[1]
        obs["namespace"] = "capture namespace:" + label
        obs["library_id"] = label
        obs["guide_id"] = "canonical-shared-guide"
    for refs in adapter["memberships"]["cell_groups"].values():
        for ref in refs:
            label = ref[1].split(":")[1]
            ref[:] = ["capture namespace:" + label, label, ref[1]]
    result = audit(receipt, adapter)
    assert not result["findings"]
    assert all(f["guide_sharing_treatment_only"]["n_unique_shared"] == 1 for f in result["folds"])
    assert all(f["batch_sharing"]["treatment_only"]["n_unique_shared"] == 0 for f in result["folds"])
    # Same textual guide ID from different declared catalogs is not assumed equal.
    for obs in adapter["observations"]:
        obs["guide_namespace"] = "catalog:" + obs["library_id"]
    result = audit(receipt, adapter)
    assert all(f["guide_sharing_treatment_only"]["n_unique_shared"] == 0 for f in result["folds"])


def test_missing_guide_catalog_is_unverified(case):
    receipt, adapter = case
    adapter["identity_scope"].pop("guide_namespace")
    result = audit(receipt, adapter)
    assert "guide_namespace_unverified" in codes(result, "missing_evidence")
    assert result["evidence_status"] == "partial"


def test_capture_namespace_disambiguates_cell_keys(case):
    receipt, adapter = case
    first, second = adapter["observations"][:2]
    original_second_id = second["cell_id"]
    first["namespace"], second["namespace"] = "capture A", "capture B"
    second["cell_id"] = first["cell_id"]
    for refs in adapter["memberships"]["cell_groups"].values():
        for ref in refs:
            if ref[1] == first["cell_id"]: ref[:] = ["capture A", ref[0], ref[1]]
            elif ref[1] == original_second_id: ref[:] = ["capture B", ref[0], first["cell_id"]]
    result = audit(receipt, adapter)
    assert not result["findings"]
    assert result["coverage"]["n_duplicate_source_identities"] == 0
    assert result["aggregate_trace"]["G0"]["treatment"]["n_unique_contributors"] == 2


@pytest.mark.parametrize("field,value", [("cell_id", " id"), ("library_id", "library\n"),
                                         ("namespace", ""), ("namespace", "bad\x00namespace")])
def test_identity_strings_are_validated(case, field, value):
    receipt, adapter = case
    adapter["observations"][0][field] = value
    result = audit(receipt, adapter)
    assert "invalid_source_identity" in codes(result)
    assert result["evidence_status"] == "invalid"


def test_invalid_exclusion_does_not_hide_orphan(case):
    receipt, adapter = case
    obs = {**adapter["observations"][0], "cell_id": "unreferenced"}
    adapter["observations"].append(obs)
    adapter["memberships"]["excluded_cells"] = [{"library_id": obs["library_id"],
        "cell_id": obs["cell_id"], "reason": "   "}]
    result = audit(receipt, adapter)
    assert "invalid_declared_exclusion" in codes(result)
    assert "orphan_source_observations" in codes(result)
    assert result["coverage"]["n_orphan_source_cells"] == 1
    assert result["coverage"]["n_valid_declared_exclusions"] == 0


def test_explicit_unmapped_targets_and_missing_mapping_receipt(case):
    receipt, adapter = case
    for run in receipt["runs"]:
        run["fit"]["missing_target_genes_zero_embedding"] = ["G0"]
    result = audit(receipt, adapter)
    assert result["gene_mapping"]["recorded_unmapped_symbols"] == ["G0"]
    assert "unmapped_target_symbols" in codes(result, "missing_evidence")
    assert result["aggregate_trace"]["G0"]["unmapped_target_symbols"] == ["G0"]
    assert any(f["n_eval_aggregates_with_unmapped_targets"] for f in result["folds"])
    receipt["runs"][0].pop("fit")
    result = audit(receipt, adapter)
    assert result["gene_mapping"]["n_runs_without_mapping_evidence"] == 1
    assert "target_mapping_evidence_missing" in codes(result, "missing_evidence")


def test_multi_group_pool_cache_preserves_repeat_edge_detection(case):
    receipt, adapter = case
    original = adapter["memberships"]["cell_groups"]["control:G0"]
    adapter["memberships"]["cell_groups"]["control:G0:a"] = original[:1]
    adapter["memberships"]["cell_groups"]["control:G0:b"] = original[1:]
    row = adapter["memberships"]["rows"][0]
    row["control_groups"] = ["control:G0:a", "control:G0:b"]
    result = audit(receipt, adapter)
    assert result["aggregate_trace"]["G0"]["control"]["n_unique_contributors"] == 2
    assert not result["findings"]
    row["control_groups"] += ["control:G0:a"]
    result = audit(receipt, adapter)
    assert "duplicate_within_aggregate_role" in codes(result)
