"""Opt-in CPU lineage audit of exact saved Norman evaluation rows.

Aggregate contributors are many-to-one treatment/control edges, not paired
single-cell targets. This module never fits a model or changes an evaluation.
See docs/norman-lineage.md for the narrow adapter and explicit audit policies.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from .norman_eval import code_provenance, digest, file_hash, leakage_audit, require_full_hash
from .paths import check_new_target, create_new, scoped, scoped_new_dir


def read_receipt(path):
    path = scoped(str(Path(path).absolute()))
    receipt = json.loads(path.read_text())
    claimed = receipt.get("result_content_sha256")
    require_full_hash(claimed, "evaluation.result_content_sha256")
    if digest({k: v for k, v in receipt.items() if k != "result_content_sha256"}) != claimed:
        raise ValueError("evaluation receipt content hash mismatch")
    return receipt, {"path": str(path), "sha256": file_hash(path)}


def read_adapter(path, receipt):
    """Partial adapters are valid evidence of missingness, never certification."""
    if path is None:
        return None
    path = scoped(str(Path(path).absolute()))
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("lineage adapter schema_version must be 1")
    packet = receipt["packet_provenance"]
    expected = {"source_sha256": packet["manifest"]["source"]["sha256"],
                "data_content_sha256": receipt["data_content_sha256"],
                **{f"{k}_sha256": v for k, v in packet["artifact_sha256"].items()}}
    bindings = manifest.get("bindings", {})
    for key, value in bindings.items():
        if key in expected:
            require_full_hash(value, f"lineage.bindings.{key}")
            if value != expected[key]:
                raise ValueError(f"lineage {key} differs from evaluated data")
    evidence = {"manifest_sha256": file_hash(path), "bindings": bindings,
                "missing_bindings": sorted(set(expected) - set(bindings)),
                "manifest": manifest, "files": {}}
    result = {"evidence": evidence, "observations": None, "memberships": None,
              "identity_scope": manifest.get("identity_scope", {}),
              "provenance_kind": manifest.get("provenance_kind", "unknown"),
              "manifest_status": manifest.get("status", "unknown"),
              "control_use": manifest.get("control_use", [])}
    for key in ("observations", "memberships"):
        spec = manifest.get(key)
        if spec is None:
            continue
        require_full_hash(spec.get("sha256"), f"lineage.{key}.sha256")
        artifact = scoped(str((path.parent / spec["path"]).absolute()))
        if file_hash(artifact) != spec["sha256"]:
            raise ValueError(f"lineage {key} byte hash mismatch")
        evidence["files"][key] = {"path": str(artifact), "sha256": spec["sha256"]}
        if key == "observations":
            with artifact.open() as stream:
                result[key] = [json.loads(line) for line in stream if line.strip()]
        else:
            result[key] = json.loads(artifact.read_text())
    return result


def _overlap(left, right):
    shared = sorted(left & right)
    return {"n_unique_shared": len(shared), "shared_set_sha256": digest(shared),
            "examples": shared[:5]}


def _identifier(value):
    return (isinstance(value, str) and bool(value) and value == value.strip()
            and not any(ord(char) < 32 or ord(char) == 127 for char in value))


def _cell_reference(ref, default_namespace):
    if not isinstance(ref, (list, tuple)) or len(ref) not in (2, 3):
        return None
    key = (default_namespace, *ref) if len(ref) == 2 else tuple(ref)
    return tuple(key) if all(_identifier(value) for value in key) else None


def audit(receipt, adapter=None, *, policy="report"):
    """Audit exact stored splits and evaluation keys; do not regenerate folds.

    Policies: report (no enforcement), source_disjoint (all contributors),
    group_disjoint (contributors plus scoped gemgroups). Group sharing is a
    separate generalization limitation, not automatically target leakage.
    """
    if policy not in {"report", "source_disjoint", "group_disjoint"}:
        raise ValueError("unknown lineage policy")
    labels = receipt["packet_provenance"]["manifest"]["target_row_labels"]
    require_full_hash(receipt.get("data_content_sha256"), "evaluation.data_content_sha256")
    findings, missing = [], []
    def issue(code, **details):
        findings.append({"code": code, **details})
    splits, unmapped, mapping_runs_missing = {}, set(), 0
    for run in receipt["runs"]:
        require_full_hash(run.get("reference_sha256"), "evaluation.run.reference_sha256")
        split = run["split"]
        key = (split["protocol"], split["seed"], split["fold"])
        if key in splits:
            issue("duplicate_split", split=list(key))
            continue
        claimed = split["sha256"]
        if digest({k: v for k, v in split.items() if k != "sha256"}) != claimed:
            raise ValueError("stored split hash mismatch")
        leakage_audit(labels, split)
        splits[key] = split
        recorded = run.get("fit", {}).get("missing_target_genes_zero_embedding")
        if not isinstance(recorded, list) or not all(_identifier(g) for g in recorded):
            mapping_runs_missing += 1
        else:
            unmapped.update(recorded)
    if mapping_runs_missing:
        missing.append({"code": "target_mapping_evidence_missing", "n_runs": mapping_runs_missing})
    if unmapped:
        missing.append({"code": "unmapped_target_symbols", "symbols": sorted(unmapped),
                        "source": "exact saved fit receipts; no alias substitution"})
    seen_eval, evaluation_links = set(), []
    arms = {row["arm"] for row in receipt["rows"]}
    evaluated = Counter()
    for row in receipt["rows"]:
        key = (row["protocol"], row["seed"], row["fold"])
        identity = (*key, row["arm"], row["label"])
        if identity in seen_eval:
            issue("duplicate_evaluation_row", row=list(identity))
        seen_eval.add(identity)
        split = splits.get(key)
        if split is None or row["label"] not in split["test"]:
            issue("orphan_evaluation_row", row=list(identity))
        elif (row.get("split_sha256") != split["sha256"]
              or row.get("data_sha256") != receipt["data_content_sha256"]
              or row.get("reference_sha256") != next(
                  run["reference_sha256"] for run in receipt["runs"] if run["split"] is split)):
            issue("evaluation_binding_mismatch", row=list(identity))
        else:
            evaluated[(*key, row["arm"])] += 1
        evaluation_links.append({"row": list(identity), "aggregate_label": row["label"]})
    for key, split in splits.items():
        for arm in arms:
            if evaluated[(*key, arm)] != len(split["test"]):
                missing.append({"code": "evaluation_coverage", "split": list(key), "arm": arm,
                                "n_expected": len(split["test"]), "n_found": evaluated[(*key, arm)]})
    if adapter is None:
        missing.append({"code": "lineage_adapter_missing"})
        adapter = {}
    observations = adapter.get("observations")
    membership = adapter.get("memberships")
    scope = adapter.get("identity_scope", {})
    namespace = scope.get("namespace")
    if not _identifier(namespace) or scope.get("status") != "canonical_physical_namespace":
        missing.append({"code": "physical_cell_namespace_unverified"})
    if adapter.get("provenance_kind") != "producer_attested":
        missing.append({"code": "aggregation_lineage_not_producer_attested"})
    if adapter.get("manifest_status") != "frozen":
        missing.append({"code": "lineage_manifest_not_frozen"})
    # Declaration fields and artifact hashes do not independently establish
    # production history or physical identity equivalence. No trust anchor is
    # supplied by this interface, so hand-flipping these strings cannot certify.
    missing.append({"code": "producer_attestations_not_independently_verified"})
    if adapter.get("evidence", {}).get("missing_bindings"):
        missing.append({"code": "incomplete_data_bindings",
                        "fields": adapter["evidence"]["missing_bindings"]})
    if not adapter.get("control_use"):
        missing.append({"code": "control_use_unspecified"})
    if observations is None:
        missing.append({"code": "source_observations_missing"})
    if membership is None:
        missing.append({"code": "aggregate_memberships_missing"})
    observations, membership = observations or [], membership or {}
    # IDs are scoped by physical capture namespace and library, not by export
    # checksum: two exports may contain the same physical cells.
    cells, ambiguous, counts = {}, set(), Counter()
    if namespace is not None and not _identifier(namespace):
        issue("invalid_identity_namespace")
    for obs in observations:
        cell_id, library = obs.get("cell_id"), obs.get("library_id")
        cell_namespace = obs.get("namespace", namespace)
        if not all(_identifier(v) for v in (cell_namespace, cell_id, library)):
            issue("invalid_source_identity")
            continue
        key = (cell_namespace, library, cell_id)
        counts[key] += 1
        if key in cells:
            ambiguous.add(key)
        cells[key] = obs
    for key in sorted(ambiguous):
        issue("duplicate_source_identity", namespace=key[0], library_id=key[1], cell_id=key[2], n_rows=counts[key])
    for key in ambiguous:
        del cells[key]  # never silently select one conflicting observation
    invalid_fields = {key: tuple(field for field in ("label", "guide_id", "gemgroup")
                                 if not _identifier(obs.get(field))) for key, obs in cells.items()}
    groups = membership.get("cell_groups", {})
    resolved_groups = {}
    for name, refs in groups.items():
        keys = []
        for ref in refs:
            key = _cell_reference(ref, namespace)
            if key is None:
                issue("invalid_contributor_reference", group=name)
                continue
            keys.append(key)
            if key not in cells:
                issue("orphan_contributor", group=name, namespace=key[0], library_id=key[1], cell_id=key[2])
        n_duplicate = len(keys) - len(set(keys))
        if n_duplicate:
            issue("duplicate_within_group", group=name, n_repeated_edges=n_duplicate)
        resolved_groups[name] = set(keys) & cells.keys()
    if membership.get("granularity") != "aggregate":
        missing.append({"code": "aggregate_granularity_unspecified"})
    row_memberships = {}
    for row in membership.get("rows", []):
        label = row.get("label")
        if label not in labels:
            issue("orphan_aggregate_row", label=label)
            continue
        if label in row_memberships:
            issue("duplicate_aggregate_row", label=label)
            continue
        row_memberships[label] = row
    treatment, control, trace, referenced = {}, {}, {}, set()
    pool_cache = {}
    fields_missing = Counter()
    for label in labels:
        row = row_memberships.get(label)
        if row is None:
            missing.append({"code": "aggregate_row_missing", "label": label})
            row = {}
        roles, role_hashes = {}, {}
        for role in ("treatment", "control"):
            group_names = row.get(f"{role}_groups", [])
            valid_names = []
            for name in group_names:
                if not _identifier(name):
                    issue("invalid_group_reference", label=label, role=role)
                    continue
                valid_names.append(name)
                if name not in resolved_groups:
                    issue("orphan_group_reference", label=label, role=role, group=name)
            pool_key = tuple(sorted(valid_names))
            if pool_key not in pool_cache:
                pools = [resolved_groups.get(name, set()) for name in pool_key]
                selected = pools[0] if len(pools) == 1 else set().union(*pools)
                pool_cache[pool_key] = (selected, sum(map(len, pools)) - len(selected), digest(sorted(selected)))
            selected, repeated, contributor_hash = pool_cache[pool_key]
            if repeated:
                issue("duplicate_within_aggregate_role", label=label, role=role,
                      n_repeated_edges=repeated)
            if not selected:
                missing.append({"code": "contributors_missing", "label": label, "role": role})
            expected_label = label if role == "treatment" else "control"
            wrong = [key for key in selected if "label" not in invalid_fields[key]
                     and cells[key]["label"] != expected_label]
            if wrong:
                issue("contributor_role_or_label_mismatch", label=label, role=role, n_cells=len(wrong))
            for key in selected:
                for field in invalid_fields[key]:
                    fields_missing[field] += 1
            roles[role] = selected
            role_hashes[role] = contributor_hash
            referenced.update(selected)
            # Many-to-one edges are resolved through the hash-bound membership
            # file's cell_groups and the observation file; shared pools stay one
            # pool, never one invented control cell per aggregate.
        treatment[label], control[label] = roles["treatment"], roles["control"]
        cross = _overlap(roles["treatment"], roles["control"])
        if cross["n_unique_shared"]:
            issue("cross_role_reuse", label=label, **cross)
        trace[label] = {"granularity": "aggregate", "membership_row_label": label,
                        "unmapped_target_symbols": sorted(set(label.split("+")) & unmapped),
                        **{role: {"groups": row.get(f"{role}_groups", []),
                                  "n_unique_contributors": len(roles[role]),
                                  "contributors_sha256": role_hashes[role]}
                           for role in ("treatment", "control")}}
    for field, count in fields_missing.items():
        missing.append({"code": "observation_field_missing", "field": field,
                        "n_role_edges_affected": count})
    excluded = set()
    for exclusion in membership.get("excluded_cells", []):
        key = _cell_reference([exclusion.get("namespace", namespace),
                               exclusion.get("library_id"), exclusion.get("cell_id")], namespace)
        reason = exclusion.get("reason")
        if (key not in cells or not isinstance(reason, str) or not reason.strip()
                or key in referenced or key in excluded):
            issue("invalid_declared_exclusion", namespace=exclusion.get("namespace", namespace),
                  library_id=exclusion.get("library_id"), cell_id=exclusion.get("cell_id"))
            continue
        excluded.add(key)
    unreferenced = set(cells) - referenced - excluded
    if unreferenced:
        issue("orphan_source_observations", n_unique_cells=len(unreferenced),
              examples=sorted(unreferenced)[:5])
    def union(table, names):
        return set().union(*(table[p] for p in names))
    guide_namespaces, guide_fallback = {}, set()
    for key, obs in cells.items():
        declared = obs.get("guide_namespace", scope.get("guide_namespace"))
        if not _identifier(declared):
            guide_fallback.add(key)
            if declared is not None:
                issue("invalid_guide_namespace", namespace=key[0], library_id=key[1], cell_id=key[2])
        guide_namespaces[key] = declared if _identifier(declared) else key[0]
    if guide_fallback:
        missing.append({"code": "guide_namespace_unverified", "n_unique_source_cells": len(guide_fallback)})
    def groups_for(keys, field):
        if field == "guide_id":
            return {(guide_namespaces[k], cells[k][field]) for k in keys if field not in invalid_fields[k]}
        return {(k[0], k[1], cells[k][field]) for k in keys if field not in invalid_fields[k]}
    def reuse(table, names):
        counts = Counter(cell for p in names for cell in table[p])
        return {"n_unique_reused_contributors": sum(n > 1 for n in counts.values()),
                "n_repeated_aggregate_edges": sum(n - 1 for n in counts.values()),
                "max_aggregate_multiplicity": max(counts.values(), default=0)}
    fold_reports = []
    for key, split in splits.items():
        trt, tet = union(treatment, split["train"]), union(treatment, split["test"])
        trc, tec = union(control, split["train"]), union(control, split["test"])
        overlaps = {"treatment_to_treatment": _overlap(trt, tet),
                    "control_to_control": _overlap(trc, tec),
                    "train_treatment_to_eval_control": _overlap(trt, tec),
                    "train_control_to_eval_treatment": _overlap(trc, tet),
                    "any_source_cell": _overlap(trt | trc, tet | tec)}
        shared = (trt | trc) & (tet | tec)
        affected = [p for p in split["test"] if (treatment[p] | control[p]) & shared]
        n_complete = sum(bool(treatment[p]) and bool(control[p]) for p in split["test"])
        fold_reports.append({"protocol": key[0], "seed": key[1], "fold": key[2],
            "split_sha256": split["sha256"], "source_cell_overlap": overlaps,
            "batch_sharing": {"all_contributors": _overlap(groups_for(trt | trc, "gemgroup"),
                                                            groups_for(tet | tec, "gemgroup")),
                              "treatment_only": _overlap(groups_for(trt, "gemgroup"), groups_for(tet, "gemgroup")),
                              "interpretation": "shared technical groups limit batch generalization; not automatically target leakage"},
            "guide_sharing_treatment_only": _overlap(groups_for(trt, "guide_id"), groups_for(tet, "guide_id")),
            "within_split_contributor_reuse": {
                "train": {"treatment": reuse(treatment, split["train"]), "control": reuse(control, split["train"]),
                          "cross_role": _overlap(trt, trc)},
                "eval": {"treatment": reuse(treatment, split["test"]), "control": reuse(control, split["test"]),
                         "cross_role": _overlap(tet, tec)}},
            "n_eval_aggregates": len(split["test"]), "n_eval_with_observed_both_roles": n_complete,
            "n_eval_affected_by_shared_source": len(affected), "affected_eval_labels": affected,
            "unmapped_eval_labels": [p for p in split["test"] if set(p.split("+")) & unmapped],
            "n_eval_aggregates_with_unmapped_targets": sum(bool(set(p.split("+")) & unmapped) for p in split["test"]),
            "n_train_unique_treatment_cells": len(trt), "n_eval_unique_treatment_cells": len(tet),
            "n_train_unique_control_cells": len(trc), "n_eval_unique_control_cells": len(tec)})
        for detail in [*overlaps.values(), fold_reports[-1]["batch_sharing"]["all_contributors"],
                       fold_reports[-1]["batch_sharing"]["treatment_only"],
                       fold_reports[-1]["guide_sharing_treatment_only"]]:
            detail["status"] = ("observed_overlap" if detail["n_unique_shared"] else
                                "unknown_or_partial" if missing or findings else "no_observed_overlap")
    violations = bool(findings)
    if policy in {"source_disjoint", "group_disjoint"}:
        violations |= any(f["source_cell_overlap"]["any_source_cell"]["n_unique_shared"] for f in fold_reports)
    if policy == "group_disjoint":
        violations |= any(f["batch_sharing"]["all_contributors"]["n_unique_shared"] for f in fold_reports)
    policy_result = ("not_enforced" if policy == "report" else
                     "fail" if violations else "unknown" if missing else "pass")
    return {"schema_version": 1, "policy": policy, "policy_result": policy_result,
            "evidence_status": "invalid" if findings else "partial" if missing else "complete",
            "evidence_validity": "invalid" if findings else "no_structural_findings",
            "evidence_completeness": "partial" if missing else "all_declared_fields_present",
            "attestation_verification": "not_independently_verified; declarations and hashes are not a trust anchor",
            "guide_identity_scope": "(declared guide namespace, guide_id); no library component; fallback to cell namespace is unverified",
            "gene_mapping": {"recorded_unmapped_symbols": sorted(unmapped),
                             "n_runs_without_mapping_evidence": mapping_runs_missing,
                             "source": "saved fit missing-target receipts; aliases and canonical mapping unverified"},
            "identity_scope": scope, "provenance_kind": adapter.get("provenance_kind", "unknown"),
            "control_use": adapter.get("control_use", []), "evidence": adapter.get("evidence"),
            "findings": findings, "missing_evidence": missing,
            "coverage": {"n_expected_aggregates": len(labels), "n_membership_rows": len(row_memberships),
                         "n_valid_unique_source_cells": len(cells), "n_referenced_unique_source_cells": len(referenced),
                         "n_duplicate_source_identities": len(ambiguous), "n_orphan_source_cells": len(unreferenced),
                         "n_valid_declared_exclusions": len(excluded),
                         "n_evaluation_rows": len(receipt["rows"])},
            "pool_cache": {"n_distinct_role_pools": len(pool_cache)},
            "aggregate_trace": trace, "evaluation_row_links": evaluation_links, "folds": fold_reports,
            "disjointness_feasibility": {
                "n_folds_with_shared_controls": sum(f["source_cell_overlap"]["control_to_control"]["n_unique_shared"] > 0 for f in fold_reports),
                "interpretation": "observed shared contributors require changing supplied memberships to obtain source-disjoint folds; no rows dropped or task changed"},
            "claims": {"biological_independence_certified": False,
                       "reported_symbol_split_semantics_unchanged": True,
                       "shared_reference_invalidates_legacy_score": False,
                       "preprocessing_predictions_and_metrics_changed": False,
                       "producer_attestations_independently_verified": False,
                       "report_exit_zero_means_evidence_accepted": False,
                       "absence_of_observed_overlap_is_not_proof_when_coverage_partial": True}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-receipt", required=True)
    parser.add_argument("--lineage-manifest")
    parser.add_argument("--policy", choices=("report", "source_disjoint", "group_disjoint"), default="report")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    check_new_target(args.output)
    receipt, receipt_ref = read_receipt(args.evaluation_receipt)
    result = audit(receipt, read_adapter(args.lineage_manifest, receipt), policy=args.policy)
    result["evaluation_receipt"] = receipt_ref
    result["code_sha256"] = file_hash(Path(__file__))
    result["code_provenance"] = code_provenance()
    result["result_content_sha256"] = digest(result)
    output = scoped_new_dir(args.output)
    data = json.dumps(result, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    create_new(str(output), "lineage-audit.json", data)
    create_new(str(output), "SHA256SUMS", (file_hash(output / "lineage-audit.json") + "  lineage-audit.json\n").encode())
    print(json.dumps({"output": str(output), "policy_result": result["policy_result"],
                      "evidence_status": result["evidence_status"], "coverage": result["coverage"]}, indent=2))
    return 2 if result["policy_result"] in {"fail", "unknown"} else 0


if __name__ == "__main__":
    raise SystemExit(main())
