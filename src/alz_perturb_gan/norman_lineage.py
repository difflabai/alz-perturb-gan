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
    splits = {}
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
    if not namespace or scope.get("status") != "canonical_physical_namespace":
        missing.append({"code": "physical_cell_namespace_unverified"})
    if adapter.get("provenance_kind") != "producer_attested":
        missing.append({"code": "aggregation_lineage_not_producer_attested"})
    if adapter.get("manifest_status") != "frozen":
        missing.append({"code": "lineage_manifest_not_frozen"})
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
    for obs in observations:
        cell_id, library = obs.get("cell_id"), obs.get("library_id")
        if not isinstance(cell_id, str) or not cell_id or not isinstance(library, str) or not library:
            issue("invalid_source_identity")
            continue
        key = (library, cell_id)
        counts[key] += 1
        if key in cells:
            ambiguous.add(key)
        cells[key] = obs
    for key in sorted(ambiguous):
        issue("duplicate_source_identity", library_id=key[0], cell_id=key[1], n_rows=counts[key])
    for key in ambiguous:
        del cells[key]  # never silently select one conflicting observation
    groups = membership.get("cell_groups", {})
    resolved_groups = {}
    for name, refs in groups.items():
        keys = []
        for ref in refs:
            if not isinstance(ref, list) or len(ref) != 2 or not all(isinstance(x, str) and x for x in ref):
                issue("invalid_contributor_reference", group=name)
                continue
            key = tuple(ref)
            keys.append(key)
            if key not in cells:
                issue("orphan_contributor", group=name, library_id=key[0], cell_id=key[1])
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
    fields_missing = Counter()
    for label in labels:
        row = row_memberships.get(label)
        if row is None:
            missing.append({"code": "aggregate_row_missing", "label": label})
            row = {}
        roles = {}
        for role in ("treatment", "control"):
            group_names = row.get(f"{role}_groups", [])
            refs = []
            for name in group_names:
                if name not in resolved_groups:
                    issue("orphan_group_reference", label=label, role=role, group=name)
                refs.extend(resolved_groups.get(name, set()))
            if len(refs) != len(set(refs)):
                issue("duplicate_within_aggregate_role", label=label, role=role,
                      n_repeated_edges=len(refs) - len(set(refs)))
            # Reuse shared pool sets: do not materialize the same 12k-control
            # contributor set once for every aggregate in a real packet.
            selected = (resolved_groups[group_names[0]] if len(group_names) == 1
                        and group_names[0] in resolved_groups else set(refs))
            if not selected:
                missing.append({"code": "contributors_missing", "label": label, "role": role})
            expected_label = label if role == "treatment" else "control"
            wrong = [key for key in selected if isinstance(cells[key].get("label"), str)
                     and cells[key]["label"] and cells[key]["label"] != expected_label]
            if wrong:
                issue("contributor_role_or_label_mismatch", label=label, role=role, n_cells=len(wrong))
            for key in selected:
                for field in ("label", "guide_id", "gemgroup"):
                    if not isinstance(cells[key].get(field), str) or not cells[key][field]:
                        fields_missing[field] += 1
            roles[role] = selected
            referenced.update(selected)
            # Many-to-one edges are resolved through the hash-bound membership
            # file's cell_groups and the observation file; shared pools stay one
            # pool, never one invented control cell per aggregate.
        treatment[label], control[label] = roles["treatment"], roles["control"]
        cross = _overlap(roles["treatment"], roles["control"])
        if cross["n_unique_shared"]:
            issue("cross_role_reuse", label=label, **cross)
        trace[label] = {"granularity": "aggregate", "membership_row_label": label,
                        **{role: {"groups": row.get(f"{role}_groups", []),
                                  "n_unique_contributors": len(roles[role]),
                                  "contributors_sha256": digest(sorted(roles[role]))}
                           for role in ("treatment", "control")}}
    for field, count in fields_missing.items():
        missing.append({"code": "observation_field_missing", "field": field,
                        "n_role_edges_affected": count})
    excluded = set()
    for exclusion in membership.get("excluded_cells", []):
        key = (exclusion.get("library_id"), exclusion.get("cell_id"))
        if key not in cells or not exclusion.get("reason") or key in referenced or key in excluded:
            issue("invalid_declared_exclusion", library_id=key[0], cell_id=key[1])
        excluded.add(key)
    unreferenced = set(cells) - referenced - excluded
    if unreferenced:
        issue("orphan_source_observations", n_unique_cells=len(unreferenced),
              examples=sorted(unreferenced)[:5])
    def union(table, names):
        return set().union(*(table[p] for p in names))
    def groups_for(keys, field):
        return {(namespace, cells[k]["library_id"], cells[k][field]) for k in keys
                if isinstance(cells[k].get(field), str) and cells[k][field]}
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
            "evidence_status": "partial" if missing else "invalid" if findings else "complete",
            "identity_scope": scope, "provenance_kind": adapter.get("provenance_kind", "unknown"),
            "control_use": adapter.get("control_use", []), "evidence": adapter.get("evidence"),
            "findings": findings, "missing_evidence": missing,
            "coverage": {"n_expected_aggregates": len(labels), "n_membership_rows": len(row_memberships),
                         "n_valid_unique_source_cells": len(cells), "n_referenced_unique_source_cells": len(referenced),
                         "n_duplicate_source_identities": len(ambiguous), "n_orphan_source_cells": len(unreferenced),
                         "n_evaluation_rows": len(receipt["rows"])},
            "aggregate_trace": trace, "evaluation_row_links": evaluation_links, "folds": fold_reports,
            "disjointness_feasibility": {
                "n_folds_with_shared_controls": sum(f["source_cell_overlap"]["control_to_control"]["n_unique_shared"] > 0 for f in fold_reports),
                "interpretation": "observed shared contributors require changing supplied memberships to obtain source-disjoint folds; no rows dropped or task changed"},
            "claims": {"biological_independence_certified": False,
                       "reported_symbol_split_semantics_unchanged": True,
                       "shared_reference_invalidates_legacy_score": False,
                       "preprocessing_predictions_and_metrics_changed": False,
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
