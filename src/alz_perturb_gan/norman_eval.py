"""CPU-only, train-fold-fitted Norman comparison and provenance harness.

No data acquisition, outcome-derived phenocopy labels, torch, or GPU imports.
Definitions and packet handoff: docs/norman-evaluation.md.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import itertools
import json
from pathlib import Path
import platform
import subprocess
from typing import Any

import numpy as np

from .paths import check_new_target, create_new, scoped, scoped_new_dir


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def array_hash(value: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(str(value.dtype).encode())
    h.update(json.dumps(value.shape).encode())
    h.update(np.ascontiguousarray(value).tobytes())
    return h.hexdigest()


def members(label: str) -> tuple[str, ...]:
    if not isinstance(label, str) or not label.strip():
        raise ValueError("nonempty perturbation labels required")
    genes = label.split("+")
    if (len(genes) not in (1, 2) or len(set(genes)) != len(genes)
            or any(g != g.strip() or not g or g.lower() in
                   {"ctrl", "control", "non-targeting"} for g in genes)):
        raise ValueError(f"expected a single or two distinct gene identities: {label!r}")
    return tuple(sorted(genes))


def validate_labels(labels: list[str]) -> None:
    canonical = [members(p) for p in labels]
    if len(set(canonical)) != len(labels):
        raise ValueError("duplicate perturbation identity (including reordered combinations)")


def _gene_set(labels: list[str]) -> set[str]:
    return {g for p in labels for g in members(p)}


def novelty(label: str, train: list[str]) -> dict[str, Any]:
    genes = members(label)
    seen = _gene_set(train)
    unseen = sum(g not in seen for g in genes)
    return {"unseen_label": label not in train, "n_members": len(genes),
            "n_unseen_genes": unseen,
            "stratum": ("single_seen_gene" if unseen == 0 else "single_unseen_gene")
            if len(genes) == 1 else f"combination_{unseen}_unseen_genes",
            "all_single_components_in_train": all(g in train for g in genes)}


def leakage_audit(labels: list[str], split: dict[str, Any]) -> dict[str, Any]:
    """Reject malformed splits; gene overlap is allowed only for label holdout."""
    validate_labels(labels)
    groups = [split[k] for k in ("train", "test", "excluded")]
    if any(len(g) != len(set(g)) for g in groups):
        raise ValueError("duplicate split membership")
    tr, te, ex = map(set, groups)
    if tr & te or tr & ex or te & ex or tr | te | ex != set(labels):
        raise ValueError("split must partition all labels into train/test/excluded")
    if not tr or not te:
        raise ValueError("empty train or test fold")
    overlap = sorted(_gene_set(sorted(tr)) & _gene_set(sorted(te)))
    if split["protocol"] == "whole_gene":
        held = set(split["heldout_genes"])
        expected_tr = {p for p in labels if not set(members(p)) & held}
        expected_te = {p for p in labels if set(members(p)) <= held}
        if overlap or tr != expected_tr or te != expected_te:
            raise ValueError("whole-gene split leaks a held-out gene or misassigns labels")
    elif split["protocol"] != "perturbation_label" or ex:
        raise ValueError("unknown protocol or excluded label in label holdout")
    return {"passed": True, "scope": "perturbation membership; cell/donor/batch independence not established",
            "label_overlap": [], "perturbed_gene_overlap": overlap,
            "n_train": len(tr), "n_test": len(te), "n_excluded": len(ex),
            "novelty": {p: novelty(p, sorted(tr)) for p in sorted(te)}}


def make_splits(labels: list[str], protocol: str, seed: int) -> list[dict[str, Any]]:
    """Five deterministic folds, invariant to packet row order.

    Whole-gene: partition genes; train outside the block, test wholly inside,
    purge bridging doubles. Thus some doubles have no strict OOF prediction.
    """
    validate_labels(labels)
    labels = sorted(labels)
    rng = np.random.default_rng(seed)
    assignments: dict[str, int] = {}
    if protocol == "perturbation_label":
        for n in (1, 2):
            group = [p for p in labels if len(members(p)) == n]
            rng.shuffle(group)
            assignments.update({p: i % 5 for i, p in enumerate(group)})
    elif protocol == "whole_gene":
        genes = sorted(_gene_set(labels))
        if len(genes) < 5:
            raise ValueError("whole-gene five-fold split needs at least five genes")
        rng.shuffle(genes)
        assignments = {g: i % 5 for i, g in enumerate(genes)}
    else:
        raise ValueError("protocol must be perturbation_label or whole_gene")
    folds = []
    for fold in range(5):
        held = sorted(g for g, f in assignments.items() if f == fold)
        if protocol == "perturbation_label":
            test = [p for p in labels if assignments[p] == fold]
            train = [p for p in labels if assignments[p] != fold]
        else:
            test = [p for p in labels if set(members(p)) <= set(held)]
            train = [p for p in labels if not set(members(p)) & set(held)]
        split = {"protocol": protocol, "seed": int(seed), "fold": fold,
                 "train": train, "test": test,
                 "excluded": sorted(set(labels) - set(train) - set(test)),
                 "heldout_genes": held if protocol == "whole_gene" else []}
        split["audit"] = leakage_audit(labels, split)
        split["sha256"] = digest(split)
        folds.append(split)
    tests = [p for s in folds for p in s["test"]]
    if len(tests) != len(set(tests)):
        raise ValueError("test labels repeat across folds")
    if protocol == "perturbation_label" and set(tests) != set(labels):
        raise ValueError("label folds must cover every label once")
    return folds


@dataclass
class Packet:
    labels: list[str]
    genes: list[str]
    delta: np.ndarray
    target_genes: list[str]
    target_delta: np.ndarray
    provenance: dict[str, Any]
    de_stat: np.ndarray | None = None

    def validate(self) -> None:
        validate_labels(self.labels)
        n, g = len(self.labels), len(self.genes)
        valid_axis = lambda axis: all(isinstance(v, str) and v.strip() == v and v for v in axis)
        if not n or g < 2 or len(set(self.genes)) != g or not valid_axis(self.genes):
            raise ValueError("nonempty unique expression gene axis required")
        if (len(set(self.target_genes)) != len(self.target_genes)
                or not self.target_genes or not valid_axis(self.target_genes)):
            raise ValueError("unique target gene axis required")
        for array, shape in ((self.delta, (n, g)),
                             (self.target_delta, (n, len(self.target_genes)))):
            if array.shape != shape or array.dtype.kind != "f" or not np.isfinite(array).all():
                raise ValueError("incorrect shape, dtype or nonfinite packet array")
        if self.de_stat is not None and (self.de_stat.shape != (n, g)
                                        or not np.isfinite(self.de_stat).all()):
            raise ValueError("invalid scoring-only DE statistic")


def load_packet(manifest_path: str | Path) -> Packet:
    """Read hash-bound existing artifacts without pickle or implicit acquisition."""
    path = scoped(str(Path(manifest_path).absolute()))
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 1 or manifest.get("status") != "frozen":
        raise ValueError("a frozen schema 1 packet is required")
    required = ("dataset", "normalization", "gene_selection", "controls",
                "source", "preparation", "terms", "limitations", "target_row_labels")
    if any(k not in manifest or not manifest[k] for k in required):
        raise ValueError("missing packet preparation/provenance fields")
    # All expression summaries must be independently per label. Transform fitting
    # on whole-cohort outcomes is not accepted as a leakage-safe packet.
    if manifest.get("outcome_transform_scope") != "per_label_only":
        raise ValueError("global outcome-fitted preprocessing is refused")
    if manifest["gene_selection"].get("fit_scope") not in {"controls_only", "external_frozen"}:
        raise ValueError("gene axis must be controls-only or externally frozen")
    loaded = {}
    hashes = {}
    for key in ("pseudobulk", "target_deltas"):
        spec = manifest[key]
        artifact = scoped(str((path.parent / spec["path"]).absolute()))
        sha = file_hash(artifact)
        if sha != spec["sha256"]:
            raise ValueError(f"{key} byte hash mismatch")
        hashes[key] = sha
        with np.load(artifact, allow_pickle=False) as arrays:
            loaded[key] = {name: arrays[name] for name in arrays.files}
    bulk, target = loaded["pseudobulk"], loaded["target_deltas"]
    if bulk["D"].dtype.kind != "f" or target["Dt"].dtype.kind != "f":
        raise ValueError("floating expression arrays required")
    labels = bulk["perts"].tolist()
    if labels != manifest["target_row_labels"]:
        raise ValueError("target row-order attestation differs from pseudobulk")
    if "perts" in target and target["perts"].tolist() != labels:
        raise ValueError("target row labels misaligned")
    # source bytes may live elsewhere; do not assert verification if unavailable.
    source = manifest["source"]
    source_verification = "not_rehashed"
    if source.get("path"):
        raw = scoped(str((path.parent / source["path"]).absolute()))
        if file_hash(raw) != source["sha256"]:
            raise ValueError("raw source hash mismatch")
        source_verification = "sha256_verified"
    packet = Packet(labels, bulk["genes"].tolist(), bulk["D"].astype(np.float64),
                    target["targets"].tolist(), target["Dt"].astype(np.float64),
                    {"manifest": manifest, "manifest_sha256": file_hash(path),
                     "artifact_sha256": hashes, "source_verification": source_verification,
                     "target_rows": "embedded_labels_verified" if "perts" in target
                     else "preparer_attestation_only"},
                    bulk.get("T"))
    packet.validate()
    return packet


def predict_baselines(train_labels: list[str], train_delta: np.ndarray,
                      train_target_delta: np.ndarray, target_genes: list[str],
                      test_labels: list[str], *, components: int = 10,
                      ridge: float = 0.1) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Only training outcomes enter fitting; held-out labels supply identities.

    PCA ridge matches the historical two-sided regularized fit. Gene embeddings
    use target-gene expression measured across TRAIN perturbations only. A gene
    may be measured without being targeted in train. This is not external gene
    knowledge and cannot transfer across an unmeasured expression axis.
    """
    if components < 1 or not np.isfinite(ridge) or ridge <= 0:
        raise ValueError("positive components and ridge required")
    y = np.asarray(train_delta, dtype=np.float64)
    dt = np.asarray(train_target_delta, dtype=np.float64)
    if (y.ndim != 2 or dt.shape != (len(train_labels), len(target_genes))
            or y.shape[0] != len(train_labels) or not len(train_labels)
            or not np.isfinite(y).all() or not np.isfinite(dt).all()):
        raise ValueError("invalid training arrays")
    mean = y.mean(axis=0)
    predictions = {"no_change": np.zeros((len(test_labels), y.shape[1])),
                   "mean_effect": np.tile(mean, (len(test_labels), 1))}
    single = {p: y[i] for i, p in enumerate(train_labels) if len(members(p)) == 1}
    predictions["additive"] = np.stack([
        sum((single.get(g, np.zeros_like(mean)) for g in members(p)),
            start=np.zeros_like(mean)) for p in test_labels])
    # Training pseudobulk PCA. Remove null singular values instead of dividing
    # by zero on tiny or rank-deficient data.
    u, s, vt = np.linalg.svd((y - mean).T, full_matrices=False)
    tol = (s[0] if len(s) else 0) * max(y.shape) * np.finfo(float).eps
    k = min(components, int(np.count_nonzero(s > tol)))
    missing = sorted(_gene_set(train_labels + test_labels) - set(target_genes))
    if k:
        basis = u[:, :k]
        coords = (dt - dt.mean(axis=0)).T @ vt[:k].T / s[:k]
        embeddings = dict(zip(target_genes, coords))
        def design(labels):
            return np.stack([sum((embeddings.get(g, np.zeros(k)) for g in members(p)),
                                 start=np.zeros(k)) for p in labels])
        p = design(train_labels)
        left = np.linalg.solve(basis.T @ basis + ridge * np.eye(k),
                               basis.T @ (y - mean).T @ p)
        w = np.linalg.solve(p.T @ p + ridge * np.eye(k), left.T).T
        predictions["pca_ridge"] = design(test_labels) @ w.T @ basis.T + mean
    else:
        predictions["pca_ridge"] = np.tile(mean, (len(test_labels), 1))
    return predictions, {"effective_components": k, "requested_components": components,
                         "ridge": ridge, "missing_target_genes_zero_embedding": missing,
                         "additive_missing_single_fallback": "zero_delta",
                         "additive_missing_single_count": {
                             p: sum(g not in single for g in members(p)) for p in test_labels},
                         "fit_labels_sha256": digest(train_labels),
                         "fit_delta_sha256": array_hash(y),
                         "fit_target_delta_sha256": array_hash(dt)}


def pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    a, b = a - a.mean(), b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return None if denom <= 1e-14 else float(np.clip(a @ b / denom, -1, 1))


def score_predictions(pred: np.ndarray, true: np.ndarray, reference: np.ndarray,
                      de_stat: np.ndarray | None = None) -> list[dict[str, float | None]]:
    """Local Systema-style centroid Pearson; reference is TRAIN centroid.

    Evaluation centroids are used only for scoring, including top-1000 gene
    selection and centroid retrieval; never supplied to a model or selector.
    Undefined Pearson is null, with explicit counts in summaries.
    """
    if pred.shape != true.shape or len(pred) == 0 or not np.isfinite(pred).all():
        raise ValueError("invalid predictions")
    rows = []
    for i, (p, y) in enumerate(zip(pred, true)):
        dist = np.mean((true - p) ** 2, axis=1)
        tied = np.isclose(dist, dist[i], rtol=1e-10, atol=1e-12)
        better = (dist < dist[i]) & ~tied
        row = {"mse": float(np.mean((p - y) ** 2)),
               "pearson_delta": pearson(p, y),
               "systema_train_centroid_pearson": pearson(p - reference, y - reference),
               "centroid_accuracy": 1.0 / int(tied.sum()) if not better.any() else 0.0,
               "centroid_rank": (float(better.sum() + (tied.sum() - 1) / 2) /
                                 (len(true) - 1)) if len(true) > 1 else None}
        if de_stat is not None:
            top = np.argsort(-np.abs(de_stat[i]), kind="stable")[:min(1000, len(y))]
            row["pearson_delta_top1000"] = pearson(p[top], y[top])
        rows.append(row)
    return rows


METRICS = ("mse", "pearson_delta", "systema_train_centroid_pearson",
           "centroid_accuracy", "centroid_rank", "pearson_delta_top1000")


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for metric in METRICS:
        valid = [r[metric] for r in rows if r.get(metric) is not None]
        out[metric] = {"mean": float(np.mean(valid)) if valid else None,
                       "n_defined": len(valid), "n_total": len(rows)}
    return out


def paired_bootstrap(a: list[dict[str, Any]], b: list[dict[str, Any]], metric: str,
                     *, seed: int = 1729, draws: int = 2000) -> dict[str, Any]:
    """Cluster by perturbation label across repeats, preserving paired arms.

    Average seed-specific differences within identity before bootstrapping.
    Conditions on observed split seeds and references; not biological replicate
    uncertainty. Strict splits can score a double in only a subset of seeds.
    """
    if draws < 1:
        raise ValueError("positive bootstrap draws required")
    key = lambda r: (r["seed"], r["fold"], r["label"])
    aa, bb = {key(r): r for r in a}, {key(r): r for r in b}
    if len(aa) != len(a) or len(bb) != len(b) or aa.keys() != bb.keys():
        raise ValueError("paired comparison requires identical unique seed/fold/label keys")
    for k in aa:
        for identity in ("split_sha256", "data_sha256", "reference_sha256"):
            if aa[k].get(identity) != bb[k].get(identity):
                raise ValueError(f"paired comparison has mismatched {identity}")
    by_label: dict[str, list[float]] = {}
    for k in sorted(aa):
        x, y = aa[k].get(metric), bb[k].get(metric)
        if x is not None and y is not None:
            if not np.isfinite([x, y]).all():
                raise ValueError("nonfinite paired metric")
            by_label.setdefault(k[2], []).append(x - y)
    d = np.array([np.mean(v) for _, v in sorted(by_label.items())])
    if not len(d):
        return {"difference": None, "ci95": None, "n_labels": 0,
                "n_paired_rows": 0, "n_excluded_undefined": len(aa)}
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(draws)])
    return {"difference": float(d.mean()), "ci95": np.quantile(boot, [.025, .975]).tolist(),
            "n_labels": len(d), "n_paired_rows": sum(map(len, by_label.values())),
            "n_excluded_undefined": len(aa) - sum(map(len, by_label.values())),
            "draws": draws, "seed": seed,
            "unit": "label; seed differences averaged within label",
            "scope": "conditional on split seeds and fitted references; not donor/gene uncertainty"}


def evaluate(packet: Packet, *, seeds=(20261022, 20261023, 20261024),
             components=10, ridge=0.1, bootstrap_draws=2000) -> dict[str, Any]:
    packet.validate()
    if len(set(seeds)) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("at least three distinct predeclared split seeds required")
    if bootstrap_draws < 1:
        raise ValueError("positive bootstrap draws required")
    index = {p: i for i, p in enumerate(packet.labels)}
    data_sha = digest({"labels": packet.labels, "genes": packet.genes,
                       "target_genes": packet.target_genes,
                       "delta": array_hash(packet.delta),
                       "target_delta": array_hash(packet.target_delta),
                       "de_stat": array_hash(packet.de_stat) if packet.de_stat is not None else None})
    runs, rows = [], []
    for protocol in ("perturbation_label", "whole_gene"):
        for seed in seeds:
            for split in make_splits(packet.labels, protocol, seed):
                tr = [index[p] for p in split["train"]]
                te = [index[p] for p in split["test"]]
                pred, fit = predict_baselines(split["train"], packet.delta[tr],
                    packet.target_delta[tr], packet.target_genes, split["test"],
                    components=components, ridge=ridge)
                reference = packet.delta[tr].mean(axis=0)
                runs.append({"split": split, "fit": fit,
                             "reference_sha256": array_hash(reference),
                             "prediction_sha256": {arm: array_hash(p) for arm, p in pred.items()}})
                for arm, p in pred.items():
                    scores = score_predictions(p, packet.delta[te], reference,
                        packet.de_stat[te] if packet.de_stat is not None else None)
                    for label, score in zip(split["test"], scores):
                        rows.append({"protocol": protocol, "seed": int(seed), "fold": split["fold"],
                                     "arm": arm, "label": label,
                                     "split_sha256": split["sha256"], "data_sha256": data_sha,
                                     "reference_sha256": array_hash(reference),
                                     **split["audit"]["novelty"][label], **score})
    summaries, comparisons, coverage = {}, {}, {}
    arms = sorted({r["arm"] for r in rows})
    for protocol in ("perturbation_label", "whole_gene"):
        rr = [r for r in rows if r["protocol"] == protocol]
        coverage[protocol] = {}
        for seed in seeds:
            scored = {r["label"] for r in rr if r["seed"] == seed}
            coverage[protocol][str(seed)] = {"n_scored": len(scored),
                "n_single": sum(len(members(p)) == 1 for p in scored),
                "n_double": sum(len(members(p)) == 2 for p in scored),
                "unscored_labels": sorted(set(packet.labels) - scored)}
        strata = ["all", "single", "combination"] + sorted({r["stratum"] for r in rr})
        summaries[protocol], comparisons[protocol] = {}, {}
        for stratum in strata:
            subset = [r for r in rr if stratum == "all" or r["stratum"] == stratum
                      or (stratum == "single" and r["n_members"] == 1)
                      or (stratum == "combination" and r["n_members"] == 2)]
            summaries[protocol][stratum] = {}
            comparisons[protocol][stratum] = {}
            for arm in arms:
                ar = [r for r in subset if r["arm"] == arm]
                summaries[protocol][stratum][arm] = {
                    "pooled_rows": summarize(ar),
                    "by_seed": {str(s): summarize([r for r in ar if r["seed"] == s]) for s in seeds}}
            for left, right in itertools.combinations(arms, 2):
                a = [r for r in subset if r["arm"] == left]
                b = [r for r in subset if r["arm"] == right]
                comparisons[protocol][stratum][f"{left}_minus_{right}"] = {
                    m: paired_bootstrap(a, b, m, draws=bootstrap_draws) for m in METRICS}
    return {"schema_version": 1, "seeds": list(seeds), "runs": runs, "rows": rows,
            "coverage": coverage, "summary": summaries, "paired_bootstrap": comparisons,
            "packet_provenance": packet.provenance, "data_content_sha256": data_sha,
            "config": {"components": components, "ridge": ridge, "bootstrap_draws": bootstrap_draws},
            "claims": {"official_rubric": False, "matched_reference_reproduction": False,
                       "reference_target_only": "historical ridge Pearson delta top1000 0.668 +/- 0.01",
                       "phenocopy_labels": "none; expression prediction and identity retrieval only",
                       "systema_variant": "local train-perturbation-centroid reference, full fixed gene axis",
                       "seed_role": "split repetitions; deterministic baselines have no training RNG"}}


def code_provenance() -> dict[str, Any]:
    repo = Path(__file__).resolve().parents[2]
    def git(*args):
        result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    return {"git_head": git("rev-parse", "HEAD"),
            "git_status": git("status", "--porcelain"),
            "files": {str(p.relative_to(repo)): file_hash(p) for p in
                      [Path(__file__), Path(__file__).with_name("paths.py")]},
            "python": platform.python_version(), "numpy": np.__version__,
            "platform": platform.platform(), "device": "cpu"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True, help="new absolute directory within ALZ_SCOPE_ROOT")
    parser.add_argument("--seeds", type=int, nargs="+", default=[20261022, 20261023, 20261024])
    parser.add_argument("--components", type=int, default=10)
    parser.add_argument("--ridge", type=float, default=0.1)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    args = parser.parse_args(argv)
    check_new_target(args.output)
    packet = load_packet(args.manifest)
    result = evaluate(packet, seeds=args.seeds, components=args.components, ridge=args.ridge,
                      bootstrap_draws=args.bootstrap_draws)
    result["code_provenance"] = code_provenance()
    result["invocation"] = vars(args)
    result["result_content_sha256"] = digest(result)
    output = scoped_new_dir(args.output)
    data = json.dumps(result, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    create_new(str(output), "results.json", data)
    create_new(str(output), "SHA256SUMS", (hashlib.sha256(data).hexdigest() + "  results.json\n").encode())
    print(json.dumps({"output": str(output), "result_content_sha256": result["result_content_sha256"],
                      "coverage": {p: {s: c["n_scored"] for s, c in v.items()}
                                   for p, v in result["coverage"].items()}}, indent=2))


if __name__ == "__main__":
    main()
