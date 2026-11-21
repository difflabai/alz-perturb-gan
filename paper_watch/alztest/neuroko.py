"""NeuroKO coverage, grouped EMPF benchmarks and cross-cell-type transfer."""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from importlib.metadata import version
import json
from pathlib import Path
from time import perf_counter, process_time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from .data import DATA_DIR, DataUnavailable

RECORD = "https://zenodo.org/records/23260389"
DOI = "10.5281/zenodo.23260389"
CONTROL = "INT"
LAYOUT = {
    "iNeurons": ("perturbation", "target_vector", "Replicate"),
    "iAstrocytes": ("target_gene", "guide_vector", "sequencing_batch"),
}
DEFAULT_DATA = DATA_DIR / "neuroko"
SEEDS = (0, 1, 2)


@dataclass
class Screen:
    name: str
    source: Path
    adata: object
    obs: pd.DataFrame
    pilot_intervals: list | None

    @property
    def groups(self):
        return sorted(self.obs["group"].unique())

    def coverage(self):
        return pd.crosstab(self.obs["target"], self.obs["group"])

    def inventory(self):
        counts = self.coverage()
        guide_counts = self.obs.groupby(["target", "group", "vector"], sort=True).size()
        guide_counts = guide_counts[guide_counts >= 20].groupby(level=(0, 1)).size()
        guide_counts = guide_counts.unstack(fill_value=0).reindex(
            index=counts.index, columns=counts.columns, fill_value=0
        )
        return {
            "path": str(self.source),
            "shape": [len(self.obs), self.adata.n_vars],
            "columns": dict(zip(("target", "vector", "group"), LAYOUT[self.name])),
            "groups": self.obs["group"].value_counts().sort_index().to_dict(),
            "targets_excluding_int": len(counts) - 1,
            "vectors": int(self.obs["vector"].nunique()),
            "control_cells_by_group": counts.loc[CONTROL].to_dict(),
            "targets_with_100_cells_in_every_group": int(
                (counts.drop(CONTROL).min(axis=1) >= 100).sum()
            ),
            "targets_with_two_vectors_of_20_cells_in_every_group": int(
                (guide_counts.drop(CONTROL).min(axis=1) >= 2).sum()
            ),
            "pilot_intervals": self.pilot_intervals,
            "coverage_by_target": counts.to_dict(orient="index"),
        }


@contextmanager
def open_screen(directory: Path, name: str, window_cells=None):
    import anndata as ad

    path = directory / f"{name}.h5ad"
    if not path.is_file():
        raise DataUnavailable(f"Missing {path}. Fetch the versioned H5AD at {RECORD}.")
    adata = ad.read_h5ad(path, backed="r")
    try:
        columns = LAYOUT[name]
        absent = sorted(set(columns) - set(adata.obs.columns))
        if absent:
            raise ValueError(f"{name}: missing release columns {absent}")
        obs = adata.obs.loc[:, list(columns)].copy()
        obs.columns = ["target", "vector", "group"]
        if obs.isna().any().any():
            raise ValueError(f"{name}: missing target, vector or group annotations")
        obs = obs.astype(str).reset_index(drop=True)
        if CONTROL not in set(obs["target"]) or obs["group"].nunique() < 2:
            raise ValueError(f"{name}: intergenic controls and at least two groups are required")
        pilot = adata.uns.get("neuroko_pilot")
        intervals = pilot["intervals"].tolist() if pilot is not None else None
        if window_cells is not None:
            if intervals is not None:
                raise ValueError("--window-cells expects the original H5AD files")
            obs = obs.groupby("group", sort=False).head(window_cells)
            rows = obs.index.to_numpy()
            breaks = np.flatnonzero(np.diff(rows) > 1) + 1
            intervals = [[int(x[0]), int(x[-1]) + 1] for x in np.split(rows, breaks)]
        yield Screen(name, path, adata, obs, intervals)
    finally:
        adata.file.close()


def shared_support(screens):
    """Return each target's minimum count across the groups of both screens."""
    counts = pd.concat([screen.coverage().min(axis=1) for screen in screens], axis=1)
    return counts.fillna(0).min(axis=1).sort_index().astype(int)


def select_targets(screens, min_cells, max_targets, requested_targets=None):
    support = shared_support(screens)
    if support[CONTROL] < min_cells:
        raise ValueError("intergenic control is below the per-group cell minimum")
    panel = support.drop(CONTROL)
    if requested_targets is not None:
        if len(set(requested_targets)) < 2:
            raise ValueError("request at least two distinct gene targets")
        if len(set(requested_targets)) != len(requested_targets):
            raise ValueError("gene targets must be unique")
        insufficient = {
            target: int(panel.get(target, 0)) for target in requested_targets
            if panel.get(target, 0) < min_cells
        }
        if insufficient:
            raise ValueError(f"requested targets below {min_cells} cells in a group: {insufficient}")
        return list(requested_targets)
    panel = panel[panel >= min_cells].sort_values(ascending=False, kind="stable")
    if len(panel) < 2:
        raise ValueError(f"only {len(panel)} targets have at least {min_cells} cells in each group")
    return panel.iloc[:max_targets].index.tolist()


def normalize_counts(matrix):
    """Apply per-cell library normalization to the complete sparse count rows."""
    matrix = csr_matrix(matrix, dtype=np.float32, copy=True)
    if not np.isfinite(matrix.data).all() or np.any(matrix.data < 0):
        raise ValueError("expected finite nonnegative counts")
    if np.any(matrix.data != np.floor(matrix.data)):
        raise ValueError("expected raw integer-valued counts in X")
    matrix.sum_duplicates()
    depths = np.asarray(matrix.sum(axis=1)).ravel()
    if np.any(depths <= 0):
        raise ValueError("expected nonzero libraries")
    matrix.data *= np.repeat(10_000 / depths, np.diff(matrix.indptr))
    np.log1p(matrix.data, out=matrix.data)
    return matrix


@dataclass
class Panel:
    x: csr_matrix
    labels: np.ndarray
    groups: np.ndarray
    features: np.ndarray
    vectors: np.ndarray


def read_panel(screen, conditions, cap, min_cells, seed=0, *, use_available_cells=False):
    rng = np.random.default_rng(seed)
    selected, labels, groups = [], [], []
    by_group = screen.obs.groupby(["group", "target"], sort=True).indices
    screen_groups = screen.groups
    for label, target in enumerate(conditions):
        available = [by_group[(group, target)] for group in screen_groups]
        size = min(cap, *(len(rows) for rows in available))
        if size < min_cells:
            raise ValueError(f"{screen.name}/{target}: only {size} usable cells per group")
        for group, rows in zip(screen_groups, available):
            group_size = min(cap, len(rows)) if use_available_cells else size
            picked = rng.choice(rows, size=group_size, replace=False)
            selected.extend(picked)
            labels.extend([label] * group_size)
            groups.extend([group] * group_size)
    order = np.argsort(selected)
    positions = np.asarray(selected)[order]
    rows = screen.obs.index.to_numpy()[positions]
    matrix = normalize_counts(screen.adata.X[rows, :])
    return Panel(matrix, np.asarray(labels)[order], np.asarray(groups)[order],
                 screen.adata.var_names.to_numpy(dtype=str), screen.obs["vector"].to_numpy()[positions])


def project_fold(train, test, n_features=512, n_components=30):
    """Fit feature selection, centering, PCA and scale on training rows once."""
    from .harness import pca_fit_transform

    mean = np.asarray(train.mean(axis=0)).ravel().astype(np.float64)
    variance = np.asarray(train.power(2).mean(axis=0)).ravel() - mean**2
    features = np.argsort(-variance, kind="stable")[:n_features]
    a = train[:, features].toarray().astype(np.float64)
    b = test[:, features].toarray().astype(np.float64)
    n_components = min(n_components, len(features), len(a) - 1)
    fit, (center, axes) = pca_fit_transform(a, n_comp=n_components)
    scale = fit.std(axis=0)
    keep = scale > 1e-8
    if not keep.any():
        raise ValueError("training PCA has no variable components")
    return fit[:, keep] / scale[keep], (b - center) @ axes[keep].T / scale[keep]


def _cosine(left, right):
    norm = np.linalg.norm(left) * np.linalg.norm(right)
    return float(left @ right / norm) if norm > 0 else None


def _target_shifts(panel, features):
    shifts = {}
    for group in sorted(set(panel.groups)):
        rows = panel.groups == group
        profiles = np.stack([
            np.asarray(panel.x[rows & (panel.labels == label)].mean(axis=0)).ravel()[features]
            for label in sorted(set(panel.labels))
        ])
        shifts[group] = profiles[1:] - profiles[0]
    return shifts


def profile_specificity(references, observed):
    denominator = np.outer(np.linalg.norm(observed, axis=1), np.linalg.norm(references, axis=1))
    scores = np.divide(observed @ references.T, denominator,
                       out=np.full(denominator.shape, np.nan), where=denominator > 0)
    result = []
    for k, row in enumerate(scores):
        same = float(row[k]) if np.isfinite(row[k]) else None
        wrong = row[np.isfinite(row) & (np.arange(len(row)) != k)]
        result.append({
            "cosine": same,
            "wrong_target_median": float(np.median(wrong)) if len(wrong) else None,
            "wrong_targets_at_least_as_high": int((wrong >= same).sum()) if same is not None else None,
            "wrong_targets_compared": len(wrong),
        })
    return result


def transfer_diagnostic(panels, targets):
    """Compare same-target transfer with within-type replication and wrong-target transfer."""
    names = list(panels)
    feature_indices = {
        name: pd.Series(np.arange(len(panel.features)), index=panel.features)
        for name, panel in panels.items()
    }
    for name, index in feature_indices.items():
        feature_indices[name] = index[~index.index.duplicated(keep=False)]
    common = feature_indices[names[0]].index.intersection(feature_indices[names[1]].index)
    if common.empty:
        raise ValueError("no uniquely named genes shared by the two screens")
    shifts = {name: _target_shifts(panels[name], feature_indices[name].loc[common].to_numpy())
              for name in names}
    non_targets = ~common.isin(targets)
    if not non_targets.any():
        raise ValueError("no shared features left after removing the panel's target genes")
    records = []
    for destination, source in (names, names[::-1]):
        transferred = np.mean(list(shifts[source].values()), axis=0)
        for group, observed in shifts[destination].items():
            within = np.mean([x for key, x in shifts[destination].items() if key != group], axis=0)
            specificity = {
                "within_type": profile_specificity(within, observed),
                "cross_type": profile_specificity(transferred, observed),
                "within_type_excluding_panel_genes": profile_specificity(within[:, non_targets],
                                                                         observed[:, non_targets]),
                "cross_type_excluding_panel_genes": profile_specificity(transferred[:, non_targets],
                                                                        observed[:, non_targets]),
            }
            for k, target in enumerate(targets):
                records.append({
                    "target": target, "destination": destination, "held_out_group": group,
                    **{key: scores[k] for key, scores in specificity.items()},
                })
    return {"common_unique_gene_symbols": len(common),
            "genes_after_excluding_panel_targets": int(non_targets.sum()), "records": records}


def target_expression_audit(panel, conditions, min_vector_cells=20):
    """Record target-gene RNA and detection by vector against group-matched INT."""
    records = []
    for label, target in enumerate(conditions[1:], start=1):
        gene = np.flatnonzero(panel.features == target)
        if len(gene) != 1:
            continue
        expression = panel.x[:, gene.item()].toarray().ravel()
        for group in sorted(set(panel.groups)):
            same_group = panel.groups == group
            control = expression[same_group & (panel.labels == 0)]
            selected = same_group & (panel.labels == label)
            for vector in sorted(set(panel.vectors[selected])):
                cells = expression[selected & (panel.vectors == vector)]
                if len(cells) >= min_vector_cells:
                    records.append({
                        "target": target, "group": group, "vector": vector,
                        "vector_cells": len(cells), "int_cells": len(control),
                        "mean_log_expression_delta": float(cells.mean() - control.mean()),
                        "detection_fraction_delta": float((cells > 0).mean() - (control > 0).mean()),
                        "int_detection_fraction": float((control > 0).mean()),
                    })
    return records


def guide_holdout_diagnostic(panel, conditions, min_cells=20):
    """Compare a withheld vector with other vectors of the same target in one group."""
    rng = np.random.default_rng(0)
    records = []
    for group in sorted(set(panel.groups)):
        in_group = panel.groups == group
        control_rows = rng.permutation(np.flatnonzero(in_group & (panel.labels == 0)))
        train_control, test_control = np.array_split(control_rows, 2)
        train_reference = np.asarray(panel.x[train_control].mean(axis=0)).ravel()
        test_reference = np.asarray(panel.x[test_control].mean(axis=0)).ravel()
        for label, target in enumerate(conditions[1:], start=1):
            at_target = in_group & (panel.labels == label)
            vectors, counts = np.unique(panel.vectors[at_target], return_counts=True)
            for vector in vectors[counts >= min_cells]:
                heldout = at_target & (panel.vectors == vector)
                others = at_target & ~heldout
                if others.sum() < min_cells:
                    continue
                prediction = np.asarray(panel.x[others].mean(axis=0)).ravel() - train_reference
                observed = np.asarray(panel.x[heldout].mean(axis=0)).ravel() - test_reference
                records.append({
                    "target": target, "group": group, "held_out_vector": vector,
                    "train_cells": int(others.sum()), "test_cells": int(heldout.sum()),
                    "shift_cosine": _cosine(prediction, observed),
                })
    return records


def expression_matches(panel, conditions, heldout):
    """Choose a wrong-target arm by gene expression in training INT cells."""
    control = panel.x[(~heldout) & (panel.labels == 0)]
    means = np.asarray(control.mean(axis=0)).ravel()
    levels = {}
    for label, target in enumerate(conditions[1:], start=1):
        genes = np.flatnonzero(panel.features == target)
        if len(genes) == 1:
            levels[label] = float(means[genes.item()])
    return {
        label: min((other for other in levels if other != label),
                   key=lambda other: (abs(level - levels[other]), conditions[other]))
        for label, level in levels.items()
    } if len(levels) > 1 else {}


def _resample(rows, n, rng):
    return rows[rng.choice(len(rows), n, replace=True)]


def score_fold(panel, group, conditions, steps, seed, projected, *, recipe_name="empf", learning_rate=0.005):
    import torch

    from .harness import ConditionalPGAN, energy_distance

    heldout = panel.groups == group
    xtrain, xtest = projected
    ytrain, ytest = panel.labels[~heldout], panel.labels[heldout]
    train = [xtrain[ytrain == k] for k in range(len(conditions))]
    test = [xtest[ytest == k] for k in range(len(conditions))]
    model_arms = {}
    fit_times = {}
    if steps:
        shuffled = np.random.default_rng(seed + 31).permutation(ytrain)
        for arm, labels in ((recipe_name, ytrain), (f"shuffled_{recipe_name}", shuffled)):
            with torch.random.fork_rng(devices=[]):
                torch.default_generator.manual_seed(seed)
                model = ConditionalPGAN(xtrain.shape[1], len(conditions), seed=seed,
                                        recipe_name=recipe_name, learning_rate=learning_rate)
            started_wall, started_cpu = perf_counter(), process_time()
            model.fit(xtrain, labels, steps=steps)
            fit_times[arm] = {
                "cpu_seconds": process_time() - started_cpu,
                "wall_seconds": perf_counter() - started_wall,
            }
            model_arms[arm] = model

    n_draws = 256
    baseline_rng = np.random.default_rng(seed)
    control = _resample(train[0], n_draws, baseline_rng)
    matches = expression_matches(panel, conditions, heldout)
    noise = baseline_rng.standard_normal((n_draws, xtrain.shape[1]))
    predicted_controls = {
        "control_only": control,
        "mean_shift": control,
        "empirical_target": control,
        "expression_matched_target": control,
        "diagonal_gaussian": train[0].mean(axis=0) + noise * train[0].std(axis=0),
    }
    predicted_controls.update({arm: model.sample(0, n_draws, seed=seed + 100)
                               for arm, model in model_arms.items()})
    records = []
    for k, target in enumerate(conditions[1:], start=1):
        shift = train[k].mean(axis=0) - train[0].mean(axis=0)
        predictions = {
            "control_only": control,
            "mean_shift": control + shift,
            "empirical_target": _resample(train[k], n_draws, baseline_rng),
            "diagonal_gaussian": train[k].mean(axis=0) + noise * train[k].std(axis=0),
        }
        if k in matches:
            predictions["expression_matched_target"] = _resample(train[matches[k]], n_draws, baseline_rng)
        predictions.update({arm: model.sample(k, n_draws, seed=seed + 100 + k)
                            for arm, model in model_arms.items()})
        observed_shift = test[k].mean(axis=0) - test[0].mean(axis=0)
        observed_spread = np.var(test[k], axis=0).sum()
        records.append({
            "target": target, "train_cells": len(train[k]), "test_cells": len(test[k]),
            "expression_matched_target": conditions[matches[k]] if k in matches else None,
            "scores": {
                arm: {
                    "energy_distance": energy_distance(predicted, test[k], seed=seed),
                    "shift_cosine": _cosine(
                        predicted.mean(axis=0) - predicted_controls[arm].mean(axis=0), observed_shift
                    ),
                    "spread_ratio": float(np.sqrt(np.var(predicted, axis=0).sum() / observed_spread))
                    if observed_spread > 0 else None,
                } for arm, predicted in predictions.items()
            },
        })
    return {"held_out_group": group, "seed": seed, "components": xtrain.shape[1],
            "fit_times": fit_times, "targets": records}


def load_comparison(screens, min_cells, cap, max_targets, requested_targets=None, *, use_available_cells=False):
    targets = select_targets(screens, min_cells, max_targets, requested_targets)
    conditions = [CONTROL, *targets]
    panels = {
        s.name: read_panel(s, conditions, cap, min_cells, use_available_cells=use_available_cells)
        for s in screens
    }
    return conditions, panels


def compare(conditions, panels):
    return {
        "targets": conditions[1:], "transfer": transfer_diagnostic(panels, conditions[1:]),
        "target_expression_by_vector": {
            name: target_expression_audit(panel, conditions) for name, panel in panels.items()
        },
        "guide_holdout_within_group": {
            name: guide_holdout_diagnostic(panel, conditions) for name, panel in panels.items()
        },
    }


def evaluate(screens, *, min_cells, cap, max_targets, steps, seeds,
             requested_targets=None, recipe_name="empf", learning_rate=0.005):
    conditions, panels = load_comparison(screens, min_cells, cap, max_targets, requested_targets)
    report = compare(conditions, panels)
    scores = {}
    for screen in screens:
        panel = panels[screen.name]
        scores[screen.name] = []
        for group in screen.groups:
            heldout = panel.groups == group
            projection = project_fold(panel.x[~heldout], panel.x[heldout])
            for seed in seeds:
                print(f"{recipe_name}: {screen.name} held out {group}, seed {seed}, {steps} steps", flush=True)
                scores[screen.name].append(
                    score_fold(panel, group, conditions, steps, seed, projection,
                               recipe_name=recipe_name, learning_rate=learning_rate)
                )
    report["folds"] = scores
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "transfer", "evaluate"))
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--window-cells", type=int, help="use the first N rows of each release group")
    parser.add_argument("--min-cells", type=int, default=100)
    parser.add_argument("--cap", type=int, default=160)
    parser.add_argument("--targets", type=int, default=6, help="maximum size of the coverage-ranked panel")
    parser.add_argument("--target", action="append", help="choose an explicit gene panel by repeating this option")
    parser.add_argument("--use-available-cells", action="store_true",
                        help="for transfer, sample up to the cap independently in each target and group")
    parser.add_argument("--steps", type=int, default=300, help="use 0 to evaluate the baselines alone")
    parser.add_argument("--recipe", choices=("empf", "unified_v11"), default="empf",
                        help="public ParticleGAN recipe for the evaluation arms")
    parser.add_argument("--learning-rate", type=float, default=0.005,
                        help="base learning rate for the model and shuffled-label control")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = parser.parse_args()
    if not 32 <= args.min_cells <= args.cap or args.targets < 2 or args.steps < 0:
        parser.error("require 32 <= min-cells <= cap, targets >= 2 and steps >= 0")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning-rate must be finite and positive")
    if args.use_available_cells and args.command != "transfer":
        parser.error("--use-available-cells applies only to transfer")
    if args.window_cells is not None and args.window_cells < 1:
        parser.error("window-cells must be positive")
    if args.command == "evaluate":
        import torch

        torch.set_num_threads(2)
    try:
        with ExitStack() as stack:
            screens = [stack.enter_context(open_screen(args.data, cell_type, args.window_cells))
                       for cell_type in LAYOUT]
            report = {
                "doi": DOI, "screens": {s.name: s.inventory() for s in screens},
                "versions": {package: version(package) for package in ("anndata", "numpy", "scipy")},
                "minimum_group_cells_by_shared_target": shared_support(screens).drop(CONTROL).to_dict(),
            }
            if args.command in ("transfer", "evaluate"):
                config = dict(min_cells=args.min_cells, cap=args.cap, max_targets=args.targets)
                if args.target is not None:
                    config["requested_targets"] = args.target
                report["config"] = config
                if args.command == "transfer":
                    config["use_available_cells"] = args.use_available_cells
                    conditions, panels = load_comparison(screens, **config)
                    report.update(compare(conditions, panels))
                else:
                    config.update(steps=args.steps, seeds=args.seeds, recipe_name=args.recipe,
                                  learning_rate=args.learning_rate)
                    report["versions"].update(torch=version("torch"), particlegan=version("particlegan"))
                    report.update(evaluate(screens, **config))
    except DataUnavailable as error:
        parser.exit(2, f"{error}\n")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
