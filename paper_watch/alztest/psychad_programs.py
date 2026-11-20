"""Read two public PsychAD gene program scores from a selected EMPF configuration.

The rate and checkpoint are chosen on an inner split of the training donors.
Only the selected configuration is scored on held-out donors, next to the
mean-shift, empirical-resampling, and shuffled-label EMPF control arms.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
from statistics import mean

import numpy as np
from scipy.stats import spearmanr
import torch

from .data import load_psychad
from .harness import ConditionalPGAN, energy_distance, mean_shift_prediction
from .ll_panel import LIPID, LYSO
from .psychad import DRAWS, GROUPS, MIN_TRAIN_DONORS, donor_profiles, split_donors

CHECKPOINTS = (300, 1200, 2400)
LEARNING_RATES = (0.0025, 0.005, 0.012)
SEEDS = (0, 1, 2)
# The pinned CELLxGENE extract has no CTSD column. Its GBA1 has this Ensembl ID.
LYSOSOME = tuple("GBA1" if gene == "GBA" else gene for gene in LYSO if gene != "CTSD")
GBA1_ENSEMBL = "ENSG00000177628"
SELECTION_METRIC = "mean_absolute_correlation_error"
ARMS = ("empf", "shuffled", "mean_shift", "empirical")
METRICS = ("mean_energy_distance", "mean_absolute_correlation_error")
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 0
NOISE_SPLITS = 200
MIN_VALIDATION_DONORS = 3


def marker_expression(adata):
    profiles, groups = donor_profiles(adata.X, adata.obs["donor_id"], adata.obs["diagnosis"])
    symbols = adata.var["feature_name"].to_numpy()
    ids = adata.var["feature_id"].to_numpy()
    indices = []
    for gene in LIPID + LYSOSOME:
        matches = np.flatnonzero(ids == GBA1_ENSEMBL if gene == "GBA1" else symbols == gene)
        if len(matches) != 1:
            raise ValueError(f"expected exactly one column for program marker {gene}")
        indices.append(matches.item())
    return profiles[:, indices].toarray(), groups


def program_scores(expression, train):
    """Map normalized donor expression to two scores using training donors only."""
    center = expression[train].mean(axis=0)
    scale = expression[train].std(axis=0)
    if not np.isfinite(scale).all() or (scale == 0).any():
        raise ValueError("program markers must have finite, nonzero spread in training donors")
    z = (expression - center) / scale
    return np.column_stack((z[:, :len(LIPID)].mean(axis=1), z[:, len(LIPID):].mean(axis=1)))


def correlation(values):
    if (np.ptp(values, axis=0) == 0).any():
        return None
    return float(spearmanr(values[:, 0], values[:, 1]).statistic)


def metrics(samples, observed):
    spread = np.var(observed, axis=0).sum()
    return {
        "energy_distance": energy_distance(samples, observed),
        "spearman": correlation(samples),
        "spread_ratio": float(np.sqrt(np.var(samples, axis=0).sum() / spread)) if spread > 0 else None,
    }


def fit_checkpoints(train, labels, observed, *, seed, rate, checkpoints, keep_samples=False):
    # Scope the CPU seed for the module and prior constructors. Sampling uses
    # the harness's independent evaluation generator.
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        model = ConditionalPGAN(train.shape[1], len(GROUPS), seed=seed, learning_rate=rate)
    trainer = model.trainer
    recipe = trainer.recipe
    regularizer = trainer.penalty.regularizer
    if recipe.reg_every != 1 or regularizer.lazy_k != 1:
        raise RuntimeError("the program comparison requires a regularizer application at every update")
    if recipe.total_steps is not None and checkpoints[-1] > recipe.total_steps:
        raise ValueError(f"checkpoint exceeds the EMPF schedule of {recipe.total_steps} updates")

    results = []

    def observe(updates):
        if updates not in checkpoints:
            return
        if regularizer.record.calls != updates:
            raise RuntimeError(f"expected {updates} regularizer calls, saw {regularizer.record.calls}")
        prior_condition = model.G.cond.clone(), model.D.cond.clone()
        try:
            samples = {
                group: model.sample(k, DRAWS, seed=seed + 100 + k)
                for k, group in enumerate(GROUPS)
            }
        finally:
            model.G.cond.copy_(prior_condition[0])
            model.D.cond.copy_(prior_condition[1])
        record = {
            "updates": updates,
            "regularizer_calls": regularizer.record.calls,
            "scores": {group: metrics(samples[group], observed[group]) for group in GROUPS},
        }
        if keep_samples:
            record["samples"] = samples
        results.append(record)

    model.fit(
        train, labels, steps=checkpoints[-1], min_condition_rows=MIN_TRAIN_DONORS,
        after_step=observe,
    )
    return {
        "seed": seed,
        "peak_rates": {
            "generator": recipe.lr,
            "critic": recipe.lr * recipe.d_lr_mult,
            "prior": recipe.lr * recipe.prior_lr_mult,
        },
        "schedule": {"kind": recipe.lr_schedule, "updates": recipe.total_steps},
        "checkpoints": results,
    }


def summarize(scored_groups):
    energy = []
    errors = []
    for scores, observed in scored_groups:
        for group in GROUPS:
            score = scores[group]
            energy.append(score["energy_distance"])
            expected = observed[group]["spearman"]
            if score["spearman"] is not None and expected is not None:
                errors.append(abs(score["spearman"] - expected))
    return {
        "mean_energy_distance": mean(energy),
        "mean_absolute_correlation_error": mean(errors) if len(errors) == len(energy) else None,
        "correlation_comparisons": len(errors),
    }


def inner_split(groups, seed):
    """Split training donors into fitting and validation donors for selection.

    Each diagnosis keeps at least MIN_TRAIN_DONORS fitting donors, so small
    groups give up fewer than a quarter of their donors to validation.
    """
    rng = np.random.default_rng(seed)
    fit, validation = [], []
    for group in GROUPS:
        rows = rng.permutation(np.flatnonzero(groups == group))
        held = min(len(rows) // 4, len(rows) - MIN_TRAIN_DONORS)
        if held < MIN_VALIDATION_DONORS:
            raise ValueError(f"insufficient {group} training donors for an inner validation split")
        validation.extend(rows[:held])
        fit.extend(rows[held:])
    return np.asarray(fit), np.asarray(validation)


def select_configuration(validation_summary):
    """Return the (rate, updates) keys with the lowest validation selection metric.

    Ties go to the lower energy distance, then to the lower rate and earlier
    checkpoint. Configurations without a defined correlation are not eligible.
    """
    candidates = [
        (summary[SELECTION_METRIC], summary["mean_energy_distance"], float(rate), int(step), rate, step)
        for rate, steps in validation_summary.items()
        for step, summary in steps.items()
        if summary[SELECTION_METRIC] is not None
    ]
    if not candidates:
        raise ValueError("no configuration has a defined validation correlation")
    best = min(candidates)
    return best[4], best[5]


def _observed_summary(observed):
    return {g: {"donors": len(x), "spearman": correlation(x)} for g, x in observed.items()}


def _distribution(values, *, interval):
    """Bootstrap mean and SD, plus a 95% percentile interval when requested."""
    values = np.asarray(values, dtype=float)
    if not len(values) or np.isnan(values).any():
        return {"bootstrap_mean": None, "bootstrap_se": None, **({"ci95": None} if interval else {})}
    result = {"bootstrap_mean": float(values.mean()), "bootstrap_se": float(values.std(ddof=1))}
    if interval:
        result["ci95"] = [float(v) for v in np.percentile(values, (2.5, 97.5))]
    return result


def donor_bootstrap(groups, splits, *, replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED):
    """Bootstrap held-out donors with fitted samples fixed.

    Each replicate draws donor multiplicities once for the whole cohort,
    stratified by diagnosis. Every split then reuses those multiplicities for
    its held-out donors, so donors shared between splits move together.
    Generated, resampled, and mean-shift draws are not refit. The intervals
    therefore describe held-out donor sampling only, not training-donor or
    fit variation.

    A resampled donor can appear more than once. Repeated rows shrink the
    within-sample distances, so energy distance and absolute correlation
    error are biased upward in the replicates, and marginal percentile
    intervals can sit above the point estimate. Each arm therefore gets a
    bootstrap mean and SE only. Paired EMPF-minus-control differences, where
    the observed side is shared, also get 95% percentile intervals.
    """
    rng = np.random.default_rng(seed)
    members = {group: np.flatnonzero(groups == group) for group in GROUPS}
    values = {arm: {metric: [] for metric in METRICS} for arm in ARMS}
    skipped = 0
    for _ in range(replicates):
        weights = np.zeros(len(groups), dtype=int)
        for rows in members.values():
            weights[rows] = rng.multinomial(len(rows), np.full(len(rows), 1 / len(rows)))
        resampled = []
        for split in splits:
            observed = {}
            for group in GROUPS:
                donors = split["held"][split["held_groups"] == group]
                counts = weights[donors]
                observed[group] = np.repeat(split["observed"][group], counts, axis=0)
            resampled.append(observed)
        if any(
            len(observed[group]) < MIN_VALIDATION_DONORS or correlation(observed[group]) is None
            for observed in resampled for group in GROUPS
        ):
            skipped += 1
            continue
        for arm in ARMS:
            summary = summarize(
                ({group: metrics(split["samples"][arm][group], observed[group]) for group in GROUPS},
                 _observed_summary(observed))
                for split, observed in zip(splits, resampled)
            )
            for metric in METRICS:
                value = summary[metric]
                values[arm][metric].append(np.nan if value is None else value)

    used = replicates - skipped
    if used < max(1, replicates // 2):
        raise RuntimeError(f"only {used} of {replicates} bootstrap replicates had defined correlations")
    intervals = {
        arm: {metric: _distribution(values[arm][metric], interval=False) for metric in METRICS}
        for arm in ARMS
    }
    differences = {
        f"empf_minus_{arm}": {
            metric: _distribution(
                np.subtract(values["empf"][metric], values[arm][metric]), interval=True,
            ) for metric in METRICS
        } for arm in ARMS if arm != "empf"
    }
    return intervals, differences, {"replicates": replicates, "used": used, "seed": seed}


def split_half_noise(observed_by_split, *, repeats=NOISE_SPLITS, seed=BOOTSTRAP_SEED):
    """Compare random halves of the same held-out donors with each other.

    This is an observed-versus-observed reference: both sides hold about half
    of a split's held-out donors, so it is noisier than a 256-draw comparison
    with all held-out donors and should be read as a scale, not a threshold.
    """
    rng = np.random.default_rng(seed)
    energy, errors = [], []
    for observed in observed_by_split:
        for group in GROUPS:
            values = observed[group]
            for _ in range(repeats):
                order = rng.permutation(len(values))
                first, second = values[order[:len(values) // 2]], values[order[len(values) // 2:]]
                energy.append(energy_distance(first, second))
                a, b = correlation(first), correlation(second)
                if a is not None and b is not None:
                    errors.append(abs(a - b))
    return {
        "mean_energy_distance": mean(energy),
        "mean_absolute_correlation_error": mean(errors) if errors else None,
        "correlation_comparisons": len(errors),
        "energy_comparisons": len(energy),
        "repeats_per_group": repeats,
    }


def select_on_inner_split(expression, groups, train, *, seed, rates, checkpoints):
    """Fit the grid on inner training donors and score it on inner validation donors."""
    fit_rows, validation_rows = inner_split(groups[train], seed + 1000)
    inner_train, validation = train[fit_rows], train[validation_rows]
    scores = program_scores(expression, inner_train)
    observed = {g: scores[validation][groups[validation] == g] for g in GROUPS}
    labels = np.array([GROUPS.index(group) for group in groups[inner_train]])
    observed_summary = _observed_summary(observed)
    grid = {}
    for rate in rates:
        fit = fit_checkpoints(
            scores[inner_train], labels, observed, seed=seed, rate=rate, checkpoints=checkpoints,
        )
        grid[str(rate)] = {
            str(row["updates"]): summarize([(row["scores"], observed_summary)])
            for row in fit["checkpoints"]
        }
    rate, step = select_configuration(grid)
    return {
        "fit_donors": {g: int(np.sum(groups[inner_train] == g)) for g in GROUPS},
        "validation": observed_summary,
        "validation_summary": grid,
        "selected": {"learning_rate": float(rate), "updates": int(step)},
    }


def run(adata, *, seeds=SEEDS, rates=LEARNING_RATES, checkpoints=CHECKPOINTS,
        bootstrap_replicates=BOOTSTRAP_REPLICATES):
    expression, groups = marker_expression(adata)
    comparisons = []
    held_out = []
    for seed in seeds:
        train, held = split_donors(groups, seed)
        selection = select_on_inner_split(
            expression, groups, train, seed=seed, rates=rates, checkpoints=checkpoints,
        )
        rate = selection["selected"]["learning_rate"]
        updates = selection["selected"]["updates"]

        scores = program_scores(expression, train)
        observed = {g: scores[held][groups[held] == g] for g in GROUPS}
        labels = np.array([GROUPS.index(group) for group in groups[train]])
        rng = np.random.default_rng(seed + 31)

        def draw(data):
            return data[rng.integers(len(data), size=DRAWS)]

        train_group = {group: scores[train][labels == k] for k, group in enumerate(GROUPS)}
        samples = {"empirical": {group: draw(train_group[group]) for group in GROUPS}}
        samples["mean_shift"] = {
            group: draw(mean_shift_prediction(train_group["control"], train_group[group]))
            for group in GROUPS
        }
        fits = {}
        for arm, condition in (("empf", labels), ("shuffled", rng.permutation(labels))):
            fit = fit_checkpoints(
                scores[train], condition, observed, seed=seed, rate=rate,
                checkpoints=(updates,), keep_samples=True,
            )
            samples[arm] = fit["checkpoints"][-1].pop("samples")
            fits[arm] = fit

        held_out.append({
            "held": held,
            "held_groups": groups[held],
            "observed": observed,
            "samples": samples,
        })
        comparisons.append({
            "seed": seed,
            "selection": selection,
            "observed": _observed_summary(observed),
            "arms": {
                arm: {group: metrics(samples[arm][group], observed[group]) for group in GROUPS}
                for arm in ARMS
            },
            "fits": fits,
        })

    intervals, differences, bootstrap = donor_bootstrap(
        groups, held_out, replicates=bootstrap_replicates,
    )
    summary = {}
    for arm in ARMS:
        summary[arm] = summarize((row["arms"][arm], row["observed"]) for row in comparisons)
        summary[arm]["bootstrap"] = intervals[arm]
    paired = {}
    for arm in ARMS[1:]:
        paired[f"empf_minus_{arm}"] = {}
        for metric in METRICS:
            model, control = summary["empf"][metric], summary[arm][metric]
            paired[f"empf_minus_{arm}"][metric] = {
                "estimate": None if model is None or control is None else model - control,
                **differences[f"empf_minus_{arm}"][metric],
            }
    return {
        "input_shape": list(adata.shape),
        "particlegan_version": version("particlegan"),
        "programs": {"lipid": list(LIPID), "lysosome": list(LYSOSOME)},
        "draws_per_group": DRAWS,
        "selected_on": "inner_train_split",
        "selection_metric": SELECTION_METRIC,
        "grid": {"learning_rates": [float(r) for r in rates], "checkpoints": [int(c) for c in checkpoints]},
        "summary": summary,
        "paired_differences": paired,
        "bootstrap": {
            **bootstrap,
            "level": 0.95,
            "unit": "held-out donor, stratified by diagnosis, multiplicities shared across splits",
            "conditional_on": "fitted samples, training donors, and splits",
        },
        "noise_floor": {
            "kind": "observed_split_half",
            **split_half_noise([split["observed"] for split in held_out]),
        },
        "splits": comparisons,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="public PsychAD microglia H5AD extract")
    parser.add_argument("--learning-rates", type=float, nargs="+", default=LEARNING_RATES)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=CHECKPOINTS)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    args = parser.parse_args()
    if any(not math.isfinite(rate) or rate <= 0 for rate in args.learning_rates):
        parser.error("learning rates must be finite and positive")
    if list(args.checkpoints) != sorted(set(args.checkpoints)) or args.checkpoints[0] < 1:
        parser.error("checkpoints must be positive and strictly increasing")
    if min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be distinct nonnegative integers")
    if len(set(args.learning_rates)) != len(args.learning_rates):
        parser.error("learning rates must be distinct")
    if args.bootstrap_replicates < 100:
        parser.error("use at least 100 bootstrap replicates")

    data = load_psychad(args.input)
    result = run(
        data, seeds=args.seeds, rates=args.learning_rates, checkpoints=args.checkpoints,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    with args.input.open("rb") as source:
        result["input_sha256"] = hashlib.file_digest(source, "sha256").hexdigest()
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
