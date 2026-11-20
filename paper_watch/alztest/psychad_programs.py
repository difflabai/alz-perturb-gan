"""Compare training budgets and rates on two public PsychAD gene program scores."""
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
from .harness import ConditionalPGAN, energy_distance
from .ll_panel import LIPID, LYSO
from .psychad import DRAWS, GROUPS, MIN_TRAIN_DONORS, donor_profiles, split_donors

CHECKPOINTS = (300, 1200, 2400)
LEARNING_RATES = (0.0025, 0.005, 0.012)
SEEDS = (0, 1, 2)
# The pinned CELLxGENE extract has no CTSD column. Its GBA1 has this Ensembl ID.
LYSOSOME = tuple("GBA1" if gene == "GBA" else gene for gene in LYSO if gene != "CTSD")
GBA1_ENSEMBL = "ENSG00000177628"


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


def fit_checkpoints(train, labels, observed, *, seed, rate, checkpoints):
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
            scores = {
                group: metrics(model.sample(k, DRAWS, seed=seed + 100 + k), observed[group])
                for k, group in enumerate(GROUPS)
            }
        finally:
            model.G.cond.copy_(prior_condition[0])
            model.D.cond.copy_(prior_condition[1])
        results.append({
            "updates": updates,
            "regularizer_calls": regularizer.record.calls,
            "scores": scores,
        })

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


def run(adata, *, seeds=SEEDS, rates=LEARNING_RATES, checkpoints=CHECKPOINTS):
    expression, groups = marker_expression(adata)
    comparisons = []
    for seed in seeds:
        train, held = split_donors(groups, seed)
        scores = program_scores(expression, train)
        observed = {g: scores[held][groups[held] == g] for g in GROUPS}
        labels = np.array([GROUPS.index(group) for group in groups[train]])
        rng = np.random.default_rng(seed + 31)
        empirical = {}
        for k, group in enumerate(GROUPS):
            source = scores[train][labels == k]
            empirical[group] = metrics(source[rng.integers(len(source), size=DRAWS)], observed[group])

        comparisons.append({
            "seed": seed,
            "observed": {g: {"donors": len(x), "spearman": correlation(x)} for g, x in observed.items()},
            "empirical": empirical,
            "fits": {
                str(rate): fit_checkpoints(
                    scores[train], labels, observed, seed=seed, rate=rate, checkpoints=checkpoints,
                ) for rate in rates
            },
        })

    summary = {
        str(rate): {
            str(step): summarize(
                (row["fits"][str(rate)]["checkpoints"][i]["scores"], row["observed"])
                for row in comparisons
            ) for i, step in enumerate(checkpoints)
        } for rate in rates
    }
    return {
        "input_shape": list(adata.shape),
        "particlegan_version": version("particlegan"),
        "programs": {"lipid": list(LIPID), "lysosome": list(LYSOSOME)},
        "draws_per_group": DRAWS,
        "summary": summary,
        "empirical_reference": summarize((r["empirical"], r["observed"]) for r in comparisons),
        "splits": comparisons,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="public PsychAD microglia H5AD extract")
    parser.add_argument("--learning-rates", type=float, nargs="+", default=LEARNING_RATES)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=CHECKPOINTS)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = parser.parse_args()
    if any(not math.isfinite(rate) or rate <= 0 for rate in args.learning_rates):
        parser.error("learning rates must be finite and positive")
    if list(args.checkpoints) != sorted(set(args.checkpoints)) or args.checkpoints[0] < 1:
        parser.error("checkpoints must be positive and strictly increasing")
    if min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be distinct nonnegative integers")
    if len(set(args.learning_rates)) != len(args.learning_rates):
        parser.error("learning rates must be distinct")

    data = load_psychad(args.input)
    result = run(data, seeds=args.seeds, rates=args.learning_rates, checkpoints=args.checkpoints)
    with args.input.open("rb") as source:
        result["input_sha256"] = hashlib.file_digest(source, "sha256").hexdigest()
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
