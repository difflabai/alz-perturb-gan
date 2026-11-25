"""Select a NeuroKO Unified v11 learning rate without using outer holdout expression."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from importlib.metadata import version
import os
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np
import torch

from .harness import ConditionalPGAN, energy_distance, shift_cosine
from .neuroko import LAYOUT, load_comparison, open_screen, project_fold

RELEASE = "0.11.1"
RECIPE = "unified_v11"
PUBLISHED_RATE = 0.014
RATES = (0.005, PUBLISHED_RATE, 0.028, 0.056, 0.112)
GROUPS = {
    "iNeurons": ("Screen1_0", "Screen1_1", "Screen1_3", "Screen1_4"),
    "iAstrocytes": ("SCREEN1_1", "SCREEN1_2"),
}
DRAWS = 256


def split_vectors(panel, available, conditions):
    """Hold out whole guide vectors with counts nearest one third per target."""
    validate = np.zeros(len(available), dtype=bool)
    assignments = {}
    for label, target in enumerate(conditions):
        rows = available & (panel.labels == label)
        values, counts = np.unique(panel.vectors[rows], return_counts=True)
        plans = {0: ()}
        for vector, count in zip(values, counts):
            for used, plan in tuple(plans.items()):
                plans.setdefault(used + int(count), (*plan, str(vector)))
        total = int(rows.sum())
        eligible = [n for n in plans if 16 <= n <= total - 32]
        if not eligible:
            raise ValueError(f"{target} needs at least 16 validation and 32 fitting cells with disjoint vectors")
        n = min(eligible, key=lambda size: (abs(size - total / 3), plans[size]))
        assignments[target] = list(plans[n])
        validate |= rows & np.isin(panel.vectors, plans[n])
    return validate, assignments


def validation_split(panel, group, conditions):
    groups = sorted(set(panel.groups))
    available = panel.groups != group
    if len(groups) == 2:
        validate, assignment = split_vectors(panel, available, conditions)
        description = {"validation_group": str(panel.groups[validate][0]), "validation_vectors": assignment}
    else:
        validation_group = groups[(groups.index(group) + 1) % len(groups)]
        validate = panel.groups == validation_group
        description = {"validation_group": validation_group}
    training = available & ~validate
    return training, validate, description


def model_samples(x, labels, n_conditions, rate, steps, seed, device, recipe_name=RECIPE):
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        model = ConditionalPGAN(x.shape[1], n_conditions, seed=seed, learning_rate=rate,
                                recipe_name=recipe_name, device=device)
    for module in (model.G, model.D, model.prior):
        if {parameter.device for parameter in module.parameters()} != {device}:
            raise RuntimeError("the generator, critic and prior must all run on the selected CUDA device")
    torch.cuda.synchronize(device)
    started = perf_counter()
    model.fit(x, labels, steps=steps)
    torch.cuda.synchronize(device)
    if model.trainer.completed_steps != steps:
        raise RuntimeError(f"completed {model.trainer.completed_steps} updates, expected {steps}")
    duration = perf_counter() - started
    samples = [model.sample(label, DRAWS, seed=seed + 100 + label) for label in range(n_conditions)]
    if not all(np.isfinite(rows).all() for rows in samples):
        raise FloatingPointError("non-finite generated samples")
    return samples, duration


def score(samples, observed, conditions):
    results = {}
    for k, target in enumerate(conditions[1:], start=1):
        observed_variance = observed[k].var(axis=0).sum()
        results[target] = {
            "energy_distance": energy_distance(samples[k], observed[k]),
            "shift_cosine": shift_cosine(samples[k], samples[0], observed[k], observed[0]),
            "spread_ratio": float(np.sqrt(samples[k].var(axis=0).sum() / observed_variance))
            if observed_variance > 0 else None,
        }
    return results


def compare_rates(panel, conditions, fit_rows, evaluation_rows, rates, steps, seed, device, recipe_name=RECIPE):
    xfit, xobserved = project_fold(panel.x[fit_rows], panel.x[evaluation_rows])
    yfit, yobserved = panel.labels[fit_rows], panel.labels[evaluation_rows]
    n_conditions = len(conditions)
    fitted = [xfit[yfit == k] for k in range(n_conditions)]
    observed = [xobserved[yobserved == k] for k in range(n_conditions)]
    if min(map(len, fitted)) < 32 or min(map(len, observed)) < 16:
        raise ValueError("a condition has insufficient cells in the requested split")
    rng = np.random.default_rng(seed)
    empirical = [rows[rng.choice(len(rows), DRAWS, replace=True)] for rows in fitted]
    records = []
    for rate in rates:
        print(f"  lr {rate:g}, seed {seed}, {steps} CUDA updates", flush=True)
        try:
            simulated, seconds = model_samples(xfit, yfit, n_conditions, rate, steps, seed, device, recipe_name)
        except FloatingPointError as error:
            records.append({"base_rate": rate, "error": str(error)})
            continue
        scores = score(simulated, observed, conditions)
        records.append({"base_rate": rate, "fit_seconds": seconds, "target_scores": scores,
                        "mean_energy_distance": mean(row["energy_distance"] for row in scores.values())})
        print(f"    mean energy distance {records[-1]['mean_energy_distance']:.6f} in {seconds:.1f}s", flush=True)
    return {
        "fit_cells": dict(zip(conditions, map(len, fitted))),
        "scored_cells": dict(zip(conditions, map(len, observed))),
        "empirical_scores": score(empirical, observed, conditions),
        "rates": records,
    }


def selected_rate(records):
    finite = [row for row in records if "mean_energy_distance" in row]
    if not finite:
        raise FloatingPointError("no finite validation fit")
    return min(finite, key=lambda row: (row["mean_energy_distance"], row["base_rate"]))["base_rate"]


def finer_rates(records):
    coarse = {row["base_rate"]: row for row in records if row["base_rate"] in RATES}
    if set(coarse) != set(RATES):
        raise ValueError("refinement requires the complete coarse grid")
    index = RATES.index(selected_rate(coarse.values()))
    center = RATES[index]
    lower = RATES[index - 1] if index > 0 else center / 4
    upper = RATES[index + 1] if index + 1 < len(RATES) else center * 4
    return tuple(float(f"{(center * neighbor) ** 0.5:.3g}") for neighbor in (lower, upper))


def refined_selection(tuning, group):
    """Return the validation choice only after the coarse grid and its refinement both finished."""
    if tuning.get("stage") != "validation" or tuning.get("held_out_group") != group:
        raise ValueError(f"{group} needs its own validation receipt before the held-out fit")
    rows = tuning["rates"]
    try:
        bracket = finer_rates(rows)
    except ValueError as error:
        raise ValueError(f"{group}: {error}; run the validation stage first") from None
    missing = sorted((set(RATES) | set(bracket)) - {row["base_rate"] for row in rows})
    if missing:
        raise ValueError(f"run the refine stage for {group} before the held-out fit; missing rates {missing}")
    selected = selected_rate(rows)
    if tuning["selected_rate"] != selected:
        raise ValueError(f"the recorded selection for {group} differs from its rate scores")
    return selected


def experiment_config(args, conditions):
    config = {"particlegan": version("particlegan"), "recipe": args.recipe, "steps": args.steps,
              "targets": conditions[1:], "data": str(args.data.resolve())}
    if args.stage == "replay":
        files = {name: (args.data / f"{name}.h5ad").stat() for name in LAYOUT}
        config["data_files"] = {name: [entry.st_size, entry.st_mtime_ns] for name, entry in files.items()}
    return config


def evaluate_group(args, conditions, panel, group, device):
    test = panel.groups == group
    validation_file = args.output / f"{group}-validation.json"
    config = experiment_config(args, conditions)
    if args.stage == "replay":
        selected, = args.rates
        split = {}
        scores = compare_rates(panel, conditions, ~test, test, args.rates, args.steps, args.seed,
                               device, args.recipe)
    elif args.stage != "heldout":
        training, validation, split = validation_split(panel, group, conditions)
        prior = json.loads(validation_file.read_text()) if validation_file.exists() else None
        if args.stage == "refine" and prior is None:
            raise ValueError(f"missing coarse validation for {group}")
        if prior is not None and (prior["config"] != config or prior["seed"] != args.seed
                                  or prior["split"] != split):
            raise ValueError(f"cached validation protocol differs for {group}")
        rates = finer_rates(prior["rates"]) if args.stage == "refine" else args.rates
        if prior is not None:
            finished = {row["base_rate"] for row in prior["rates"]}
            rates = [rate for rate in rates if rate not in finished]
            if not rates:
                return prior
        scores = compare_rates(panel, conditions, training, validation, rates, args.steps, args.seed, device)
        if prior is not None:
            for key in ("fit_cells", "scored_cells", "empirical_scores"):
                if scores[key] != prior[key]:
                    raise ValueError(f"cached validation inputs differ for {group}: {key}")
            scores["rates"] = sorted(prior["rates"] + scores["rates"], key=lambda row: row["base_rate"])
        selected = selected_rate(scores["rates"])
    else:
        tuning = json.loads(validation_file.read_text())
        if tuning["config"] != config or tuning["held_out_group"] != group:
            raise ValueError(f"validation receipt does not match {group}")
        selected = refined_selection(tuning, group)
        split = {"validation_seed": tuning["seed"]}
        rates = tuple(dict.fromkeys((selected, PUBLISHED_RATE)))
        scores = compare_rates(panel, conditions, ~test, test, rates, args.steps, args.seed, device)
    stage = "validation" if args.stage == "refine" else args.stage
    return {"stage": stage,
            "held_out_group": group, "config": config, "seed": args.seed,
            "device": str(device), "gpu": torch.cuda.get_device_name(device), "split": split,
            "torch_version": torch.__version__, "selected_rate": selected, **scores}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("validation", "refine", "heldout", "replay"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group", choices=sum(GROUPS.values(), ()), action="append", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--recipe", choices=(RECIPE, "empf"), default=RECIPE,
                        help="public recipe for a fixed-rate replay")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rates", type=float, nargs="+", default=RATES)
    args = parser.parse_args()
    if args.steps < 1 or len(set(args.rates)) != len(args.rates) or any(
        not np.isfinite(rate) or rate <= 0 for rate in args.rates
    ):
        parser.error("steps and distinct finite learning rates must be positive")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("this experiment requires CUDA")
    if args.stage == "replay":
        if len(args.rates) != 1:
            parser.error("replay requires exactly one explicit rate via --rates")
    elif version("particlegan") != RELEASE or args.recipe != RECIPE:
        parser.error(f"rate selection and confirmation require {RECIPE} on ParticleGAN {RELEASE}")
    if device.index is not None:
        torch.cuda.set_device(device)
    device = torch.device("cuda", torch.cuda.current_device())
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        screens = [stack.enter_context(open_screen(args.data, name)) for name in LAYOUT]
        conditions, panels = load_comparison(screens, 100, 160, 6)
        for group in args.group:
            panel = next(panel for name, panel in panels.items() if group in GROUPS[name])
            if args.stage == "replay":
                filename = f"{group}-replay-{args.recipe}-{version('particlegan')}-seed{args.seed}.json"
            elif args.stage == "heldout":
                filename = f"{group}-heldout-seed{args.seed}.json"
            else:
                filename = f"{group}-validation.json"
            path = args.output / filename
            if args.stage == "replay" and path.is_file():
                completed = json.loads(path.read_text())
                if (completed["config"] == experiment_config(args, conditions)
                        and completed["selected_rate"] == args.rates[0]
                        and completed["seed"] == args.seed and completed["held_out_group"] == group
                        and completed["torch_version"] == torch.__version__
                        and completed["gpu"] == torch.cuda.get_device_name(device)
                        and "target_scores" in completed["rates"][0]):
                    print(f"reused {path}", flush=True)
                    continue
            print(f"{args.stage} {group}, {torch.cuda.get_device_name(device)}", flush=True)
            result = evaluate_group(args, conditions, panel, group, device)
            temporary = path.with_suffix(f".{os.getpid()}.tmp")
            temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
            temporary.replace(path)
            print(f"  selected {result['selected_rate']:g}, wrote {path}", flush=True)


if __name__ == "__main__":
    main()
