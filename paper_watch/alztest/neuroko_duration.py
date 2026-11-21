"""Measure held-out NeuroKO scores and synchronized CUDA timing through one fit."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np
import torch

from .harness import ConditionalPGAN
from .neuroko import LAYOUT, load_comparison, open_screen, project_fold
from .neuroko_lr import DRAWS, GROUPS, score


def optimizer_rates(model):
    trainer = model.trainer
    rates = {}
    for role, module, optimizer in (("generator", model.G, trainer.opt_g),
                                    ("critic", model.D, trainer.opt_d),
                                    ("particles", model.prior, trainer.opt_g)):
        parameters = {id(parameter) for parameter in module.parameters() if parameter.requires_grad}
        values = {float(group["lr"]) for group in optimizer.param_groups
                  if any(id(parameter) in parameters for parameter in group["params"])}
        if len(values) != 1:
            raise ValueError(f"expected one optimizer rate for {role}, found {values}")
        rates[role] = values.pop()
    return rates


def function_space_gains(trainer):
    if trainer.recipe.effective_critic_formulation != "unified_v11":
        return {}
    gains = {}
    for role, optimizer in (("generator", trainer.opt_g), ("critic", trainer.opt_d)):
        state = optimizer.state_dict()["unified_v11_equil"]
        gains[role] = {key: state[key] for key in ("ema_func_gain", "last_step_gain")}
    return gains


def write_receipt(path, contents):
    part = path.with_suffix(f".{os.getpid()}.tmp")
    part.write_text(json.dumps(contents, indent=2, allow_nan=False) + "\n")
    part.replace(path)


def effective_rank(samples):
    """Return the covariance participation ratio, in effective dimensions."""
    cov = np.atleast_2d(np.cov(samples, rowvar=False))
    squared_norm = np.square(cov).sum()
    return float(np.trace(cov) ** 2 / squared_norm) if squared_norm else 0.0


def run(args, panel, conditions, device):
    test = panel.groups == args.group
    xfit, xobserved = project_fold(panel.x[~test], panel.x[test])
    yfit, yobserved = panel.labels[~test], panel.labels[test]
    n_conditions = len(conditions)
    observed = [xobserved[yobserved == k] for k in range(n_conditions)]
    fitted = [xfit[yfit == k] for k in range(n_conditions)]
    observed_ranks = {target: effective_rank(rows) for target, rows in zip(conditions[1:], observed[1:])}
    rng = np.random.default_rng(args.seed)
    empirical = [rows[rng.choice(len(rows), DRAWS, replace=True)] for rows in fitted]
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(args.seed)
        model = ConditionalPGAN(xfit.shape[1], n_conditions, seed=args.seed, learning_rate=args.rate,
                                recipe_name=args.recipe, device=device,
                                use_recipe_defaults=args.use_recipe_defaults)
    if any(parameter.device != device for module in (model.G, model.D, model.prior)
           for parameter in module.parameters()):
        raise RuntimeError("all fitted parameters must reside on CUDA")
    recipe = model.trainer.recipe
    receipt = {
        "particlegan": version("particlegan"), "torch": str(torch.__version__), "recipe": args.recipe,
        "device": str(device), "gpu": torch.cuda.get_device_name(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "group": args.group, "seed": args.seed, "base_rate": recipe.lr,
        "use_recipe_defaults": args.use_recipe_defaults,
        "latent_dim": recipe.z_dim, "particles": recipe.num_particles, "batch_size": recipe.batch_size,
        "schedule": {"name": recipe.lr_schedule, "steps": recipe.total_steps,
                     "network_floor": recipe.resolved_network_lr_floor, "particle_floor": recipe.lr_floor},
        "initial_optimizer_rates": optimizer_rates(model),
        "fit_cells": dict(zip(conditions, map(len, fitted))),
        "heldout_cells": dict(zip(conditions, map(len, observed))),
        "heldout_effective_rank": observed_ranks,
        "empirical_scores": score(empirical, observed, conditions), "checkpoints": [],
    }
    checkpoints = set(args.checkpoints)
    endpoint = max(checkpoints)
    trainer = model.trainer
    if args.state_file:
        digest = sha256()
        for arr in (xfit, yfit, xobserved, yobserved):
            digest.update(np.ascontiguousarray(arr).view(np.uint8))
        receipt["projected_data_sha256"] = digest.hexdigest()
        if args.state_file.exists():
            saved = torch.load(args.state_file, weights_only=True)
            previous = saved["receipt"]
            if any(previous[k] != v for k, v in receipt.items() if k != "checkpoints"):
                raise ValueError("checkpoint and requested run configurations differ")
            state = saved["trainer"]
            cap = state.get("max_steps", recipe.total_steps)
            if cap > trainer.max_steps:
                trainer.extend_execution(cap)
            trainer.load_state_dict(state)
            receipt = previous
            scored = {row["updates"] for row in previous["checkpoints"]}
            if (trainer.completed_steps != max(scored) or endpoint < trainer.completed_steps
                    or any(k <= trainer.completed_steps and k not in scored for k in checkpoints)):
                raise ValueError("requested checkpoints are incompatible with the saved fit")
            write_receipt(args.out, receipt)
            print(f"Resuming {args.recipe} at {trainer.completed_steps} updates", flush=True)
    if endpoint > trainer.max_steps:
        trainer.extend_execution(endpoint)
    torch.cuda.synchronize(device)
    started = perf_counter()

    def observe(update):
        nonlocal started
        if update not in checkpoints:
            return
        torch.cuda.synchronize(device)
        seconds = perf_counter() - started
        generator_condition, critic_condition = model.G.cond.clone(), model.D.cond.clone()
        try:
            samples = [model.sample(k, DRAWS, seed=args.seed + 100 + k) for k in range(n_conditions)]
        finally:
            model.G.cond.copy_(generator_condition)
            model.D.cond.copy_(critic_condition)
        if not all(np.isfinite(rows).all() for rows in samples):
            raise FloatingPointError(f"non-finite samples at step {update}")
        scores = score(samples, observed, conditions)
        for target, rows in zip(conditions[1:], samples[1:]):
            scores[target]["effective_rank"] = effective_rank(rows)
        energy = mean(values["energy_distance"] for values in scores.values())
        receipt["checkpoints"].append({
            "updates": update, "segment_fit_seconds": seconds, "optimizer_rates": optimizer_rates(model),
            "function_space_gains": function_space_gains(model.trainer),
            "mean_energy_distance": energy, "scores": scores,
        })
        if args.state_file:
            part = args.state_file.with_suffix(f".{os.getpid()}.tmp")
            torch.save({"trainer": trainer.state_dict(), "receipt": receipt}, part)
            part.replace(args.state_file)
        write_receipt(args.out, receipt)
        print(f"{args.group} {args.recipe}: {update} updates, energy {energy:.6f}, {seconds:.1f}s segment", flush=True)
        torch.cuda.synchronize(device)
        started = perf_counter()

    if trainer.completed_steps < endpoint:
        model.fit(xfit, yfit, steps=endpoint - trainer.completed_steps,
                  replay_batches=trainer.completed_steps, after_step=observe)
    if model.trainer.completed_steps != max(checkpoints):
        raise RuntimeError("the requested fit did not complete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--window-cells", type=int,
                        help="use the first N cells per release group from the original H5ADs")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--state-file", type=Path, help="save and resume a local trainer checkpoint")
    parser.add_argument("--group", choices=sum(GROUPS.values(), ()), required=True)
    parser.add_argument("--recipe", choices=("empf", "unified_v11"), required=True)
    configuration = parser.add_mutually_exclusive_group(required=True)
    configuration.add_argument("--rate", type=float, help="base rate for the small NeuroKO configuration")
    configuration.add_argument("--use-recipe-defaults", action="store_true",
                               help="use the published recipe without size, rate or prior overrides")
    parser.add_argument("--checkpoints", nargs="+", type=int, default=(300, 600, 1200))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.rate is not None and (not np.isfinite(args.rate) or args.rate <= 0):
        parser.error("the base rate must be positive and finite")
    if min(args.checkpoints) <= 0:
        parser.error("the checkpoints must be positive")
    if args.window_cells is not None and args.window_cells < 1:
        parser.error("window-cells must be positive")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("this experiment requires CUDA")
    if device.index is not None:
        torch.cuda.set_device(device)
    device = torch.device("cuda", torch.cuda.current_device())
    torch.set_num_threads(2)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.state_file:
        if args.state_file.resolve() == args.out.resolve():
            parser.error("choose separate paths for scores and trainer state")
        args.state_file.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        screens = [stack.enter_context(open_screen(args.data, name, args.window_cells)) for name in LAYOUT]
        conditions, panels = load_comparison(screens, 100, 160, 6)
        panel = next(panel for name, panel in panels.items() if args.group in GROUPS[name])
        run(args, panel, conditions, device)


if __name__ == "__main__":
    main()
