"""Compare fixed 1,200-update NeuroKO configurations on CUDA.

Hold out a training group for validation, then refit a fixed configuration for Screen1_0.
Each configuration after --trial is a JSON object with name, base rate (lr),
and optional TrainingSettings fields. Each completed fit writes one receipt.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np
import torch

from .harness import ConditionalPGAN, TrainingSettings, energy_distance
from .neuroko import LAYOUT, load_comparison, open_screen, project_fold
from .neuroko_duration import effective_rank, function_space_gains, optimizer_rates, write_receipt
from .neuroko_lr import DRAWS, score


def verify_fixed_schedule(recipe, updates):
    from particlegan import learning_rate_scales

    for step in range(updates + 1):
        scales = learning_rate_scales(step, recipe)
        if tuple(scales) != (1.0, 1.0):
            raise ValueError(f"{recipe.name} changes network or particle LR at update {step}: {scales}")


def optimizer_momentum(trainer):
    """Read the state that each released optimizer uses to calculate its step."""
    result = {}
    for role, optimizer in (("generator_and_particles", trainer.opt_g), ("critic", trainer.opt_d)):
        state = {"class": type(optimizer).__name__}
        if trainer.recipe.optimizer_family == "dualnorm":
            state["momentum"] = optimizer.momentum
        else:
            state["group_betas"] = [list(group["betas"]) for group in optimizer.param_groups]
        result[role] = state
    return result


def diagnose_energy(samples, observed, training, labels, conditions):
    """Measure centroid and spread contributions after the primary evaluation."""
    targets = {}
    for k, target in enumerate(conditions[1:], 1):
        generated, real, fitted = samples[k], observed[k], training[labels == k]
        generated_mean, real_mean, fitted_mean = (rows.mean(0) for rows in (generated, real, fitted))
        centered = generated - generated_mean
        variance = centered.var(0).sum()
        rescaled = centered * np.sqrt(real.var(0).sum() / variance) if variance > 0 else centered
        targets[target] = {
            "generated_to_observed_centroid": float(np.linalg.norm(generated_mean - real_mean)),
            "generated_to_training_centroid": float(np.linalg.norm(generated_mean - fitted_mean)),
            "training_to_observed_centroid": float(np.linalg.norm(fitted_mean - real_mean)),
            "energy_at_observed_centroid": energy_distance(centered + real_mean, real),
            "energy_at_observed_centroid_and_spread": energy_distance(rescaled + real_mean, real),
            "energy_to_training_cells": energy_distance(generated, fitted),
        }
    return targets


def draw_metrics(samples, scored_targets, *, diversity=True):
    metrics = {"mean_energy_distance": mean(row["energy_distance"] for row in scored_targets.values())}
    if diversity:
        metrics["spread_ratio"] = mean(row["spread_ratio"] for row in scored_targets.values())
        metrics["effective_rank"] = mean(effective_rank(rows) for rows in samples[1:])
    return metrics


def score_draw(model, seed, observed, conditions):
    generated = [model.sample(i, DRAWS, seed=seed + i) for i in range(len(conditions))]
    return draw_metrics(generated, score(generated, observed, conditions))


def score_alternate_noise(model, training_sigma, alternate_sigma, seeds, observed, conditions):
    from particlegan.priors import NormalNoiseManipulator

    noises = [m for m in model.prior.modules() if isinstance(m, NormalNoiseManipulator)]
    if len(noises) != 1 or noises[0].sigma != training_sigma:
        raise ValueError("Expected one native Gaussian noise manipulator with the training sigma")
    noise = noises[0]
    try:
        noise.sigma = alternate_sigma
        return {str(seed): score_draw(model, seed, observed, conditions) for seed in seeds}
    finally:
        noise.sigma = training_sigma


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("validation", "heldout"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-cells", type=int,
                        help="use the first N cells of each group from the public files")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--recipe", choices=("unified_v12", "unified_v11", "empf"), default="unified_v11")
    parser.add_argument("--trial", action="append", type=json.loads, required=True,
                        help='JSON object, e.g. {"name":"v11","lr":0.028,"batch_size":256}')
    parser.add_argument("--require-constant-lr", action="store_true",
                        help="check network and particle learning rates on every update (automatic for v12)")
    parser.add_argument("--trace-every", type=int, help="also score validation or holdout at this update interval")
    parser.add_argument("--evaluation-seeds", type=int, nargs="+", metavar="BASE",
                        help="also evaluate each fitted model with these sampling seeds plus the condition index")
    parser.add_argument("--score-noise", type=float, metavar="SIGMA",
                        help="also score the fitted generator at this Gaussian latent noise using the same draws")
    parser.add_argument("--deterministic", action="store_true",
                        help="seed PyTorch and require deterministic CUDA kernels for repeatability checks")
    parser.add_argument("--diagnose-energy", action="store_true",
                        help="also score the final samples after matching their observed centroid and spread")
    parser.add_argument("--release", choices=("0.10.0", "0.11.1", "0.12.0"),
                        help="pin the installed wheel for the requested recipe")
    parser.add_argument("--validation-group", choices=("Screen1_1", "Screen1_3", "Screen1_4"),
                        default="Screen1_1")
    parser.add_argument("--self-plan", choices=("observe", "exclude-same-id", "opposite-halves", "discount-same-id"),
                        help="trace the v11 solver or run an explicit self-pairing counterfactual")
    args = parser.parse_args()
    args.require_constant_lr = args.require_constant_lr or args.recipe == "unified_v12"
    if args.self_plan and args.recipe != "unified_v11":
        parser.error("the self-plan diagnostic requires unified_v11")
    if args.trace_every is not None and not 0 < args.trace_every <= 1200:
        parser.error("--trace-every must be between 1 and 1200")
    if args.evaluation_seeds and len(set(args.evaluation_seeds)) != len(args.evaluation_seeds):
        parser.error("evaluation seed bases must be distinct")
    if args.score_noise is not None and (not np.isfinite(args.score_noise) or args.score_noise <= 0):
        parser.error("--score-noise must be positive and finite to preserve the Gaussian noise draws")
    release = version("particlegan")
    supported = {"unified_v12": ("0.12.0",), "unified_v11": ("0.11.1", "0.12.0"),
                 "empf": ("0.10.0", "0.11.1", "0.12.0")}
    expected = args.release or supported[args.recipe][0]
    if release != expected or release not in supported[args.recipe]:
        parser.error(f"unexpected ParticleGAN release {release} for {args.recipe}")
    if args.window_cells is not None and args.window_cells < 1:
        parser.error("--window-cells must be positive")
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        parser.error("the 1,200-update benchmark requires CUDA")
    device = torch.device("cuda", torch.cuda.current_device())
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        screens = [stack.enter_context(open_screen(args.data, name, args.window_cells)) for name in LAYOUT]
        conditions, panels = load_comparison(screens, 100, 160, 6)
    panel = panels["iNeurons"]
    evaluation = panel.groups == "Screen1_0"
    fit = ~evaluation
    if args.stage == "validation":
        evaluation = panel.groups == args.validation_group
        fit &= ~evaluation
    x, y = project_fold(panel.x[fit], panel.x[evaluation])
    labels = panel.labels[fit]
    observed = [y[panel.labels[evaluation] == idx] for idx in range(len(conditions))]
    if min(np.bincount(labels, minlength=len(conditions))) < 32 or min(map(len, observed)) < 16:
        raise ValueError("a condition has insufficient fitting or evaluation cells")
    digest = sha256()
    for arr in (x, labels, y, panel.labels[evaluation]):
        digest.update(np.ascontiguousarray(arr).view(np.uint8))
    for trial in args.trial:
        if not isinstance(trial, dict) or not {"name", "lr"} <= trial.keys():
            parser.error("each trial needs a name and a numeric lr")
        values = trial.copy()
        name, lr = values.pop("name"), values.pop("lr")
        if type(lr) not in (int, float) or not np.isfinite(lr) or lr <= 0:
            parser.error("trial learning rate must be positive and finite")
        if not isinstance(name, str) or not name or not all(c.isalnum() or c in "_.-" for c in name):
            parser.error("trial name must use letters, numbers, periods, hyphens or underscores")
        try:
            settings = TrainingSettings(**values)
        except (TypeError, ValueError) as exc:
            parser.error(f"invalid settings for {name}: {exc}")
        path = args.output / f"{args.stage}-{name}-s{args.seed}.json"
        config = {"name": name, "lr": lr, "settings": asdict(settings), "recipe": args.recipe,
                  "particlegan": release, "torch": str(torch.__version__), "seed": args.seed,
                  "stage": args.stage, "projected_data_sha256": digest.hexdigest(), "updates": 1200}
        if args.stage == "validation" and args.validation_group != "Screen1_1":
            path = path.with_stem(f"{path.stem}-{args.validation_group}")
        if args.deterministic:
            config["deterministic"] = True
            config["cublas_workspace_config"] = os.environ["CUBLAS_WORKSPACE_CONFIG"]
            path = path.with_stem(f"{path.stem}-deterministic")
        if args.diagnose_energy:
            config["diagnose_energy"] = True
            path = path.with_stem(f"{path.stem}-diagnosis")
        if args.trace_every:
            config["trace_every"] = args.trace_every
            path = path.with_stem(f"{path.stem}-trace{args.trace_every}")
        if args.evaluation_seeds:
            config["evaluation_seeds"] = args.evaluation_seeds
            path = path.with_stem(f"{path.stem}-resampled")
        if args.score_noise is not None:
            config["score_noise"] = args.score_noise
            path = path.with_stem(f"{path.stem}-noise{args.score_noise}")
        if args.self_plan:
            config["diagnostic_self_plan"] = args.self_plan
            path = path.with_stem(f"{path.stem}-{args.self_plan}")
        if path.exists():
            previous = json.loads(path.read_text())
            previous["settings"] = asdict(TrainingSettings(**previous["settings"]))
            if any(previous[k] != v for k, v in config.items()):
                raise ValueError(f"incompatible cached run at {path}")
            if not args.require_constant_lr:
                continue
            from particlegan import get_recipe

            verify_fixed_schedule(get_recipe(args.recipe, lr=lr, **settings.recipe_overrides()), config["updates"])
            rates = previous["initial_rates"]
            trace = previous.get("rate_trace", [])
            if previous["final_rates"] != rates or any(row[k] != value for row in trace for k, value in rates.items()):
                raise ValueError(f"cached optimizer rates changed at {path}")
            if previous.get("constant_lr_verified_updates") == config["updates"]:
                continue
            # Replay older fits that only recorded the rates every 300 updates.
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(args.seed)
            model = ConditionalPGAN(x.shape[1], len(conditions), seed=args.seed, device=device,
                                    recipe_name=args.recipe, learning_rate=lr, settings=settings)
        if any(p.device != device for m in (model.G, model.D, model.prior) for p in m.parameters()):
            raise RuntimeError("all trainable modules must reside on the selected GPU")
        if args.self_plan:
            from .neuroko_radial import trace_training_plans

            plan_trace = trace_training_plans(model.trainer, args.self_plan)
        if settings.stratified_prior:
            from particlegan.priors import StratifiedSamplingManipulator

            if not any(isinstance(module, StratifiedSamplingManipulator) for module in model.prior.modules()):
                raise RuntimeError("the requested stratified prior must be active")
            config["prior_manipulators"] = model.prior.manipulator_spec()
        recipe = model.trainer.recipe
        formulation = "unified_v11" if args.recipe == "unified_v12" else args.recipe
        optimizer = {"unified_v12": "unified_v12_equil", "unified_v11": "unified_v11_equil", "empf": "dualnorm"}
        if (recipe.name != args.recipe or recipe.effective_critic_formulation != formulation
                or recipe.prior_regularizer != formulation or recipe.optimizer_family != optimizer[args.recipe]):
            raise RuntimeError("the requested recipe, critic formulation, prior and optimizer must be active")
        if args.recipe == "unified_v12" and any(
            getattr(opt, "recipe_optimizer_family", None) != "unified_v12_equil"
            for opt in (model.trainer.opt_g, model.trainer.opt_d)
        ):
            raise RuntimeError("both optimizers must use the released v12 implementation")
        if settings.reg_kappa is not None:
            threshold = model.trainer.penalty.regularizer.kappa
            if threshold != settings.reg_kappa:
                raise RuntimeError(f"critic penalty threshold is {threshold}, expected {settings.reg_kappa}")
            config["critic_penalty_threshold"] = threshold
        if args.require_constant_lr:
            verify_fixed_schedule(recipe, config["updates"])
        rates = optimizer_rates(model)
        initialization = sha256()
        for module in (model.G, model.D, model.prior):
            for parameter in module.parameters():
                initialization.update(parameter.detach().cpu().contiguous().numpy().tobytes())
        batch_digest = sha256()

        def record_batch(label, indices):
            batch_digest.update(label.to_bytes(4, "little"))
            batch_digest.update(indices.astype("<i8", copy=False).tobytes())

        print(f"Training {name} {args.stage} seed {args.seed} on {device}, rates {rates}", flush=True)
        torch.cuda.synchronize(device)
        start = perf_counter()
        trace = []
        rate_trace = []
        gain_trace = []
        scoring_seconds = 0.0

        def progress(step):
            nonlocal scoring_seconds
            if args.trace_every and step < 1200 and step % args.trace_every == 0:
                torch.cuda.synchronize(device)
                tick = perf_counter()
                condition = model.G.cond.clone()
                try:
                    generated = [model.sample(i, DRAWS, seed=args.seed + 100 + i) for i in range(len(conditions))]
                    measured = score(generated, observed, conditions)
                finally:
                    model.G.cond.copy_(condition)
                    model.D.cond.copy_(condition)
                trace.append({"updates": step, **draw_metrics(generated, measured, diversity=bool(args.evaluation_seeds))})
                print(f"{name} update {step} energy {trace[-1]['mean_energy_distance']:.6f}", flush=True)
                scoring_seconds += perf_counter() - tick
            if args.require_constant_lr or step % 300 == 0:
                live_rates = optimizer_rates(model)
                if args.require_constant_lr and live_rates != rates:
                    raise RuntimeError(f"optimizer rates changed at update {step}: {rates} -> {live_rates}")
            if step % 300 == 0:
                rate_trace.append({"update": step, **live_rates})
                gains = function_space_gains(model.trainer)
                if gains:
                    gain_trace.append({"update": step, **gains})
                print(f"{name} completed {step}/1200", flush=True)

        model.fit(x, labels, steps=1200, after_step=progress, on_batch=record_batch)
        torch.cuda.synchronize(device)
        seconds = perf_counter() - start - scoring_seconds
        if model.trainer.completed_steps != 1200:
            raise RuntimeError("expected exactly 1,200 completed updates")
        samples = [model.sample(i, DRAWS, seed=args.seed + 100 + i) for i in range(len(conditions))]
        metrics = score(samples, observed, conditions)
        if args.diagnose_energy:
            config["energy_diagnosis"] = diagnose_energy(samples, observed, x, labels, conditions)
        averages = {key: mean(row[key] for row in metrics.values()) for key in next(iter(metrics.values()))}
        if args.trace_every:
            trace.append({"updates": 1200, **draw_metrics(samples, metrics, diversity=bool(args.evaluation_seeds))})
            config["trace"] = trace
        if args.evaluation_seeds:
            config["observed_effective_rank"] = mean(effective_rank(rows) for rows in observed[1:])
            config["evaluation_draws"] = {str(base): score_draw(model, base, observed, conditions)
                                          for base in args.evaluation_seeds}
        if args.score_noise is not None:
            bases = [args.seed + 100, *(args.evaluation_seeds or ())]
            config["alternate_noise_draws"] = score_alternate_noise(
                model, settings.prior_sigma, args.score_noise, bases, observed, conditions)
            if not all(np.array_equal(model.sample(i, DRAWS, seed=args.seed + 100 + i), original)
                       for i, original in enumerate(samples)):
                raise RuntimeError("Failed to reproduce the original samples after restoring the training noise")
        if args.require_constant_lr:
            config["constant_lr_verified_updates"] = model.trainer.completed_steps
        if gain_trace:
            config["function_space_gain_trace"] = gain_trace
        if rate_trace:
            config["rate_trace"] = rate_trace
        if args.self_plan:
            if len(plan_trace) != 48:
                raise RuntimeError("expected 48 observed self plans")
            config["training_plans"] = plan_trace
        write_receipt(path, {**config, "gpu": torch.cuda.get_device_name(device),
                             "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                             "fit_groups": sorted(set(panel.groups[fit])),
                             "score_groups": sorted(set(panel.groups[evaluation])),
                             "fit_cells": dict(zip(conditions, np.bincount(labels).tolist())),
                             "scored_cells": dict(zip(conditions, map(len, observed))),
                             "formulation": recipe.effective_critic_formulation,
                             "optimizer_family": recipe.optimizer_family, "prior_regularizer": recipe.prior_regularizer,
                             "optimizer_momentum": optimizer_momentum(model.trainer),
                             "initial_rates": rates, "final_rates": optimizer_rates(model),
                             "initial_parameters_sha256": initialization.hexdigest(),
                             "training_batches_sha256": batch_digest.hexdigest(),
                             "seconds": seconds, "means": averages, "targets": metrics,
                             "effective_rank": {k: effective_rank(s) for k, s in zip(conditions[1:], samples[1:])}})
        print(f"{name} {averages} {seconds:.1f}s", flush=True)


if __name__ == "__main__":
    main()
