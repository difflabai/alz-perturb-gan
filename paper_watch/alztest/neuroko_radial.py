"""Measure the radial gradient of the released v11 transport loss at a matched law."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from pathlib import Path
from statistics import mean, stdev
from unittest.mock import patch

import numpy as np
from particlegan import get_recipe
import torch

from .neuroko import LAYOUT, load_comparison, open_screen, project_fold
from .neuroko_duration import write_receipt


def probe(primitive, fake, real, center, scale, particle_ids, exclude_siblings):
    """Positive d(loss)/d(log scale) pushes a generator to contract under descent."""
    log_scale = torch.tensor(np.log(scale), device=fake.device, dtype=fake.dtype, requires_grad=True)
    x = center + log_scale.exp() * (fake - center)
    logits = torch.zeros(len(real), device=real.device)
    same_particle = particle_ids[:, None] == particle_ids[None, :]
    plans = []
    solve = primitive._sinkhorn_slack

    def observe_plan(*args, **kwargs):
        if exclude_siblings and len(plans) == 1:
            mask = kwargs.get("pair_mask")
            kwargs["pair_mask"] = ~same_particle if mask is None else (~same_particle & mask)
        plan = solve(*args, **kwargs)
        plans.append(plan.detach())
        return plan

    with patch.object(primitive, "_sinkhorn_slack", side_effect=observe_plan):
        loss, _ = primitive(x, real, logits, logits, record=None, return_metrics=False)
    if len(plans) != 2:
        raise RuntimeError(f"expected a cross plan and a self plan, saw {len(plans)}")
    cross, within = [v[:-1, :-1] for v in plans]
    return {
        "radial_gradient": torch.autograd.grad(loss, log_scale)[0].item(),
        "self_diagonal_fraction": (within.diag().sum() / within.sum()).item(),
        "self_same_particle_fraction": (within[same_particle].sum() / within.sum()).item(),
        "cross_active_mass": cross.sum().item(),
        "cross_column_l1_residual": (plans[0].sum(0)[:-1] - 1 / len(real)).abs().sum().item(),
        "self_column_l1_residual": (plans[1].sum(0)[:-1] - 1 / len(fake)).abs().sum().item(),
    }


def trace_training_plans(trainer, mode):
    """Observe the released solver, optionally test a row-ID mask on the fake self plan."""
    primitive = trainer.loss.t1c
    sample_prior = trainer._sample_training_prior
    sinkhorn = primitive._sinkhorn_slack
    ids = None
    inside = False
    calls = 0
    observation = None
    trace = []

    def sample(n):
        nonlocal ids
        latent, ids = sample_prior(n)
        return latent, ids

    def begin(module, args):
        nonlocal inside, calls, observation
        inside, calls, observation = True, 0, None
        step = trainer.completed_steps + 1
        if step % 25 == 0:
            fake, real = args[:2]
            observation = {"step": step, "spread_ratio": (fake.var(0).sum() / real.var(0).sum()).sqrt().item()}
            observed = observation
            direction = (fake.detach() - fake.detach().mean(0)).detach()

            def radial_gradient(gradient):
                observed["live_radial_gradient"] = (gradient.detach() * direction).sum().item()

            def logit_gradient(gradient):
                observed["live_logit_gradient_l1"] = gradient.detach().abs().sum().item()

            if fake.requires_grad:
                fake.register_hook(radial_gradient)
            if args[2].requires_grad:
                args[2].register_hook(logit_gradient)

    def end(module, args, output):
        nonlocal inside
        inside = False
        if output is not None and calls != 2:
            raise RuntimeError(f"expected a cross plan and a self plan, observed {calls} solves")

    def solve(*args, **kwargs):
        nonlocal calls
        if not inside:
            return sinkhorn(*args, **kwargs)
        is_self = calls == 1
        calls += 1
        same = None
        if is_self and (observation is not None or mode in ("exclude-same-id", "discount-same-id")):
            same = ids[:, None] == ids[None, :]
            if same.shape != args[0].shape:
                raise RuntimeError("sampled prior IDs do not correspond to the self plan")
        permitted = None
        if is_self and mode == "discount-same-id":
            # For the released unscreened plan, this divides same-ID kernel affinities by the batch size.
            cost, *rest = args
            args = (cost + same.to(cost.dtype) * (primitive.eps * np.log(len(ids))), *rest)
        elif is_self and mode == "exclude-same-id":
            permitted = ~same
        elif is_self and mode == "opposite-halves":
            half = torch.arange(len(ids), device=ids.device) < len(ids) // 2
            permitted = half[:, None] != half[None, :]
        if permitted is not None:
            mask = kwargs.get("pair_mask")
            kwargs["pair_mask"] = permitted if mask is None else mask & permitted
        plan = sinkhorn(*args, **kwargs)
        if observation is not None:
            active = plan[:-1, :-1]
            if is_self:
                mass = active.sum()
                observation.update(unique_particles=ids.unique().numel(),
                                   diagonal_fraction=(active.diag().sum() / mass).item(),
                                   same_particle_fraction=(active[same].sum() / mass).item())
                trace.append(observation)
            else:
                observation["cross_active_mass"] = active.sum().item()
        return plan

    trainer._sample_training_prior = sample
    primitive.register_forward_pre_hook(begin)
    primitive.register_forward_hook(end, always_call=True)
    primitive._sinkhorn_slack = solve
    return trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--window-cells", type=int)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=32)
    parser.add_argument("--quantile", type=float, default=0.25)
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--rho", type=float, default=3.0)
    parser.add_argument("--eps", nargs="+", type=float, default=(0.03, 0.1, 0.3, 1.0, 3.0))
    parser.add_argument("--self-coupling", choices=("feydy_discrete", "hollow"), default="feydy_discrete")
    parser.add_argument("--repeat-particles", action="store_true",
                        help="draw 64 fake rows with replacement from 64 batch-specific prototypes and add 0.025 jitter")
    parser.add_argument("--exclude-siblings", action="store_true",
                        help="diagnostic counterfactual excluding pairs with the same prototype from the self plan")
    args = parser.parse_args()
    if args.draws < 2:
        parser.error("use at least two independent draws")
    device = torch.device("cuda", torch.cuda.current_device())
    with ExitStack() as stack:
        screens = [stack.enter_context(open_screen(args.data, n, args.window_cells)) for n in LAYOUT]
        conditions, panels = load_comparison(screens, 100, 160, 6)
    panel = panels["iNeurons"]
    fit = np.isin(panel.groups, ["Screen1_3", "Screen1_4"])
    x, _ = project_fold(panel.x[fit], panel.x[fit][:1])
    labels = panel.labels[fit]
    data = [torch.as_tensor(x[labels == i], dtype=torch.float32, device=device) for i in range(len(conditions))]
    results = []
    for source in ("neurons", "gaussian"):
        for eps in args.eps:
            for scale in (0.6, 0.8, 1.0, 1.2):
                rng = np.random.default_rng(139)
                observations = []
                for i in range(args.draws):
                    if source == "neurons":
                        pool = data[i % len(data)]
                        indices = rng.choice(len(pool), 128, replace=False)
                        fa, re = pool[indices].chunk(2)
                        center = pool.mean(0)
                    else:
                        pair = rng.standard_normal((128, 30)).astype(np.float32)
                        fa, re = torch.from_numpy(pair).to(device).chunk(2)
                        center = torch.zeros_like(re[0])
                    ids = torch.arange(64, device=device)
                    if args.repeat_particles:
                        ids = torch.from_numpy(rng.integers(0, 64, 64)).to(device)
                        noise = torch.from_numpy(rng.normal(0, 0.025, (64, 30)).astype(np.float32)).to(device)
                        fa = fa[ids] + noise
                    primitive = get_recipe("unified_v11", rhbi_eps=eps, rhbi_rho=args.rho,
                                           rhbi_nn_quantile=args.quantile, rhbi_iters=args.iters,
                                           rhbi_self_coupling=args.self_coupling).make_loss().t1c.to(device)
                    observations.append(probe(primitive, fa, re, center, scale, ids, args.exclude_siblings))
                means = {key: mean(r[key] for r in observations) for key in observations[0]}
                se = stdev(r["radial_gradient"] for r in observations) / len(observations) ** 0.5
                results.append({"source": source, "eps": eps, "scale": scale, **means, "gradient_se": se})
                print(source, eps, scale, means, flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_receipt(args.out, {"draws": args.draws, "repeat_particles": args.repeat_particles,
                            "exclude_siblings": args.exclude_siblings, "batch_size": 64, "output_dimensions": 30, "self_coupling": args.self_coupling, "rho": args.rho, "nn_quantile": args.quantile, "iters": args.iters,
                            "device": torch.cuda.get_device_name(), "results": results})


if __name__ == "__main__":
    main()
