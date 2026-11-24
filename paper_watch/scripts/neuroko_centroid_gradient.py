"""Measure how the released generator loss moves the conditional mean in output space."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from functools import partial
from importlib.metadata import version
import json
import multiprocessing
import os
from pathlib import Path
from statistics import mean

from compare_neuroko_v12 import GROUPS, write_frozen
from neuroko_critic_motion import STEPS, measure, probe_folds
from neuroko_rate_noise import CENTROID_WEIGHTS, REPLICATION_SEEDS, centroid_weight_summary


RULE = ("At fixed critic learning rate 0.0049 measure the gradient on the generator's final bias at "
        "each of 64 updates. Compare it with the generated minibatch mean minus the full training mean "
        "for the same condition. For each weight average the cosine within each of the four scored "
        "targets, then across targets, then across nine seed–group pairs. Select the highest mean, "
        "breaking ties in favor of weight 2, then smaller weight. Require improvement over weight 2 "
        "on at least six pairs. Only the selected weight can enter a separate 1200-update energy study.")


@contextmanager
def observe(model, *, means, trace, first_gradients, checks, direct_weight=None):
    import torch

    loss = model.trainer.loss
    native = loss.g_loss
    coupling = loss.t1c
    original_weight = coupling.relativistic_weight
    if original_weight != 2.:
        raise ValueError("The diagnostic requires the trained reference weight of 2")
    bias = model.G.net[-1].bias

    def inspect(fake_logits, real_logits):
        parts = loss._resolve_pair_parts(fake_logits, real_logits)
        fake = parts[0]
        if not parts[-1] or tuple(fake.shape) != (64, 30):
            raise ValueError("Expected the captured 64 generated rows in the benchmark projection")
        label = int(model.G.cond.argmax().item())
        error = (fake.detach().mean(0) - means[label]).double()
        coefficient = (torch.tensor(original_weight, device=fake.device, requires_grad=True)
                       if direct_weight is None else direct_weight)
        coupling.relativistic_weight = coefficient
        try:
            value = native(fake_logits, real_logits)
        finally:
            coupling.relativistic_weight = original_weight
        total, = torch.autograd.grad(value, bias, retain_graph=True)
        total = total.detach().double()
        if direct_weight is not None:
            expected = first_gradients[direct_weight]
            relative_error = float((total - expected).norm() / expected.norm())
            if not relative_error < 2e-4:
                raise ValueError(f"The direct weight-{direct_weight} gradient differs by {relative_error}")
            checks[str(direct_weight)] = relative_error
            return value

        unit_term, = torch.autograd.grad(value, coefficient, create_graph=True, retain_graph=True)
        derivative, = torch.autograd.grad(unit_term, bias, retain_graph=True)
        derivative = derivative.detach().double()
        if not (total.norm() > 0 and derivative.norm() > 0 and error.norm() > 0):
            raise ValueError("Expected a nonzero translation, weight derivative and centroid error")
        cosines = {}
        for weight in CENTROID_WEIGHTS:
            gradient = total + (weight - original_weight) * derivative
            cosines[str(weight)] = float(torch.nn.functional.cosine_similarity(gradient, error, dim=0))
            if not trace:
                first_gradients[weight] = gradient
        trace.append(dict(condition=label, centroid_error=float(error.norm()), total_gradient=float(total.norm()),
                          gradient_per_unit_weight=float(derivative.norm()), cosines=cosines))
        return value

    loss.g_loss = inspect
    try:
        yield
    finally:
        loss.g_loss = native


def score_trace(trace):
    if {row["condition"] for row in trace} != set(range(5)):
        raise ValueError("Expected the control and all four scored targets")
    return {str(w): mean(mean(row["cosines"][str(w)] for row in trace if row["condition"] == target)
                         for target in range(1, 5)) for w in CENTROID_WEIGHTS}


def probe_gpu(args, gpu):
    import torch

    if version("particlegan") != "0.12.0":
        raise ValueError("Use ParticleGAN 0.12.0 for the generator gradient study")
    cells = []
    for i, group, x, labels, probes, data_hash, probe_hash in probe_folds(args, gpu):
        means = torch.stack([torch.as_tensor(x[labels == k], device=probes.device).mean(0) for k in range(len(probes))])
        for seed in REPLICATION_SEEDS:
            if (seed + i) % 2 != gpu:
                continue
            path = args.output / f"seed{seed}-{group}.json"
            ref = json.loads((args.reference / path.name).read_text())
            if (ref["projected_data_sha256"], ref["probe_indices_sha256"], ref["gpu"]) != (data_hash, probe_hash, gpu):
                raise ValueError("The reference has different projected cells, probes or physical GPU")
            trace, first, checks = [], {}, {}
            context = partial(observe, means=means, trace=trace, first_gradients=first, checks=checks)
            params = (x, labels, probes, seed, "unified_v12", probes.device, .0049, .1)
            row, _ = measure(*params, fit_context=context)
            original = ref["rows"]["unified_v12"]
            if any(row[k] != original[k] for k in row) or len(trace) != STEPS:
                raise ValueError("The gradient measurement changed the original training trajectory")
            for weight in CENTROID_WEIGHTS:
                if weight != 2.:
                    measure(*params, steps=1, fit_context=partial(context, direct_weight=weight))
            summary = dict(seed=seed, group=group, gpu=gpu, projected_data_sha256=data_hash,
                           initial=row["initial_parameters_sha256"], final=row["final_parameters_sha256"],
                           batches=row["training_batches_sha256"], first_step_relative_errors=checks,
                           mean_cosine=score_trace(trace))
            write_frozen(path, dict(summary=summary, trace=trace))
            print(seed, group, summary["mean_cosine"], flush=True)
            cells.append(summary)
    return cells


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "reference", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--window-cells", type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    protocol = dict(rule=RULE, weights=CENTROID_WEIGHTS, steps=STEPS, seeds=REPLICATION_SEEDS, groups=GROUPS,
                    rates=dict(generator=.014, critic=.0049, particles=.0406), prior_sigma=.15, prior_reg=.1)
    write_frozen(args.output / "protocol.json", protocol)
    with ProcessPoolExecutor(2, mp_context=multiprocessing.get_context("spawn")) as pool:
        jobs = [pool.submit(probe_gpu, args, gpu) for gpu in (0, 1)]
        cells = sorted([row for job in jobs for row in job.result()], key=lambda row: (row["seed"], row["group"]))
    result = dict(protocol=protocol, cells=cells, **centroid_weight_summary(cells))
    write_frozen(args.output / "result.json", result)
    print("Generator gradient comparison", result["scores"], "selected", result["selected_weight"], flush=True)


if __name__ == "__main__":
    main()
