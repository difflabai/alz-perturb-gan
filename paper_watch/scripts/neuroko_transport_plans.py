"""Measure how the transport grid changes matching on identical real and generated minibatches."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from copy import copy
from functools import partial
from importlib.metadata import version
import json
import multiprocessing
import os
from pathlib import Path
from statistics import median
from unittest.mock import patch

from compare_neuroko_v12 import GROUPS, write_frozen
from neuroko_critic_motion import STEPS, measure, probe_folds


def plan_metrics(plan):
    rows = plan[:-1].double()
    matched = rows[:, :-1]
    mass = matched.sum(dim=1)
    # Conditional participation ratio of real matches, weighted by each row's matched mass.
    counts = mass.square() / matched.square().sum(dim=1).clamp_min(1e-100)
    return rows[:, -1].sum() / rows.sum(), (mass * counts).sum() / mass.sum().clamp_min(1e-100)


@contextmanager
def inspect_plans(model, settings, trace):
    import torch

    loss = model.trainer.loss
    coupling = loss.t1c
    generator_loss, native_plan = loss.g_loss, coupling._sinkhorn_slack
    if (coupling.tau, coupling.eps, coupling.rho, coupling.self_coupling, coupling.rapidity_screening) != (2., .1, 3., "hollow", False):
        raise ValueError("Expected the calibrated hollow baseline")
    in_generator = False

    def observe_generator(*args, **kw):
        nonlocal in_generator
        in_generator = True
        try:
            return generator_loss(*args, **kw)
        finally:
            in_generator = False

    def observe_plan(cost, row_scores, col_scores, hollow=False, pair_mask=None):
        plan = native_plan(cost, row_scores, col_scores, hollow=hollow, pair_mask=pair_mask)
        if in_generator and not hollow:
            probe = copy(coupling)
            values = []
            for tau, eps, rho in settings:
                probe.tau, probe.eps, probe.rho = tau, eps, rho
                candidate = type(probe)._sinkhorn_slack(probe, cost, row_scores, col_scores,
                                                       hollow=False, pair_mask=pair_mask)
                if (tau, eps, rho) == (2., .1, 3.) and not torch.equal(candidate, plan):
                    raise ValueError("The unmodified probe must exactly reproduce the native cross plan")
                values.append(torch.stack(plan_metrics(candidate)))
            trace.append(torch.stack(values).tolist())
        return plan

    def check_count(step):
        if len(trace) != step:
            raise ValueError(f"Expected one generator cross plan per update through step {step}")

    with patch.object(loss, "g_loss", observe_generator), patch.object(coupling, "_sinkhorn_slack", observe_plan):
        yield check_count


def run_gpu(args, settings, seeds, gpu):
    if version("particlegan") != "0.12.0":
        raise ValueError("Use the released v12 wheel")
    cells = []
    for i, group, x, labels, probes, data_hash, probe_hash in probe_folds(args, gpu):
        for seed in seeds:
            if (seed + i) % 2 != gpu:
                continue
            name = f"seed{seed}-{group}.json"
            ref = json.loads((args.reference / name).read_text())
            if (ref["gpu"], ref["projected_data_sha256"], ref["probe_indices_sha256"]) != (gpu, data_hash, probe_hash):
                raise ValueError("Mismatched reference data, probes or GPU")
            trace = []
            row, _ = measure(x, labels, probes, seed, "unified_v12", probes.device, .0049, .1,
                             fit_context=partial(inspect_plans, settings=settings, trace=trace))
            if len(trace) != STEPS or any(v != ref["rows"]["unified_v12"][k] for k, v in row.items()):
                raise ValueError("Expected one cross plan per update and an unchanged baseline training trajectory")
            metrics = [[median(step[j][k] for step in trace) for k in (0, 1)] for j in range(len(settings))]
            cell = dict(seed=seed, group=group, gpu=gpu, projected_data_sha256=data_hash,
                        fingerprints={k: v for k, v in row.items() if k.endswith("sha256")}, medians=metrics)
            write_frozen(args.output / name, dict(cell=cell, trace=trace))
            cells.append(cell)
    return cells


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("data", "reference", "output"):
        parser.add_argument("--" + field, type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--grid", type=Path, help="frozen tau and epsilon grid")
    source.add_argument("--rho-study", type=Path, help="frozen single-rho training protocol")
    parser.add_argument("--window-cells", type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    study = json.loads((args.grid or args.rho_study).read_text())
    if args.grid:
        pairs = sorted([(2., .1), *((a[2]["rhbi_tau"], a[2]["rhbi_eps"]) for a in study["candidates"])])
        if pairs != [(tau, eps) for tau in (1., 2., 4.) for eps in (.05, .1, .2)]:
            raise ValueError("Expected the frozen three-by-three transport grid")
        settings = [(tau, eps, 3.) for tau, eps in pairs]
        metadata = dict(settings=pairs)
        seeds = study["first"]
    else:
        release, recipe, trial = study["arms"][0]
        unchanged = {k: v for k, v in trial.items() if k not in ("name", "rhbi_rho")}
        baseline = dict(unchanged, name="original-v11-critic-49")
        if (release != "0.12.0" or recipe != "unified_v12" or trial.get("rhbi_rho") not in (1.5, 6.)
                or [release, recipe, baseline] not in study["arms"]):
            raise ValueError("Expected the calibrated v12 comparison with only rho halved or doubled")
        settings = [(2., .1, 3.), (2., .1, trial["rhbi_rho"]), (2., .2, 3.)]
        metadata = dict(settings=settings, setting_columns=["tau", "epsilon", "rho"])
        seeds = study["screen_seeds"]
    if tuple(study["groups"]) != GROUPS or seeds != [22, 23, 24]:
        raise ValueError("Expected the three calibrated seeds and groups")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    with ProcessPoolExecutor(2, mp_context=multiprocessing.get_context("spawn")) as pool:
        jobs = [pool.submit(run_gpu, args, settings, seeds, gpu) for gpu in (0, 1)]
        cells = sorted([cell for job in jobs for cell in job.result()], key=lambda c: (c["seed"], c["group"]))
    overall = [[median(c["medians"][j][k] for c in cells) for k in (0, 1)] for j in range(len(settings))]
    write_frozen(args.output / "result.json", dict(**metadata, steps=STEPS, seeds=seeds,
                 metrics=["fraction_of_generated_mass_unmatched", "effective_real_matches_per_matched_row"],
                 medians=overall, cells=cells))


if __name__ == "__main__":
    main()
