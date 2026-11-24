"""Calibrate the fixed generator LR from output movement on the original ParticleGAN releases."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import partial
from hashlib import sha256
from importlib.metadata import version
import json
from math import isclose, isfinite, log
import os
from pathlib import Path
from statistics import mean, median
import subprocess
import sys

from compare_neuroko_v12 import GROUPS, ROOT, write_frozen
from neuroko_critic_motion import STEPS, measure, probe_folds
from neuroko_rate_noise import REPLICATION_SEEDS


RELEASES = {"0.11.1": "unified_v11", "0.12.0": "unified_v12"}
RULE = ("On a fixed panel of the 64 initial latent rows with independent noise 0.15, measure the RMS "
        "change in generator outputs over all four scored targets at each of 64 updates. Compare the "
        "median change per fit on original v11 and v12 at weight 0.5 and identical fixed role rates. "
        "When at least seven of nine ratios agree in direction, divide 0.014 by their median and round "
        "to four decimal places. Replay v12 at that constant generator LR. Proceed to a 1200-update "
        "comparison only if the median ratio is closer to one in log space.")


@contextmanager
def observe(model, *, seed, means, result):
    import torch

    table = model.prior.z.detach()
    if tuple(table.shape) != (64, 16) or model.n_cond != 5:
        raise ValueError("Expected the benchmark particle table and five conditions")
    rng = torch.Generator(device=table.device).manual_seed(seed + 7162)
    z = table + .15 * torch.randn(table.shape, device=table.device, generator=rng)
    labels = torch.eye(model.n_cond, device=z.device)[1:].repeat_interleave(len(z), dim=0)
    inputs = torch.cat((z.repeat(model.n_cond - 1, 1), labels), dim=1)
    result["probe_sha256"] = sha256(inputs.cpu().numpy().tobytes()).hexdigest()

    @torch.no_grad()
    def outputs():
        return model.G.net(inputs).reshape(4, 64, 30).double()

    previous = outputs()
    result["initial_outputs_sha256"] = sha256(previous.cpu().numpy().tobytes()).hexdigest()
    trace = result["trace"] = []

    def after(step):
        nonlocal previous
        current = outputs()
        delta = current - previous
        translation = delta.mean(1)
        residual = delta - translation[:, None, :]
        before = previous.mean(1) - means
        after_error = current.mean(1) - means
        trace.append(dict(step=step, rms=delta.square().mean().sqrt().item(),
                          centroid_rms=translation.square().mean().sqrt().item(),
                          centered_rms=residual.square().mean().sqrt().item(),
                          mean_centroid_squared_error_change=(after_error.square() - before.square()).mean().item()))
        previous = current

    yield after


def worker(args):
    import torch

    if version("particlegan") != args.release:
        raise ValueError("Generator probe imported a different ParticleGAN release")
    for index, group, x, y, probes, fingerprint, _ in probe_folds(args, args.gpu):
        means = torch.stack([torch.as_tensor(x[y == k], device=probes.device).double().mean(0) for k in range(1, 5)])
        for seed in REPLICATION_SEEDS:
            if (seed + index) % 2 != args.gpu:
                continue
            result = dict(release=args.release, gpu=args.gpu, seed=seed, group=group, projected_data_sha256=fingerprint)
            fixed = dict(monitor=False, generator_lr=args.generator_lr, relativistic_weight=.5)
            setup = (x, y, probes, seed, RELEASES[args.release], probes.device, .0049, .1)
            row, _ = measure(*setup, fit_context=partial(observe, seed=seed, means=means, result=result), **fixed)
            plain, _ = measure(*setup, **fixed)
            if row != plain or row["steps"] != STEPS or len(result["trace"]) != STEPS:
                raise ValueError("Generator observation changed the fit or missed an update")
            result.update({k: row[k] for k in ("initial_parameters_sha256", "final_parameters_sha256",
                                                "training_batches_sha256", "rates")})
            if not all(isfinite(v["rms"]) and v["rms"] > 0 for v in result["trace"]):
                raise ValueError("Expected positive finite generator motion at every step")
            path = args.output / f"{args.release}-g{args.generator_lr}-s{seed}-{group}.json"
            write_frozen(path, result)
            print(args.release, args.generator_lr, seed, group, flush=True)


def run_release(args, release, lr):
    wheel = args.wheels.resolve() / release
    if not (wheel / "particlegan").is_dir():
        raise FileNotFoundError(wheel)
    pythonpath = os.pathsep.join((str(wheel), str(ROOT), os.environ.get("PYTHONPATH", "")))
    env = dict(os.environ, PYTHONPATH=pythonpath, CUBLAS_WORKSPACE_CONFIG=":4096:8",
               OPENBLAS_NUM_THREADS="2", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    extra = [] if args.window_cells is None else ["--window-cells", str(args.window_cells)]
    cmd = [sys.executable, "-Bu", __file__, "--release", release, "--generator-lr", str(lr),
           "--data", str(args.data.resolve()), "--wheels", str(args.wheels.resolve()),
           "--output", str(args.output.resolve()), *extra]
    with ThreadPoolExecutor(2) as pool:
        jobs = [pool.submit(subprocess.run, [*cmd, "--gpu", str(gpu)], cwd=ROOT, env=env, check=True) for gpu in (0, 1)]
        for job in jobs:
            job.result()


def paired_ratios(output, lr):
    rows = []
    fingerprints = []
    for seed in REPLICATION_SEEDS:
        for group in GROUPS:
            a, b = [json.loads((output / f"{release}-g{rate}-s{seed}-{group}.json").read_text())
                    for release, rate in (("0.11.1", .014), ("0.12.0", lr))]
            for field in ("seed", "group", "gpu", "projected_data_sha256", "probe_sha256", "initial_outputs_sha256",
                          "initial_parameters_sha256", "training_batches_sha256"):
                if a[field] != b[field]:
                    raise ValueError(f"Unpaired generator probe at {seed} {group} ({field})")
            summary = {}
            for row in (a, b):
                trace = row["trace"]
                expected = (.014 if row["release"] == "0.11.1" else lr, .0049, .0406)
                rates = tuple(row["rates"][role] for role in ("generator", "critic", "particles"))
                if (not all(isclose(x, y, rel_tol=1e-12) for x, y in zip(rates, expected, strict=True))
                        or [t["step"] for t in trace] != list(range(1, STEPS + 1))):
                    raise ValueError("Generator traces must cover every update at the specified rates")
                if any(not isclose(t["rms"] ** 2, t["centroid_rms"] ** 2 + t["centered_rms"] ** 2,
                                   rel_tol=1e-10) for t in trace):
                    raise ValueError("Centroid and centered motion do not account for the whole output update")
                summary[row["release"]] = {k: median(t[k] for t in trace) for k in ("rms", "centroid_rms", "centered_rms")}
                summary[row["release"]]["mean_centroid_squared_error_change"] = mean(
                    t["mean_centroid_squared_error_change"] for t in trace)
            ratio = summary["0.12.0"]["rms"] / summary["0.11.1"]["rms"]
            rows.append(dict(seed=seed, group=group, gpu=a["gpu"], ratio=ratio, motion=summary))
            fingerprints.append(dict(seed=seed, group=group, gpu=a["gpu"],
                                     projected_data_sha256=a["projected_data_sha256"], probe_sha256=a["probe_sha256"],
                                     initial_outputs_sha256=a["initial_outputs_sha256"],
                                     initial_parameters_sha256=a["initial_parameters_sha256"],
                                     training_batches_sha256=a["training_batches_sha256"],
                                     verified_final_hashes={r["release"]: r["final_parameters_sha256"] for r in (a, b)}))
    write_frozen(output / f"matched-g{lr}.json", fingerprints)
    return dict(generator_lr=lr, median_ratio=median(c["ratio"] for c in rows), pairs=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("data", "wheels", "output"):
        parser.add_argument("--" + field, required=True, type=Path)
    parser.add_argument("--window-cells", type=int)
    parser.add_argument("--release", choices=RELEASES)
    parser.add_argument("--gpu", type=int, choices=(0, 1))
    parser.add_argument("--generator-lr", type=float, default=.014)
    args = parser.parse_args()
    if args.release:
        if args.gpu is None:
            parser.error("a probe worker needs --gpu")
        worker(args)
        return
    args.output.mkdir(exist_ok=True, parents=True)
    plan = dict(rule=RULE, seeds=REPLICATION_SEEDS, groups=GROUPS, releases=RELEASES, steps=STEPS,
                fixed_rates=dict(generator=.014, critic=.0049, particles=.0406), prior_reg=.1,
                prior_sigma=.15, relativistic_weight=.5, latent_probe_noise_seed_offset=7162)
    write_frozen(args.output / "protocol.json", plan)
    for release in RELEASES:
        run_release(args, release, .014)
    first = paired_ratios(args.output, .014)
    ratio = first["median_ratio"]
    agree = sum((c["ratio"] - 1) * (ratio - 1) > 0 for c in first["pairs"])
    candidate = round(.014 / ratio, 4)
    result = dict(protocol=plan, initial=first, agreeing_pairs=agree, candidate_lr=candidate, checked=None, qualifies=False)
    if agree >= 7 and candidate > 0 and candidate != .014:
        run_release(args, "0.12.0", candidate)
        checked = paired_ratios(args.output, candidate)
        result.update(checked=checked, qualifies=abs(log(checked["median_ratio"])) < abs(log(ratio)))
    write_frozen(args.output / "result.json", result)
    print("Generator motion", first["median_ratio"], candidate, result["qualifies"], flush=True)


if __name__ == "__main__":
    main()
