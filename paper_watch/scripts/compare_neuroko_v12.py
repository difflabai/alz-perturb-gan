"""Select a v12 setting on training groups, then test the frozen setting and controls on two GPUs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
from statistics import mean, median, stdev
import subprocess
import sys
from uuid import uuid4


GROUPS = ("Screen1_1", "Screen1_3", "Screen1_4")
OUTER_SEEDS = range(72, 84)
HOLLOW = {"rhbi_self_coupling": "hollow", "rhbi_relativistic_weight": 2.0}
NATIVE_OPTIMIZER_SMOOTHING = {"v12": .0061, "matched_v11": .0061, "matched_empf": .001}


def with_shared_smoothing(arms, smoothing):
    adjusted = dict(arms)
    for role, default in NATIVE_OPTIMIZER_SMOOTHING.items():
        if smoothing != default:
            release, recipe, trial = arms[role]
            adjusted[role] = [release, recipe, dict(trial, name=f"{trial['name']}-smoothing{smoothing}",
                                                     optimizer_smoothing=smoothing)]
    return adjusted


def rates(name, generator, critic, particles, **settings):
    return dict(name=name, lr=generator, d_lr_mult=critic / generator,
                prior_lr_mult=particles / generator, **settings)


CANDIDATES = [
    {"name": "default", "lr": .014},
    {"name": "half", "lr": .007},
    {"name": "double", "lr": .028},
    rates("slow-critic", .014, .0075, .0406),
    rates("slow-particles", .014, .021, .0125),
    rates("slow-both", .014, .0075, .0125),
    {"name": "hollow", "lr": .014, "rhbi_self_coupling": "hollow"},
    {"name": "active2", "lr": .014, "rhbi_relativistic_weight": 2.0},
    {"name": "hollow-active2", "lr": .014, **HOLLOW},
    rates("v11-transfer", .014, .0075, .0125, **HOLLOW),
    rates("v11-transfer-low-g", .007, .0075, .0125, **HOLLOW),
    rates("v11-transfer-high-g", .028, .0075, .0125, **HOLLOW),
]
BY_NAME = {trial["name"]: trial for trial in CANDIDATES}
ROOT = Path(__file__).resolve().parents[2]


def write_frozen(path, value):
    encoded = json.dumps(value, indent=2, allow_nan=False) + "\n"
    if not path.exists():
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            temporary.write_text(encoded)
            try:
                os.link(temporary, path)
            except FileExistsError:
                pass
        finally:
            temporary.unlink(missing_ok=True)
    if json.loads(path.read_text()) != json.loads(encoded):
        raise ValueError(f"protocol or selection differs from {path}")


def receipt(output, stage, release, recipe, trial, seed, group):
    suffix = f"-{group}" if stage == "validation" and group != GROUPS[0] else ""
    return output / stage / f"{recipe}-{release}" / f"{stage}-{trial['name']}-s{seed}{suffix}-deterministic.json"


def reusable_fit(path, require_constant_lr):
    if not path.is_file():
        return False
    row = json.loads(path.read_text())
    if row.get("settings", {}).get("output_noise_std") is not None and row.get("fixed_output_noise_verified_updates") != 1200:
        return False
    return not require_constant_lr or row.get("constant_lr_verified_updates") == row.get("updates") == 1200


def fit(args, stage, seed, group, gpu, arms, *, options=()):
    for release, recipe, trial in arms:
        target = receipt(args.output, stage, release, recipe, trial, seed, group)
        target.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="2",
                   OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
        wheel = args.wheels / release
        if not (wheel / "particlegan").is_dir():
            raise FileNotFoundError(f"install ParticleGAN {release} into {wheel}")
        env["PYTHONPATH"] = os.pathsep.join([str(wheel), str(ROOT), str(ROOT / "paper_watch"),
                                             os.environ.get("PYTHONPATH", "")])
        cmd = [sys.executable, "-Bu", "-m", "paper_watch.alztest.neuroko_tune", stage,
               "--release", release, "--recipe", recipe, "--seed", str(seed), "--data", str(args.data),
               "--output", str(target.parent), "--deterministic", "--trial", json.dumps(trial)]
        if stage == "validation":
            cmd.extend(["--validation-group", group])
        if getattr(args, "require_constant_lr", False):
            cmd.append("--require-constant-lr")
        if args.window_cells:
            cmd.extend(["--window-cells", str(args.window_cells)])
        cmd.extend(options)
        with target.with_suffix(".log").open("a") as log:
            # Replications may request the same control; the second worker validates the cached fit.
            fcntl.flock(log, fcntl.LOCK_EX)
            subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def schedule(args, jobs):
    with ThreadPoolExecutor(args.workers_per_gpu) as first, ThreadPoolExecutor(args.workers_per_gpu) as second:
        pending = [(first, second)[gpu].submit(fit, args, stage, seed, group, gpu, arms)
                   for stage, seed, group, gpu, arms in jobs]
        for future in pending:
            future.result()


def validation(args, names, seed):
    jobs = [("validation", seed, group, (idx + seed) % 2, [("0.12.0", "unified_v12", BY_NAME[n])])
            for n in sorted(names, key=lambda n: n == "default") for idx, group in enumerate(GROUPS)]
    schedule(args, jobs)
    return {n: [json.loads(receipt(args.output, "validation", "0.12.0", "unified_v12",
                                   BY_NAME[n], seed, group).read_text())["means"]["energy_distance"]
                for group in GROUPS] for n in names}


def summarize(output):
    from scipy.stats import t

    freeze = json.loads((output / "freeze.json").read_text())
    protocol = json.loads((output / "protocol.json").read_text())
    hashes = ("projected_data_sha256", "initial_parameters_sha256", "training_batches_sha256")
    paired = {}
    summaries = {}
    by_arm = {}
    for release, recipe, trial in freeze["arms"]:
        name = trial["name"]
        key = f"{release}/{recipe}/{name}"
        records = [json.loads(receipt(output, "heldout", release, recipe, trial, s, GROUPS[0]).read_text())
                   for s in OUTER_SEEDS]
        first = records[0]
        for seed, row in zip(OUTER_SEEDS, records):
            matching = {h: row[h] for h in hashes}
            if paired.setdefault(seed, matching) != matching:
                raise ValueError(f"{key} has a mismatched data, initialization or minibatch hash at seed {seed}")
            if (row["updates"] != 1200 or row.get("constant_lr_verified_updates") != 1200
                    or not row["deterministic"] or row["seed"] != seed
                    or row["visible_devices"] != str(seed % 2) or row["particlegan"] != release
                    or row["recipe"] != recipe or row["name"] != name
                    or row["score_groups"] != ["Screen1_0"] or row["fit_groups"] != list(GROUPS)
                    or row["initial_rates"] != row["final_rates"] or row["initial_rates"] != first["initial_rates"]
                    or any(row["settings"][k] != v for k, v in {"batch_size": 64, "z_dim": 16,
                                                                  "num_particles": 64}.items())):
                raise ValueError(f"{key} violates the paired protocol at seed {seed}")
        metrics = [r["means"] for r in records]
        summaries[key] = {
            "rates": first["initial_rates"], "optimizer": first["optimizer_momentum"],
            "optimizer_family": first["optimizer_family"], "prior": first["prior_regularizer"],
            "critic_formulation": first["formulation"], "gpu": first["gpu"], "torch": first["torch"],
            "fit_cells": first["fit_cells"], "scored_cells": first["scored_cells"],
            "mean": {m: mean(row[m] for row in metrics) for m in metrics[0]},
            "median_seconds": median(r["seconds"] for r in records),
            "targets": {target: {m: mean(r["targets"][target][m] for r in records)
                                 for m in first["targets"][target]} for target in first["targets"]},
            "seeds": [{"seed": r["seed"], **r["means"], "seconds": r["seconds"],
                       "effective_rank": r["effective_rank"], "targets": r["targets"]} for r in records],
        }
        by_arm[key] = [r["energy_distance"] for r in metrics]
    winner = f"0.12.0/unified_v12/{freeze['winner']}"
    matched = "0.12.0/empf/matched"
    v11 = f"0.12.0/unified_v11/{freeze['winner']}"
    if any(summaries[winner]["rates"] != summaries[k]["rates"] for k in (v11, matched)):
        raise ValueError("actual optimizer rates differ between matched arms")
    comparisons = {}
    for key in by_arm:
        if key == winner:
            continue
        diff = [a - b for a, b in zip(by_arm[winner], by_arm[key])]
        center = mean(diff)
        radius = float(t.ppf(.975, len(diff) - 1)) * stdev(diff) / len(diff) ** .5
        comparisons[key] = {"mean_selected_minus_control": center, "paired_t_ci95": [center-radius, center+radius],
                            "seeds_won": sum(d < 0 for d in diff), "paired_differences": diff}
    validation_hashes = {}
    for path in (output / "validation" / "unified_v12-0.12.0").glob("*.json"):
        row = json.loads(path.read_text())
        group, = row["score_groups"]
        if group == "Screen1_0" or "Screen1_0" in row["fit_groups"] or group in row["fit_groups"]:
            raise ValueError(f"invalid validation split in {path}")
        key = f"{group}/{row['seed']}"
        matching = {h: row[h] for h in hashes}
        if validation_hashes.setdefault(key, matching) != matching:
            raise ValueError(f"validation pair mismatch in {path}")
    result = {"protocol": protocol, "validation_and_frozen_selection": freeze,
              "validation_hashes_by_group_seed": validation_hashes, "heldout_hashes_by_seed": paired,
              "heldout_arms": summaries, "heldout_paired_comparisons": comparisons}
    target = output / "report.json"
    target.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("data", "wheels", "output"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--window-cells", type=int, help="use 10000 for the original public H5ADs")
    parser.add_argument("--workers-per-gpu", type=int, default=2)
    parser.set_defaults(require_constant_lr=True)
    args = parser.parse_args()
    args.data, args.wheels, args.output = (p.resolve() for p in (args.data, args.wheels, args.output))
    args.output.mkdir(parents=True, exist_ok=True)
    write_frozen(args.output / "protocol.json", {
        "candidates": CANDIDATES, "groups": list(GROUPS), "seeds": [0, 1], "outer_seeds": list(OUTER_SEEDS),
        "selection": "Advance the three lowest seed-0 mean energies plus default and v11-transfer. "
                     "Select the lowest equal-weight mean over the three validation groups and both seeds. "
                     "Break ties by trial name. Use Screen1_0 only after the choice is frozen.",
        "controls": ["v12 default", "v12 v11-transfer", "same-wheel v11 with selected settings",
                     "0.11.1 v11 with prior selected settings", "same-wheel EMPF at matching actual rates",
                     "same-wheel EMPF at independently tuned rates (.020/.030/.058)"],
    })
    first = validation(args, list(BY_NAME), 0)
    finalists = sorted(set(sorted(first, key=lambda n: (mean(first[n]), n))[:3]) | {"default", "v11-transfer"})
    second = validation(args, finalists, 1)
    scores = {n: mean(first[n] + second[n]) for n in finalists}
    winner = min(scores, key=lambda n: (scores[n], n))
    selected = BY_NAME[winner]
    matched = {k: selected[k] for k in ("lr", "d_lr_mult", "prior_lr_mult") if k in selected}
    matched.setdefault("d_lr_mult", 1.5)
    matched.setdefault("prior_lr_mult", 2.9)
    v12_names = list(dict.fromkeys([winner, "default", "v11-transfer"]))
    arms = [("0.12.0", "unified_v12", BY_NAME[n]) for n in v12_names]
    arms.extend([("0.12.0", "unified_v11", selected), ("0.11.1", "unified_v11", BY_NAME["v11-transfer"]),
                 ("0.12.0", "empf", dict(name="matched", **matched)),
                 ("0.12.0", "empf", rates("tuned", .020, .030, .058))])
    write_frozen(args.output / "freeze.json", {"winner": winner, "seed0": first, "seed1": second,
                                               "means": scores, "arms": [list(a) for a in arms]})
    print("Frozen selection", winner, scores, flush=True)
    jobs = [("heldout", s, GROUPS[0], s % 2, arms[s % len(arms):] + arms[:s % len(arms)]) for s in OUTER_SEEDS]
    schedule(args, jobs)
    summarize(args.output)


if __name__ == "__main__":
    main()
