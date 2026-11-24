"""Replay six matched training settings and partition the generated variance by particle row."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from statistics import mean

from compare_neuroko_v12 import GROUPS, fit, receipt, reusable_fit, write_frozen
from neuroko_rate_noise import NOISY_PRIOR, WIDE_NOISY_PRIOR
from neuroko_role_factorial import paired

PAIRS = ((22, "Screen1_1"), (24, "Screen1_3"), (23, "Screen1_4"))
REPEATS = 32
HASHES = ("projected_data_sha256", "initial_parameters_sha256", "training_batches_sha256")


def replay(args, seed, group):
    gpu = (seed + GROUPS.index(group)) % 2
    common = None
    measurements = {}
    for lower, higher in zip(paired(NOISY_PRIOR), paired(WIDE_NOISY_PRIOR), strict=True):
        recipe = lower[1]
        measurements[recipe] = {}
        for arm in (lower, higher):
            base = receipt(args.output, "validation", *arm, seed, group)
            path = base.with_stem(f"{base.stem}-panel{REPEATS}")
            if not reusable_fit(path, True):
                fit(args, "validation", seed, group, gpu, [arm], options=["--particle-panel", str(REPEATS)])
            row = json.loads(path.read_text())
            ref = json.loads(receipt(args.reference, "validation", *arm, seed, group).read_text())
            for field in (*HASHES, "targets", "means", "settings", "initial_rates", "final_rates",
                          "score_groups", "fit_groups", "recipe", "particlegan", "seed", "visible_devices"):
                if row[field] != ref[field]:
                    raise ValueError(f"{field} differs from the original fit at {path}")
            panel = row["particle_panel"]
            if (row["visible_devices"] != str(gpu) or row["constant_lr_verified_updates"] != 1200
                    or row["initial_rates"] != row["final_rates"] or panel["particles"] != 64
                    or panel["repeats"] != REPEATS or panel["seed"] != 73219
                    or set(panel["targets"]) != set(ref["targets"])):
                raise ValueError(f"Incomplete particle panel or fixed-rate fit at {path}")
            hashes = {k: row[k] for k in HASHES}
            if common is not None and common != hashes:
                raise ValueError("The six paired fits must use identical inputs and initialization")
            common = hashes
            sigma = arm[2]["prior_sigma"]
            measurements[recipe][str(sigma)] = dict(panel=panel, energy=row["means"]["energy_distance"])
        write_frozen(args.output / f"{recipe}-s{seed}-{group}.json", measurements[recipe])
    return dict(seed=seed, group=group, gpu=gpu, hashes=common, measurements=measurements)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("data", "wheels", "reference", "output"):
        parser.add_argument("--" + field, type=Path, required=True)
    parser.add_argument("--window-cells", type=int)
    args = parser.parse_args()
    args.require_constant_lr = True
    for field in ("data", "wheels", "reference", "output"):
        setattr(args, field, getattr(args, field).resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    plan = dict(pairs=PAIRS, lower=paired(NOISY_PRIOR), higher=paired(WIDE_NOISY_PRIOR), repeats=REPEATS,
                seed=73219, updates=1200, rule="Use one distinct training seed per group. For each target, "
                "draw fresh Gaussian jitter 32 times from each of the 64 table rows with the release sampler. "
                "Within and between describe this balanced finite panel. Average the per-target quantities "
                "equally across four targets and three fitted pairs. Diagnostic only.")
    write_frozen(args.output / "protocol.json", plan)
    with ThreadPoolExecutor(1) as first, ThreadPoolExecutor(1) as second:
        jobs = [(first, second)[(s + GROUPS.index(g)) % 2].submit(replay, args, s, g) for s, g in PAIRS]
        cells = [job.result() for job in jobs]
    summary = {}
    for recipe, settings in cells[0]["measurements"].items():
        summary[recipe] = {}
        for sigma in settings:
            fits = [c["measurements"][recipe][sigma]["panel"] for c in cells]
            targets = [t for f in fits for t in f["targets"].values()]
            summary[recipe][sigma] = dict(
                within_fraction=mean(t["within"] / t["total"] for t in targets),
                within_over_observed=mean(t["within"] / t["observed"] for t in targets),
                between_over_observed=mean(t["between"] / t["observed"] for t in targets),
                centroid_distance=mean(t["centroid_distance"] for t in targets),
                table_variance=mean(f["table_variance"] for f in fits),
                table_effective_rank=mean(f["table_effective_rank"] for f in fits))
    write_frozen(args.output / "result.json", dict(protocol=plan, summary=summary, cells=cells))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
