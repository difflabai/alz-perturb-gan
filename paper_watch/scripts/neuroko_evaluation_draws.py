"""Replay the matched original releases to measure learning curves and sampling variability."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from statistics import mean

from compare_neuroko_v12 import GROUPS, ROOT, fit, receipt, reusable_fit, write_frozen
from neuroko_stratified import key


EVALUATION_SEEDS = list(range(100000, 100256, 32))
CHECKPOINTS = (300, 600, 900, 1200)
METRICS = ("mean_energy_distance", "spread_ratio", "effective_rank")
ORIGINAL = ROOT / "paper_watch/notes/neuroko_v12_prior_weight_original_releases.json"


def replay(args, arm, cell, original):
    seed, group = cell["seed"], cell["group"]
    gpu = (seed + GROUPS.index(group)) % 2
    path = receipt(args.output, "validation", *arm, seed, group)
    path = path.with_stem(path.stem + "-trace300-resampled")
    if not reusable_fit(path, require_constant_lr=True):
        options = ["--trace-every", "300", "--evaluation-seeds", *map(str, EVALUATION_SEEDS)]
        fit(args, "validation", seed, group, gpu, [arm], options=options)
    row = json.loads(path.read_text())
    for field in ("seed", "targets", "means", "initial_rates", "final_rates", "score_groups",
                  "constant_lr_verified_updates"):
        if row[field] != original[field]:
            raise ValueError(f"Replay differs from the published fit in {field} at {path}")
    for field in ("projected_data_sha256", "initial_parameters_sha256", "training_batches_sha256", "visible_devices"):
        if row[field] != cell[field]:
            raise ValueError(f"Replay differs in {field} at {path}")
    if (row["particlegan"], row["recipe"], row["name"]) != (arm[0], arm[1], arm[2]["name"]):
        raise ValueError(f"Unexpected formulation in {path}")
    if (row["constant_lr_verified_updates"] != 1200 or row["evaluation_seeds"] != EVALUATION_SEEDS
            or set(row["evaluation_draws"]) != set(map(str, EVALUATION_SEEDS))
            or tuple(r["updates"] for r in row["trace"]) != CHECKPOINTS
            or row["trace"][-1]["mean_energy_distance"] != row["means"]["energy_distance"]):
        raise ValueError(f"Incomplete diagnostic in {path}")
    return {k: row[k] for k in ("seed", "score_groups", "trace", "evaluation_draws", "observed_effective_rank")}


def summarize(records):
    curves, integrated, by_draw, by_cell = {}, {}, {}, {}
    for name, rows in records.items():
        curves[name] = {step: {m: mean(r["trace"][i][m] for r in rows) for m in METRICS}
                        for i, step in enumerate(CHECKPOINTS)}
        by_draw[name] = {s: mean(r["evaluation_draws"][str(s)]["mean_energy_distance"] for r in rows)
                         for s in EVALUATION_SEEDS}
        by_cell[name] = [mean(v["mean_energy_distance"] for v in r["evaluation_draws"].values()) for r in rows]
        integrated[name] = {m: mean(v[m] for r in rows for v in r["evaluation_draws"].values()) for m in METRICS}
    v12 = next(name for name in records if "/unified_v12/" in name)
    comparisons = {}
    for name in records:
        if name == v12:
            continue
        gaps = [by_draw[v12][s] - by_draw[name][s] for s in EVALUATION_SEEDS]
        comparisons[name] = {"v12_minus_arm_by_draw": gaps, "mean_difference": mean(gaps),
                             "v12_pairs_won_after_averaging_draws":
                                 sum(a < b for a, b in zip(by_cell[v12], by_cell[name], strict=True))}
    return dict(curves=curves, repeated_draw_means=integrated, comparisons=comparisons, records=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("data", "output", "wheels"):
        parser.add_argument("--" + flag, required=True, type=Path)
    parser.add_argument("--window-cells", type=int)
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    args = parser.parse_args()
    args.require_constant_lr = True
    args.output, args.data, args.wheels = (p.resolve() for p in (args.output, args.data, args.wheels))
    args.output.mkdir(parents=True, exist_ok=True)
    published = json.loads(ORIGINAL.read_text())
    arms = [arm for arm in published["protocol"]["arms"] if arm[0] != "0.12.0" or arm[1] == "unified_v12"]
    cells = published["cells"]
    protocol = dict(arms=arms, cells=cells, checkpoints=CHECKPOINTS, evaluation_seed_bases=EVALUATION_SEEDS,
                    draws_per_target=256, source=ORIGINAL.name,
                    rule="Replay all nine published training pairs. Verify each original target and input hash. "
                         "Score the original draws every 300 updates. At 1200 also average all eight fixed, "
                         "shared draw sets within each fit. This diagnostic does not select models.")
    write_frozen(args.output / "protocol.json", protocol)
    records = {key(a): [] for a in arms}
    with ThreadPoolExecutor(args.workers_per_gpu) as first, ThreadPoolExecutor(args.workers_per_gpu) as second:
        jobs = []
        for i, cell in enumerate(cells):
            pool = (first, second)[(cell["seed"] + GROUPS.index(cell["group"])) % 2]
            for arm in arms:
                name = key(arm)
                jobs.append((name, pool.submit(replay, args, arm, cell, published["per_arm"][name][i])))
        for name, job in jobs:
            records[name].append(job.result())
    result = summarize(records)
    write_frozen(args.output / "result.json", result)
    print("Repeated evaluation", result["repeated_draw_means"], result["comparisons"], flush=True)


if __name__ == "__main__":
    main()
