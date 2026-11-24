"""Compare the calibrated v12 and original controls at 1,200, 2,400 and 3,600 fixed-rate updates."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from statistics import mean

from compare_neuroko_v12 import GROUPS, ROOT, fit, receipt, write_frozen

STEPS = (1200, 2400, 3600)
NOTES = ROOT / "paper_watch/notes"
METRICS = ("mean_energy_distance", "spread_ratio", "effective_rank")


def run_fold(args, arms, seed, group, original):
    gpu = (seed + GROUPS.index(group)) % 2
    points, fingerprints = {}, {}
    for role in ("v12", "tuned_empf", "tuned_v11", "matched_empf", "matched_v11"):
        arm = arms[role]
        fit(args, "validation", seed, group, gpu, [arm], options=("--updates", "3600", "--trace-every", "1200"))
        base = receipt(args.output, "validation", *arm, seed, group)
        row = json.loads(base.with_stem(base.stem + "-u3600-trace1200").read_text())
        trace = row["trace"]
        rates = row["initial_rates"]
        trial = arm[2]
        expected_rates = dict(generator=trial["lr"], critic=trial["lr"] * trial["d_lr_mult"],
                              particles=trial["lr"] * trial["prior_lr_mult"])
        if (rates != expected_rates or (row["particlegan"], row["recipe"]) != tuple(arm[:2])
                or row["score_groups"] != [group] or set(row["fit_groups"]) != set(GROUPS) - {group}
                or row["seed"] != seed or not row["deterministic"]
                or row["updates"] != STEPS[-1] or row["constant_lr_verified_updates"] != STEPS[-1]
                or row["fixed_lr_optimizer_steps"] != {"generator_and_particles": STEPS[-1], "critic": STEPS[-1]}
                or row["final_rates"] != rates or row["visible_devices"] != str(gpu)
                or [p["updates"] for p in trace] != list(STEPS)
                or (original is not None and trace[0]["mean_energy_distance"] != original[role])
                or trace[-1]["mean_energy_distance"] != row["means"]["energy_distance"]
                or trace[-1]["training_batches_sha256"] != row["training_batches_sha256"]):
            raise ValueError(f"Invalid fixed-rate curve or 1,200-update replay for {role}, seed {seed}, {group}")
        hashes = dict(data=row["projected_data_sha256"], initialization=row["initial_parameters_sha256"],
                      training_batches=[p["training_batches_sha256"] for p in trace])
        if fingerprints and hashes != fingerprints:
            raise ValueError("Curves must use identical projected cells, initialization and training batches")
        fingerprints = hashes
        points[role] = trace
        print(seed, group, role, [round(p["mean_energy_distance"], 5) for p in trace], flush=True)
    result = dict(seed=seed, group=group, gpu=gpu, curves=points, fingerprints=fingerprints)
    write_frozen(args.output / f"curves-{seed}-{group}.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("data", "wheels", "output"):
        parser.add_argument("--" + field, type=Path, required=True)
    parser.add_argument("--window-cells", type=int)
    parser.add_argument("--fresh", action="store_true", help="replicate the fixed curves on seeds 838 through 843")
    parser.set_defaults(require_constant_lr=True)
    args = parser.parse_args()
    args.data, args.wheels, args.output = (p.resolve() for p in (args.data, args.wheels, args.output))
    args.output.mkdir(parents=True, exist_ok=True)
    sources = [json.loads((NOTES / f"neuroko_v12_original_control_benchmark_{cohort}.json").read_text())
               for cohort in ("first", "second")]
    arms = sources[0]["protocol"]["arms"]
    anchors = {}
    for source, seed in zip(sources, (22, 251), strict=True):
        if (source["protocol"]["arms"] != arms or source["protocol"]["groups"] != list(GROUPS)
                or source["result"]["seeds"][0] != seed):
            raise ValueError("Unexpected published control comparison")
        for i, group in enumerate(GROUPS):
            anchors[seed, group] = {role: source["result"]["scores"][role][i] for role in arms}
    if args.fresh:
        seeds = list(range(838, 844))
        rule = ("Freeze the five pilot configurations. Run every pair through 3,600 updates with the "
                "same data, initialization and minibatches across arms at each fresh seed. Compare "
                "the mean across all 18 pairs at the fixed endpoint to each original control, and "
                "report individual seed differences and paired wins for matched and tuned settings.")
    else:
        seeds = [22, 251]
        rule = ("Run all five arms through 3,600 updates on every pair regardless of intermediate scores. "
                "Reproduce the published 1,200-update value and match data, weights and minibatches. "
                "Assess whether v12 beats all four controls at 3,600 updates on each seed.")
    write_frozen(args.output / "protocol.json", dict(arms=arms, seeds=seeds, groups=list(GROUPS),
                 checkpoints=STEPS, metric="four-target mean energy distance", rule=rule))
    with ThreadPoolExecutor(1) as first, ThreadPoolExecutor(1) as second:
        jobs = []
        for seed in seeds:
            for i, group in enumerate(GROUPS):
                jobs.append((first, second)[(i + seed) % 2].submit(run_fold, args, arms, seed, group,
                                                                   anchors.get((seed, group))))
        cells = [job.result() for job in jobs]
    comparisons = []
    for seed in seeds:
        cohort = [cell for cell in cells if cell["seed"] == seed]
        for i, step in enumerate(STEPS):
            means = {role: {metric: mean(cell["curves"][role][i][metric] for cell in cohort) for metric in METRICS}
                     for role in arms}
            energy = means["v12"]["mean_energy_distance"]
            gaps = {role: energy - values["mean_energy_distance"] for role, values in means.items() if role != "v12"}
            comparisons.append(dict(seed=seed, updates=step, means=means, v12_minus_control=gaps))
    endpoint = [row for row in comparisons if row["updates"] == STEPS[-1]]
    result = dict(cells=cells, comparisons=comparisons)
    if args.fresh:
        result["endpoint_means"] = {role: mean(row["means"][role]["mean_energy_distance"] for row in endpoint)
                                    for role in arms}
        summary = {}
        for role in ("tuned_empf", "tuned_v11", "matched_empf", "matched_v11"):
            gaps = [row["v12_minus_control"][role] for row in endpoint]
            wins = sum(cell["curves"]["v12"][-1]["mean_energy_distance"] < cell["curves"][role][-1]["mean_energy_distance"]
                       for cell in cells)
            summary[role] = dict(mean_v12_minus_control=mean(gaps), pairs_won=wins,
                                 seed_means_won=sum(gap < 0 for gap in gaps))
        result["endpoint_comparisons"] = summary
        result["leads_all_controls_on_average"] = all(row["mean_v12_minus_control"] < 0 for row in summary.values())
    else:
        result["leads_both_seeds"] = all(max(row["v12_minus_control"].values()) < 0 for row in endpoint)
    write_frozen(args.output / "comparison.json", result)
    print("3,600-update comparison", {k: v for k, v in result.items() if k not in ("cells", "comparisons")}, flush=True)


if __name__ == "__main__":
    main()
