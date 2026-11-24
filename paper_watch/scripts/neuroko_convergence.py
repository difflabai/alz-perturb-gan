"""Compare the calibrated v12 and original controls through 3,600 or 40,000 fixed-rate updates."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import copy
import json
from math import sqrt
from pathlib import Path
from statistics import mean, stdev

from compare_neuroko_v12 import GROUPS, ROOT, fit, receipt, write_frozen

STEPS = (1200, 2400, 3600)
LONG_STEPS = (*range(1200, 40000, 1200), 40000)
NOTES = ROOT / "paper_watch/notes"
METRICS = ("mean_energy_distance", "spread_ratio", "effective_rank")


def optimizer_difference(wheels, trial):
    """Resolve both recipes against the same installed 0.12.0 wheel."""
    from dataclasses import asdict
    from importlib.metadata import version
    import sys

    wheel = (wheels / "0.12.0").resolve()
    sys.path[:0] = [p for p in (str(wheel), str(ROOT / "paper_watch")) if p not in sys.path]
    import particlegan
    from alztest.harness import TrainingSettings

    if Path(particlegan.__file__).resolve().parent != wheel / "particlegan" or version("particlegan") != "0.12.0":
        raise ValueError("The optimizer comparison requires the supplied 0.12.0 installation")
    settings = TrainingSettings(**{k: v for k, v in trial.items() if k not in ("name", "lr")})
    recipes = [asdict(particlegan.get_recipe(name, lr=trial["lr"], **settings.recipe_overrides()))
               for name in ("unified_v12", "unified_v11")]
    v12, v11 = recipes
    changes = {k: dict(v12=v12[k], same_wheel_v11=v11[k]) for k in v12 if v12[k] != v11[k]}
    if set(changes) != {"name", "optimizer_family"}:
        raise ValueError(f"Unexpected differences between the two resolved recipes: {list(changes)}")
    return changes


def run_fold(args, arms, seed, group, original, *, read_only=False):
    holdout = group == "Screen1_0"
    stage = "heldout" if holdout else "validation"
    fit_groups = set(GROUPS) if holdout else set(GROUPS) - {group}
    gpu = seed % 2 if holdout else (seed + GROUPS.index(group)) % 2
    steps = getattr(args, "checkpoints", STEPS)
    endpoint = steps[-1]
    points, fingerprints, targets = {}, {}, {}
    source = getattr(args, "same_wheel_from", None)
    for role in ("v12", "tuned_empf", "tuned_v11", "matched_empf", "matched_v11", "same_wheel_v11"):
        if role not in arms:
            continue
        arm = arms[role]
        reuse_v12 = source is not None and role == "v12"
        if not read_only and not reuse_v12:
            fit(args, stage, seed, group, gpu, [arm], options=("--updates", str(endpoint), "--trace-every", "1200"))
        base = receipt(source if reuse_v12 else args.output, stage, *arm, seed, group)
        row = json.loads(base.with_stem(base.stem + f"-u{endpoint}-trace1200").read_text())
        trace = row["trace"]
        rates = row["initial_rates"]
        trial = arm[2]
        expected_rates = dict(generator=trial["lr"], critic=trial["lr"] * trial["d_lr_mult"],
                              particles=trial["lr"] * trial["prior_lr_mult"])
        if (rates != expected_rates or (row["particlegan"], row["recipe"]) != tuple(arm[:2])
                or row["stage"] != stage or row["score_groups"] != [group] or set(row["fit_groups"]) != fit_groups
                or row["name"] != trial["name"] or row["lr"] != trial["lr"]
                or any(row["settings"][k] != v for k, v in trial.items() if k not in ("lr", "name"))
                or row["seed"] != seed or not row["deterministic"]
                or row["updates"] != endpoint or row["constant_lr_verified_updates"] != endpoint
                or row["fixed_lr_optimizer_steps"] != {"generator_and_particles": endpoint, "critic": endpoint}
                or row["final_rates"] != rates or row["visible_devices"] != str(gpu)
                or [p["updates"] for p in trace] != list(steps)
                or (original is not None and trace[0]["mean_energy_distance"] != original[role])
                or trace[-1]["mean_energy_distance"] != row["means"]["energy_distance"]
                or trace[-1]["training_batches_sha256"] != row["training_batches_sha256"]):
            raise ValueError(f"Invalid fixed-rate curve or 1,200-update replay for {role}, seed {seed}, {group}")
        hashes = dict(data=row["projected_data_sha256"], initialization=row["initial_parameters_sha256"],
                      training_batches=[p["training_batches_sha256"] for p in trace])
        if source is not None:
            if row["optimizer_family"] != args.recipe_differences["optimizer_family"][role]:
                raise ValueError(f"Unexpected optimizer in {role}, seed {seed}, {group}")
            if reuse_v12:
                paired_settings = row["settings"]
            elif row["settings"] != paired_settings:
                raise ValueError(f"Training settings differ between the two optimizers for seed {seed}, {group}")
        if endpoint == LONG_STEPS[-1]:
            anchor = args.pilot_anchors[seed, group]
            prefix = [dict(updates=p["updates"], **{m: p[m] for m in METRICS}) for p in trace[:len(STEPS)]]
            prior = anchor["fingerprints"]
            if (prefix != (anchor["curves"] | anchor.get("other_control_curves", {}))[role]
                    or hashes["data"] != prior["projected_data"] or hashes["initialization"] != prior["initial_parameters"]
                    or hashes["training_batches"][:len(STEPS)] != prior["training_batches"]):
                raise ValueError(f"The long run differs from its published 3,600-update pilot for {role}, seed {seed}, {group}")
        if fingerprints and hashes != fingerprints:
            raise ValueError("Curves must use identical projected cells, initialization and training batches")
        fingerprints = hashes
        points[role] = trace
        if holdout:
            targets[role] = row["targets"]
            if (len(targets[role]) != 4 or (targets["v12"].keys() != targets[role].keys())
                    or mean(t["energy_distance"] for t in targets[role].values()) != row["means"]["energy_distance"]):
                raise ValueError(f"Inconsistent biological target scores for {role}, seed {seed}")
        if not read_only:
            print(seed, group, role, [round(p["mean_energy_distance"], 5) for p in trace], flush=True)
    result = dict(seed=seed, group=group, gpu=gpu, curves=points, fingerprints=fingerprints)
    if holdout:
        result["endpoint_targets"] = targets
    path = args.output / f"curves-{seed}-{group}.json"
    if read_only:
        if json.loads(path.read_text()) != result:
            raise ValueError(f"Archived curves differ from the optimizer receipts for seed {seed}, {group}")
    else:
        write_frozen(path, result)
    return result


def heldout_qualification(args, arms):
    fresh = copy(args)
    fresh.output = args.confirm_from.resolve()
    fresh.checkpoints = STEPS
    plan = json.loads((fresh.output / "protocol.json").read_text())
    seeds = list(range(838, 844))
    if (plan["seeds"] != seeds or plan["groups"] != list(GROUPS) or plan["checkpoints"] != list(STEPS)
            or plan["arms"] != arms):
        raise ValueError("Biological confirmation requires the frozen six-seed development comparison")
    report = json.loads((fresh.output / "comparison.json").read_text())
    cells = [run_fold(fresh, arms, seed, group, None, read_only=True) for seed in seeds for group in GROUPS]
    if cells != report["cells"]:
        raise ValueError("Fresh comparison differs from its 18 verified group–seed pairs")
    by_seed = [{role: mean(cell["curves"][role][-1]["mean_energy_distance"] for cell in cells if cell["seed"] == seed)
                for role in arms} for seed in seeds]
    gaps = {role: mean(s["v12"] - s[role] for s in by_seed) for role in arms if role != "v12"}
    passes = all(gap < 0 for gap in gaps.values())
    if (passes != report["leads_all_controls_on_average"]
            or any(gap != report["endpoint_comparisons"][role]["mean_v12_minus_control"] for role, gap in gaps.items())):
        raise ValueError("Fresh endpoint ranking differs from its optimizer receipts")
    return dict(seeds=seeds, groups=list(GROUPS), updates=STEPS[-1], v12_minus_control=gaps, qualifies=passes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("data", "wheels", "output"):
        parser.add_argument("--" + field, type=Path, required=True)
    parser.add_argument("--window-cells", type=int)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--fresh", action="store_true", help="replicate the fixed curves on seeds 838 through 843")
    mode.add_argument("--forty-thousand", action="store_true", help="extend all five pilot arms on Screen1_1 to 40,000 updates")
    mode.add_argument("--confirm-from", type=Path, metavar="FRESH_DIR",
                      help="verify the completed six-seed replication and conditionally evaluate Screen1_0")
    mode.add_argument("--same-wheel-from", type=Path, metavar="CONVERGENCE_DIR",
                      help="compare the v12 and v11 optimizers inside release 0.12.0 at the source run's endpoint")
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
    args.checkpoints = LONG_STEPS if args.forty_thousand else STEPS
    groups = GROUPS
    extra_protocol = {}
    if args.same_wheel_from:
        args.same_wheel_from = args.same_wheel_from.resolve()
        if args.output == args.same_wheel_from:
            parser.error("use a distinct output directory for the optimizer comparison")
        plan = json.loads((args.same_wheel_from / "protocol.json").read_text())
        seeds = plan["seeds"]
        if plan["checkpoints"] == list(LONG_STEPS):
            args.checkpoints = LONG_STEPS
            groups = GROUPS[:1]
            valid_seeds = seeds == [22, 251]
        else:
            valid_seeds = seeds in ([22, 251], list(range(838, 844)))
        if (plan["arms"] != arms or plan["groups"] != list(groups)
                or plan["checkpoints"] != list(args.checkpoints) or not valid_seeds):
            raise ValueError("Expected a frozen five-arm NeuroKO convergence comparison")
        v12 = arms["v12"]
        arms = dict(v12=v12, same_wheel_v11=[v12[0], "unified_v11", v12[2]])
        args.recipe_differences = optimizer_difference(args.wheels, v12[2])
        extra_protocol["resolved_recipe_differences"] = args.recipe_differences
        if seeds == [22, 251]:
            for seed, file, section in ((22, "fixed_critic_matched_development", "comparison"),
                                        (251, "fixed_critic_second_originals", "result")):
                published = json.loads((NOTES / f"neuroko_v12_{file}.json").read_text())
                name = f"{v12[0]}/unified_v11/{v12[2]['name']}"
                scores = published[section]["scores"][name]
                if published["protocol"].get("seeds", published["protocol"].get("development"))[0] != seed:
                    raise ValueError("Unexpected published 1,200-update optimizer comparison")
                for i, group in enumerate(groups):
                    anchors[seed, group]["same_wheel_v11"] = scores[i]
        rule = ("Reuse the original v12 fits and train the v11 optimizer from the same wheel on identical "
                "cells, weights, batches and fixed rates. Compare their paired mean energy at 3,600 updates "
                "and show the 1,200 and 2,400 checkpoints. Use both pilot seeds and replicate on seeds "
                "838–843 regardless of the pilot ordering.")
        if args.checkpoints == LONG_STEPS:
            rule = ("Reuse the 40,000-update v12 fits and train v11 from the same wheel on Screen1_1 for "
                    "both pilot seeds irrespective of earlier scores. Match cells, initialization, minibatches "
                    "and fixed learning rates. Replay the first three published checkpoints for each optimizer. "
                    "Report paired endpoint energies and the time-weighted mean from 30,000 to 40,000 updates.")
    elif args.confirm_from:
        if args.output == args.confirm_from.resolve():
            parser.error("use a distinct output directory for biological confirmation")
        qualification = heldout_qualification(args, arms)
        write_frozen(args.output / "qualification.json", qualification)
        if not qualification["qualifies"]:
            print("Fresh v12 did not lead all four controls at 3,600 updates.", qualification, flush=True)
            return
        seeds = list(range(844, 860))
        groups = ("Screen1_0",)
        rule = ("After v12 beats all four original controls on mean energy across the complete fresh "
                "development comparison, train every frozen configuration on Screen1_1, Screen1_3 and "
                "Screen1_4. Score Screen1_0 at 3,600 updates for seeds 844–859. Report all paired endpoints, "
                "target scores and seed-level 95% t intervals. The 1,200 and 2,400 checkpoints are descriptive.")
    elif args.forty_thousand:
        groups = GROUPS[:1]
        seeds = [22, 251]
        rule = ("Run all five frozen configurations on the first listed development group Screen1_1 "
                "and both pilot seeds through 40,000 optimizer updates. Reproduce the first three "
                "published checkpoints and their minibatch hashes. Compare the two-seed mean energy "
                "at 40,000 with each control and report the time-weighted mean energy over updates "
                "30,000 through 40,000 using linear interpolation between checkpoints.")
    elif args.fresh:
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
    if args.checkpoints == LONG_STEPS:
        archive = ("neuroko_v12_optimizer_3600_pilot.json" if args.same_wheel_from
                   else "neuroko_v12_convergence_3600_tuned_empf.json")
        pilot = json.loads((NOTES / archive).read_text())
        selected = [cell for cell in pilot["cells"] if cell["group"] in groups]
        args.pilot_anchors = {(cell["seed"], cell["group"]): cell for cell in selected}
        if (pilot["arms"] != arms or len(selected) != len(seeds) * len(groups)
                or set(args.pilot_anchors) != {(s, g) for s in seeds for g in groups}
                or any(set(c["curves"] | c.get("other_control_curves", {})) != set(arms) for c in selected)):
            raise ValueError("The long run needs a published pilot curve for every arm and seed")
    write_frozen(args.output / "protocol.json", dict(arms=arms, seeds=seeds, groups=list(groups),
                 checkpoints=args.checkpoints, metric="four-target mean energy distance", rule=rule, **extra_protocol))
    with ThreadPoolExecutor(1) as first, ThreadPoolExecutor(1) as second:
        jobs = []
        for seed in seeds:
            for i, group in enumerate(groups):
                jobs.append((first, second)[(i + seed) % 2].submit(run_fold, args, arms, seed, group,
                                                                   anchors.get((seed, group))))
        cells = [job.result() for job in jobs]
    comparisons = []
    for seed in seeds:
        cohort = [cell for cell in cells if cell["seed"] == seed]
        for i, step in enumerate(args.checkpoints):
            means = {role: {metric: mean(cell["curves"][role][i][metric] for cell in cohort) for metric in METRICS}
                     for role in arms}
            energy = means["v12"]["mean_energy_distance"]
            gaps = {role: energy - values["mean_energy_distance"] for role, values in means.items() if role != "v12"}
            comparisons.append(dict(seed=seed, updates=step, means=means, v12_minus_control=gaps))
    endpoint = [row for row in comparisons if row["updates"] == args.checkpoints[-1]]
    result = dict(cells=cells, comparisons=comparisons)
    if args.fresh or args.forty_thousand or args.confirm_from or args.same_wheel_from:
        result["endpoint_means"] = {role: mean(row["means"][role]["mean_energy_distance"] for row in endpoint)
                                    for role in arms}
        summary = {}
        for role in arms:
            if role == "v12":
                continue
            gaps = [row["v12_minus_control"][role] for row in endpoint]
            wins = sum(cell["curves"]["v12"][-1]["mean_energy_distance"] < cell["curves"][role][-1]["mean_energy_distance"]
                       for cell in cells)
            summary[role] = dict(mean_v12_minus_control=mean(gaps), pairs_won=wins,
                                 seed_means_won=sum(gap < 0 for gap in gaps))
            if args.confirm_from or (args.same_wheel_from and len(seeds) > 2):
                from scipy.stats import t
                margin = float(t.ppf(.975, len(gaps) - 1)) * stdev(gaps) / sqrt(len(gaps))
                summary[role]["paired_95_t_interval"] = [mean(gaps) - margin, mean(gaps) + margin]
        result["endpoint_comparisons"] = summary
        result["leads_all_controls_on_average"] = all(row["mean_v12_minus_control"] < 0 for row in summary.values())
    else:
        result["leads_both_seeds"] = all(max(row["v12_minus_control"].values()) < 0 for row in endpoint)
    if args.checkpoints == LONG_STEPS:
        late = {}
        for role in arms:
            seed_means = []
            for cell in cells:
                tail = [p for p in cell["curves"][role] if p["updates"] >= 30000]
                area = sum((b["updates"] - a["updates"]) * (a["mean_energy_distance"] + b["mean_energy_distance"]) / 2
                           for a, b in zip(tail, tail[1:]))
                seed_means.append(area / (tail[-1]["updates"] - tail[0]["updates"]))
            late[role] = mean(seed_means)
        result["mean_energy_30000_to_40000"] = late
    write_frozen(args.output / "comparison.json", result)
    print(f"{args.checkpoints[-1]:,}-update comparison", {k: v for k, v in result.items() if k not in ("cells", "comparisons")}, flush=True)


if __name__ == "__main__":
    main()
