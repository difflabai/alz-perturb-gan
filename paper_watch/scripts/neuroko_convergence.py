"""Compare the calibrated v12 and original controls through 3,600 or 40,000 fixed-rate updates."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import copy
import json
from math import isfinite, sqrt
from pathlib import Path
from statistics import mean, stdev

from compare_neuroko_v12 import (
    GROUPS, NATIVE_OPTIMIZER_SMOOTHING, ROOT, fit, receipt, with_shared_smoothing, write_frozen,
)

STEPS = (1200, 2400, 3600)
LONG_STEPS = (*range(1200, 40000, 1200), 40000)
NOTES = ROOT / "paper_watch/notes"
METRICS = ("mean_energy_distance", "spread_ratio", "effective_rank")
SMOOTHING = NATIVE_OPTIMIZER_SMOOTHING["v12"]


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


def shared_smoothing_pilot(arms, anchors, smoothing=SMOOTHING):
    archive = json.loads((NOTES / "neuroko_v12_shared_optimizer_smoothing.json").read_text())
    study = archive["settings"][str(smoothing)]
    expected = with_shared_smoothing(arms, smoothing)
    if study["protocol"]["arms"] != expected:
        raise ValueError("Expected the published release configurations at the requested optimizer smoothing")
    factory = {role: dict(generator_and_particles=smoothing, critic=smoothing) for role in NATIVE_OPTIMIZER_SMOOTHING}
    if study["matched_optimizer_smoothing_from_native_factories"] != factory:
        raise ValueError("The matched release optimizers must share both smoothing values")
    inputs = {}
    for cohort in study["cohorts"].values():
        for row in cohort["rows"]:
            pair = row["seed"], row["group"]
            if pair in anchors:
                if any(row["energy"][k] != anchors[pair][k] for k in arms if expected[k] == arms[k]):
                    raise ValueError("Published original-release energies differ between studies")
                anchors[pair] = row["energy"]
                inputs[pair] = row["hashes"]
    if inputs.keys() != anchors.keys():
        raise ValueError("Missing 1,200-update smoothing comparison for the pilot seeds")
    return expected, inputs


def late_energy(curve):
    tail = [p for p in curve if p["updates"] >= 30000]
    area = sum((b["updates"] - a["updates"]) * (a["mean_energy_distance"] + b["mean_energy_distance"]) / 2
               for a, b in zip(tail, tail[1:]))
    return area / (tail[-1]["updates"] - tail[0]["updates"])


def run_fold(args, arms, seed, group, original, *, read_only=False, include_measurements=False):
    holdout = group == "Screen1_0"
    stage = "heldout" if holdout else "validation"
    fit_groups = set(GROUPS) if holdout else set(GROUPS) - {group}
    gpu = seed % 2 if holdout else (seed + GROUPS.index(group)) % 2
    steps = getattr(args, "checkpoints", STEPS)
    endpoint = steps[-1]
    points, fingerprints, targets, measurements = {}, {}, {}, {}
    source = getattr(args, "same_wheel_from", None)
    smoothing_source = getattr(args, "shared_smoothing_from", None)
    smoothing_replication = getattr(args, "smoothing_replication", None)
    smoothing = getattr(args, "matched_smoothing", SMOOTHING)
    training = args
    if holdout and smoothing_replication:
        # Unmodified configurations use the same deterministic fit in both confirmations.
        training = copy(args)
        training.output = args.output.parent
    if smoothing_source and steps == LONG_STEPS:
        pilot = json.loads((args.smoothing_pilot / f"curves-{seed}-{group}.json").read_text())
        if ((pilot["seed"], pilot["group"], pilot["gpu"]) != (seed, group, gpu)
                or set(pilot["curves"]) != set(arms)):
            raise ValueError("Expected the paired smoothing pilot on the same seed, group and GPU")
    for role in ("v12", "tuned_empf", "tuned_v11", "matched_empf", "matched_v11", "same_wheel_v11"):
        if role not in arms:
            continue
        arm = arms[role]
        origin = training.output
        if source and role == "v12":
            origin = source
        elif smoothing_source and (role not in NATIVE_OPTIMIZER_SMOOTHING or NATIVE_OPTIMIZER_SMOOTHING[role] == smoothing):
            origin = smoothing_source
        if not read_only and origin == training.output:
            fit(training, stage, seed, group, gpu, [arm], options=("--updates", str(endpoint), "--trace-every", "1200"))
        base = receipt(origin, stage, *arm, seed, group)
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
            if role == "v12":
                paired_settings = row["settings"]
            elif row["settings"] != paired_settings:
                raise ValueError(f"Training settings differ between the two optimizers for seed {seed}, {group}")
        if smoothing_source or smoothing_replication:
            prior = args.smoothing_inputs.get((seed, group))
            if prior and (hashes["data"], hashes["initialization"], hashes["training_batches"][0], str(gpu)) != (
                    prior["projected_data_sha256"], prior["initial_parameters_sha256"],
                    prior["training_batches_sha256"], prior["visible_devices"]):
                raise ValueError(f"Inputs differ from the published smoothing comparison for seed {seed}, {group}")
            if role in NATIVE_OPTIMIZER_SMOOTHING:
                settings = row["settings"]
                expected = None if smoothing == NATIVE_OPTIMIZER_SMOOTHING[role] else smoothing
                must_check_objects = expected is not None or "optimizer_smoothing" in row
                actual = row.get("optimizer_smoothing")
                if settings["optimizer_smoothing"] != expected or (must_check_objects and actual != {
                        "generator_and_particles": smoothing, "critic": smoothing}):
                    raise ValueError(f"Optimizer smoothing mismatch for {role}, seed {seed}, {group}")
                # EMPF has its own objective. The two transport parameters apply only to the Unified loss.
                shared = {k: v for k, v in settings.items()
                          if k not in ("rhbi_relativistic_weight", "rhbi_self_coupling", "optimizer_smoothing")}
                if role == "v12":
                    paired_settings = shared, rates
                elif (shared, rates) != paired_settings:
                    raise ValueError(f"Unmatched training settings for {role}, seed {seed}, {group}")
        if smoothing_source and steps == LONG_STEPS:
            prior = pilot["fingerprints"]
            if (trace[:len(STEPS)] != pilot["curves"][role] or hashes["data"] != prior["data"]
                    or hashes["initialization"] != prior["initialization"]
                    or hashes["training_batches"][:len(STEPS)] != prior["training_batches"]):
                raise ValueError(f"The long run differs from its shared-smoothing pilot for {role}, seed {seed}, {group}")
        elif endpoint == LONG_STEPS[-1]:
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
        if holdout or include_measurements:
            targets[role] = row["targets"]
            if (len(targets[role]) != 4 or (targets["v12"].keys() != targets[role].keys())
                    or mean(t["energy_distance"] for t in targets[role].values()) != row["means"]["energy_distance"]):
                raise ValueError(f"Inconsistent biological target scores for {role}, seed {seed}")
        if include_measurements:
            if not isfinite(row["seconds"]) or row["seconds"] <= 0:
                raise ValueError(f"Invalid training time for {role}, seed {seed}")
            measurements[role] = dict(fit_seconds=row["seconds"], fixed_rates=rates,
                                      verified_optimizer_steps=row["fixed_lr_optimizer_steps"],
                                      endpoint_targets=targets[role])
            if "optimizer_smoothing" in row:
                measurements[role]["measured_optimizer_smoothing"] = row["optimizer_smoothing"]
            if steps == LONG_STEPS:
                measurements[role]["mean_energy_30000_to_40000"] = late_energy(trace)
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
    if include_measurements:
        result["training"] = measurements
    return result


def heldout_qualification(args, arms):
    fresh = copy(args)
    fresh.output = args.confirm_from.resolve()
    replication = getattr(args, "smoothing_replication", None)
    if replication:
        fresh.shared_smoothing_from = fresh.output
        fresh.output = replication.resolve()
    fresh.checkpoints = STEPS
    plan = json.loads((fresh.output / "protocol.json").read_text())
    seeds = list(range(838, 844))
    if (plan["seeds"] != seeds or plan["groups"] != list(GROUPS) or plan["checkpoints"] != list(STEPS)
            or plan["arms"] != arms
            or plan.get("matched_optimizer_smoothing") != (getattr(args, "matched_smoothing", SMOOTHING) if replication else None)):
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
    parser.add_argument("--report-completed", action="store_true",
                        help="read completed paired curves and print verified timings, rates and target scores as JSON")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--fresh", action="store_true", help="replicate the fixed curves on seeds 838 through 843")
    mode.add_argument("--forty-thousand", action="store_true", help="extend all five pilot arms on Screen1_1 to 40,000 updates")
    mode.add_argument("--confirm-from", type=Path, metavar="FRESH_DIR",
                      help="verify the completed six-seed replication and conditionally evaluate Screen1_0")
    mode.add_argument("--same-wheel-from", type=Path, metavar="CONVERGENCE_DIR",
                      help="compare the v12 and v11 optimizers inside release 0.12.0 at the source run's endpoint")
    mode.add_argument("--shared-smoothing-from", type=Path, metavar="CONVERGENCE_DIR",
                      help="compare the original releases with matched optimizer smoothing at the source endpoint")
    parser.add_argument("--smoothing-pilot", type=Path, metavar="PILOT_DIR",
                        help="3,600-update shared-smoothing pilot to verify when extending it to 40,000")
    parser.add_argument("--smoothing-replication", type=Path, metavar="SHARED_FRESH_DIR",
                        help="qualify the matched-smoothing replication alongside the original --confirm-from directory")
    parser.add_argument("--matched-smoothing", type=float, choices=sorted(set(NATIVE_OPTIMIZER_SMOOTHING.values())),
                        help="use this release smoothing value for all three matched models (default 0.0061)")
    parser.set_defaults(require_constant_lr=True)
    args = parser.parse_args()
    if args.smoothing_replication and not args.confirm_from:
        parser.error("--smoothing-replication requires --confirm-from with the original six-seed run")
    if args.matched_smoothing is not None and not (args.shared_smoothing_from or args.smoothing_replication):
        parser.error("--matched-smoothing requires a shared-smoothing comparison or confirmation")
    if args.matched_smoothing is None:
        args.matched_smoothing = SMOOTHING
    args.data, args.wheels, args.output = (p.resolve() for p in (args.data, args.wheels, args.output))
    if not args.report_completed:
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
    comparison_source = args.same_wheel_from or args.shared_smoothing_from
    if comparison_source:
        if args.output == comparison_source.resolve():
            parser.error("use a distinct output directory for the optimizer comparison")
        plan = json.loads((comparison_source / "protocol.json").read_text())
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
    if bool(args.smoothing_pilot) != bool(args.shared_smoothing_from and args.checkpoints == LONG_STEPS):
        parser.error("--smoothing-pilot is required specifically for the 40,000-update shared-smoothing comparison")
    if args.shared_smoothing_from:
        args.shared_smoothing_from = args.shared_smoothing_from.resolve()
        smoothing = args.matched_smoothing
        arms, args.smoothing_inputs = shared_smoothing_pilot(arms, anchors, smoothing)
        extra_protocol["matched_optimizer_smoothing"] = smoothing
        reuse = ("Reuse the four original unchanged fits and train matched EMPF" if smoothing == SMOOTHING else
                 "Reuse original EMPF and the independently tuned controls and train matched v12 and original v11")
        rule = (f"Give v12, matched original v11 and matched original EMPF optimizer smoothing {smoothing}. "
                f"{reuse} on identical cells, initial "
                "weights and minibatches with fixed learning rates. Report the paired 3,600-update "
                "endpoint and the two earlier checkpoints on the pilot and fresh seeds.")
        if args.smoothing_pilot:
            args.smoothing_pilot = args.smoothing_pilot.resolve()
            pilot = json.loads((args.smoothing_pilot / "protocol.json").read_text())
            if (pilot["arms"] != arms or pilot["groups"] != list(GROUPS) or pilot["seeds"] != seeds
                    or pilot["checkpoints"] != list(STEPS) or pilot["matched_optimizer_smoothing"] != smoothing):
                raise ValueError("Expected the 3,600-update pilot with the same matched optimizer smoothing")
            adjusted_fits = ("the adjusted EMPF fits, reuse the other four configurations" if smoothing == SMOOTHING else
                             "the adjusted v12 and original v11 fits, reuse the other three configurations")
            rule = ("Compare the original releases through 40,000 updates on Screen1_1 for seeds 22 and 251. "
                    f"Set both optimizers' smoothing to {smoothing} for the three matched formulations. Train only "
                    f"{adjusted_fits} and replay all five pilot "
                    "curves through 3,600 on identical inputs. Compare the paired endpoints and the "
                    "time-weighted mean from 30,000 to 40,000 updates regardless of earlier rankings.")
    elif args.same_wheel_from:
        args.same_wheel_from = args.same_wheel_from.resolve()
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
        if args.smoothing_replication:
            native = json.loads((args.confirm_from / "protocol.json").read_text())
            if (native["arms"] != arms or native["seeds"] != list(range(838, 844))
                    or native["groups"] != list(GROUPS) or native["checkpoints"] != list(STEPS)):
                raise ValueError("Expected the original six-seed convergence comparison")
            shared_output = args.output / f"smoothing-{args.matched_smoothing}"
            if (args.output == args.smoothing_replication.resolve()
                    or shared_output.resolve() in (args.confirm_from.resolve(), args.smoothing_replication.resolve())):
                parser.error("use a distinct output directory for biological confirmation")
            arms, args.smoothing_inputs = shared_smoothing_pilot(arms, anchors, args.matched_smoothing)
            extra_protocol["matched_optimizer_smoothing"] = args.matched_smoothing
            args.output = shared_output
            if not args.report_completed:
                args.output.mkdir(exist_ok=True)
        qualification = heldout_qualification(args, arms)
        write_frozen(args.output / "qualification.json", qualification, read_only=args.report_completed)
        if not qualification["qualifies"]:
            if args.report_completed:
                print(json.dumps(dict(status="did_not_qualify", qualification=qualification), indent=2))
            else:
                print("Fresh v12 did not lead all four controls at 3,600 updates.", qualification, flush=True)
            return
        seeds = list(range(844, 860))
        groups = ("Screen1_0",)
        rule = ("After v12 beats all four original controls on mean energy across the complete fresh "
                "development comparison, train every frozen configuration on Screen1_1, Screen1_3 and "
                "Screen1_4. Score Screen1_0 at 3,600 updates for seeds 844–859. Report all paired endpoints, "
                "target scores and seed-level 95% t intervals. The 1,200 and 2,400 checkpoints are descriptive.")
        if args.smoothing_replication:
            rule += f" Use shared optimizer smoothing of {args.matched_smoothing} for the three rate-matched formulations at both stages."
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
    if args.checkpoints == LONG_STEPS and not args.shared_smoothing_from:
        archive = ("neuroko_v12_optimizer_3600_pilot.json" if args.same_wheel_from
                   else "neuroko_v12_convergence_3600_tuned_empf.json")
        pilot = json.loads((NOTES / archive).read_text())
        selected = [cell for cell in pilot["cells"] if cell["group"] in groups]
        args.pilot_anchors = {(cell["seed"], cell["group"]): cell for cell in selected}
        if (pilot["arms"] != arms or len(selected) != len(seeds) * len(groups)
                or set(args.pilot_anchors) != {(s, g) for s in seeds for g in groups}
                or any(set(c["curves"] | c.get("other_control_curves", {})) != set(arms) for c in selected)):
            raise ValueError("The long run needs a published pilot curve for every arm and seed")
    protocol = dict(arms=arms, seeds=seeds, groups=list(groups), checkpoints=args.checkpoints,
                    metric="four-target mean energy distance", rule=rule, **extra_protocol)
    write_frozen(args.output / "protocol.json", protocol, read_only=args.report_completed)
    if args.report_completed:
        cells = [run_fold(args, arms, seed, group, anchors.get((seed, group)), read_only=True, include_measurements=True)
                 for seed in seeds for group in groups if (args.output / f"curves-{seed}-{group}.json").is_file()]
        expected = len(seeds) * len(groups)
        report = dict(status="complete" if len(cells) == expected else "partial",
                      completed_pairs=len(cells), expected_pairs=expected, protocol=protocol, cells=cells)
        if len(cells) == expected:
            report["endpoint_means"] = {}
            for role in arms:
                seed_means = [mean(c["curves"][role][-1]["mean_energy_distance"] for c in cells if c["seed"] == seed)
                              for seed in seeds]
                report["endpoint_means"][role] = mean(seed_means)
            if args.checkpoints == LONG_STEPS:
                report["mean_energy_30000_to_40000"] = {
                    r: mean(c["training"][r]["mean_energy_30000_to_40000"] for c in cells) for r in arms}
        print(json.dumps(report, indent=2, allow_nan=False))
        return
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
    if args.fresh or args.forty_thousand or args.confirm_from or comparison_source:
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
            if args.confirm_from or (comparison_source and len(seeds) > 2):
                from scipy.stats import t
                margin = float(t.ppf(.975, len(gaps) - 1)) * stdev(gaps) / sqrt(len(gaps))
                summary[role]["paired_95_t_interval"] = [mean(gaps) - margin, mean(gaps) + margin]
        result["endpoint_comparisons"] = summary
        result["leads_all_controls_on_average"] = all(row["mean_v12_minus_control"] < 0 for row in summary.values())
    else:
        result["leads_both_seeds"] = all(max(row["v12_minus_control"].values()) < 0 for row in endpoint)
    if args.checkpoints == LONG_STEPS:
        result["mean_energy_30000_to_40000"] = {r: mean(late_energy(c["curves"][r]) for c in cells) for r in arms}
    write_frozen(args.output / "comparison.json", result)
    print(f"{args.checkpoints[-1]:,}-update comparison", {k: v for k, v in result.items() if k not in ("cells", "comparisons")}, flush=True)


if __name__ == "__main__":
    main()
