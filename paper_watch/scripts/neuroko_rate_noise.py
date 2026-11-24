"""Select a rate and latent-noise combination using paired comparisons across three seeds."""
import argparse
from copy import copy
import json
from math import exp, isfinite, log, sqrt
from pathlib import Path
from statistics import mean, median

from compare_neuroko_v12 import GROUPS, ROOT, receipt, reusable_fit, write_frozen
from neuroko_matched_rate_replication import (
    SEEDS as REPLICATION_SEEDS, TRIAL as MATCHED_RATE_TRIAL, comparison, original_release_scores,
)
from neuroko_role_factorial import REFERENCES, TRIALS as RATES, paired
from neuroko_stratified import NOISY, confirm_candidate, control_arms, evaluate, key
from neuroko_transport_scale import TRIALS as TRANSPORT, validate_finalist
from retune_neuroko_v12 import BASE, RELEASE, V12

RATE_NAMES = ("g-high-d-high-p-low", "g-low-d-low-p-high")
NOISY_PRIOR = dict(NOISY, name="noise150-prior100", d_lr_mult=1.5, prior_lr_mult=2.9, prior_reg=0.1)
WIDE_NOISY_PRIOR = dict(NOISY_PRIOR, name="noise250-prior100", prior_sigma=.25)
MATCHED_PRIOR = dict(MATCHED_RATE_TRIAL, name="matched-rates-prior100", prior_reg=0.1)
MOTION_STUDIES = ("critic-motion", "matched-critic-motion", "balanced-particle-prior")
ROLE_INDEPENDENT_SEEDS = list(range(290, 296))
ROLE_OUTER_SEEDS = list(range(396, 412))
PRELIMINARY_STUDIES = ("original-v11-critic", "original-v11-critic-wide-noise", "generator-output-noise")
TWO_COHORT_STUDIES = ("strong-prior-noise", "individual-role-rates", "prior-target-spread", *PRELIMINARY_STUDIES, *MOTION_STUDIES)
TRIALS = {
    f"{name}-sigma{sigma}": dict(RATES[name], name=f"{name}-sigma{sigma}", prior_sigma=sigma / 1000.)
    for name in RATE_NAMES for sigma in (75, 150)
}
GROUPS_BY_SEED = {19: GROUPS[:2], 20: GROUPS[1:], 21: (GROUPS[2], GROUPS[0])}
RULE = (
    "Cross the two leading seed-15 v12 rate combinations with Gaussian latent sigma 0.075 or 0.15. "
    "Score each v12, v11 and EMPF at identical rates and noise on six training fold–seed pairs. "
    "Each of three training groups and each of seeds 19–21 occurs twice. For each candidate subtract "
    "the lowest mean of its matched v11, matched EMPF, centered v11 and independently tuned EMPF. "
    "Choose the smallest difference and always replicate that candidate on all three training folds "
    "at new seeds 22–24. Break ties by lower v12 energy and then name. For an outer comparison on "
    "seeds 164–179 require the nine-cell v12 mean to beat all matched and established references "
    "and v12 at the same rates with the original noise 0.025. Training and evaluation share the "
    "specified noise across formulations. Keep the network, particle shape, draws, data, training "
    "budget of 1200 updates and constant role rates fixed."
)


def original_noise(trial):
    return dict(trial, name="same-rates-sigma025", prior_sigma=0.025)


def critic_motion_probe(path, v12_critic_lr):
    measurement = json.loads(path.read_text())
    spec, cells = measurement["protocol"], measurement["cells"]
    expected = {(s, g) for s in REPLICATION_SEEDS for g in GROUPS}
    if (spec["release"] != RELEASE or spec["settings"] != NOISY_PRIOR or spec["steps"] != 64
            or spec.get("v12_critic_lr", 0.021) != v12_critic_lr
            or spec.get("v12_prior_weight", NOISY_PRIOR["prior_reg"]) != NOISY_PRIOR["prior_reg"]
            or len(cells) != len(expected) or {(r["seed"], r["group"]) for r in cells} != expected):
        raise ValueError("Critic calibration requires the nine paired 64-update probes at the release rates")
    ratios, reference = [], {}
    for cell in cells:
        for role, lr in (("unified_v11", .021), (V12, v12_critic_lr)):
            if cell["rows"][role]["rates"] != dict(generator=.014, critic=lr, particles=.0406):
                raise ValueError("The training probe used unexpected role rates")
        v11, v12 = (cell["rows"][role]["centered_rms"] for role in ("unified_v11", V12))
        if len(v11) != 64 or len(v12) != 64 or not all(isfinite(x) and x > 0 for x in (*v11, *v12)):
            raise ValueError("Incomplete or zero critic displacement")
        signatures = [tuple(cell["rows"][role][k] for k in ("initial_parameters_sha256", "training_batches_sha256"))
                      for role in ("unified_v11", V12)]
        if signatures[0] != signatures[1]:
            raise ValueError("Critics used different initial parameters or training minibatches")
        reference[cell["seed"], cell["group"]] = (cell["projected_data_sha256"], cell["probe_indices_sha256"],
                                                   signatures[0], v11)
        ratios.append(median(v12) / median(v11))
    ratio = median(ratios)
    if ratio != measurement["summary"]["median_step_ratio"]:
        raise ValueError("The published critic ratio differs from the measured displacements")
    return ratio, reference


def calibrated_critic(path, check=None):
    ratio, reference = critic_motion_probe(path, .021)
    critic = round(0.021 / ratio, 4)
    if critic <= 0:
        raise ValueError("The measured ratio gives a nonpositive critic learning rate after rounding")
    calibration = dict(measured_median_ratio=ratio, reference_critic_lr=0.021)
    if check is not None:
        low_ratio, second_reference = critic_motion_probe(check, critic)
        if reference != second_reference:
            raise ValueError("Critic calibration requires identical v11 probes, training inputs and measurements")
        if not (low_ratio < 1 < ratio):
            raise ValueError("The two measured critic responses must bracket the v11 reference")
        calibration.update(lower_critic_lr=critic, lower_median_ratio=low_ratio)
        fraction = -log(low_ratio) / log(ratio / low_ratio)
        critic = round(exp(log(critic) + fraction * log(0.021 / critic)), 4)
    trial = dict(NOISY_PRIOR, name=f"motion-critic-{round(critic * 10000)}", d_lr_mult=critic / NOISY_PRIOR["lr"])
    return trial, dict(calibration, fixed_critic_lr=critic)


def balanced_prior(path, trial):
    result = json.loads(path.read_text())
    cells = result["cells"]
    if (result["calibrated_critic_rate"] != trial["lr"] * trial["d_lr_mult"]
            or result["original_critic_rate"] != .021 or result["original_prior_weight"] != trial["prior_reg"]
            or [(c["seed"], c["group"]) for c in cells] != [(s, g) for s in REPLICATION_SEEDS for g in GROUPS]):
        raise ValueError("Particle-gradient measurements must match the fixed critic, prior and nine training pairs")
    scales = []
    for c in cells:
        ratios = [c[name]["ratios_by_update"] for name in ("v12_original", "v12_calibrated")]
        if any(len(row) != 64 or not all(isfinite(v) and v > 0 for v in row) for row in ratios):
            raise ValueError("Incomplete paired particle gradients")
        scales.append(median(ratios[1]) / median(ratios[0]))
    scale = median(scales)
    weight = round(trial["prior_reg"] / scale, 3)
    if scale != result["median_relative_strength"] or weight != result["suggested_prior_weight"] or weight <= 0:
        raise ValueError("The measured gradient ratios disagree with the proposed prior weight")
    return dict(trial, prior_reg=weight, name=f"{trial['name']}-prior{round(weight * 1000)}"), scale


def prior_effects(arms, scores, seeds):
    grid = {}
    for arm in arms:
        if arm[1] != V12:
            continue
        trial = arm[2]
        generator = trial["lr"]
        rates = (generator, generator * trial.get("d_lr_mult", 1.5), generator * trial.get("prior_lr_mult", 2.9))
        cell = (rates, trial.get("prior_sigma", 0.025), trial.get("prior_reg", 0.02))
        values = scores[key(arm)]
        if cell in grid or len(values) != len(seeds) * len(GROUPS):
            raise ValueError(f"Duplicate or incomplete v12 comparison at {cell}")
        grid[cell] = values
    rate_sets = sorted({cell[0] for cell in grid})
    sigmas = (0.025, 0.15)
    weights = (0.02, 0.1)
    if len(rate_sets) != 2 or set(grid) != {
        (rates, sigma, weight) for rates in rate_sets for sigma in sigmas for weight in weights
    }:
        raise ValueError("Expected two rate settings crossed with two noise levels and two prior weights")
    result = []
    for rates in rate_sets:
        noise_levels = []
        for sigma in sigmas:
            weak, strong = (grid[rates, sigma, weight] for weight in weights)
            delta = [x - y for x, y in zip(strong, weak, strict=True)]
            noise_levels.append(dict(sigma=sigma, mean_energy_by_weight={"0.02": mean(weak), "0.1": mean(strong)},
                                     strong_minus_default=mean(delta), paired_differences=delta,
                                     pairs_improved=sum(d < 0 for d in delta)))
        low, high = (level["paired_differences"] for level in noise_levels)
        interaction = [x - y for x, y in zip(high, low, strict=True)]
        result.append(dict(rates=dict(zip(("generator", "critic", "particles"), rates)), noise_levels=noise_levels,
                           noise_by_weight=mean(interaction), paired_interactions=interaction))
    return dict(seeds=seeds, groups=list(GROUPS), rate_settings=result)


def compare_releases(args, trial, seeds):
    screen = json.loads((args.output / "screen.json").read_text())
    plan = json.loads((args.output / "search.json").read_text())
    if plan["arms"][0] != [RELEASE, V12, trial] or plan["independent_seeds"] != seeds:
        raise ValueError("the frozen study has a different candidate or independent seeds")
    if not screen["passes"]:
        raise ValueError("the prior-weight trial must qualify for fresh training seeds")
    if plan.get("second_screen_seeds"):
        second = args.output / "second_screen.json"
        if not second.is_file() or not json.loads(second.read_text())["passes"]:
            raise ValueError("both development cohorts must pass before fitting the original releases")
    arms, labels = control_arms([trial])
    by_key = {key(arm): arm for arm in arms}
    v12, v11, empf = (by_key[labels[n]] for n in ("selected_v12", "matched_v11", "matched_empf"))
    original_release_scores(args, args.output, (v12, v11, empf), seeds,
                            "Mean four-target energy at 1200 fixed-rate updates on the independent training folds.")


def individual_rates(args):
    """Vary G or the particles alone around the strongest prior/noise setting."""
    base = NOISY_PRIOR
    trials = {}
    for name, generator, particles in (("g010", .010, .0406), ("g020", .020, .0406),
                                       ("p0287", .014, .0287), ("p0574", .014, .0574)):
        trial = dict(base, name=f"noise150-prior100-{name}", lr=generator,
                     d_lr_mult=.021 / generator, prior_lr_mult=particles / generator)
        trials[trial["name"]] = trial
    seeds = REPLICATION_SEEDS
    protocol = dict(trials=list(trials.values()), baseline=base, screen_seeds=seeds, groups=list(GROUPS),
                    second_screen_seeds=[251, 252, 253], independent_seeds=ROLE_INDEPENDENT_SEEDS,
                    outer_seeds=ROLE_OUTER_SEEDS,
                    rule="Change one absolute role rate per trial. Choose the lowest nine-pair v12 mean "
                         "on seeds 22–24, breaking ties by name. Stop if the winner fails to improve the "
                         "fixed 0.014/0.021/0.0406 v12 baseline. Otherwise test only this candidate against "
                         "matched and established controls separately on both development cohorts. "
                         "Test fresh seeds with the original releases before a qualifying biological holdout.")
    write_frozen(args.output / "role_grid.json", protocol)
    shortlist = args.output / "role_shortlist.json"
    if args.stage == "search":
        screen_args = copy(args)
        if args.shared_screen is not None:
            screen_args.output = args.shared_screen.resolve()
        arms = [[RELEASE, V12, trial] for trial in (base, *trials.values())]
        scores = evaluate(screen_args, arms, seeds)
        means = {arm[2]["name"]: mean(scores[key(arm)]) for arm in arms}
        name = min(trials, key=lambda n: (means[n], n))
        write_frozen(shortlist, dict(scores=scores, means=means, chosen=name, passes=means[name] < means[base["name"]]))
    selected = json.loads(shortlist.read_text())
    if not selected["passes"]:
        if args.stage != "search":
            raise ValueError("No individual role rate passed the fixed v12 baseline")
        write_frozen(args.output / "selection.json", dict(shortlist=selected, advances=False, winner=None))
        return None
    return trials[selected["chosen"]]


def compare_noise_controls(args):
    """Compare the role-rate candidates with the stronger noise-0.25 v11 and EMPF controls."""
    grid = json.loads((args.output / "role_grid.json").read_text())
    controls = paired(WIDE_NOISY_PRIOR)[1:]
    baseline = [RELEASE, V12, NOISY_PRIOR]
    trials = [[RELEASE, V12, t] for t in grid["trials"]]
    protocol = dict(controls=controls, baseline=baseline, trials=trials,
                    first=grid["screen_seeds"], second=grid["second_screen_seeds"],
                    rule="The completed noise-0.25 comparison identified stronger v11 and EMPF controls "
                         "during the role-rate search. Score these two frozen controls on both development "
                         "cohorts. Compare every first-cohort role-rate score to them. On the second cohort "
                         "compare the chosen rate only if its nine runs exist under the original protocol. "
                         "Report the additional comparisons separately from the original selection.")
    write_frozen(args.output / "wide_noise_control_protocol.json", protocol)
    shortlist = args.output / "role_shortlist.json"
    for cohort, shared in (("first", args.shared_screen), ("second", args.second_screen)):
        seeds = protocol[cohort]
        local = copy(args)
        if shared is not None:
            local.output = shared.resolve()
        reference_arms = [baseline, *controls]
        reference_scores = evaluate(local, reference_arms, seeds)
        means = {k: mean(v) for k, v in reference_scores.items()}
        write_frozen(args.output / f"wide_noise_{cohort}_controls.json", dict(scores=reference_scores, means=means))
        if not shortlist.is_file():
            continue
        ranking = json.loads(shortlist.read_text())
        chosen = [arm for arm in trials if arm[2]["name"] == ranking["chosen"]]
        candidates = trials if cohort == "first" else chosen
        if cohort == "second" and (not ranking["passes"] or not all(
            reusable_fit(receipt(local.output, "validation", *chosen[0], seed, group), True)
            for seed in seeds for group in GROUPS
        )):
            continue
        scores = evaluate(local, [*reference_arms, *candidates], seeds)
        if cohort == "first" and any(ranking["scores"][key(arm)] != scores[key(arm)] for arm in [baseline, *trials]):
            raise ValueError("Role-rate scores differ from the frozen shortlist")
        differences = {}
        for arm in candidates:
            values = scores[key(arm)]
            gaps = {key(c): [a - b for a, b in zip(values, scores[key(c)], strict=True)] for c in controls}
            differences[key(arm)] = {k: dict(mean=mean(v), pairs_won=sum(d < 0 for d in v)) for k, v in gaps.items()}
        write_frozen(args.output / f"wide_noise_{cohort}_rates.json", dict(scores=scores, differences=differences))


def original_v11_critic_rate(path):
    data = json.loads(path.read_text())
    cells = data["cells"]
    if (data["original_v11_release"] != "0.11.1" or data["v12_release"] != RELEASE
            or data["settings"] != NOISY_PRIOR or data["original_v11_critic_lr"] != .021
            or [(c["seed"], c["group"]) for c in cells] != [(s, g) for s in REPLICATION_SEEDS for g in GROUPS]):
        raise ValueError("Expected the nine original v11 probes at the reference settings")
    ratios = {}
    for rate in (.0035, .0046, .021):
        observations = [c["v12_to_original_by_critic_lr"][str(rate)] for c in cells]
        if not all(isfinite(v) and v > 0 for v in observations):
            raise ValueError("Invalid original-v11 critic movement")
        ratios[rate] = median(observations)
        if ratios[rate] != data["median_ratios"][str(rate)]:
            raise ValueError("Recorded critic ratios differ from the measurements")
    low, high = .0046, .021
    if not ratios[.0035] < ratios[low] < 1 < ratios[high]:
        raise ValueError("The two nearest critic rates must bracket the original v11 movement")
    fraction = -log(ratios[low]) / log(ratios[high] / ratios[low])
    critic = round(exp(log(low) + fraction * log(high / low)), 4)
    return dict(NOISY_PRIOR, name=f"original-v11-critic-{round(critic * 10000)}",
                d_lr_mult=critic / NOISY_PRIOR["lr"]), ratios


def critic_references(args):
    lower, _ = calibrated_critic(args.motion_result)
    middle, _ = calibrated_critic(args.motion_result, args.motion_check)
    balanced, _ = balanced_prior(args.gradient_result, middle)
    return [(ref["name"] + "-" + arm[1], arm) for ref in (lower, middle, balanced) for arm in paired(ref)]


def spread_target(path):
    measurement = json.loads(path.read_text())
    name = key([RELEASE, V12, NOISY_PRIOR])
    rows = measurement["screen_pairs"]
    if [(row["seed"], row["group"]) for row in rows] != [(s, g) for s in REPLICATION_SEEDS for g in GROUPS]:
        raise ValueError("Expected the nine matched training pairs for the prior target")
    spreads = [r["arms"][name]["spread"] for r in rows]
    if not all(isfinite(v) and v > 0 for v in spreads):
        raise ValueError("Expected finite positive spread ratios")
    average = mean(spreads)
    if average != measurement["screen_means"][name]["spread"]:
        raise ValueError("The published mean spread does not match the nine pairs")
    target = round(1. / average, 2)
    return dict(NOISY_PRIOR, name=f"noise150-prior100-target{round(target * 100)}",
                prior_reg_target_std=target), average


def replicate_calibrated_releases(args):
    if args.study not in ("balanced-particle-prior", "original-v11-critic"):
        raise ValueError("The matched replication requires a calibrated prior or critic study")
    search = json.loads((args.output / "search.json").read_text())
    trial = search["arms"][0][2]
    arms = paired(trial)
    if search["arms"][:3] != arms or search["screen_seeds"] != REPLICATION_SEEDS:
        raise ValueError("The calibrated study must contain all three matched formulations")
    selected = key(arms[0])
    root = args.output / "matched-releases"
    root.mkdir(exist_ok=True)
    fresh = list(range(452, 458))
    rule = ("Reproduce the calibrated-prior setting on the nine original development pairs with all "
            "five formulation and release combinations. Only if v12 has the lowest mean across all five, "
            "repeat all five on six fresh training seeds. This comparison measures the matched formulation "
            "effect and does not change the original qualification against tuned controls.")
    if args.study == "original-v11-critic":
        fresh = list(range(524, 530))
        rule = ("At the fixed critic rate calibrated against original v11, check every release at identical "
                "rates, noise 0.15 and prior weight 0.1 on the nine development pairs. The preceding v12 "
                "run must have improved over its original critic rate. Replicate all five on six fresh "
                "training seeds only if v12 leads every release at these shared settings.")
    write_frozen(root / "protocol.json", dict(arms=arms, original_releases=["0.10.0", "0.11.1"],
                                               groups=GROUPS, development=REPLICATION_SEEDS, fresh=fresh, rule=rule))
    if args.study == "original-v11-critic":
        screen = json.loads((args.output / "preliminary.json").read_text())["scores"]
        if mean(screen[selected]) >= mean(screen[key(paired(NOISY_PRIOR)[0])]):
            raise ValueError("The calibrated v12 must first improve on its original critic rate")
        published = {selected: screen[selected]}
    else:
        screen = json.loads((args.output / "screen.json").read_text())["scores"]
        published = {key(arm): screen[key(arm)] for arm in arms}
        if any(mean(published[selected]) >= mean(published[key(arm)]) for arm in arms[1:]):
            raise ValueError("The calibrated v12 must first beat both same-wheel matched controls")
    for cohort, seeds in (("development", REPLICATION_SEEDS), ("fresh", fresh)):
        target = root / cohort
        target.mkdir(exist_ok=True)
        local = copy(args)
        local.output = (args.shared_screen or args.output) if cohort == "development" else root
        summaries = original_release_scores(local, target, arms, seeds, rule,
                                            published=published if cohort == "development" else None)
        lowest = all(summaries[selected]["energy"] < summary["energy"]
                     for name, summary in summaries.items() if name != selected)
        write_frozen(target / "decision.json", dict(v12_lowest_matched_mean=lowest, summaries=summaries))
        if not lowest:
            break


def calibrated_output_noise(path):
    measurement = json.loads(path.read_text())
    plan = measurement["protocol"]
    pairs = [(22, "Screen1_1"), (24, "Screen1_3"), (23, "Screen1_4")]
    cells = measurement["cells"]
    if (plan["lower"] != paired(NOISY_PRIOR) or plan["updates"] != 1200 or plan["repeats"] != 32
            or [(c["seed"], c["group"]) for c in cells] != pairs):
        raise ValueError("Expected the three paired baseline particle panels")
    excess = []
    for cell in cells:
        panel = cell["measurements"][V12]["0.15"]["panel"]
        if panel["particles"] != 64 or panel["repeats"] != 32 or len(panel["targets"]) != 4:
            raise ValueError("The output variance calibration needs 64 particle rows and all four targets")
        for target in panel["targets"].values():
            if not all(isfinite(target[k]) and target[k] > 0 for k in ("total", "observed")):
                raise ValueError("Invalid projected variance")
            excess.append((target["total"] - target["observed"]) / 30)
    variance = mean(excess)
    if variance <= 0:
        raise ValueError("The baseline has no positive mean excess variance")
    sigma = round(sqrt(variance), 2)
    return dict(NOISY_PRIOR, name=f"noise150-prior100-output{round(sigma * 100)}", output_noise_std=sigma), variance


def compare_prior_weight(args):
    screen_only = []
    pinned = []
    calibration = None
    if args.study == "matched-prior-weight":
        trial = MATCHED_PRIOR
        baseline = MATCHED_RATE_TRIAL
        references = [(ref["name"], [RELEASE, V12, ref]) for ref in (MATCHED_RATE_TRIAL, NOISY_PRIOR)]
        independent_seeds = [254, 255, 256]
        outer_seeds = list(range(228, 244))
        rule = (
            "At the independently replicated rates 0.020, 0.042 and 0.0203 and noise 0.025, change "
            "only the native prior weight to 0.1 in the matched v12, v11 and EMPF fits. On all three "
            "training folds at seeds 22–24 require v12 to beat both matched formulations and v12 at "
            "the same rates and noise with default weight 0.02. Report the preceding noise-0.15 "
            "weight-0.1 candidate as well. If the screen passes, require a lower v12 mean than every "
            "matched and established reference, including both preceding v12 settings, on fresh "
            "training seeds 254–256 before scoring the biological holdout on seeds 228–243."
        )
    elif args.study == "double-prior-weight":
        trial = dict(NOISY_PRIOR, name="noise150-prior200", prior_reg=0.2)
        baseline = NOISY_PRIOR
        references = [("prior100-" + arm[1], arm) for arm in paired(NOISY_PRIOR)]
        independent_seeds = [257, 258, 259]
        outer_seeds = list(range(300, 316))
        rule = (
            "At fixed generator/critic/particle rates 0.014/0.021/0.0406 and latent noise 0.15, "
            "double the prior weight from 0.1 to 0.2 in all three formulations. Require lower mean "
            "v12 energy than the two matched weight-0.2 controls and all three weight-0.1 models "
            "on training seeds 22–24. If this passes, require the frozen v12 to beat all matched "
            "and established references, including all three weight-0.1 models, on fresh training "
            "seeds 257–259 before scoring the biological holdout on seeds 300–315."
        )
    elif args.study == "prior-interaction":
        baseline = TRIALS["g-high-d-high-p-low-sigma150"]
        trial = dict(baseline, name="matched-rates-noise150-prior100", prior_reg=0.1)
        references = [("release-prior100-" + arm[1], arm) for arm in paired(NOISY_PRIOR)]
        references.append(("same-rates-default-prior", [RELEASE, V12, baseline]))
        release_low_noise = dict(BASE, name="previous-regularized", prior_reg=0.1)
        screen_only = [[RELEASE, V12, ref] for ref in (release_low_noise, MATCHED_RATE_TRIAL, MATCHED_PRIOR)]
        independent_seeds = list(range(260, 266))
        outer_seeds = list(range(316, 332))
        rule = (
            "Complete the eight v12 combinations of two fixed role-rate sets, latent noise 0.025 or "
            "0.15, and prior weight 0.02 or 0.1 on the same nine training pairs at seeds 22–24. "
            "Only the new combination of rates 0.020/0.042/0.0203, noise 0.15 and weight 0.1 can "
            "advance. It must beat v12 at those rates and noise with weight 0.02 and every v11 and "
            "EMPF arm in the screen, including its weight-matched controls. On fresh training seeds "
            "260–265 it must beat every reference, including all three release-rate models at noise "
            "0.15 and weight 0.1. Only then score the biological holdout on seeds 316–331."
        )
    elif args.study == "generator-output-noise":
        baseline = NOISY_PRIOR
        trial, variance = calibrated_output_noise(args.particle_spread_result)
        write_frozen(args.output / "output_noise_calibration.json",
                     dict(mean_excess_variance_per_component=variance, fixed_output_noise_std=trial["output_noise_std"]))
        references = [(ref["name"] + "-" + arm[1], arm)
                      for ref in (NOISY_PRIOR, WIDE_NOISY_PRIOR) for arm in paired(ref)]
        references.extend(critic_references(args))
        independent_seeds = list(range(480, 486))
        outer_seeds = list(range(486, 502))
        rule = ("Use the square root of the mean baseline excess variance per projected coordinate, rounded "
                "to 0.01, as the fixed training-only generator output noise. Use native clean sampling at "
                "evaluation. Keep latent noise 0.15, prior weight 0.1 and constant role learning rates. "
                "On seeds 22–24 first require a lower v12 mean than baseline v12 and both noise-0.25 controls. "
                "If it passes, compare against all matched and previously tuned controls on 22–24 and 251–253. "
                "Reserve fresh seeds 480–485 including original releases, then biological holdout seeds 486–501.")
    elif args.study in ("original-v11-critic", "original-v11-critic-wide-noise"):
        baseline = NOISY_PRIOR
        trial, ratios = original_v11_critic_rate(args.original_v11_motion)
        write_frozen(args.output / "original_critic_calibration.json",
                     dict(median_ratios=ratios, fixed_critic_lr=trial["lr"] * trial["d_lr_mult"]))
        references = [(ref["name"] + "-" + arm[1], arm)
                      for ref in (NOISY_PRIOR, WIDE_NOISY_PRIOR) for arm in paired(ref)]
        references.extend(critic_references(args))
        independent_seeds = list(range(458, 464))
        outer_seeds = list(range(464, 480))
        rule = ("At the published role rates with noise 0.15 and prior weight 0.1, match the "
                "original 0.11.1 v11 critic's median centered output displacement by log-interpolating "
                "the two bracketing v12 rates, rounding to 0.0001. First compare this one v12 rate on "
                "nine training pairs to the baseline and both stronger noise-0.25 controls. Only after "
                "a lower mean than all three, fit the matched controls and test all established arms on "
                "22–24 and 251–253 separately. Fresh seeds 458–463 include the two original releases. "
                "Reserve biological holdout seeds 464–479 for a passing candidate.")
        if args.study == "original-v11-critic-wide-noise":
            baseline = trial
            references.append(("same-critic-noise150", [RELEASE, V12, baseline]))
            trial = dict(baseline, name=baseline["name"] + "-sigma250", prior_sigma=WIDE_NOISY_PRIOR["prior_sigma"])
            independent_seeds = list(range(502, 508))
            outer_seeds = list(range(508, 524))
            rule = ("At the original-v11 calibrated fixed critic rate change latent noise from 0.15 to 0.25. "
                    "Keep generator and particle rates, prior weight, architecture, and evaluation draws fixed. "
                    "On seeds 22–24 require lower v12 energy than noise 0.15 at the same critic rate and both "
                    "tuned noise-0.25 controls at the release rates. Then compare all matched and established "
                    "arms separately on 22–24 and 251–253. If it passes, replicate on seeds 502–507 including "
                    "both original releases before assessing biological holdout seeds 508–523.")
    elif args.study == "prior-target-spread":
        baseline = NOISY_PRIOR
        trial, spread = spread_target(args.spread_result)
        write_frozen(args.output / "spread_calibration.json",
                     dict(mean_observed_spread_ratio=spread, prior_target_std=trial["prior_reg_target_std"]))
        references = [(ref["name"] + "-" + arm[1], arm)
                      for ref in (NOISY_PRIOR, WIDE_NOISY_PRIOR) for arm in paired(ref)]
        references.extend(critic_references(args))
        independent_seeds = list(range(430, 436))
        outer_seeds = list(range(436, 452))
        rule = ("Set the native particle prior's target standard deviation to the reciprocal of "
                "the mean generated-to-observed spread ratio of the baseline v12, rounded to 0.01. "
                "Use the same target, noise 0.15, penalty weight 0.1 and constant role rates in all "
                "three formulations. Require lower mean four-target energy after 1200 updates than "
                "all matched and established references on 22–24 and 251–253 separately, including "
                "the controls at noise 0.25. Qualify on fresh seeds 430–435 with the two original "
                "releases before evaluating biological holdout seeds 436–451.")
    elif args.study == "individual-role-rates":
        trial = individual_rates(args)
        if trial is None:
            return
        baseline = NOISY_PRIOR
        references = [("original-rate-" + arm[1], arm) for arm in paired(baseline)]
        references.extend(critic_references(args))
        independent_seeds = ROLE_INDEPENDENT_SEEDS
        outer_seeds = ROLE_OUTER_SEEDS
        rule = ("Take the single v12 configuration selected in role_shortlist.json. With critic LR 0.021, "
                "noise 0.15 and weight 0.1, give the selected generator and particle LRs to all three "
                "matched formulations. At 1200 updates require a lower v12 energy than all matched and "
                "established references, including the original rate controls and the three recent "
                "critic-rate and particle-weight configurations. Test seeds 22–24 and 251–253 separately, "
                "then six fresh training seeds with both original releases before the biological holdout.")
    elif args.study == "strong-prior-noise":
        baseline = NOISY_PRIOR
        trial = WIDE_NOISY_PRIOR
        references = [("noise150-" + arm[1], arm) for arm in paired(baseline)]
        independent_seeds = list(range(266, 272))
        outer_seeds = list(range(332, 348))
        rule = (
            "At fixed rates 0.014/0.021/0.0406 and prior weight 0.1, increase only latent noise "
            "from 0.15 to 0.25. Apply the same noise and weight to v11 and EMPF. On training seeds "
            "22–24 require lower mean v12 energy than every matched and established reference and "
            "all three lower-noise arms. If successful, require the same result separately on the "
            "second development cohort, seeds 251–253. Only a passing candidate goes to six new "
            "independent training seeds 266–271, including exact-setting original 0.11.1 v11 and "
            "0.10.0 EMPF. Only a win over every independent reference qualifies for the biological "
            "holdout on seeds 332–347. All decisions use energy at 1200 constant-rate updates."
        )
    elif args.study in MOTION_STUDIES:
        baseline = NOISY_PRIOR
        trial, calibration = calibrated_critic(args.motion_result)
        references = [("release-critic-" + arm[1], arm) for arm in paired(baseline)]
        independent_seeds = list(range(272, 278))
        outer_seeds = list(range(348, 364))
        rule = (
            "Divide the release critic LR 0.021 by the median v12-to-v11 centered critic displacement "
            "ratio across nine paired 64-update training probes, rounding to 0.0001. Keep this single "
            "rate fixed for all 1200 updates. Keep generator 0.014, particles 0.0406, noise 0.15 "
            "and prior weight 0.1. Give v11 and EMPF the same calibrated critic LR. Require v12 to "
            "beat every reference, including all three original-rate models, separately on development "
            "seeds 22–24 and 251–253, then on six fresh seeds 272–277 including the original v11 and "
            "EMPF. Only then score the frozen candidate on outer seeds 348–363."
        )
        if args.study in ("matched-critic-motion", "balanced-particle-prior"):
            references.extend(("first-calibration-" + arm[1], arm) for arm in paired(trial))
            trial, calibration = calibrated_critic(args.motion_result, args.motion_check)
            independent_seeds = list(range(278, 284))
            outer_seeds = list(range(364, 380))
            rule = (
                "Interpolate in log learning rate and log centered critic displacement ratio between "
                "the original 0.021 rate and its first calibrated rate. Target the median 64-step v11 "
                "movement at 0.021 and round the estimated v12 rate to 0.0001. Use the resulting fixed "
                "critic rate in all three matched formulations with generator 0.014, particles 0.0406, "
                "noise 0.15 and prior weight 0.1. On each development cohort (22–24 and 251–253), "
                "require lower energy after 1200 updates than all matched and established references, "
                "including all three at the original and first calibrated rates. Repeat on fresh seeds "
                "278–283 with both original releases before the biological holdout on seeds 364–379."
            )
        if args.study == "balanced-particle-prior":
            references.extend(("matched-motion-original-prior-" + arm[1], arm) for arm in paired(trial))
            trial, scale = balanced_prior(args.gradient_result, trial)
            calibration.update(relative_particle_gradient_strength=scale, fixed_prior_weight=trial["prior_reg"])
            independent_seeds = list(range(284, 290))
            outer_seeds = list(range(380, 396))
            rule = (
                "At the fixed movement-matched critic rate, divide the native prior weight 0.1 by the median "
                "paired increase in its gradient strength relative to the adversarial gradient on the particle "
                "table. Round to 0.001 and give the same weight and fixed generator, critic and particle rates "
                "to v12, v11 and EMPF. At 1200 updates v12 must beat every reference on each development cohort "
                "(22–24, then 251–253), including all three models at weight 0.1 at the original and both "
                "calibrated critic rates. Fresh seeds 284–289 must also beat both original releases before "
                "the biological holdout on seeds 380–395."
            )
    else:
        trial = NOISY_PRIOR
        baseline = NOISY
        references = ()
        independent_seeds = [251, 252, 253]
        outer_seeds = list(range(212, 228))
        rule = (
            "At the release v12 role rates and noise 0.15, set the native prior penalty weight to 0.1 "
            "in v12, v11 and EMPF. On all three training folds at seeds 22–24 require lower mean v12 "
            "energy than both weight-matched formulations and v12 at its default weight 0.02. "
            "If this passes, test the fixed candidate on training seeds 251–253. A lower independent "
            "mean than every matched and established reference qualifies for the biological holdout."
        )
    if args.study in TWO_COHORT_STUDIES:
        candidates, labels = control_arms([trial])
        by_key = {key(arm): arm for arm in candidates}
        for role, release in (("matched_v11", "0.11.1"), ("matched_empf", "0.10.0")):
            pinned.append(("original-" + role, [release, *by_key[labels[role]][1:]]))
    if args.stage == "compare-releases":
        compare_releases(args, trial, independent_seeds)
        return
    if args.stage == "confirm":
        decision = json.loads((args.output / "selection.json").read_text())
        protocol = json.loads((args.output / "search.json").read_text())
        if protocol["arms"][0] != [RELEASE, V12, trial] or protocol["outer_seeds"] != outer_seeds:
            raise ValueError("the frozen study has a different candidate or holdout seeds")
        if not decision["advances"] or decision["winner"] != trial["name"]:
            raise ValueError("the prior-weight candidate did not pass independent validation")
        # confirm_candidate already includes the rate-matched original EMPF arm.
        outer_refs = [*references, *(p for p in pinned if p[1][1] == "unified_v11")]
        confirm_candidate(args, decision, trial, outer_seeds, compare_historical_empf=True,
                          previous_transport=TRANSPORT["q75-noise25"], reference_arms=outer_refs)
        return

    matches, arms = comparison(trial, False)
    arms.extend(arm for _, arm in references)
    arms[1:1] = screen_only
    protocol = dict(rule=rule, arms=arms, groups=list(GROUPS),
                    screen_seeds=REPLICATION_SEEDS, independent_seeds=independent_seeds, outer_seeds=outer_seeds)
    if args.study in TWO_COHORT_STUDIES:
        protocol.update(second_screen_seeds=[251, 252, 253], pinned_independent_arms=[arm for _, arm in pinned])
    if calibration is not None:
        protocol["critic_calibration"] = calibration
    write_frozen(args.output / "search.json", protocol)
    screen_args = copy(args)
    if args.shared_screen is not None:
        screen_args.output = args.shared_screen.resolve()
    if args.study in PRELIMINARY_STUDIES:
        preliminary = [matches[0], [RELEASE, V12, baseline], *paired(WIDE_NOISY_PRIOR)[1:]]
        initial_scores = evaluate(screen_args, preliminary, REPLICATION_SEEDS)
        initial_means = {k: mean(v) for k, v in initial_scores.items()}
        own = key(matches[0])
        passes = all(initial_means[own] < value for name, value in initial_means.items() if name != own)
        early = dict(scores=initial_scores, means=initial_means, passes=passes)
        write_frozen(args.output / "preliminary.json", early)
        if not passes:
            write_frozen(args.output / "selection.json", dict(preliminary=early, advances=False, winner=None))
            return
    scores = evaluate(screen_args, arms, REPLICATION_SEEDS)
    means = {k: mean(v) for k, v in scores.items()}
    own, v11, empf = [key(arm) for arm in matches]
    required = [v11, empf, key([RELEASE, V12, baseline])]
    if args.study == "double-prior-weight":
        required.extend(key(arm) for _, arm in references)
    elif args.study == "prior-interaction":
        required.extend(key(arm) for arm in arms if arm[1] != V12)
    elif args.study in TWO_COHORT_STUDIES:
        required = [key(arm) for arm in arms if key(arm) != own]
    passes = all(means[own] < means[control] for control in required)
    screened = dict(scores=scores, means=means, passes=passes)
    write_frozen(args.output / "screen.json", screened)
    if args.study == "prior-interaction":
        write_frozen(args.output / "prior_effects.json", prior_effects(arms, scores, REPLICATION_SEEDS))
    decision = dict(screening=screened, advances=False, winner=None)
    if passes and args.study in TWO_COHORT_STUDIES:
        second_args = copy(args)
        if args.second_screen is not None:
            second_args.output = args.second_screen.resolve()
        second_scores = evaluate(second_args, arms, protocol["second_screen_seeds"])
        second_means = {name: mean(values) for name, values in second_scores.items()}
        passes = all(second_means[own] < second_means[control] for control in required)
        decision["second_screening"] = dict(scores=second_scores, means=second_means, passes=passes)
        write_frozen(args.output / "second_screen.json", decision["second_screening"])
    if passes:
        decision.update(validate_finalist(args, trial, independent_seeds, reference_arms=[*references, *pinned]))
    write_frozen(args.output / "selection.json", decision)
    print("Prior-weight screen passed", passes, "qualified for holdout", decision["advances"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("data", "wheels", "output"):
        parser.add_argument("--" + field, type=Path, required=True)
    parser.add_argument("--workers-per-gpu", type=int, default=2)
    parser.add_argument("--window-cells", type=int)
    parser.add_argument("--stage", choices=("search", "confirm", "compare-releases", "compare-noise-controls",
                                          "replicate-balanced-releases", "replicate-calibrated-critic"), default="search")
    parser.add_argument("--study", choices=("rates", "prior-weight", "matched-prior-weight", "double-prior-weight",
                                            "prior-interaction", *TWO_COHORT_STUDIES),
                        default="rates")
    parser.add_argument("--particle-spread-result", type=Path, default=ROOT / "paper_watch/notes/neuroko_v12_particle_spread.json")
    parser.add_argument("--original-v11-motion", type=Path, default=ROOT / "paper_watch/notes/neuroko_original_v11_critic_motion.json")
    parser.add_argument("--spread-result", type=Path, default=ROOT / "paper_watch/notes/neuroko_v12_prior_weight_geometry.json")
    parser.add_argument("--shared-screen", type=Path,
                        help="reuse the seed 22–24 reference receipts for the prior-weight comparison")
    parser.add_argument("--second-screen", type=Path,
                        help="reuse seeds 251–253 from the prior-weight study in a two-cohort search")
    parser.add_argument("--motion-result", type=Path, default=ROOT / "paper_watch/notes/neuroko_v12_critic_motion.json",
                        help="paired critic-output measurements used by the critic-motion study")
    parser.add_argument("--motion-check", type=Path, default=ROOT / "paper_watch/notes/neuroko_v12_critic_motion_check.json",
                        help="measured response at the first calibrated rate for matched-critic-motion")
    parser.add_argument("--gradient-result", type=Path, default=ROOT / "paper_watch/notes/neuroko_v12_particle_balance.json",
                        help="paired particle gradients used by the balanced-particle-prior study")
    parser.set_defaults(require_constant_lr=True)
    args = parser.parse_args()
    if args.second_screen is not None and args.study not in TWO_COHORT_STUDIES:
        parser.error("--second-screen applies to a two-cohort study")
    for field in ("data", "wheels", "output"):
        setattr(args, field, getattr(args, field).resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    replication = {"replicate-balanced-releases": "balanced-particle-prior",
                   "replicate-calibrated-critic": "original-v11-critic"}
    if args.stage in replication:
        if args.study != replication[args.stage]:
            parser.error(f"--stage {args.stage} requires --study {replication[args.stage]}")
        replicate_calibrated_releases(args)
        return
    if args.stage == "compare-noise-controls":
        if args.study != "individual-role-rates":
            parser.error("--stage compare-noise-controls applies to individual-role-rates")
        compare_noise_controls(args)
        return
    if args.study != "rates":
        compare_prior_weight(args)
        return
    if args.stage == "compare-releases":
        parser.error("--stage compare-releases applies to a prior-weight study")
    if args.shared_screen is not None:
        parser.error("--shared-screen applies to a prior-weight study")

    if args.stage == "confirm":
        decision = json.loads((args.output / "selection.json").read_text())
        if not decision["advances"]:
            raise ValueError("rate and noise combination did not pass independent validation")
        trial = TRIALS[decision["winner"]]
        confirm_candidate(args, decision, trial, list(range(164, 180)), compare_historical_empf=True,
                          previous_transport=TRANSPORT["q75-noise25"], noise_reference=original_noise(trial))
        return

    cells = [{"seed": s, "groups": list(groups)} for s, groups in GROUPS_BY_SEED.items()]
    write_frozen(args.output / "search.json", dict(rule=RULE, trials=TRIALS, cells=cells))
    references = REFERENCES[1:]
    arms = [arm for trial in TRIALS.values() for arm in paired(trial)]
    screen = evaluate(args, [*arms, *references], list(GROUPS_BY_SEED), groups_by_seed=GROUPS_BY_SEED)
    scores = {name: mean(values) for name, values in screen.items()}
    gaps = {}
    for name, trial in TRIALS.items():
        own, v11, empf = [scores[key(arm)] for arm in paired(trial)]
        gaps[name] = own - min(v11, empf, *(scores[key(ref)] for ref in references))
    chosen = min(TRIALS, key=lambda n: (gaps[n], scores[key(paired(TRIALS[n])[0])], n))
    screened = dict(screen=screen, means=scores, gaps=gaps, chosen=chosen)
    write_frozen(args.output / "screen.json", screened)
    trial = TRIALS[chosen]
    result = validate_finalist(args, trial, [22, 23, 24], noise_reference=original_noise(trial))
    decision = dict(screening=screened, finalist=chosen, **result)
    write_frozen(args.output / "selection.json", decision)
    print("Replicated rate and noise comparison", chosen, decision["advances"], flush=True)


if __name__ == "__main__":
    main()
