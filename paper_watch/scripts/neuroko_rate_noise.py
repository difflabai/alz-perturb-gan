"""Select a rate and latent-noise combination using paired comparisons across three seeds."""
import argparse
from copy import copy
import json
from pathlib import Path
from statistics import mean

from compare_neuroko_v12 import GROUPS, receipt, write_frozen
from neuroko_matched_rate_replication import SEEDS as REPLICATION_SEEDS, TRIAL as MATCHED_RATE_TRIAL, comparison
from neuroko_role_factorial import REFERENCES, TRIALS as RATES, paired
from neuroko_stratified import NOISY, confirm_candidate, control_arms, evaluate, key
from neuroko_transport_scale import TRIALS as TRANSPORT, validate_finalist
from retune_neuroko_v12 import BASE, RELEASE, V12

RATE_NAMES = ("g-high-d-high-p-low", "g-low-d-low-p-high")
NOISY_PRIOR = dict(NOISY, name="noise150-prior100", d_lr_mult=1.5, prior_lr_mult=2.9, prior_reg=0.1)
MATCHED_PRIOR = dict(MATCHED_RATE_TRIAL, name="matched-rates-prior100", prior_reg=0.1)
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
    arms, labels = control_arms([trial])
    by_key = {key(arm): arm for arm in arms}
    v12, v11, empf = (by_key[labels[n]] for n in ("selected_v12", "matched_v11", "matched_empf"))
    pinned = [["0.10.0", *empf[1:]], ["0.11.1", *v11[1:]]]
    comparisons = [*pinned, v12, v11, empf]
    write_frozen(args.output / "release_comparison_protocol.json",
                 dict(arms=comparisons, seeds=seeds, groups=list(GROUPS),
                      metric="Mean four-target energy at 1200 fixed-rate updates on the independent training folds."))
    scores = evaluate(args, comparisons, seeds)
    own = scores[key(v12)]
    summaries = {}
    for arm in comparisons:
        values = scores[key(arm)]
        rows = [json.loads(receipt(args.output, "validation", *arm, seed, group).read_text())
                for seed in seeds for group in GROUPS]
        if any(row.get("constant_lr_verified_updates") != 1200 for row in rows):
            raise ValueError(f"missing a full per-update learning-rate check for {key(arm)}")
        gaps = [x - y for x, y in zip(own, values, strict=True)]
        summaries[key(arm)] = dict(energy=mean(values), v12_minus_arm=mean(gaps),
                                  v12_pairs_won=sum(d < 0 for d in gaps))
    write_frozen(args.output / "release_comparison.json", dict(scores=scores, summaries=summaries))
    print("Original-release comparison", summaries, flush=True)


def compare_prior_weight(args):
    screen_only = []
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
        confirm_candidate(args, decision, trial, outer_seeds, compare_historical_empf=True,
                          previous_transport=TRANSPORT["q75-noise25"], reference_arms=references)
        return

    matches, arms = comparison(trial, False)
    arms.extend(arm for _, arm in references)
    arms[1:1] = screen_only
    write_frozen(args.output / "search.json", dict(rule=rule, arms=arms, groups=list(GROUPS),
                 screen_seeds=REPLICATION_SEEDS, independent_seeds=independent_seeds, outer_seeds=outer_seeds))
    screen_args = copy(args)
    if args.shared_screen is not None:
        screen_args.output = args.shared_screen.resolve()
    scores = evaluate(screen_args, arms, REPLICATION_SEEDS)
    means = {k: mean(v) for k, v in scores.items()}
    own, v11, empf = [key(arm) for arm in matches]
    required = [v11, empf, key([RELEASE, V12, baseline])]
    if args.study == "double-prior-weight":
        required.extend(key(arm) for _, arm in references)
    elif args.study == "prior-interaction":
        required.extend(key(arm) for arm in arms if arm[1] != V12)
    passes = all(means[own] < means[control] for control in required)
    screened = dict(scores=scores, means=means, passes=passes)
    write_frozen(args.output / "screen.json", screened)
    if args.study == "prior-interaction":
        write_frozen(args.output / "prior_effects.json", prior_effects(arms, scores, REPLICATION_SEEDS))
    decision = dict(screening=screened, advances=False, winner=None)
    if passes:
        decision.update(validate_finalist(args, trial, independent_seeds, reference_arms=references))
    write_frozen(args.output / "selection.json", decision)
    print("Prior-weight screen passed", passes, "qualified for holdout", decision["advances"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("data", "wheels", "output"):
        parser.add_argument("--" + field, type=Path, required=True)
    parser.add_argument("--workers-per-gpu", type=int, default=2)
    parser.add_argument("--window-cells", type=int)
    parser.add_argument("--stage", choices=("search", "confirm", "compare-releases"), default="search")
    parser.add_argument("--study", choices=("rates", "prior-weight", "matched-prior-weight", "double-prior-weight",
                                            "prior-interaction"),
                        default="rates")
    parser.add_argument("--shared-screen", type=Path,
                        help="reuse the seed 22–24 reference receipts for the prior-weight comparison")
    parser.set_defaults(require_constant_lr=True)
    args = parser.parse_args()
    for field in ("data", "wheels", "output"):
        setattr(args, field, getattr(args, field).resolve())
    args.output.mkdir(parents=True, exist_ok=True)
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
