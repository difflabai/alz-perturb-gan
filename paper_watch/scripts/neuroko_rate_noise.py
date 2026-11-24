"""Select a rate and latent-noise combination using paired comparisons across three seeds."""
import argparse
from copy import copy
import json
from pathlib import Path
from statistics import mean

from compare_neuroko_v12 import GROUPS, receipt, write_frozen
from neuroko_matched_rate_replication import SEEDS as REPLICATION_SEEDS, comparison
from neuroko_role_factorial import REFERENCES, TRIALS as RATES, paired
from neuroko_stratified import NOISY, confirm_candidate, control_arms, evaluate, key
from neuroko_transport_scale import TRIALS as TRANSPORT, validate_finalist
from retune_neuroko_v12 import RELEASE, V12

RATE_NAMES = ("g-high-d-high-p-low", "g-low-d-low-p-high")
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
    trial = dict(NOISY, name="noise150-prior100", d_lr_mult=1.5, prior_lr_mult=2.9, prior_reg=0.1)
    independent_seeds = [251, 252, 253]
    outer_seeds = list(range(212, 228))
    if args.stage == "compare-releases":
        compare_releases(args, trial, independent_seeds)
        return
    if args.stage == "confirm":
        decision = json.loads((args.output / "selection.json").read_text())
        if not decision["advances"]:
            raise ValueError("the prior-weight candidate did not pass independent validation")
        confirm_candidate(args, decision, trial, outer_seeds, compare_historical_empf=True,
                          previous_transport=TRANSPORT["q75-noise25"])
        return

    matches, arms = comparison(trial, False)
    rule = (
        "At the release v12 role rates and noise 0.15, set the native prior penalty weight to 0.1 "
        "in v12, v11 and EMPF. On all three training folds at seeds 22–24 require lower mean v12 "
        "energy than both weight-matched formulations and v12 at its default weight 0.02. "
        "If this passes, test the fixed candidate on training seeds 251–253. A lower independent "
        "mean than every matched and established reference qualifies for the biological holdout."
    )
    write_frozen(args.output / "search.json", dict(rule=rule, arms=arms, groups=list(GROUPS),
                 screen_seeds=REPLICATION_SEEDS, independent_seeds=independent_seeds, outer_seeds=outer_seeds))
    screen_args = copy(args)
    if args.shared_screen is not None:
        screen_args.output = args.shared_screen.resolve()
    scores = evaluate(screen_args, arms, REPLICATION_SEEDS)
    means = {k: mean(v) for k, v in scores.items()}
    own, v11, empf = [key(arm) for arm in matches]
    baseline = key([RELEASE, V12, NOISY])
    passes = means[own] < min(means[v11], means[empf], means[baseline])
    screened = dict(scores=scores, means=means, passes=passes)
    write_frozen(args.output / "screen.json", screened)
    decision = dict(screening=screened, advances=False, winner=None)
    if passes:
        decision.update(validate_finalist(args, trial, independent_seeds))
    write_frozen(args.output / "selection.json", decision)
    print("Prior-weight screen passed", passes, "qualified for holdout", decision["advances"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("data", "wheels", "output"):
        parser.add_argument("--" + field, type=Path, required=True)
    parser.add_argument("--workers-per-gpu", type=int, default=2)
    parser.add_argument("--window-cells", type=int)
    parser.add_argument("--stage", choices=("search", "confirm", "compare-releases"), default="search")
    parser.add_argument("--study", choices=("rates", "prior-weight"), default="rates")
    parser.add_argument("--shared-screen", type=Path,
                        help="reuse the seed 22–24 reference receipts for the prior-weight comparison")
    parser.set_defaults(require_constant_lr=True)
    args = parser.parse_args()
    for field in ("data", "wheels", "output"):
        setattr(args, field, getattr(args, field).resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    if args.study == "prior-weight":
        compare_prior_weight(args)
        return
    if args.stage == "compare-releases":
        parser.error("--stage compare-releases applies to --study prior-weight")
    if args.shared_screen is not None:
        parser.error("--shared-screen applies to --study prior-weight")

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
