"""Replicate the fixed-rate v12 candidate that beat both exact-rate controls on seed 15."""
import argparse
from copy import copy
import json
from pathlib import Path
from statistics import mean

from compare_neuroko_v12 import GROUPS, write_frozen
from neuroko_role_factorial import TRIALS, paired
from neuroko_stratified import confirm_candidate, control_arms, evaluate, key
from neuroko_transport_scale import TRIALS as TRANSPORT
from retune_neuroko_v12 import BASE

TRIAL = dict(TRIALS["g-high-d-high-p-low"], name="seed15-matched-leader", prior_sigma=0.025)
EMPF_RATES = {**BASE, "name": "at-tuned-empf-rates", "lr": 0.020, "d_lr_mult": 1.5,
              "prior_lr_mult": 2.9, "prior_sigma": 0.025}
SEEDS = [22, 23, 24]
REFERENCES = ("previous_best_v11", "previous_best_empf", "previous_v12", "tuned_empf",
              "selected_v12-alternative", "matched_v11-alternative", "matched_empf-alternative")
RULE = (
    "Replicate the seed-15 high-generator/high-critic/low-particle candidate unchanged on all three "
    "training folds at seeds 22–24. Primary matched comparison: v12 must beat both v11 and EMPF "
    "with the same three constant rates and latent noise. Report this separately from the overall "
    "comparison against the existing uniform v12, v11 and EMPF controls and independently tuned EMPF. "
    "Only a candidate with lower nine-cell mean energy than every frozen comparator can go to the "
    "outer biological group on paired seeds 180–195. Always report the training result."
)

EMPF_RULE = (
    "Compare the centered hollow, relativistic-weight-2 v12 and v11 formulations at the independently "
    "tuned EMPF rates 0.020, 0.030 and 0.058, with uniform particle sampling and sigma 0.025 for all. "
    "Use all three training folds on fresh seeds 22–24 and the identical tuned EMPF reference fits. "
    "Report the exact-rate result and the comparison against all previously frozen uniform controls. "
    "Only a lower nine-cell v12 mean than every reference qualifies for outer seeds 196–211."
)


def comparison(trial, at_empf_rates):
    # These historical reference names are fixed across the two studies that share their CUDA receipts.
    all_arms, labels = control_arms([TRIAL])
    keys = {labels[n] for n in REFERENCES}
    refs = [arm for arm in all_arms if key(arm) in keys]
    matches = paired(trial)
    if at_empf_rates:
        matches[2] = next(arm for arm in refs if key(arm) == labels["tuned_empf"])
    arms = {key(arm): arm for arm in [*matches, *refs]}
    return matches, list(arms.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("data", "wheels", "output"):
        parser.add_argument("--" + field, type=Path, required=True)
    parser.add_argument("--shared-validation", type=Path,
                        help="study root where compatible training receipts are shared with the noise comparison")
    parser.add_argument("--rates", choices=("seed15", "empf"), default="seed15")
    parser.add_argument("--workers-per-gpu", type=int, default=2)
    parser.add_argument("--window-cells", type=int)
    parser.add_argument("--stage", choices=("search", "confirm"), default="search")
    parser.set_defaults(require_constant_lr=True)
    args = parser.parse_args()
    for field in ("data", "wheels", "output"):
        setattr(args, field, getattr(args, field).resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    at_empf_rates = args.rates == "empf"
    trial = EMPF_RATES if at_empf_rates else TRIAL
    rule = EMPF_RULE if at_empf_rates else RULE
    if args.stage == "confirm":
        decision = json.loads((args.output / "selection.json").read_text())
        if not decision["advances"]:
            raise ValueError("the unchanged rate configuration did not beat every reference")
        seeds = list(range(196, 212)) if at_empf_rates else list(range(180, 196))
        confirm_candidate(args, decision, trial, seeds,
                          compare_historical_empf=True, previous_transport=TRANSPORT["q75-noise25"])
        return

    matches, arms = comparison(trial, at_empf_rates)
    write_frozen(args.output / "search.json", dict(rule=rule, arms=arms, seeds=SEEDS, groups=list(GROUPS)))
    run_args = copy(args)
    if args.shared_validation is not None:
        run_args.output = args.shared_validation.resolve()
    scores = evaluate(run_args, arms, SEEDS)
    means = {name: mean(values) for name, values in scores.items()}
    own, v11, empf = (key(arm) for arm in matches)
    differences = {}
    for name, values in scores.items():
        if name != own:
            gaps = [a - b for a, b in zip(scores[own], values, strict=True)]
            differences[name] = {"mean": mean(gaps), "pairs_won": sum(x < 0 for x in gaps),
                                 "by_seed": {str(seed): mean(gaps[i * len(GROUPS):(i + 1) * len(GROUPS)])
                                             for i, seed in enumerate(SEEDS)}}
    matched_win = means[own] < min(means[v11], means[empf])
    advances = all(d["mean"] < 0. for d in differences.values())
    decision = dict(trial=trial, validation=scores, means=means, matched_win=matched_win,
                    differences=differences, advances=advances, winner=trial["name"] if advances else None)
    write_frozen(args.output / "selection.json", decision)
    print("Unchanged rate replication", "matched", matched_win, "all", advances, flush=True)


if __name__ == "__main__":
    main()
