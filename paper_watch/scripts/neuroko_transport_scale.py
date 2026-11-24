"""Compare fixed-rate transport or critic-penalty settings with matched v11 and EMPF."""
import argparse
import json
from pathlib import Path
from statistics import mean

from compare_neuroko_v12 import GROUPS, write_frozen
from neuroko_stratified import confirm_candidate, control_arms, evaluate, key
from retune_neuroko_v12 import BASE, RELEASE, V12

TRIALS = {
    f"q{q}-noise{sigma}": dict(
        name=f"q{q}-noise{sigma}", **BASE, rhbi_nn_quantile=q / 100, prior_sigma=sigma / 1000
    )
    for q in (75, 95) for sigma in (25, 150)
}
RULE = (
    "Screen four uniform-sampling combinations on the three training groups at seed 6. "
    "For each v12 trial, subtract the lower mean of its matched v11 and EMPF controls. "
    "Choose the smallest negative difference, breaking ties by v12 mean then trial name. "
    "On independent training seeds 7–9, require the chosen v12 mean to beat both matched controls, "
    "the default-bandwidth v11/EMPF/v12 references, the 0.15-bandwidth v11/EMPF/v12 references, "
    "and independently tuned EMPF. Only a qualifying trial is evaluated on outer seeds 116–131. "
    "All fits use 1200 updates, constant role rates, and the same capacity and input fingerprints."
)

CAP_TRIALS = {
    f"cap{percent}": dict(name=f"cap{percent}", **BASE, prior_sigma=0.025, reg_kappa=percent / 100)
    for percent in (75, 125)
}
CAP_RULE = (
    "Change the native critic gradient-penalty threshold from 1.0 to 0.75 or 1.25, matched across "
    "v12, v11 and EMPF. Hold latent noise at 0.025, use uniform particle sampling and keep role LRs "
    "constant. At seed 11 choose the lowest negative difference from the better matched control, "
    "breaking ties by v12 energy and name. Repeat on independent training seeds 12–14, requiring "
    "the chosen v12 to beat both matched controls, the old uniform references at either bandwidth, "
    "and independently tuned EMPF. Only a qualifying trial goes to new outer seeds 132–147, including "
    "the earlier quantile-0.75 v12 and v11 and both original 0.10.0 EMPF controls. "
    "Every fit has the same 1200 updates, model capacity, initial weights and training minibatches within seed."
)


def matched(trial):
    # EMPF uses the same Gaussian noise and absolute role rates. Its loss has no RHBI scale.
    shared = dict(lr=trial["lr"], d_lr_mult=1.5, prior_lr_mult=2.9, prior_sigma=trial["prior_sigma"])
    name = f"noise{round(1000 * shared['prior_sigma'])}"
    if "reg_kappa" in trial:
        name = trial["name"]
        shared["reg_kappa"] = trial["reg_kappa"]
    return [
        [RELEASE, V12, trial],
        [RELEASE, "unified_v11", trial],
        [RELEASE, "empf", dict(name=name, **shared)],
    ]


def validate_finalist(args, trial, validation_seeds):
    comparisons, labels = control_arms([trial])
    kept = ("selected_v12", "matched_v11", "matched_empf", "previous_best_v11", "previous_best_empf",
            "previous_v12", "tuned_empf", "selected_v12-alternative", "matched_v11-alternative",
            "matched_empf-alternative")
    keys = {labels[label] for label in kept}
    validation = evaluate(args, [arm for arm in comparisons if key(arm) in keys], validation_seeds)
    means = {name: mean(values) for name, values in validation.items()}
    chosen = labels["selected_v12"]
    advances = all(means[chosen] < score for name, score in means.items() if name != chosen)
    return {"validation": validation, "means": means, "labels": {k: labels[k] for k in kept},
            "advances": advances, "winner": trial["name"] if advances else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("data", "wheels", "output"):
        parser.add_argument("--" + field, type=Path, required=True)
    parser.add_argument("--workers-per-gpu", type=int, default=2)
    parser.add_argument("--window-cells", type=int)
    parser.add_argument("--study", choices=["transport-scale", "critic-cap"], default="transport-scale")
    parser.add_argument("--stage", choices=["search", "confirm"], default="search")
    parser.set_defaults(require_constant_lr=True)
    args = parser.parse_args()
    for field in ("data", "wheels", "output"):
        setattr(args, field, getattr(args, field).resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    if args.study == "critic-cap":
        trials, rule, screen_seed = CAP_TRIALS, CAP_RULE, 11
        validation_seeds, outer_seeds = [12, 13, 14], list(range(132, 148))
        previous_transport = TRIALS["q75-noise25"]
    else:
        trials, rule, screen_seed = TRIALS, RULE, 6
        validation_seeds, outer_seeds = [7, 8, 9], list(range(116, 132))
        previous_transport = None
    if args.stage == "confirm":
        selection = json.loads((args.output / "selection.json").read_text())
        if not selection["advances"]:
            raise ValueError("no candidate passed training validation")
        confirm_candidate(args, selection, trials[selection["winner"]], outer_seeds,
                          compare_historical_empf=True, previous_transport=previous_transport)
        return

    write_frozen(args.output / "search.json", dict(trials=trials, groups=list(GROUPS), rule=rule))
    arms = {key(arm): arm for trial in trials.values() for arm in matched(trial)}
    screen = evaluate(args, list(arms.values()), [screen_seed])
    gaps = {}
    for name, trial in trials.items():
        own, v11, empf = (mean(screen[key(arm)]) for arm in matched(trial))
        gaps[name] = (own - min(v11, empf), own)
    finalist = min(gaps, key=lambda name: (*gaps[name], name))
    decision = dict(screen=screen, gaps=gaps, finalist=None, validation={}, means={}, advances=False, winner=None)
    if gaps[finalist][0] < 0:
        trial = trials[finalist]
        decision.update(finalist=finalist, **validate_finalist(args, trial, validation_seeds))
    # Store lists in the JSON protocol so a deterministic rerun compares equal.
    decision["gaps"] = {name: list(values) for name, values in gaps.items()}
    write_frozen(args.output / "selection.json", decision)
    print(args.study, "training selection", decision["finalist"], decision["advances"], flush=True)


if __name__ == "__main__":
    main()
