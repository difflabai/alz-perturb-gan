"""Replicate the fixed-rate v12 candidate that beat both exact-rate controls on seed 15."""
import argparse
from copy import copy
import json
import os
from pathlib import Path
from statistics import mean
import subprocess
import sys

from compare_neuroko_v12 import GROUPS, ROOT, receipt, write_frozen
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


def original_release_scores(args, destination, matches, seeds, metric, *, published=None):
    v12, v11, empf = matches
    pinned = [["0.10.0", *empf[1:]], ["0.11.1", *v11[1:]]]
    comparisons = [*pinned, v12, v11, empf]
    write_frozen(destination / "release_comparison_protocol.json",
                 dict(arms=comparisons, seeds=seeds, groups=list(GROUPS),
                      metric=metric))
    scores = evaluate(args, comparisons, seeds)
    if published is not None and any(scores[name] != values for name, values in published.items()):
        raise ValueError("The same-wheel controls do not reproduce the published scores")
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
    write_frozen(destination / "release_comparison.json", dict(scores=scores, summaries=summaries))
    print("Original-release comparison", summaries, flush=True)
    return summaries


def prior_equivalence(release, arms):
    """Resolve full recipes in a process importing just this ParticleGAN release."""
    from importlib.metadata import version
    from paper_watch.alztest.harness import TrainingSettings
    from particlegan import get_recipe

    if version("particlegan") != release:
        raise ValueError(f"Expected ParticleGAN {release}")
    result = {}
    for _, recipe, trial in arms:
        settings = TrainingSettings(**{k: v for k, v in trial.items() if k not in ("name", "lr")})
        overrides = settings.recipe_overrides()
        native = get_recipe(recipe, lr=trial["lr"], **overrides)
        explicit = get_recipe(recipe, lr=trial["lr"], **(overrides | {"prior_reg": .02}))
        same = native == explicit
        if explicit.prior_reg != .02 or same != (recipe != "empf") or native.prior_reg != (.01 if recipe == "empf" else .02):
            raise ValueError(f"Unexpected prior defaults for {release} {recipe}")
        result[recipe] = dict(native_weight=native.prior_reg, shared_weight=explicit.prior_reg,
                              identical_full_recipe=same)
    return result


def compare_shared_prior(run_args, destination, matches, published):
    destination = destination / "shared-prior-002"
    destination.mkdir(exist_ok=True)
    v12, v11, empf = matches
    by_release = {"0.12.0": matches, "0.11.1": [v11], "0.10.0": [empf]}
    resolved = {}
    for release, arms in by_release.items():
        wheel = run_args.wheels / release
        if not (wheel / "particlegan").is_dir():
            raise FileNotFoundError(wheel)
        pythonpath = os.pathsep.join([str(wheel), str(ROOT), str(ROOT / "paper_watch/scripts"),
                                      os.environ.get("PYTHONPATH", "")])
        cmd = [sys.executable, "-B", "-c",
               "import json,sys; from neuroko_matched_rate_replication import prior_equivalence; "
               "print(json.dumps(prior_equivalence(*json.load(sys.stdin))))"]
        check = subprocess.run(cmd, input=json.dumps([release, arms]), text=True, capture_output=True,
                               env=dict(os.environ, PYTHONPATH=pythonpath), cwd=ROOT)
        if check.returncode:
            raise RuntimeError(f"Could not verify {release} recipes: {check.stderr.strip()}")
        resolved[release] = json.loads(check.stdout)
    write_frozen(destination / "prior_equivalence.json", resolved)
    shared_empf = [*empf[:2], dict(empf[2], name=empf[2]["name"] + "-prior020", prior_reg=.02)]
    original_release_scores(run_args, destination, [v12, v11, shared_empf], SEEDS,
                            "Mean four-target energy on the same nine training pairs at fixed role rates, "
                            "latent noise 0.025 and shared prior weight 0.02. Reuse full-equivalent Unified "
                            "recipes and refit both EMPF releases with the explicit coefficient.",
                            published={key(a): published[key(a)] for a in (v12, v11)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("data", "wheels", "output"):
        parser.add_argument("--" + field, type=Path, required=True)
    parser.add_argument("--shared-validation", type=Path,
                        help="study root where compatible training receipts are shared with the noise comparison")
    parser.add_argument("--rates", choices=("seed15", "empf"), default="seed15")
    parser.add_argument("--workers-per-gpu", type=int, default=2)
    parser.add_argument("--window-cells", type=int)
    parser.add_argument("--stage", choices=("search", "confirm", "compare-releases", "compare-shared-prior"), default="search")
    parser.set_defaults(require_constant_lr=True)
    args = parser.parse_args()
    for field in ("data", "wheels", "output"):
        setattr(args, field, getattr(args, field).resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    at_empf_rates = args.rates == "empf"
    trial = EMPF_RATES if at_empf_rates else TRIAL
    rule = EMPF_RULE if at_empf_rates else RULE
    if args.stage == "confirm":
        search = json.loads((args.output / "search.json").read_text())
        decision = json.loads((args.output / "selection.json").read_text())
        if search["rule"] != rule or decision["trial"] != trial or decision["winner"] != trial["name"]:
            raise ValueError(f"selection.json in {args.output} was not made for --rates {args.rates}")
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
    if args.stage in ("compare-releases", "compare-shared-prior"):
        decision = json.loads((args.output / "selection.json").read_text())
        if decision["trial"] != trial:
            raise ValueError("The published replication used a different candidate")
        if args.stage == "compare-shared-prior":
            compare_shared_prior(run_args, args.output, matches, decision["validation"])
            return
        metric = ("Mean four-target energy after 1200 fixed-rate updates on the original nine "
                  "training pairs at this rate setting. This release check does not select a candidate.")
        original_release_scores(run_args, args.output, matches, SEEDS, metric,
                                published={key(a): decision["validation"][key(a)] for a in matches})
        return
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
