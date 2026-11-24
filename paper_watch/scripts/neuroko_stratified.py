"""Test the released without-replacement particle sampler under constant learning rates."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from statistics import mean

from compare_neuroko_v12 import GROUPS, fit, receipt, write_frozen
from retune_neuroko_v12 import BASE, RELEASE, V12, constant_rates, controls, summarize


PLAIN = dict(name="plain", **BASE)
NOISY = dict(name="plain-noise", **BASE, prior_sigma=0.15)
STRATIFIED = {**BASE, "stratified_prior": True}
VARIANTS = {
    "stratified": {},
    "stratified-noise": {"prior_sigma": 0.15},
    "stratified-slow-p": {"prior_lr_mult": 0.0203 / 0.014},
    "stratified-active-half": {"rhbi_relativistic_weight": 0.5},
    "stratified-feydy": {"rhbi_self_coupling": "feydy_discrete"},
    "stratified-sharp": {"rhbi_eps": 0.05},
    "stratified-regularized": {"prior_reg": 0.1},
    "stratified-q75": {"rhbi_nn_quantile": 0.75},
}
TRIALS = {n: dict(name=n, **(STRATIFIED | changes)) for n, changes in VARIANTS.items()}


def key(arm):
    return "/".join([arm[0], arm[1], arm[2]["name"]])


def evaluate(args, arms, seeds):
    tasks = []
    paths = {}
    requested = {key(arm): arm for arm in arms}
    for arm in arms:
        for seed in seeds:
            for i, group in enumerate(GROUPS):
                path = receipt(args.output, "validation", *arm, seed, group)
                paths[key(arm), seed, group] = path
                if not path.is_file():
                    tasks.append(((i + seed) % 2, seed, group, [arm]))
    # Keep all arms for a given fold and seed on the same physical card.
    with ThreadPoolExecutor(args.workers_per_gpu) as a, ThreadPoolExecutor(args.workers_per_gpu) as b:
        futures = [
            (a, b)[gpu].submit(fit, args, "validation", seed, group, gpu, arm)
            for gpu, seed, group, arm in tasks
        ]
        for future in futures:
            future.result()

    fingerprints = {}
    results = {}
    for (name, seed, group), path in paths.items():
        d = json.loads(path.read_text())
        version, recipe, trial = requested[name]
        expected_rates = {
            "generator": trial["lr"],
            "critic": trial["lr"] * trial.get("d_lr_mult", 1.5),
            "particles": trial["lr"] * trial.get("prior_lr_mult", 2.9),
        }
        signature = tuple(
            d[k] for k in ("projected_data_sha256", "training_batches_sha256", "initial_parameters_sha256")
        )
        if (
            d["updates"] != 1200
            or d["seed"] != seed
            or not d["deterministic"]
            or (d["particlegan"], d["recipe"], d["name"]) != (version, recipe, trial["name"])
            or d["lr"] != trial["lr"]
            or any(d["settings"][k] != v for k, v in trial.items() if k not in ("lr", "name"))
            or [row["update"] for row in d["rate_trace"]] != [300, 600, 900, 1200]
            or d["score_groups"] != [group]
            or set(d["fit_groups"]) != set(GROUPS) - {group}
            or d["initial_rates"] != expected_rates
            or not constant_rates(d)
            or d["visible_devices"] != str((GROUPS.index(group) + seed) % 2)
            or fingerprints.setdefault((seed, group), signature) != signature
        ):
            raise ValueError(f"Invalid validation pair at {path}")
        results.setdefault(name, []).append(d["means"]["energy_distance"])
    return results


def control_arms(candidates):
    arms, labels = controls(candidates + [NOISY])
    # The previous study has two independent choices. Compare the stronger regularization as well.
    prior = dict(BASE, prior_reg=0.1)
    for r, trial in [
        (V12, {"name": "previous-regularized", **prior}),
        ("unified_v11", {"name": "previous-regularized-v11", **prior}),
        (
            "empf",
            {
                "name": "previous-regularized-empf",
                "lr": 0.014,
                "d_lr_mult": 1.5,
                "prior_lr_mult": 2.9,
                "prior_reg": 0.1,
            },
        ),
    ]:
        arm = [RELEASE, r, trial]
        arms.append(arm)
        labels[trial["name"]] = key(arm)
    return arms, labels


def validation_controls(finalists):
    arms = []
    mapping = {}
    shared = {}
    for n in finalists:
        local, labels = controls([TRIALS[n]])
        selected_labels = (
            "selected_v12",
            "matched_v11",
            "matched_empf",
            "previous_best_v11",
            "previous_best_empf",
            "tuned_empf",
        )
        by_key = {key(a): a for a in local}
        mapping[n] = {}
        for label in selected_labels:
            ar = by_key[labels[label]]
            params = {k: v for k, v in ar[2].items() if k != "name"}
            identity = json.dumps([ar[0], ar[1], params], sort_keys=True)
            if identity not in shared:
                name = n if label == "selected_v12" else f"{label}-{n}"
                ar = [ar[0], ar[1], {"name": name, **params}]
                shared[identity] = key(ar)
                arms.append(ar)
            mapping[n][label] = shared[identity]
    references = [[RELEASE, V12, PLAIN], [RELEASE, V12, NOISY]]
    return arms, references, mapping


def confirm_candidate(args, selection, chosen, seeds, *, compare_historical_empf=False):
    arms, labels = control_arms([chosen])
    if compare_historical_empf:
        matched = next(arm[2] for arm in arms if key(arm) == labels["matched_empf"])
        historical = [
            {"name": "empf-010-matched", **{k: v for k, v in matched.items() if k != "name"}},
            {"name": "empf-010-historical", "lr": 0.005, "d_lr_mult": 1.5, "prior_lr_mult": 2.5},
        ]
        for trial in historical:
            arm = ["0.10.0", "empf", trial]
            arms.append(arm)
            labels[trial["name"]] = key(arm)
    freeze = {"selection": selection, "arms": arms, "labels": labels, "seeds": seeds}
    write_frozen(args.output / "freeze.json", freeze)

    def outer(seed):
        k = seed % len(arms)
        fit(args, "heldout", seed, GROUPS[0], seed % 2, arms[k:] + arms[:k])

    with ThreadPoolExecutor(args.workers_per_gpu) as a, ThreadPoolExecutor(args.workers_per_gpu) as b:
        futures = [(a, b)[s % 2].submit(outer, s) for s in seeds]
        for future in futures:
            future.result()
    summarize(args, freeze)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("data", "wheels", "output"):
        parser.add_argument("--" + field, type=Path, required=True)
    parser.add_argument("--window-cells", type=int)
    parser.add_argument("--workers-per-gpu", type=int, default=2)
    parser.add_argument("--stage", choices=("search", "confirm"), default="search")
    parser.set_defaults(require_constant_lr=True)
    args = parser.parse_args()
    for field in ("data", "wheels", "output"):
        setattr(args, field, getattr(args, field).resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage == "confirm":
        selection = json.loads((args.output / "selection.json").read_text())
        if not selection["advances"]:
            raise ValueError("no candidate passed the training-group comparisons")
        confirm_candidate(args, selection, TRIALS[selection["winner"]], list(range(100, 116)))
        return

    write_frozen(
        args.output / "search.json",
        {
            "trials": TRIALS,
            "initial_seed": 3,
            "confirmation_seeds": [3, 4, 5],
            "groups": list(GROUPS),
            "rule": "Rank eight stratified v12 candidates by three-fold energy at seed 3. Advance the two lowest. "
            "Compare each against same-wheel v11 and EMPF at matched settings, uniform v11 and EMPF at the "
            "incumbent rates and uniform EMPF at its independently tuned rates on nine seed-fold cells. "
            "Advance to fresh outer seeds 100–115 only if v12's nine-cell mean is lower than every one "
            "of those five controls and both uniform v12 references. Choose the lowest qualifying energy. "
            "Use lexical tie breaking. Share the evaluation sampler and keep all learning rates constant.",
        },
    )
    candidates = [[RELEASE, V12, t] for t in [*TRIALS.values(), PLAIN, NOISY]]
    first = evaluate(args, candidates, [3])
    finalists = sorted(TRIALS, key=lambda n: (mean(first[key([RELEASE, V12, TRIALS[n]])]), n))[:2]
    arms, references, mapping = validation_controls(finalists)
    scores = evaluate(args, [*arms, *references], [3, 4, 5])
    means = {arm: mean(values) for arm, values in scores.items()}
    passed = []
    for n, labels in mapping.items():
        own = means[labels["selected_v12"]]
        comparators = [means[v] for k, v in labels.items() if k != "selected_v12"]
        if own < min(comparators + [means[key(a)] for a in references]):
            passed.append(n)
    winner = min(passed, key=lambda n: (means[mapping[n]["selected_v12"]], n)) if passed else None
    selection = {
        "screen": first,
        "finalists": mapping,
        "validation": scores,
        "means": means,
        "advances": bool(passed),
        "winner": winner,
    }
    write_frozen(args.output / "selection.json", selection)
    print("Training-group decision", passed, winner, flush=True)


if __name__ == "__main__":
    main()
