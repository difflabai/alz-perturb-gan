"""Screen release-native v12 hyperparameters on NeuroKO training replicates."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
from statistics import mean, median, stdev
from compare_neuroko_v12 import GROUPS, fit, receipt, write_frozen


RELEASE = "0.12.0"
V12 = "unified_v12"
BASE = dict(lr=.014, rhbi_self_coupling="hollow", rhbi_relativistic_weight=2.)
SETTINGS = {
    "hollow-active2": {},
    "g-low-fixed": {"lr": .010, "d_lr_mult": .021 / .010, "prior_lr_mult": .0406 / .010},
    "g-high-fixed": {"lr": .020, "d_lr_mult": .021 / .020, "prior_lr_mult": .0406 / .020},
    "d-low-fixed": {"d_lr_mult": .0075 / .014},
    "d-high-fixed": {"d_lr_mult": .042 / .014},
    "p-low-fixed": {"prior_lr_mult": .0125 / .014},
    "p-high-fixed": {"prior_lr_mult": .0812 / .014},
    "momentum-half": {"betas": [.5, .999]},
    "momentum-nine": {"betas": [.9, .999]},
    "smooth-low": {"optimizer_smoothing": .0006},
    "smooth-high": {"optimizer_smoothing": .04},
    "transport-medium": {"rhbi_eps": .4, "rhbi_rho": 6.},
    "transport-wide": {"rhbi_eps": 1., "rhbi_rho": 12.},
    "active-half": {"rhbi_relativistic_weight": .5},
    "active-five": {"rhbi_relativistic_weight": 5.},
    "weak-critic-penalty": {"reg_coeff": .25},
    "strong-prior-penalty": {"prior_reg": .1},
}
NOISE = {"hollow-active2": {}, **{f"noise-{n}": {"prior_sigma": sigma}
                                 for n, sigma in (("015", .15), ("04", .4), ("08", .8), ("16", 1.6))}}


def constant_rates(row):
    initial = row["initial_rates"]
    return (row["final_rates"] == initial
            and all(all(step[role] == value for role, value in initial.items())
                    for step in row.get("rate_trace", []))
            and row["settings"].get("lr_floor") in (None, 1.)
            and row["settings"].get("network_lr_floor") in (None, 1.))


def validation(args, candidates, names, seeds):
    jobs = []
    paths = {}
    for i, name in enumerate(names):
        for seed in seeds:
            for g, group in enumerate(GROUPS):
                trial = candidates[name]
                path = receipt(args.output, "validation", RELEASE, V12, trial, seed, group)
                old = receipt(args.previous, "validation", RELEASE, V12, trial, seed, group)
                if name == "hollow-active2" and old.is_file():
                    path = old
                elif not path.is_file():
                    gpu = (i + g + seed) % 2
                    jobs.append((gpu, seed, group, [(RELEASE, V12, trial)]))
                paths[name, seed, group] = path
    with ThreadPoolExecutor(args.workers_per_gpu) as first, ThreadPoolExecutor(args.workers_per_gpu) as second:
        pending = [(first, second)[gpu].submit(fit, args, "validation", seed, group, gpu, arm)
                   for gpu, seed, group, arm in jobs]
        for future in pending:
            future.result()
    results = {}
    fingerprints = {}
    for (name, seed, group), path in paths.items():
        row = json.loads(path.read_text())
        wanted = candidates[name]
        if (row["updates"] != 1200 or row["recipe"] != V12 or row["particlegan"] != RELEASE
                or row["score_groups"] != [group] or set(row["fit_groups"]) != set(GROUPS) - {group}
                or row["seed"] != seed or not row["deterministic"] or not constant_rates(row)
                or row["lr"] != wanted["lr"]
                or any(row["settings"][k] != v for k, v in wanted.items() if k not in ("lr", "name"))):
            raise ValueError(f"Incompatible validation receipt at {path}")
        fingerprint = tuple(row[k] for k in ("initial_parameters_sha256", "projected_data_sha256",
                                              "training_batches_sha256"))
        if fingerprints.setdefault((seed, group), fingerprint) != fingerprint:
            raise ValueError(f"Validation data, weights or batches differ at {path}")
        results.setdefault(name, []).append(row["means"]["energy_distance"])
    return results


def controls(finalists):
    trials = []
    labels = {}

    def add(label, version, recipe, name, settings):
        arm = next((arm for arm in trials if arm[:2] == [version, recipe]
                    and {k: v for k, v in arm[2].items() if k != "name"} == settings), None)
        if arm is None:
            arm = [version, recipe, dict(name=name, **settings)]
            trials.append(arm)
        labels[label] = "/".join([version, recipe, arm[2]["name"]])

    for i, selected in enumerate(finalists):
        suffix = "" if i == 0 else "-alternative"
        settings = {k: v for k, v in selected.items() if k != "name"}
        matched = {k: v for k, v in settings.items() if not k.startswith("rhbi_")}
        matched.setdefault("d_lr_mult", 1.5)
        matched.setdefault("prior_lr_mult", 2.9)
        add("selected_v12" + suffix, RELEASE, V12, selected["name"], settings)
        add("matched_v11" + suffix, RELEASE, "unified_v11", "matched-v11" + suffix, settings)
        add("matched_empf" + suffix, RELEASE, "empf", "matched-empf" + suffix, matched)
    slow = dict(BASE, d_lr_mult=.0075 / .014, prior_lr_mult=.0125 / .014)
    add("previous_best_v11", RELEASE, "unified_v11", "v11-incumbent", BASE)
    add("previous_best_empf", RELEASE, "empf", "empf-mid", dict(lr=.014, d_lr_mult=1.5, prior_lr_mult=2.9))
    add("tuned_empf", RELEASE, "empf", "empf-tuned", dict(lr=.020, d_lr_mult=1.5, prior_lr_mult=2.9))
    add("pinned_v11", "0.11.1", "unified_v11", "v11-transfer", slow)
    add("previous_v12", RELEASE, V12, "hollow-active2", BASE)
    add("slow_v12", RELEASE, V12, "v11-transfer", slow)
    add("default_v12", RELEASE, V12, "default", {"lr": .014})
    return trials, labels


def summarize(args, freeze):
    from scipy.stats import t

    hashes = {}
    records = {}
    for version, recipe, trial in freeze["arms"]:
        key = "/".join([version, recipe, trial["name"]])
        rows = []
        for seed in freeze["seeds"]:
            path = receipt(args.output, "heldout", version, recipe, trial, seed, GROUPS[0])
            row = json.loads(path.read_text())
            fingerprint = {k: row[k] for k in ("initial_parameters_sha256", "projected_data_sha256",
                                               "training_batches_sha256")}
            if hashes.setdefault(seed, fingerprint) != fingerprint:
                raise ValueError(f"Initialization, cells or batches differ at {path}")
            if (row["seed"] != seed or row["updates"] != 1200 or row["visible_devices"] != str(seed % 2)
                    or row["fit_groups"] != list(GROUPS) or row["score_groups"] != ["Screen1_0"]
                    or row["recipe"] != recipe or row["particlegan"] != version or not row["deterministic"]
                    or row["lr"] != trial["lr"] or row["name"] != trial["name"] or not constant_rates(row)
                    or [x["update"] for x in row.get("rate_trace", [])] != [300, 600, 900, 1200]
                    or any(row["settings"][k] != v for k, v in trial.items() if k not in ("lr", "name"))
                    or any(row["settings"][k] != v for k, v in {"batch_size": 64, "z_dim": 16, "num_particles": 64}.items())):
                raise ValueError(f"Unmatched confirmation protocol at {path}")
            rows.append(row)
        records[key] = rows

    labels = freeze["labels"]
    chosen = records[labels["selected_v12"]]
    for suffix in ("", "-alternative"):
        label = "selected_v12" + suffix
        if label not in labels:
            continue
        comparators = ["matched_empf" + suffix, "matched_v11" + suffix]
        if not suffix and "empf-010-matched" in labels:
            comparators.append("empf-010-matched")
        for control in comparators:
            for a, b in zip(records[labels[label]], records[labels[control]]):
                for field in ("initial_rates", "final_rates", "rate_trace"):
                    if a.get(field) != b.get(field):
                        raise ValueError(f"Actual training rates differ for {control}, seed {a['seed']}")
                if a["settings"]["prior_sigma"] != b["settings"]["prior_sigma"]:
                    raise ValueError("Compared formulations must share latent noise bandwidth")
    summaries = {}
    for key, rows in records.items():
        diff = [a["means"]["energy_distance"] - b["means"]["energy_distance"] for a, b in zip(chosen, rows)]
        margin = float(t.ppf(.975, len(diff) - 1)) * stdev(diff) / len(diff) ** .5
        summaries[key] = {"energy": mean(r["means"]["energy_distance"] for r in rows),
                          "spread": mean(r["means"]["spread_ratio"] for r in rows),
                          "median_seconds": median(r["seconds"] for r in rows),
                          "v12_minus_arm": mean(diff), "paired_ci95": [mean(diff) - margin, mean(diff) + margin],
                          "v12_seeds_won": sum(x < 0 for x in diff),
                          "initial_rates": rows[0]["initial_rates"], "final_rates": rows[0]["final_rates"],
                          "optimizer": rows[0]["optimizer_momentum"], "rate_trace": rows[0].get("rate_trace"),
                          "per_seed": [{k: r[k] for k in ("seed", "means", "seconds", "targets", "effective_rank")}
                                       for r in rows]}
    result = {"search": json.loads((args.output / "search.json").read_text()), "freeze": freeze,
              "seed_hashes": hashes, "summaries": summaries}
    path = args.output / "result.json"
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def confirmation(args):
    searches = {}
    candidates = []
    for directory in [args.output] + ([args.companion] if args.companion else []):
        search = json.loads((directory / "search.json").read_text())
        selection = json.loads((directory / "selection.json").read_text())
        winner = selection["winner"]
        candidates.append((selection["means"][winner], winner, search["candidates"][winner]))
        searches[directory.name] = {"search": search, "selection": selection}
    finalists = [settings for _, _, settings in sorted(candidates, key=lambda v: (v[0], v[1]))]
    arms, labels = controls(finalists)
    seeds = list(range(84, 100))
    freeze = {"searches": searches, "arms": arms, "labels": labels, "seeds": seeds,
              "primary": "The finalist with the lowest nine-cell training validation mean across both "
                         "searches is the primary candidate. Score mean four-target energy after 1200 "
                         "updates on Screen1_0. Report both finalists, all competing arms and paired "
                         "uncertainty conditional on this one biological replicate."}
    write_frozen(args.output / "freeze.json", freeze)

    def run_seed(seed):
        offset = seed % len(arms)
        fit(args, "heldout", seed, GROUPS[0], seed % 2, arms[offset:] + arms[:offset])
    with ThreadPoolExecutor(args.workers_per_gpu) as first, ThreadPoolExecutor(args.workers_per_gpu) as second:
        pending = [(first, second)[s % 2].submit(run_seed, s) for s in seeds]
        for future in pending:
            future.result()
    summarize(args, freeze)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--companion", type=Path, help="second training-only search for joint confirmation")
    parser.add_argument("--stage", choices=("search", "confirm"), default="search")
    parser.add_argument("--study", choices=("training", "latent-noise"), default="training")
    for flag in ("data", "wheels", "output", "previous"):
        parser.add_argument("--" + flag, required=True, type=Path)
    parser.add_argument("--window-cells", type=int)
    parser.add_argument("--workers-per-gpu", type=int, default=3)
    parser.set_defaults(require_constant_lr=True)
    args = parser.parse_args()
    for key in ("data", "wheels", "output", "previous"):
        setattr(args, key, getattr(args, key).resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage == "confirm":
        confirmation(args)
        return
    settings = SETTINGS if args.study == "training" else NOISE
    candidates = {n: dict(name=n, **(BASE | override)) for n, override in settings.items()}
    count = 4 if args.study == "training" else 2
    write_frozen(args.output / "search.json", {
        "candidates": candidates, "validation_groups": list(GROUPS), "screen_seed": 0,
        "rule": "Advance the " + {4:"four", 2:"two"}[count] +
                " lowest seed-0 three-fold means plus the previous v12 incumbent. "
                "Select by the nine equally weighted cells across seeds 0, 1 and 2 and three folds, "
                "with lexical tie breaking. Screen1_0 has no role in this choice.",
        "confirmation": "Evaluate locked settings on fresh paired seeds 84-99. Include prior strongest "
                        "v11 and EMPF controls and equal-setting v11/equal-rate EMPF controls. Record "
                        "point ordering and paired uncertainty. No candidate selection on outer scores.",
    })
    first = validation(args, candidates, list(candidates), [0])
    finalists = sorted(set(sorted(first, key=lambda n: (mean(first[n]), n))[:count]) | {"hollow-active2"})
    followup = validation(args, candidates, finalists, [1, 2])
    scores = {n: mean(first[n] + followup[n]) for n in finalists}
    selection = dict(screen=first, confirm=followup, means=scores, winner=min(scores, key=lambda n: (scores[n], n)))
    write_frozen(args.output / "selection.json", selection)
    print("Selection", selection["winner"], scores, flush=True)


if __name__ == "__main__":
    main()
