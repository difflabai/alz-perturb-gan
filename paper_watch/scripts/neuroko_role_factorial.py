"""Cross constant generator, critic and particle rates on matched NeuroKO training inputs."""
import argparse
import json
from pathlib import Path
from statistics import mean

from compare_neuroko_v12 import GROUPS, write_frozen
from neuroko_stratified import confirm_candidate, evaluate, key
from neuroko_transport_scale import TRIALS as TRANSPORT, validate_finalist
from retune_neuroko_v12 import BASE, RELEASE, V12

TRIALS = {}
for g_name, gen in (("low", 0.010), ("high", 0.020)):
    for d_name, critic in (("low", 0.0105), ("high", 0.042)):
        for p_name, particles in (("low", 0.0203), ("high", 0.0812)):
            name = f"g-{g_name}-d-{d_name}-p-{p_name}"
            TRIALS[name] = {**BASE, "name": name, "lr": gen, "d_lr_mult": critic / gen,
                            "prior_lr_mult": particles / gen}

CENTER = dict(name="center", **BASE)
REFERENCES = [
    [RELEASE, V12, CENTER],
    [RELEASE, "unified_v11", CENTER],
    [RELEASE, "empf", dict(name="tuned-reference", lr=0.020, d_lr_mult=1.5, prior_lr_mult=2.9)],
]
RULE = (
    "Evaluate eight uniform-prior v12 rate combinations at seed 15 on the three training folds, with "
    "hollow coupling, relativistic weight 2, noise 0.025, default critic penalty and 1200 updates. "
    "Take the three lowest v12 energies. For each, run v11 and EMPF at the same three absolute rates. "
    "Keep candidates beating both matched controls, centered v12, centered v11 and independently "
    "tuned EMPF. Choose the lowest eligible v12 energy, breaking ties by trial name. Require that "
    "candidate to beat the matched controls and established uniform references across training seeds "
    "16–18 before scoring outer seeds 148–163. All role learning rates stay constant."
)


def paired(trial):
    empf = {k: trial[k] for k in ("name", "lr", "d_lr_mult", "prior_lr_mult")}
    return [[RELEASE, V12, trial], [RELEASE, "unified_v11", trial], [RELEASE, "empf", empf]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("data", "wheels", "output"):
        parser.add_argument("--" + field, type=Path, required=True)
    parser.add_argument("--workers-per-gpu", type=int, default=2)
    parser.add_argument("--window-cells", type=int)
    parser.add_argument("--stage", choices=["search", "confirm"], default="search")
    parser.set_defaults(require_constant_lr=True)
    args = parser.parse_args()
    for field in ("data", "wheels", "output"):
        setattr(args, field, getattr(args, field).resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage == "confirm":
        selection = json.loads((args.output / "selection.json").read_text())
        if not selection["advances"]:
            raise ValueError("no combination passed training validation")
        confirm_candidate(args, selection, TRIALS[selection["winner"]], list(range(148, 164)),
                          compare_historical_empf=True, previous_transport=TRANSPORT["q75-noise25"])
        return

    write_frozen(args.output / "search.json", dict(trials=TRIALS, groups=list(GROUPS), rule=RULE))
    candidates = {name: paired(t)[0] for name, t in TRIALS.items()}
    first = evaluate(args, [*candidates.values(), *REFERENCES], [15])
    finalists = sorted(candidates, key=lambda n: (mean(first[key(candidates[n])]), n))[:3]
    write_frozen(args.output / "screen.json", dict(energies=first, finalists=finalists))

    arms = [arm for n in finalists for arm in paired(TRIALS[n])]
    second = evaluate(args, [*arms, *REFERENCES], [15])
    reference_scores = [mean(second[key(ref)]) for ref in REFERENCES]
    eligible = []
    for n in finalists:
        own, v11, empf = [mean(second[key(arm)]) for arm in paired(TRIALS[n])]
        if own < min(v11, empf, *reference_scores):
            eligible.append(n)
    selected = min(eligible, key=lambda n: (mean(second[key(candidates[n])]), n)) if eligible else None
    decision = dict(screen=first, paired_screen=second, screened=finalists, finalist=selected,
                    validation={}, means={}, advances=False, winner=None)
    write_frozen(args.output / "paired_screen.json", {"energies": second, "finalist": selected})
    if selected is not None:
        decision.update(validate_finalist(args, TRIALS[selected], [16, 17, 18]))
    write_frozen(args.output / "selection.json", decision)
    print("Joint rate comparison", selected, decision["advances"], flush=True)


if __name__ == "__main__":
    main()
