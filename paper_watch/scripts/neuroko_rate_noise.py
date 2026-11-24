"""Select a rate and latent-noise combination using paired comparisons across three seeds."""
import argparse
import json
from pathlib import Path
from statistics import mean

from compare_neuroko_v12 import GROUPS, write_frozen
from neuroko_role_factorial import REFERENCES, TRIALS as RATES, paired
from neuroko_stratified import confirm_candidate, evaluate, key
from neuroko_transport_scale import TRIALS as TRANSPORT, validate_finalist

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("data", "wheels", "output"):
        parser.add_argument("--" + field, type=Path, required=True)
    parser.add_argument("--workers-per-gpu", type=int, default=2)
    parser.add_argument("--window-cells", type=int)
    parser.add_argument("--stage", choices=("search", "confirm"), default="search")
    parser.set_defaults(require_constant_lr=True)
    args = parser.parse_args()
    for field in ("data", "wheels", "output"):
        setattr(args, field, getattr(args, field).resolve())
    args.output.mkdir(parents=True, exist_ok=True)

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
