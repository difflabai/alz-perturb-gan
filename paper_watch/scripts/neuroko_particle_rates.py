"""Compare particle learning rates at two fixed critic rates on paired development fits."""
from copy import copy
import json
from statistics import mean

from compare_neuroko_v12 import GROUPS, write_frozen
from neuroko_role_factorial import paired
from neuroko_stratified import evaluate, key


def particle_arms(baseline):
    trials = [dict(baseline, name=baseline["name"] + suffix, prior_lr_mult=rate / baseline["lr"])
              for suffix, rate in (("-p0287", .0287), ("-p0574", .0574))]
    return [paired(trial)[0] for trial in (trials[0], baseline, trials[1])]


def select_particle_rate(cohorts, candidates, references):
    gaps = {}
    for name in candidates:
        gaps[name] = {
            cohort: mean(scores[name]) - min(mean(scores[ref]) for ref in references)
            for cohort, scores in cohorts.items()
        }
    winner = min(candidates, key=lambda name: (max(gaps[name].values()), name))
    return dict(winner=winner, gaps=gaps, passes=max(gaps[winner].values()) < 0)


def particle_rate_grid(args, calibrated, original, wide_noise, first_seeds):
    low = particle_arms(calibrated)
    high = particle_arms(original)
    refs = [low[1], *paired(wide_noise)[1:]]
    plan = dict(calibrated_critic=low, original_critic=high, references=refs, groups=list(GROUPS),
                first=list(first_seeds), second=[251, 252, 253], fresh=list(range(728, 734)),
                outer=list(range(734, 750)),
                rule="At fixed generator LR 0.014 compare particle LRs 0.0287, 0.0406 and 0.0574 "
                     "at critic LR 0.0049 on both seed cohorts. Choose the alternative whose worst "
                     "cohort mean gap to the best of the baseline and both tuned controls is smallest. "
                     "Break ties by arm name. Require a negative gap on each cohort. Compare paired "
                     "particle-rate effects to the same sweep at critic LR 0.021 on the first cohort. "
                     "A qualifying candidate must beat all matched and tuned references on each cohort "
                     "and on fresh training seeds before the biological holdout.")
    write_frozen(args.output / "particle_grid.json", plan)
    target = args.output / "particle_scores.json"
    if args.stage == "search":
        local = copy(args)
        local.output = args.shared_screen or args.output
        first = evaluate(local, [*low, *high, *refs[1:]], plan["first"])
        local.output = args.second_screen or args.output
        second = evaluate(local, [*low, *refs[1:]], plan["second"])
        cohorts = dict(first=first, second=second)
        result = select_particle_rate(cohorts, [key(low[i]) for i in (0, 2)], [key(arm) for arm in refs])
        effects = {}
        for i in (0, 2):
            low_alt, low_base, high_alt, high_base = (first[key(arm)] for arm in (low[i], low[1], high[i], high[1]))
            effects[key(low[i])] = [(la - lb) - (ha - hb)
                                    for la, lb, ha, hb in zip(low_alt, low_base, high_alt, high_base, strict=True)]
        write_frozen(target, dict(**result, scores=cohorts, paired_critic_by_particle_interaction=effects))
    result = json.loads(target.read_text())
    if result["passes"]:
        return next(arm[2] for arm in low if key(arm) == result["winner"]), plan
    if args.stage != "search":
        raise ValueError("Neither particle rate beat all preliminary controls on both development cohorts")
    write_frozen(args.output / "selection.json", dict(advances=False, winner=None, preliminary=result))
    return None, plan
