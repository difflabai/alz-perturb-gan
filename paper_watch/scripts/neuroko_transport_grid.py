"""Search the released coupling's critic-score coefficient and transport entropy."""
from copy import copy
import json

from compare_neuroko_v12 import GROUPS, write_frozen
from neuroko_role_factorial import paired
from neuroko_stratified import evaluate, key, rank_cohort_gaps


def transport_arms(baseline):
    return [paired(dict(baseline, name=f"{baseline['name']}-tau{tau}-eps{round(eps*100):02}",
                        rhbi_tau=float(tau), rhbi_eps=eps))[0]
            for tau in (1, 2, 4) for eps in (.05, .1, .2) if (tau, eps) != (2, .1)]


def transport_grid(args, baseline, wide_noise, first_seeds):
    candidates = transport_arms(baseline)
    half = dict(baseline, name=baseline["name"] + "-rel5", rhbi_relativistic_weight=.5)
    refs = [paired(baseline)[0], paired(half)[0], *paired(wide_noise)[1:]]
    ref_keys = [key(arm) for arm in refs]
    plan = dict(candidates=candidates, references=refs, groups=list(GROUPS),
                first=list(first_seeds), second=[251, 252, 253], fresh=list(range(750, 756)),
                outer=list(range(756, 772)),
                rule="Cross critic-score coefficient tau 1, 2 or 4 with entropy epsilon 0.05, 0.1 or 0.2. "
                     "Use the released 2/0.1 combination as the baseline. Screen each new combination on "
                     "all three groups at seeds 22 and 251. Keep the two with the smallest worst-seed gap "
                     "to the best of the baseline, hollow weight 0.5, tuned v11 and tuned EMPF. "
                     "Evaluate both finalists on the other four development seeds. Select the smallest "
                     "worst gap over the remaining seeds of each cohort and each full cohort, breaking "
                     "ties by arm name. All four gaps must be negative before testing the winner against "
                     "every matched and tuned control on both development cohorts and six fresh seeds.")
    write_frozen(args.output / "transport_grid.json", plan)
    target = args.output / "transport_finalists.json"
    if args.stage == "search":
        local = copy(args)
        screen, remaining, complete = {}, {}, {}
        paths = {"first": args.shared_screen or args.output, "second": args.second_screen or args.output}
        for cohort, path in paths.items():
            local.output = path
            screen[cohort] = evaluate(local, [*candidates, *refs], plan[cohort][:1])
        ranks, gaps = rank_cohort_gaps(screen, [key(arm) for arm in candidates], ref_keys)
        write_frozen(args.output / "transport_shortlist.json",
                     dict(scores=screen, gaps=gaps, shortlist=ranks[:2]))
        finalists = [arm for arm in candidates if key(arm) in ranks[:2]]
        for cohort, path in paths.items():
            local.output = path
            remaining[cohort] = evaluate(local, [*finalists, *refs], plan[cohort][1:])
            complete[cohort] = {name: screen[cohort][name] + scores for name, scores in remaining[cohort].items()}
        cohorts = {**complete, **{name + "_remaining": rows for name, rows in remaining.items()}}
        ranking, gaps = rank_cohort_gaps(cohorts, ranks[:2], ref_keys)
        winner = ranking[0]
        write_frozen(target, dict(scores=complete, fresh_development_scores=remaining, gaps=gaps, winner=winner,
                                  passes=max(gaps[winner].values()) < 0))
    result = json.loads(target.read_text())
    if result["passes"]:
        return next(arm[2] for arm in candidates if key(arm) == result["winner"]), plan
    if args.stage != "search":
        raise ValueError("Neither transport finalist qualified on both development cohorts")
    write_frozen(args.output / "selection.json", dict(advances=False, winner=None, preliminary=result))
    return None, plan
