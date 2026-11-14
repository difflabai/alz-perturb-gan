"""Prospective numerical gates; empirical replicate noise is included."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]/'runs/improve_20261109/results_r5/empirical_challenge_20261114'
CANDIDATES = ('blind32', 'radial32')
HISTORICAL = ('pb_rmse', 'meanlog_rmse_gene', 'nonzero_rmse_gene',
              'logvar_rmse_gene', 'pc_energy_rep', 'cell_energy_rep')
GUARDS = (*HISTORICAL[1:], 'abs_log_spread')
TARGETS = ('sd/interval_score', 'combined/interval_score')
COVERAGE = ('sd/coverage_distance', 'combined/coverage_distance')
METRICS = (*HISTORICAL, 'abs_log_spread', *[
    f'{g}/{s}' for g in ('mean', 'sd', 'combined')
    for s in ('coverage_distance', 'width', 'interval_score')])


def value(row, metric):
    if metric in HISTORICAL:
        out = row['historical'][metric]
    elif metric == 'abs_log_spread':
        out = abs(math.log(row['historical']['pc_spread_rep']))
    else:
        group, stat = metric.split('/')
        block = row['intervals']['heldout']['percentile'][group]
        out = abs(block['coverage']-.8) if stat == 'coverage_distance' else block[stat]
    if not math.isfinite(out):
        raise ValueError('nonfinite comparison metric')
    return out


def paired(differences):
    folds = sorted(differences)
    if not folds or any(len(differences[f]) != 2 for f in folds):
        raise ValueError('exactly two complete computational replicates per fold required')
    per_fold = {f: statistics.mean(differences[f]) for f in folds}
    mean = statistics.mean(per_fold.values())
    noise = math.sqrt(sum(statistics.variance(differences[f])/2 for f in folds))/len(folds)
    scatter = statistics.stdev(per_fold.values())/math.sqrt(len(folds)) if len(folds)>1 else None
    return {'mean_diff': mean, 'per_fold': per_fold, 'se_computational_replicate': noise,
            'se_fold_scatter': scatter, 'n_folds_lower': sum(v<0 for v in per_fold.values()),
            'n_pairs': 2*len(folds)}


def metrics(candidate, reference, folds):
    return {metric: paired({f:[value(candidate[f, r], metric)-value(reference[f, r], metric)
                              for r in (0, 1)] for f in folds}) for metric in METRICS}


def means(rows):
    rows = list(rows.values())
    return {'historical':{m:statistics.mean(r['historical'][m] for r in rows)
                          for m in (*HISTORICAL, 'pc_spread_rep')},
            'intervals':{split:{g:{s:statistics.mean(r['intervals'][split]['percentile'][g][s] for r in rows)
                                  for s in ('coverage', 'below', 'above', 'width', 'interval_score')}
                              for g in ('mean', 'sd', 'combined')} for split in ('train', 'heldout')}}


def improves(block):
    return block['mean_diff'] < -2*block['se_computational_replicate']


def nonregression(block):
    return block['mean_diff'] <= 2*block['se_computational_replicate']


def verdict(candidate, current, empirical, original, folds, confirmation=False):
    expected = {(f, r) for f in folds for r in (0, 1)}
    if any(set(rows) != expected for rows in (candidate, current, empirical, original)):
        return {'status':'incomplete', 'passes':False, 'required_pairs':len(expected)}
    comparisons = {name:metrics(candidate, rows, folds) for name, rows in
                   (('current32', current), ('empirical32', empirical), ('original32', original))}
    gates = {}
    for name in ('current32', 'empirical32'):
        stats = comparisons[name]
        for target in TARGETS:
            block = stats[target]
            key = name+'/'+target
            gates[key+'/more_than_two_SE'] = improves(block)
            gates[key+'/fold_direction'] = block['n_folds_lower'] >= (5 if confirmation else 3)
            if confirmation:
                for repeat in (1, 2):
                    vals = [d for f, d in block['per_fold'].items() if f//3 == repeat]
                    gates[key+f'/repeat{repeat}'] = bool(vals) and statistics.mean(vals)<0
        for metric in COVERAGE:
            gates[name+'/'+metric+'/nonregression'] = nonregression(stats[metric])
    for metric in GUARDS:
        gates['current32/'+metric+'/nonregression'] = nonregression(comparisons['current32'][metric])
    profile = comparisons['original32']['pb_rmse']
    original_mean = statistics.mean(value(row, 'pb_rmse') for row in original.values())
    upper = profile['mean_diff']+2*profile['se_computational_replicate']
    gates['profile_total_half_percent_budget'] = upper <= .005*original_mean
    costs = {f:statistics.mean(value(candidate[f,r], 'pb_rmse') for r in (0,1))/
             statistics.mean(value(original[f,r], 'pb_rmse') for r in (0,1))-1 for f in folds}
    gates['profile_every_fold_half_percent_budget'] = all(cost<=.005 for cost in costs.values())
    return {'status':'complete', 'passes':all(gates.values()), 'n_pairs':len(expected),
            'gates':gates, 'comparisons':comparisons,
            'means':{'candidate':means(candidate), 'current32':means(current),
                     'empirical32':means(empirical), 'original32':means(original)},
            'profile_cost_upper_2computational_SE':upper,
            'profile_total_allowance':.005*original_mean, 'profile_relative_cost_by_fold':costs,
            'scope':'Same37 reused development donors; computational noise, not biological significance'}


def load(recipe, folds):
    rows, hashes = {}, {}
    expected_sources = json.loads((ROOT/'source_manifest.json').read_text())
    if any(hashlib.sha256(Path(p).read_bytes()).hexdigest() != h for p,h in expected_sources.items()):
        raise ValueError('queue source changed before comparison')
    for f in folds:
        for r in (0, 1):
            p = ROOT/recipe/f'fold{f}/seed{r}.json'
            if not p.exists():
                continue
            row = json.loads(p.read_text())
            meta = row.get('challenge', {})
            if (meta.get('recipe') != recipe or meta.get('fold') != f or meta.get('replicate') != r
                    or row['fold'] != f or row['M'] != (128 if recipe.endswith('128') else 32)
                    or not row.get('pruned_source_sha256') or meta.get('generation_device') != 'cpu'
                    or meta.get('sources') != expected_sources):
                raise ValueError('challenge score identity/metadata mismatch')
            rows[f, r] = row
            hashes[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    return rows, hashes


def report(folds, confirmation=False, eligible=None):
    reference, hashes = {}, {}
    for name in ('current32', 'empirical32', 'original32'):
        reference[name], found = load(name, folds)
        hashes.update(found)
    arms = {}
    for name in CANDIDATES:
        if confirmation and eligible is not None and name not in eligible:
            arms[name] = {'status':'not_triggered_screen_failed', 'passes':False}
            continue
        rows, found = load(name, folds)
        hashes.update(found)
        arms[name] = verdict(rows, reference['current32'], reference['empirical32'],
                             reference['original32'], folds, confirmation)
    passing = [name for name in CANDIDATES if arms[name]['passes']]
    passing.sort(key=lambda name:(arms[name]['means']['candidate']['intervals']['heldout']['combined']['interval_score'],
                                  arms[name]['means']['candidate']['intervals']['heldout']['sd']['interval_score'], name))
    ties=[]
    for i,name in enumerate(passing):
        for other in passing[i+1:]:
            left,_=load(name,folds);right,_=load(other,folds)
            paired_difference=metrics(left,right,folds)
            ties.append({'recipes':[name,other],'combined_within_two_computational_SE':
                abs(paired_difference['combined/interval_score']['mean_diff']) <=
                2*paired_difference['combined/interval_score']['se_computational_replicate'],
                'sd_within_two_computational_SE':abs(paired_difference['sd/interval_score']['mean_diff']) <=
                2*paired_difference['sd/interval_score']['se_computational_replicate'],
                'scope':'Computational tie report, not biological uncertainty or automatic promotion'})
    controls = {}
    for name in ('replay32', 'current128', 'empirical128'):
        rows, found = load(name, folds)
        hashes.update(found)
        if len(rows) == 2*len(folds):
            controls[name] = means(rows)
    return {'phase':'confirmation' if confirmation else 'screen', 'folds':list(folds),
            'arms':arms, 'passing_recipes_ranked':passing, 'passing_computational_ties':ties,
            'controls_not_eligible':controls,
            'score_sha256':hashes, 'default_changed':False,
            'note':'Numerical gates only; source/calibration artifact audit, independent review and CLI proof remain separate'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--confirmation', action='store_true')
    args = parser.parse_args()
    eligible=None
    if args.confirmation:
        screen=report((0,1,2))
        if any(row['status']!='complete' for row in screen['arms'].values()):
            raise ValueError('complete source-verified eligibility screen required')
        eligible=screen['passing_recipes_ranked']
    out = report(tuple(range(3,9)) if args.confirmation else (0,1,2), args.confirmation, eligible)
    ROOT.mkdir(parents=True, exist_ok=True)
    path = ROOT/('confirmation_comparison.json' if args.confirmation else 'screen_comparison.json')
    path.write_text(json.dumps(out, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'phase':out['phase'], 'passing':out['passing_recipes_ranked']}, indent=2))
