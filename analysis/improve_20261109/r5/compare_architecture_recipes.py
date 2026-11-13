"""Numeric final-recipe gates; never changes the default or approves an arm.

Implements PROTOCOL_ARCHITECTURE_FINAL_SELECTION_20261113.txt. Architecture
screen/confirmation, artifact audit and complementary-direction diagnostics
remain separate requirements. Both references stay fixed at the promotion
comparison; passing here alone is insufficient for default promotion.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from pzero_compare import paired

BASE = Path('/ml2/alz/runs/improve_20261109/results_r5/architecture_pass/shape_confirmation')
EVIDENCE = Path(__file__).resolve().parents[1]/'default_config_distribution_evidence_20261113.json'
PROTOCOL = EVIDENCE.parent/'PROTOCOL_ARCHITECTURE_FINAL_SELECTION_20261113.txt'
# Promotion evidence from dac8c95. A later promotion cannot silently rebase
# either reference or its total profile-error allowance.
EVIDENCE_SHA256 = '3fb5bcb23efd58c47b25946a00cf63cd051b40f9b86d1b9ad4f7ec62498dcbdc'
EXPECTED = {(f, s) for f in range(3, 9) for s in (0, 1)}
BIASCAL = 'biascal@stagehalf'
COUNT_SHAPE = 'cellshape@0.5:biascal:stagehalf'
GUARDS = ('meanlog_rmse_gene', 'nonzero_rmse_gene', 'logvar_rmse_gene',
          'pc_energy_rep', 'cell_energy_rep', 'abs_log_spread', 'combined_coverage_distance')
METRICS = (*GUARDS, 'sd_interval_score', 'sd_coverage_distance', 'pb_rmse')


def read_scores(directory):
    return {(int(p.parent.name[4:]), int(p.stem[4:])): json.loads(p.read_text())
            for p in Path(directory).glob('fold*/seed*.json')}


def fixed_references():
    raw = EVIDENCE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != EVIDENCE_SHA256:
        raise ValueError('original promotion evidence changed; references cannot be rebased')
    artifacts = json.loads(raw)['artifact_audit']['artifacts']
    rows = {'default': {}, 'count_shape': {}}
    for artifact in artifacts:
        label, key = artifact['label'], (artifact['fold'], artifact['seed'])
        path = BASE/label/f'fold{key[0]}/seed{key[1]}.json'
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != artifact['score_sha256']:
            raise ValueError(f'frozen reference score changed: {path}')
        if key in rows[label]:
            raise ValueError('duplicate frozen reference score')
        rows[label][key] = json.loads(raw)
    validate(rows['default'], 'original profile reference', BIASCAL)
    validate(rows['count_shape'], 'fixed distribution reference', COUNT_SHAPE)
    return rows['count_shape'], rows['default']


def metric(row, name):
    held = row['intervals']['heldout']['percentile']
    if name == 'sd_interval_score':
        return float(held['sd']['interval_score'])
    if name.endswith('_coverage_distance'):
        return abs(float(held[name.split('_')[0]]['coverage']) - .8)
    if name == 'abs_log_spread':
        spread = float(row['historical']['pc_spread_rep'])
        if spread <= 0:
            raise ValueError('spread ratio must be positive')
        return abs(float(np.log(spread)))
    return float(row['historical'][name])


def validate(rows, label, recipe=None):
    if set(rows) != EXPECTED:
        raise ValueError(f'{label}: exactly folds3-8,seeds0/1 required; '
                         f'missing={sorted(EXPECTED-set(rows))}, unexpected={sorted(set(rows)-EXPECTED)}')
    recipes = {row.get('generation_recipe') for row in rows.values()}
    if len(recipes) != 1 or not recipes <= {BIASCAL, COUNT_SHAPE} or (recipe and recipes != {recipe}):
        raise ValueError(f'{label}: wrong or mixed readout: {recipes}')
    for (fold, seed), row in rows.items():
        if row.get('fold') != fold or row.get('M') != 32 or row.get('generation_device') != 'cpu':
            raise ValueError(f'{label} fold{fold}/seed{seed}: needs matched CPU,M32 score')
        if not all(np.isfinite(metric(row, name)) and metric(row, name) >= 0 for name in METRICS):
            raise ValueError(f'{label}: nonfinite or negative decision metric')
        for split in ('train', 'heldout'):
            for group in ('mean', 'sd', 'combined'):
                values = row['intervals'][split]['percentile'][group]
                coverage, width, score = (float(values[name]) for name in ('coverage', 'width', 'interval_score'))
                if not (np.isfinite([coverage, width, score]).all() and 0 <= coverage <= 1
                        and width >= 0 and score >= width - 1e-10):
                    raise ValueError(f'{label}: invalid interval summary')


def difference(candidate, reference, name):
    return paired({fold: [metric(candidate[(fold, seed)], name) - metric(reference[(fold, seed)], name)
                          for seed in (0, 1)] for fold in range(3, 9)})


def repeats(block):
    return {str(repeat): float(np.mean([value for fold, value in block['per_fold'].items()
                                       if int(fold) // 3 == repeat])) for repeat in (1, 2)}


def summaries(rows):
    return {'decision_metric_means': {name: float(np.mean([metric(row, name) for row in rows.values()]))
                                     for name in METRICS},
            'intervals': {split: {group: {name: float(np.mean([
                row['intervals'][split]['percentile'][group][name] for row in rows.values()]))
                for name in ('coverage', 'width', 'interval_score')}
                for group in ('mean', 'sd', 'combined')} for split in ('train', 'heldout')}}


def compare(candidate, distribution_reference, profile_reference):
    validate(candidate, 'candidate')
    validate(distribution_reference, 'fixed distribution reference', COUNT_SHAPE)
    validate(profile_reference, 'original profile reference', BIASCAL)
    for key in sorted(EXPECTED):
        fingerprints = [rows[key]['fingerprint'] for rows in
                        (candidate, distribution_reference, profile_reference)]
        for field in ('cache_sha256', 'score_code_sha256', 'interval_code_sha256'):
            if not fingerprints[0].get(field) or len({f.get(field) for f in fingerprints}) != 1:
                raise ValueError(f'{key}: unmatched {field}')
    distribution = {name: difference(candidate, distribution_reference, name) for name in METRICS}
    profile = difference(candidate, profile_reference, 'pb_rmse')
    score, coverage = distribution['sd_interval_score'], distribution['sd_coverage_distance']
    score_repeats, coverage_repeats = repeats(score), repeats(coverage)
    guards = {name: block['mean_diff'] <= 2 * block['se_seed_noise']
              for name in GUARDS for block in [distribution[name]]}
    profile_mean = float(np.mean([metric(row, 'pb_rmse') for row in profile_reference.values()]))
    if profile_mean <= 0:
        raise ValueError('positive original profile reference required')
    fold_relative = {fold: float(cost / np.mean([metric(profile_reference[(fold, s)], 'pb_rmse')
                                                for s in (0, 1)]))
                     for fold, cost in profile['per_fold'].items()}
    upper = profile['mean_diff'] + 2 * profile['se_seed_noise']
    gates = {
        'sd_score': bool(score['mean_diff'] < -2 * score['se_seed_noise']
                         and score['n_folds_lower'] >= 5 and all(v < 0 for v in score_repeats.values())),
        'sd_coverage_distance': bool(coverage['mean_diff'] < 0 and
                                     all(v < 0 for v in coverage_repeats.values())),
        'profile_budget_from_original_reference': bool(upper <= .005 * profile_mean and
                                                       all(v <= .005 for v in fold_relative.values())),
        'distribution_guards': bool(all(guards.values())),
    }
    return {'complete_pairs': 12, 'passes_numeric_final_recipe_rule': all(gates.values()),
            'gates': gates, 'distribution_guards': guards, 'distribution_comparison': distribution,
            'sd_score_repeat_differences': score_repeats, 'sd_coverage_distance_repeat_differences': coverage_repeats,
            'profile_comparison_against_original_reference': profile,
            'profile_reference_mean': profile_mean, 'profile_cost_upper_2seedSE': upper,
            'profile_allowance': .005 * profile_mean, 'profile_relative_cost_by_fold': fold_relative,
            'means': {label: summaries(rows) for label, rows in (
                ('candidate', candidate), ('fixed_distribution_reference', distribution_reference),
                ('original_profile_reference', profile_reference))},
            'scope': 'development-conditioned comparison on the same37 donors; seed SE is computational noise',
            'additional_requirements': ['original architecture screen', 'extra-fold architecture confirmation',
                'artifact/source/receipt audit including K256 and fixed random streams',
                'fixed random-direction diagnostic', 'deterministic tie selection if needed',
                'actual CLI verification, documentation and commit before default change']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    distribution, profile = fixed_references()
    value = compare(read_scores(args.candidate), distribution, profile)
    value['candidate_directory'] = str(args.candidate.resolve())
    value['reference_directories'] = {'distribution': str(BASE/'count_shape'), 'profile': str(BASE/'default')}
    value['original_promotion_evidence_sha256'] = EVIDENCE_SHA256
    value['comparison_code_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    value['protocol_sha256'] = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    value['candidate_score_sha256'] = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in sorted(args.candidate.glob('fold*/seed*.json'))}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    print(json.dumps({key: value[key] for key in ('complete_pairs', 'passes_numeric_final_recipe_rule', 'gates')}, indent=2))


if __name__ == '__main__':
    main()
