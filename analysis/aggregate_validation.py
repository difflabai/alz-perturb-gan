"""Aggregate the three predeclared seeds without ranking or replacing any seed."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path('/ml2/alz')
IDENTITY = ('donors', 'training_seed', 'generation_replicates', 'calibration_manifest_SHA256',
            'trace_manifest_SHA256', 'sampling_contract')


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def architecture(value):
    return dict(z_dim=value['z_dim'], stage_balance=value.get('stage_balance', False),
                cell_z_dim=value.get('cell_z_dim', 8), theta_init_scale=value.get('theta_init_scale', 1.),
                fourier_feature_scale=value.get('fourier_feature_scale', 1.))


def scientific_sources(auth):
    return {path: pin for path, pin in auth['research_source_SHA256'].items()
            if Path(path).parent in (ROOT, ROOT / 'model')}


def verified_deployment(attempt):
    deployment = read(ROOT / 'artifacts/allocations' / attempt / 'deployment.json')
    auth = read(deployment['authorization_path'])
    ledger = Path(deployment['ledger_path'])
    settlement = read(ledger.with_name(attempt + '-settlement.json'))
    job = read(ledger)['jobs'][0]
    assert settlement['uncertain'] is None and settlement['settlement']['status'] == 'settled'
    assert job['status'] == 'completed' and job['exit_code'] == 0 and job['reservation_released']
    candidate = read(Path(deployment['output_dir']) / 'completed.json')
    assert candidate['completed_updates'] == 20000 and candidate['integer_count_contract_passed']
    assert sha(candidate['checkpoint_path']) == candidate['checkpoint_SHA256']
    return deployment, auth, candidate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed1-attempt', required=True)
    parser.add_argument('--seed2-attempt', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), 'Keep existing confirmation evidence; never overwrite'
    incumbent_path = ROOT / 'incumbent.json'
    incumbent = read(incumbent_path)
    incumbent_pin = sha(incumbent_path)
    assert incumbent['training_seed'] == 0
    matches = [p.parent.name for p in (ROOT / 'artifacts/allocations').glob('*/deployment.json')
               if read(p)['allocation_id'] == incumbent['allocation_id']]
    assert len(matches) == 1, 'Frozen seed0 allocation is ambiguous'
    seed0_attempt = matches[0]
    result0 = read(ROOT / 'analysis' / (seed0_attempt + '-result.json'))
    assert result0['decision']['accept'] and result0['independent_saved_scores_checkpoint_and_count_exemplars_verified']
    baseline0 = read(ROOT / 'analysis/baseline_calibration_seed0.json')
    assert baseline0['both_original_seed0_arrays_bitwise_reproduced']
    _, auth0, candidate0 = verified_deployment(seed0_attempt)
    assert candidate0 == {k: v for k, v in incumbent.items() if k != 'candidate_commit'}
    records = []
    for seed, attempt in enumerate([seed0_attempt, args.seed1_attempt, args.seed2_attempt]):
        deployment, auth, candidate = verified_deployment(attempt)
        baseline = read(ROOT / 'analysis' / f'baseline_calibration_seed{seed}.json')
        for key in IDENTITY:
            assert candidate[key] == baseline[key], 'Unmatched baseline: ' + key
        assert candidate['training_seed'] == seed and candidate['generation_replicates'] == 2
        assert architecture(candidate['variant_config']) == architecture(incumbent['variant_config'])
        assert candidate.get('hybrid_config') == incumbent.get('hybrid_config')
        assert candidate['method'] == incumbent['method']
        assert scientific_sources(auth) == scientific_sources(auth0), 'Model/scorer sources changed between seeds'
        if seed:
            confirmation = read(ROOT / 'analysis' / (attempt + '-confirmation.json'))
            assert confirmation['status'] == 'SEED_CONFIRMATION_RECORDED_NO_MODEL_SELECTION'
            assert confirmation['candidate'] == candidate and confirmation['baseline'] == baseline
            assert confirmation['independently_verified_saved_scores_counts_and_checkpoint']
            assert auth['files_SHA256'][str(incumbent_path)] == incumbent_pin, 'Seed was not run against this frozen incumbent'
        records.append({'seed': seed, 'attempt': attempt, 'energy_score': candidate['energy_score'],
                        'matched_baseline_energy_score': baseline['energy_score'],
                        'cell_energy_distance': candidate['cell_energy_distance'],
                        'matched_baseline_cell_energy_distance': baseline['cell_energy_distance'],
                        'spread_ratio': candidate['spread_ratio'],
                        'joint_improvement': candidate['energy_score'] < baseline['energy_score']
                        and candidate['cell_energy_distance'] <= baseline['cell_energy_distance']
                        and .7 <= candidate['spread_ratio'] <= 1.3})
    improvement = 1 - sum(q['energy_score'] for q in records) / sum(q['matched_baseline_energy_score'] for q in records)
    passed = improvement >= .05 and all(q['joint_improvement'] for q in records)
    assert sha(incumbent_path) == incumbent_pin
    result = {'status': 'FROZEN_THREE_SEED_CALIBRATION_TARGET_PASSED' if passed else 'FROZEN_THREE_SEED_CALIBRATION_TARGET_NOT_MET',
              'stable_joint_target_passed': passed, 'mean_ES_relative_improvement': improvement,
              'records': records, 'incumbent_SHA256': incumbent_pin, 'incumbent_unchanged': True,
              'training_seeds_were_not_ranked': True,
              'scope': 'Same four adaptive calibration donors; three training seeds are not twelve independent donors. Evaluation donors excluded. No independent biological confirmation or external SOTA claim.'}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
