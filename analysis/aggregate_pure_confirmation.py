"""Aggregate a frozen pure model's exact seeds 0/1/2; never select a seed."""
import argparse
import json
import math
from pathlib import Path

from aggregate_validation import IDENTITY, architecture, read, sha, scientific_sources, verified_deployment
from freeze_pure_confirmation import confirmation_identities, seed0_gate

ROOT = Path('/ml2/alz')
PROTOCOL = ROOT / 'analysis/pure_confirmation_protocol.json'


def frozen_input_pins(auth, candidate):
    science = Path(auth['science_root'])
    manifests = [(science / 'REVIEW_MANIFEST.json', 'source_SHA256', False),
                 (science / 'NATIVE_SOURCE_PINS.json', 'files_SHA256', False),
                 (Path(auth['cache_dir']) / 'manifest.json', 'files_SHA256', True),
                 (science / 'traces/manifest.json', 'files_SHA256', True),
                 (ROOT / 'data/calibration_fold0/manifest.json', 'files_SHA256', True)]
    pins = {}
    for path, key, relative in manifests:
        pin = sha(path); assert pin == auth['files_SHA256'][str(path)]
        pins[str(path)] = pin
        for name, digest in read(path)[key].items():
            source = path.parent / name if relative else Path(name)
            assert sha(source) == digest == auth['files_SHA256'][str(source)]
            pins[str(source)] = digest
    assert pins[str(science / 'traces/manifest.json')] == candidate['trace_manifest_SHA256']
    assert pins[str(ROOT / 'data/calibration_fold0/manifest.json')] == candidate['calibration_manifest_SHA256']
    return pins


def stable_target(records):
    if [q.get('seed') for q in records] != [0, 1, 2] or any(type(q.get('seed')) is not int for q in records):
        raise ValueError('Require all predeclared seeds exactly once, in order 0/1/2')
    fields = ['energy_score', 'matched_baseline_energy_score', 'cell_energy_distance',
              'matched_baseline_cell_energy_distance', 'spread_ratio']
    if any(type(q.get(k)) not in (float, int) or not math.isfinite(q[k]) for q in records for k in fields):
        raise ValueError('Finite numerical measurements required for every seed')
    if any(q['matched_baseline_energy_score'] <= 0 for q in records):
        raise ValueError('Positive matched baseline energy score required')
    improvement = 1 - sum(q['energy_score'] for q in records) / sum(q['matched_baseline_energy_score'] for q in records)
    joint = [q['energy_score'] < q['matched_baseline_energy_score']
             and q['cell_energy_distance'] <= q['matched_baseline_cell_energy_distance']
             and .7 <= q['spread_ratio'] <= 1.3 for q in records]
    passed = improvement >= .05 and all(joint)
    return {'status': 'FROZEN_PURE_THREE_SEED_CALIBRATION_TARGET_PASSED' if passed else 'FROZEN_PURE_THREE_SEED_CALIBRATION_TARGET_NOT_MET',
            'stable_joint_target_passed': passed, 'mean_ES_relative_improvement': improvement,
            'per_seed_joint_improvement': joint}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--seed1-attempt', required=True)
    parser.add_argument('--seed2-attempt', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), 'Preserve existing confirmation evidence'
    freeze = read(args.freeze); freeze_pin = sha(args.freeze)
    assert freeze['status'] == 'FROZEN_PURE_THREE_SEED_CALIBRATION_CONFIRMATION'
    assert freeze['training_seeds'] == [0, 1, 2] and freeze['confirmation_training_seeds'] == [1, 2]
    assert freeze['ranking_seeds_allowed'] is False
    assert sha(PROTOCOL) == freeze['pure_confirmation_protocol_SHA256']
    incumbent_path = ROOT / 'pure_incumbent.json'
    assert sha(incumbent_path) == freeze['pure_incumbent_SHA256']
    attempts = [freeze['seed0_attempt'], args.seed1_attempt, args.seed2_attempt]
    assert len(set(attempts)) == 3
    deployment0, auth0, candidate0 = verified_deployment(attempts[0])
    assert auth0.get('cpu_fixture') is False and auth0['status'] == 'APPROVED_FOR_BOUNDED_EXECUTION'
    assert sha(deployment0['authorization_path']) == freeze['seed0_authorization_SHA256']
    assert candidate0 == freeze['seed0_candidate'] == {k: v for k, v in read(incumbent_path).items() if k != 'candidate_commit'}
    identities = confirmation_identities(attempts[0], candidate0['checkpoint_SHA256'])
    assert freeze['confirmation_identities'] == identities
    assert attempts[1:] == [identities[str(seed)]['attempt_id'] for seed in (1, 2)]
    assert seed0_gate(candidate0, read(ROOT / 'analysis/baseline_calibration_seed0.json'))['ready']
    decision_path = ROOT / 'analysis' / (attempts[0] + '-result.json')
    assert sha(decision_path) == freeze['seed0_decision_SHA256']
    decision = read(decision_path)
    assert decision['candidate'] == candidate0 and decision['decision']['accept']
    assert decision['independent_saved_scores_checkpoint_and_count_exemplars_verified']
    assert scientific_sources(auth0) == freeze['scientific_source_SHA256']
    assert all(sha(path) == pin for path, pin in freeze['scientific_source_SHA256'].items())
    immutable_pins = frozen_input_pins(auth0, candidate0)
    records = []; evidence_pins = {str(args.freeze.resolve()): freeze_pin, str(decision_path): sha(decision_path)}
    for seed, attempt in enumerate(attempts):
        deployment, auth, candidate = verified_deployment(attempt)
        assert auth.get('cpu_fixture') is False and auth['status'] == 'APPROVED_FOR_BOUNDED_EXECUTION'
        base_record = freeze['matched_baselines'][str(seed)]
        baseline_path = ROOT / 'analysis' / f'baseline_calibration_seed{seed}.json'
        assert str(baseline_path) == base_record['path'] and sha(baseline_path) == base_record['SHA256']
        baseline = read(baseline_path)
        for key in IDENTITY:
            assert candidate[key] == baseline[key], 'Unmatched seed baseline: ' + key
        assert candidate['metric_role'] == baseline['metric_role'] == 'calibration_selection'
        assert candidate['training_seed'] == seed and candidate['generation_replicates'] == 2
        assert candidate['status'] == 'completed'
        assert candidate['pure_only'] is True and candidate['real_bag_replacements'] == 0 and candidate.get('hybrid_config') is None
        assert candidate['completed_updates'] == candidate0['completed_updates'] == candidate['fixed_endpoint']
        assert architecture(candidate['variant_config']) == architecture(candidate0['variant_config'])
        for key in ['method', 'native_recipe', 'network_rate_config', 'semantic_config', 'learned_observation_config',
                    'released_hyperparameters', 'particlegan_version', 'penalty_options']:
            assert candidate.get(key) == candidate0.get(key), 'Frozen model configuration changed: ' + key
        if candidate0.get('particlegan_version') == '0.9.5':
            for key in ('released_hyperparameters', 'particlegan_version', 'penalty_options'):
                assert auth.get(key) == candidate0[key], 'Native authorization differs from frozen configuration: ' + key
        assert scientific_sources(auth) == freeze['scientific_source_SHA256']
        assert all(auth['files_SHA256'].get(path) == pin for path, pin in immutable_pins.items())
        if seed:
            assert auth['cell'] == identities[str(seed)]['cell']
            assert auth['attempt_id'] == identities[str(seed)]['attempt_id']
            assert auth['allocation_id'] == candidate['allocation_id'] == identities[str(seed)]['allocation_id']
            assert auth['operation'] == candidate['operation'] == 'confirm_pure'
            assert auth['frozen_seed0_operation'] == candidate0['operation']
            assert auth['pure_confirmation_freeze_SHA256'] == freeze_pin
            assert auth['files_SHA256'].get(str(args.freeze.resolve())) == freeze_pin
            assert auth['files_SHA256'][str(incumbent_path)] == freeze['pure_incumbent_SHA256']
            path = ROOT / 'analysis' / (attempt + '-confirmation.json'); confirmation = read(path)
            assert confirmation['status'] == 'PURE_SEED_CONFIRMATION_RECORDED_NO_MODEL_SELECTION'
            assert confirmation['candidate'] == candidate and confirmation['baseline'] == baseline
            assert confirmation['pure_confirmation_freeze_SHA256'] == freeze_pin
            for key in ['independently_verified_saved_scores_counts_and_checkpoint', 'full_checkpoint_and_native_recipe_verified',
                        'owned_release_and_clean_settlement_verified', 'training_seed_was_not_ranked', 'incumbent_unchanged']:
                assert confirmation[key] is True
            audit = confirmation['checkpoint_audit']; assert sha(audit['path']) == audit['SHA256']
            checkpoint_audit = read(audit['path'])
            assert checkpoint_audit['status'] == 'VERIFIED_FROZEN_PURE_CONFIRMATION_CHECKPOINT'
            assert checkpoint_audit['training_seed'] == seed and checkpoint_audit['checkpoint_SHA256'] == candidate['checkpoint_SHA256']
            assert checkpoint_audit['seed0_checkpoint_SHA256'] == candidate0['checkpoint_SHA256']
            for key in ['recipe_exactly_matches_frozen_seed0', 'complete_model_keys_shapes_dtypes_finite_verified',
                        'policy_update_count_and_seed_streams_verified']:
                assert checkpoint_audit[key] is True
            evidence_pins.update({str(path): sha(path), audit['path']: audit['SHA256']})
        records.append({'seed': seed, 'attempt': attempt, 'energy_score': candidate['energy_score'],
                        'matched_baseline_energy_score': baseline['energy_score'], 'cell_energy_distance': candidate['cell_energy_distance'],
                        'matched_baseline_cell_energy_distance': baseline['cell_energy_distance'], 'spread_ratio': candidate['spread_ratio']})
    result = {**stable_target(records), 'records': records, 'pure_confirmation_freeze_SHA256': freeze_pin,
              'incumbent_SHA256': freeze['pure_incumbent_SHA256'], 'incumbent_unchanged': True,
              'training_seeds_were_not_ranked': True, 'evidence_SHA256': evidence_pins,
              'scope': 'Same four adaptive calibration donors and all three predeclared training seeds. Not twelve independent donors, independent biological confirmation or external SOTA. Evaluation donors excluded.'}
    assert sha(incumbent_path) == freeze['pure_incumbent_SHA256'] and sha(args.freeze) == freeze_pin
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
