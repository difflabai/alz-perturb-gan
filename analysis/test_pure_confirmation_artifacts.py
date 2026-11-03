"""Private counterfactual metadata fixtures; no models, native jobs or real results."""
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import aggregate_pure_confirmation as aggregate
import aggregate_validation as common
from freeze_pure_confirmation import confirmation_identities


class ArtifactBindings(unittest.TestCase):
    def write(self, path, value):
        path = Path(path)
        assert path.resolve().is_relative_to(self.root.resolve())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2) + '\n')

    def setUp(self):
        self.real_pins = {p: common.sha(p) for p in [Path('/ml2/alz/pure_incumbent.json'), Path('/ml2/alz/incumbent.json')]}
        self.temp = tempfile.TemporaryDirectory(prefix='alz-pure-confirmation-metadata-')
        self.root = Path(self.temp.name)
        self.protocol = self.root / 'analysis/pure_confirmation_protocol.json'
        self.write(self.protocol, {'fixture': 'COUNTERFACTUAL_METADATA_NOT_A_MODEL_RESULT'})
        science = self.root / 'science'; cache = science / 'cache'
        self.native = science / 'native.txt'
        literals = [self.root / 'scorer.py', self.root / 'model/count.py', self.native, science / 'critic.txt']
        for p in literals:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text('SYNTHETIC_LITERAL_NOT_EXECUTED = True\n')
        self.sources = {str(p): common.sha(p) for p in literals[:2]}
        pins = {str(p): common.sha(p) for p in literals}
        for path, key, body in [
            (science / 'REVIEW_MANIFEST.json', 'source_SHA256', {str(literals[-1]): pins[str(literals[-1])]}),
            (science / 'NATIVE_SOURCE_PINS.json', 'files_SHA256', {str(self.native): pins[str(self.native)]})]:
            self.write(path, {key: body}); pins[str(path)] = common.sha(path)
        for folder in [cache, science / 'traces', self.root / 'data/calibration_fold0']:
            data = folder / 'synthetic.bin'; data.parent.mkdir(parents=True, exist_ok=True)
            data.write_bytes(b'COUNTERFACTUAL_BYTES_NOT_ALZ_DATA')
            manifest = folder / 'manifest.json'
            self.write(manifest, {'files_SHA256': {data.name: common.sha(data)}})
            pins.update({str(data): common.sha(data), str(manifest): common.sha(manifest)})
        base = {'metric_role': 'calibration_selection', 'donors': ['A', 'B', 'C', 'D'], 'generation_replicates': 2,
                'calibration_manifest_SHA256': common.sha(self.root / 'data/calibration_fold0/manifest.json'),
                'trace_manifest_SHA256': common.sha(science / 'traces/manifest.json'), 'sampling_contract': 'PRIVATE_METADATA_FIXTURE',
                'status': 'completed', 'completed_updates': 20000, 'fixed_endpoint': 20000, 'integer_count_contract_passed': True,
                'pure_only': True, 'real_bag_replacements': 0, 'hybrid_config': None, 'energy_score': 1.8,
                'cell_energy_distance': .2, 'spread_ratio': 1., 'variant_config': {'z_dim': 8},
                'method': 'pg_network_rate_half', 'native_recipe': 'bispf_optimized',
                'network_rate_config': {'lr': .006, 'prior_lr_mult': 5.}}
        self.attempt0 = 'CAL_GPU0_V1_fixture_seed0_f0_pg_s0'
        self.candidates = []; self.auths = []; self.deployments = []; self.baselines = []
        ids = None
        for seed in (0, 1, 2):
            if seed == 0:
                attempt = self.attempt0; allocation = 'ALZ_CAL_V1_fixture_seed0_f0_pg_s0'
            else:
                attempt = ids[str(seed)]['attempt_id']; allocation = ids[str(seed)]['allocation_id']
            output = self.root / 'runs' / attempt; checkpoint = output / 'fit/final.pt'
            checkpoint.parent.mkdir(parents=True); checkpoint.write_bytes(f'NOT_A_TORCH_MODEL_seed{seed}'.encode())
            candidate = {**copy.deepcopy(base), 'training_seed': seed, 'allocation_id': allocation,
                         'operation': 'train_network_rate' if seed == 0 else 'confirm_pure',
                         'checkpoint_path': str(checkpoint), 'checkpoint_SHA256': common.sha(checkpoint)}
            self.write(output / 'completed.json', candidate)
            if seed == 0:
                ids = confirmation_identities(attempt, candidate['checkpoint_SHA256'])
            ledger = self.root / 'ledgers' / (attempt + '.json')
            self.write(ledger, {'jobs': [{'status': 'completed', 'exit_code': 0, 'reservation_released': True}]})
            self.write(ledger.with_name(attempt + '-settlement.json'), {'uncertain': None, 'settlement': {'status': 'settled'}})
            auth = {'cpu_fixture': False, 'status': 'APPROVED_FOR_BOUNDED_EXECUTION',
                    'fixture_note': 'PRIVATE COUNTERFACTUAL WIRE FORMAT. NO NATIVE JOB EVER EXECUTED.',
                    'files_SHA256': dict(pins), 'research_source_SHA256': dict(self.sources),
                    'science_root': str(science), 'cache_dir': str(cache), 'operation': candidate['operation'],
                    'allocation_id': allocation, 'attempt_id': attempt}
            if seed:
                auth.update(cell=ids[str(seed)]['cell'], frozen_seed0_operation='train_network_rate')
            auth_path = self.root / 'artifacts/allocations' / attempt / 'authorization.json'
            deployment = {'allocation_id': allocation, 'authorization_path': str(auth_path),
                          'output_dir': str(output), 'ledger_path': str(ledger), 'endpoint': 20000}
            self.write(auth_path, auth)
            self.write(auth_path.with_name('deployment.json'), deployment)
            baseline = {**copy.deepcopy(base), 'training_seed': seed, 'energy_score': 2., 'cell_energy_distance': .25,
                        'both_original_seed0_arrays_bitwise_reproduced': True}
            self.write(self.root / 'analysis' / f'baseline_calibration_seed{seed}.json', baseline)
            self.candidates.append(candidate); self.auths.append(auth); self.deployments.append(deployment); self.baselines.append(baseline)
        self.ids = ids
        selected = self.root / 'pure_incumbent.json'
        self.write(selected, {**self.candidates[0], 'candidate_commit': 'PRIVATE_FIXTURE_ONLY'})
        decision_path = self.root / 'analysis' / (self.attempt0 + '-result.json')
        self.write(decision_path, {'candidate': self.candidates[0], 'decision': {'accept': True},
                                   'independent_saved_scores_checkpoint_and_count_exemplars_verified': True})
        self.freeze = self.root / 'analysis/frozen.json'
        self.write(self.freeze, {'status': 'FROZEN_PURE_THREE_SEED_CALIBRATION_CONFIRMATION',
            'training_seeds': [0, 1, 2], 'confirmation_training_seeds': [1, 2], 'ranking_seeds_allowed': False,
            'pure_confirmation_protocol_SHA256': common.sha(self.protocol), 'pure_incumbent_SHA256': common.sha(selected),
            'seed0_attempt': self.attempt0, 'seed0_candidate': self.candidates[0], 'confirmation_identities': ids,
            'seed0_authorization_SHA256': common.sha(self.deployments[0]['authorization_path']),
            'seed0_decision_SHA256': common.sha(decision_path), 'scientific_source_SHA256': self.sources,
            'matched_baselines': {str(s): {'path': str(self.root / 'analysis' / f'baseline_calibration_seed{s}.json'),
                                         'SHA256': common.sha(self.root / 'analysis' / f'baseline_calibration_seed{s}.json')} for s in (0, 1, 2)}})
        for seed in (1, 2):
            auth = self.auths[seed]; auth['pure_confirmation_freeze_SHA256'] = common.sha(self.freeze)
            auth['files_SHA256'].update({str(self.freeze): common.sha(self.freeze), str(selected): common.sha(selected)})
            self.write(self.deployments[seed]['authorization_path'], auth)
            audit_path = self.root / 'analysis' / f'checkpoint-seed{seed}-fixture.json'
            self.write(audit_path, {'status': 'VERIFIED_FROZEN_PURE_CONFIRMATION_CHECKPOINT', 'training_seed': seed,
                'checkpoint_SHA256': self.candidates[seed]['checkpoint_SHA256'], 'seed0_checkpoint_SHA256': self.candidates[0]['checkpoint_SHA256'],
                'recipe_exactly_matches_frozen_seed0': True, 'complete_model_keys_shapes_dtypes_finite_verified': True,
                'policy_update_count_and_seed_streams_verified': True})
            self.write(self.root / 'analysis' / (ids[str(seed)]['attempt_id'] + '-confirmation.json'), {
                'status': 'PURE_SEED_CONFIRMATION_RECORDED_NO_MODEL_SELECTION', 'candidate': self.candidates[seed],
                'baseline': self.baselines[seed], 'pure_confirmation_freeze_SHA256': common.sha(self.freeze),
                'independently_verified_saved_scores_counts_and_checkpoint': True, 'full_checkpoint_and_native_recipe_verified': True,
                'owned_release_and_clean_settlement_verified': True, 'training_seed_was_not_ranked': True, 'incumbent_unchanged': True,
                'checkpoint_audit': {'path': str(audit_path), 'SHA256': common.sha(audit_path)}})
        self.output = self.root / 'analysis/aggregate-PRIVATE-FIXTURE.json'

    def tearDown(self):
        self.assertEqual(self.real_pins, {p: common.sha(p) for p in self.real_pins})
        self.temp.cleanup()

    def run_aggregate(self, seed1=None):
        argv = ['aggregate', '--freeze', str(self.freeze), '--seed1-attempt', seed1 or self.ids['1']['attempt_id'],
                '--seed2-attempt', self.ids['2']['attempt_id'], '--output', str(self.output)]
        with patch.object(aggregate, 'ROOT', self.root), patch.object(aggregate, 'PROTOCOL', self.protocol), \
             patch.object(common, 'ROOT', self.root), patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()):
            aggregate.main()

    def test_complete_private_wire_artifacts_aggregate_without_touching_real_incumbents(self):
        self.run_aggregate(); q = common.read(self.output)
        self.assertTrue(q['stable_joint_target_passed'])
        self.assertEqual([r['seed'] for r in q['records']], [0, 1, 2])
        with self.assertRaises(AssertionError):
            self.run_aggregate()

    def test_unplanned_repeat_attempt_refuses_without_output(self):
        with self.assertRaises(AssertionError):
            self.run_aggregate(self.ids['1']['attempt_id'] + '_repeat')
        self.assertFalse(self.output.exists())

    def test_native_source_drift_refuses(self):
        self.native.write_text('CHANGED_PRIVATE_NATIVE_LITERAL\n')
        with self.assertRaises(AssertionError):
            self.run_aggregate()
        self.assertFalse(self.output.exists())

    def test_changed_checkpoint_refuses(self):
        Path(self.candidates[1]['checkpoint_path']).write_bytes(b'CHANGED_PRIVATE_CHECKPOINT')
        with self.assertRaises(AssertionError):
            self.run_aggregate()
        self.assertFalse(self.output.exists())

    def test_missing_confirmation_checkpoint_audit_refuses(self):
        path = self.root / 'analysis' / (self.ids['1']['attempt_id'] + '-confirmation.json')
        Path(common.read(path)['checkpoint_audit']['path']).unlink()
        with self.assertRaises(FileNotFoundError):
            self.run_aggregate()
        self.assertFalse(self.output.exists())

    def test_uncertain_settlement_refuses(self):
        path = Path(self.deployments[1]['ledger_path']).with_name(self.ids['1']['attempt_id'] + '-settlement.json')
        q = common.read(path); q['uncertain'] = 'PRIVATE_FIXTURE_UNCERTAINTY'; self.write(path, q)
        with self.assertRaises(AssertionError):
            self.run_aggregate()
        self.assertFalse(self.output.exists())

    def test_one_regressed_seed_is_retained_as_failed_confirmation(self):
        q = self.candidates[1]; q['energy_score'] = 2.05
        self.write(Path(self.deployments[1]['output_dir']) / 'completed.json', q)
        path = self.root / 'analysis' / (self.ids['1']['attempt_id'] + '-confirmation.json')
        confirmation = common.read(path); confirmation['candidate'] = q; self.write(path, confirmation)
        self.run_aggregate(); result = common.read(self.output)
        self.assertGreater(result['mean_ES_relative_improvement'], .05)
        self.assertFalse(result['stable_joint_target_passed'])
        self.assertEqual(result['per_seed_joint_improvement'], [True, False, True])
        self.assertEqual([r['seed'] for r in result['records']], [0, 1, 2])

    def test_CPU_fixture_flag_cannot_be_aggregated_as_native_model_evidence(self):
        q = self.auths[1]; q['cpu_fixture'] = True; self.write(self.deployments[1]['authorization_path'], q)
        with self.assertRaises(AssertionError):
            self.run_aggregate()
        self.assertFalse(self.output.exists())

    def test_changed_real_selection_identity_refuses(self):
        selected = self.root / 'pure_incumbent.json'; q = common.read(selected)
        q['candidate_commit'] = 'CHANGED_PRIVATE_SELECTION'; self.write(selected, q)
        with self.assertRaises(AssertionError):
            self.run_aggregate()
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
