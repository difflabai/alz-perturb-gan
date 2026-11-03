"""Counterfactual arithmetic/seed rules only; never publish a model result."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from aggregate_pure_confirmation import frozen_input_pins, read, sha, stable_target


class PureAggregation(unittest.TestCase):
    def setUp(self):
        self.rows = [{'seed': seed, 'energy_score': 1.8, 'matched_baseline_energy_score': 2.,
                      'cell_energy_distance': .2, 'matched_baseline_cell_energy_distance': .25,
                      'spread_ratio': 1.} for seed in range(3)]

    def test_all_three_joint_improvements_and_mean_target_pass(self):
        q = stable_target(self.rows)
        self.assertTrue(q['stable_joint_target_passed'])
        self.assertAlmostEqual(q['mean_ES_relative_improvement'], .1)
        self.assertEqual(q['per_seed_joint_improvement'], [True] * 3)

    def test_cannot_drop_duplicate_reorder_or_substitute_a_seed(self):
        for rows in [self.rows[:2], self.rows + [self.rows[0]], list(reversed(self.rows)),
                     [self.rows[0], self.rows[0], self.rows[2]],
                     [{**self.rows[0], 'seed': False}, *self.rows[1:]]]:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                stable_target(rows)

    def test_one_regressed_seed_fails_even_when_average_is_good(self):
        for key, value in [('energy_score', 2.), ('cell_energy_distance', .25001),
                           ('spread_ratio', .69999), ('spread_ratio', 1.30001)]:
            rows = copy.deepcopy(self.rows); rows[1][key] = value
            with self.subTest(key=key, value=value):
                q = stable_target(rows)
                self.assertFalse(q['stable_joint_target_passed'])
                self.assertFalse(q['per_seed_joint_improvement'][1])
        rows = [{**q, 'energy_score': 1.92} for q in self.rows]
        q = stable_target(rows)
        self.assertEqual(q['per_seed_joint_improvement'], [True] * 3)
        self.assertFalse(q['stable_joint_target_passed'])

    def test_nonfinite_boolean_missing_or_nonpositive_baseline_refuses(self):
        for key in ['energy_score', 'matched_baseline_energy_score', 'cell_energy_distance',
                    'matched_baseline_cell_energy_distance', 'spread_ratio']:
            for value in [True, None, float('nan'), float('inf')]:
                rows = copy.deepcopy(self.rows); rows[2][key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    stable_target(rows)
        for value in [0., -1.]:
            rows = copy.deepcopy(self.rows); rows[0]['matched_baseline_energy_score'] = value
            with self.assertRaises(ValueError):
                stable_target(rows)

    def real_binding(self):
        root = Path('/ml2/alz')
        deployment = read(root / 'artifacts/allocations/CAL_GPU0_V1_learned_obs_v1_f0_pg_s0/deployment.json')
        return read(deployment['authorization_path']), read(Path(deployment['output_dir']) / 'completed.json')

    def test_actual_closed_attempt_immutable_input_binding_without_model_import(self):
        auth, candidate = self.real_binding()
        pins = frozen_input_pins(auth, candidate)
        self.assertGreater(len(pins), 400)
        self.assertEqual(pins[str(Path(auth['science_root']) / 'traces/manifest.json')], candidate['trace_manifest_SHA256'])
        self.assertNotIn('torch', sys.modules)

    def test_source_drift_and_changed_trace_identity_refuse_without_writes(self):
        auth, candidate = self.real_binding()
        manifest = Path(auth['science_root']) / 'REVIEW_MANIFEST.json'
        with patch('aggregate_pure_confirmation.sha', lambda path: '0' * 64 if Path(path) == manifest else sha(path)):
            with self.assertRaises(AssertionError):
                frozen_input_pins(auth, candidate)
        with self.assertRaises(AssertionError):
            frozen_input_pins(auth, {**candidate, 'trace_manifest_SHA256': '0' * 64})


if __name__ == '__main__':
    unittest.main()
