"""Estimator and reporting regression tests, not evidence of real theta fits."""
import copy
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path('/ml2/alz')
PROJECT = pathlib.Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark')
sys.path[:0] = [str(ROOT / 'analysis'), str(ROOT),
    str(PROJECT / 'current_base_20261029/runtime/lib/python3.12/site-packages'),
    str(PROJECT / '.venv312/lib/python3.12/site-packages')]
import numpy as np
import audit_dispersion097_completed_study as m


class ThetaFinalAudit(unittest.TestCase):
    def test_energy_score_uses_unbiased_pair_correction_and_ED_uses_V_statistic(self):
        y = np.zeros((4, 2, 6), dtype=np.float64)
        for stage in range(4):
            y[stage, :, 0] = [stage, stage + 2]
        real = np.zeros((4, 6))
        real[:, 0] = np.arange(4) + 1
        cells = np.zeros((4, 2, 1))
        cells[:, 1] = 2
        refs = np.ones((4, 2, 1))
        score = m.independent_metrics(y, cells, real, np.arange(4), refs.reshape(-1, 1), refs)
        self.assertEqual(score['energy_score'], 0.)
        self.assertEqual(score['per_donor_energy_score'], [0.] * 4)
        self.assertEqual(score['cell_energy_distance'], 1.)
        self.assertEqual(score['coverage80'], 1.)

    def test_independent_score_matches_declared_scorer_on_fullwidth_synthetic_arrays(self):
        import calibration
        rng = np.random.default_rng(331)
        y = rng.normal(size=(4, 200, 26)).astype(np.float32)
        c = rng.normal(size=(4, 375, 50)).astype(np.float32)
        _, targets, cells = calibration.targets()
        try:
            expected = calibration.score(y, c)
            actual = m.independent_metrics(y, c, targets['summary'], targets['stage'], cells['equal_stage'], cells['by_stage'])
            m.compare_metrics(actual, expected)
        finally:
            targets.close()
            cells.close()

    def test_nonfinite_or_degenerate_targets_refuse(self):
        y = np.zeros((4, 2, 6));c = np.zeros((4, 2, 1));real = np.ones((4, 6))
        with self.assertRaises(AssertionError):
            m.independent_metrics(y, c, real, np.arange(4), np.ones((8, 1)), c)
        y[0, 0, 0] = np.nan
        with self.assertRaises(AssertionError):
            m.independent_metrics(y, c, real, np.arange(4), np.ones((8, 1)), c)

    def fixture(self):
        rows = []
        for seed in m.SEEDS:
            for arm in m.ARMS:
                metrics = {k: (0.2 if arm == 'theta1' else 0.3) for k in m.METRICS}
                metrics['energy_score'] = 2. if arm == 'theta1' else 1.
                rows.append(dict(arm=arm, seed=seed, metrics=metrics,
                                 baseline_metrics={k: .4 for k in m.METRICS}))
        means = {a: rows[i]['metrics'].copy() for a, i in [('theta1', 0), ('theta5', 1)]}
        deltas = [dict(seed=s, **{k: means['theta5'][k] - means['theta1'][k] for k in m.METRICS}) for s in m.SEEDS]
        report = dict(mean_metrics_by_arm=means, paired_theta5_minus_theta1=deltas,
            paired_mean_differences={k: deltas[0][k] for k in m.METRICS},
            calibration_preferred_arm='theta1', theta5_ED_paired_wins=0)
        return rows, report

    def test_ED_decides_despite_better_donor_energy_score(self):
        rows, q = self.fixture()
        out = m.validate_summary(rows, q)
        self.assertEqual(out['calibration_preferred_arm'], 'theta1')
        self.assertEqual(len(out['PG_minus_matched_resampling_by_seed']), 6)

    def test_missing_negative_outcome_or_duplicate_seed_refuses(self):
        rows, q = self.fixture()
        with self.assertRaises(AssertionError):
            m.validate_summary(rows[:-1], q)
        rows[-1] = copy.deepcopy(rows[0])
        with self.assertRaises(AssertionError):
            m.validate_summary(rows, q)

    def test_changed_paired_delta_or_wrong_preferred_arm_refuses(self):
        rows, q = self.fixture();q['paired_theta5_minus_theta1'][1]['cell_energy_distance'] = -.1
        with self.assertRaises(AssertionError):
            m.validate_summary(rows, q)
        rows, q = self.fixture();q['calibration_preferred_arm'] = 'theta5'
        with self.assertRaises(AssertionError):
            m.validate_summary(rows, q)

    def test_exact_ED_tie_keeps_theta1(self):
        rows, q = self.fixture()
        for row in rows:
            row['metrics']['cell_energy_distance'] = .2
        q['mean_metrics_by_arm']['theta5']['cell_energy_distance'] = .2
        q['paired_mean_differences']['cell_energy_distance'] = 0.
        for row in q['paired_theta5_minus_theta1']:
            row['cell_energy_distance'] = 0.
        self.assertEqual(m.validate_summary(rows, q)['calibration_preferred_arm'], 'theta1')

    def test_matched_baseline_difference_refuses(self):
        rows, q = self.fixture();rows[1]['baseline_metrics']['energy_score'] += .1
        with self.assertRaises(AssertionError):
            m.validate_summary(rows, q)

    def test_count_contract_rejects_fractional_negative_nonfinite_and_wrong_width(self):
        counts = np.zeros((1, 400, 2000))
        m.count_contract(counts)
        for bad in (.5, -1., np.nan):
            counts[0, 0, 0] = bad
            with self.assertRaises(AssertionError):
                m.count_contract(counts)
        with self.assertRaises(AssertionError):
            m.count_contract(np.zeros((1, 400, 1999)))

    def test_matched_array_fingerprint_is_content_and_dtype_sensitive(self):
        a = dict(x=np.arange(4), y=np.ones(2))
        self.assertEqual(m.array_fingerprint(a), m.array_fingerprint(dict(reversed(list(a.items())))))
        b = copy.deepcopy(a);b['x'][0] = 1
        self.assertNotEqual(m.array_fingerprint(a), m.array_fingerprint(b))
        b = copy.deepcopy(a);b['x'] = b['x'].astype(np.float64)
        self.assertNotEqual(m.array_fingerprint(a), m.array_fingerprint(b))

    def test_partial_quality_is_not_decoded_and_presence_is_not_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory);paths = m.required_paths(root)
            for path in paths[:-1]:
                path.parent.mkdir(parents=True, exist_ok=True);path.write_bytes(b'OPAQUE_NOT_JSON')
            q = m.readiness(root)
            self.assertEqual(q['missing_paths'], [str(paths[-1])])
            self.assertFalse(q['quality_values_read'])
            paths[-1].parent.mkdir(parents=True, exist_ok=True)
            paths[-1].write_bytes(b'OPAQUE_NOT_JSON')
            q = m.readiness(root)
            self.assertEqual(q['status'], 'INPUT_PATHS_PRESENT_NOT_A_COMPLETION_CLAIM')
            self.assertFalse(q['full_goal_complete']);self.assertFalse(q['SOTA_claim'])


if __name__ == '__main__':
    unittest.main()
