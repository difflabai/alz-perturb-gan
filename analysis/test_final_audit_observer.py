"""Check the delayed audit boundary without loading any scientific outcomes."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import after_released097_final_audits as observer


class Gates(unittest.TestCase):
    def test_live_handle_prevents_audit_even_with_completed_peers(self):
        for name in observer.PARENTS:
            states = dict.fromkeys(observer.PARENTS, 'completed')
            states[name] = 'live'
            self.assertIsNone(observer.audit_plan(states, reservations_open=False))

    def test_open_reservation_prevents_every_audit(self):
        self.assertIsNone(observer.audit_plan(dict.fromkeys(observer.PARENTS, 'completed'), reservations_open=True))

    def test_full_completion_schedules_both_independent_audits(self):
        self.assertEqual(observer.audit_plan(dict.fromkeys(observer.PARENTS, 'completed'), reservations_open=False), ['benchmark_scalar', 'theta'])

    def test_theta_deferred_or_failed_keeps_prior_completed_studies_auditable(self):
        for status in ('failed', 'deferred'):
            self.assertEqual(observer.audit_plan(dict(benchmark='completed', scalar='completed', theta=status), reservations_open=False), ['benchmark_scalar'])

    def test_missing_or_failed_prior_study_never_reads_quality(self):
        with self.assertRaises(AssertionError):
            observer.audit_plan(dict(benchmark='completed', scalar='completed', theta='missing'), reservations_open=False)
        self.assertEqual(observer.audit_plan(dict(benchmark='failed', scalar='failed', theta='failed'), reservations_open=False), [])

    def test_live_exact_identity_wins_over_receipt_presence(self):
        expected = dict(pid=123, birth=456, boot='fixture')
        with patch.object(observer, 'identity', return_value=expected), patch.object(observer, 'read', side_effect=AssertionError('No receipt reads while live')):
            self.assertEqual(observer.parent_state('theta', expected), 'live')

    def test_absent_or_reused_pid_needs_real_terminal_receipt(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(observer, 'ROOT', Path(folder)), patch.object(observer, 'identity', return_value=None):
            self.assertEqual(observer.parent_state('theta', dict(pid=123)), 'missing')

    def test_unknown_or_incomplete_state_set_refused(self):
        for states in ({}, dict(benchmark='completed', scalar='completed', theta='unknown')):
            with self.assertRaises(AssertionError):
                observer.audit_plan(states, reservations_open=False)

    def test_conflicting_terminal_receipts_refused(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(observer, 'ROOT', Path(folder)), patch.object(observer, 'identity', return_value=None):
            base = Path(folder) / 'artifacts' / observer.PARENTS['theta'][0]
            base.mkdir(parents=True)
            (base / 'completed.json').write_text('{}')
            (base / 'failure.json').write_text('{}')
            with self.assertRaises(AssertionError):
                observer.parent_state('theta', dict(pid=123))

    def test_reused_pid_does_not_validate_wrong_completion_receipt(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(observer, 'ROOT', Path(folder)), patch.object(observer, 'identity', return_value=dict(pid=123, birth=999)):
            base = Path(folder) / 'artifacts' / observer.PARENTS['scalar'][0]
            base.mkdir(parents=True)
            (base / 'completed.json').write_text('{"status":"COMPLETED_OTHER_WORK"}')
            with self.assertRaises(AssertionError):
                observer.parent_state('scalar', dict(pid=123, birth=456))


if __name__ == '__main__':
    unittest.main()
