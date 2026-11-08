"""Synthetic terminal-boundary cases; never an actual ALZ failure declaration."""
from copy import deepcopy
import unittest
import verify_cfgen_postfit_cap_failure as gate


def fixture():
    auth = dict(attempt_id=gate.ATTEMPT, cell=deepcopy(gate.CELL), initial_completed_updates=0,
                new_model_updates=106200, is_continuation=False, reuse_completed_model=False,
                source_model_path=None, source_model_SHA256=None, cap_seconds=21600)
    job = dict(status='failed', exit_code=1, reservation_released=True, allowed_seconds=21600,
               started_unix=100., ended_unix=21585., error='Worker exited nonzero')
    failure = dict(status='incomplete', error=gate.TIMEOUT, from_scratch=True,
                   initial_completed_updates=0, automatic_retry=False)
    result = dict(status='INCOMPLETE_BENCHMARK097_CELL_CLEAN_SETTLEMENT_AUDIT',
                  attempt=gate.ATTEMPT, cell=deepcopy(gate.CELL), complete=False,
                  all_new_training_from_scratch=True, initial_completed_updates=0,
                  reuse_completed_model=False, automatic_retry=False,
                  own_processes_absent=True, owned_registry_lease_absent=True,
                  charged_GPU_seconds=21485., failure=failure)
    fit = dict(status='completed', fold=1, training_seed=2, AE_epochs=300, FM_epochs=1500,
               AE_updates=17700, FM_updates=88500, total_optimizer_updates=106200,
               validation_cells=0, early_stopping=False,
               frozen_encoder_parameters_buffers_and_modes_preserved=True, FM_checkpoint_SHA256='a'*64)
    restore = dict(status='EXACT_NATIVE_CFGEN_CUDA_RESTORE_PASS', native_global_step=88500,
                   native_epoch=1500, checkpoint_SHA256='a'*64)
    return dict(auth=auth, auth_sha256=gate.AUTH_SHA256, job=job,
                settlement=dict(uncertain=None, settlement=dict(status='settled', charged_GPU_seconds=21485.)),
                result=result, result_sha256='b'*64, expected_result_sha256='b'*64,
                fit=fit, restore=restore, processes_absent=True, lease_absent=True)


class TerminalBoundary(unittest.TestCase):
    def rejects(self, mutate):
        case = fixture()
        mutate(case)
        with self.assertRaises(AssertionError):
            gate.classify(**case)

    def test_only_exact_complete_terminal_fixture_passes(self):
        record = gate.classify(**fixture())
        self.assertTrue(record['future_replacement_must_train_from_zero'])
        self.assertFalse(record['failed_weights_may_initialize_training'])
        self.assertFalse(record['recovery_qualified_or_launched'])

    def test_canonical_nonzero_exit_may_have_null_guard_error(self):
        case = fixture()
        case['job']['error'] = None
        self.assertEqual(gate.classify(**case)['retained_native_training_updates'], 106200)

    def test_live_or_unsettled_job_is_never_a_failure(self):
        for mutate in (lambda c:c['job'].update(status='active'),
                       lambda c:c['job'].update(reservation_released=False),
                       lambda c:c['settlement'].update(uncertain='pending'),
                       lambda c:c.update(processes_absent=False),
                       lambda c:c.update(lease_absent=False)):
            self.rejects(mutate)

    def test_other_failure_causes_and_early_exit_refused(self):
        for mutate in (lambda c:c['result']['failure'].update(error='RuntimeError(out of memory)'),
                       lambda c:c['job'].update(error='Protected/registered GPU growth no longer fits'),
                       lambda c:c['job'].update(ended_unix=1000.)):
            self.rejects(mutate)

    def test_partial_native_fit_or_wrong_restore_refused(self):
        for mutate in (lambda c:c['fit'].update(FM_epochs=1499),
                       lambda c:c['fit'].update(FM_updates=88000),
                       lambda c:c['fit'].update(status='incomplete'),
                       lambda c:c['restore'].update(checkpoint_SHA256='c'*64),
                       lambda c:c['restore'].update(native_global_step=88000)):
            self.rejects(mutate)

    def test_wrong_case_or_authorization_and_missing_pinned_result_refused(self):
        for mutate in (lambda c:c['auth']['cell'].update(seed=3),
                       lambda c:c.update(auth_sha256='0'*64),
                       lambda c:c.update(expected_result_sha256=None),
                       lambda c:c.update(result_sha256='0'*64)):
            self.rejects(mutate)

    def test_success_charge_tamper_and_warmstart_refused(self):
        for mutate in (lambda c:c['result'].update(complete=True),
                       lambda c:c['job'].update(exit_code=0),
                       lambda c:c['result'].update(charged_GPU_seconds=1.),
                       lambda c:c['auth'].update(is_continuation=True),
                       lambda c:c['auth'].update(source_model_path='prior.ckpt')):
            self.rejects(mutate)


if __name__ == '__main__':
    unittest.main()
