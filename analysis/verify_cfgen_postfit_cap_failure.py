"""Recognize one settled post-fit timeout; never start or resume any model.

An active job or a duration forecast cannot establish failure. Any later recovery
still requires its own qualified lineage and an entirely fresh training fit.
"""
import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path('/ml2/alz')
ATTEMPT = 'CAL_GPU0_V1_external097_f1_cfgen_s2'
CELL = dict(experiment='external097', fold=1, method='cfgen', seed=2)
AUTH_SHA256 = '5db243583c234164643ba340f49f3ae408fb8f7c64c4c929bf8f84d4e1d3c0dd'
TIMEOUT = "TimeoutError('Finite benchmark allocation ending; preserve incomplete outcome')"


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def classify(*, auth, auth_sha256, job, settlement, result, result_sha256,
             expected_result_sha256, fit, restore, processes_absent, lease_absent):
    assert auth_sha256 == AUTH_SHA256 and auth['attempt_id'] == ATTEMPT and auth['cell'] == CELL
    assert auth['initial_completed_updates'] == 0 and auth['new_model_updates'] == 106200
    assert not auth['is_continuation'] and not auth['reuse_completed_model']
    assert auth['source_model_path'] is auth['source_model_SHA256'] is None
    assert job['status'] == 'failed' and type(job['exit_code']) is int and job['exit_code'] != 0
    assert job['reservation_released'] and processes_absent is True and lease_absent is True
    assert settlement['uncertain'] is None and settlement['settlement']['status'] == 'settled'
    assert auth['cap_seconds'] == job['allowed_seconds'] == 21600
    assert math.isfinite(job['ended_unix']) and job['ended_unix']-job['started_unix'] >= 21600-150
    assert 'Protected/registered GPU growth' not in job.get('error', '')
    assert result['status'] == 'INCOMPLETE_BENCHMARK097_CELL_CLEAN_SETTLEMENT_AUDIT'
    assert result['attempt'] == ATTEMPT and result['cell'] == CELL and result['complete'] is False
    assert result['all_new_training_from_scratch'] and result['initial_completed_updates'] == 0
    assert not result['reuse_completed_model'] and not result['automatic_retry']
    assert result['own_processes_absent'] and result['owned_registry_lease_absent']
    assert isinstance(expected_result_sha256, str) and len(expected_result_sha256) == 64
    assert all(c in '0123456789abcdef' for c in expected_result_sha256)
    assert result_sha256 == expected_result_sha256
    charge = settlement['settlement']['charged_GPU_seconds']
    assert math.isfinite(charge) and charge > 0 and result['charged_GPU_seconds'] == charge
    failure = result['failure']
    assert failure['status'] == 'incomplete' and failure['error'] == TIMEOUT
    assert failure['from_scratch'] and failure['initial_completed_updates'] == 0 and not failure['automatic_retry']
    assert fit['status'] == 'completed' and fit['fold'] == 1 and fit['training_seed'] == 2
    assert (fit['AE_epochs'], fit['FM_epochs'], fit['AE_updates'], fit['FM_updates'], fit['total_optimizer_updates']) == (300, 1500, 17700, 88500, 106200)
    assert fit['validation_cells'] == 0 and not fit['early_stopping']
    assert fit['frozen_encoder_parameters_buffers_and_modes_preserved']
    assert restore['status'] == 'EXACT_NATIVE_CFGEN_CUDA_RESTORE_PASS'
    assert restore['native_global_step'] == 88500 and restore['native_epoch'] == 1500
    assert restore['checkpoint_SHA256'] == fit['FM_checkpoint_SHA256']
    return dict(status='ONE_CLEANLY_SETTLED_FULLFIT_POSTFIT_NATIVE_CAP_FAILURE_VERIFIED',
                attempt=ATTEMPT, result_SHA256=result_sha256,
                retained_native_training_updates=106200,
                future_replacement_must_train_from_zero=True, failed_weights_may_initialize_training=False,
                recovery_qualified_or_launched=False, full_goal_complete=False, SOTA_claim=False)


def identity(pid):
    try:
        path = Path('/proc') / str(pid)
        stat = (path / 'stat').read_text().rsplit(')', 1)[1].split()
        if stat[0] == 'Z' or path.stat().st_uid != os.getuid():
            return None
        return dict(pid=pid, birth=int(stat[19]), boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    except (FileNotFoundError, ProcessLookupError):
        return None


def check_current():
    folder = ROOT / 'artifacts/allocations' / ATTEMPT
    auth_path = folder / 'authorization.json'
    assert sha(auth_path) == AUTH_SHA256
    auth = read(auth_path)
    job = read(auth['ledger_path'])['jobs'][0]
    if job['status'] == 'active':
        for name in ('guard_identity', 'child_identity'):
            expected = job[name]
            assert identity(expected['pid']) == expected
        return dict(status='EXACT_NATIVE_JOB_STILL_LIVE_NO_FAILURE_DECLARED', attempt=ATTEMPT,
                    recovery_qualified_or_launched=False, quality_values_read=False)
    # Classification is deliberately not automatic: a later qualified amendment
    # must first freeze the actual result hash and inspect all source/file pins.
    return dict(status='NATIVE_JOB_NO_LONGER_ACTIVE_REQUIRES_CLEAN_TERMINAL_AUDIT_AND_PINNED_RESULT',
                attempt=ATTEMPT, observed_job_status=job['status'],
                recovery_qualified_or_launched=False, quality_values_read=False)


if __name__ == '__main__':
    print(json.dumps(check_current(), indent=2))
