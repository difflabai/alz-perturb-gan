"""Observe shared capacity only; never deploy, activate, allocate or train."""
import datetime
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path('/ml2/alz')
sys.path.insert(0, str(ROOT / 'released_rate_g100d100p050_runtime'))
import binding as b
import prepare
import resource_profile as p
import preset_scope as s

FOLDER = ROOT / 'artifacts/diagnostics/released_G_D_full_GPU1_capacity'


def main():
    assert b.read(s.PLAN)['enabled'] is False
    proof = b.read(s.RUNTIME_PROOF)
    assert proof['tests_passed'] == 24
    for path, pin in proof['source_SHA256'].items():
        assert b.sha(path) == pin, path
    FOLDER.mkdir(parents=True, exist_ok=True)
    stable = None
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        auth, _ = prepare.build_authorization(check_only=True)
        controller, admission, _ = b.modules(auth)
        pressure = Path('/proc/pressure/memory').read_text()
        full = float(re.search(r'^full avg10=([0-9.]+)', pressure, re.M)[1])
        record = {'status': 'WAITING_UNCHANGED_GPU1_CAPACITY_AND_GLOBAL_MEMORY_PSI',
                  'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'observer_PID': __import__('os').getpid(), 'full_memory_PSI_avg10': full,
                  'candidate_id': s.CANDIDATE_ID, 'CPU_qualification_passed': True,
                  'GPU_job_or_allocation_started': False, 'foreign_controls_or_guard_changes': False}
        try:
            observation = admission.snapshot()
            record['forecast'] = p.capacity(observation, admission)
            p.free_floor()
            p.previous_trial_released()
            assert not controller.Controller(b.PROJECT)._read()['open_reservations'], 'Another canonical phase is active'
            assert not b.read(ROOT / 'artifacts/gpu0-released-g050d100p050-20261104/live_registry/registry.json')['jobs']
            assert full < 10, 'Existing full memory PSI threshold is not clear'
            stable = time.monotonic() if stable is None else stable
            record['stable_clear_seconds'] = time.monotonic() - stable
        except Exception as error:
            if type(error).__name__ == 'Uncertain' and str(error) == 'GPU PID identity not observable':
                pass
            elif isinstance(error, (AssertionError, b.Refusal)) and any(text in str(error) for text in
                    ('growth forecasts do not fit', 'host available below', 'GPU free below', 'memory PSI', 'canonical phase is active')):
                pass
            else:
                raise
            stable = None
            record['gate_refusal'] = str(error)
        (FOLDER / 'progress.json').write_text(json.dumps(record, indent=2) + '\n')
        if stable is not None and record['stable_clear_seconds'] >= 30:
            record.update(status='UNCHANGED_SHARED_GPU1_CAPACITY_AND_PSI_READY_NO_ADMISSION',
                          utc_unix=time.time(), boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
            (FOLDER / 'first_observation.json').write_text(json.dumps(record, indent=2) + '\n')
            print(json.dumps({k: v for k, v in record.items() if k != 'forecast'}), flush=True)
            return
        time.sleep(5)
    raise TimeoutError('Capacity still unsettled after15minutes; no GPU admission performed')


if __name__ == '__main__':
    main()
