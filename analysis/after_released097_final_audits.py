"""Run prepared independent CPU audits after the existing native queues drain.

This observer neither launches training nor signals any process. Completed-path
presence alone never authorizes a numerical read: exact upstream processes must
have exited, terminal receipts must agree, and canonical reservations must drain.
"""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/ml2/alz')
STATE = ROOT / 'artifacts/released097-final-independent-audits'
PROOF = ROOT / 'analysis/released097_final_audits_observer_CPU_check_v5_20261108.json'
PROJECT = Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark')
PG = PROJECT / 'current_base_20261029/runtime/bin/python'
PARENTS = {
    'benchmark': ('benchmark097-infra1-pipeline', 'RECOVERED45_57_53_READY_FRESH_SCALAR_CALIBRATION_AND_FINAL_AUDIT_REMAIN'),
    'scalar': ('scalarcal097-after-recovered-science-v5', 'SCALARCAL097_COMPLETE_CALIBRATION_REPORT_FINAL_RESEARCH_AUDIT_REMAINS'),
    'theta': ('dispersion097-after-scalarcal097', 'DISPERSION097_COMPLETE_CALIBRATION_REPORT_FINAL_RESEARCH_AUDIT_REMAINS'),
}


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(pid):
    try:
        path = Path('/proc') / str(pid)
        stat = (path / 'stat').read_text().rsplit(')', 1)[1].split()
        if stat[0] == 'Z' or path.stat().st_uid != os.getuid():
            return None
        return dict(pid=pid, birth=int(stat[19]), boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    except (FileNotFoundError, ProcessLookupError):
        return None


def write(path, value):
    path = Path(path)
    temp = path.with_suffix('.partial')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def audit_plan(states, *, reservations_open):
    """Inspect only terminal metadata; never consume partial quality reports."""
    assert set(states) == set(PARENTS)
    assert all(state in ('live', 'completed', 'failed', 'deferred', 'missing') for state in states.values())
    if 'live' in states.values() or reservations_open:
        return None
    assert 'missing' not in states.values(), 'Absent upstream has no verified terminal receipt'
    assert states['benchmark'] != 'deferred' and states['scalar'] != 'deferred'
    if states['benchmark'] != 'completed' or states['scalar'] != 'completed':
        return []
    return ['benchmark_scalar'] + (['theta'] if states['theta'] == 'completed' else [])


def parent_state(name, expected):
    if identity(expected['pid']) == expected:
        return 'live'
    folder, completed_status = PARENTS[name]
    base = ROOT / 'artifacts' / folder
    receipts = [kind for kind in ('completed', 'failure', 'deferred') if (base / (kind + '.json')).is_file()]
    assert len(receipts) <= 1, 'Conflicting terminal upstream receipts: ' + name
    if not receipts:
        return 'missing'
    kind = receipts[0]
    receipt = read(base / (kind + '.json'))
    if kind == 'completed':
        assert receipt['status'] == completed_status
        return 'completed'
    if kind == 'deferred':
        assert name == 'theta' and receipt['actual_fresh_fits'] == receipt['new_GPU_allocations'] == 0
        assert receipt['status'] in ('ALL_SIX_THETA_FITS_DEFERRED_UNSTARTED_INSUFFICIENT_REMAINING_WINDOW',
                                    'ALL_SIX_THETA_FITS_DEFERRED_UNSTARTED_CAPACITY_WAIT_EXHAUSTED_WINDOW')
        return 'deferred'
    assert receipt.get('error') or receipt.get('status')
    return 'failed'


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True)
    proof = read(PROOF)
    assert proof['status'] == 'FINAL_AUDIT_OBSERVER_CPU_GATES_PASS_NO_REAL_AUDIT_YET'
    for path, pin in proof['source_SHA256'].items():
        assert sha(path) == pin, path
    parents = proof['upstream_identities']
    for name, expected in parents.items():
        assert identity(expected['pid']) == expected, name
    expiry = read(ROOT / 'analysis/research_window_20261104_five_days.json')['expires_unix'] + 7200
    STATE.mkdir(exist_ok=False)
    with (STATE / 'observer.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(STATE / 'identity.json', dict(**identity(os.getpid()), source_SHA256=sha(__file__),
              qualification_SHA256=sha(PROOF), upstream=parents, CPU_expiry_unix=expiry, new_GPU_allocation=False))
        try:
            while True:
                assert time.time() < expiry, 'Finite CPU observation ended; upstream handles retained without signals'
                states = {name: parent_state(name, expected) for name, expected in parents.items()}
                reserved = bool(read(PROJECT / 'project_gpu_budget.json')['open_reservations'])
                plan = audit_plan(states, reservations_open=reserved)
                write(STATE / 'progress.json', dict(status='WAITING_FOR_UPSTREAM_TERMINALS_AND_RESERVATION_DRAIN',
                      upstream_states=states, reservations_open=reserved, new_GPU_allocation=False,
                      numerical_reports_read=False, full_goal_complete=False))
                if plan is not None:
                    break
                time.sleep(30)
            for path, pin in proof['source_SHA256'].items():
                assert sha(path) == pin, path
            outputs = []
            jobs = {
                'benchmark_scalar': ('audit_released097_completed_studies.py', 'released097_completed_studies_final_independent_audit_20261108.json'),
                'theta': ('audit_dispersion097_completed_study.py', 'dispersion097_completed_study_final_independent_audit_20261108.json'),
            }
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
            for name in plan:
                source, filename = jobs[name]
                output = ROOT / 'analysis' / filename
                assert not output.exists()
                result = subprocess.run([str(PG), '-I', '-S', '-B', str(ROOT / 'analysis' / source), '--output', str(output)],
                        cwd=ROOT, env=env, capture_output=True, text=True, timeout=max(1, min(3600, expiry-time.time())))
                (STATE / (name + '.stdout')).write_text(result.stdout)
                (STATE / (name + '.stderr')).write_text(result.stderr)
                assert result.returncode == 0, 'Independent audit failed; preserve outputs: ' + name
                value = read(output)
                assert not value['full_goal_complete'] and not value['SOTA_claim']
                subprocess.run(['git', 'add', '--', str(output)], cwd=ROOT, check=True)
                subprocess.run(['git', 'commit', '--only', '-m', 'Retain independent completed-study audit without SOTA promotion', '--', str(output)],
                               cwd=ROOT, check=True, capture_output=True, text=True)
                outputs.append(dict(path=str(output), SHA256=sha(output)))
            record = dict(status='AVAILABLE_COMPLETED_STUDY_AUDITS_FINISHED' if plan else 'AUDITS_UNAVAILABLE_UPSTREAM_STUDIES_INCOMPLETE',
                          UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), upstream_states=states,
                          audit_outputs=outputs, all_planned_studies_audited=plan == ['benchmark_scalar', 'theta'],
                          no_training_or_GPU_allocation=True, full_goal_complete=False, SOTA_claim=False)
            write(STATE / ('completed.json' if plan else 'incomplete.json'), record)
            sys.path.insert(0, str(ROOT / 'benchmark097_balanced_runtime'))
            from babel_reporting import report
            report('completed' if plan else 'failed', 'released097-final-independent-audits-terminal',
                   'Independent CPU audit observer finished; inspect retained outputs and incomplete-study flags. Full research goal remains active.', record, log_root=STATE)
        except BaseException as exc:
            write(STATE / 'failure.json', dict(error=repr(exc), upstream_or_foreign_signals=False,
                  automatic_retry=False, new_GPU_allocation=False, full_goal_complete=False, SOTA_claim=False))
            raise


if __name__ == '__main__':
    main()
