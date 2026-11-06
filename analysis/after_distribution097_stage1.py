"""At the exact closed stage1 boundary, audit and qualify the next fresh search.

One use, finite CPU-only coordinator. Never activates a plan or allocates a GPU.
"""
import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/ml2/alz')
STATE = ROOT / 'artifacts/distribution097-next-boundary'
PG = Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark/current_base_20261029/runtime/bin/python')
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')

def read(p):
    return json.loads(Path(p).read_bytes())
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p, q):
    Path(p).write_text(json.dumps(q, indent=2, allow_nan=False) + '\n')
def alive(i):
    p = Path('/proc', str(i['pid']), 'stat')
    if not p.exists():
        return False
    s = p.read_text().rsplit(')', 1)[1].split()
    return s[0] != 'Z' and int(s[19]) == i['birth'] and Path('/proc/sys/kernel/random/boot_id').read_text().strip() == i['boot']
def command(name, args, timeout=120):
    p = subprocess.run(list(map(str, args)), cwd=ROOT, env=ENV, capture_output=True, text=True, timeout=timeout)
    (STATE / (name + '.stdout')).write_text(p.stdout)
    (STATE / (name + '.stderr')).write_text(p.stderr)
    assert p.returncode == 0, f'{name} failed; retained stdout/stderr, no retry or activation'
    return p.stdout
def commit(paths, message):
    command('git-add', ['git', 'add', '--', *map(str, paths)])
    command('git-commit', ['git', 'commit', '--only', '-m', message, '--', *map(str, paths)])
def report(kind, event, summary, details):
    spec = importlib.util.spec_from_file_location('existing_reporter', ROOT / 'analysis/external_benchmark_driver.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.report(kind, event, summary, json.dumps(details))

def main():
    STATE.mkdir(exist_ok=True)
    with (STATE / 'lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not (STATE / 'identity.json').exists(), 'One-use boundary coordinator already consumed'
        parent = read(ROOT / 'artifacts/distribution097-stage1/driver_identity.json')
        assert alive(parent), 'Expected live stage1 driver absent; inspect instead of replacing it'
        sources = [Path(__file__), ROOT / 'analysis/audit_distribution097.py',
                   ROOT / 'analysis/prepare_distribution097b.py', ROOT / 'analysis/distribution097_next_search_rule_20261106.json',
                   *list((ROOT / 'distribution097_runtime').glob('*.py')),
                   *list((ROOT / 'analysis/distribution097/model').glob('*.py')),
                   ROOT / 'analysis/distribution097/protocol.json']
        pins = {str(p): sha(p) for p in sources}
        write(STATE / 'identity.json', dict(pid=os.getpid(), birth=int(Path('/proc', str(os.getpid()), 'stat').read_text().rsplit(')', 1)[1].split()[19]),
              boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(), parent=parent, source_SHA256=pins,
              finite_expiry_unix=time.time() + 3600, activation_permitted=False, GPU_allocation_permitted=False))
        try:
            expiry = read(STATE / 'identity.json')['finite_expiry_unix']
            while alive(parent):
                assert time.time() < expiry, 'Finite boundary wait expired; inspect the same parent handle, no restart'
                time.sleep(10)
            for path, pin in pins.items():
                assert sha(path) == pin, 'Prepared boundary source changed; no qualification/activation'
            assert not (ROOT / 'artifacts/distribution097-stage1/failure.json').exists()
            status = read(ROOT / 'artifacts/distribution097-stage1/progress.json')
            assert status['activation_reverted'] and status['goal_still_active']
            assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True)
            audit = ROOT / 'analysis/distribution097_stage1_final_independent_audit_20261106.json'
            command('complete-stage1-audit', [PG, '-I', '-S', '-B', ROOT / 'analysis/audit_distribution097.py', '--complete', '--output', audit], timeout=600)
            commit([audit], 'Independently verify all fresh native097 stage1 endpoints and actual closed-resource rollback')
            command('materialize-stage2', [PG, '-I', '-S', '-B', ROOT / 'analysis/prepare_distribution097b.py', '--stage1-audit', audit])
            runtime = ROOT / 'distribution097b_runtime'
            study = ROOT / 'analysis/distribution097b'
            command('qualify-stage2-CPU', [PG, '-I', '-S', '-B', runtime / 'qualify.py'], timeout=600)
            q = read(study / 'runtime_CPU_qualification.json')
            assert q['actual_runtime_tests'] == 22 and q['fullwidth_native_optimizer_pairs'] == 12
            assert not q['GPU_allocation_created'] and not q['CUDA_initialized']
            assert not read(runtime / 'plan.json')['enabled']
            gp = ROOT / 'analysis/current_research_goal.json'
            g = read(gp)
            g.update(study_status='STAGE1_FULLY_AUDITED_STAGE2_CPU_QUALIFIED_DISABLED',
                     completed_distribution_stage1_audit_path=str(audit),
                     next_distribution_protocol_path=str(study / 'protocol.json'),
                     next_distribution_CPU_qualification_path=str(study / 'runtime_CPU_qualification.json'),
                     native_goal_status='active')
            write(gp, g)
            commit([study, runtime, ROOT / 'analysis/run_distribution097b.py', gp],
                   'Freeze full20k rate and penalty refinement from audited stage1; qualify disabled fresh097 runtime')
            result = dict(status='STAGE1_CLOSED_AUDITED_NEXT_FRESH20K_SEARCH_CPU_QUALIFIED_DISABLED',
                          UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                          stage1_audit_path=str(audit), next_protocol_path=str(study / 'protocol.json'),
                          CPU_qualification_path=str(study / 'runtime_CPU_qualification.json'),
                          next_anchor=read(study / 'protocol.json')['anchor_candidate_id'],
                          next_steps='Review exact prepared source/resource checks and enable only the qualified new plan; then three new fresh20k fits. Fresh-seed confirmation and full external matrix remain.',
                          activation=False, GPU_allocation_created=False, full_goal_complete=False, SOTA_claim=False)
            write(STATE / 'completed.json', result)
            report('completed', 'distribution097-stage1-closed-next-adaptation-cpu-qualified-disabled-20261106',
                   'All three fresh20k stage1 endpoints independently verified and rolled back; next frozen rate/penalty search CPU-qualified but disabled, no new GPU allocation.', result)
        except BaseException as exc:
            failure = dict(status='BOUNDARY_PREPARATION_STOPPED_RETAINING_EVIDENCE', UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                           error=repr(exc), automatic_retry=False, activation=False, GPU_allocation_created=False)
            write(STATE / 'failure.json', failure)
            report('failed', 'distribution097-boundary-preparation-operational-stop-20261106',
                   'Post-stage1 CPU audit/preparation stopped; evidence retained, no activation or GPU allocation.', failure)
            raise

if __name__ == '__main__':
    main()
