"""Concurrent runtime starts on a shared host: locked host activation and distinct CPU sets.

Regression tests for two failures seen when the 17-fit confirmation queue
started eight processes at once (analysis/coverage_empf_confirm_20261120,
commit a662ad3): host.activate rewrote its derived files non-atomically, so a
process could read a half-written file and stop with "existing derived EMPF
host differs"; and every process pinned itself to the same first three CPUs.
The host here is a stub that reproduces host.py's read-compare-truncate-write
pattern with a widened truncation window; no torch or data is needed.
"""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('microglia_runtime_concurrency', ROOT/'scripts/microglia_runtime.py')
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)

STUB_HOST = '''
import os, time
from pathlib import Path
from types import SimpleNamespace

DERIVED = Path(os.environ['STUB_DERIVED'])


def expected(name):
    return '# Derived by the stub host\\n' + name * 4000


def activate(lane='native'):
    # Same pattern as analysis/empf_comparison_20261117/host.py: compare, then truncate and rewrite.
    for name in ('release_env.py', 'trainer.py', 'generate.py'):
        path = DERIVED / name
        if path.exists() and path.read_text() != expected(name):
            raise RuntimeError('existing derived EMPF host differs: ' + str(path))
        with open(path, 'w') as handle:   # truncated here
            time.sleep(float(os.environ.get('STUB_WRITE_DELAY', '0.3')))
            handle.write(expected(name))
    torch = SimpleNamespace(
        set_num_threads=lambda n: None, use_deterministic_algorithms=lambda flag: None,
        backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace()), cudnn=SimpleNamespace()),
        cuda=SimpleNamespace(is_initialized=lambda: False))
    return SimpleNamespace(torch=torch), SimpleNamespace(), {'stub': True}
'''

CHILD = textwrap.dedent('''
    import importlib.util, json, os, sys, time
    from pathlib import Path
    index, mode = int(sys.argv[1]), sys.argv[2]
    work = Path(os.environ['STUB_WORK'])
    spec = importlib.util.spec_from_file_location('microglia_runtime', sys.argv[3])
    R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
    R.STUDY = work / 'study'
    R.HOST_LOCK = work / 'host.lock'
    while not (work / 'go').exists():
        time.sleep(.005)
    time.sleep(.1 * index)  # staggered starts inside the truncation window of the first process
    result = {'index': index}
    try:
        if mode == 'unlocked':
            sys.path.insert(0, str(R.STUDY))
            import host
            host.activate('native')
        else:
            T, G, provenance, cpus = R.activate('cpu', .04)
            result['cpus'] = cpus['cpus']
            result['source'] = cpus['source']
            result['affinity'] = sorted(os.sched_getaffinity(0))
        result['ok'] = True
    except Exception as exc:
        result.update(ok=False, error=f'{type(exc).__name__}: {exc}')
    (work / f'ready{index}').write_text('')
    while not (work / 'release').exists():  # hold CPU slots until every process has chosen
        time.sleep(.005)
    print(json.dumps(result))
''')


def run_concurrently(tmp_path, n, mode, extra_env=None):
    study = tmp_path/'study'
    study.mkdir()
    (study/'host.py').write_text(STUB_HOST)
    (study/'mode_cellshape.py').write_text('')
    derived = tmp_path/'derived'
    derived.mkdir()
    for name in ('release_env.py', 'trainer.py', 'generate.py'):  # an already-derived host, as in production
        (derived/name).write_text('# Derived by the stub host\n' + name * 4000)
    child = tmp_path/'child.py'
    child.write_text(CHILD)
    env = dict(os.environ, STUB_WORK=str(tmp_path), STUB_DERIVED=str(derived),
               MICROGLIA_CPU_LOCK_DIR=str(tmp_path/'cpu-locks'), PYTHONDONTWRITEBYTECODE='1')
    env.pop('MICROGLIA_CPU_SLOT', None)
    env.update(extra_env or {})
    procs = [subprocess.Popen([sys.executable, '-B', str(child), str(i), mode, str(ROOT/'scripts/microglia_runtime.py')],
                              env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
             for i in range(n)]
    time.sleep(1.0)  # let every interpreter reach the start gate
    (tmp_path/'go').write_text('')
    deadline = time.monotonic() + 60
    while not all((tmp_path/f'ready{i}').exists() for i in range(n)):
        if time.monotonic() > deadline or any(p.poll() not in (None, 0) for p in procs):
            break
        time.sleep(.02)
    (tmp_path/'release').write_text('')
    results = []
    for p in procs:
        out, err = p.communicate(timeout=60)
        assert p.returncode == 0, err
        results.append(json.loads(out.strip().splitlines()[-1]))
    return results


def test_stub_host_reproduces_the_unlocked_activation_race(tmp_path):
    """Control: without the lock the stub fails the way fold 8 seed 0 did."""
    results = run_concurrently(tmp_path, 3, 'unlocked')
    errors = [r['error'] for r in results if not r['ok']]
    assert errors and all('existing derived EMPF host differs' in e for e in errors)


def test_concurrent_activation_is_serialized_and_pins_distinct_cpus(tmp_path):
    n = 3
    if len(os.sched_getaffinity(0)) < 3 * n:
        pytest.skip('needs at least nine allowed CPUs to give three disjoint three-CPU sets')
    results = run_concurrently(tmp_path, n, 'locked')
    assert all(r['ok'] for r in results), results
    for r in results:
        assert len(r['cpus']) == 3 and r['affinity'] == r['cpus'] and r['source'] == 'slot_lock'
    sets = [frozenset(r['cpus']) for r in results]
    assert len(set().union(*sets)) == 3 * n, f'CPU sets overlap: {sets}'
    for name in ('release_env.py', 'trainer.py', 'generate.py'):
        assert (tmp_path/'derived'/name).read_text() == '# Derived by the stub host\n' + name * 4000


def test_inherited_small_mask_is_respected(tmp_path):
    cpus, info = R.choose_cpus({7, 9}, env={}, lock_dir=tmp_path, load=lambda c: {})
    assert cpus == [7, 9] and info['source'] == 'inherited_mask'
    cpus, info = R.choose_cpus({4, 5, 6}, env={}, lock_dir=tmp_path, load=lambda c: {})
    assert cpus == [4, 5, 6] and info['source'] == 'inherited_mask'


def test_explicit_cpu_slot_selects_a_block():
    allowed = set(range(24))
    assert R.choose_cpus(allowed, env={'MICROGLIA_CPU_SLOT': '2'})[0] == [6, 7, 8]
    assert R.choose_cpus(allowed, env={'MICROGLIA_CPU_SLOT': '9'})[0] == [3, 4, 5]  # wraps over 8 blocks
    with pytest.raises(ValueError):
        R.choose_cpus(allowed, env={'MICROGLIA_CPU_SLOT': '-1'})


def test_slot_claims_prefer_idle_cpus_and_do_not_repeat(tmp_path, monkeypatch):
    monkeypatch.setattr(R, '_CPU_SLOT', None)
    busy = {c: (1. if c < 3 else 0.) for c in range(12)}  # another job saturates 0-2
    first, info = R.choose_cpus(set(range(12)), env={}, lock_dir=tmp_path, load=lambda c: busy)
    assert first == [3, 4, 5] and info['source'] == 'slot_lock'
    held = R._CPU_SLOT
    # Simulate a second process: keep the first claim open under another handle.
    monkeypatch.setattr(R, '_CPU_SLOT', None)
    second, _ = R.choose_cpus(set(range(12)), env={}, lock_dir=tmp_path, load=lambda c: busy)
    assert second == [6, 7, 8]
    R._CPU_SLOT.close(); held.close()


def test_all_slots_claimed_falls_back_to_least_loaded(tmp_path, monkeypatch):
    import fcntl
    handles = []
    for block in ([0, 1, 2], [3, 4, 5]):
        h = open(tmp_path/('cpus-' + '-'.join(map(str, block)) + '.lock'), 'a+')
        fcntl.flock(h, fcntl.LOCK_EX | fcntl.LOCK_NB)
        handles.append(h)
    monkeypatch.setattr(R, '_CPU_SLOT', None)
    busy = {0: .9, 1: .1, 2: .8, 3: .0, 4: .7, 5: .2}
    cpus, info = R.choose_cpus(set(range(6)), env={}, lock_dir=tmp_path, load=lambda c: busy)
    assert cpus == [1, 3, 5] and info['source'] == 'least_loaded'
    for h in handles:
        h.close()
