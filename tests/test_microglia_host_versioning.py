"""Versioned derived-host activation (vendored_host): ABSENT -> DERIVING -> DERIVED, CORRUPT -> rederive.

Runs in a copied checkout under tmp (no torch, data or native release needed). The loader used here
compiles every derived file under the shared lock instead of importing the model, which needs the
native release; a torn or half-written file would raise SyntaxError or fail its digest.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ['analysis/empf_comparison_20261117/host.py', 'analysis/empf_comparison_20261117/adapter.py',
          'analysis/default_empf_20261117/vendored_host.py', 'analysis/improve_20261109/r5/trainer.py',
          'analysis/improve_20261109/r5/generate.py', 'analysis/faithful097/model/release_env.py',
          'src/alz_perturb_gan/__init__.py', 'src/alz_perturb_gan/durable_state.py',
          *(f'analysis/default_empf_20261117/vendored/{n}' for n in
            ('pg_model.py', 'critic.py', 'usage.py', 'contracts.py', 'stage_law.py', 'bench_core.py'))]
RELEASE = 'artifacts/native-releases/particlegan-0.10.0'

CHILD = textwrap.dedent('''
    import json, os, sys, time
    from pathlib import Path
    root = Path(sys.argv[1]); work = Path(sys.argv[2]); index = int(sys.argv[3])
    sys.path.insert(0, str(root/'analysis/default_empf_20261117'))
    import vendored_host as V
    ds = V._ds()
    real_write = ds.atomic_write_bytes
    def slow_write(path, data):  # widen the derivation window
        time.sleep(float(os.environ.get('SLOW_WRITE', '0')))
        real_write(path, data)
    ds.atomic_write_bytes = slow_write
    derived = []
    real_ensure = V.ensure_host
    def ensure(lane='native'):
        out = real_ensure(lane)
        derived.append(out[2])
        return out
    V.ensure_host = ensure
    def loader(directory):
        for name in V.DERIVED_FILES:
            compile((directory/name).read_text(), str(directory/name), 'exec')
        return directory, None
    while not (work/'go').exists():
        time.sleep(.005)
    time.sleep(.03 * index)
    result = {'index': index, 'pid': os.getpid()}
    try:
        directory, _, provenance = V.activate('native', loader=loader)
        result.update(ok=True, directory=str(directory), derived=any(derived),
                      legacy=bool(provenance.get('legacy_directory')), sha=provenance['derivation_sha256'])
    except Exception as exc:
        result.update(ok=False, error=f'{type(exc).__name__}: {exc}')
    print(json.dumps(result))
''')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree(directory):
    return {p.name: sha(p) for p in sorted(Path(directory).iterdir()) if p.is_file()}


@pytest.fixture
def checkout(tmp_path):
    root = tmp_path/'alz'
    for rel in INPUTS:
        (root/rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/rel, root/rel)
    (tmp_path/'child.py').write_text(CHILD)
    return root


def activate_all(root, n, slow=0.0):
    work = root.parent
    (work/'go').unlink(missing_ok=True)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', SLOW_WRITE=str(slow))
    procs = [subprocess.Popen([sys.executable, '-B', str(work/'child.py'), str(root), str(work), str(i)],
                              env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
             for i in range(n)]
    time.sleep(.5)
    (work/'go').write_text('')
    results = []
    for p in procs:
        out, err = p.communicate(timeout=120)
        assert p.returncode == 0, err
        assert 'existing derived EMPF host differs' not in out + err and 'ImportError' not in out + err and 'SyntaxError' not in out + err
        results.append(json.loads(out.strip().splitlines()[-1]))
    return results


def run_in(root, code):
    prelude = ('import json, sys\n'
               f'sys.path.insert(0, {str(root/"analysis/default_empf_20261117")!r})\n'
               'import vendored_host as V\n')
    out = subprocess.run([sys.executable, '-B', '-c', prelude + code], capture_output=True, text=True,
                         env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
    assert out.returncode == 0, out.stderr
    return out.stdout


def test_host_machine_table():
    sys.path.insert(0, str(ROOT/'analysis/default_empf_20261117'))
    import vendored_host as V
    table = {(s.name, e): t.name for (s, e), t in V.HOST_MACHINE.transitions.items()}
    assert table == {('ABSENT', 'derive'): 'DERIVING', ('DERIVING', 'commit'): 'DERIVED',
                     ('DERIVING', 'crash'): 'CORRUPT', ('DERIVED', 'damaged'): 'CORRUPT',
                     ('CORRUPT', 'rederive'): 'DERIVING'}


def test_six_concurrent_activations_from_absent_derive_once(checkout):
    results = activate_all(checkout, 6, slow=.1)
    assert all(r['ok'] for r in results), results
    assert sum(r['derived'] for r in results) == 1, results
    assert len({r['directory'] for r in results}) == 1
    directory = Path(results[0]['directory'])
    assert directory.name == f'alz-host-empf-native-vendored-{results[0]["sha"][:12]}'
    marker = json.loads((directory/'derivation.json').read_text())
    for name, record in marker['files'].items():
        assert sha(directory/name) == record['derived_sha256']
    assert not [p for p in directory.iterdir() if p.name.endswith('.tmp')]
    # Everyone loads in parallel once DERIVED: nobody derives again.
    again = activate_all(checkout, 6)
    assert all(r['ok'] and not r['derived'] for r in again), again


def test_changed_input_gets_a_new_directory_and_the_old_one_stays(checkout):
    first = activate_all(checkout, 2)
    old = Path(first[0]['directory'])
    before = tree(old)
    with open(checkout/'analysis/improve_20261109/r5/generate.py', 'a') as handle:  # a derivation input changes
        handle.write('# refrozen\n')
    second = activate_all(checkout, 3, slow=.05)
    assert all(r['ok'] for r in second) and sum(r['derived'] for r in second) == 1
    new = Path(second[0]['directory'])
    assert new != old and new.parent == old.parent and second[0]['sha'] != first[0]['sha']
    assert tree(old) == before  # the earlier epoch's host is intact and still loadable
    out = run_in(checkout, f'from pathlib import Path\nr, _ = V.derivation_record("native")\n'
                           f'print(V.inspect(Path({str(new)!r}), r)[0].name, V.inspect(Path({str(old)!r}), r)[0].name)')
    assert out.split() == ['DERIVED', 'CORRUPT']  # the old directory is not this derivation's


def test_truncated_file_is_corrupt_and_rederive_repairs_it(checkout):
    directory = Path(activate_all(checkout, 1)[0]['directory'])
    good = (directory/'trainer.py').read_bytes()
    (directory/'trainer.py').write_bytes(good[:len(good)//2])
    state = run_in(checkout, 'r, _ = V.derivation_record("native"); print(V.inspect(V.runtime_dir("native"), r))')
    assert 'CORRUPT' in state and 'trainer.py does not match' in state
    results = activate_all(checkout, 3)
    assert all(r['ok'] for r in results) and sum(r['derived'] for r in results) == 1
    assert (directory/'trainer.py').read_bytes() == good


def test_missing_marker_is_corrupt_and_rederived(checkout):
    directory = Path(activate_all(checkout, 1)[0]['directory'])
    (directory/'derivation.json').unlink()  # a crash between the files and the commit marker
    (directory/'.trainer.py.deadbeef.999999999.tmp').write_text('orphan of a dead deriver')
    state = run_in(checkout, 'r, _ = V.derivation_record("native"); print(V.inspect(V.runtime_dir("native"), r)[0].name)')
    assert state.strip() == 'CORRUPT'
    results = activate_all(checkout, 2)
    assert all(r['ok'] for r in results) and sum(r['derived'] for r in results) == 1
    assert (directory/'derivation.json').exists() and not (directory/'.trainer.py.deadbeef.999999999.tmp').exists()


def test_rederive_refuses_a_directory_of_another_derivation(checkout):
    directory = Path(activate_all(checkout, 1)[0]['directory'])
    marker = json.loads((directory/'derivation.json').read_text())
    marker['derivation_sha256'] = marker['derivation_sha256'][:12] + '0'*52  # same prefix, other derivation
    (directory/'derivation.json').write_text(json.dumps(marker))
    out = subprocess.run([sys.executable, '-B', '-c',
                          f'import sys; sys.path.insert(0, {str(checkout/"analysis/default_empf_20261117")!r})\n'
                          'import vendored_host as V; V.write_host("native")'], capture_output=True, text=True)
    assert out.returncode != 0 and 'refusing to rederive' in out.stderr


def test_legacy_directory_is_used_read_only_only_when_identical(checkout):
    bodies = json.loads(run_in(checkout, 'r, data = V.derivation_record("native")\n'
                                         'print(json.dumps({n: b.decode() for n, b in data.items()}))'))
    legacy = checkout/RELEASE/'alz-host-empf-native-vendored'
    legacy.mkdir(parents=True)
    for name, body in bodies.items():
        (legacy/name).write_text(body)
    (legacy/'derivation.json').write_text('{"recorded": "by the previous runtime"}\n')
    before = tree(legacy)
    results = activate_all(checkout, 3)
    assert all(r['ok'] and r['legacy'] and not r['derived'] for r in results), results
    assert tree(legacy) == before
    assert not list((checkout/RELEASE).glob('alz-host-empf-native-vendored-*[0-9a-f]'))
    (legacy/'trainer.py').write_text('# an older derivation\n')
    before = tree(legacy)
    results = activate_all(checkout, 2)
    assert all(r['ok'] and not r['legacy'] for r in results) and sum(r['derived'] for r in results) == 1
    assert tree(legacy) == before  # never rewritten


def test_runtime_activation_does_not_serialize_a_self_locking_host(tmp_path, monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location('microglia_runtime_hostver', ROOT/'scripts/microglia_runtime.py')
    R = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(R)
    host = tmp_path/'selflock'/'selflocking_host.py'
    host.parent.mkdir()
    host.write_text('LOCKS_ITSELF = True\ndef activate(lane):\n    return "called", lane\n')
    lock = tmp_path/'outer.lock'
    assert R.activate_host(host, lock) == ('called', 'native')
    assert not lock.exists()  # no outer lock for a host that locks itself
