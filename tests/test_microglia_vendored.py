"""Vendored EMPF model sources and the checkout-relative host (no torch, data or GPU needed).

The runtime used to import pg_model, critic, usage, contracts, stage_law and
bench_core from absolute /ml2/codex-storage directories. They are now vendored
byte for byte, and analysis/default_empf_20261117/vendored_host.py derives the
host so that every location it names is inside the checkout.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SELECTION = ROOT/'analysis/default_empf_20261117'
VENDORED = SELECTION/'vendored'
MANIFEST = SELECTION/'runtime_source_manifest.json'
# Digests of the out-of-repo originals, as pinned before vendoring (the first three by the runtime and
# the concluded EMPF comparison; the last three were imported but never pinned).
ORIGINALS = {
    'pg_model.py': ('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark/pg_model.py',
                    '47740d4039e680c48c9e5855753d7ebd797fe4f68109aaa75c3d56b57f13af85'),
    'critic.py': ('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/core-comparison-20261103/critic.py',
                  '2921c0f7706cf3ac7bff0e3ebb09be2f2699a97e93d1faf1fb5376d6b9c11a5f'),
    'usage.py': ('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/core-comparison-20261103/usage.py',
                 'de5255a5f594967adf6fa4bd27d6c37b4f2dd83d9e1d4f0c75fd28d5897f7ae7'),
    'contracts.py': ('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/core-comparison-20261103/contracts.py',
                     '1ef929b020b87c5a1e7e4a1194a42ded2c0f61c7589f3e8bb49a6f437ae57131'),
    'stage_law.py': ('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/core-comparison-20261103/stage_law.py',
                     '403b04bbaa285f5cc2ffbace9b67c133448c83c7304083da2aba9cc4daa6c5ae'),
    'bench_core.py': ('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark/bench_core.py',
                      'fd56394690123ec404fc5aad89d7b381c16aadc837bc9c534465efb53d053a46'),
}
HOST_INPUTS = ['analysis/empf_comparison_20261117/host.py', 'analysis/empf_comparison_20261117/adapter.py',
               'analysis/default_empf_20261117/vendored_host.py', 'analysis/improve_20261109/r5/trainer.py',
               'analysis/improve_20261109/r5/generate.py', 'analysis/faithful097/model/release_env.py',
               *(f'analysis/default_empf_20261117/vendored/{n}' for n in ORIGINALS)]
# Assignments in the derived files that name a location.
LOCATION = re.compile(r"^(HERE|MODEL|SCIENCE|BENCH|RUNS|DEFAULT_OUT_ROOT) = Path\((.*)\)$|^(SITE|B)=Path\((.*)\)$", re.M)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_vendored_files_are_byte_identical_to_the_pinned_originals():
    manifest = json.loads(MANIFEST.read_text())['sources']
    record = json.loads((VENDORED/'SOURCES.json').read_text())['files']
    assert sorted(p.name for p in VENDORED.glob('*.py')) == sorted(ORIGINALS)
    for name, (original, digest) in ORIGINALS.items():
        assert sha(VENDORED/name) == digest, name
        assert manifest[f'analysis/default_empf_20261117/vendored/{name}'] == digest
        assert record[name]['original'] == original and record[name]['sha256'] == digest
        if Path(original).exists():  # only on the workstation that holds the originals
            assert (VENDORED/name).read_bytes() == Path(original).read_bytes()
    study = json.loads((ROOT/'analysis/empf_comparison_20261117/source_manifest.json').read_text())
    for name in ('pg_model.py', 'critic.py', 'usage.py'):
        assert study[ORIGINALS[name][0]] == ORIGINALS[name][1]


def test_runtime_manifest_pins_no_absolute_path():
    manifest = json.loads(MANIFEST.read_text())
    assert [k for k in manifest['sources'] if Path(k).is_absolute()] == []
    assert 'analysis/default_empf_20261117/vendored_host.py' in manifest['sources']


def copy_tree(dest, paths):
    for rel in paths:
        (dest/rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/rel, dest/rel)


def run_in_copy(root, code):
    """Run code against vendored_host in a fresh interpreter rooted at a copied checkout."""
    prelude = ('import json, sys\n'
               f'sys.path.insert(0, {str(root/"analysis/default_empf_20261117")!r})\n'
               'import vendored_host as V\n')
    out = subprocess.run([sys.executable, '-B', '-c', prelude + code], capture_output=True, text=True,
                         env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'), cwd=root)
    assert out.returncode == 0, out.stderr
    return out.stdout


@pytest.fixture
def copied(tmp_path):
    root = tmp_path/'elsewhere'/'alz'
    copy_tree(root, HOST_INPUTS)
    return root


def test_derived_host_names_only_paths_inside_a_relocated_checkout(copied):
    bodies = json.loads(run_in_copy(copied, 'print(json.dumps(V.derive_bodies("native")[0]))'))
    assert sorted(bodies) == ['contracts.py', 'generate.py', 'release_env.py', 'trainer.py']
    found = {}
    for name, body in bodies.items():
        assert 'codex-storage' not in body, name
        assert body.startswith('# Derived by analysis/default_empf_20261117/vendored_host.py')
        for m in LOCATION.finditer(body):
            key, literal = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
            found[(name, key)] = literal
    vendored = repr(str(copied/'analysis/default_empf_20261117/vendored'))
    assert found[('trainer.py', 'SCIENCE')] == found[('trainer.py', 'BENCH')] == vendored
    assert found[('contracts.py', 'B')] == vendored
    assert found[('trainer.py', 'MODEL')] == repr(str(copied/'analysis/faithful097/model'))
    assert found[('trainer.py', 'RUNS')] == repr(str(copied/'runs/improve_20261109'))
    assert found[('release_env.py', 'SITE')] == repr(str(copied/'artifacts/native-releases/particlegan-0.10.0/site-packages'))
    for (name, key), literal in found.items():
        assert literal.startswith(repr(str(copied))[:-1]), (name, key, literal)


def test_vendored_host_differs_from_the_study_host_only_in_locations(copied):
    """Derive with the unchanged study host.py too; the two hosts differ only in the location lines."""
    code = ('sys.path.insert(0, str(V.STUDY)); import host\n'
            'try:\n    host.activate("native")\n'
            'except Exception as exc:\n    err = type(exc).__name__\n'  # stops at the particlegan import
            'else:\n    err = None\n'
            'print(json.dumps([err, V.derive_bodies("native")[0]]))')
    err, bodies = json.loads(run_in_copy(copied, code))
    assert err is not None  # no native release in the copy, so the study host stops after writing its files
    study = copied/'artifacts/native-releases/particlegan-0.10.0/alz-host-empf-native'
    vendored = repr(str(copied/'analysis/default_empf_20261117/vendored'))
    expected = {'trainer.py': {"MODEL = Path('/ml2/alz/analysis/faithful097/model')":
                                   f"MODEL = Path({str(copied/'analysis/faithful097/model')!r})",
                               "SCIENCE = Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/core-comparison-20261103')":
                                   f'SCIENCE = Path({vendored})',
                               "BENCH = Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark')":
                                   f'BENCH = Path({vendored})',
                               "RUNS = Path('/ml2/alz/runs/improve_20261109')":
                                   f"RUNS = Path({str(copied/'runs/improve_20261109')!r})"},
                'release_env.py': {"SITE=Path('/ml2/alz/artifacts/native-releases/particlegan-0.10.0/site-packages')":
                                       f"SITE=Path({str(copied/'artifacts/native-releases/particlegan-0.10.0/site-packages')!r})"},
                'generate.py': {}}
    for name, rewrites in expected.items():
        before = (study/name).read_text().splitlines()[1:]  # drop each host's own header line
        after = bodies[name].splitlines()[1:]
        assert len(before) == len(after), name
        assert {a: b for a, b in zip(before, after) if a != b} == rewrites, name
    contracts = (copied/'analysis/default_empf_20261117/vendored/contracts.py').read_text().splitlines()
    derived = bodies['contracts.py'].splitlines()[1:]
    assert len(contracts) == len(derived)
    assert {a: b for a, b in zip(contracts, derived) if a != b} == {
        "B=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark')": f'B=Path({vendored})'}


def test_write_host_uses_a_new_directory_and_refuses_a_changed_file(copied):
    old = copied/'artifacts/native-releases/particlegan-0.10.0/alz-host-empf-native'
    old.mkdir(parents=True)
    (old/'trainer.py').write_text('previous derived host\n')
    code = ('d, p = V.write_host("native"); d2, p2 = V.write_host("native")\n'
            'print(json.dumps([str(d), str(d2), p == p2, sorted(p["files"]), sorted(p["vendored"])]))')
    d, d2, same, files, vendored = json.loads(run_in_copy(copied, code))
    assert d == d2 == str(copied/'artifacts/native-releases/particlegan-0.10.0/alz-host-empf-native-vendored')
    assert same and files == ['contracts.py', 'generate.py', 'release_env.py', 'trainer.py']
    assert vendored == sorted(ORIGINALS)
    assert (old/'trainer.py').read_text() == 'previous derived host\n'
    assert not list(Path(d).glob('.*.tmp'))
    with open(Path(d)/'trainer.py', 'a') as handle:
        handle.write('# drift\n')
    prelude = ('import json, sys\n'
               f'sys.path.insert(0, {str(copied/"analysis/default_empf_20261117")!r})\n'
               'import vendored_host as V\nV.write_host("native")\n')
    out = subprocess.run([sys.executable, '-B', '-c', prelude], capture_output=True, text=True, cwd=copied)
    assert out.returncode != 0 and 'existing derived EMPF host differs' in out.stderr


def test_manifest_check_passes_from_a_checkout_at_another_path(tmp_path):
    """Copy every repository-relative pinned file elsewhere; the manifest still verifies there."""
    manifest = json.loads(MANIFEST.read_text())
    root = tmp_path/'moved'/'alz'
    tracked = [k for k in manifest['sources'] if not k.startswith('artifacts/')]
    copy_tree(root, [*tracked, str(MANIFEST.relative_to(ROOT)), manifest['archived_previous_entrypoint']['archive'],
                     'scripts/refreeze_runtime_manifest.py'])
    artifacts = [k for k in manifest['sources'] if k.startswith('artifacts/')]
    have_artifacts = all((ROOT/k).exists() for k in artifacts)
    if have_artifacts:
        (root/'artifacts').symlink_to(ROOT/'artifacts')
    code = ('import importlib.util, json, sys\n'
            f'spec = importlib.util.spec_from_file_location("rt", {str(root/"scripts/microglia_runtime.py")!r})\n'
            'R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)\n'
            f'spec = importlib.util.spec_from_file_location("rf", {str(root/"scripts/refreeze_runtime_manifest.py")!r})\n'
            'F = importlib.util.module_from_spec(spec); spec.loader.exec_module(F)\n'
            'm = json.loads(R.SOURCE_MANIFEST.read_text())\n'
            'ok, missing, changed, borrowed, entry_ok = F.scan(m, R.ROOT)\n'
            f'full = {have_artifacts!r}\n'
            'full and R.check_sources()\n'
            'print(json.dumps([str(R.ROOT), len(ok), missing, sorted(changed), entry_ok, full]))')
    out = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True,
                         env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'), cwd=tmp_path)
    assert out.returncode == 0, out.stderr
    where, ok, missing, changed, entry_ok, full = json.loads(out.stdout)
    assert where == str(root) and changed == [] and entry_ok
    assert sorted(missing) == ([] if full else sorted(artifacts))
    assert ok == len(manifest['sources']) - len(missing)
