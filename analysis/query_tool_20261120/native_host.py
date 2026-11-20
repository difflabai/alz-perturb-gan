"""Read-only loader for the derived native EMPF host modules.

``analysis/empf_comparison_20261117/host.activate`` rewrites the derived
release_env/trainer/generate files on every call. A concurrent confirmation
queue calls it too (the recorded "EMPF activation race"). This loader never
writes there: it snapshots the three derived files into a private directory,
checks every byte against the hashes recorded in ``derivation.json`` (and
retries while a concurrent identical rewrite is in progress), then imports
the private copies. Released ParticleGAN and all pinned sources are unchanged.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path('/ml2/alz')
DERIVED = ROOT / 'artifacts/native-releases/particlegan-0.10.0/alz-host-empf-native'
R5 = ROOT / 'analysis/improve_20261109/r5'
NAMES = ('release_env.py', 'trainer.py', 'generate.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot(dest, attempts=20):
    """Copy the derived files to ``dest`` and verify them against derivation.json."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for _ in range(attempts):
        try:
            record = json.loads((DERIVED / 'derivation.json').read_text())
            ok = record.get('lane') == 'native' and record.get('release') == '0.10.0'
            for name in NAMES:
                shutil.copyfile(DERIVED / name, dest / name)
                ok = ok and sha(dest / name) == record['files'][name]['derived_sha256']
                ok = ok and sha(record['files'][name]['original']) == record['files'][name]['original_sha256']
            if ok:
                return {name: sha(dest / name) for name in NAMES} | {'derivation': record}
        except (OSError, ValueError, KeyError):
            pass
        time.sleep(0.5)
    raise RuntimeError('could not obtain a hash-consistent snapshot of the derived EMPF host')


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def activate(private_dir):
    """Return (trainer, generate, provenance) imported from a verified private snapshot."""
    if any(k in sys.modules for k in ('particlegan', 'trainer', 'generate', 'release_env')):
        raise RuntimeError('native host requires a fresh process')
    provenance = snapshot(private_dir)
    _load('release_env', Path(private_dir) / 'release_env.py')
    T = _load('trainer', Path(private_dir) / 'trainer.py')
    G = _load('generate', Path(private_dir) / 'generate.py')
    import mode_cellshape  # noqa: F401  (r5 is on sys.path via trainer; registers the readout)
    return T, G, provenance
