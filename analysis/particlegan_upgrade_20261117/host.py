"""Explicit, isolated v0.10 host derivation; frozen ALZ sources stay untouched.

The compatibility lane changes only the import pin and output directory.
Training/generation bodies are materialized with their source hashes before use.
Never import this module into a process that already imported ParticleGAN.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT / 'analysis/improve_20261109/r5'
RELEASE = ROOT / 'artifacts/native-releases/particlegan-0.10.0'
RUNS = ROOT / 'runs/particlegan_upgrade_20261117'
COMMIT = '952c3b62ba434f7da194067923351ff5748ef8dd'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def replace_once(body, before, after):
    if body.count(before) != 1:
        raise ValueError(f'derivation expected one occurrence: {before[:80]}')
    return body.replace(before, after)


def activate(lane='compat'):
    if lane not in ('compat', 'rhbi'):
        raise ValueError('unknown release-test lane')
    if any(k in sys.modules for k in ('particlegan', 'trainer', 'generate', 'release_env')):
        raise RuntimeError('use a fresh subprocess for the isolated release host')
    runtime = RELEASE / ('alz-host-' + lane)
    runtime.mkdir(parents=True, exist_ok=True)
    originals = {
        'release_env.py': ROOT / 'analysis/faithful097/model/release_env.py',
        'trainer.py': OLD / 'trainer.py',
        'generate.py': OLD / 'generate.py',
    }
    bodies = {k: p.read_text() for k, p in originals.items()}
    bodies['release_env.py'] = bodies['release_env.py'].replace('particlegan-0.9.7', 'particlegan-0.10.0').replace("VERSION='0.9.7'", "VERSION='0.10.0'")
    bodies['trainer.py'] = replace_once(bodies['trainer.py'],
        'HERE = Path(__file__).resolve().parent', f'HERE = Path({str(OLD)!r})')
    bodies['trainer.py'] = replace_once(bodies['trainer.py'],
        "DEFAULT_OUT_ROOT = RUNS / 'fits_r5'", f'DEFAULT_OUT_ROOT = Path({str(RUNS / "fits" / lane)!r})')
    if lane == 'rhbi':
        bodies['release_env.py'] = replace_once(bodies['release_env.py'],
            "PENALTY_OPTIONS={'input_domain':'continuous'}", 'PENALTY_OPTIONS={}')
        from rhbi_adapter import derive
        bodies['trainer.py'] = derive(bodies['trainer.py'])
    records = {}
    for name, body in bodies.items():
        path = runtime / name
        prefix = '# Derived by analysis/particlegan_upgrade_20261117/host.py; see derivation.json.\n'
        expected = prefix + body
        if path.exists() and path.read_text() != expected:
            raise RuntimeError(f'existing derived host differs: {path}')
        path.write_text(expected)
        records[name] = {'original': str(originals[name]), 'original_sha256': sha(originals[name]),
                         'derived': str(path), 'derived_sha256': sha(path)}
    provenance = {'release': '0.10.0', 'commit': COMMIT, 'lane': lane,
                  'host_derivation_sha256': sha(__file__), 'files': records}
    (runtime / 'derivation.json').write_text(json.dumps(provenance, indent=2) + '\n')
    load('release_env', runtime / 'release_env.py')
    trainer = load('trainer', runtime / 'trainer.py')
    generate = load('generate', runtime / 'generate.py')
    return trainer, generate, provenance
