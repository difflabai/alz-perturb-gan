"""Isolated EMPF host; preserves all earlier release-study sources."""
from pathlib import Path
import importlib.util
import hashlib
import json
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT / 'analysis/particlegan_upgrade_20261117'
OLD = ROOT / 'analysis/improve_20261109/r5'
RELEASE = ROOT / 'artifacts/native-releases/particlegan-0.10.0'
RUNS = ROOT / 'runs/empf_comparison_20261117'
COMMIT = '952c3b62ba434f7da194067923351ff5748ef8dd'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def replace_once(body, before, after):
    if body.count(before) != 1:
        raise ValueError('derivation expected one occurrence: ' + before[:80])
    return body.replace(before, after)


def activate(lane='native'):
    if lane not in ('native', 'matched'):
        raise ValueError('unknown EMPF lane')
    if any(k in sys.modules for k in ('particlegan', 'trainer', 'generate', 'release_env')):
        raise RuntimeError('EMPF lane requires a fresh subprocess')
    runtime = RELEASE / ('alz-host-empf-' + lane); runtime.mkdir(parents=True, exist_ok=True)
    originals = {'release_env.py': ROOT/'analysis/faithful097/model/release_env.py',
                 'trainer.py': OLD/'trainer.py', 'generate.py': OLD/'generate.py'}
    bodies = {n: p.read_text() for n, p in originals.items()}
    bodies['release_env.py'] = bodies['release_env.py'].replace('particlegan-0.9.7', 'particlegan-0.10.0').replace("VERSION='0.9.7'", "VERSION='0.10.0'")
    bodies['trainer.py'] = replace_once(bodies['trainer.py'],
        'HERE = Path(__file__).resolve().parent', f'HERE = Path({str(OLD)!r})')
    bodies['trainer.py'] = replace_once(bodies['trainer.py'],
        "DEFAULT_OUT_ROOT = RUNS / 'fits_r5'", f'DEFAULT_OUT_ROOT = Path({str(RUNS/"fits"/lane)!r})')
    from adapter import derive
    bodies['trainer.py'] = derive(bodies['trainer.py'], lane)
    records = {}
    for name, body in bodies.items():
        path = runtime/name
        expected = '# Derived by analysis/empf_comparison_20261117/host.py; see derivation.json.\n' + body
        if path.exists() and path.read_text() != expected:
            raise RuntimeError('existing derived EMPF host differs: ' + str(path))
        path.write_text(expected)
        records[name] = dict(original=str(originals[name]), original_sha256=sha(originals[name]),
                             derived=str(path), derived_sha256=sha(path))
    provenance = dict(release='0.10.0', commit=COMMIT, lane=lane,
                      host_derivation_sha256=sha(__file__), files=records)
    (runtime/'derivation.json').write_text(json.dumps(provenance, indent=2)+'\n')
    load('release_env', runtime/'release_env.py')
    return load('trainer', runtime/'trainer.py'), load('generate', runtime/'generate.py'), provenance
