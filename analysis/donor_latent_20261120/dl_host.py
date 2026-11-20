"""Private native-EMPF host for the donor-latent study.

Same derivation as analysis/empf_comparison_20261117/host.py ``activate('native')``
(release_env -> 0.10.0, r5 trainer through adapter.derive), but:
* sources are read from THIS worktree (byte-identical to the pinned /ml2/alz
  copies; checked against the frozen runtime source manifest below), and
* the derived files are written under RUNS (outside git and outside /ml2/alz's
  artifacts/), so the running confirmation queue's host directory is never touched.
Then the ``g.dre`` plug-in is registered in the frozen r5 arm registry.
"""
from __future__ import annotations

import os
os.environ.update(OMP_NUM_THREADS='3', OPENBLAS_NUM_THREADS='3', MKL_NUM_THREADS='3',
                  CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONDONTWRITEBYTECODE='1')
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT / 'analysis/improve_20261109/r5'
EMPF = ROOT / 'analysis/empf_comparison_20261117'
RUNS = Path('/ml2/alz/runs/research_20261120/donor-latent')
PINNED_MANIFEST = Path('/ml2/alz/analysis/default_empf_20261117/runtime_source_manifest.json')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def check_pinned():
    """Every worktree source used here equals the application's pinned source hash."""
    pins = json.loads(PINNED_MANIFEST.read_text())['sources']
    prefixes = ('analysis/improve_20261109/r5/', 'analysis/faithful097/', 'analysis/empf_comparison_20261117/')
    used = sorted(rel for rel in pins if rel.startswith(prefixes))
    for rel in ('analysis/improve_20261109/r5/trainer.py', 'analysis/improve_20261109/r5/generate.py',
                'analysis/improve_20261109/r5/arms.py', 'analysis/empf_comparison_20261117/adapter.py'):
        if rel not in used:
            raise ValueError(f'expected pinned source missing from manifest: {rel}')
    for rel in used:
        if sha(ROOT / rel) != pins[rel]:
            raise ValueError(f'worktree source differs from the pinned application source: {rel}')
    return {'n_pinned_sources_checked': len(used), 'manifest_sha256': sha(PINNED_MANIFEST)}


def activate(device='cpu', memory=.04, task='default'):
    if any(k in sys.modules for k in ('particlegan', 'trainer', 'generate', 'release_env')):
        raise RuntimeError('donor-latent host requires a fresh subprocess')
    checked = check_pinned()
    runtime = RUNS / 'host' / task
    runtime.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(EMPF))
    from host import replace_once
    from adapter import derive
    originals = {'release_env.py': ROOT / 'analysis/faithful097/model/release_env.py',
                 'trainer.py': OLD / 'trainer.py', 'generate.py': OLD / 'generate.py'}
    bodies = {n: p.read_text() for n, p in originals.items()}
    bodies['release_env.py'] = bodies['release_env.py'].replace('particlegan-0.9.7', 'particlegan-0.10.0').replace(
        "VERSION='0.9.7'", "VERSION='0.10.0'")
    bodies['trainer.py'] = replace_once(bodies['trainer.py'], 'HERE = Path(__file__).resolve().parent', f'HERE = Path({str(OLD)!r})')
    bodies['trainer.py'] = replace_once(bodies['trainer.py'], "DEFAULT_OUT_ROOT = RUNS / 'fits_r5'",
                                        f'DEFAULT_OUT_ROOT = Path({str(RUNS / "fits")!r})')
    bodies['trainer.py'] = derive(bodies['trainer.py'], 'native')
    records = {}
    for name, body in bodies.items():
        path = runtime / name
        text = '# Derived by analysis/donor_latent_20261120/dl_host.py (native EMPF lane).\n' + body
        if not path.exists() or path.read_text() != text:
            tmp = path.with_suffix('.tmp'); tmp.write_text(text); tmp.replace(path)
        records[name] = dict(original=str(originals[name]), original_sha256=sha(originals[name]), derived_sha256=sha(path))
    load('release_env', runtime / 'release_env.py')
    T = load('trainer', runtime / 'trainer.py')
    G = load('generate', runtime / 'generate.py')
    importlib.import_module('mode_cellshape')
    import arms
    if Path(arms.__file__).resolve().parent != OLD:
        raise ImportError('wrong arms module')
    sys.path.insert(0, str(HERE))
    import arm_dre
    arm_dre.register(arms)
    torch = T.torch
    torch.set_num_threads(3)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if device == 'cuda':
        torch.cuda.set_per_process_memory_fraction(memory)
    provenance = dict(release='0.10.0', lane='native', host='donor_latent_20261120/dl_host.py',
                      host_sha256=sha(__file__), arm_dre_sha256=sha(HERE / 'arm_dre.py'), files=records, sources=checked)
    return T, G, arms, provenance
