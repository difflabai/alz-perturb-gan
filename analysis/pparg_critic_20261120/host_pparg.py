"""Activate the native EMPF host (same derivation as analysis/empf_comparison_20261117/host.py)
for this study, writing derived sources under the study's own run directory.

Differences from host.activate('native'), none of which change the training code path:
* derived trainer/generate/release_env files are written to RUN/host (outside git and
  outside /ml2/alz/artifacts), once, atomically, so concurrent processes never rewrite
  a file another process is importing;
* after activation, this study's plug-in arm ``offpc`` (arm_offpc.py) is registered.
Released ParticleGAN 0.10.0 is imported unchanged from its isolated site-packages.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
COMP = ROOT / 'analysis/empf_comparison_20261117'
OLD = ROOT / 'analysis/improve_20261109/r5'
MODEL = ROOT / 'analysis/faithful097/model'
RUN = Path(os.environ.get('PPARG_CRITIC_RUN', '/ml2/alz/runs/research_20261120/pparg-critic'))
CACHE = Path('/ml2/alz/runs/improve_20261109/caches/hvg2000')
APP_MODE = 'cellshape@0.5:biascal:stagehalf'
RAW_MODE = 'snapens'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _write_once(path, text):
    if path.exists():
        if path.read_text() != text:
            raise RuntimeError(f'existing derived host differs: {path}')
        return
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    tmp.write_text(text); tmp.replace(path)


def derived_bodies(runtime):
    """Exactly host.activate('native')'s derivation, except the default out root (unused: --out-root is explicit)."""
    sys.path.insert(0, str(COMP))
    from host import replace_once
    from adapter import derive
    originals = {'release_env.py': MODEL/'release_env.py', 'trainer.py': OLD/'trainer.py', 'generate.py': OLD/'generate.py'}
    bodies = {n: p.read_text() for n, p in originals.items()}
    bodies['release_env.py'] = bodies['release_env.py'].replace('particlegan-0.9.7', 'particlegan-0.10.0').replace("VERSION='0.9.7'", "VERSION='0.10.0'")
    bodies['trainer.py'] = replace_once(bodies['trainer.py'], 'HERE = Path(__file__).resolve().parent', f'HERE = Path({str(OLD)!r})')
    bodies['trainer.py'] = replace_once(bodies['trainer.py'], "DEFAULT_OUT_ROOT = RUNS / 'fits_r5'", f'DEFAULT_OUT_ROOT = Path({str(RUN/"fits")!r})')
    bodies['trainer.py'] = derive(bodies['trainer.py'], 'native')
    header = '# Derived by analysis/pparg_critic_20261120/host_pparg.py (= empf_comparison host native derivation).\n'
    return originals, {n: header + b for n, b in bodies.items()}


def activate():
    if any(k in sys.modules for k in ('particlegan', 'trainer', 'generate', 'release_env')):
        raise RuntimeError('activation requires a fresh process')
    runtime = RUN / 'host'; runtime.mkdir(parents=True, exist_ok=True)
    originals, bodies = derived_bodies(runtime)
    records = {}
    for name, body in bodies.items():
        _write_once(runtime/name, body)
        records[name] = dict(original=str(originals[name]), original_sha256=sha(originals[name]),
                             derived=str(runtime/name), derived_sha256=sha(runtime/name))
    load('release_env', runtime/'release_env.py')
    T = load('trainer', runtime/'trainer.py')
    G = load('generate', runtime/'generate.py')
    import mode_cellshape  # noqa: F401  registers the application readout (as microglia_runtime.activate)
    sys.path.insert(0, str(HERE))
    import arm_offpc  # noqa: F401  registers the study arm
    provenance = dict(release='0.10.0', lane='native', host='host_pparg', files=records,
                      host_sha256=sha(__file__), arm_offpc_sha256=sha(HERE/'arm_offpc.py'))
    return T, G, provenance


def app_streams(T, G):
    """Application readout streams (scripts/microglia_runtime.py): SeedSequence([20261114,6,fold,index,0])."""
    def streams(fold, index):
        children = T.np.random.SeedSequence([20261114, 6, fold, index, 0]).spawn(5)
        return {key: int(child.generate_state(1, dtype=T.np.uint64)[0]) % (2**63-16) for key, child in zip(G.GEN_STREAMS, children)}
    G.generation_streams = streams


def deterministic(T, device, memory):
    T.torch.set_num_threads(3)
    T.torch.use_deterministic_algorithms(True)
    T.torch.backends.cuda.matmul.allow_tf32 = False
    T.torch.backends.cudnn.allow_tf32 = False
    T.torch.backends.cudnn.benchmark = False
    T.torch.backends.cudnn.deterministic = True
    if device == 'cuda':
        T.torch.cuda.set_per_process_memory_fraction(memory)
