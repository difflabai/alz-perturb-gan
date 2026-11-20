"""Matched SEA-AD EMPF fits: native vs condition-aware wide residual.

Both lanes use the committed native EMPF derivation of the ALZ host
(analysis/empf_comparison_20261117/{host,adapter}.py, lane 'native'), the same
cache, seed, streams, recipe (total_steps 20000) and an equal execution cap of
``--updates`` updates. The adapter lane additionally calls
``cond_residual.install`` on the critic penalty right after ``trainer.build``.

The derived host files are written to a private directory under
runs/research_20261120/cond-residual/host/<lane>_fold<k>, never to the shared
artifacts/native-releases/... runtime directory used by other sessions.

Phases:  train  (GPU, TRAIN-only cache payloads)
         generate (CPU; pilot readout cellshape@0.5:biascal:stagehalf, pilot
                   generation streams; frozen architecture_score scoring)
"""
import os
os.environ.update(OMP_NUM_THREADS='3', OPENBLAS_NUM_THREADS='3', MKL_NUM_THREADS='3',
                  CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONDONTWRITEBYTECODE='1')
import argparse
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = Path('/ml2/alz')
STUDY = ROOT / 'analysis/empf_comparison_20261117'
OLD = ROOT / 'analysis/improve_20261109/r5'
OUT = ROOT / 'runs/research_20261120/cond-residual'
MODE = 'cellshape@0.5:biascal:stagehalf'
LANES = ('native', 'adapter')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def activate(lane, fold):
    """host.activate('native') with a private runtime directory."""
    if lane not in LANES:
        raise ValueError(lane)
    sys.path.insert(0, str(STUDY))
    import host  # analysis/empf_comparison_20261117/host.py (import only; its activate is not called)
    from adapter import derive
    runtime = OUT / 'host' / f'{lane}_fold{fold}'; runtime.mkdir(parents=True, exist_ok=True)
    originals = {'release_env.py': ROOT/'analysis/faithful097/model/release_env.py',
                 'trainer.py': OLD/'trainer.py', 'generate.py': OLD/'generate.py'}
    bodies = {n: p.read_text() for n, p in originals.items()}
    bodies['release_env.py'] = bodies['release_env.py'].replace('particlegan-0.9.7', 'particlegan-0.10.0').replace("VERSION='0.9.7'", "VERSION='0.10.0'")
    bodies['trainer.py'] = host.replace_once(bodies['trainer.py'], 'HERE = Path(__file__).resolve().parent', f'HERE = Path({str(OLD)!r})')
    bodies['trainer.py'] = host.replace_once(bodies['trainer.py'], "DEFAULT_OUT_ROOT = RUNS / 'fits_r5'",
                                             f'DEFAULT_OUT_ROOT = Path({str(OUT/"fits"/lane)!r})')
    bodies['trainer.py'] = derive(bodies['trainer.py'], 'native')
    files = {}
    for name, body in bodies.items():
        path = runtime / name
        path.write_text('# Derived by analysis/cond_residual_20261120/alz_run.py (native EMPF host derivation).\n' + body)
        files[name] = dict(original=str(originals[name]), original_sha256=sha(originals[name]), derived_sha256=sha(path))
    # Same native derivation as the shared runtime the selected pilot used:
    shared = ROOT / 'artifacts/native-releases/particlegan-0.10.0/alz-host-empf-native'
    for name in files:
        a = (runtime/name).read_text().split('\n', 1)[1]; b = (shared/name).read_text().split('\n', 1)[1]
        if a != b:
            raise RuntimeError(f'private derivation differs from the native pilot host: {name}')
    load('release_env', runtime/'release_env.py')
    T = load('trainer', runtime/'trainer.py'); G = load('generate', runtime/'generate.py')
    importlib.import_module('mode_cellshape')
    torch = T.torch
    torch.set_num_threads(3)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if lane == 'adapter':
        sys.path.insert(0, str(HERE))
        import cond_residual as CR
        original = T.build

        def build(ctx, seed, **kw):
            out = original(ctx, seed, **kw)
            CR.install(out[0][6])
            return out
        T.build = build
    return T, G, dict(lane=lane, files=files, adapter_sha256=sha(HERE/'cond_residual.py'), runner_sha256=sha(__file__))


def fit_dir(lane, fold, seed):
    return OUT / 'fits' / lane / f'pbulk/hvg2000/fold{fold}/seed{seed}'


def write(path, value):
    path = Path(path); tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n'); tmp.replace(path)


def train(lane, fold, seed, updates, memory):
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise SystemExit('set CUDA_VISIBLE_DEVICES')
    T, G, prov = activate(lane, fold)
    T.torch.cuda.set_per_process_memory_fraction(memory)
    cache = ROOT / f'runs/improve_20261109/caches/hvg2000/fold{fold}'
    args = T.parser().parse_args(['--cache', str(cache), '--arm', 'pbulk', '--seed', str(seed), '--updates', str(updates),
                                  '--out-root', str(OUT/'fits'/lane), '--device', 'cuda', '--threads', '3',
                                  '--gpu-mem-fraction', str(memory)])
    original = G.generate_fit
    G.generate_fit = lambda *a, **kw: (None, {'status': 'deferred'})
    try:
        fit = T.run(args)
    finally:
        G.generate_fit = original
    assert fit == fit_dir(lane, fold, seed), fit
    q = T.torch.load(fit/'final.pt', map_location='cpu', weights_only=False)
    record = q['policy']['optimizers'][1]['regularizer']['record']
    assert record['calls'] == record['observed_steps'] == q['completed_updates'] == updates
    assert q['recipe']['reg_every'] == 1
    write(fit/'cond_residual_receipt.json', dict(status='trained', lane=lane, fold=fold, seed=seed, updates=updates,
          regularizer_calls=record['calls'], ema_residual_final=record['empf']['ema_residual'],
          provenance=prov, final_sha256=sha(fit/'final.pt'),
          cache_manifest_sha256=sha(cache/'manifest.json')))
    print(f'trained {lane} fold{fold} seed{seed}: {fit}', flush=True)


def generate(lane, fold, seed):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    T, G, prov = activate(lane, fold)
    T.arms.parse_mode(MODE)
    fit = fit_dir(lane, fold, seed)
    receipt = json.loads((fit/'cond_residual_receipt.json').read_text())
    assert receipt['final_sha256'] == sha(fit/'final.pt')
    cache = ROOT / f'runs/improve_20261109/caches/hvg2000/fold{fold}'
    out = OUT / 'generated' / f'{lane}_fold{fold}_seed{seed}.npz'; out.parent.mkdir(parents=True, exist_ok=True)
    score = OUT / 'results' / f'{lane}_fold{fold}_seed{seed}.json'; score.parent.mkdir(parents=True, exist_ok=True)

    def streams(fold_, index):  # pilot streams (repeat0), identical for both lanes
        children = T.np.random.SeedSequence([20261114, 6, fold_, index, 0]).spawn(5)
        return {key: int(child.generate_state(1, dtype=T.np.uint64)[0]) % (2**63-16)
                for key, child in zip(G.GEN_STREAMS, children)}
    G.generation_streams = streams
    if not out.exists():
        _, gen_receipt = G.generate_fit(fit, cache, mode=MODE, out=out, device='cpu')
    else:
        gen_receipt = json.loads(out.with_name(out.stem + '.receipt.json').read_text())
    E = importlib.import_module('architecture_score')
    row = E.score(out, cache, score)
    write(OUT/'results'/f'{lane}_fold{fold}_seed{seed}.meta.json',
          dict(lane=lane, fold=fold, seed=seed, mode=MODE, states_used=gen_receipt['states_used'],
               generated_sha256=sha(out), provenance=prov, score_sha256=sha(score)))
    print(json.dumps({'lane': lane, 'fold': fold, 'historical_pb_rmse': row['historical']['pb_rmse'],
                      'heldout': row['intervals']['heldout']['percentile']}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('phase', choices=('train', 'generate'))
    p.add_argument('--lane', choices=LANES, required=True)
    p.add_argument('--fold', type=int, required=True); p.add_argument('--seed', type=int, default=0)
    p.add_argument('--updates', type=int, default=5000); p.add_argument('--gpu-mem-fraction', type=float, default=.04)
    a = p.parse_args()
    if a.phase == 'train':
        train(a.lane, a.fold, a.seed, a.updates, a.gpu_mem_fraction)
    else:
        generate(a.lane, a.fold, a.seed)
