#!/usr/bin/env python3
"""Train / generate for the donor-latent study (native control ``pbulk`` vs variant ``pbulk,g.dre``).

    run.py train    --fold K --seed S --arm pbulk|pbulk,g.dre --updates U --device cuda --gpu-mem-fraction F
    run.py generate --fold K --seed S --arm ... --updates U [--mode cellshape@0.5:biascal:stagehalf]

Fits:      RUNS/fits/u{U}/{arm}/hvg2000/fold{K}/seed{S}
Generated: RUNS/generated/u{U}/{arm}/fold{K}/seed{S}/{mode}.npz (+ .receipt.json)
"""
import os
os.environ.update(OMP_NUM_THREADS='3', OPENBLAS_NUM_THREADS='3', MKL_NUM_THREADS='3',
                  CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONDONTWRITEBYTECODE='1')
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dl_host

CACHES = Path('/ml2/alz/runs/improve_20261109/caches/hvg2000')
ARMS = ('pbulk', 'g.dre,pbulk')
DEFAULT_MODE = 'cellshape@0.5:biascal:stagehalf'


def canonical(arm):
    flags = ','.join(sorted(arm.split(',')))
    if flags not in ARMS:
        raise SystemExit(f'arm must be one of {ARMS}')
    return flags


def fit_path(arm, fold, seed, updates):
    return dl_host.RUNS / 'fits' / f'u{updates}' / arm / 'hvg2000' / f'fold{fold}' / f'seed{seed}'


def gen_path(arm, fold, seed, updates, mode):
    return dl_host.RUNS / 'generated' / f'u{updates}' / arm / f'fold{fold}' / f'seed{seed}' / f'{mode}.npz'


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp'); tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n'); tmp.replace(path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('train', 'generate'))
    p.add_argument('--fold', type=int, required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--arm', required=True)
    p.add_argument('--updates', type=int, required=True)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    p.add_argument('--gpu-mem-fraction', type=float, default=.06)
    p.add_argument('--mode', default=DEFAULT_MODE)
    a = p.parse_args(argv)
    arm = canonical(a.arm)
    if not 0 < a.gpu_mem_fraction <= .08:
        raise SystemExit('GPU memory fraction must be in (0, .08]')
    if a.device == 'cuda' and not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise SystemExit('set CUDA_VISIBLE_DEVICES explicitly')
    if a.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    cache = CACHES / f'fold{a.fold}'
    task = f'{a.command}-{arm}-f{a.fold}-s{a.seed}-u{a.updates}'.replace(',', '+')
    T, G, arms, provenance = dl_host.activate(a.device, a.gpu_mem_fraction, task=task)
    fit = fit_path(arm, a.fold, a.seed, a.updates)
    if a.command == 'train':
        out_root = dl_host.RUNS / 'fits' / f'u{a.updates}'
        args = T.parser().parse_args(['--cache', str(cache), '--arm', arm, '--seed', str(a.seed), '--updates', str(a.updates),
                                      '--out-root', str(out_root), '--device', a.device, '--threads', '3',
                                      '--gpu-mem-fraction', str(a.gpu_mem_fraction)])
        original = G.generate_fit
        G.generate_fit = lambda *x, **kw: (None, {'status': 'deferred_to_explicit_generation'})
        try:
            done = T.run(args)
        finally:
            G.generate_fit = original
        if Path(done).resolve() != fit.resolve():
            raise RuntimeError(f'unexpected fit folder {done}')
        q = T.torch.load(fit / 'final.pt', map_location='cpu', weights_only=False)
        record = q['policy']['optimizers'][1]['regularizer']['record']
        audit = dict(completed_updates=q['completed_updates'], penalty_calls=record['calls'],
                     observed_steps=record['observed_steps'], native_recipe=q['native_recipe'],
                     every_update_regularized=record['calls'] == record['observed_steps'] == q['completed_updates'] == a.updates,
                     provenance=provenance)
        if 'g.dre' in arm:
            gs = q['policy']['models']['generator']
            audit['random_effect_fro'] = {k: float(gs[k].norm()) for k in ('re_scale.weight', 're_loc.weight')}
        write(fit / 'donor_latent_audit.json', audit)
        print(json.dumps({k: v for k, v in audit.items() if k != 'provenance'}), flush=True)
    else:
        out = gen_path(arm, a.fold, a.seed, a.updates, a.mode)
        if out.exists():
            raise SystemExit(f'{out} exists; inspect before regenerating')

        def streams(fold, index):  # the application's named generation streams (repeat 0)
            children = T.np.random.SeedSequence([20261114, 6, fold, index, 0]).spawn(5)
            return {key: int(child.generate_state(1, dtype=T.np.uint64)[0]) % (2 ** 63 - 16)
                    for key, child in zip(G.GEN_STREAMS, children)}
        G.generation_streams = streams
        _, receipt = G.generate_fit(fit, cache, mode=a.mode, out=out, device=a.device)
        receipt.update(generation_streams=[20261114, 6, a.fold, 'development donor index', 0], provenance=provenance,
                       arm=arm, updates=a.updates)
        write(out.with_name(out.stem + '.receipt.json'), receipt)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
