"""Train one matched fold-0 fit (native control ``pbulk`` or variant ``offpc,pbulk``) with the
native EMPF host, then write the two declared readouts outside the fit directory.

python -B run_fit.py --arm pbulk|offpc,pbulk --seed S --updates U [--device cuda --gpu-mem-fraction F]
"""
import os
os.environ.update(OMP_NUM_THREADS='3', OPENBLAS_NUM_THREADS='3', MKL_NUM_THREADS='3',
                  CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONDONTWRITEBYTECODE='1')
import argparse
import json
from pathlib import Path
import host_pparg as H

DEFAULT_FIT = Path('/ml2/alz/runs/empf_comparison_20261117/fits/native/pbulk/hvg2000/fold0/seed0')
ARMS = ('pbulk', 'offpc,pbulk')


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--arm', choices=ARMS, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--updates', type=int, required=True)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    p.add_argument('--gpu-mem-fraction', type=float, default=.06)
    p.add_argument('--fold', type=int, default=0)
    p.add_argument('--skip-generate', action='store_true')
    a = p.parse_args(argv)
    if a.device == 'cuda' and not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise SystemExit('set CUDA_VISIBLE_DEVICES explicitly')
    if not 0 < a.gpu_mem_fraction <= .08:
        raise SystemExit('GPU fraction must be in (0, .08]')
    T, G, prov = H.activate()
    H.deterministic(T, a.device, a.gpu_mem_fraction)
    want = json.loads((DEFAULT_FIT/'config.json').read_text())['recipe']
    if json.loads(json.dumps(T.recipe().to_dict())) != want:
        raise ValueError('live recipe differs from the retained native EMPF fit recipe')
    cache = H.CACHE/f'fold{a.fold}'
    out_root = H.RUN/'fits'
    args = T.parser().parse_args(['--cache', str(cache), '--arm', a.arm, '--seed', str(a.seed), '--updates', str(a.updates),
                                  '--out-root', str(out_root), '--device', a.device, '--threads', '3',
                                  '--gpu-mem-fraction', str(a.gpu_mem_fraction)])
    original = G.generate_fit
    G.generate_fit = lambda *x, **kw: (None, {'status': 'deferred'})
    try:
        fit = T.run(args)
    finally:
        G.generate_fit = original
    q = T.torch.load(fit/'final.pt', map_location='cpu', weights_only=False)
    record = q['policy']['optimizers'][1]['regularizer']['record']
    audit = dict(arm=a.arm, seed=a.seed, updates=a.updates, completed_updates=q['completed_updates'],
                 penalty_calls=record['calls'], observed_steps=record['observed_steps'],
                 native_recipe=q['native_recipe'], recipe_equals_default_fit=q['recipe'] == want,
                 penalty_options=q['penalty_options'], particlegan_version=q['particlegan_version'], provenance=prov)
    assert record['calls'] == record['observed_steps'] == q['completed_updates'] == a.updates, audit
    del q
    (fit/'study_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(f'trained {fit}', flush=True)
    if a.skip_generate:
        return
    H.app_streams(T, G)
    gen_root = H.RUN/'generated'/a.arm.replace(',', '+')/f'fold{a.fold}'/f'seed{a.seed}'
    gen_root.mkdir(parents=True, exist_ok=True)
    for tag, mode in (('raw', H.RAW_MODE), ('app', H.APP_MODE)):
        out = gen_root/f'{tag}.npz'
        if out.exists():
            continue
        _, receipt = G.generate_fit(fit, cache, mode=mode, out=out, device='cpu')
        receipt.update(generation_streams=[20261114, 6, a.fold, 'development donor index', 0])
        (out.with_name(out.stem + '.receipt.json')).write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
        print(f'generated {out}', flush=True)


if __name__ == '__main__':
    main()
