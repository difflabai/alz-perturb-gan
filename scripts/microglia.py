#!/usr/bin/env python3
"""Train or generate with the committed development-selected microglia recipe.

The root package is the older synthetic smoke study. This entry point routes
to the human-microglia runtime and its explicit configuration. --dry-run and
config work without raw cells, checkpoints, torch or the original environment.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'analysis/improve_20261109'
RUNTIME = STUDY / 'r5'
CONFIG = STUDY / 'default_config.json'


def load_config(path=CONFIG):
    cfg = json.loads(Path(path).read_text())
    t, g, runtime = (cfg[k] for k in ('training', 'generation', 'runtime'))
    if cfg['schema_version'] != 1 or t['arm'] != 'pbulk' or t['panel'] != 'hvg2000' or t['updates'] != 20000:
        raise ValueError('configuration is not the validated pbulk/hvg2000/20000 development recipe')
    expected = {'particlegan_version': '0.9.7', 'recipe': 'bispf_optimized', 'particles': 512, 'z_dim': 8,
                'batch_donors': 32, 'cells_per_bag': 128, 'generator_lr': .012, 'critic_lr': .018,
                'particle_lr': .06, 'reg_coeff': 1., 'reg_kappa': 1., 'prior_reg': .01, 'native_ema_decay': 0.}
    if any(t.get(k) != v for k, v in expected.items()):
        raise ValueError('configuration metadata differs from the pinned trainer recipe')
    if g['generated_donors_per_target'] != 32 or g['cells_per_generated_donor'] != 256:
        raise ValueError('the evaluated generation contract is M=32, K=256')
    if g['mode'] not in ('biascal@stagehalf', 'cellshape@0.5:biascal:stagehalf'):
        raise ValueError('generation mode is outside the documented default-selection pass')
    if runtime['threads'] != 3 or not 0 < runtime['gpu_memory_fraction'] <= .04:
        raise ValueError('configuration exceeds the shared-machine resource limits')
    if cfg['data']['external_donors'] or cfg['data']['sealed_evaluation_donors']:
        raise ValueError('the default uses development donors only')
    return cfg


def cache_path(cfg, fold):
    return ROOT / 'runs/improve_20261109/caches' / cfg['training']['panel'] / f'fold{fold}'


def resolve_profile(cfg, name):
    """Choose an explicit read-out independently of the committed default.

    Keep the original profile-RMSE reference available if a distribution
    read-out later passes its prospective promotion rule.
    """
    if name == 'default':
        return cfg
    profiles = {**cfg.get('experimental_profiles', {}), **cfg.get('reference_profiles', {})}
    profile = profiles[name]
    resolved = copy.deepcopy(cfg)
    resolved['name'] += '-' + name
    resolved['generation']['mode'] = profile['mode']
    resolved['generation']['calibration'] = profile['calibration']
    resolved['status'] = profile['status']
    return resolved


def generation_command(cfg, fit, cache, out, device, memory):
    mode = cfg['generation']['mode']
    module = 'mode_cellshape.py' if mode.startswith('cellshape@') else 'generate.py'
    cmd = [sys.executable, '-B', str(RUNTIME / module), '--fit', str(fit), '--cache', str(cache),
           '--mode', mode, '--out', str(out), '--device', device]
    if device == 'cuda':
        cmd += ['--gpu-mem-fraction', str(memory)]
    return cmd


def check_fit(fit, cfg, fold, seed):
    saved = json.loads((fit / 'config.json').read_text())
    expected = {'arm': cfg['training']['arm'], 'panel': cfg['training']['panel'],
                'updates': cfg['training']['updates'], 'fold': fold, 'seed': seed}
    if any(saved.get(k) != v for k, v in expected.items()) or not saved.get('protocol_split'):
        raise ValueError('fit does not match the default recipe, fold, seed and development split')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['config', 'train', 'generate'], nargs='?', default='config')
    p.add_argument('--fold', type=int, choices=range(9), default=0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--fit', type=Path, help='existing compatible fit; otherwise use the default fit output root')
    p.add_argument('--out', type=Path, help='generation NPZ output; original fit folders are never modified')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--profile', choices=['default', 'count-shape', 'profile-rmse'], default='default',
                   help='choose the committed default, experimental count-shape, or original profile-RMSE read-out')
    p.add_argument('--gpu-mem-fraction', type=float, default=None)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args(argv)
    cfg = resolve_profile(load_config(), args.profile)
    if args.command == 'config':
        print(json.dumps(cfg, indent=2))
        return 0
    if args.seed < 0:
        p.error('--seed must be nonnegative')
    memory = cfg['runtime']['gpu_memory_fraction'] if args.gpu_mem_fraction is None else args.gpu_mem_fraction
    if not 0 < memory <= cfg['runtime']['gpu_memory_fraction']:
        p.error('--gpu-mem-fraction must be positive and no larger than the committed cap')
    if args.command == 'train' and args.fit is not None:
        p.error('--fit is for generate; train always uses the default fit output root')
    t = cfg['training']
    cache = cache_path(cfg, args.fold)
    fit = (args.fit.resolve() if args.fit else ROOT / cfg['runtime']['fit_output_root'] /
           t['arm'] / t['panel'] / f'fold{args.fold}/seed{args.seed}')
    out = args.out.resolve() if args.out else ROOT / cfg['runtime']['generation_output_root'] / \
        f'fold{args.fold}/seed{args.seed}' / f"generated_{cfg['generation']['mode']}.npz"
    if out.parent == fit or fit in out.parents:
        p.error('--out must be outside the fit directory')
    commands = []
    if args.command == 'train':
        cmd = [sys.executable, '-B', str(RUNTIME / 'trainer.py'), '--cache', str(cache), '--arm', t['arm'],
               '--seed', str(args.seed), '--updates', str(t['updates']), '--out-root',
               str(ROOT / cfg['runtime']['fit_output_root']), '--device', args.device,
               '--threads', str(cfg['runtime']['threads'])]
        if args.device == 'cuda':
            cmd += ['--gpu-mem-fraction', str(memory)]
        commands.append(cmd)
    commands.append(generation_command(cfg, fit, cache, out, args.device, memory))
    if args.dry_run:
        print(json.dumps({'config': str(CONFIG), 'recipe': cfg['name'], 'commands': commands}, indent=2))
        return 0
    if not (cache / 'manifest.json').exists():
        p.error(f'missing development cache {cache}; see analysis/improve_20261109/README.md')
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS='3', OPENBLAS_NUM_THREADS='3', MKL_NUM_THREADS='3', PYTHONDONTWRITEBYTECODE='1')
    if args.device == 'cpu':
        env['CUDA_VISIBLE_DEVICES'] = ''
    elif not env.get('CUDA_VISIBLE_DEVICES'):
        p.error('set CUDA_VISIBLE_DEVICES explicitly to a shared GPU before using --device cuda')
    if args.command == 'train':
        subprocess.run(commands.pop(0), check=True, env=env)
    check_fit(fit, cfg, args.fold, args.seed)
    subprocess.run(commands[0], check=True, env=env)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f'microglia: {exc}')
