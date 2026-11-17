#!/usr/bin/env python3
"""Train or generate with the owner-selected experimental EMPF default.

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
RUNTIME = ROOT / 'scripts/microglia_runtime.py'
CONFIG = ROOT / 'configs/microglia_default.json'
RECIPE = ROOT / 'analysis/empf_comparison_20261117/resolved_recipes.json'


def load_config(path=CONFIG):
    cfg = json.loads(Path(path).read_text())
    t, g, runtime = (cfg[k] for k in ('training', 'generation', 'runtime'))
    if cfg['schema_version'] != 1 or t['arm'] != 'pbulk' or t['panel'] != 'hvg2000' or t['updates'] != 20000:
        raise ValueError('configuration is not the validated pbulk/hvg2000/20000 development recipe')
    expected = {'particlegan_version': '0.10.0', 'recipe': 'empf', 'particles': 512, 'z_dim': 8,
                'batch_donors': 32, 'cells_per_bag': 128, 'generator_lr': .012, 'critic_lr': .018,
                'particle_lr': .03, 'reg_coeff': 1., 'reg_kappa': 1., 'prior_reg': .01, 'native_ema_decay': 0.,
                'reg_every': 1, 'lazy_k': 1, 'ebi_beta': 6., 'optimizer_smoothing': .001,
                'optimizer_dense_profile': 'newton_schulz_hybrid_v1', 'optimizer_svd_backend': 'batched',
                'training_precision': 'bf16_channels_last'}
    if any(t.get(k) != v for k, v in expected.items()):
        raise ValueError('configuration metadata differs from the pinned trainer recipe')
    if g['generated_donors_per_target'] != 32 or g['cells_per_generated_donor'] != 256:
        raise ValueError('the evaluated generation contract is M=32, K=256')
    if g['mode'] not in ('biascal@stagehalf', 'cellshape@0.5:biascal:stagehalf'):
        raise ValueError('generation mode is outside the documented default-selection pass')
    if g.get('generation_stream_namespace') != [20261114, 6]:
        raise ValueError('generation streams differ from the evaluated EMPF pilot')
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
    resolved['evidence'] = profile.get('evidence', cfg['evidence'])
    return resolved


def generation_command(cfg, fit, cache, out, device, memory, seed=0):
    mode = cfg['generation']['mode']
    cmd = [sys.executable, '-B', str(RUNTIME), 'generate', '--fit', str(fit), '--cache', str(cache),
           '--mode', mode, '--out', str(out), '--device', device, '--threads', str(cfg['runtime']['threads']),
           '--seed', str(seed)]
    if device == 'cuda':
        cmd += ['--gpu-mem-fraction', str(memory)]
    return cmd


def check_fit(fit, cfg, fold, seed):
    fit = Path(fit)
    saved = json.loads((fit / 'config.json').read_text())
    expected = {'arm': cfg['training']['arm'], 'panel': cfg['training']['panel'],
                'updates': cfg['training']['updates'], 'fold': fold, 'seed': seed,
                'particlegan_version': '0.10.0', 'native_recipe': 'empf', 'batch': [32, 128],
                'flags': ['pbulk'], 'critic_features': 2150}
    if any(saved.get(k) != v for k, v in expected.items()) or not saved.get('protocol_split'):
        raise ValueError('fit does not match the default recipe, fold, seed and development split')
    spec = json.loads(RECIPE.read_text())['native']
    if saved.get('recipe') != spec['recipe'] or saved.get('hyperparameters') != spec['hyperparameters']:
        raise ValueError('fit contains a different native EMPF recipe or hyperparameters')
    receipt = json.loads((fit / 'fit_receipt.json').read_text())
    audit = json.loads((fit / 'no_skip_audit.json').read_text())
    if receipt.get('completed_updates') != 20000 or receipt.get('endpoint') != 20000 or receipt.get('snapshots_missing'):
        raise ValueError('fit has not completed the full20000-update endpoint')
    if any(audit.get(k) != 20000 for k in ('penalty_calls', 'observed_steps', 'completed_updates')) or audit.get('lazy_k') != 1:
        raise ValueError('fit does not verify regularization at every update')
    if audit.get('final_checkpoint_sha256') != receipt.get('final_SHA256'):
        raise ValueError('regularizer audit is for a different final checkpoint')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['config', 'train', 'generate'], nargs='?', default='config')
    p.add_argument('--fold', type=int, choices=range(9), default=0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--fit', type=Path, help='existing compatible fit; otherwise use the default fit output root')
    p.add_argument('--out', type=Path, help='generation NPZ output; original fit folders are never modified')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--profile', choices=['default', 'count-shape', 'profile-rmse'], default='default',
                   help='choose the committed default or an explicit distribution/profile-RMSE read-out')
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
    fit = (ROOT / cfg['runtime']['fit_output_root'] / t['arm'] / t['panel'] /
           f'fold{args.fold}/seed{args.seed}')
    if args.fit:
        fit = args.fit.resolve()
    elif args.command == 'generate' and (args.fold, args.seed) == (cfg['runtime']['selected_fold'], cfg['runtime']['selected_seed']):
        fit = ROOT / cfg['runtime']['selected_fit']
    out = args.out.resolve() if args.out else ROOT / cfg['runtime']['generation_output_root'] / \
        f'fold{args.fold}/seed{args.seed}' / f"generated_{cfg['generation']['mode']}.npz"
    if out.parent == fit or fit in out.parents:
        p.error('--out must be outside the fit directory')
    commands = []
    if args.command == 'train':
        cmd = [sys.executable, '-B', str(RUNTIME), 'train', '--cache', str(cache), '--arm', t['arm'],
               '--seed', str(args.seed), '--updates', str(t['updates']), '--out-root',
               str(ROOT / cfg['runtime']['fit_output_root']), '--device', args.device,
               '--threads', str(cfg['runtime']['threads'])]
        if args.device == 'cuda':
            cmd += ['--gpu-mem-fraction', str(memory)]
        commands.append(cmd)
    commands.append(generation_command(cfg, fit, cache, out, args.device, memory,args.seed))
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
