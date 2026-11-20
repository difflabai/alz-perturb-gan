#!/usr/bin/env python3
"""Train or generate with the owner-selected experimental EMPF default.

The root package is the older synthetic smoke study. This entry point routes
to the human-microglia runtime and its explicit configuration. --dry-run and
config work without raw cells, checkpoints, torch or the original environment.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
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
SOURCE_MANIFEST = ROOT / 'analysis/default_empf_20261117/runtime_source_manifest.json'
NATIVE_RELEASE = ROOT / 'artifacts/native-releases/particlegan-0.10.0/site-packages/particlegan'
# Exit codes of `microglia_runtime.py train` (scripts/microglia_runtime.py).
TRAIN_ALREADY_DONE, TRAIN_DAMAGED, TRAIN_BUSY = 3, 4, 75


def recorded_fit_state(fit):
    """The state recorded in <fit>/fit_state.json (read-only): None when absent, 'UNREADABLE' when damaged."""
    try:
        return json.loads((Path(fit) / 'fit_state.json').read_text())['state']
    except FileNotFoundError:
        return None
    except (OSError, ValueError, KeyError, TypeError):
        return 'UNREADABLE'


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


def generation_command(cfg, fit, cache, out, device, memory, seed=0, profile='default', force=False):
    mode = cfg['generation']['mode']
    cmd = [sys.executable, '-B', str(RUNTIME), 'generate', '--fit', str(fit), '--cache', str(cache),
           '--mode', mode, '--profile', profile, '--out', str(out), '--device', device,
           '--threads', str(cfg['runtime']['threads']), '--seed', str(seed)]
    if device == 'cuda':
        cmd += ['--gpu-mem-fraction', str(memory)]
    if force:
        cmd.append('--force')
    return cmd


def receipt_path(out):
    return Path(out).with_name(Path(out).stem + '.receipt.json')


def choose_fit(cfg, command, fold, seed, explicit=None, source='auto'):
    """Return (fit, source). The retained selected fit and newly trained fits never share a path."""
    t, runtime = cfg['training'], cfg['runtime']
    trained = ROOT / runtime['fit_output_root'] / t['arm'] / t['panel'] / f'fold{fold}/seed{seed}'
    if explicit is not None:
        return Path(explicit).resolve(), 'explicit'
    is_selected = (fold, seed) == (runtime['selected_fold'], runtime['selected_seed'])
    if source == 'selected' and not is_selected:
        raise ValueError(f"the selected fit is fold{runtime['selected_fold']}/seed{runtime['selected_seed']}")
    if command == 'generate' and is_selected and source in ('auto', 'selected'):
        return ROOT / runtime['selected_fit'], 'selected'
    return trained, 'trained'


def default_out(cfg, fold, seed, fit_source):
    root = ROOT / cfg['runtime']['generation_output_root']
    if fit_source == 'selected':
        root = root / 'selected'
    return root / f'fold{fold}/seed{seed}' / f"generated_{cfg['generation']['mode']}.npz"


def source_status():
    try:
        manifest = json.loads(SOURCE_MANIFEST.read_text())
    except (OSError, ValueError) as exc:
        return {'ok': False, 'error': str(exc)}
    bad = []
    for name, digest in manifest['sources'].items():
        path = Path(name) if Path(name).is_absolute() else ROOT / name
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            bad.append(name)
    return {'ok': not bad, 'sources': len(manifest['sources']), 'missing_or_changed': len(bad),
            'first_missing_or_changed': bad[:5]}


def preflight(cfg, cache, fit, fold, seed, out):
    """Non-fatal readiness report for --dry-run."""
    try:
        check_fit(fit, cfg, fold, seed)
        fit_state = 'complete'
    except FileNotFoundError:
        fit_state = 'absent or incomplete'
    except (ValueError, KeyError, OSError) as exc:
        fit_state = f'not usable: {exc}'
    return {'cache': (cache / 'manifest.json').exists(), 'fit': fit_state, 'fit_state': recorded_fit_state(fit),
            'output_exists': [str(x) for x in (out, receipt_path(out)) if x.exists()],
            'native_release': NATIVE_RELEASE.is_dir(), 'torch': importlib.util.find_spec('torch') is not None,
            'sources': source_status()}


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
    p.add_argument('--fit-source', choices=['auto', 'selected', 'trained'], default='auto',
                   help='generate: auto uses the retained selected fit at its fold/seed and a trained fit elsewhere')
    p.add_argument('--gpu-mem-fraction', type=float, default=None)
    p.add_argument('--force', action='store_true', help='replace an existing generation output and its receipt')
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
    if args.command == 'train' and (args.fit is not None or args.fit_source != 'auto'):
        p.error('--fit/--fit-source are for generate; train always uses the default fit output root')
    if args.fit is not None and args.fit_source != 'auto':
        p.error('give either --fit or --fit-source')
    t = cfg['training']
    cache = cache_path(cfg, args.fold)
    try:
        fit, fit_source = choose_fit(cfg, args.command, args.fold, args.seed, args.fit, args.fit_source)
    except ValueError as exc:
        p.error(str(exc))
    out = args.out.resolve() if args.out else default_out(cfg, args.fold, args.seed, fit_source)
    if out.parent == fit or fit in out.parents:
        p.error('--out must be outside the fit directory')
    plan = {'config': str(CONFIG), 'recipe': cfg['name'], 'profile': args.profile,
            'fit': str(fit), 'fit_source': fit_source, 'out': str(out)}
    trained, _ = choose_fit(cfg, 'train', args.fold, args.seed)
    if fit_source == 'selected' and (trained / 'final.pt').exists():
        plan['notice'] = f'using the retained selected fit; a trained fit also exists at {trained} (--fit-source trained)'
    commands = []
    train_needed = args.command == 'train'
    # Skip only a fit whose recorded state (if any) is AUDITED: a RECEIPTED fit with a passing
    # no_skip_audit.json (the audit landed, its state write did not) goes to the runtime, which
    # reconciles it to AUDITED and exits 3; generate would refuse it.
    if train_needed and (fit / 'no_skip_audit.json').exists() and recorded_fit_state(fit) in (None, 'AUDITED'):
        try:
            check_fit(fit, cfg, args.fold, args.seed)
            train_needed = False  # complete and audited: never re-audit it under different sources
        except (ValueError, KeyError, OSError):
            pass
    if train_needed:
        cmd = [sys.executable, '-B', str(RUNTIME), 'train', '--cache', str(cache), '--arm', t['arm'],
               '--seed', str(args.seed), '--updates', str(t['updates']), '--out-root',
               str(ROOT / cfg['runtime']['fit_output_root']), '--device', args.device,
               '--threads', str(cfg['runtime']['threads'])]
        if args.device == 'cuda':
            cmd += ['--gpu-mem-fraction', str(memory)]
        commands.append(cmd)
    commands.append(generation_command(cfg, fit, cache, out, args.device, memory, args.seed,
                                       args.profile, args.force))
    if args.command == 'train' and not train_needed:
        plan['training'] = 'skipped: fit is complete and audited'
    plan['commands'] = commands
    if args.dry_run:
        plan['preflight'] = preflight(cfg, cache, fit, args.fold, args.seed, out)
        print(json.dumps(plan, indent=2))
        return 0
    existing = [str(x) for x in (out, receipt_path(out)) if x.exists()]
    if existing and not args.force:
        p.error(f'generation output exists: {", ".join(existing)}; use --force to replace it or choose --out')
    if not (cache / 'manifest.json').exists():
        p.error(f'missing development cache {cache}; see analysis/improve_20261109/README.md')
    print(f'microglia: fit {fit} ({fit_source}); output {out}', flush=True)
    if 'notice' in plan:
        print(f"microglia: {plan['notice']}", flush=True)
    if args.command == 'train' and not train_needed:
        print('microglia: fit is complete and audited; generating only', flush=True)
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS='3', OPENBLAS_NUM_THREADS='3', MKL_NUM_THREADS='3', PYTHONDONTWRITEBYTECODE='1')
    if args.device == 'cpu':
        env['CUDA_VISIBLE_DEVICES'] = ''
    elif not env.get('CUDA_VISIBLE_DEVICES'):
        p.error('set CUDA_VISIBLE_DEVICES explicitly to a shared GPU before using --device cuda')
    if train_needed:
        cmd = commands.pop(0)
        rc = subprocess.run(cmd, env=env).returncode
        if rc == TRAIN_BUSY:
            print(f'microglia: fit {fit} is busy: another process is training it; rerun when it has finished',
                  file=sys.stderr, flush=True)
            return TRAIN_BUSY
        if rc == TRAIN_DAMAGED:
            raise ValueError(f'fit {fit} is DAMAGED (see the runtime message above); inspect it and run '
                             f'scripts/microglia_runtime.py train ... --repair')
        if rc not in (0, TRAIN_ALREADY_DONE):
            raise subprocess.CalledProcessError(rc, cmd)
    check_fit(fit, cfg, args.fold, args.seed)
    subprocess.run(commands[0], check=True, env=env)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, RuntimeError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f'microglia: {exc}')
