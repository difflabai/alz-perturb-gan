"""A separately declared20k->40k continuation through the public execution cap."""
import hashlib
import json
from pathlib import Path
import shutil
import signal
import time

ROOT = Path('/ml2/alz')
INITIAL = 20000
TARGET = 40000
HORIZON = 20000
DIAGNOSTICS = (30000,)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_module():
    import released_adapter as model
    if Path(model.__file__).resolve() != ROOT / 'analysis/released_tuning_v096/released_adapter.py':
        raise ValueError('Frozen released count-model import origin differs')
    return model


def check_limits(ctx, initial, target, horizon, diagnostics):
    if ctx.device.type == 'cuda' and (initial, target, horizon, tuple(diagnostics)) != (INITIAL, TARGET, HORIZON, DIAGNOSTICS):
        raise ValueError('Native duration experiment is exactly20k->40k with30k diagnostic and unchanged20k recipe horizon')
    if any(type(value) is not int for value in (initial, target, horizon)) or not 0 < initial < target or horizon <= 0:
        raise ValueError('Finite positive integral extension clocks required')
    if any(type(value) is not int or not initial < value < target for value in diagnostics):
        raise ValueError('Diagnostic clocks must be strictly inside the extension')


def validate_parent(q, auth):
    model = model_module()
    parent = auth['extension']
    if (parent['initial_completed_updates'], parent['fixed_endpoint']) != (INITIAL, TARGET):
        raise ValueError('Wrong declared extension endpoint')
    if q.get('particlegan_version') != '0.9.6' or q.get('penalty_options') != {'input_domain': 'continuous'}:
        raise ValueError('Exact native096 continuous-domain state required')
    if q.get('schema') != 1 or q.get('method') != 'pg' or q.get('native_recipe') != 'bispf_optimized':
        raise ValueError('Exact released parent model required')
    if q.get('status') != 'completed' or q.get('completed_updates') != q.get('fixed_endpoint') or q['completed_updates'] != INITIAL:
        raise ValueError('Complete immutable20k parent required')
    if q.get('fold') != auth['cell']['fold'] or q.get('seed') != auth['cell']['seed'] or (q['fold'], q['seed']) != (0, 0):
        raise ValueError('Same fold0/seed0 required')
    if q.get('policy', {}).get('completed_steps') != INITIAL or q['policy'].get('device') != 'cuda:0' or q.get('usage', {}).get('steps') != INITIAL:
        raise ValueError('Original native CUDA and usage clocks required')
    if q.get('released_hyperparameters') != auth['released_hyperparameters'] or q.get('variant_config') != auth['variant_config']:
        raise ValueError('Parent hyperparameters or architecture changed')
    if q.get('cache_manifest_SHA256') != sha(Path(auth['cache_dir']) / 'manifest.json'):
        raise ValueError('Original training cache differs')
    if q.get('training_streams') != model.training_streams(0, 0):
        raise ValueError('Original eleven training streams required')
    if q.get('recipe') != model.recipe().to_dict() or q['policy'].get('recipe') != q['recipe'] or q['recipe']['total_steps'] != HORIZON:
        raise ValueError('Saved recipe must retain its original20k schedule horizon')


def snapshot(ctx, bundle, generators, rng, usage, seed, streams, status, *, initial=INITIAL, target=TARGET, horizon=HORIZON):
    check_limits(ctx, initial, target, horizon, () if ctx.device.type != 'cuda' else DIAGNOSTICS)
    if not initial <= bundle[-1].completed_steps <= target or bundle[0].total_steps != horizon:
        raise ValueError('Extension state clock or recipe horizon differs')
    # Only ALZ-owned external execution metadata changes. The native recipe,
    # native policy state, models, optimizers and all random states are intact.
    q = model_module().snapshot(ctx, bundle, generators, rng, usage, seed, streams, status)
    q.update(fixed_endpoint=target, initial_completed_updates=initial, native_recipe_schedule_horizon=horizon)
    return q


def restore(ctx, q, *, initial=INITIAL, target=TARGET, horizon=HORIZON):
    check_limits(ctx, initial, target, horizon, () if ctx.device.type != 'cuda' else DIAGNOSTICS)
    import torch
    model = model_module()
    n = q.get('completed_updates')
    if type(n) is not int or not initial <= n <= target or q.get('policy', {}).get('completed_steps') != n or q.get('usage', {}).get('steps') != n:
        raise ValueError('Extension policy/usage/host clocks differ')
    if q.get('fixed_endpoint') != target or q.get('initial_completed_updates') != initial or q.get('native_recipe_schedule_horizon') != horizon:
        raise ValueError('External endpoint or unchanged schedule horizon differs')
    if q.get('status') not in ('synthetic', 'diagnostic', 'incomplete', 'completed') or (q['status'] == 'completed' and n != target):
        raise ValueError('Incomplete extension cannot stand in for its fixed endpoint')
    if q.get('particlegan_version') != '0.9.6' or q.get('penalty_options') != {'input_domain': 'continuous'}:
        raise ValueError('Exact native096 continuous-domain state required')
    if q.get('schema') != 1 or q.get('method') != 'pg' or q.get('native_recipe') != 'bispf_optimized':
        raise ValueError('Released extension schema differs')
    if q.get('released_hyperparameters') != dict(model.HYPER) or q.get('variant_config') != dict(model.CONFIG):
        raise ValueError('Extension parameters or architecture differ')
    if q.get('cache_manifest_SHA256') != sha(ctx.cache / 'manifest.json') or q.get('fold') != ctx.fold or q.get('seed') != 0:
        raise ValueError('Extension cache/fold/seed differs')
    bundle, generators, rng, usage, streams = model.build(ctx.mu, ctx.theta, ctx.device, ctx.fold, q['seed'])
    rec, policy = bundle[0], bundle[-1]
    if rec.total_steps != horizon or q['recipe'] != q['policy']['recipe'] or q['recipe'] != rec.to_dict() or q['training_streams'] != streams:
        raise ValueError('Native recipe or seed streams must remain unchanged')
    policy.load_state_dict(q['policy'])
    rec.restore_prior_regularizer(bundle[-2], q['prior_regularizer'].get('prior_regularizer'))
    for key, state in q['generators'].items():
        generators[key].set_state(state.cpu())
    rng.bit_generator.state = q['numpy_rng']
    for key in ('draws', 'selected_steps', 'moved_steps', 'unsampled_moved_steps'):
        setattr(usage, key, q['usage'][key].to(ctx.device).clone())
    usage.steps = q['usage']['steps']
    usage.unique_batch_histogram = list(q['usage']['unique_batch_histogram'])
    torch.set_rng_state(q['torch_cpu_rng'].cpu())
    if ctx.device.type == 'cuda':
        torch.cuda.set_rng_state(q['torch_device_rng'].cpu(), ctx.device)
    return bundle, generators, rng, usage, streams


def run_remaining(ctx, bundle, generators, rng, usage, seed, streams, out, *, end_monotonic,
                  initial=INITIAL, target=TARGET, horizon=HORIZON, diagnostics=DIAGNOSTICS):
    check_limits(ctx, initial, target, horizon, diagnostics)
    import torch
    model = model_module()
    if bundle[0].total_steps != horizon or not initial <= bundle[-1].completed_steps < target:
        raise ValueError('Saved native clock and schedule horizon differ')
    out = Path(out)
    stopped = []
    handlers = {sig: signal.signal(sig, lambda signum, frame: stopped.append(signum)) for sig in (signal.SIGINT, signal.SIGTERM)}
    def save(status):
        path = out / ('final.pt' if status == 'completed' else 'recovery.pt')
        temp = path.with_suffix('.partial.pt')
        torch.save(snapshot(ctx, bundle, generators, rng, usage, seed, streams, status,
                            initial=initial, target=target, horizon=horizon), temp)
        temp.replace(path)
        model.dump(out / 'checkpoint_receipt.json', {'path': path.name, 'SHA256': sha(path), 'status': status,
                                                    'completed_updates': bundle[-1].completed_steps})
        return path
    try:
        while bundle[-1].completed_steps < target:
            if stopped or time.monotonic() >= end_monotonic:
                raise InterruptedError('Finite extension allocation ending; preserve original native state')
            metrics = model.advance(ctx, bundle, generators, rng, usage, execution_limit=target)
            if metrics['step'] % 250 in (0, 1):
                metrics['particle_usage'] = model.passive_particle_diagnostics(bundle, usage)
                model.dump(out / 'progress.json', metrics)
                with (out / 'learning_curve.jsonl').open('a') as stream:
                    stream.write(json.dumps(metrics, allow_nan=False) + '\n')
            if metrics['step'] in diagnostics:
                path = out / ('checkpoint%05d.pt' % metrics['step'])
                if path.exists():
                    raise FileExistsError('Do not replace a declared extension diagnostic')
                torch.save(snapshot(ctx, bundle, generators, rng, usage, seed, streams, 'diagnostic',
                                    initial=initial, target=target, horizon=horizon), path)
            if metrics['step'] % 1000 == 0:
                save('recovery')
        return bundle, save('completed')
    except BaseException as error:
        if bundle[-1]._phase == 'ready':
            save('incomplete')
        model.dump(out / 'failure.json', {'error': repr(error), 'completed_updates': bundle[-1].completed_steps,
                                         'phase': bundle[-1]._phase, 'primary_endpoint_complete': False})
        raise
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def fit(ctx, out, seed, auth, *, end_monotonic):
    if ctx.device.type != 'cuda':
        raise RuntimeError('Real extension fitting requires separately guarded GPU admission; CPU fixtures are synthetic only')
    import torch
    from state_fingerprint import fingerprint
    model = model_module()
    parent = auth['extension']
    path = Path(parent['checkpoint_path'])
    if seed != 0 or sha(path) != parent['checkpoint_SHA256']:
        raise ValueError('Immutable parent checkpoint or seed differs')
    q = torch.load(path, map_location='cpu', weights_only=False)
    validate_parent(q, auth)
    bundle, generators, rng, usage, streams = model.restore(ctx, q)
    whole = fingerprint(q)
    if fingerprint(bundle[-1].state_dict()) != fingerprint(q['policy']):
        raise ValueError('Full public native CUDA policy restore differs')
    if fingerprint(model.snapshot(ctx, bundle, generators, rng, usage, seed, streams, q['status'])) != whole or whole != parent['whole_parent_scientific_fingerprint']:
        raise ValueError('Full original models/optimizers/usage/eleven streams were not restored byte-exact')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(path, out / 'checkpoint20000.pt')
    if sha(out / 'checkpoint20000.pt') != parent['checkpoint_SHA256']:
        raise ValueError('Copied original20k checkpoint differs')
    history = parent['training_log']
    if sha(history['immutable_copy_path']) != history['SHA256']:
        raise ValueError('Original20k training log differs')
    shutil.copyfile(history['immutable_copy_path'], out / 'learning_curve.jsonl')
    model.dump(out / 'native_parent_restore.json', {
        'status': 'EXACT_COMPLETED20K_PARENT_ALL_NATIVE_AND_SCIENTIFIC_STATE_RESTORED_BEFORE_EXTENSION',
        'parent_checkpoint_SHA256': parent['checkpoint_SHA256'], 'whole_parent_scientific_fingerprint': whole,
        'native_policy_fingerprint': fingerprint(q['policy']), 'initial_completed_updates': INITIAL,
        'fixed_endpoint': TARGET, 'new_model_updates': TARGET - INITIAL, 'native_recipe_schedule_horizon': HORIZON,
        'public_execution_limit': TARGET, 'native_recipe_rewritten': False, 'native_device_metadata_relabelled': False,
        'native_or_usage_or_data_clock_reset': False, 'original_log_prefix_byte_exact': True,
    })
    del q
    return run_remaining(ctx, bundle, generators, rng, usage, seed, streams, out, end_monotonic=end_monotonic)
