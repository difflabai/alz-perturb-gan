"""Read-only CPU audit of one actually published stage-prior recovery snapshot.

Preserve the native snapshot and CUDA metadata. No updates, generation, target
access, ranking, allocation, or edits to the running trial's frozen sources.
"""
import datetime
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path('/ml2/alz')
sys.path.insert(0, str(ROOT / 'stage_runtime'))
import binding as b
import preset_scope as scope


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    attempt = 'CAL_GPU0_V1_stage_prior_v1_f0_pg_s0'
    allocation = ROOT / 'artifacts/allocations' / attempt
    auth = b.read(allocation / 'authorization.json')
    scope.validate_authorization(auth)
    fit = ROOT / 'runs' / attempt / 'fit'
    for _ in range(5):
        receipt = b.read(fit / 'checkpoint_receipt.json')
        payload = (fit / receipt['path']).read_bytes()
        pin = hashlib.sha256(payload).hexdigest()
        if receipt == b.read(fit / 'checkpoint_receipt.json') and pin == receipt['SHA256']:
            break
        time.sleep(.25)
    else:
        raise RuntimeError('Publication changed; reobserve the native receipt')
    step = receipt['completed_updates']
    assert 0 < step < 20000 and step % 1000 == 0
    assert receipt['status'] == 'recovery'
    folder = ROOT / 'artifacts/checkpoint_milestones' / attempt / (str(step) + '-' + pin)
    folder.mkdir(parents=True, exist_ok=False)
    preserved = folder / 'recovery.pt'
    preserved.write_bytes(payload)
    assert b.sha(preserved) == pin
    sys.path.extend(map(str, b.native_site_roots(scope.CELL, auth['files_SHA256'])))
    sys.path[:0] = [str(ROOT / 'stage_model'), str(ROOT / 'model'), str(b.SCIENCE), str(b.PROJECT)]
    import torch
    from particlegan.policy import _validate_optimizer_state
    from stage_pg_adapter import build, training_streams

    assert Path(sys.modules[build.__module__].__file__).resolve() == ROOT / 'stage_model/stage_pg_adapter.py'
    torch.set_num_threads(1)
    q = torch.load(preserved, map_location='cpu', weights_only=False)
    assert q['schema'] == 1 and q['method'] == 'pg'
    assert q['status'] == 'recovery' and q['completed_updates'] == q['policy']['completed_steps'] == step
    assert q['fixed_endpoint'] == 20000 and q['native_recipe'] == 'bispf_optimized'
    assert q['stage_prior_config'] == auth['stage_prior_config'] == scope.STAGE
    assert q['variant_config'] == auth['variant_config'] == scope.CFG
    assert set(q['generators']) == {'cell', 'count'}
    assert q['cache_manifest_SHA256'] == b.sha(Path(auth['cache_dir']) / 'manifest.json')
    assert q['training_streams'] == training_streams(0, 0) and q['fold'] == q['seed'] == 0
    bundle, *_ = build(torch.ones(2000), torch.ones(2000), 'cpu', 0, 0)
    assert q['recipe'] == q['policy']['recipe'] == bundle[0].to_dict()
    assert bundle[3].z.requires_grad and len(list(bundle[3].parameters())) == 1
    assert len(bundle[4].param_groups) == len(q['policy']['optimizers'][0]['param_groups']) == 2
    initial = bundle[3].z.detach().clone()
    trained = q['policy']['models']['prior']['z']
    assert not torch.equal(trained, initial)
    group_movement = (trained - initial).square().sum(1).reshape(4, 128).sum(1).sqrt()
    assert bool((group_movement > 0).all())
    assert q['prior_regularizer'] == {}
    usage = q['usage']
    assert usage['steps'] == step and int(usage['draws'].sum()) == 32 * step
    assert int(usage['moved_steps'].sum()) > 0
    assert int(usage['unsampled_moved_steps'].sum()) == 0
    group_draws = usage['draws'].reshape(4, 128).sum(1)
    assert bool((group_draws > 0).all())
    assert len(usage['unique_batch_histogram']) == 33 and sum(usage['unique_batch_histogram']) == step
    models = {'generator': bundle[1], 'critic': bundle[2], 'prior': bundle[3]}
    assert q['policy']['models'].keys() == models.keys()
    floats = 0
    for name, module in models.items():
        expected, saved = module.state_dict(), q['policy']['models'][name]
        assert saved.keys() == expected.keys()
        for key, value in saved.items():
            assert isinstance(value, torch.Tensor)
            assert value.shape == expected[key].shape and value.dtype == expected[key].dtype
            if value.is_floating_point():
                assert bool(torch.isfinite(value).all())
                floats += value.numel()
        module.load_state_dict(saved, strict=True)
    assert floats == 371177

    def finite_tree(value):
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            assert bool(torch.isfinite(value).all())
        elif isinstance(value, dict):
            for child in value.values():
                finite_tree(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                finite_tree(child)

    finite_tree(q)
    expected_averages = bundle[-1].state_dict()['averages']
    assert q['policy']['averages'].keys() == expected_averages.keys()
    for name, expected in expected_averages.items():
        saved = q['policy']['averages'][name]
        assert saved.keys() == expected.keys()
        for key, value in saved.items():
            assert value.shape == expected[key].shape and value.dtype == expected[key].dtype
    assert len(q['policy']['optimizers']) == 2
    for optimizer, saved in zip(bundle[4:6], q['policy']['optimizers']):
        _validate_optimizer_state(optimizer, saved)
        optimizer.load_state_dict(saved)
    assert not torch.cuda.is_initialized()
    result = {
        'status': 'ACTUAL_INTERMEDIATE_STAGE_GPU_CHECKPOINT_CPU_PORTABILITY_AUDIT_PASSED',
        'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'attempt': attempt,
        'completed_updates': step, 'fixed_endpoint': 20000,
        'native_recovery_checkpoint_SHA256': pin, 'preserved_checkpoint_path': str(preserved),
        'full_model_floating_values': floats,
        'native_optimizer_and_average_states_verified': True,
        'stage_prior_config': q['stage_prior_config'],
        'learned_particle_bank_differs_from_exact_seeded_initial_bank': True,
        'particle_group_L2_displacement_from_initial': group_movement.tolist(),
        'total_generator_particle_draws': int(usage['draws'].sum()),
        'generator_particle_draws_per_stage_group': group_draws.tolist(),
        'all_four_stage_groups_sampled_and_moved': True,
        'unsampled_raw_particle_movement_zero': True,
        'all_saved_floating_tensors_finite_including_usage_and_regularizer': True,
        'GPU_policy_device_metadata_preserved': q['policy']['device'],
        'source_SHA256': {str(Path(__file__).resolve()): b.sha(__file__)},
        'authorization_SHA256': b.sha(allocation / 'authorization.json'),
        'CUDA_initialized': False, 'training_update_executed': False,
        'calibration_forecasts_generated': False, 'quality_measured': False,
        'scope': 'Actual stable intermediate snapshot copied by matching native receipt/hash before and after read. Full CPU models/optimizers/EMA validate; saved CUDA policy and RNG metadata remain unchanged. No GPU next-update replay, quality selection, completed endpoint, confirmation or new admission.',
    }
    output = ROOT / 'analysis' / (attempt + f'-checkpoint-{step}-portability.json')
    assert not output.exists()
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
