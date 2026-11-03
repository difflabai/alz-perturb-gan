"""Same-device complete-state qualification before any longer ALZ continuation."""
import copy
from pathlib import Path

import numpy as np
import torch

import variant_pg_adapter as variant
import variant_pg_update as update


def same(a, b, path='checkpoint'):
    if isinstance(a, torch.Tensor):
        if not isinstance(b, torch.Tensor) or a.dtype != b.dtype or not torch.equal(a.cpu(), b.cpu()):
            raise AssertionError('Complete-state tensor mismatch: ' + path)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b, err_msg=path)
    elif isinstance(a, dict):
        if not isinstance(b, dict) or a.keys() != b.keys():
            raise AssertionError('Complete-state keys differ: ' + path)
        for key in a:
            same(a[key], b[key], path + '.' + str(key))
    elif isinstance(a, (tuple, list)):
        if type(a) is not type(b) or len(a) != len(b):
            raise AssertionError('Complete-state sequence differs: ' + path)
        for index, (x, y) in enumerate(zip(a, b)):
            same(x, y, path + '.' + str(index))
    elif a != b:
        raise AssertionError('Complete-state value differs: ' + path)


def qualify(ctx, checkpoint, out):
    assert ctx.device.type == 'cuda' and checkpoint['completed_updates'] == 20000
    bundle, generators, rng, usage, streams = variant.restore(ctx, checkpoint)
    expected_initial = copy.deepcopy(checkpoint)
    expected_initial['variant_config'] = update.normalized_config(checkpoint.get('variant_config', {'z_dim': checkpoint['recipe']['z_dim']}))
    observed_initial = variant.snapshot(ctx, bundle, generators, rng, usage, checkpoint['seed'], streams, checkpoint['status'])
    same(expected_initial, observed_initial)
    expected_metrics = variant.advance(ctx, bundle, generators, rng, usage, execution_limit=40000)
    expected = variant.snapshot(ctx, bundle, generators, rng, usage, checkpoint['seed'], streams, 'qualification_only')
    rb, rg, rr, ru, rs = variant.restore(ctx, checkpoint)
    observed_metrics = variant.advance(ctx, rb, rg, rr, ru, execution_limit=40000)
    observed = variant.snapshot(ctx, rb, rg, rr, ru, checkpoint['seed'], rs, 'qualification_only')
    same(expected_metrics, observed_metrics)
    same(expected, observed)
    assert observed['completed_updates'] == 20001 and observed['recipe']['total_steps'] == 20000
    path = Path(out) / 'qualification_step20001.pt'
    torch.save(observed, path)
    return {'status': 'SAME_GPU_COMPLETE_STATE_AND_NEXT_UPDATE_BITWISE_VERIFIED',
            'qualification_only': True, 'parent_completed_updates': 20000,
            'qualified_completed_updates': 20001, 'executed_optimizer_pairs': 2,
            'replay_branches': 2, 'updates_per_branch': 1, 'native_recipe_schedule_horizon': 20000,
            'caller_execution_limit': 40000, 'complete_initial_state_restored_exactly': True,
            'complete_next_state_and_metrics_bitwise_equal': True,
            'qualification_checkpoint_path': str(path),
            'calibration_or_evaluation_scores_generated': False, 'automatic_successor': False}
