"""Full 2000-gene synthetic native Halloween update and integer-count serving."""
import datetime
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import halloween_pg_adapter as a
from generation import iter_pg
from contracts import sha

ROOT = Path('/ml2/alz')


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    torch.set_num_threads(1)
    x = torch.randint(1, 8, (24, 2000), generator=torch.Generator().manual_seed(20261104111)).float()
    ctx = object.__new__(a.Context)
    ctx.device, ctx.fold = torch.device('cpu'), 0
    ctx.ids = np.arange(6)
    ctx.by = [np.arange(4 * i, 4 * i + 4) for i in ctx.ids]
    ctx.X, ctx.conditions = x, torch.zeros(6, 7)
    ctx.conditions[:, 3] = torch.tensor([0., 1/3, 2/3, 1., 1., 1.])
    ctx.fitc = ctx.conditions.numpy().copy()
    ctx.mean, ctx.sd = torch.zeros(2000), torch.ones(2000)
    ctx.W = torch.randn(2000, 50, generator=torch.Generator().manual_seed(20261104112)) / np.sqrt(2000)
    ctx.trainpcs, ctx.mu = ctx.project(x), x.mean(0)
    ctx.theta = (ctx.mu.square() / (x.var(0, unbiased=False) - ctx.mu).clamp_min(1e-4)).clamp(.1, 10000)
    b, g, r, usage, _ = a.build(ctx.mu, ctx.theta, 'cpu', 0, 0)
    before = [m.detach().clone() for m in [b[1].out[-1].weight, b[1].log_theta, b[3].z]]
    metrics = a.advance(ctx, b, g, r, usage, execution_limit=20000)
    diagnostics = [
        {'parameter': name, 'changed_values': int((old != p.detach()).sum()),
         'gradient_L2': float(p.grad.norm()) if p.grad is not None else None,
         'movement_L2': float((old - p.detach()).norm())}
        for name, old, p in zip(['G_output_weight', 'G_log_theta', 'prior_z'], before,
                               [b[1].out[-1].weight, b[1].log_theta, b[3].z])
    ]
    for key in ('G_gradient_norm', 'prior_gradient_norm', 'D_gradient_norm'):
        assert metrics[key] > 0 and np.isfinite(metrics[key])
    for old, p in zip(before, [b[1].out[-1].weight, b[1].log_theta, b[3].z]):
        assert p.grad is not None
        assert bool(torch.isfinite(p.grad).all()) and bool(p.grad.abs().sum() > 0)
        group = next(group for group in b[4].param_groups if any(p is owned for owned in group['params']))
        state = b[4].state[p]
        rate = group['lr'] * math.sqrt(1 - group['_tf_beta2_power']) / (1 - group['_tf_beta1_power'])
        # Reconstruct the actual native dense application, including float32
        # rounding. The native generator epsilon is deliberately substantial;
        # a nonzero gradient need not move every parameter by one representable ULP.
        expected = old.clone().addcdiv_(state['exp_avg'], state['exp_avg_sq'].sqrt().add_(group['eps']), value=-rate)
        assert torch.equal(expected, p.detach())
    assert diagnostics[0]['changed_values'] > 0 and diagnostics[2]['changed_values'] > 0
    assert metrics['D_penalty'] == 0 and metrics['G_adversarial'] == metrics['G_total']
    assert int(usage.draws.sum()) == 32 and int(usage.unsampled_moved_steps.sum()) == 0
    floats = 0
    for module in b[1:4]:
        for v in module.state_dict().values():
            if v.is_floating_point():
                assert bool(torch.isfinite(v).all())
                floats += v.numel()
    assert floats == 371105
    for opt in b[4:6]:
        assert opt.state_dict()['tensorflow_v1'] == {'schema_version': 1, 'gradient_law': 'dense_real'}
        for group in opt.param_groups:
            assert group['_tf_step'] == 1
            assert group['_tf_beta1_power'] == group['betas'][0]
            assert group['_tf_beta2_power'] == group['betas'][1]
    c = np.zeros(7, dtype=np.float32)
    c[3] = 2/3
    prior_mode, generator_mode = b[3].training, b[1].training
    iterator = iter_pg(ctx, b[1], b[3], SimpleNamespace(libraries=np.full((200, 400), 8000, dtype=np.float32)), c, 20261104113)
    try:
        start, pcs, counts, totals = next(iterator)
        assert start == 0 and pcs.shape == (5, 400, 50) and counts.shape == (1, 400, 2000)
        assert np.isfinite(pcs).all() and np.isfinite(counts).all()
        assert (counts >= 0).all() and np.equal(counts, np.floor(counts)).all()
        assert totals.shape == (2000,) and np.isfinite(totals).all()
        assert (totals >= 0).all() and np.equal(totals, np.floor(totals)).all()
        np.testing.assert_array_equal(totals[:400], counts.sum(-1).reshape(-1))
    finally:
        iterator.close()
    assert b[3].training == prior_mode and b[1].training == generator_mode
    assert not torch.cuda.is_initialized()
    result = {
        'status': 'PASSED_FULL2000GENE_NATIVE_HALLOWEEN_UPDATE_AND_UNCHANGED_COUNT_GENERATION_CPU_CHECK',
        'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'full_model_floating_values': floats,
        'native_G_D_dispersion_and_P_nonzero_gradients_verified': True,
        'native_generator_output_and_prior_parameter_movement_verified': True,
        'first_update_diagnostic': diagnostics,
        'exact_TensorFlowV1Adam_float32_application_including_dispersion_rounding_verified': True,
        'native_TensorFlowV1Adam_all_role_clocks_powers_and_marker_verified': True,
        'native_zero_penalty_and_prior_regularization_verified': True,
        'unsampled_raw_particle_movement_zero': True,
        'unchanged_iter_pg_first_five_bags_PC_integer_counts_and_mode_restoration_passed': True,
        'CUDA_initialized': False, 'quality_measured': False,
        'source_SHA256': {str(p): sha(p) for p in [
            Path(__file__).resolve(), *(ROOT / 'halloween_model').glob('*.py'),
            Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/core-comparison-20261103/generation.py'),
        ]},
        'scope': 'One synthetic fullwidth native optimizer pair and first five bags from unchanged iter_pg. G output and prior move; log_theta has a finite nonzero gradient but its first synthetic step rounds to zero in float32 under the unmodified native generator epsilon. Exact native application is independently recomputed, including that rounding. This does not establish real-data dispersion immobility or quality. No real data, targets, calibration score, GPU context or allocation. This is qualification, not a completed native Halloween comparison.',
    }
    out = ROOT / 'analysis/halloween_native_full_width_CPU_qualification.json'
    assert not out.exists()
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
