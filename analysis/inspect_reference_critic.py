"""CPU fitting-only critic response probe; no fitting or model modification."""
import json
import os
from pathlib import Path
import sys

ROOT = Path('/ml2/alz')
B = Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark')
SCIENCE = Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/core-comparison-20261103')


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    sys.path.extend([str(B / 'current_base_20261029/runtime/lib/python3.12/site-packages'),
                     str(B / '.venv312/lib/python3.12/site-packages')])
    sys.path.insert(0, str(SCIENCE))
    import numpy as np
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from pg_adapter import Context, restore
    from pg_model import critic_features
    from critic import bag_features
    from contracts import sha
    checkpoint = SCIENCE / 'runs_gpu0_v1/GPU0_V1_f0_pg_s0/fit/final.pt'
    assert sha(checkpoint) == 'c6fdab29e87dca26b5742208999e7552392ff25bab07056214da747cfb9e4fea'
    ctx = Context(SCIENCE / 'caches/fold0', 'cpu')
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    bundle, _, _, _, _ = restore(ctx, state)
    G, D, P = bundle[1], bundle[2], bundle[3]
    G.eval(); D.eval(); D.requires_grad_(False)
    rng = np.random.default_rng(20261107)
    cell_rng = torch.Generator().manual_seed(20261108)
    count_rng = torch.Generator().manual_seed(20261109)
    latent_rng = torch.Generator().manual_seed(20261110)
    torch.manual_seed(20261111)
    donor_stage = np.rint(ctx.fitc[:, 3] * 3).astype(int)
    rows = []
    for stage in range(4):
        group = np.flatnonzero(donor_stage == stage)
        selected = rng.choice(group, size=8, replace=True)
        index = torch.as_tensor(np.stack([rng.choice(ctx.by[d], 128, replace=True) for d in selected]))
        real = bag_features(ctx.trainpcs[index]).detach().requires_grad_(True)
        c = ctx.conditions[torch.as_tensor(selected)]
        libraries = ctx.X[index].sum(-1)
        z, _ = P.sample(8, generator=latent_rng)
        rates = G.rates(z, c, 128, libraries, cell_rng)
        counts = torch.poisson(rates.detach(), generator=count_rng)
        fake = bag_features(critic_features(counts, rates, ctx.mean, ctx.sd) @ ctx.W)
        dr, df = D(real, c), D(fake, c)
        grad_real = torch.autograd.grad(dr.sum(), real)[0]
        grad_fake = torch.autograd.grad(df.sum(), fake, retain_graph=True)[0]
        loss = torch.nn.functional.softplus(-df).mean()
        G.zero_grad(); P.zero_grad(); loss.backward()
        def norm(module):
            return float(sum(p.grad.detach().square().sum() for p in module.parameters() if p.grad is not None).sqrt())
        row = {'stage': stage, 'optimization_donors_at_stage': len(group),
               'probe_bags': 8, 'cells_per_bag': 128,
               'real_logit_mean': float(dr.detach().mean()),
               'fake_logit_mean': float(df.detach().mean()),
               'critic_gap': float((dr - df).detach().mean()),
               'NS_generator_loss': float(loss.detach()),
               'raw_adversarial_G_gradient_norm': norm(G),
               'raw_adversarial_prior_gradient_norm': norm(P),
               'real_input_gradient_norm_mean': float(grad_real.norm(dim=1).mean()),
               'fake_input_gradient_norm_mean': float(grad_fake.norm(dim=1).mean())}
        for name, start in [('means', 0), ('SDs', 50), ('Fourier', 100)]:
            row[name + '_fake_input_gradient_norm'] = float(grad_fake[:, start:start+50].norm(dim=1).mean())
            row[name + '_mean_feature_error_L2'] = float((fake[:, start:start+50].detach().mean(0) - real[:, start:start+50].detach().mean(0)).norm())
        assert all(np.isfinite(v) for v in row.values())
        rows.append(row)
    assert not torch.cuda.is_initialized()
    result = {'status': 'FITTING_ONLY_REFERENCE_CRITIC_PROBE', 'records': rows,
              'checkpoint_SHA256': sha(checkpoint),
              'scope': 'Fixed eight generated/real bags per stage on optimization donors only. Raw gradients before native optimizer geometry. Descriptive probe, no causal attribution.',
              'GPU_used': False, 'model_updates': 0, 'calibration_or_evaluation_targets_used': False}
    (ROOT / 'analysis/reference_critic_probe.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
