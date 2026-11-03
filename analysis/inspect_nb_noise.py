"""Read-only fitting-data NB variance decomposition; never resumes a CUDA policy."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path('/ml2/alz')
B = Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark')
SCIENCE = Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/core-comparison-20261103')


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'runs/CAL_GPU0_V1_latent8_v1_f0_pg_s0/fit/final.pt')
    parser.add_argument('--expected-sha', default='444c22a5829b79d8ed0cd300be255d9410916d6c2248f930cf4cc0f886744da2')
    parser.add_argument('--output', type=Path, default=ROOT / 'analysis/nb_noise_diagnostic.json')
    args = parser.parse_args()
    sys.path.extend([str(B / 'current_base_20261029/runtime/lib/python3.12/site-packages'), str(B / '.venv312/lib/python3.12/site-packages')])
    sys.path[:0] = [str(ROOT / 'model'), str(SCIENCE), str(B)]
    import numpy as np
    import torch
    torch.set_num_threads(1)
    import variant_pg_update
    from variant_pg_adapter import Context, build
    from contracts import sha
    checkpoint = args.checkpoint
    assert sha(checkpoint) == args.expected_sha
    saved_checkpoint = torch.load(checkpoint, map_location='cpu', weights_only=False)
    config = variant_pg_update.normalized_config(saved_checkpoint.get('variant_config', {'z_dim': saved_checkpoint['recipe']['z_dim']}))
    variant_pg_update.configure(config)
    ctx = Context(SCIENCE / 'caches/fold0', 'cpu')
    saved = saved_checkpoint['policy']
    bundle, _, _, _, _ = build(ctx.mu, ctx.theta, 'cpu', 0, saved_checkpoint['seed'])
    for name, module in bundle[-1]._training_modules().items():
        module.load_state_dict(saved['models'][name], strict=True)
    with torch.no_grad():
        bundle[3].z.copy_(saved['table'])
    G, P = bundle[1].eval(), bundle[3].eval()
    rng = np.random.default_rng(20261113)
    latent = torch.Generator().manual_seed(20261114)
    cell = torch.Generator().manual_seed(20261115)
    with torch.no_grad():
        _, condition, library = ctx.batch(rng)
        z, _ = P.sample(32, generator=latent)
        mu, theta = G.parameters_nb(z, condition, 128, library, cell)
        mu = mu.reshape(-1, 2000)
        decoder_variance = mu.var(0, unbiased=False)
        poisson = mu.mean(0)
        gamma = (mu.square() / theta).mean(0)
        total = decoder_variance + poisson + gamma
        independent_fraction = (poisson + gamma) / total.clamp_min(1e-12)
        gamma_fraction = gamma / total.clamp_min(1e-12)
    def quantiles(x):
        return np.percentile(x.detach().numpy(), [0, 10, 50, 90, 100]).tolist()
    report = {
        'status': 'FITTING_ONLY_NB_VARIANCE_DIAGNOSTIC',
        'checkpoint_SHA256': sha(checkpoint), 'probe_cells': 4096,
        'checkpoint_status': saved_checkpoint['status'],
        'completed_updates': saved_checkpoint['completed_updates'],
        'variant_config': config,
        'theta_initial_quantiles': quantiles(ctx.theta * config['theta_init_scale']),
        'theta_learned_quantiles': quantiles(theta),
        'theta_learned_over_initial_quantiles': quantiles(theta / (ctx.theta * config['theta_init_scale'])),
        'independent_count_noise_fraction_per_gene_quantiles': quantiles(independent_fraction),
        'Gamma_overdispersion_fraction_per_gene_quantiles': quantiles(gamma_fraction),
        'aggregate_independent_variance_fraction': float((poisson + gamma).sum() / total.sum()),
        'aggregate_gamma_variance_fraction': float(gamma.sum() / total.sum()),
        'decoder_variance_sum': float(decoder_variance.sum()),
        'Poisson_variance_sum': float(poisson.sum()), 'Gamma_variance_sum': float(gamma.sum()),
        'scope': 'Count-space law-of-total-variance estimate under the fitted decoder and fitting-only library/condition templates. Log/PCA variance is nonlinear; this is not a direct decomposition of the scoring metric or a causal diagnosis.',
        'CPU_only': True, 'new_model_updates': 0, 'calibration_or_evaluation_targets_used': False,
    }
    assert not torch.cuda.is_initialized()
    assert sha(checkpoint) == args.expected_sha
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
