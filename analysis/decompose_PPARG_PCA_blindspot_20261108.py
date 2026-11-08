"""Exact visible/null-space decomposition of the frozen PPARG mean error.

The critic only receives features of the fixed 50-PC cell representation.
Differences in its orthogonal complement cannot reach any downstream critic.
This is an algebraic diagnosis on actual generated counts, not an intervention.
"""
import datetime
import importlib.util
import json
import os
from pathlib import Path
import signal
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/PPARG-PCA-blindspot-CPU-20261108'
RESULT = ROOT / 'analysis/PPARG_PCA_blindspot_CPU_20261108.json'


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    spec = importlib.util.spec_from_file_location('probe_utils', ROOT / 'analysis/probe_PPARG_critic_gradient_20261108.py')
    util = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(util)
    spec = importlib.util.spec_from_file_location('strict_probe_loader', ROOT / 'analysis/diagnose_t005_calibration_checkpoints_multiseed_cpu.py')
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    m = loader.m
    torch, np = m.torch, m.np
    from pg_model import standardize_counts
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    assert not torch.cuda.is_initialized() and not OUT.exists() and not RESULT.exists()
    OUT.mkdir()
    firstpath = ROOT / 'analysis/PPARG_critic_gradient_CPU_20261108.json'
    first = json.loads(firstpath.read_bytes())
    plan = json.loads((ROOT / 'analysis/primamente_inference_hypothesis_plan_20261108.json').read_bytes())
    pins = dict(first['protocol']['input_SHA256'])
    pins.update({str(Path(__file__).resolve()): util.sha(__file__), str(firstpath): util.sha(firstpath)})
    for p, h in pins.items():
        assert util.sha(p) == h, p
    protocol = dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    input_SHA256=pins, seeds=list(range(5)), replicates=4,
                    exact_replay='Same CPU base draws as the critic-gradient probe, verified by all stage mu/log means/detection.',
                    algebra='delta = delta_visible + delta_null; delta_visible = delta @ U @ U.T; U spans the fixed PC columns.',
                    units='Mean log1p(panel CP10k); no new model fitting.', new_optimizer_updates=0)
    util.write(OUT / 'protocol.json', protocol)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('CPU probe cap')))
    signal.alarm(1200)
    started = time.monotonic()
    ctx = m.Context(Path(plan['fitting_cache']), 'cpu')
    meta = np.load(ctx.cache / 'train/metadata.npz', allow_pickle=False)
    j = list(meta['genes']).index('PPARG')
    rawW = torch.tensor(ctx.space.W, dtype=torch.float64)
    # QR removes only ~1e-6 stored-basis roundoff and leaves its column space intact.
    U = torch.linalg.qr(rawW, mode='reduced').Q
    assert float((U.T @ U - torch.eye(50)).abs().max()) < 1e-12
    assert float((rawW - U @ (U.T @ rawW)).abs().max()) < 1e-12
    mean, sd = ctx.mean.double(), ctx.sd.double()
    # Training-cell variance accounting, descriptive rather than held-out R2.
    real_gene = []
    real_reconstructed = []
    for start in range(0, len(ctx.X), 256):
        std = standardize_counts(ctx.X[start:start + 256].double(), mean, sd)
        real_gene.append(std[:, j])
        real_reconstructed.append((std @ U) @ U[j])
    real_gene = torch.cat(real_gene)
    real_reconstructed = torch.cat(real_reconstructed)
    variance_ratio = float((real_gene - real_reconstructed).var(unbiased=False) / real_gene.var(unbiased=False))
    records = []
    max_null_projection = 0.
    for model in plan['models']:
        seed = model['seed']
        q = torch.load(model['checkpoint'], map_location='cpu', weights_only=False)
        G, P = loader.generation_modules(ctx, q, seed, 20000, plan)
        before = m.fingerprint(G), m.fingerprint(P)
        for rep in range(4):
            rows = []
            for stage in range(4):
                c0 = np.zeros(7, dtype=np.float32)
                c0[3] = stage / 3
                rows.append(ctx.law.schedule(c0, 2026110900 + rep * 4 + stage, m=8, k=128).rows)
            real = ctx.X[torch.as_tensor(np.concatenate(rows))]
            c = torch.zeros(32, 7)
            c[:, 3] = torch.arange(4).repeat_interleave(8) / 3
            random_seed = 202611080000 + seed * 100 + rep
            torch.manual_seed(random_seed)
            z, _ = P.sample(32, generator=torch.Generator().manual_seed(random_seed + 1))
            mu, theta = G.parameters_nb(z, c, 128, real.sum(-1), torch.Generator().manual_seed(random_seed + 2))
            rate = torch.distributions.Gamma(theta, theta / mu).rsample()
            counts = torch.poisson(rate, generator=torch.Generator().manual_seed(random_seed + 3))
            real_std = standardize_counts(real.double(), mean, sd)
            model_std = standardize_counts(counts.double(), mean, sd)
            target = next(r for r in first['records'] if r['seed'] == seed and r['replicate'] == rep)
            stages = []
            for stage in range(4):
                sl = slice(stage * 8, (stage + 1) * 8)
                ref = target['by_stage'][stage]
                real_means = real_std[sl].mean((0, 1))
                model_means = model_std[sl].mean((0, 1))
                real_log = float(real_means[j] * sd[j] + mean[j])
                model_log = float(model_means[j] * sd[j] + mean[j])
                assert abs(real_log - ref['observed_logmean']) < 1e-12
                assert abs(model_log - ref['model_logmean']) < 1e-12
                assert float(mu[sl, :, j].mean()) == ref['predicted_mu']
                assert float((counts[sl, :, j] > 0).double().mean()) == ref['model_detection']
                delta = real_means - model_means
                visible = (delta @ U) @ U.T
                invisible = delta - visible
                null_error = float((invisible @ ctx.W.double()).abs().max())
                max_null_projection = max(max_null_projection, null_error)
                assert null_error < 1e-6
                gap = float(delta[j] * sd[j])
                visible_gap = float(visible[j] * sd[j])
                hidden_gap = float(invisible[j] * sd[j])
                assert abs(gap - visible_gap - hidden_gap) < 1e-12
                stages.append(dict(stage=stage, observed_logmean=real_log, generated_logmean=model_log,
                    actual_mean_gap=gap, PC_visible_component_of_mean_gap=visible_gap,
                    PC_invisible_component_of_mean_gap=hidden_gap,
                    real_PC_reconstructed_logmean=float(((real_means @ U) @ U[j]) * sd[j] + mean[j]),
                    generated_PC_reconstructed_logmean=float(((model_means @ U) @ U[j]) * sd[j] + mean[j]),
                    hidden_difference_projection_max_error=null_error))
            record = dict(seed=seed, replicate=rep, by_stage=stages)
            records.append(record)
            util.write(OUT / f'seed{seed}-rep{rep}.json', record)
        assert before == (m.fingerprint(G), m.fingerprint(P))
        print(json.dumps(dict(seed=seed, elapsed_seconds=time.monotonic() - started)), flush=True)
    fields = ['observed_logmean', 'generated_logmean', 'actual_mean_gap', 'PC_visible_component_of_mean_gap',
              'PC_invisible_component_of_mean_gap', 'real_PC_reconstructed_logmean', 'generated_PC_reconstructed_logmean']
    summary = []
    for stage in range(4):
        values = [r['by_stage'][stage] for r in records]
        means = {f: float(np.mean([v[f] for v in values])) for f in fields}
        summary.append(dict(stage=stage, **means,
            invisible_share_of_signed_mean_gap=means['PC_invisible_component_of_mean_gap'] / means['actual_mean_gap'],
            seed_means={str(s): {f: float(np.mean([r['by_stage'][stage][f] for r in records if r['seed'] == s])) for f in fields} for s in range(5)}))
    for p, h in pins.items():
        assert util.sha(p) == h, p
    assert not torch.cuda.is_initialized()
    result = dict(status='COMPLETE_EXACT_PPARG_PCA_VISIBLE_VS_INVISIBLE_ERROR_DECOMPOSITION',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), protocol=protocol, summary=summary, records=records,
        original_fitting_cell_PPARG_variance_outside_PC_column_space_fraction=variance_ratio,
        max_invisible_error_projection_magnitude=max_null_projection,
        replay_of_all_twenty_original_probe_cases_verified=True, new_optimizer_updates=0, new_training_fits=0,
        new_GPU_allocations=0, CUDA_initialized=False, generator_prior_checkpoint_states_unchanged=True,
        interpretation='The null-space component of actual standardized-log expression differences is absent from every cell PC, hence absent from downstream PC bag summaries and critic inputs.',
        limitations=['This splits a vector of mean differences, not the total energy-distance or a fraction of causal training responsibility.',
                     'Signed invisible shares can exceed100% when visible and invisible components oppose one another.',
                     'PCA reconstructions are linear proxies and can take values outside nonnegative expression.',
                     'No matched retraining ablation has established which representation or count-gradient change would fix the issue.'],
        elapsed_seconds=time.monotonic() - started)
    util.write(RESULT, result)
    signal.alarm(0)
    print(json.dumps(dict(result=str(RESULT), summary=summary,
        original_fitting_cell_PPARG_variance_outside_PC_column_space_fraction=variance_ratio)), flush=True)


if __name__ == '__main__':
    main()
