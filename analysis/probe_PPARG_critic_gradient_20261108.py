"""Frozen CPU probes of PPARG representation, critic, and count gradient.

No optimizer is constructed and no model parameter is changed. The weak
Poisson derivative is for an independent log-mean coordinate with theta and
other gene rates fixed, not the full softmax/network parameter gradient.
"""
import copy
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/PPARG-critic-gradient-CPU-20261108'
RESULT = ROOT / 'analysis/PPARG_critic_gradient_CPU_20261108.json'
REPLICATES = 4
BAGS = 32
CELLS = 128


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, q):
    p = Path(p)
    tmp = p.with_suffix('.partial')
    tmp.write_text(json.dumps(q, indent=2, allow_nan=False) + '\n')
    tmp.replace(p)


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    spec = importlib.util.spec_from_file_location('strict_probe_loader', ROOT / 'analysis/diagnose_t005_calibration_checkpoints_multiseed_cpu.py')
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    m = loader.m
    torch, np = m.torch, m.np
    from critic import bag_features, VectorCritic
    from pg_model import critic_features, standardize_counts
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    assert not torch.cuda.is_initialized()
    assert not OUT.exists() and not RESULT.exists()
    OUT.mkdir()
    planpath = ROOT / 'analysis/primamente_inference_hypothesis_plan_20261108.json'
    plan = json.loads(planpath.read_bytes())
    pins = {p: h for p, h in plan['input_SHA256'].items()
            if p.endswith(('final.pt', 'counts_hvg.npz', 'metadata.npz', 'space.npz', 'manifest.json', 'pg_model.py', 'released_adapter.py', 'released_update.py', 'contracts.py', 'stage_law.py'))}
    pins.update({str(planpath): sha(planpath), str(Path(__file__).resolve()): sha(__file__),
                 str(Path(m.SCIENCE) / 'critic.py'): sha(Path(m.SCIENCE) / 'critic.py'),
                 str(Path(loader.__file__).resolve()): sha(loader.__file__)})
    for p, h in pins.items():
        assert sha(p) == h, p
    protocol = dict(created_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    mode='Frozen CPU sensitivity probes; no checkpoint selection or training',
                    seeds=list(range(5)), replicates=REPLICATES, bags_per_replicate=BAGS,
                    cells_per_bag=CELLS, stages='8 bags per stage; donors uniform within stage',
                    independent_log_mu_coordinate='PPARG Gamma rate multiplied by exp(alpha); other rates and theta fixed',
                    weak_derivative='sum_i lambda_i * [L(n+e_i)-L(n)]',
                    straight_through='Exact existing critic_features with a free per-bag log-rate coordinate',
                    correction='Replace only standardized log PPARG feature by paired real-cell value; synthetic feature probe, not a gene intervention',
                    CPU_wall_cap_seconds=1800, new_optimizer_updates=0, input_SHA256=pins)
    write(OUT / 'protocol.json', protocol)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('CPU probe cap')))
    signal.alarm(protocol['CPU_wall_cap_seconds'])
    started = time.monotonic()
    ctx = m.Context(Path(plan['fitting_cache']), 'cpu')
    meta = np.load(ctx.cache / 'train/metadata.npz', allow_pickle=False)
    j = list(meta['genes']).index('PPARG')
    W, mean, sd = ctx.W.double(), ctx.mean.double(), ctx.sd.double()
    rawW = torch.tensor(ctx.space.W, dtype=torch.float64)
    leverage = rawW.square().sum(1)
    gram_error = float((rawW.T @ rawW - torch.eye(50)).abs().max())
    assert gram_error < 1e-4
    # dE[f(N)]/d log(lambda) = lambda E[f(N+1)-f(N)].
    # Verify independently by differentiated Poisson probabilities and finite differences.
    import math
    def expectation(lam):
        return sum(math.exp(-lam + n * math.log(lam) - math.lgamma(n + 1)) * math.log1p(n) for n in range(120))
    lam = 2.3
    weak = lam * sum(math.exp(-lam + n * math.log(lam) - math.lgamma(n + 1)) * (math.log1p(n + 1) - math.log1p(n)) for n in range(120))
    exact = sum(math.exp(-lam + n * math.log(lam) - math.lgamma(n + 1)) * (n - lam) * math.log1p(n) for n in range(120))
    finite = (expectation(lam * math.exp(1e-5)) - expectation(lam * math.exp(-1e-5))) / 2e-5
    assert abs(weak - exact) < 1e-12 and abs(weak - finite) < 1e-8
    records = []
    for model in plan['models']:
        seed = model['seed']
        q = torch.load(model['checkpoint'], map_location='cpu', weights_only=False)
        G, P = loader.generation_modules(ctx, q, seed, 20000, plan)
        Dnative = VectorCritic().eval()
        Dnative.load_state_dict(q['policy']['models']['critic'], strict=True)
        for key, value in Dnative.state_dict().items():
            assert torch.equal(value, q['policy']['models']['critic'][key])
        before = tuple(m.fingerprint(x) for x in (G, P, Dnative))
        D = copy.deepcopy(Dnative).double().eval()
        for module in (G, P, Dnative, D):
            module.requires_grad_(False)
        loss = m.Recipe.from_dict(q['recipe']).make_loss()
        for rep in range(REPLICATES):
            # Same real schedules across trained seeds; separate latent/count streams.
            schedule_rows = []
            for stage in range(4):
                c0 = np.zeros(7, dtype=np.float32)
                c0[3] = stage / 3
                schedule_rows.append(ctx.law.schedule(c0, 2026110900 + rep * 4 + stage, m=8, k=CELLS).rows)
            rows = torch.as_tensor(np.concatenate(schedule_rows))
            real = ctx.X[rows]
            libs = real.sum(-1)
            c = torch.zeros(BAGS, 7)
            c[:, 3] = torch.arange(4).repeat_interleave(8) / 3
            random_seed = 202611080000 + seed * 100 + rep
            torch.manual_seed(random_seed)
            z, _ = P.sample(BAGS, generator=torch.Generator().manual_seed(random_seed + 1))
            mu, theta = G.parameters_nb(z, c, CELLS, libs, torch.Generator().manual_seed(random_seed + 2))
            rate = torch.distributions.Gamma(theta, theta / mu).rsample()
            counts = torch.poisson(rate, generator=torch.Generator().manual_seed(random_seed + 3))
            pcs_native = ctx.project(counts)
            native_logits = Dnative(bag_features(pcs_native), c)
            assert torch.allclose(loss.g_loss(native_logits), torch.nn.functional.softplus(-native_logits).mean(), atol=1e-7)
            x, r, cd = counts.double(), rate.double(), c.double()
            standardized = standardize_counts(x, mean, sd)
            pcs = standardized @ W
            logits = D(bag_features(pcs), cd)
            parity = float((logits - native_logits.double()).abs().max())
            assert parity < 1e-3, parity
            base = torch.nn.functional.softplus(-logits)
            # Feature-space correction keeps all other standardized gene features fixed.
            real_std = standardize_counts(real.double(), mean, sd)
            corrected = pcs + (real_std[:, :, j] - standardized[:, :, j])[:, :, None] * W[j]
            corrected_loss = torch.nn.functional.softplus(-D(bag_features(corrected), cd))
            realpcs = real_std @ W
            erased = realpcs + ((-mean[j] / sd[j]) - real_std[:, :, j])[:, :, None] * W[j]
            erase_delta = torch.nn.functional.softplus(-D(bag_features(erased), cd)) - torch.nn.functional.softplus(-D(bag_features(realpcs), cd))
            with torch.enable_grad():
                alpha = torch.zeros(BAGS, dtype=torch.float64, requires_grad=True)
                rp = r.clone()
                rp[:, :, j] = r[:, :, j] * alpha.exp()[:, None]
                f = critic_features(x, rp, mean, sd) @ W
                st_forward = D(bag_features(f), cd)
                assert torch.equal(st_forward, logits)
                st_loss = torch.nn.functional.softplus(-st_forward)
                st = torch.autograd.grad(st_loss.sum(), alpha)[0].detach()
            # A projection is cellwise. Compute each increment once, then place it
            # into its own counterfactual bag, retaining all other cell coordinates.
            xp = x.clone()
            xp[:, :, j] += 1
            incremented = standardize_counts(xp, mean, sd) @ W
            differences = torch.empty(BAGS, CELLS, dtype=torch.float64)
            for b in range(BAGS):
                counter = pcs[b:b + 1].expand(CELLS, -1, -1).clone()
                ii = torch.arange(CELLS)
                counter[ii, ii] = incremented[b]
                if b == 0 and rep == 0:
                    direct = x[:1].clone()
                    direct[0, 0, j] += 1
                    assert torch.equal(counter[0], (standardize_counts(direct, mean, sd) @ W)[0])
                values = torch.nn.functional.softplus(-D(bag_features(counter), cd[b:b + 1].expand(CELLS, -1)))
                differences[b] = values - base[b]
            weak_gradient = (r[:, :, j] * differences).sum(1)
            result = dict(seed=seed, replicate=rep, native_float32_vs_float64_logit_max_error=parity,
                          forward_ST_parity_passed=True, weak_increment_forward_parity_passed=True,
                          by_stage=[])
            for stage in range(4):
                sl = slice(stage * 8, (stage + 1) * 8)
                result['by_stage'].append(dict(stage=stage,
                    predicted_mu=float(mu[sl, :, j].mean()), observed_count_mean=float(real[sl, :, j].mean()),
                    model_detection=float((x[sl, :, j] > 0).double().mean()), observed_detection=float((real[sl, :, j] > 0).double().mean()),
                    model_logmean=float((standardized[sl, :, j] * sd[j] + mean[j]).mean()),
                    observed_logmean=float((real_std[sl, :, j] * sd[j] + mean[j]).mean()),
                    base_G_loss=float(base[sl].mean()), correction_G_loss_delta=float((corrected_loss - base)[sl].mean()),
                    real_erasure_G_loss_delta=float(erase_delta[sl].mean()),
                    ST_log_mu_gradient=float(st[sl].mean()), weak_log_mu_gradient=float(weak_gradient[sl].mean()),
                    per_bag_ST_log_mu_gradient=st[sl].tolist(), per_bag_weak_log_mu_gradient=weak_gradient[sl].tolist()))
            records.append(result)
            write(OUT / f'seed{seed}-rep{rep}.json', result)
            print(json.dumps(dict(seed=seed, replicate=rep, elapsed=time.monotonic() - started,
                                  ST=[a['ST_log_mu_gradient'] for a in result['by_stage']],
                                  weak=[a['weak_log_mu_gradient'] for a in result['by_stage']],
                                  correction=[a['correction_G_loss_delta'] for a in result['by_stage']])), flush=True)
        assert before == tuple(m.fingerprint(x) for x in (G, P, Dnative))
        assert sha(model['checkpoint']) == model['checkpoint_SHA256']
    summary = []
    for stage in range(4):
        values = [r['by_stage'][stage] for r in records]
        metrics = ['predicted_mu', 'observed_count_mean', 'model_detection', 'observed_detection', 'model_logmean', 'observed_logmean',
                   'base_G_loss', 'correction_G_loss_delta', 'real_erasure_G_loss_delta', 'ST_log_mu_gradient', 'weak_log_mu_gradient']
        summary.append(dict(stage=stage, **{k: float(np.mean([v[k] for v in values])) for k in metrics},
                            seed_means={str(s): {k: float(np.mean([r['by_stage'][stage][k] for r in records if r['seed'] == s]))
                                                for k in metrics} for s in range(5)}))
    output = dict(status='COMPLETE_FROZEN_PPARG_CRITIC_AND_GRADIENT_PROBES', UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  protocol=protocol, summary=summary, records=records,
                  PCA=dict(PPARG_coordinate_retained_squared_projection=float(leverage[j]),
                           median_gene_coordinate_retained_squared_projection=float(leverage.median()),
                           PPARG_percentile=float((leverage <= leverage[j]).double().mean()),
                           raw_basis_orthogonality_max_error=gram_error),
                  mathematical_check=dict(poisson_identity=weak, differentiated_probability=exact, centered_finite_difference=finite),
                  sign_convention='Negative G-loss delta/gradient rewards increasing or correcting PPARG.',
                  new_training_fits=0, new_optimizer_updates=0, new_GPU_allocations=0, CUDA_initialized=torch.cuda.is_initialized(),
                  all_generator_prior_critic_states_and_checkpoint_hashes_unchanged=True,
                  limitations=['Frozen local probes do not reconstruct historical optimizer causation.',
                               'The independent PPARG log-mean derivative omits softmax coupling to other genes and shared network parameters.',
                               'Feature replacement breaks cross-gene dependence and is not a biological intervention.',
                               'Sampling replicates and trained seeds are not independent donors or external validation.',
                               'Stage-balanced diagnostics differ from the donor-uniform stage proportions used in training.'],
                  elapsed_seconds=time.monotonic() - started, protocol_SHA256=sha(OUT / 'protocol.json'),
                  per_probe_SHA256={str(p): sha(p) for p in sorted(OUT.glob('seed*.json'))})
    for p, h in pins.items():
        assert sha(p) == h, p
    assert not torch.cuda.is_initialized()
    write(RESULT, output)
    signal.alarm(0)
    print(json.dumps(dict(result_path=str(RESULT), summary=summary, PCA=output['PCA'])), flush=True)


if __name__ == '__main__':
    main()
