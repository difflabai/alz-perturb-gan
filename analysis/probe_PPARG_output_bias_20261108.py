"""Frozen actual-output-bias response; coupled valid Gamma-Poisson counts.

This is a diagnostic landscape, not a tuned or retrained model. Parameter
substitution is temporary through torch.func; source checkpoints stay intact.
"""
import copy
import datetime
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/PPARG-output-bias-CPU-20261108'
RESULT = ROOT / 'analysis/PPARG_output_bias_CPU_20261108.json'
FACTORS = [.25, .5, 1., 2., 4., 8.]
REPLICATES = 4


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    spec = importlib.util.spec_from_file_location('frozen_probe', ROOT / 'analysis/probe_PPARG_critic_gradient_20261108.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    spec = importlib.util.spec_from_file_location('strict_probe_loader', ROOT / 'analysis/diagnose_t005_calibration_checkpoints_multiseed_cpu.py')
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    m = loader.m
    torch, np = m.torch, m.np
    from critic import VectorCritic, bag_features
    from pg_model import critic_features, standardize_counts
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    assert not torch.cuda.is_initialized() and not OUT.exists() and not RESULT.exists()
    OUT.mkdir()
    planpath = ROOT / 'analysis/primamente_inference_hypothesis_plan_20261108.json'
    plan = json.loads(planpath.read_bytes())
    upstream = json.loads((ROOT / 'analysis/PPARG_critic_gradient_CPU_20261108.json').read_bytes())
    pins = dict(upstream['protocol']['input_SHA256'])
    pins.update({str(Path(__file__).resolve()): helper.sha(__file__),
                 str(ROOT / 'analysis/PPARG_critic_gradient_CPU_20261108.json'): helper.sha(ROOT / 'analysis/PPARG_critic_gradient_CPU_20261108.json')})
    for p, h in pins.items():
        assert helper.sha(p) == h, p
    protocol = dict(created_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    seeds=list(range(5)), replicates=REPLICATES, bags=32, cells_per_bag=128,
                    output_bias_multipliers=FACTORS,
                    meaning='Add log(multiplier) to PPARG output bias through functional_call; softmax/clamp/renormalization remain exact.',
                    count_coupling='Given base rates lambda and counts N, use Binomial(N,min(ratio,1))+Poisson(lambda*max(ratio-1,0)). Marginally Poisson(lambda*ratio).',
                    no_model_selection=True, new_optimizer_updates=0, CPU_wall_cap_seconds=1800,
                    input_SHA256=pins)
    helper.write(OUT / 'protocol.json', protocol)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('CPU probe cap')))
    signal.alarm(1800)
    started = time.monotonic()
    ctx = m.Context(Path(plan['fitting_cache']), 'cpu')
    meta = np.load(ctx.cache / 'train/metadata.npz', allow_pickle=False)
    j = list(meta['genes']).index('PPARG')
    mean, sd, W = ctx.mean.double(), ctx.sd.double(), ctx.W.double()
    class MeanWrapper(torch.nn.Module):
        def __init__(self, generator):
            super().__init__()
            self.generator = generator
        def forward(self, z, c, k, lib, cell):
            return self.generator.parameters_nb(z, c, k, lib, cell)
    records = []
    for model in plan['models']:
        seed = model['seed']
        q = torch.load(model['checkpoint'], map_location='cpu', weights_only=False)
        G, P = loader.generation_modules(ctx, q, seed, 20000, plan)
        D = VectorCritic().eval()
        D.load_state_dict(q['policy']['models']['critic'], strict=True)
        before = tuple(m.fingerprint(x) for x in (G, P, D))
        D = D.double()
        Dcopy = copy.deepcopy(D)
        for module in (G, P, D):
            module.requires_grad_(False)
        wrapper = MeanWrapper(G)
        direction = torch.zeros_like(G.out[-1].bias)
        direction[j] = 1.
        for rep in range(REPLICATES):
            rows = []
            for stage in range(4):
                c0 = np.zeros(7, dtype=np.float32)
                c0[3] = stage / 3
                rows.append(ctx.law.schedule(c0, 2026110900 + rep * 4 + stage, m=8, k=128).rows)
            real = ctx.X[torch.as_tensor(np.concatenate(rows))]
            libs = real.sum(-1)
            c = torch.zeros(32, 7)
            c[:, 3] = torch.arange(4).repeat_interleave(8) / 3
            random_seed = 202611080000 + seed * 100 + rep
            torch.manual_seed(random_seed)
            z, _ = P.sample(32, generator=torch.Generator().manual_seed(random_seed + 1))
            def conditional_mean(alpha):
                bias = G.out[-1].bias + alpha * direction
                return torch.func.functional_call(wrapper, {'generator.out.3.bias': bias},
                    (z, c, 128, libs, torch.Generator().manual_seed(random_seed + 2)))
            mu, theta = conditional_mean(0.)
            rate = torch.distributions.Gamma(theta, theta / mu).rsample()
            counts = torch.poisson(rate, generator=torch.Generator().manual_seed(random_seed + 3))
            base_std = standardize_counts(counts.double(), mean, sd)
            base = torch.nn.functional.softplus(-D(bag_features(base_std @ W), c.double()))
            real_log = (standardize_counts(real.double(), mean, sd)[:, :, j] * sd[j] + mean[j]).mean(1)
            with torch.enable_grad():
                alpha = torch.tensor(0., requires_grad=True)
                new_mu, _ = conditional_mean(alpha)
                rp = rate.double() * (new_mu / mu).double()
                loss = torch.nn.functional.softplus(-D(bag_features(critic_features(counts.double(), rp, mean, sd) @ W), c.double()))
                st_by_stage = [float(torch.autograd.grad(loss[s * 8:(s + 1) * 8].mean(), alpha, retain_graph=True)[0]) for s in range(4)]
            items = []
            for factor in FACTORS:
                new_mu, new_theta = conditional_mean(math.log(factor))
                assert torch.equal(theta, new_theta)
                ratio = new_mu / mu
                torch.testing.assert_close(new_mu.sum(-1), libs, atol=.02, rtol=1e-5)
                if factor == 1.:
                    assert torch.equal(new_mu, mu)
                    new_counts = counts
                else:
                    stream = torch.Generator().manual_seed(random_seed + 999)
                    new_counts = torch.binomial(counts, ratio.clamp(max=1), generator=stream)
                    new_counts += torch.poisson(rate * (ratio - 1).clamp_min(0), generator=stream)
                transformed = standardize_counts(new_counts.double(), mean, sd)
                new_loss = torch.nn.functional.softplus(-D(bag_features(transformed @ W), c.double()))
                log_gene = transformed[:, :, j] * sd[j] + mean[j]
                stage_records = []
                for stage in range(4):
                    sl = slice(stage * 8, (stage + 1) * 8)
                    stage_records.append(dict(stage=stage, output_bias_ST_gradient=st_by_stage[stage],
                        predicted_raw_mean=float(new_mu[sl, :, j].mean()),
                        generated_nonzero=float((new_counts[sl, :, j] > 0).float().mean()),
                        generated_logmean=float(log_gene[sl].mean()), observed_logmean=float(real_log[sl].mean()),
                        per_bag_logmean_absolute_error=float((log_gene.mean(1)[sl] - real_log[sl]).abs().mean()),
                        G_loss_delta=float((new_loss - base)[sl].mean()),
                        per_bag_G_loss_delta=(new_loss - base)[sl].tolist()))
                items.append(dict(multiplier=factor, by_stage=stage_records))
            rec = dict(seed=seed, replicate=rep, items=items)
            helper.write(OUT / f'seed{seed}-rep{rep}.json', rec)
            records.append(rec)
            print(json.dumps(dict(seed=seed, replicate=rep, seconds=time.monotonic() - started,
                                  NotAD=[dict(f=x['multiplier'], log=x['by_stage'][0]['generated_logmean'], loss_delta=x['by_stage'][0]['G_loss_delta']) for x in items])), flush=True)
        # D's dtype conversion changes its fingerprint, so compare it with its
        # untouched double copy; compare G/P with exact original fingerprints.
        assert before[:2] == tuple(m.fingerprint(x) for x in (G, P))
        assert m.fingerprint(D) == m.fingerprint(Dcopy)
        assert helper.sha(model['checkpoint']) == model['checkpoint_SHA256']
    summary = []
    for stage in range(4):
        items = []
        for index, factor in enumerate(FACTORS):
            values = [r['items'][index]['by_stage'][stage] for r in records]
            fields = ['predicted_raw_mean', 'generated_nonzero', 'generated_logmean', 'observed_logmean', 'per_bag_logmean_absolute_error', 'G_loss_delta', 'output_bias_ST_gradient']
            items.append(dict(multiplier=factor, **{f: float(np.mean([x[f] for x in values])) for f in fields},
                              seed_means={str(s): {f: float(np.mean([r['items'][index]['by_stage'][stage][f] for r in records if r['seed'] == s])) for f in fields} for s in range(5)}))
        summary.append(dict(stage=stage, items=items))
    for p, h in pins.items():
        assert helper.sha(p) == h, p
    assert not torch.cuda.is_initialized()
    result = dict(status='COMPLETE_FROZEN_ACTUAL_PPARG_OUTPUT_BIAS_LANDSCAPE', UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  summary=summary, records=records, protocol=protocol, elapsed_seconds=time.monotonic() - started,
                  CUDA_initialized=False, new_optimizer_updates=0, new_training_fits=0, new_GPU_allocations=0,
                  source_model_states_and_checkpoint_files_unchanged=True,
                  limitations=['Finite Monte Carlo frozen-critic landscape, not a new fitted or selected model.',
                               'This probes the actual output-bias coordinate, including other-gene softmax redistribution, but not a full optimizer step.',
                               'A local critic failure does not prove which training design change would fix it.',
                               'Pooled matched fitting bags, not the within-subtype donor-balanced report estimand or held-out validation.'])
    helper.write(RESULT, result)
    signal.alarm(0)
    print(json.dumps(dict(result=str(RESULT), summary=summary)), flush=True)


if __name__ == '__main__':
    main()
