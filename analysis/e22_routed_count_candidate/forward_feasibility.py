"""CPU-only complete routed count forward; not a trained/qualified ALZ adapter.

Native row controls stay enabled. The packed raw-count/observation output lets
the full routed callback retain the existing transformed-rate gradient while
exposing exact integer counts through the same served forward.
"""
import copy
import datetime
import hashlib
import json
import math
import os
from pathlib import Path

import torch
from torch import nn
from particlegan import E22Policy, RoutedRows, get_recipe
from pg_model import CountGenerator, critic_features, standardize_counts
from critic import VectorCritic, bag_features
from variant_pg_adapter import training_streams

ROOT = Path('/ml2/alz')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class Router(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.register_buffer('log_mass', torch.zeros(n))


class CountForward:
    def __init__(self, genes):
        self.genes = genes
        self.mean, self.sd = torch.zeros(genes), torch.ones(genes)
        self.W = torch.randn(genes, 50, generator=torch.Generator().manual_seed(20261104201)) / math.sqrt(genes)

    def route(self, models, context, candidate):
        logits = context[:, :8] @ candidate.table.T / math.sqrt(8) / .25
        return (logits + candidate.log_mass).softmax(-1)

    @staticmethod
    def unpack_seeds(context):
        words = context[:, -12:].detach().to(device='cpu', dtype=torch.int64)
        assert bool(((words >= 0) & (words < 65536)).all())
        return [[sum(int(row[4*k+j]) << (16*j) for j in range(4)) & ((1 << 63)-1)
                 for k in range(3)] for row in words]

    def generate(self, models, context, candidate, weights):
        assert candidate.codes is not None
        assert candidate.codes.shape == (len(context), 8)
        rows = []
        for i, (gamma_seed, cell_seed, count_seed) in enumerate(self.unpack_seeds(context)):
            code = candidate.codes[i:i+1]
            condition, libraries = context[i:i+1, 8:15], context[i:i+1, 15:-12]
            assert libraries.shape[1] > 0
            devices = [code.device.index] if code.device.type == 'cuda' else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(gamma_seed)
                cell = torch.Generator(device=code.device).manual_seed(cell_seed)
                count = torch.Generator(device=code.device).manual_seed(count_seed)
                rate = models['generator'].rates(code, condition, libraries.shape[1], libraries, cell)
                counts = torch.poisson(rate.detach(), generator=count)
                observed = critic_features(counts, rate, self.mean.to(code), self.sd.to(code))
                rows.append(torch.cat([counts, observed], -1))
        return torch.cat(rows, 0)

    def observation(self, packed):
        assert packed.ndim == 3 and packed.shape[-1] == 2*self.genes
        return bag_features(packed[..., self.genes:] @ self.W.to(packed))

    def features(self, models, context, samples, targets):
        return models['critic'].phi(torch.cat([self.observation(samples), context[:, 8:15]], -1))


def context(seed=20261104202, batch=32, cells=8):
    g = torch.Generator().manual_seed(seed)
    q, c = torch.randn(batch, 8, generator=g), torch.zeros(batch, 7)
    c[:, 3] = torch.arange(batch).remainder(4)/3
    lib = torch.full((batch, cells), 40.)
    words = torch.randint(65536, (batch, 12), generator=g).float()
    return torch.cat([q, c, lib, words], -1)


def build(genes=8):
    seeds = training_streams(0, 0)
    rec = get_recipe('e22_routed', total_steps=None, batch_size=32, num_particles=512, z_dim=8)
    torch.manual_seed(seeds['generator_init'])
    G = CountGenerator(torch.ones(genes), torch.full((genes,), 3.), zd=8, zc=8)
    torch.manual_seed(seeds['critic_init'])
    D = VectorCritic()
    P = rec.make_prior(generator=torch.Generator().manual_seed(seeds['prior_init']))
    router, forward = Router(512), CountForward(genes)
    rows = RoutedRows(route=forward.route, generate=forward.generate, features=forward.features, probe_interval=20)
    og, od = rec.make_optimizers(G, D, P, ema_critic=copy.deepcopy(D))
    penalty = rec.make_critic_penalty(od)
    policy = E22Policy(rec, G, D, prior=P, router=router, routed_rows=rows,
                       generator_optimizer=og, critic_optimizer=od, penalty=penalty,
                       row_semantics='dense_soft', seed=0,
                       streams={name: torch.Generator().manual_seed(seeds['policy_'+key])
                                for name, key in [('latent_generator','latent'),('penalty_generator','penalty'),
                                                  ('eval_generator','eval'),('noise_generator','noise')]})
    return rec, policy, forward


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    torch.set_num_threads(1)
    rec, policy, forward = build()
    assert rec.total_steps is None and rec.particle_birth_death and rec.row_evidence_gate
    assert rec.row_policy == 'routed_paired' and policy.row_semantics == 'dense_soft'
    assert policy.routed_control is not None and policy.controller is not None
    assert rec.output_noise_mode == 'learnable' and policy.log_output_sigma is not None
    x = context()
    before = torch.get_rng_state().clone()
    packed = policy.routed_generate(x, sigma=0, perturb=False)
    assert torch.equal(before, torch.get_rng_state())
    counts, observed = packed[..., :8], packed[..., 8:]
    assert counts.shape == (32,8,8) and bool(torch.isfinite(packed).all())
    assert bool((counts >= 0).all()) and torch.equal(counts, counts.floor())
    assert torch.equal(observed, standardize_counts(counts, forward.mean, forward.sd))
    permutation = torch.arange(31,-1,-1)
    assert torch.equal(packed[permutation], policy.routed_generate(x[permutation], sigma=0, perturb=False))
    assert torch.equal(packed[::3], policy.routed_generate(x[::3], sigma=0, perturb=False))
    replay = policy.routed_generate(x, sigma=0, perturb=False)
    assert torch.equal(packed, replay)
    served = policy.served_model()
    clean = served.routed_forward(x, perturb=False, output_noise=False)
    assert clean.shape == packed.shape and torch.equal(clean, packed)
    loss = forward.observation(packed).square().mean()
    policy.opt_g.zero_grad()
    loss.backward()
    assert policy.G.log_theta.grad is not None and bool(policy.G.log_theta.grad.abs().sum() > 0)
    assert policy.table.grad is not None and bool(policy.table.grad.abs().sum() > 0)
    assert not torch.cuda.is_initialized()
    out = ROOT/'analysis/e22_routed_count_candidate/forward_feasibility.json'
    assert not out.exists()
    result = {
        'status': 'ACTUAL_PUBLIC_E22_ROUTED_POLICY_AND_COMPLETE_SYNTHETIC_COUNT_FORWARD_FEASIBLE_NOT_UPDATE_QUALIFIED',
        'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'resolved_recipe': rec.to_dict(), 'row_controls_disabled': False,
        'raw_Gamma_Poisson_integer_forward_verified': True,
        'auxiliary_observation_exactly_matches_raw_count_transform_forward': True,
        'same_context_deterministic_under_batch_permutation_subsetting_replay': True,
        'ambient_Torch_RNG_restored_by_complete_callback': True,
        'public_clean_served_routed_forward_matches_raw_fast_counts': True,
        'nonzero_synthetic_dispersion_and_table_gradients': True,
        'native_optimizer_pairs_executed': 0, 'native_model_qualified': False,
        'fit_guard_partition_implemented': False, 'real_data_accessed': False,
        'GPU_context_created': False, 'quality_measured': False,
        'source_SHA256': {str(Path(__file__).resolve()): sha(__file__),
                          str(ROOT/'analysis/e22_routed_count_feasibility_hypothesis.json'): sha(ROOT/'analysis/e22_routed_count_feasibility_hypothesis.json')},
        'scope': 'Synthetic8gene/8cell public-policy construction and complete count callback only. No native optimizer pair, fit/guard datasets, probe/acceptance lifecycle, checkpoint continuation, full-width qualification or GPU runtime. Not a completed E22 ALZ quality trial. Existing scalar conditional refusal remains valid; no native mechanisms were disabled.',
    }
    out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='resolved_recipe'}, indent=2))


if __name__ == '__main__':
    main()
