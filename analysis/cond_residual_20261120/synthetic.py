"""Synthetic grouped-Gaussian EMPF task: native vs condition-aware wide residual.

Four conditions (7-slot condition vector, stage code in slot 3 as in the ALZ
host). Eight informative coordinates have known per-group means and SDs; the
SDs form a Latin square over {0.5, 1, 1.5, 2}, so every informative
coordinate has the same pooled SD mixture and only the condition tells the
groups' spreads apart. The other D-8 coordinates are N(0, 0.1^2) for every
group. The training loop mirrors the ALZ host's ``advance`` (UpdatePolicy with
conditional rows, every-step EMPF regulariser, lazy_k=1, particle prior,
constraint-geometry generator step). Both arms share init, minibatches,
latent/penalty streams and hyperparameters; the only difference is
``cond_residual.install`` on the critic penalty.

Usage: synthetic.py train --arm native|adapter --seed S --width D --updates U --out file.json
       synthetic.py controls --width D --out file.json
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '3'); os.environ.setdefault('MKL_NUM_THREADS', '3')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

GROUPS = 4
INFO = 8
LEVELS = (0.5, 1.0, 1.5, 2.0)
NOISE_SD = 0.1
BATCH = 32
EVAL_N = 4096
ALPHA = 0.2


def truth(width):
    sd = np.full((GROUPS, width), NOISE_SD)
    mean = np.zeros((GROUPS, width))
    for g in range(GROUPS):
        for j in range(INFO):
            sd[g, j] = LEVELS[(g + j) % GROUPS]
            if j >= INFO // 2:
                mean[g, j] = 0.75 * (g - 1.5) * (1 if j % 2 else -1)
    return mean, sd


DEVICE = torch.device('cpu')


def condition(groups):
    c = torch.zeros(len(groups), 7, device=DEVICE)
    c[:, 3] = torch.as_tensor(np.asarray(groups), dtype=torch.float32, device=DEVICE)
    return c


def draw(mean, sd, groups, rng):
    eps = rng.standard_normal((len(groups), mean.shape[1]))
    return mean[groups] + sd[groups] * eps


class Generator(nn.Module):
    def __init__(self, z_dim, width, h=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(z_dim + 7, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, width))

    def forward(self, z, c):
        return self.net(torch.cat([z, c], 1))


class Critic(nn.Module):
    """Same layout as the ALZ WideVectorCritic (phi/rho with the condition at both)."""

    def __init__(self, width, h=128):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(width + 7, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU())
        self.rho = nn.Sequential(nn.Linear(h + 7, h), nn.SiLU(), nn.Linear(h, 1))

    def forward(self, x, c):
        f = self.phi(torch.cat([x, c], 1))
        return self.rho(torch.cat([f, c], 1)).squeeze(1)


def build(arm, seed, width, updates):
    from particlegan import UpdatePolicy, get_recipe
    from particlegan.empf import bind_empf_record
    import cond_residual as CR
    rec = get_recipe('gan', trainer='empf', total_steps=updates, batch_size=BATCH,
                     num_particles=512, z_dim=8, reg_every=1)
    torch.manual_seed(1000 + seed); G = Generator(rec.z_dim, width).to(DEVICE)
    torch.manual_seed(2000 + seed); D = Critic(width).to(DEVICE)
    P = rec.make_prior(generator=torch.Generator().manual_seed(3000 + seed)).to(DEVICE)
    og, od = rec.make_optimizers(G, D, P)
    pen = rec.make_critic_penalty(od, input_domain='continuous')
    if arm == 'adapter':
        CR.install(pen)
    elif arm != 'native':
        raise ValueError(arm)
    loss = rec.make_loss()
    reg = rec.make_prior_regularizer(prior=P, reference=P, weight=1.)
    bind_empf_record(od.record, loss, reg)
    streams = {name: torch.Generator(device=DEVICE).manual_seed(4000 + 10 * seed + i)
               for i, name in enumerate(('latent_generator', 'penalty_generator', 'eval_generator', 'noise_generator'))}
    policy = UpdatePolicy(rec, G, D, prior=P, generator_optimizer=og, critic_optimizer=od, penalty=pen,
                          seed=seed, row_semantics='conditional', streams=streams)
    return rec, G, D, P, og, od, pen, loss, reg, policy


def advance(bundle, batch, updates):
    """ALZ host ``advance`` without the count/feature specifics."""
    from particlegan import serial_autograd
    from particlegan.optim.constraint_geometry import constraint_geometry_backward
    rec, G, D, P, og, od, pen, loss, reg, policy = bundle
    with serial_autograd():
        real, c = batch()
        noise = policy.begin_step(real, game_record=pen.regularizer.record, execution_limit=updates)
        assert noise.input_sigma == 0 and float(noise.output_sigma) == 0
        D.train(); G.eval()
        with torch.no_grad():
            z, _ = P.sample(len(real), generator=policy.latent_generator)
            policy.observe_support(z)
            f = G(z, c)
        policy.observe_critic_pair(real, f)
        dr, df = D(real, c), D(f, c); adv = loss.d_loss(dr, df)
        penalty = pen(D, real, f, c); ld = adv + penalty
        od.zero_grad(); ld.backward(); od.step(); policy.after_critic_step()
        D.eval(); G.train(); D.requires_grad_(False)
        real2, c2 = batch()
        z, idx = P.sample(len(real2), generator=policy.latent_generator)
        lg = loss.g_loss(D(G(z, c2), c2))
        raw = reg(P.z); total = lg + rec.prior_reg * raw
        og.zero_grad(); policy.observe_sampled_rows(idx)
        constraint_geometry_backward(total, og, (lg,))
        policy.after_generator_backward(loss_gan=lg.detach(), loss_critic=adv.detach())
        og.step(); policy.after_generator_step(); D.requires_grad_(True); policy.finish_step()
    rec_ = pen.regularizer.record
    return dict(ema_residual=float(rec_.ema_residual), D_adv=float(adv.detach()), penalty=float(penalty.detach()),
                G_adv=float(lg.detach()))


@torch.no_grad()
def sample_model(G, P, width, seed):
    gen = torch.Generator(device=DEVICE).manual_seed(9000 + seed)
    out = []
    for g in range(GROUPS):
        z, _ = P.sample(EVAL_N, generator=gen)
        out.append(G(z, condition([g] * EVAL_N)).cpu().numpy().astype(np.float64))
    return np.stack(out)  # [G, N, D]


def score(samples, width, eval_seed=777):
    """Group mean/SD error, 80% coverage, width and interval score against independent truth draws."""
    mean, sd = truth(width)
    rng = np.random.default_rng(eval_seed)
    test = np.stack([draw(mean, sd, np.full(EVAL_N, g), rng) for g in range(GROUPS)])
    lo, hi = np.quantile(samples, [ALPHA / 2, 1 - ALPHA / 2], axis=1)  # [G, D]
    inside = (test >= lo[:, None]) & (test <= hi[:, None])
    isc = (hi - lo)[:, None] + (2 / ALPHA) * (np.maximum(lo[:, None] - test, 0) + np.maximum(test - hi[:, None], 0))
    gm, gs = samples.mean(1), samples.std(1)
    out = {}
    for name, sl in (('informative', slice(0, INFO)), ('noise', slice(INFO, width))):
        lr = np.log(gs[:, sl] / sd[:, sl])
        out[name] = dict(mean_rmse=float(np.sqrt(((gm[:, sl] - mean[:, sl]) ** 2).mean())),
                         sd_rmse=float(np.sqrt(((gs[:, sl] - sd[:, sl]) ** 2).mean())),
                         logsd_rmse=float(np.sqrt((lr ** 2).mean())), logsd_bias=float(lr.mean()),
                         coverage80=float(inside[:, :, sl].mean()), width80=float((hi - lo)[:, sl].mean()),
                         interval_score=float(isc[:, :, sl].mean()),
                         per_group_logsd_bias=[float(x) for x in lr.mean(1)])
    return out


def controls(width):
    mean, sd = truth(width)
    rng = np.random.default_rng(31337)
    correct = np.stack([draw(mean, sd, np.full(EVAL_N, g), rng) for g in range(GROUPS)])
    swapped_sd = sd[[1, 0, 3, 2]]  # exchange spreads between groups 0<->1 and 2<->3, keep means
    swapped = np.stack([mean[g] + swapped_sd[g] * rng.standard_normal((EVAL_N, width)) for g in range(GROUPS)])
    pooled = np.concatenate(list(correct))
    collapse = np.stack([mean[g] + 0.5 * (pooled[rng.integers(len(pooled), size=EVAL_N)] - pooled.mean(0))
                         for g in range(GROUPS)])
    return {'correct_sampler': score(correct, width), 'swapped_spread_sampler': score(swapped, width),
            'half_spread_pooled_sampler': score(collapse, width)}


def train(arm, seed, width, updates, log_every=250, device='cpu', memory=.04):
    global DEVICE
    DEVICE = torch.device(device)
    if DEVICE.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(memory)
    torch.set_num_threads(3)
    torch.use_deterministic_algorithms(True)
    mean, sd = truth(width)
    data_rng = np.random.default_rng(5000 + seed)

    def batch():
        g = data_rng.integers(GROUPS, size=BATCH)
        return torch.as_tensor(draw(mean, sd, g, data_rng), dtype=torch.float32, device=DEVICE), condition(g)
    bundle = build(arm, seed, width, updates)
    curve, t0 = [], time.monotonic()
    for step in range(updates):
        m = advance(bundle, batch, updates)
        if (step + 1) % log_every == 0 or step == 0:
            curve.append(dict(step=step + 1, **m))
    rec, G, D, P, *_, pen, loss, reg, policy = bundle
    G.eval()
    samples = sample_model(G, P, width, seed)
    assert policy.completed_steps == updates and pen.regularizer.record.calls == updates
    cond_calls = getattr(pen.regularizer, 'cond_calls', 0)
    assert (cond_calls == updates) if arm == 'adapter' else cond_calls == 0
    return dict(arm=arm, seed=seed, width=width, updates=updates, device=device, seconds=time.monotonic() - t0,
                regularizer_calls=pen.regularizer.record.calls, adapter_calls=cond_calls,
                score=score(samples, width), curve=curve)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command', choices=('train', 'controls'))
    p.add_argument('--arm', default='native'); p.add_argument('--seed', type=int, default=0)
    p.add_argument('--width', type=int, default=128); p.add_argument('--updates', type=int, default=3000)
    p.add_argument('--device', default='cpu'); p.add_argument('--gpu-mem-fraction', type=float, default=.04)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f'{out} exists')
    value = train(a.arm, a.seed, a.width, a.updates, device=a.device, memory=a.gpu_mem_fraction) if a.command == 'train' else controls(a.width)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(value, indent=1, allow_nan=False) + '\n')
    print(json.dumps(value.get('score', value))[:400])
