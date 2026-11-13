"""Worked example of the round-2 arm plug-in API (r2/arms.py docstring).  NOT a scientific arm.

Flag ``example`` or ``example@s`` (0 <= s <= 1, default 0.5); mode
``exshrink@v`` (0 <= v <= 1).  Every hook is exercised (test_core.py):

setup        train-only centring vector (mean log1p CP10k of train-donor
             pseudobulks) and a fixed random projection [G,8] seeded from
             SeedSequence([20261109, crc32('example'), 1]) (not a training stream).
(a)          ``conditions``: slot 0 (Braak slot, unused in every arm) = s * slot 3.
(b)          ``batch_condition``: zero slot 0 for each batch donor with
             probability 1/2, drawn from the arm's own numpy stream.
(c)          ``generator_class``: adds ``example_bias`` [G] (zeros) to the
             output logits via HookedCountGenerator.logits; ``init_generator``
             fills it with zeros again (shows the hook; no RNG);
             ``generator_grad_params`` asserts it gets a gradient at update 0.
(d)          two critic blocks: ``pb8`` (linear-in-counts bag pseudobulk:
             log1p CP10k of the summed counts, centred, projected on the fixed
             [G,8] matrix; fake bags use counts + (rate - rate.detach()) so
             the gradient flows through the rate) and ``depth1`` (log of the
             mean cell library, centred on the train mean).
(e)          ``generation_condition``: identity apart from a check; mode
             ``exshrink@v``: for a random half of the generated donors (mode
             stream) slot 0 is multiplied by v, and ``prepare`` multiplies
             the generator's ``example_bias`` by v when present.
(f)          ``state``: s, centring vector, projection, train mean log depth;
             blocks report their own state.
"""
from __future__ import annotations

import zlib

import numpy as np
import torch
from torch import nn

import arms
from arms import Arm, FeatureBlock, register, register_mode, canonical_float

NAME = 'example'


def _log_cp10k_pb(x):
    """[B, cells, G] counts (float, may carry a straight-through gradient) -> [B, G] log1p CP10k of the bag sum."""
    pb = x.sum(1)
    return torch.log1p(pb * (1e4 / pb.sum(-1, keepdim=True).clamp_min(1)))


class PseudobulkBlock(FeatureBlock):
    name = 'pb8'
    width = 8

    def __init__(self, arm):
        super().__init__()
        self.arm = arm
        self.info = {'what': 'centred log1p CP10k bag pseudobulk @ fixed random [G,8] / sqrt(G) * 4'}

    def _f(self, x):
        return (_log_cp10k_pb(x) - self.arm.center) @ self.arm.proj

    def real(self, ctx, idx, ds):
        return self._f(ctx.X[idx])

    def fake(self, ctx, cf, rate, counts):
        return self._f(counts + (rate - rate.detach()))

    def state(self):
        return {'width': self.width}


class DepthBlock(FeatureBlock):
    name = 'depth1'
    width = 1

    def __init__(self, arm):
        super().__init__()
        self.arm = arm

    def real(self, ctx, idx, ds):
        return torch.log(ctx.X[idx].sum(-1).mean(1, keepdim=True)) - self.arm.log_depth

    def fake(self, ctx, cf, rate, counts):
        x = counts + (rate - rate.detach())
        return torch.log(x.sum(-1).mean(1, keepdim=True).clamp_min(1)) - self.arm.log_depth


@register
class ExampleArm(Arm):
    name = NAME
    value_policy = 'optional'
    conflicts = ('offset',)
    needs_Z = False

    @classmethod
    def parse_value(cls, text):
        return canonical_float(text, lo=0.0, hi=1.0)

    def setup(self, ctx):
        self.s = 0.5 if self.value is None else float(self.value)
        with torch.no_grad():
            pbs = torch.stack([_log_cp10k_pb(ctx.X[torch.as_tensor(r, device=ctx.device)][None])[0] for r in ctx.by])
            self.center = pbs.mean(0)
            self.log_depth = float(torch.log(ctx.fitlibs.double().mean()))
        rng = np.random.default_rng(np.random.SeedSequence([20261109, zlib.crc32(NAME.encode()), 1]))
        self.proj = torch.as_tensor(rng.standard_normal((ctx.n_genes, 8)) / np.sqrt(ctx.n_genes) * 4, dtype=torch.float32, device=ctx.device)
        self.info = {'s': self.s, 'log_depth_train': self.log_depth}

    def conditions(self, ctx, raw_c, cond):
        out = cond.copy()
        out[:, 0] = np.float32(self.s) * cond[:, 3]
        return out

    def batch_condition(self, ctx, c, ds, rng, phase):
        keep = torch.as_tensor(rng.random(len(ds)) >= 0.5, dtype=c.dtype, device=c.device)
        out = c.clone()
        out[:, 0] = c[:, 0] * keep
        return out

    def generator_class(self, ctx, cls):
        n_genes = ctx.n_genes

        class ExampleGenerator(cls):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self.example_bias = nn.Parameter(torch.zeros(n_genes))  # no RNG draw

            def logits(self, h, z, c):
                return super().logits(h, z, c) + self.example_bias
        return ExampleGenerator

    def init_generator(self, ctx, G):
        G.example_bias.zero_()

    def generator_grad_params(self, G):
        return [G.example_bias]

    def feature_blocks(self, ctx):
        return [PseudobulkBlock(self), DepthBlock(self)]

    def generation_condition(self, ctx, raw_c, cond, is_heldout):
        assert len(is_heldout) == len(cond)
        return cond

    def state(self):
        return {'s': self.s, 'center': self.center.cpu(), 'proj': self.proj.cpu(), 'log_depth': self.log_depth}


def _exshrink_plan(ctx, value, target, rng, states):
    cond = np.repeat(target.default_cond[None], target.M, 0)
    pick = rng.random(target.M) < 0.5
    cond[pick, 0] *= np.float32(value)
    return cond, ['final'] * target.M


def _exshrink_prepare(ctx, value, state_name, G, P):
    if hasattr(G, 'example_bias'):
        G.example_bias.mul_(float(value))


register_mode('exshrink', _exshrink_plan, value_policy='required', parse_value=lambda t: canonical_float(t, lo=0.0, hi=1.0),
              prepare=_exshrink_prepare, doc='example: slot 0 x v for a random half of the generated donors; example_bias x v')
