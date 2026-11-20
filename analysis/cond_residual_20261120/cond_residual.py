"""Opt-in condition-aware transport residual for EMPF's wide continuous path.

ParticleGAN 0.10.0 EMPF measures its transport residual, for conditioned
continuous rows wider than 64 features, with ``unconditioned_chamfer_residual``:
a pooled, bidirectional nearest-neighbour distance that ignores the condition.
Exchanging two conditions' spreads keeps the pooled rows, so it reports no
change (analysis/particlegan_review_20261117).

This adapter changes exactly one thing. On that wide path it replaces the
pooled Chamfer value with the same Chamfer statistic restricted to pairs that
share a condition row: each real row's nearest fake row is searched only among
fake rows with an identical condition, and vice versa. The median scale and
``data_scale`` are the native ones, so with a single condition the value is
bit-identical to native. Everything else (critic gradients, EBI cap, gauge,
chords, optimizer, other widths/domains) is the released code path; the
residual only reaches training through the released kernel's own uses of
``record.ema_residual``.

Use: ``install(penalty)`` on a ``recipe.make_critic_penalty(...)`` object.
Nothing in the released package is modified.
"""
from __future__ import annotations

import math

import torch

from particlegan._empf.penalty import _EMPFPenaltyKernel, EMPFCriticPenalty
from particlegan._stats import pairwise_sq_dist

WIDE_MIN = 65  # native uses the chord/conditional path at width <= 64


@torch.no_grad()
def condition_group_chamfer_residual(x_real, x_fake, cond_tensors, *, eps_scale=1e-12):
    """Chamfer residual restricted to same-condition pairs; native scale.

    ``cond_tensors`` are row-aligned with both ``x_real`` and ``x_fake`` (the
    released penalty applies one condition batch to both). Returns
    ``(residual, data_scale)`` as 0-d tensors in ``x_fake``'s dtype.
    """
    b = x_real.shape[0]
    if x_fake.shape[0] != b:
        raise ValueError('condition-aware residual needs row-aligned real and fake batches')
    r = x_real.detach().flatten(1).to(dtype=torch.float32)
    f = x_fake.detach().flatten(1).to(dtype=torch.float32)
    cost = pairwise_sq_dist(r, f)
    scale = cost.median().clamp_min(eps_scale)
    c = torch.cat([t.detach().reshape(b, -1).to(device=r.device, dtype=torch.float32)
                   for t in cond_tensors], dim=1)
    same = (c.unsqueeze(1) == c.unsqueeze(0)).all(dim=-1)  # [real i, fake j]
    masked = cost.masked_fill(~same, float('inf'))
    mean_sq = 0.5 * (masked.min(dim=1).values.mean() + masked.min(dim=0).values.mean()) / scale
    residual = torch.sqrt(mean_sq.clamp_min(0.0))
    return residual.to(dtype=x_fake.dtype), scale.sqrt().to(dtype=x_fake.dtype)


class ConditionAwareEMPFKernel(_EMPFPenaltyKernel):
    """Released EMPF kernel whose wide continuous residual is condition-aware."""

    def _cond_route(self, x_real, x_fake, step, is_conditioned, cond_tensors):
        if not is_conditioned or not cond_tensors or x_real.ndim != 2 or x_fake.ndim != 2:
            return False
        if x_real.shape[0] != x_fake.shape[0] or self.coeff == 0.0:
            return False
        if self.lazy_k > 1 and step % self.lazy_k != 0:
            return False
        dim = max(1, x_real[0].numel())
        if dim < WIDE_MIN:
            return False
        discrete = self.input_domain == 'categorical' or (self.input_domain == 'auto' and dim >= 170)
        return not discrete

    def penalty(self, live, x_real, x_fake, step, collect_stats=False, *, is_conditioned=False,
                cond_tensors=(), **kwargs):
        override = None
        if self._cond_route(x_real, x_fake, step, is_conditioned, cond_tensors):
            res, scale = condition_group_chamfer_residual(x_real, x_fake, cond_tensors)
            override = (float(res.item()), float(scale.item()))
        self._cond_override = override
        self._cond_consumed = False
        try:
            value, stats = super().penalty(live, x_real, x_fake, step, collect_stats,
                                           is_conditioned=is_conditioned, cond_tensors=cond_tensors, **kwargs)
        finally:
            pending = self._cond_override
            self._cond_override = None
        if override is not None:
            if pending is not None or not self._cond_consumed:
                raise RuntimeError('condition-aware residual was computed but the native wide path did not observe it')
            self.cond_calls += 1
            if collect_stats:
                stats = dict(stats, cond_residual=override[0], cond_residual_adapter=True)
        return value, stats

    def _observe(self, res, data_scale, *, ema_decay=None):
        override = getattr(self, '_cond_override', None)
        if override is not None:
            res, data_scale = override
            self._cond_override = None
            self._cond_consumed = True
        super()._observe(res, data_scale, ema_decay=ema_decay)


def install(penalty):
    """Switch a released EMPF critic penalty to the condition-aware wide residual (in place)."""
    if not isinstance(penalty, EMPFCriticPenalty):
        raise TypeError('install expects recipe.make_critic_penalty(...) for EMPF')
    kernel = penalty.regularizer
    if type(kernel) is not _EMPFPenaltyKernel:
        raise TypeError('penalty kernel is not the released EMPF kernel')
    kernel.__class__ = ConditionAwareEMPFKernel
    kernel._cond_override = None
    kernel._cond_consumed = False
    kernel.cond_calls = 0
    penalty.cond_residual_adapter = True
    return penalty


def native_chamfer(x_real, x_fake):
    """The released pooled residual, for side-by-side diagnostics."""
    from particlegan._empf.transport import unconditioned_chamfer_residual
    res, scale = unconditioned_chamfer_residual(x_real, x_fake)
    return float(res.item()), float(scale.item())


assert math.isfinite(WIDE_MIN)
