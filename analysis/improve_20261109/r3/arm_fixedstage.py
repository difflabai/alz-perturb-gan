"""Round-3 arm ``fixedstage[@w]`` (R6): stage-blind generator + a FIXED, non-learned stage x gene logit offset.

Flag ``fixedstage`` (w = 1) or ``fixedstage@w`` (0 <= w <= 100, w != 1; one fit-folder name per w).

Generator (application only; released update rule unchanged; no random number drawn):

* Donor FiLM MLP: receives the condition with ALL 7 slots set to 0 (as ``nostage``), so the shared
  generator learns one stage-pooled donor distribution.
* Head: logits + w * A[s(t)] before the softmax, A = round-1 ``offset_init`` table [4, G] computed
  from the fold's TRAIN donors only (arms.py docstring: donor-averaged per-stage pseudobulk
  log-ratio, shrunk by the empirical-Bayes weight tau2 / (tau2 + v_g (1/n_s - 1/n)) with one global
  method-of-moments tau2; the fixer verified that this tau2 is about 0 under permuted stage labels,
  which is the sense in which the rule is permutation-calibrated).  A is a registered BUFFER
  (``fixed_stage_offset``), not a parameter: it is in no optimizer, never changes, is saved with
  every checkpoint/snapshot/shadow EMA and is recomputed from the cache and compared at resume and
  generation (hook f).  The stage row is chosen with piecewise-linear weights on the knots
  c3 = 0, 1/3, 2/3, 1 (one-hot on the grid; off-grid values from ``condshrink`` interpolate), so
  every stage mode is defined.  Training and generation use the same head, so the critic sees the
  offset in every fake bag.

Critic: conditions on stage exactly as released (condition slot 3 = ADNC/3, other slots 0).  Why:
the arm claims P(cells | stage s) = shared stage-blind generator tilted by the fixed offset w A_s;
a stage-conditional critic tests exactly that conditional (fake bags of stage s against real bags
of stage s), so the shared part is fitted to the stage-weighted average of what is left after the
offset.  With a stage-blind critic the offset would be judged only through the stage-pooled mixture
(as extra between-donor spread), and the shared generator would be pushed to cancel that spread
instead of learning the pooled distribution.  Consequence, stated in advance: whatever stage
structure the real bags have beyond w A_s is visible to the critic but cannot be produced by the
generator (its stage path is fixed), so the critic gap will stay above base's, as for round-2
``stageanchor``.

Measured before building (r3/stage_failures.py tau2/anchor, hvg2000, folds 0-2): the global tau2
is 0.0031, -0.027 (floored to 1e-6) and -0.0054 (floored) with stage-permutation p = 0.38, 0.91,
0.51, so A has RMS 0.009-0.024 in fold 0 (2-18% of the raw log-ratio) and about 1e-5 in folds 1 and
2.  For w of order 1, ``fixedstage@w`` is therefore numerically a stage-blind generator with a
stage-conditional critic in 2 of 3 folds, which is the round-2 ``stageanchor`` configuration in
those folds (there: pb_rmse 0.989 in fold 1, 0.658 in fold 2; level bias exploded in fold 1).
Large w (e.g. 20-50) would reinstate an unshrunk, noise-dominated stage effect in fold 0 only.
This arm is built because the task asked for it; ``nostage`` (critic blind too) is the cleaner test.

Conflicts: ``offset``, ``nostage``, ``stageanchor``, ``stagedrop``, ``covars``, ``covdrop``,
``covbin`` (covariates would reach the critic but not the generator), and ``g.disp`` (repair
TR-4: g.disp's dispersion map reads the condition directly, dispersion(z, c), so the stage would
reach the generator through the dispersion; max |log-dispersion difference| between ADNC 0 and 3
was 1.70 with a random map).  ``g.cdisp`` and ``g.dispz`` are allowed: g.cdisp reads the FiLM-ed
cell state, which fixedstage computes from the all-zero condition, and g.dispz reads z only.

Stage-blind generator flag (repair E9a).  When the effective offset table is numerically zero
(max per-stage RMS below ``STAGE_BLIND_RMS`` = 1e-3, which is the case in folds 1 and 2 where tau2
is floored to 1e-6: RMS about 3e-6), the generated cells do not depend on stage at all, and the
arm is a stage-blind generator facing a stage-conditional critic.  ``info`` then carries
``stage_blind_generator: True``; r3/evaluate.py lists any unit with such a fit as STAGE-BLIND
(a unit is blind when any of its fits is), so fixedstage cannot pass T1 as a stage model through
folds in which it ignores stage.  Released generator parameters
are created first under the generator_init seed, so their initial tensors equal ``base``.
"""
from __future__ import annotations

import numpy as np
import torch

import arms
from arms import Arm, register, canonical_float, stage_of
import arm_nostage  # noqa: F401  (registers the stage modes too)

NAME = 'fixedstage'
SNAP_EPS = 1e-5
STAGE_BLIND_RMS = 1e-3


def stage_weights(c3):
    """Piecewise-linear weights [B,4] over the knots 0, 1/3, 2/3, 1 (one-hot on the grid; no gradient)."""
    t = (c3.detach() * 3).clamp(0, 3)
    lo = torch.floor(t).clamp(max=2)
    frac = t - lo
    frac = torch.where(frac < SNAP_EPS, torch.zeros_like(frac), torch.where(frac > 1 - SNAP_EPS, torch.ones_like(frac), frac))
    lo = lo.long()
    w = torch.zeros(len(c3), 4, dtype=c3.dtype, device=c3.device)
    w.scatter_add_(1, lo[:, None], (1 - frac)[:, None])
    w.scatter_add_(1, (lo + 1)[:, None], frac[:, None])
    return w


def _parse(text):
    v, t = canonical_float(text, lo=0.0, hi=100.0)
    if v == 1.0:
        raise ValueError('fixedstage@1 is fixedstage; use the bare name')
    return v, t


@register
class FixedStageArm(Arm):
    name = NAME
    value_policy = 'optional'
    conflicts = ('offset', 'nostage', 'stageanchor', 'stagedrop', 'covars', 'covdrop', 'covbin', 'g.disp')

    @classmethod
    def parse_value(cls, text):
        return _parse(text)

    @property
    def w(self):
        return 1.0 if self.value is None else float(self.value)

    def setup(self, ctx):
        table, info = arms.offset_init(ctx.X, ctx.by, stage_of(ctx.raw_c))
        self.table = torch.as_tensor(table, dtype=torch.float32) * float(self.w)   # CPU; becomes a generator buffer
        self.info = {'offset': 'w * round-1 offset_init table (EB-shrunk train stage log-ratios), FIXED buffer', 'w': self.w,
                     'generator_condition': 'all slots 0 in the donor FiLM MLP', 'critic_condition': 'released (slot 3 = ADNC/3)',
                     'offset_info': info,
                     'effective_rms_per_stage': [float(self.table[s].double().square().mean().sqrt()) for s in range(4)]}
        self.info['stage_blind_generator'] = bool(max(self.info['effective_rms_per_stage']) < STAGE_BLIND_RMS)

    def conditions(self, ctx, raw_c, cond):
        other = np.delete(cond, 3, axis=1)
        if np.any(other != 0):
            raise ValueError('fixedstage: condition slots other than the stage are already used by another arm')
        return cond

    def generator_class(self, ctx, cls):
        table = self.table.clone()

        class FixedStageGenerator(cls):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self.register_buffer('fixed_stage_offset', table.clone())

            def film(self, z, c):
                return super().film(z, torch.zeros_like(c))

            def logits(self, h, z, c):
                off = stage_weights(c[:, 3]).to(self.fixed_stage_offset.dtype) @ self.fixed_stage_offset
                return super().logits(h, z, c) + off[:, None, :]
        return FixedStageGenerator

    def init_generator(self, ctx, G):
        if not torch.equal(G.fixed_stage_offset, self.table):
            raise ValueError('fixed stage offset buffer differs from the train table')

    def state(self):
        return {'offset': self.table.clone(), 'w': self.w}
