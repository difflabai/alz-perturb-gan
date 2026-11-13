"""Arm ``stageanchor``: anchored stage path (round-2 fix for F2, aimed also at F3).

Flag ``stageanchor`` (no value).  Conflicts with round-1 ``offset`` (two stage
paths on the logits) and with ``stagedrop``.

What it changes (application only; released update rule, loss, optimizer and
policy calls unchanged):

1. Fixed direction, train donors only.  A[4,G] = round-1 ``offset_init``
   table (r2/arms.py docstring): per stage s, the empirical-Bayes shrunken
   donor-averaged train pseudobulk log-ratio w_sg * r_sg (rows of stages
   without train donors are 0).  A is a buffer (``stage_anchor``), never
   trained, recomputed from the cache at resume/generation and compared
   exactly (hook f).
2. Learnable gains.  ``stage_gain`` [4], initialised to 1 (constant, no RNG
   draw), in the generator optimizer group.  The output logits become
       logits + (sum_s w_s(c3) * gain_s * A_s)[:, None, :]
   where w(c3) are piecewise-linear interpolation weights on the knots
   c3 = 0, 1/3, 2/3, 1 (on the grid, w is one-hot on the donor's ADNC stage;
   values within 1e-5 of a knot snap to it).  Off-grid c3 (e.g. the
   ``condshrink`` mode) interpolates between adjacent stage rows.
3. No other generator stage path.  The donor FiLM MLP receives the condition
   with slot 3 set to 0 for every donor (``film`` override), so the stage can
   reach the generator only through the 4 gains along the fixed directions.
   The critic still sees the full condition (critic conditional on stage, as
   released), which is what drives the gains.  Other slots (covars 4-6) are
   passed unchanged.

Released parameters are created first, under the generator_init seed, so
their initial tensors equal ``base``.  The released DualNorm optimizer moves
a 1-D parameter by a normalised step (direction / |direction| times the
scheduled rate), so the 4 gains take steps of fixed length whatever the
gradient size (measured on CPU, fold 0: |gain - 1| = 0.002 after 3 updates);
expect a jitter of about that per-step size around their equilibrium.

Why it should help (F2): the round-1 ``offset`` table had 4 x G free
parameters on top of the MLP stage path; it moved but learned structure that
correlates -0.07..0.60 with the real stage log-ratio, duplicating the ADNC
path.  Here the stage effect has 4 degrees of freedom and is confined to the
empirical-Bayes train direction, so it cannot carry non-stage structure, and
with the MLP path removed it is not duplicated.  (F3): the generators' stage
deviations are 3-8x the real train deviations; on round-1 fits about 55-65%
(base) and 90% (genemean) of that is seed-specific noise in the stage path
and the rest mostly does not replicate on held-out donors.  An effect bounded
to gain_s x A_s has size |gain_s| x |A_s| with |A_s| already shrunk (the
shrunk stage-mean baseline is about as good as the grand mean on held-out
donors), so a gain near 1 reproduces that baseline's stage effect, and the
per-seed random stage structure has no parameters to live in.

What could go wrong: (i) the critic may push gains far above 1 (the critic
gap is driven by noisy per-stage differences); the mode ``anchorgain@v``
(generation only, multiplies every gain by v; FINAL STATE ONLY: its plan uses
'final' for every generated donor, and no mode combines anchorgain with the
snapshot ensemble - repair 2026-11-10, review FID-4, corrected this text) and
``condshrink``/``condmix`` remain available to shrink at generation.  (ii)
Real stage-related cell-state shifts that are not in the pseudobulk
direction (e.g. a change in the fraction of DAM-like cells, which changes
the cell distribution, not only the mean) can no longer be represented: the
FiLM MLP is stage-blind, so the within-donor cell distribution is shared by
all stages and only the mean proportions are tilted.  PC-space metrics of
stage-specific cell states may get worse.  (iii) A_s is estimated from 2-3
Not AD train donors; with gain 1 its noise enters every Not AD prediction
(the EB shrinkage limits it).  (iv) Rows of stages absent from the train
donors are 0: such targets get no stage effect.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

import arms
from arms import Arm, register, register_mode, canonical_float, stage_of

NAME = 'stageanchor'
SNAP_EPS = 1e-5


def interp_weights(c3):
    """Piecewise-linear weights [B,4] over the stage knots 0,1/3,2/3,1 (no gradient)."""
    t = (c3.detach() * 3).clamp(0, 3)
    lo = torch.floor(t).clamp(max=2)
    frac = t - lo
    frac = torch.where(frac < SNAP_EPS, torch.zeros_like(frac), torch.where(frac > 1 - SNAP_EPS, torch.ones_like(frac), frac))
    lo = lo.long()
    w = torch.zeros(len(c3), 4, dtype=c3.dtype, device=c3.device)
    w.scatter_add_(1, lo[:, None], (1 - frac)[:, None])
    w.scatter_add_(1, (lo + 1)[:, None], frac[:, None])
    return w


@register
class StageAnchorArm(Arm):
    name = NAME
    value_policy = 'none'
    conflicts = ('offset', 'stagedrop')

    def setup(self, ctx):
        table, info = arms.offset_init(ctx.X, ctx.by, stage_of(ctx.raw_c))
        self.table = torch.as_tensor(table, dtype=torch.float32)  # CPU; becomes a generator buffer
        self.info = {'anchor': 'round-1 offset_init table (EB-shrunk train stage log-ratios), fixed', 'gain_init': 1.0,
                     'mlp_stage_input': 'removed (slot 3 zeroed in the FiLM input)', 'anchor_info': info}

    def generator_class(self, ctx, cls):
        table = self.table.clone()

        class StageAnchorGenerator(cls):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self.register_buffer('stage_anchor', table.clone())
                self.stage_gain = nn.Parameter(torch.ones(4))      # constant init, no RNG draw

            def film(self, z, c):
                c0 = c.clone()
                c0[:, 3] = 0
                return super().film(z, c0)

            def logits(self, h, z, c):
                w = interp_weights(c[:, 3]).to(self.stage_gain.dtype)
                off = (w * self.stage_gain) @ self.stage_anchor
                return super().logits(h, z, c) + off[:, None, :]
        return StageAnchorGenerator

    def init_generator(self, ctx, G):
        if not torch.equal(G.stage_anchor, self.table):
            raise ValueError('stage anchor buffer differs from the train table')
        G.stage_gain.fill_(1.0)

    def generator_grad_params(self, G):
        return [G.stage_gain]

    def state(self):
        return {'anchor': self.table.clone(), 'gain_init': 1.0}


def _anchorgain_plan(ctx, value, target, rng, states):
    if NAME not in {arms.flag_name(f) for f in ctx.flags}:
        raise ValueError('anchorgain applies only to stageanchor fits')
    return np.repeat(target.default_cond[None], target.M, 0), ['final'] * target.M


def _anchorgain_prepare(ctx, value, state_name, G, P):
    G.stage_gain.mul_(float(value))


register_mode('anchorgain', _anchorgain_plan, value_policy='required',
              parse_value=lambda t: canonical_float(t, lo=0.0, hi=1.0), prepare=_anchorgain_prepare,
              doc='stageanchor fits: every stage gain multiplied by v at generation (final state)')
