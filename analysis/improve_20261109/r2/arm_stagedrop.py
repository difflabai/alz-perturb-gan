"""Arm ``stagedrop@p``: stage-condition dropout in training (round-2 fix for F3), mode ``condunk@lam``.

Flag ``stagedrop@p`` (value required, 0 < p < 1, canonical '%g'; suggested
screen value p = 0.5).  Conflicts with round-1 ``offset`` (its table row is
chosen by round(3*c3)) and with ``stageanchor``.

Training (application only; released update rule unchanged).  Hook (b):
in each half-update ('critic' batch and 'generator' batch separately) every
batch donor independently gets, with probability p (drawn from the arm's own
numpy stream, ``ctx.arm_seed('stagedrop', 0)``; the 11 released streams are
untouched), the "stage unknown" condition
    slot 3 = cbar  (mean over the fold's TRAIN donors of ADNC/3, float32)
    slot 2 = 1     (unknown flag; slot 2 is 0 in every other condition)
instead of its own; other slots (covars 4-6) are kept.  The same modified
condition goes to the generator (fake bags) and to the critic (real and fake
bags) of that half-update, so for "unknown" rows the critic compares the
generator's unknown-condition bags with real bags of all stages in train
proportions: the generator learns the stage marginal under the unknown
condition and the stage conditionals under the known conditions, with the
same networks.  Hook (a) checks that slot 2 is 0 in the train/target
conditions (no other plug-in may use it together with this arm).

Generation.  Default ``generated.npz``: the known stage (as trained, hook e
identity).  Mode ``condunk@lam`` (0 <= lam <= 1, stagedrop fits only): of the
M = 32 generated donors of each target the last floor(lam*M + 0.5) get the
unknown condition (final state), so the mean prediction is
(1 - lam) * P(t | stage) + lam * P(t | unknown), a shrinkage toward the
model's trained marginal.  ``condmix``/``condshrink`` (arm_condshrink.py)
also apply.

Why it should help: on round-1 fits the stage deviations are 3-8x the real
train deviations, mostly seed-specific noise (base ~60%, genemean ~90%) plus
a reproducible part that does not replicate on held-out donors.  With
dropout, (i) generation can shrink toward a marginal the network was
actually trained to produce (instead of an off-grid stage value, as in
``condshrink``, or an average of four inflated conditionals, as in
``condmix``), and (ii) a share p of the critic's updates carries no stage
information, so less of the critic's capacity goes to per-stage
discrimination from 2-12 donors per stage.

What could go wrong: (i) dropout does not by itself shrink the known-stage
conditionals: they are still trained against the same few donors per stage,
now with a fraction (1-p) of the batches, which can make them noisier, not
smaller; the benefit then comes only from condunk at generation. (ii) the
unknown flag is a second input path: the generator can learn an arbitrary
difference between known and unknown outputs (e.g. a level shift) that the
critic, seeing only marginal real bags, cannot fully pin down with 32-bag
batches. (iii) default generation at p = 0.5 sees half as many
stage-labelled batches as base, so in-sample stage fit (and the DAM
direction) can weaken.  The pre-registered comparison is condunk@1 and
condunk@0.5 against base, condmix and the grand mean on held-out pb_rmse.
"""
from __future__ import annotations

import numpy as np
import torch

import arms
from arms import Arm, register, register_mode, canonical_float

NAME = 'stagedrop'
UNKNOWN_SLOT = 2


@register
class StageDropArm(Arm):
    name = NAME
    value_policy = 'required'
    conflicts = ('offset', 'stageanchor')

    @classmethod
    def parse_value(cls, text):
        return canonical_float(text, lo=0.0, hi=1.0, lo_open=True, hi_open=True)

    def setup(self, ctx):
        self.p = float(self.value)
        self.cbar = np.float32(np.asarray(ctx.raw_c, dtype=np.float64)[:, 3].mean())
        self.info = {'p': self.p, 'unknown_condition': {'slot3': float(self.cbar), 'slot2': 1.0},
                     'stream': 'SeedSequence([stream_root,7,fold,seed,crc32(stagedrop),0]) uniform per batch donor'}

    def conditions(self, ctx, raw_c, cond):
        if np.any(cond[:, UNKNOWN_SLOT] != 0):
            raise ValueError('stagedrop needs condition slot 2 free (another arm uses it)')
        return cond.copy()

    def unknown(self, cond):
        """Unknown-stage version of conditions (numpy [n,7] or tensor [B,7])."""
        out = cond.clone() if isinstance(cond, torch.Tensor) else np.array(cond, dtype=np.float32, copy=True)
        out[:, 3] = float(self.cbar)
        out[:, UNKNOWN_SLOT] = 1.0
        return out

    def batch_condition(self, ctx, c, ds, rng, phase):
        drop = torch.as_tensor(rng.random(len(ds)) < self.p, device=c.device)
        return torch.where(drop[:, None], self.unknown(c), c)

    def state(self):
        return {'p': self.p, 'cbar': float(self.cbar)}


def _arm(ctx):
    for a in ctx.plugins:
        if a.name == NAME:
            return a
    raise ValueError('condunk applies only to stagedrop fits (the unknown condition was never trained otherwise)')


def condunk_plan(ctx, value, target, rng, states):
    a = _arm(ctx)
    M = target.M
    n = int(np.floor(float(value) * M + 0.5))
    cond = np.repeat(target.default_cond[None], M, 0).astype(np.float32)
    if n:
        cond[M - n:] = a.unknown(cond[M - n:])
    return cond, ['final'] * M


register_mode('condunk', condunk_plan, value_policy='required', parse_value=lambda t: canonical_float(t, lo=0.0, hi=1.0),
              doc='stagedrop fits: last floor(lam M + .5) generated donors get the trained stage-unknown condition (final state)')
