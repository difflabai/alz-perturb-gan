"""Round-2 arm ``covdrop@p`` (F1): covariate conditioning with covariate dropout during training.

Failure addressed (RESULTS.md section 2): round-1 ``covars`` puts (Male, APOE4
dose, standardised age) into the condition; nearly every train donor has a
unique tuple, the generator uses it as a donor key and a held-out donor gets
the profile of the train donor with the nearest covariates.  ``covdrop@p``
keeps the covariates but, in every training batch, replaces the covariate
vector of each batch donor independently with probability p by a "missing"
code, and generates by default with covariates missing (stage only).

Flag: ``covdrop@p``, p in (0, 1], value required (one fit-folder name per p).
Recommended screen value p = 0.5 (fixed in advance, not tuned): the default
generation uses the missing state, so it should get a large share of the
training signal (half of all batch rows, i.e. on average 16 donors per
half-update), while the other half keeps the covariate-present pathway
trained for the ``owncov`` / ``margcov`` modes.  ``covdrop@1`` drops every
row: its conditions are then bitwise base's, so the fit is bitwise equal to
``base`` (tested), which pins down that the dropout machinery changes nothing
else.

Encoding (slots 4-6; slots 0-3 as base, slot 3 = ADNC/3 = stage):

    present:  slot 4 = 2*Male - 1 in {-1, +1}
              slot 5 = (APOE4 dose - m_apoe) / s_apoe
              slot 6 = (age - m_age) / s_age
    missing:  slots 4, 5, 6 = 0, 0, 0

m/s are population mean/SD over the inner fold's TRAIN donors (m_age, s_age
are exactly round-1 covars' ctx.age_mean/age_sd).  Missing is
distinguishable from every real value because slot 4 is +-1 for a present
covariate vector and 0 only when missing; slots 5-6 at 0 are the train-mean
(mean imputation), so the missing point is also the centre of the present
distribution.  The condition stays 7 wide and covdrop uses no slot outside
4-6, so it composes with plug-ins that use slots 0-2.  With ``covbin`` in the
same arm (``covbin,covdrop@p``) covdrop keeps covbin's signed coarse codes
(slot 4 is +-1 there too) and only applies the dropout and missing code.

Whole-vector vs per-covariate dropout: whole vector.  The failure is that a
covariate tuple, and even age alone (train ages are nearly unique, see
covbin's tuple_stats), is a donor key; per-covariate dropout would still show
partial keys (age only) in most rows and would train the all-missing state
used at generation in only p^3 of the rows.  Whole-vector dropout trains
exactly two states: full covariates or none.

Same dropped vector for generator and critic: the hook returns one [B,7]
condition per half-update, which the core uses for the fake bags, the real
bags and the critic in that half-update (trainer.advance; the critic and the
generator phase draw their own independent masks because they use different
batch donors).

Random stream: the mask is ``rng.random(B) < p`` from this arm's own numpy
Generator (core hook b: SeedSequence([stream_root, 7, fold, seed,
crc32('covdrop'), 0]), checkpointed as ``arm_streams``).  It draws exactly B =
32 uniforms per batch call and never touches the 11 released streams, so the
donor sampler, cell/count/latent/penalty generators and torch global RNG
follow base's sequence (tested after several updates).

Generation: default ``generated.npz`` conditions on stage only with
covariates missing (hook e sets slots 4-6 to 0 for every target); mode
``owncov`` uses the target's own present covariates and ``margcov`` those of
random same-stage train donors (mode_margcov.py).  Nothing else differs from
base (released generator class, critic with 150 features, update rule).
Conflicts with round-1 ``covars``.
"""
from __future__ import annotations

import numpy as np
import torch

import arms
from arms import Arm, register, canonical_float
import mode_margcov  # noqa: F401  (registers owncov / margcov)

NAME = 'covdrop'


@register
class CovDropArm(Arm):
    name = NAME
    value_policy = 'required'
    conflicts = ('covars',)

    @classmethod
    def parse_value(cls, text):
        return canonical_float(text, lo=0.0, hi=1.0, lo_open=True)

    def setup(self, ctx):
        self.p = float(self.value)
        self.binned = 'covbin' in {arms.flag_name(f) for f in ctx.flags}
        dose = ctx.raw_c[:, 5]
        self.m_apoe, self.s_apoe = float(dose.mean()), float(dose.std())
        self.m_age, self.s_age = float(ctx.age_mean), float(ctx.age_sd)
        if not (self.s_apoe > 0 and self.s_age > 0):
            raise ValueError('covdrop needs positive train APOE4-dose and age SDs')
        self.info = {'p': self.p, 'dropout': 'whole covariate vector, independently per batch donor and half-update',
                     'missing_code': 'slots 4-6 = 0 (present slot 4 is +-1)', 'binned_by_covbin': self.binned,
                     'encoding': ('covbin codes' if self.binned else
                                  'slot4 = 2*Male-1, slot5 = (APOE4 dose - m)/s, slot6 = (age - m)/s (train donors)'),
                     'apoe_mean_sd': [self.m_apoe, self.s_apoe], 'age_mean_sd': [self.m_age, self.s_age],
                     'default_generation': 'covariates missing (stage only)'}

    def conditions(self, ctx, raw_c, cond):
        out = cond.copy()
        if self.binned:
            if not np.isin(cond[:, 4], (-1.0, 1.0)).all():
                raise ValueError('covdrop with covbin: slot 4 is not the covbin sex code')
            return out
        if np.any(cond[:, 4:7] != 0):
            raise ValueError('covdrop: condition slots 4-6 already used by another arm')
        male = raw_c[:, 4]
        if not np.isin(male, (0.0, 1.0)).all():
            raise ValueError('Male must be 0/1')
        out[:, 4] = (2 * male - 1).astype(np.float32)
        out[:, 5] = ((raw_c[:, 5] - self.m_apoe) / self.s_apoe).astype(np.float32)
        out[:, 6] = ((raw_c[:, 6] - self.m_age) / self.s_age).astype(np.float32)
        return out

    def batch_condition(self, ctx, c, ds, rng, phase):
        if not bool((c[:, 4].abs() == 1).all()):
            raise ValueError('covdrop: slot 4 of the training condition is not +-1 (another arm overwrote slots 4-6?)')
        drop = torch.as_tensor(rng.random(len(ds)) < self.p, device=c.device)
        out = c.clone()
        out[drop, 4:7] = 0
        return out

    def generation_condition(self, ctx, raw_c, cond, is_heldout):
        out = cond.copy()
        out[:, 4:7] = 0
        return out

    def state(self):
        return {'p': self.p, 'binned': self.binned, 'apoe_mean_sd': [self.m_apoe, self.s_apoe], 'age_mean_sd': [self.m_age, self.s_age]}
