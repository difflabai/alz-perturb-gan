"""Round-3 arm ``nostage`` (R6): a stage-blind model (generator AND critic do not see stage).

Flag ``nostage`` (no value).  Also imports ``mode_stagedefault`` (modes ``stagehalf``,
``stageblind``) and ``mode_biascal`` (mode ``biascal``) so that the plug-in scan registers them.

What it changes (application only; released update rule, recipe, optimizers, penalty and policy
calls unchanged; no random number drawn, so every training stream is base's):

    condition vector = 0 (all 7 slots) for every train donor and every generation target.

The condition is shared by the generator's donor FiLM MLP and the critic (trainer: ``ctx.conditions``
feeds both, ``Context.target_conditions`` feeds generation), so

* the generator learns ONE donor distribution, the stage-pooled one (the latent z is the only donor
  input), and
* the critic compares stage-pooled real bags with stage-pooled fake bags (it no longer conditions
  on stage).  Donors are drawn uniformly as released, so the pooled target is the train-donor
  mixture, i.e. stages weighted by their train donor counts.

Why (R6 diagnosis, r3/stage_failures.py): (1) on the whole 2,000-gene panel the train stage effect
is indistinguishable from a stage-label permutation null in all three folds (empirical-Bayes prior
variance tau2 at or near 0; permutation p = 0.38, 0.91, 0.51), so a stage path can only add noise;
(2) the generators' stage deviations are 3-8x the real ones and mostly seed-specific; in the best
round-2 read-outs the stage-deviation term is the largest part of the gap to the grand mean;
(3) every stage-blind READ-OUT of a stage-conditional model (condmix@1) beat its stage-conditional
read-out, but it is generated from a model whose capacity and critic signal were partly spent on a
stage path; ``nostage`` removes stage inflation at the source and tests whether a generator freed
from the stage path fits the donor and cell distribution better (T2 metrics, PC/cell energies).
(4) Not the stageanchor design: round-2 ``stageanchor`` made the GENERATOR (effectively)
stage-blind while the critic still conditioned on stage; its critic gap rose to 2-2.5x base and its
per-gene level bias exploded in 2 of 3 folds (stage_failures.py anchor).  Here both sides are blind,
so the critic cannot reward stage structure the generator cannot produce.

What it gives up: every stage contrast (it is a stage-blind model; the evaluator must list it with
the stage-blind units, PROTOCOL_R3 "Additions" item 3) and any genuine stage-specific cell state.
Read-out (PROTOCOL_R3 item 4): as trained (``snapens`` for constant learning rates).  Stage modes
(condmix, stagehalf, ...) are no-ops on it (all conditions are 0 whatever the stage).

Conflicts: every other condition or stage arm (round-1 ``offset``, ``covars``; ``covdrop``,
``covbin``, ``stageanchor``, ``stagedrop``, ``fixedstage``).  The conditions hook additionally
refuses to run after a plug-in that already wrote a nonzero slot other than the stage slot.
"""
from __future__ import annotations

import numpy as np

import arms
from arms import Arm, register
import mode_stagedefault  # noqa: F401  (registers stagehalf / stageblind)
import mode_biascal  # noqa: F401  (registers biascal)

NAME = 'nostage'


@register
class NoStageArm(Arm):
    name = NAME
    value_policy = 'none'
    conflicts = ('offset', 'covars', 'covdrop', 'covbin', 'stageanchor', 'stagedrop', 'fixedstage')

    def setup(self, ctx):
        st = arms.stage_of(ctx.raw_c)
        self.info = {'condition': 'all 7 slots 0 for generator and critic (stage-blind model)',
                     'stage_blind': True, 'train_donors_per_stage': np.bincount(st, minlength=4).tolist(),
                     'pooled_target': 'uniform train-donor mixture (released donor sampling)'}

    def conditions(self, ctx, raw_c, cond):
        other = np.delete(cond, 3, axis=1)
        if np.any(other != 0):
            raise ValueError('nostage: a condition slot other than the stage is already used by another arm')
        return np.zeros_like(cond, dtype=np.float32)

    def state(self):
        return {'stage_blind': True}
