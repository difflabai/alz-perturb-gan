"""Round-3 pre-declared generation-time stage handling (R6): modes ``stagehalf`` and ``stageblind``.

Declared on 2026-11-11, BEFORE any round-3 score exists.  Registered as generation modes (r3/arms.py
hook e) and imported by ``arm_nostage.py`` so that the plug-in scan finds them (mode files are found
only through ``arm_*.py``).  No retraining; works on round-1, round-2 and round-3 fits.

``stagehalf[@R]``  - THE pre-declared default stage-shrink read-out for target T1 (PROTOCOL_R3.md
    "Additions" item 4: stage-conditional arms are read with stagehalf and, separately,
    stage-blind).  Conditions = ``condmix@0.5`` (arm_condshrink.py): of the M = 32 generated donors
    of a target, n = floor(0.5 M + 0.5) = 16 get stages apportioned by the TRAIN-donor stage
    frequencies (largest remainder, ties to the lower stage; ``ctx.target_conditions`` of the
    target's raw_c with only ADNC replaced, so every arm hook applies), the other 16 keep the
    target's own stage.  The mean generated profile is then about
        0.5 P(t | own stage) + 0.5 sum_s pi_s P(t | stage s),
    i.e. the model's stage deviation shrunk by one half toward its own train-weighted stage marginal.
``stageblind[@R]`` - the stage-blind companion (= condmix@1 conditions: all 32 generated donors get
    the train stage mixture; the prediction does not depend on the target's stage).  Its pb_rmse is
    a calibration result, NOT a stage model (evaluator: label 'stage-blind', blank stage metrics).

Read-out state R (which model weights generate):
    (none)   the PRIMARY read-out of PROTOCOL_R3 item 4, fixed in advance:
             t.ema fit -> 'final' (= native EMA weights); t.anneal fit -> 'final' (final weights);
             any other fit with snapshots in the last 25% of training (constant learning rates,
             round 2 and round 3) -> the ``snapens`` states (those snapshots + raw final);
             otherwise (round-1 fits) -> 'final'.
    final    'final' only (stagehalf@final is bitwise ``condmix@0.5``; stageblind@final is bitwise
             ``condmix@1``: same conditions, same order, same streams).
    raw      the raw live final weights (differs from final only for t.ema fits).
    snapens  the snapens states (raises if the fit has none).
    ema<D>   shadow EMA of the final update at decay D, e.g. ``stagehalf@ema0.9995`` (round-3 fits).

Interleaving with several states.  With one state the layout is exactly condmix's (mixed donors
last).  With several states (snapens: contiguous runs of 6,6,5,5,5,5 donors on a 20,000-update
round-3 fit; 7,7,6,6,6 on round-2 fits) putting the 16 mixed donors last would give them only the
latest states; instead the mixed positions are taken round-robin over the state runs (first slot of
every run, then the second slot of every run, ...; for stagehalf only every second slot of a run is
eligible) and the apportioned stages, sorted ascending, are assigned in that round-robin order, so
every state carries half own-stage and half mixed donors (up to rounding) and each mixed stage is
spread over the states.  The state of every generated donor is exactly the snapens one.

Why one half, and why this was chosen (selection history, stated plainly): the choice of a 0.5
shrink as the T1 default was informed by round-1 and round-2 HELD-OUT results on the same 37
development donors: generation-time shrinkage helped every stage-conditional arm (round 2, seed 0:
condmix@0.5 0.641 vs raw 0.690 for base, 0.586 vs 0.651 for genemean; condmix@1 better still), the
round-1 train-chosen lambda was 0.65-0.9, and the stage task's decomposition of the best round-2
read-outs (stage_failures.py decomp) found the stage-deviation term to be the largest part of the
remaining gap, mostly seed-specific.  condmix@1 is better on those held-out numbers but discards the
target's stage entirely; 0.5 keeps a stage model while removing about three quarters of the
excess stage error in MSE (the error is quadratic in the shrink factor).  Any round-3 result read
with stagehalf is therefore selection-conditioned development evidence, like everything else in
round 3.

Evaluator note: ``r3/evaluate.py`` must label ``stagehalf*`` units 'stage-mixture' and
``stageblind*`` units 'stage-blind' (the round-2 ``stage_blind_mode`` only knows condmix /
condunk / condshrink / anchorgain).
"""
from __future__ import annotations

import numpy as np

import arms
from arms import register_mode, canonical_float, snapshot_states, raw_final_state
import arm_condshrink as MC

HALF = 0.5
READOUTS = ('final', 'raw', 'snapens')


def parse_readout(text):
    t = str(text)
    if t in READOUTS:
        return t, t
    if t.startswith('ema'):
        d, dt = canonical_float(t[3:], lo=0.0, lo_open=True, hi=1.0, hi_open=True)
        return ('ema', d), 'ema' + dt
    raise ValueError(f'read-out {text!r}: want final, raw, snapens or ema<D> (e.g. ema0.9995)')


def primary_states(ctx, target, states):
    """Per-donor state names of the PRIMARY read-out (PROTOCOL_R3 item 4; module docstring)."""
    topt = getattr(ctx, 'topt', None)
    if topt is not None and (topt.ema is not None or topt.anneal is not None):
        return ['final'] * target.M
    if snapshot_states(states) and target.updates is not None:
        try:
            return list(arms._snapens_plan(ctx, None, target, None, states)[1])
        except ValueError:
            pass
    return ['final'] * target.M


def readout_states(ctx, value, target, states):
    if value is None:
        return primary_states(ctx, target, states)
    if value == 'final':
        return ['final'] * target.M
    if value == 'raw':
        return list(arms._raw_plan(ctx, None, target, None, states)[1])
    if value == 'snapens':
        return list(arms._snapens_plan(ctx, None, target, None, states)[1])
    if isinstance(value, tuple) and value[0] == 'ema':
        return list(arms._ema_plan(ctx, value[1], target, None, states)[1])
    raise ValueError(f'unknown read-out {value!r}')


def _runs(names):
    out, a = [], 0
    for m in range(1, len(names) + 1):
        if m == len(names) or names[m] != names[a]:
            out.append(list(range(a, m)))
            a = m
    return out


def mixed_conditions(ctx, target, lam, names):
    """(cond float32 [M,7], stage per generated donor) for a stage mixture of weight lam over the plan's states."""
    M = target.M
    if len(set(names)) == 1:
        cond, stages = MC.condmix_conditions(ctx, target, lam)        # exactly condmix's layout
        return cond, stages
    n = MC.n_mixed(lam, M)
    k = MC.apportion(n, MC.train_stage_proportions(ctx))
    mixed_stages = [s for s in range(4) for _ in range(k[s])]
    runs = _runs(names)
    step = 1 if n >= M else 2                                         # stagehalf: every second slot of a run
    eligible = [[r[i] for i in range(len(r) - 1, -1, -step)][::-1] for r in runs]   # last slot of a run first eligible
    order = []
    depth = max(len(e) for e in eligible)
    for j in range(depth):
        for e in eligible:
            if j < len(e):
                order.append(e[j])
    if len(order) < n:   # odd run lengths: fill from the remaining slots, round-robin again
        rest = [[i for i in r if i not in set(order)] for r in runs]
        for j in range(max(len(e) for e in rest)):
            for e in rest:
                if j < len(e):
                    order.append(e[j])
    pos = sorted(order[:n])
    assign = dict(zip(order[:n], mixed_stages))
    stages = np.full(M, target.stage, dtype=int)
    cond = np.repeat(target.default_cond[None], M, 0).astype(np.float32)
    if n:
        raws = np.repeat(np.asarray(target.raw_c, dtype=np.float64)[None], 4, 0)
        raws[:, 3] = np.arange(4) / 3.0
        per_stage = ctx.target_conditions(raws, np.full(4, bool(target.is_heldout)))
        for m in pos:
            stages[m] = assign[m]
            cond[m] = per_stage[assign[m]]
    return cond, stages


def _plan(lam):
    def plan(ctx, value, target, rng, states):
        names = readout_states(ctx, value, target, states)
        cond, _ = mixed_conditions(ctx, target, lam, names)
        return cond, names
    plan.__module__ = __name__
    return plan


stagehalf_plan = _plan(HALF)
stageblind_plan = _plan(1.0)

register_mode('stagehalf', stagehalf_plan, value_policy='optional', parse_value=parse_readout,
              doc='PRE-DECLARED T1 stage shrink: condmix@0.5 conditions (16 of 32 donors get the train stage mixture) on the '
                  'primary read-out states (snapens for constant-lr fits with snapshots, final for t.anneal / t.ema / round-1 fits); '
                  '@final/@raw/@snapens/@ema<D> fix the state')
register_mode('stageblind', stageblind_plan, value_policy='optional', parse_value=parse_readout,
              doc='STAGE-BLIND companion of stagehalf: condmix@1 conditions on the same read-out states')


def stage_label(mode):
    """'stage-mixture' for stagehalf*, 'stage-blind' for stageblind*, else None (for the evaluator)."""
    name = str(mode or '').split('@', 1)[0]
    return {'stagehalf': 'stage-mixture', 'stageblind': 'stage-blind'}.get(name)
