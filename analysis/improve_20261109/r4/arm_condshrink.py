"""Generation-time stage shrinkage modes (round-2 fix for F3; no retraining).

Registers two generation modes (r2/arms.py hook e); no arm flag.  Both work on
round-1 fits (``r2/generate.py --fit ... --out ...``, final state only) and on
round-2 fits.  They act on the stage slot (3, ADNC/3) of the per-target
condition only; every other slot (e.g. covars 4-6) stays the target's own.

``condshrink@lam`` (0 <= lam <= 1)
    Every generated donor of target t is conditioned on
        c3' = (1 - lam) * c3_t + lam * cbar,
    cbar = mean over the fold's TRAIN donors of ADNC/3 (``ctx.raw_c``, donor
    weighted; float32).  lam = 0 is the default generation, lam = 1 gives every
    target the same off-grid stage value.  Model states: final only.
    Why it should help: the generators' predicted stage deviations are 3-8x
    larger than the real train stage deviations and uncorrelated with held-out
    ones (RESULTS.md section 3), so moving every target toward a common stage
    input removes most of that between-stage spread.
    What can go wrong: the generator was trained only at the four values
    0, 1/3, 2/3, 1; between them its output is an uncontrolled interpolation
    (the donor MLP is smooth, but nothing ties c3 = cbar to the stage
    marginal).  The shrunk prediction at lam = 1 is NOT the model's marginal
    over stages; it can carry its own bias.  Refused for fits with the round-1
    ``offset`` flag (OffsetCountGenerator selects its table row by
    round(3*c3), so an off-grid c3 silently picks a different stage row).
    For ``stageanchor`` fits the anchored path interpolates linearly between
    the stage rows (see arm_stageanchor.py), so the mode is defined there too.

``condmix@lam`` (0 <= lam <= 1)
    On-grid alternative.  Of the M = 32 generated donors of each target the
    first M - n keep the target's own stage, the last n = floor(lam*M + 0.5)
    get stage s in proportion to the TRAIN-donor stage frequencies pi_s
    (deterministic largest-remainder apportionment, ties to the lower stage;
    donors sorted by stage; no random draws).  Their condition is
    ``ctx.target_conditions`` of the target's raw_c with only ADNC replaced
    by s/3, so every arm hook applies as in default generation.  The mean
    generated profile is then (up to the rounding of n and Monte Carlo noise)
        (1 - lam) * P(t | own stage) + lam * sum_s pi_s P(t | stage s),
    i.e. linear shrinkage of the model's stage deviation toward the model's
    own train-weighted stage marginal, without off-grid inputs.  Works for
    every arm, including round-1 ``offset`` and ``covars`` fits.
    What can go wrong: the generated donor population of a target becomes a
    stage mixture, so donor-level distribution metrics (spread, coverage)
    describe a mixture, not the target's stage; and the model's marginal
    still carries the generator's per-gene level bias (only a level
    recalibration removes that, see the posthoc task).

The first M - n generated donors of condmix use exactly the default
conditions and streams, so with CHUNK = 8 the chunks that contain no mixed
donor are bitwise equal to generated.npz (common random numbers).

Choice of lam: fixed in advance (0.5 and 1) or chosen by train-only
(in-sample) pb_rmse on the fold's train targets; see r2/mode_condshrink.py,
which runs both modes on round-1 fits and reports held-out results.

STAGE-BLIND MODES (repair 2026-11-10, reviews FID-3 / S4).  ``condmix@1``
(n = floor(M + .5) = M: every generated donor of every target gets the same
train-proportion stage mixture; the target's own stage is never used) and
``condshrink@1`` (every target at c3 = cbar) make predictions that do not
depend on the target's stage: apart from library sizes, covariates and Monte
Carlo noise they are the model's stage marginal, i.e. a generator-made
grand-mean predictor.  Their pb_rmse is a CALIBRATION result (compare with
the grand-mean baseline and direct:train_mean), NOT evidence that stage
inflation (F3) is fixed, and their stage metrics (xfit, DAM, contrasts,
within-stage spreads) are degenerate.  ``evaluate.stage_blind_mode`` flags
them; the evaluator masks their stage metrics and labels them 'stage-blind'.
Partial values (condmix@0.5) keep half of the stage effect and are the ones
that test stage shrinkage; their donor-level distribution metrics describe a
mixture (flagged 'stage-mixture').

Selection history (repair, review S3): lam = 1 is the best round-1
generation-only value on HELD-OUT donors of the same 37 donors (the
train-chosen lam was 0.65-0.9), and condmix was preferred over condshrink on
held-out genemean results.  condmix@1 and condmix@0.5 are therefore both run
as pre-specified candidates on every round-2 screen arm and both reported; a
round-2 result for either is selection-conditioned development evidence.
"""
from __future__ import annotations

import numpy as np

import arms
from arms import canonical_float, register_mode, stage_of


def train_mean_stage_value(ctx):
    """cbar = mean over train donors of ADNC/3 (float32).  Round 4: cohort-0 (SEA-AD) train donors only (ctx.target_idx)."""
    return np.float32(np.asarray(ctx.raw_c, dtype=np.float64)[_target_rows(ctx), 3].mean())


def _target_rows(ctx):
    """Round-4 edit: the train donors that are generation targets (cohort 0; all of them for round-1 caches)."""
    return np.asarray(getattr(ctx, 'target_idx', np.arange(len(ctx.raw_c))), dtype=np.int64)


def train_stage_proportions(ctx):
    st = stage_of(np.asarray(ctx.raw_c)[_target_rows(ctx)])   # round 4: SEA-AD train donors only
    return np.array([(st == s).sum() for s in range(4)], dtype=np.float64) / len(st)


def apportion(n, pi):
    """Largest-remainder apportionment of n donors over stages (ties to the lower stage)."""
    pi = np.asarray(pi, dtype=np.float64)
    q = n * pi
    k = np.floor(q).astype(int)
    rem = q - k
    left = n - int(k.sum())
    order = sorted(range(len(pi)), key=lambda s: (-rem[s], s))
    for s in order[:left]:
        k[s] += 1
    assert k.sum() == n and (k >= 0).all()
    return k


def n_mixed(lam, M):
    return int(np.floor(lam * M + 0.5))


def _refuse_offset(ctx, mode):
    if 'offset' in arms.round1_names(ctx.flags):
        raise ValueError(f'{mode} is not defined for round-1 offset fits (stage row chosen by round(3*c3)); use condmix')


def condshrink_plan(ctx, value, target, rng, states):
    _refuse_offset(ctx, 'condshrink')
    lam = float(value)
    cond = np.repeat(target.default_cond[None], target.M, 0).astype(np.float32)
    cbar = train_mean_stage_value(ctx)
    c3 = cond[:, 3].astype(np.float64)
    cond[:, 3] = ((1.0 - lam) * c3 + lam * float(cbar)).astype(np.float32)
    if lam == 0.0:
        cond = np.repeat(target.default_cond[None], target.M, 0).astype(np.float32)
    return cond, ['final'] * target.M


def condmix_conditions(ctx, target, lam):
    """(cond float32 [M,7], stage per generated donor int [M]) for condmix@lam."""
    M = target.M
    n = n_mixed(lam, M)
    k = apportion(n, train_stage_proportions(ctx))
    stages = [target.stage] * (M - n) + [s for s in range(4) for _ in range(k[s])]
    cond = np.repeat(target.default_cond[None], M, 0).astype(np.float32)
    if n:
        raws = np.repeat(np.asarray(target.raw_c, dtype=np.float64)[None], 4, 0)
        raws[:, 3] = np.arange(4) / 3.0
        per_stage = ctx.target_conditions(raws, np.full(4, bool(target.is_heldout)))
        for m in range(M - n, M):
            cond[m] = per_stage[stages[m]]
    return cond, np.array(stages)


def condmix_plan(ctx, value, target, rng, states):
    cond, _ = condmix_conditions(ctx, target, float(value))
    return cond, ['final'] * target.M


register_mode('condshrink', condshrink_plan, value_policy='required',
              parse_value=lambda t: canonical_float(t, lo=0.0, hi=1.0),
              doc='stage slot moved toward the train-donor mean ADNC/3: c3 <- (1-lam) c3 + lam cbar (final state)')
register_mode('condmix', condmix_plan, value_policy='required',
              parse_value=lambda t: canonical_float(t, lo=0.0, hi=1.0),
              doc='last floor(lam M + .5) generated donors get stages apportioned by train stage frequencies (final state)')
