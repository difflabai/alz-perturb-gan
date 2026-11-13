"""Round-2 generation modes for covariate arms (F1): ``margcov`` and ``owncov``.

Imported by ``arm_covdrop.py`` and ``arm_covbin.py`` (the registry finds a mode
by importing ``arm_<mode>.py`` or, failing that, every ``arm_*.py``; this file
is not ``arm_*`` so the two arm files import it).  Both modes work for every
fit whose flags contain a covariate arm: round-1 ``covars`` (including the
round-1 fits under ``runs/improve_20261109/fits/``, written with ``--out``),
``covdrop@p``, ``covbin`` and ``covbin,covdrop@p``.  Fits without a covariate
flag are refused.

Covariate slots.  Every covariate arm keeps the covariates in condition slots
4, 5, 6 (round-1 ``covars``: Male, APOE4 dose, standardised age; ``covdrop``
and ``covbin``: their own signed encodings, see those files).  The "present"
covariate value of a donor is the hook-(a) condition chain
``ctx.condition_matrix(raw_c)`` (round-1 ``build_conditions`` + plug-in
``conditions`` hooks), i.e. what the covariate rows looked like in training
before any dropout.  A mode condition is the target's DEFAULT generation
condition (``target.default_cond``: the full hook (a) + (e) chain, so every
other arm's generation-time hooks still apply) with slots 4-6 replaced by
present covariate values.  Each plan asserts that changing the covariate
columns of raw_c changes no slot other than 4-6 of ``condition_matrix``
(i.e. no other arm reads the covariates), otherwise it refuses.

owncov
    All M generated donors get the target donor's OWN present covariates.  For
    round-1 ``covars`` and for ``covbin`` this equals default generation
    (generated.npz) bitwise; for ``covdrop`` (whose default marks covariates
    missing) it is the own-covariate counterpart.

margcov
    Covariates marginalised over same-stage TRAIN donors: generated donor m of
    target t takes the raw covariates (Male, APOE4 dose, age = raw_c columns
    4-6) of train donor ``pool[j_m]``, where ``pool`` = the fold's train
    donors (cache order) with the target's ADNC stage, excluding the target
    itself when the target is a train donor (leave-self-out, so a train target
    never sees its own donor key), and ``j_m = rng.integers(len(pool), size=M)``
    with the registry's mode stream ``SeedSequence([20261109, 5, fold,
    dev_index, crc32('margcov')])``.  The draws depend only on fold and target
    (common random numbers across arms, seeds and fits of the same fold); the
    generation streams and library draws are the default ones (registry).  The
    picked covariates are then encoded by the fit's own arm
    (``condition_matrix``), so the same donor draws feed covars (raw 0/1, dose,
    z-age), covdrop (signed, z-scored) and covbin (bins).  Stage is the
    target's own (pool is same-stage).  ``margcov_picks`` reproduces the picked
    donor ids for an evaluator.

Neither mode retrains or changes model state; both use the 'final' state.
"""
from __future__ import annotations

import numpy as np

import arms
from arms import register_mode, stage_of

COV_SLOTS = slice(4, 7)
RAW_COV = slice(4, 7)       # raw_c columns: Male, APOE4 dose, raw age
COV_ARMS = ('covars', 'covbin', 'covdrop')


def covariate_flags(flags):
    return sorted({arms.flag_name(f) for f in flags} & set(COV_ARMS))


def _require_cov(ctx, mode):
    if not covariate_flags(ctx.flags):
        raise ValueError(f'mode {mode} needs a covariate arm ({"/".join(COV_ARMS)}); fit flags {list(ctx.flags)}')


def _present(ctx, target, raw):
    """Present-covariate conditions for raw rows (hook a chain), checking that only slots 4-6 depend on covariates."""
    raw = np.asarray(raw, dtype=np.float64)
    pres = ctx.condition_matrix(raw)
    own = ctx.condition_matrix(np.asarray(target.raw_c, dtype=np.float64)[None])
    other = np.ones(7, bool)
    other[COV_SLOTS] = False
    if not np.array_equal(pres[:, other], np.repeat(own[:, other], len(raw), 0)):
        raise ValueError('another arm reads the covariate columns of raw_c; covariate modes cannot isolate slots 4-6')
    return pres


def margcov_pool(train_ids, train_stages, target_donor, target_stage, is_heldout):
    pool = np.flatnonzero(np.asarray(train_stages) == int(target_stage))
    if not is_heldout:
        pool = pool[np.asarray(train_ids)[pool] != target_donor]
    if len(pool) == 0:
        raise ValueError(f'margcov: no other same-stage train donor for {target_donor} (stage {target_stage})')
    return pool


def margcov_picks(train_ids, train_raw_c, target_donor, target_stage, is_heldout, fold, dev_index, M=32):
    """Train-donor indices (cache order) whose covariates margcov gives the M generated donors of one target."""
    pool = margcov_pool(train_ids, stage_of(train_raw_c), target_donor, target_stage, is_heldout)
    rng = arms.mode_rng(fold, dev_index, 'margcov')
    return pool[rng.integers(len(pool), size=M)]


def _owncov_plan(ctx, value, target, rng, states):
    _require_cov(ctx, 'owncov')
    pres = _present(ctx, target, target.raw_c[None])
    cond = np.repeat(target.default_cond[None], target.M, 0).astype(np.float32)
    cond[:, COV_SLOTS] = pres[0, COV_SLOTS]
    return cond, ['final'] * target.M


def _margcov_plan(ctx, value, target, rng, states):
    _require_cov(ctx, 'margcov')
    pool = margcov_pool(ctx.ids, stage_of(ctx.raw_c), target.donor, target.stage, target.is_heldout)
    pick = pool[rng.integers(len(pool), size=target.M)]
    raw = np.repeat(np.asarray(target.raw_c, dtype=np.float64)[None], target.M, 0)
    raw[:, RAW_COV] = ctx.raw_c[pick][:, RAW_COV]
    pres = _present(ctx, target, raw)
    cond = np.repeat(target.default_cond[None], target.M, 0).astype(np.float32)
    cond[:, COV_SLOTS] = pres[:, COV_SLOTS]
    return cond, ['final'] * target.M


register_mode('owncov', _owncov_plan,
              doc="covariate arms: every generated donor gets the target's own present covariates (slots 4-6)")
register_mode('margcov', _margcov_plan,
              doc='covariate arms: generated donor m gets the covariates of a random same-stage train donor '
                  '(leave-self-out for train targets; mode stream, common random numbers across arms)')
