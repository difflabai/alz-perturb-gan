"""Round-2 arm ``covbin`` (F1): coarse covariates so that covariate tuples are shared by several donors.

Failure addressed (RESULTS.md section 2, F1): round-1 ``covars`` conditions on
(Male, APOE4 dose, standardised age); 22-24 of the 24-25 train donors of a
fold have a unique tuple, so the covariates act as a donor key (generator
routes 98-99% of its donor-layer variance through them, identification rate
0.85).  ``covbin`` replaces them by three coarse codes:

    slot 4 = +1 male / -1 female                      (2*Male - 1)
    slot 5 = +1 APOE4 carrier (dose >= 1) / -1 non-carrier
    slot 6 = age tertile - 1 in {-1, 0, +1}

Age tertile thresholds ``t1, t2 = np.quantile(train ages, [1/3, 2/3])``
(numpy default linear interpolation) over the inner fold's TRAIN donors
only; tertile = (age > t1) + (age > t2), applied unchanged to held-out
donors.  The raw age is the cache's raw_c column 6 (bench_core encoding).
Slots 0-3 are as in base (slot 3 = ADNC/3 = stage); nothing else changes
(released generator, critic, update rule, streams: covbin consumes no random
numbers, so every training stream is bitwise base's).  With 2 x 2 x 3 = 12
possible tuples and about 25 train donors, tuples are shared; the measured
counts per fold are in ``tuple_stats`` (config.json r2_arms.covbin.info) and
in the structured report of the F1 task.  The signed encoding (no 0 for a
present value in slot 4) is shared with ``covdrop``: ``covbin,covdrop@p``
applies covariate dropout to the binned covariates (covdrop then encodes
"missing" as 0 in slots 4-6, see arm_covdrop.py).

Generation: default ``generated.npz`` conditions on the target's own binned
covariates; modes ``owncov`` (= default, bitwise) and ``margcov``
(covariates of random same-stage train donors) are in mode_margcov.py.
Conflicts with round-1 ``covars`` (both own slots 4-6).
"""
from __future__ import annotations

import numpy as np

import arms
from arms import Arm, register
import mode_margcov  # noqa: F401  (registers owncov / margcov)

NAME = 'covbin'


def age_thresholds(train_age):
    t1, t2 = np.quantile(np.asarray(train_age, dtype=np.float64), [1 / 3, 2 / 3])
    return float(t1), float(t2)


def bin_covariates(raw_c, t1, t2):
    """raw_c float64 [n,7] -> float32 [n,3] (sex +-1, APOE4 carrier +-1, age tertile - 1)."""
    raw = np.asarray(raw_c, dtype=np.float64)
    male, dose, age = raw[:, 4], raw[:, 5], raw[:, 6]
    if not np.isin(male, (0.0, 1.0)).all():
        raise ValueError('Male must be 0/1')
    if not np.isin(dose, (0.0, 1.0, 2.0)).all():
        raise ValueError('APOE4 dose must be 0/1/2')
    tert = (age > t1).astype(np.float64) + (age > t2)
    return np.stack([2 * male - 1, np.where(dose >= 1, 1.0, -1.0), tert - 1], 1).astype(np.float32)


def tuple_stats(train_raw, heldout_raw=None, train_stage=None):
    """Sharing of covariate tuples among train donors: covbin codes vs round-1 covars (Male, dose, raw age)."""
    t1, t2 = age_thresholds(train_raw[:, 6])
    b = bin_covariates(train_raw, t1, t2)

    def share(keys):
        keys = [tuple(np.round(np.asarray(k, dtype=np.float64), 6)) for k in keys]
        u, cnt = np.unique(np.array(keys), axis=0, return_counts=True)
        per_donor = np.array([cnt[np.flatnonzero((u == np.array(k)).all(1))[0]] for k in keys])
        return {'n_donors': len(keys), 'unique_tuples': int(len(u)), 'donors_per_tuple': sorted(cnt.tolist(), reverse=True),
                'singleton_donors': int((cnt == 1).sum()), 'mean_tuple_size_per_donor': float(per_donor.mean()),
                'min_tuple_size': int(cnt.min())}
    out = {'age_thresholds': [t1, t2], 'age_tertile_counts': np.bincount((b[:, 2] + 1).astype(int), minlength=3).tolist(),
           'covbin': share(b), 'covars_round1': share(train_raw[:, 4:7])}
    if train_stage is not None:
        out['covbin_with_stage'] = share(np.c_[np.asarray(train_stage, dtype=np.float64), b])
        out['covars_round1_with_stage'] = share(np.c_[np.asarray(train_stage, dtype=np.float64), train_raw[:, 4:7]])
    if heldout_raw is not None and len(heldout_raw):
        hb = bin_covariates(heldout_raw, t1, t2)
        seen = {tuple(x) for x in b.tolist()}
        out['heldout_tuple_absent_from_train'] = int(sum(tuple(x) not in seen for x in hb.tolist()))
        out['heldout_n'] = int(len(hb))
    return out


@register
class CovBinArm(Arm):
    name = NAME
    value_policy = 'none'
    conflicts = ('covars',)

    def setup(self, ctx):
        self.t1, self.t2 = age_thresholds(ctx.raw_c[:, 6])
        st = arms.stage_of(ctx.raw_c)
        self.info = {'encoding': 'slot4 = 2*Male-1, slot5 = +1 APOE4 carrier else -1, slot6 = age tertile - 1 (train thresholds)',
                     'age_thresholds': [self.t1, self.t2], 'tuple_stats': tuple_stats(ctx.raw_c, None, st)}

    def conditions(self, ctx, raw_c, cond):
        if np.any(cond[:, 4:7] != 0):
            raise ValueError('covbin: condition slots 4-6 already used by another arm')
        out = cond.copy()
        out[:, 4:7] = bin_covariates(raw_c, self.t1, self.t2)
        return out

    def state(self):
        return {'age_thresholds': [self.t1, self.t2]}
