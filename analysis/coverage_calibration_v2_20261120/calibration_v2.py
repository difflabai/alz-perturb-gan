"""Donor-level SD-only interval correction, schema 2 (NumPy only).

This is a follow-up to ``analysis/coverage_empf_20261117/calibration.py``,
which stays pinned and unchanged. Differences from schema 1:

* ``fit_sd`` records the finite-sample nominal level ``rank/(n+1)`` and warns
  when ``rank == n``, i.e. the correction is the per-coordinate maximum of the
  calibration scores. With n=6 and alpha=.2 the rank is 6, so the nominal
  marginal level is 6/7 = 85.7%, not 80%.
* ``apply_sd`` also rejects evaluation donors that were in GAN training.
* Optional asymmetric lower/upper corrections, each at alpha/2 (the joint
  nominal level then follows from a union bound).
* ``loo_sd`` assembles leave-one-donor-out corrections over every held-out
  donor. The base intervals come from a generator that never saw these donors
  and are not refit, so CV+/jackknife+ reduce here to split conformal with the
  other n-1 donors as the calibration set for each evaluation donor.
* ``pooled_loo_sd`` does the same across several fits (folds/seeds), leaving
  out every row of the evaluation donor's identity.

Guarantees are marginal over exchangeable donors for one coordinate at a time.
They are not simultaneous over coordinates or donors, and they do not hold
if the calibration donors were used to select the model or readout.
"""
import math
import warnings

import numpy as np

SCHEMA = 2
D = 50


class RankAtMaximumWarning(UserWarning):
    """The conformal rank equals the calibration size (correction = max score)."""


def arrays(low, high, observed=None):
    low, high = np.asarray(low, dtype=float), np.asarray(high, dtype=float)
    if low.ndim != 2 or low.shape[1] != 2*D or high.shape != low.shape:
        raise ValueError('expected donor by 100 mean/SD coordinates')
    if not np.isfinite(low).all() or not np.isfinite(high).all() or np.any(low > high):
        raise ValueError('nonfinite or reversed bounds')
    if np.any(low[:, D:] < 0):
        raise ValueError('SD bounds must have nonnegative support')
    if observed is not None:
        observed = np.asarray(observed, dtype=float)
        if observed.shape != low.shape or not np.isfinite(observed).all() or np.any(observed[:, D:] < 0):
            raise ValueError('invalid one-bag-per-donor observations')
    return low, high, observed


def identities(donors, count):
    donors = [str(d) for d in donors]
    if len(donors) != count or len(set(donors)) != count:
        raise ValueError('one unique identity is required per donor')
    return donors


def conformal_rank(n, alpha):
    """Return (rank, nominal) for n calibration scores at miscoverage alpha.

    ``nominal = rank/(n+1)`` is the finite-sample marginal lower bound under
    exchangeability; it is at least 1-alpha and can be well above it.
    """
    if not 0 < alpha < 1:
        raise ValueError('alpha must be between zero and one')
    rank = math.ceil((n + 1) * (1 - alpha) - 1e-12)
    if n < 1 or rank > n:
        raise ValueError('insufficient independent calibration donors for a finite correction')
    return rank, rank / (n + 1)


def scores(low, high, observed):
    """Nonnegative lower, upper and symmetric SD nonconformity scores."""
    lower = np.maximum(0, low[:, D:] - observed[:, D:])
    upper = np.maximum(0, observed[:, D:] - high[:, D:])
    return lower, upper, np.maximum(lower, upper)


def fit_sd(low, high, observed, donors, training_donors, alpha=.2, asymmetric=False):
    low, high, observed = arrays(low, high, observed)
    donors = identities(donors, len(low))
    training_donors = sorted({str(d) for d in training_donors})
    if set(donors) & set(training_donors):
        raise ValueError('calibration donor was used in GAN training')
    n = len(donors)
    lower, upper, symmetric = scores(low, high, observed)
    if asymmetric:
        rank, side = conformal_rank(n, alpha / 2)
        ordered_lower, ordered_upper = np.sort(lower, axis=0), np.sort(upper, axis=0)
        correction = dict(lower=ordered_lower[rank-1].tolist(), upper=ordered_upper[rank-1].tolist())
        nominal = max(0., 1 - 2 * (1 - side))
    else:
        rank, nominal = conformal_rank(n, alpha)
        value = np.sort(symmetric, axis=0)[rank-1].tolist()
        correction = dict(lower=value, upper=value)
    if rank == n:
        warnings.warn(f'conformal rank {rank} equals the {n} calibration donors: each correction is '
                      f'the per-coordinate maximum score and the nominal level is {nominal:.4f}, '
                      f'not 1-alpha={1-alpha:.2f}', RankAtMaximumWarning, stacklevel=2)
    return dict(schema=SCHEMA, method='nonnegative_donor_split_SD_only_CQR_v2', alpha=alpha,
                asymmetric=bool(asymmetric), calibration_donors=donors,
                training_donors=training_donors, n_calibration_donors=n, rank=rank,
                rank_is_maximum=rank == n, nominal_coverage=nominal,
                sd_correction_lower=correction['lower'], sd_correction_upper=correction['upper'],
                lower_support=0., mean_coordinates_unchanged=True)


def check_record(record):
    cal = identities(record['calibration_donors'], record['n_calibration_donors'])
    if set(cal) & set(record['training_donors']):
        raise ValueError('calibration donor was used in GAN training')
    n, alpha = len(cal), record['alpha']
    rank, nominal = conformal_rank(n, alpha / 2 if record['asymmetric'] else alpha)
    if record['asymmetric']:
        nominal = max(0., 1 - 2 * (1 - nominal))
    lower = np.asarray(record['sd_correction_lower'], dtype=float)
    upper = np.asarray(record['sd_correction_upper'], dtype=float)
    if (record['schema'] != SCHEMA or record['method'] != 'nonnegative_donor_split_SD_only_CQR_v2' or
            record['rank'] != rank or abs(record['nominal_coverage'] - nominal) > 1e-12 or
            record['rank_is_maximum'] != (rank == n) or record['lower_support'] != 0 or
            record['mean_coordinates_unchanged'] is not True or
            (not record['asymmetric'] and not np.array_equal(lower, upper))):
        raise ValueError('invalid SD calibration artifact')
    for value in (lower, upper):
        if value.shape != (D,) or not np.isfinite(value).all() or np.any(value < 0):
            raise ValueError('invalid SD calibration artifact')
    return cal, lower, upper


def apply_sd(low, high, record, donors):
    low, high, _ = arrays(low, high)
    donors = identities(donors, len(low))
    cal, lower, upper = check_record(record)
    if set(donors) & set(cal):
        raise ValueError('evaluation and calibration donor identities overlap')
    if set(donors) & set(record['training_donors']):
        raise ValueError('evaluation donor was used in GAN training')
    lo, hi = low.copy(), high.copy()
    lo[:, D:] = np.maximum(0, lo[:, D:] - lower)
    hi[:, D:] += upper
    return lo, hi


def loo_sd(low, high, observed, donors, training_donors, alpha=.2, asymmetric=False):
    """Leave-one-donor-out corrected bounds for every supplied held-out donor.

    Each donor's bounds depend only on the other donors' observations; its own
    observation never enters its correction. Returns (lo, hi, records).
    """
    low, high, observed = arrays(low, high, observed)
    donors = identities(donors, len(low))
    lo, hi, records = low.copy(), high.copy(), []
    for i in range(len(donors)):
        keep = np.arange(len(donors)) != i
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RankAtMaximumWarning)
            record = fit_sd(low[keep], high[keep], observed[keep], np.asarray(donors)[keep],
                            training_donors, alpha, asymmetric)
        lo[i:i+1], hi[i:i+1] = apply_sd(low[i:i+1], high[i:i+1], record, [donors[i]])
        records.append(record)
    return lo, hi, records


def pooled_loo_sd(blocks, alpha=.2, asymmetric=False):
    """Leave-one-donor-identity-out calibration pooled across fits.

    ``blocks`` is a list of dicts with keys low, high, observed, donors,
    training_donors and fit. Each row must be a held-out donor of its own fit.
    A row's correction uses scores from every other row whose donor identity
    differs, across all fits, so repeated identities across folds/seeds are
    never calibrated on themselves. Pooled rows from different fits are only
    approximately exchangeable; report the pooled scheme as such.
    """
    rows = []
    for block in blocks:
        low, high, observed = arrays(block['low'], block['high'], block['observed'])
        donors = identities(block['donors'], len(low))
        if set(donors) & {str(d) for d in block['training_donors']}:
            raise ValueError(f"fit {block['fit']}: evaluation donor was used in GAN training")
        lower, upper, symmetric = scores(low, high, observed)
        rows.extend((block['fit'], d, i, lower[i], upper[i], symmetric[i]) for i, d in enumerate(donors))
    identity = np.array([r[1] for r in rows])
    lower = np.stack([r[3] for r in rows]); upper = np.stack([r[4] for r in rows])
    symmetric = np.stack([r[5] for r in rows])
    out = []
    for block in blocks:
        low, high, _ = arrays(block['low'], block['high'])
        lo, hi = low.copy(), high.copy()
        for i, d in enumerate(identities(block['donors'], len(low))):
            keep = identity != d
            n = int(keep.sum())
            if asymmetric:
                rank, _ = conformal_rank(n, alpha / 2)
                down = np.sort(lower[keep], axis=0)[rank-1]; up = np.sort(upper[keep], axis=0)[rank-1]
            else:
                rank, _ = conformal_rank(n, alpha)
                down = up = np.sort(symmetric[keep], axis=0)[rank-1]
            lo[i, D:] = np.maximum(0, low[i, D:] - down)
            hi[i, D:] = high[i, D:] + up
        out.append(dict(fit=block['fit'], low=lo, high=hi))
    return out
