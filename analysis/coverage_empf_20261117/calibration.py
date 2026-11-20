"""Donor-level, SD-only interval correction; generated cells are unchanged."""
import math
import numpy as np


def arrays(low, high, observed=None):
    low, high = np.asarray(low, dtype=float), np.asarray(high, dtype=float)
    if low.ndim != 2 or low.shape[1] != 100 or high.shape != low.shape:
        raise ValueError('expected donor by 100 mean/SD coordinates')
    if not np.isfinite(low).all() or not np.isfinite(high).all() or np.any(low > high):
        raise ValueError('nonfinite or reversed bounds')
    if np.any(low[:, 50:] < 0):
        raise ValueError('SD bounds must have nonnegative support')
    if observed is not None:
        observed = np.asarray(observed, dtype=float)
        if observed.shape != low.shape or not np.isfinite(observed).all() or np.any(observed[:, 50:] < 0):
            raise ValueError('invalid one-bag-per-donor observations')
    return low, high, observed


def identities(donors, count):
    donors = [str(d) for d in donors]
    if len(donors) != count or len(set(donors)) != count:
        raise ValueError('one unique identity is required per donor')
    return donors


def fit_sd(low, high, observed, donors, training_donors, alpha=.2):
    low, high, observed = arrays(low, high, observed)
    donors = identities(donors, len(low))
    training_donors = [str(d) for d in training_donors]
    if set(donors) & set(training_donors):
        raise ValueError('calibration donor was used in GAN training')
    if not 0 < alpha < 1:
        raise ValueError('alpha must be between zero and one')
    n = len(donors)
    rank = math.ceil((n + 1) * (1 - alpha))
    if not n or rank > n:
        raise ValueError('insufficient independent calibration donors for a finite correction')
    score = np.maximum(0, np.maximum(low[:, 50:] - observed[:, 50:],
                                    observed[:, 50:] - high[:, 50:]))
    correction = np.sort(score, axis=0)[rank - 1]
    return dict(schema=1, method='nonnegative_donor_split_SD_only_CQR', alpha=alpha,
                calibration_donors=donors, training_donors=training_donors,
                n_calibration_donors=n, rank=rank, sd_correction=correction.tolist(),
                lower_support=0., mean_coordinates_unchanged=True)


def apply_sd(low, high, record, donors):
    low, high, _ = arrays(low, high)
    donors = identities(donors, len(low))
    cal = identities(record['calibration_donors'], record['n_calibration_donors'])
    if set(donors) & set(cal):
        raise ValueError('evaluation and calibration donor identities overlap')
    if set(cal) & set(record['training_donors']):
        raise ValueError('calibration donor was used in GAN training')
    correction = np.asarray(record['sd_correction'], dtype=float)
    alpha, n = record['alpha'], len(cal)
    if (record['schema'] != 1 or record['method'] != 'nonnegative_donor_split_SD_only_CQR' or
            not 0 < alpha < 1 or record['rank'] != math.ceil((n + 1) * (1 - alpha)) or
            record['rank'] > n or record['lower_support'] != 0 or
            record['mean_coordinates_unchanged'] is not True or
            correction.shape != (50,) or not np.isfinite(correction).all() or np.any(correction < 0)):
        raise ValueError('invalid SD calibration artifact')
    lo, hi = low.copy(), high.copy()
    lo[:, 50:] = np.maximum(0, lo[:, 50:] - correction)
    hi[:, 50:] += correction
    return lo, hi
