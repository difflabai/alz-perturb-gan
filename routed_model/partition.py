"""Fixed fitting-only cell split for subsequent E22 optimization/protection.

The common feature transform predates this split; protection is local to the
subsequent optimizer/evidence/proposal operations, not independent preprocessing.
"""
from dataclasses import dataclass
import hashlib
import json

import numpy as np

SALT = 20261104
DENOMINATOR = 5
SCHEMA = 'ALZ_E22_GUARD_V1'


def key(donor, cell):
    return json.dumps([SCHEMA, SALT, donor, cell], ensure_ascii=False, separators=(',', ':')).encode('utf-8')


def membership_sha(pairs):
    return hashlib.sha256(json.dumps(sorted(pairs), ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


@dataclass(frozen=True)
class Partition:
    donors: tuple
    fitting_by_donor: tuple
    protected_by_donor: tuple
    fitting: np.ndarray
    protected: np.ndarray
    membership_SHA256: dict

    def summary(self):
        return {'schema': SCHEMA, 'salt': SALT, 'protected_denominator': DENOMINATOR,
                'rounding': 'floor(n/5), at least1protected and1fitting cell per donor',
                'donors': list(self.donors), 'fitting_cells': len(self.fitting),
                'protected_cells': len(self.protected), 'per_donor': [
                    {'donor': d, 'fitting_cells': len(f), 'protected_cells': len(g)}
                    for d, f, g in zip(self.donors, self.fitting_by_donor, self.protected_by_donor)],
                'membership_SHA256': self.membership_SHA256}


def partition(observed_donors, cells, expected_donors):
    observed, names, expected = map(np.asarray, (observed_donors, cells, expected_donors))
    if any(x.ndim != 1 or x.dtype.kind != 'U' for x in (observed, names, expected)):
        raise ValueError('One-dimensional Unicode donor/cell identifiers required')
    if len(observed) != len(names) or not len(observed) or not len(expected):
        raise ValueError('Nonempty aligned donor/cell identities required')
    if any(bool((x == '').any()) for x in (observed, names, expected)):
        raise ValueError('Empty identities are invalid')
    donors = tuple(expected.tolist())
    pairs = list(zip(observed.tolist(), names.tolist()))
    if len(set(donors)) != len(donors) or set(observed.tolist()) != set(donors):
        raise ValueError('Exact unique optimization donor cohort required')
    if len(set(pairs)) != len(pairs):
        raise ValueError('Duplicate donor/cell identity')
    fit, guard = [], []
    for donor in donors:
        ids = np.flatnonzero(observed == donor)
        if len(ids) < 2:
            raise ValueError('Each fitting donor needs distinct optimization and protected cells')
        ranked = sorted(ids.tolist(), key=lambda i: (hashlib.sha256(key(donor, str(names[i]))).digest(), str(names[i])))
        ng = max(1, min(len(ranked)-1, len(ranked)//DENOMINATOR))
        guard.append(np.asarray(ranked[:ng], dtype=np.int64))
        fit.append(np.asarray(ranked[ng:], dtype=np.int64))
    fitting, protected = np.concatenate(fit), np.concatenate(guard)
    if len(set(fitting.tolist()) & set(protected.tolist())) or not np.array_equal(
            np.sort(np.concatenate([fitting, protected])), np.arange(len(observed))):
        raise AssertionError('Internal partition is not disjoint and exhaustive')
    fp = [pairs[i] for i in fitting]
    gp = [pairs[i] for i in protected]
    pins = {'all': membership_sha(pairs), 'fitting': membership_sha(fp), 'protected': membership_sha(gp)}
    for array in (*fit, *guard, fitting, protected):
        array.flags.writeable = False
    return Partition(donors, tuple(fit), tuple(guard), fitting, protected, pins)
