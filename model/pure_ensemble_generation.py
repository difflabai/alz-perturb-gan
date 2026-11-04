"""Compose complete, sequential native PG branches by whole generated bags."""
import numpy as np

MASK_SEED = 20261108


def model_mask(weight, replicate, stage):
    if type(weight) not in (float, int) or weight not in (0., .25, 1.):
        raise ValueError('Only parent controls or the registered quarter mixture')
    if type(replicate) is not int or replicate not in (0, 1) or type(stage) is not int or stage not in range(4):
        raise ValueError('Fixed generation replicate and stage required')
    mask = np.zeros(200, dtype=bool)
    selected = np.random.default_rng(MASK_SEED + 100 * replicate + stage).choice(200, int(200 * weight), replace=False)
    mask[selected] = True
    return mask


def collect_branch(factory, check_deadline):
    pcs = []; totals = []; exemplar0 = None; next_start = 0
    for start, values, exemplar, libraries in factory():
        check_deadline()
        if start != next_start or values.ndim != 3 or values.shape[1:] != (400, 50) or len(values) != min(5, 200 - start):
            raise ValueError('Exact native five-bag chunk contract required')
        if not np.isfinite(values).all() or np.asarray(libraries).size != len(values) * 400 or not np.isfinite(libraries).all():
            raise ValueError('Finite complete PC and library values required')
        if start == 0:
            if exemplar is None or exemplar.shape != (1, 400, 2000) or not np.isfinite(exemplar).all() or not (exemplar >= 0).all() or not np.equal(exemplar, np.floor(exemplar)).all():
                raise ValueError('Native first-bag integer-count exemplar required')
            exemplar0 = exemplar.copy()
        pcs.append(values.copy()); totals.append(np.asarray(libraries).reshape(len(values), 400).copy())
        next_start += len(values)
    if next_start != 200:
        raise ValueError('Incomplete native parent branch')
    return np.concatenate(pcs), np.concatenate(totals), exemplar0


def iter_ensemble(factories, *, weight, replicate, stage, check_deadline):
    if set(factories) != {0, 1}:
        raise ValueError('Two explicitly named native parent factories required')
    mask = model_mask(weight, replicate, stage)
    active = ([0] if weight == 0 else [1] if weight == 1 else [0, 1])
    # Exhaust a branch before starting the next. Gamma.rsample uses global RNG.
    branches = {index: collect_branch(factories[index], check_deadline) for index in active}
    for start in range(0, 200, 5):
        check_deadline(); stop = start + 5
        if len(active) == 1:
            pcs, totals, exemplar = branches[active[0]]
            yield start, pcs[start:stop], exemplar if start == 0 else None, totals[start:stop].reshape(-1)
        else:
            a, b = branches[0], branches[1]; chosen = mask[start:stop]
            pcs = np.where(chosen[:, None, None], b[0][start:stop], a[0][start:stop])
            totals = np.where(chosen[:, None], b[1][start:stop], a[1][start:stop])
            exemplar = branches[int(mask[0])][2] if start == 0 else None
            yield start, pcs, exemplar, totals.reshape(-1)
