"""Paired pooled-cell energy contrasts with real-donor cluster resampling.

The real-real distance cancels in a paired contrast. Generated reservoirs stay
fixed; uncertainty is conditional on those fitted models and reservoirs. No
cell, generation replicate, or initialization seed is counted as a donor.
"""
import numpy as np
from scipy.spatial.distance import cdist


def paired_components(generated_pg, generated_other, real_cells, donor_labels,
                      donor_order):
    pg, other, real = map(lambda x: np.asarray(x, dtype=np.float64),
                          (generated_pg, generated_other, real_cells))
    labels, order = np.asarray(donor_labels), np.asarray(donor_order)
    if (any(x.ndim != 2 or not len(x) or not np.isfinite(x).all()
            for x in (pg, other, real))
            or len({x.shape[1] for x in (pg, other, real)}) != 1
            or labels.shape != (len(real),) or order.ndim != 1
            or len(set(order.tolist())) != len(order)
            or set(labels.tolist()) != set(order.tolist())):
        raise ValueError('Finite paired reservoirs and every declared donor required')
    # Each element is the paired cross-distance contribution of one real cell.
    cross = 2 * (cdist(pg, real).mean(0) - cdist(other, real).mean(0))
    sums = np.array([cross[labels == donor].sum() for donor in order])
    counts = np.array([(labels == donor).sum() for donor in order], dtype=np.int64)
    constant = float(cdist(other, other).mean() - cdist(pg, pg).mean())
    return dict(donors=order.tolist(), cross_sums=sums, cell_counts=counts,
                generated_constant=constant,
                point_delta=float(sums.sum() / counts.sum() + constant))


def average_fixed_models(components):
    """Average paired model/replicate statistics before donor resampling."""
    if not components:
        raise ValueError('No paired fitted models')
    first = components[0]
    for item in components:
        if (item['donors'] != first['donors']
                or not np.array_equal(item['cell_counts'], first['cell_counts'])):
            raise ValueError('Real donor reservoir identity/count pairing differs')
    sums = np.mean([item['cross_sums'] for item in components], axis=0)
    constant = float(np.mean([item['generated_constant'] for item in components]))
    counts = first['cell_counts'].copy()
    return dict(donors=first['donors'], cross_sums=sums, cell_counts=counts,
                generated_constant=constant,
                point_delta=float(sums.sum() / counts.sum() + constant))


def donor_cluster_interval(fold_components, stages_by_fold, *, draws=10000,
                          seed=20261106, interval_percent=98.33333333333333):
    """Equal-fold mean paired contrast, donor-cluster bootstrap within stage.

    Three primary comparator contrasts use Bonferroni 98 1/3 percent intervals.
    This function requires the declared three folds and four stage strata. It
    does not infer biological independence from overlapping training folds or
    historical exposure to this cohort.
    """
    if (len(fold_components) != 3 or len(stages_by_fold) != 3
            or type(draws) is not int or draws < 1
            or not 0 < interval_percent < 100):
        raise ValueError('Three complete folds and a positive resampling budget required')
    rng = np.random.default_rng(seed)
    boot = np.zeros(draws)
    points = []
    for component, stages in zip(fold_components, stages_by_fold):
        sums = np.asarray(component['cross_sums'], dtype=np.float64)
        counts = np.asarray(component['cell_counts'], dtype=np.int64)
        stages = np.asarray(stages)
        if (stages.shape != sums.shape or counts.shape != sums.shape
                or set(stages.tolist()) != set(range(4))
                or (counts <= 0).any() or not np.isfinite(sums).all()):
            raise ValueError('All four donor-stage strata and nonempty clusters required')
        sampled_sums = np.zeros(draws)
        sampled_counts = np.zeros(draws)
        for stage in range(4):
            indices = np.flatnonzero(stages == stage)
            chosen = rng.choice(indices, size=(draws, len(indices)), replace=True)
            sampled_sums += sums[chosen].sum(1)
            sampled_counts += counts[chosen].sum(1)
        constant = component['generated_constant']
        boot += (sampled_sums / sampled_counts + constant) / 3
        points.append(float(sums.sum() / counts.sum() + constant))
    alpha = (100 - interval_percent) / 2
    lower, upper = np.percentile(boot, [alpha, 100 - alpha])
    return dict(mean_delta=float(np.mean(points)), lower=float(lower),
                upper=float(upper), fold_point_deltas=points,
                interval_percent=interval_percent, draws=draws, seed=seed,
                unit='real donor cluster, stratified by fold and stage',
                model_and_generation_reservoirs_fixed=True,
                independent_population_confirmation=False)
