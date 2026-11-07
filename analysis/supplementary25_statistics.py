"""Conditional contrasts for one subject cohort seen through three fit views.

The same subject is resampled once and reused in every fitting-fold view.
No fold, seed, generation replicate or cell is a new biological subject.
"""
import numpy as np

def shared_donor_interval(components, stages, *, draws=10000, seed=20261107,
                          interval_percent=98.33333333333333):
    if (len(components)!=3 or type(draws) is not int or draws<1
            or not 0<interval_percent<100):
        raise ValueError('Three fixed fitting views and positive bootstrap budget required')
    order=components[0]['donors'];stages=np.asarray(stages)
    if (not order or len(order)!=len(set(order)) or stages.shape!=(len(order),)
            or set(stages.tolist())!=set(range(4))):
        raise ValueError('One common unique subject order and all four stage strata required')
    validated=[]
    for c in components:
        sums=np.asarray(c['cross_sums'],dtype=np.float64)
        raw_counts=np.asarray(c['cell_counts'])
        constant=c['generated_constant']
        if (c['donors']!=order or sums.shape!=(len(order),)
                or raw_counts.shape!=sums.shape or not np.isfinite(sums).all()
                or not np.isfinite(raw_counts).all() or (raw_counts<=0).any()
                or not np.equal(raw_counts,np.floor(raw_counts)).all()
                or not np.isfinite(constant)):
            raise ValueError('Every subject needs finite paired contributions and positive integer reservoir counts in every view')
        validated.append((sums,raw_counts.astype(np.int64),float(constant)))
    rng=np.random.default_rng(seed)
    # Draw subject indices once. Reusing these exact indices across all views
    # preserves the shared-target dependence even when cell counts differ.
    chosen=np.concatenate([rng.choice(np.flatnonzero(stages==stage),
        size=(draws,int((stages==stage).sum())),replace=True) for stage in range(4)],axis=1)
    boot=np.zeros(draws);points=[]
    for sums,counts,constant in validated:
        boot+=(sums[chosen].sum(1)/counts[chosen].sum(1)+constant)/3
        points.append(float(sums.sum()/counts.sum()+constant))
    alpha=(100-interval_percent)/2;lower,upper=np.percentile(boot,[alpha,100-alpha])
    return dict(mean_delta=float(np.mean(points)),lower=float(lower),upper=float(upper),
        fitting_view_point_deltas=points,draws=draws,seed=seed,interval_percent=interval_percent,
        biological_subject_count=len(order),same_subject_draw_reused_across_all_three_views=True,
        unit='one global real donor cluster, stratified by stage',
        model_and_generation_reservoirs_fixed=True,independent_population_confirmation=False)
