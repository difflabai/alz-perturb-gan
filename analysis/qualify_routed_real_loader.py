"""Read fitting counts only; qualify actual loader/statistics/trace boundaries."""
import datetime
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
import routed_pg_adapter as a
from generation import trace

ROOT=Path('/ml2/alz')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';torch.set_num_threads(1)
    previous=ROOT/'artifacts/allocations/CAL_GPU0_V1_halloween_native_v1_f0_pg_s0/authorization.json'
    cache=Path(json.loads(previous.read_text())['cache_dir'])
    ctx=a.Context(cache,'cpu')
    assert ctx.X.shape==(14762,2000) and len(ctx.ids)==33
    split=ctx.partition
    assert len(split.fitting)==11825 and len(split.protected)==2937
    old=json.loads((ROOT/'analysis/routed_fitting_partition_CPU_qualification.json').read_text())
    assert split.summary()==old['actual_fold0_metadata_partition']
    expected_mu,expected_theta=a.fitting_statistics(ctx.X,split)
    assert torch.equal(expected_mu,ctx.mu) and torch.equal(expected_theta,ctx.theta)
    protected=torch.as_tensor(split.protected.copy())
    backup=ctx.X[protected].clone()
    try:
        ctx.X[protected]+=10000
        changed_mu,changed_theta=a.fitting_statistics(ctx.X,split)
        assert torch.equal(ctx.mu,changed_mu) and torch.equal(ctx.theta,changed_theta)
    finally:ctx.X[protected]=backup
    for role,pool,seed in [('fitting',split.fitting,20261104241),('protected',split.protected,20261104242)]:
        rng=np.random.default_rng(seed)
        for _ in range(8):
            counts,c,lib,rows=ctx.draw(role,rng)
            assert np.isin(rows,pool).all()
            assert torch.equal(counts,ctx.X[torch.as_tensor(rows)])
            assert torch.equal(lib,counts.sum(-1))
            for i,row in enumerate(rows):
                d=ctx.fit_labels[row[0]];assert set(ctx.fit_labels[row])=={d}
                ix=ctx.ids.tolist().index(d);assert torch.equal(c[i],ctx.conditions[ix])
    traces={}
    for stage in range(4):
        s,c,schedule=trace(ctx,0,0,stage)
        assert np.array_equal(schedule.libraries,ctx.fitlibs[torch.as_tensor(schedule.rows)].numpy())
        traces[str(stage)]={'template_seed':s,'rows':list(schedule.rows.shape),
                            'libraries_SHA256':hashlib.sha256(schedule.libraries.tobytes()).hexdigest()}
    assert not torch.cuda.is_initialized()
    data=ctx.manifest['files_SHA256']
    for rel in ('train/counts_hvg.npz','train/metadata.npz'):
        assert sha(cache/rel)==data[rel]
    output=ROOT/'analysis/routed_real_loader_CPU_qualification.json';assert not output.exists()
    proof={'status':'ACTUAL_FITTING_ONLY_ROUTED_COUNT_LOADER_STATS_SAMPLERS_AND_COMMON_TRACES_PASSED',
           'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'total_original_fitting_cells':len(ctx.X),'gradient_fitting_cells':len(split.fitting),
           'protected_cells':len(split.protected),'fitting_donors':len(ctx.ids),
           'partition':split.summary(),'fit_only_initial_mu_theta_verified':True,
           'protected_values_perturbed_in_private_memory_do_not_change_initial_stats':True,
           'sixteen_real_batches_match_declared_pool_counts_conditions_and_libraries':True,
           'four_registered_seed0_trace_sources_and_libraries_unchanged':traces,
           'initial_mu_SHA256':hashlib.sha256(ctx.mu.numpy().tobytes()).hexdigest(),
           'initial_theta_SHA256':hashlib.sha256(ctx.theta.numpy().tobytes()).hexdigest(),
           'source_SHA256':{str(p):sha(p) for p in [Path(__file__).resolve(),*(ROOT/'routed_model').glob('*.py')]},
           'input_SHA256':{str(p):sha(p) for p in [cache/'manifest.json',cache/'train/counts_hvg.npz',cache/'train/metadata.npz']},
           'GPU_context_created':False,'generative_model_or_optimizer_created':False,
           'calibration_or_evaluation_count_arrays_loaded':False,'quality_measured':False,
           'scope':'Actual33fittingdonor count loader only. Protected-value perturbation occurs in private CPU tensor and is restored; immutable count/cache files unchanged. Common frozen preprocessing predates protection and uses all original fitting donors/cells. Registered exposure traces retain that original fitting-source pool. Integrity hashing may read immutable evaluation bytes through checked_cache; evaluation count arrays are never loaded for statistics, fitting or selection. No adversarial update, GPU model/admission or quality result.'}
    output.write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps({k:v for k,v in proof.items() if k not in ('partition','source_SHA256','input_SHA256')},indent=2))


if __name__=='__main__':main()
