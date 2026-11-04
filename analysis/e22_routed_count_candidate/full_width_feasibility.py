"""Full 2000-gene synthetic native optimizer pair and 400-cell count readout."""
import datetime
import json
import os
import resource
import time
from pathlib import Path

import torch
from particlegan import RoutedBatch
from pg_model import standardize_counts
from forward_feasibility import ROOT, build, context, sha
from update_feasibility import advance


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    torch.set_num_threads(1)
    started=time.monotonic()
    rec,policy,forward=build(genes=2000)
    counts=torch.poisson(torch.full((32,128,2000),5.),generator=torch.Generator().manual_seed(20261104211))
    guard=torch.poisson(torch.full((32,128,2000),5.),generator=torch.Generator().manual_seed(20261104212))
    fit_context,guard_context=context(20261104213,cells=128),context(20261104214,cells=128)
    fit_context[:,15:-12],guard_context[:,15:-12]=counts.sum(-1),guard.sum(-1)
    pack=lambda x:torch.cat([x,standardize_counts(x,forward.mean,forward.sd)],-1)
    batch=RoutedBatch(fit_context,pack(counts),guard_context,pack(guard))
    before=policy.table.detach().clone()
    stamp=time.monotonic();metrics=advance(rec,policy,forward,batch);update_seconds=time.monotonic()-stamp
    assert metrics['step']==1 and policy._phase=='ready'
    for value in metrics['gradients'].values():assert value is not None and value>0 and torch.isfinite(torch.tensor(value))
    assert not torch.equal(before,policy.table)
    for model in (policy.G,policy.D,policy.prior):
        for p in model.parameters():assert bool(torch.isfinite(p).all())
    print(json.dumps({'full_width_optimizer_pair':metrics,'CPU_update_seconds':update_seconds}),flush=True)
    served=policy.served_model();sources=[]
    stamp=time.monotonic()
    with torch.no_grad():
        for stage in range(4):
            query=context(20261104215+stage,batch=5,cells=400)
            query[:,8:15]=0;query[:,11]=stage/3;query[:,15:-12]=8000
            ambient=torch.get_rng_state().clone()
            value=served.routed_forward(query,perturb=False,output_noise=False)
            raw,aux=value[...,:2000],value[...,2000:]
            assert raw.shape==(5,400,2000) and bool(torch.isfinite(value).all())
            assert bool((raw>=0).all()) and torch.equal(raw,raw.floor())
            assert torch.equal(aux,standardize_counts(raw,forward.mean,forward.sd))
            assert torch.equal(value,served.routed_forward(query,perturb=False,output_noise=False))
            assert torch.equal(value[::2],served.routed_forward(query[::2],perturb=False,output_noise=False))
            assert torch.equal(value.flip(0),served.routed_forward(query.flip(0),perturb=False,output_noise=False))
            average=policy.routed_generate(query,sigma=0,perturb=False,averaged=True)
            avg_raw=average[...,:2000]
            assert bool(torch.isfinite(average).all()) and bool((avg_raw>=0).all())
            assert torch.equal(avg_raw,avg_raw.floor())
            assert torch.equal(average[...,2000:],standardize_counts(avg_raw,forward.mean,forward.sd))
            assert torch.equal(ambient,torch.get_rng_state())
            sources.append({'stage':stage,'bags':5,'cells_per_bag':400,'served_source':served.source})
            print('Full-width fast served and averaged count paths verified stage',stage,flush=True)
    serving_seconds=time.monotonic()-stamp
    assert not torch.cuda.is_initialized()
    result={'status':'ACTUAL_FULL2000GENE_ROUTED_E22_CPU_UPDATE_AND_CLEAN400CELL_COUNT_READOUT_PASSED',
            'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'native_optimizer_pairs_executed':1,
            'full_width_metrics':metrics,'CPU_update_seconds':update_seconds,
            'all_serving_checks_CPU_seconds':serving_seconds,'total_CPU_wall_seconds':time.monotonic()-started,
            'peak_process_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            'served_and_averaged_paths':sources,'full_width_query_replay_permutation_subsetting_exact':True,
            'raw_count_and_auxiliary_transform_exact_forward_verified':True,
            'full_width_nonzero_G_dispersion_table_and_learned_noise_gradients':True,
            'native_row_controls_disabled':False,'full_width_probe_boundary_exercised':False,
            'real_fitting_partition_implemented':False,'GPU_context_created':False,
            'native_admission_ready':False,'quality_measured':False,
            'source_SHA256':{str(p):sha(p) for p in [Path(__file__).resolve(),
                ROOT/'analysis/e22_routed_count_candidate/forward_feasibility.py',
                ROOT/'analysis/e22_routed_count_candidate/update_feasibility.py']},
            'scope':'Synthetic full2000gene/128cell native optimizer pair and all4stages five400cell bags each via current served and native averaged paths. CPU memory/timings are observations, not GPU admission or throughput guarantees. No real fitting data, calibration score, fullwidth20update probe boundary or native allocation.'}
    output=Path(__file__).with_suffix('.json');assert not output.exists()
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
