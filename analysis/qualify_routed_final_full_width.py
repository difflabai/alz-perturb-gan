"""Full-width final adapter: dynamic protected sampling, probe replay and serving."""
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import resource
import tempfile
import time
from types import SimpleNamespace

import numpy as np
import torch
import routed_pg_adapter as a
from routed_generation import iter_routed_counts
from test_routed_model import fixture,same

ROOT=Path('/ml2/alz')


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';torch.set_num_threads(1)
    started=time.monotonic()
    with tempfile.TemporaryDirectory() as temp:
        cache=Path(temp);(cache/'manifest.json').write_text('{}')
        ctx=fixture(cache,genes=2000);runtime=a.build(ctx,0)
        for i in range(19):
            a.advance(ctx,runtime)
            if i in (0,4,9,14,18):print('Final adapter full2000gene optimizer pairs',i+1,flush=True)
        folder=ROOT/'artifacts/diagnostics/routed_final_adapter_CPU_fullwidth_probe20_v1'
        folder.mkdir(parents=True,exist_ok=False)
        checkpoint=folder/'before19.pt';torch.save(a.snapshot(ctx,runtime,'synthetic'),checkpoint)
        before=torch.load(checkpoint,map_location='cpu',weights_only=False)
        metrics20=a.advance(ctx,runtime);state20=a.snapshot(ctx,runtime,'synthetic')
        assert state20['policy']['routing']['probe_clock']=={'observed_updates':20,'last_probe_update':20}
        assert state20['policy']['routing']['counters']['probes']==8
        metrics21=a.advance(ctx,runtime);state21=a.snapshot(ctx,runtime,'synthetic')
        replay=a.restore(ctx,before)
        same(metrics20,a.advance(ctx,replay));same(state20,a.snapshot(ctx,replay,'synthetic'))
        same(metrics21,a.advance(ctx,replay));same(state21,a.snapshot(ctx,replay,'synthetic'))
        del replay,before,state21
        print('Final adapter fullwidth dynamic sampler/probe/native state byte-replay passed',flush=True)
        served=runtime.policy.served_model();observed=[]
        schedule=SimpleNamespace(libraries=np.full((200,400),8000,dtype=np.float32))
        for stage in range(4):
            c=np.zeros(7,dtype=np.float32);c[3]=stage/3
            iterator=iter_routed_counts(ctx,served,runtime.forward,schedule,c,20261104+stage)
            try:first=next(iterator)
            finally:iterator.close()
            assert first[0]==0 and first[1].shape==(5,400,50) and first[2].shape==(1,400,2000) and first[3].shape==(2000,)
            assert np.isfinite(first[1]).all() and np.isfinite(first[2]).all()
            assert (first[2]>=0).all() and np.equal(first[2],np.floor(first[2])).all()
            np.testing.assert_array_equal(first[3][:400],first[2].sum(-1).reshape(-1))
            repeated=iter_routed_counts(ctx,served,runtime.forward,schedule,c,20261104+stage)
            try:second=next(repeated)
            finally:repeated.close()
            same(first,second)
            if stage==0:
                old=runtime.policy.G.out[-1].bias.detach().clone()
                try:
                    with torch.no_grad():runtime.policy.G.out[-1].bias[0]+=100
                    third=iter_routed_counts(ctx,served,runtime.forward,schedule,c,20261104+stage)
                    try:same(first,next(third))
                    finally:third.close()
                finally:
                    with torch.no_grad():runtime.policy.G.out[-1].bias.copy_(old)
            observed.append({'stage':stage,'bags_verified':5,'cells_per_bag':400,'native_served_source':served.source})
            print('Final native served400cell count iterator verified stage',stage,flush=True)
        for module in (runtime.policy.G,runtime.policy.D,runtime.policy.prior):
            for p in module.parameters():assert bool(torch.isfinite(p).all())
        assert not torch.cuda.is_initialized()
        sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
        output=ROOT/'analysis/routed_final_full_width_CPU_qualification.json';assert not output.exists()
        proof={'status':'ACTUAL_FINAL_ROUTED_ADAPTER_FULL2000GENE_PROBE_REPLAY_AND_COUNT_ITERATOR_PASSED',
               'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'original_optimizer_pairs':21,'replayed_optimizer_pairs':2,
               'step20_probe_clock':state20['policy']['routing']['probe_clock'],
               'step20_native_counters':state20['policy']['routing']['counters'],
               'full_native_and_both_data_sampler_states_after20and21_byte_identical':True,
               'fitting_and_protection_contexts_seen':runtime.fitting_contexts_seen,
               'actual_native_400cell_count_iterator':observed,
               'native_served_count_snapshot_independent_of_live_generator_mutation':True,
               'checkpoint_path':str(checkpoint),'checkpoint_SHA256':sha(checkpoint),
               'CPU_wall_seconds':time.monotonic()-started,
               'peak_process_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
               'source_SHA256':{str(p):sha(p) for p in [Path(__file__).resolve(),ROOT/'analysis/test_routed_model.py',*(ROOT/'routed_model').glob('*.py')]},
               'native_row_controls_disabled':False,'GPU_context_created':False,
               'real_data_used_for_optimizer_update':False,'native_admission_ready':False,'quality_measured':False,
               'scope':'Finaladapter synthetic2000gene/128cell dynamic fitting/protected sampler native21updates with probe20 and complete serialized20/21 replay, plus clean frozen served five400cell bags for all4stages. Same existing transformed-rate estimator and directintegercounts. CPU memory/timings do not establish GPU capacity or sixhour throughput. No calibration score/evaluationdata/GPU allocation.'}
        output.write_text(json.dumps(proof,indent=2,allow_nan=False)+'\n');print(json.dumps(proof,indent=2))


if __name__=='__main__':main()
