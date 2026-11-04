"""Actual native probe clock and serialized continuation, synthetic CPU only."""
import copy
import datetime
import json
import os
import struct
from pathlib import Path

import numpy as np
import torch
from forward_feasibility import ROOT, build, sha
from update_feasibility import advance, synthetic_batch


def same(a,b):
    if isinstance(a,torch.Tensor):
        assert isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape
        # Native stationarity histories use NaN as a documented dropped-pair
        # marker. Compare actual bytes, preserving even NaN payloads and signed
        # zero; no numeric tolerance or blanket finite-state waiver is applied.
        assert torch.equal(a.contiguous().reshape(-1).view(torch.uint8),
                           b.contiguous().reshape(-1).view(torch.uint8))
    elif isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a:same(a[key],b[key])
    elif isinstance(a,(tuple,list)):
        assert type(a)==type(b) and len(a)==len(b)
        for x,y in zip(a,b):same(x,y)
    elif isinstance(a,float):assert isinstance(b,float) and struct.pack('<d',a)==struct.pack('<d',b)
    else:assert a==b,(a,b)


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    torch.set_num_threads(1)
    rec,policy,forward=build();batch=synthetic_batch(forward)
    for i in range(19):
        advance(rec,policy,forward,batch)
        if i in (0,9,18):print('Actual CPU native optimizer pairs',i+1,flush=True)
    assert policy.completed_steps==19 and policy.routed_control.counters['probes']==0
    folder=ROOT/'artifacts/diagnostics/e22_routed_count_CPU_probe20_v2'
    folder.mkdir(parents=True,exist_ok=False)
    checkpoint=folder/'before_probe19.pt';torch.save(copy.deepcopy(policy.state_dict()),checkpoint)
    before=torch.load(checkpoint,map_location='cpu',weights_only=False)
    expected20=advance(rec,policy,forward,batch)
    state20=copy.deepcopy(policy.state_dict())
    clock=state20['routing']['probe_clock'];counters=state20['routing']['counters']
    assert clock=={'observed_updates':20,'last_probe_update':20}
    assert counters['evals']==1 and counters['probes']==8
    expected21=advance(rec,policy,forward,batch);state21=copy.deepcopy(policy.state_dict())
    print('Actual probe completed; validating complete native serialized replay',flush=True)
    restored_rec,restored,restored_forward=build()
    restored.load_state_dict(before)
    same(expected20,advance(restored_rec,restored,restored_forward,batch))
    same(state20,restored.state_dict())
    same(expected21,advance(restored_rec,restored,restored_forward,batch))
    same(state21,restored.state_dict())
    assert not torch.cuda.is_initialized()
    result={'status':'ACTUAL_NATIVE_ROUTED_E22_PROBE20_AND_FULL_SERIALIZED_NEXT_STEP_CPU_REPLAY_PASSED',
            'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'original_completed_updates':21,'replayed_optimizer_pairs':2,'total_actual_optimizer_pairs':23,
            'native_probe_clock_after_step20':clock,'native_counters_after_step20':counters,
            'complete_native_policy_state_after_probe_and_following_update_bitwise_equal':True,
            'all_tensor_bytes_including_documented_native_NaN_sentinels_exactly_equal':True,
            'checkpoint_path':str(checkpoint),'checkpoint_SHA256':sha(checkpoint),
            'native_row_controls_disabled':False,'GPU_context_created':False,'real_data_accessed':False,
            'full2000gene_qualification_performed':False,'model_native_admission_ready':False,'quality_measured':False,
            'source_SHA256':{str(p):sha(p) for p in [Path(__file__).resolve(),
                ROOT/'analysis/e22_routed_count_candidate/forward_feasibility.py',
                ROOT/'analysis/e22_routed_count_candidate/update_feasibility.py']},
            'scope':'Synthetic8gene/8cell CPU policy lifecycle only: actual20update probe clock and8complete deletion counterfactuals, complete native models/optimizers/averages/controllers/fit_guard reservoirs/evidence/RNG serialized replay at20and21. A move or proposal is claimed only if actual counters show it; no real fitting-cell partition, fullwidth/count-quality/native runtime/lease/confirmation result.'}
    out=Path(__file__).with_suffix('.json');assert not out.exists();out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
