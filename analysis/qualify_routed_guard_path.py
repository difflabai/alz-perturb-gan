"""Final callback guard rejection in the previously declared synthetic challenge."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

import torch
from particlegan import RoutedBatch
import routed_pg_adapter as a
from test_routed_model import fixture

ROOT=Path('/ml2/alz')


def challenge_context(seed,counts):
    g=torch.Generator().manual_seed(seed)
    q=torch.randn(32,8,generator=g);c=torch.zeros(32,7);c[:,3]=torch.arange(32).remainder(4)/3
    words=torch.randint(65536,(32,12),generator=g).float()
    # Canonicalize the high word to reproduce the earlier immutable toy
    # challenge's effective63bit seed law; no native mechanism is changed.
    words[:,[3,7,11]]=words[:,[3,7,11]].remainder(32768)
    return torch.cat([q,c,counts.sum(-1),words],-1)


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as temp:
        ctx=fixture(temp,genes=2000)
        ctx.mu=torch.ones(2000);ctx.theta=torch.full((2000,),3.)
        ctx.W=torch.randn(2000,50,generator=torch.Generator().manual_seed(20261104201))/(2000**.5)
        runtime=a.build(ctx,0)
        counts=torch.poisson(torch.full((32,128,2000),5.),generator=torch.Generator().manual_seed(20261104211))
        guard=torch.poisson(torch.full((32,128,2000),5.),generator=torch.Generator().manual_seed(20261104212))
        fit_context=challenge_context(20261104213,counts);guard_context=challenge_context(20261104214,guard)
        batch=RoutedBatch(fit_context,runtime.forward.targets(counts),guard_context,runtime.forward.targets(guard))
        def fixed_batch(c,r):
            assert c is ctx and r is runtime
            r.fitting_contexts_seen+=32;r.protected_contexts_seen+=32
            return batch
        with patch.object(a,'training_batch',side_effect=fixed_batch):
            for i in range(20):
                a.advance(ctx,runtime)
                if i in (0,9,18):print('Final callback synthetic guard-path pairs',i+1,flush=True)
        counters=dict(runtime.policy.routed_control.counters)
        assert counters['evals']==1 and counters['probes']==8
        assert counters['proposals']>0 and counters['guard_rejections']>0 and counters['moves']==0
        assert runtime.recipe.row_evidence_gate and runtime.recipe.particle_birth_death
        assert not torch.cuda.is_initialized()
        sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
        output=ROOT/'analysis/routed_final_guard_path_CPU_qualification.json';assert not output.exists()
        proof={'status':'ACTUAL_FINAL_COUNT_CALLBACK_NATIVE_PROPOSAL_AND_GUARD_REJECTION_PATH_PASSED',
               'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'actual_native_optimizer_pairs':20,
               'native_counters':counters,'native_probe_clock':dict(runtime.policy.routed_control.probe_clock),
               'native_mechanisms_or_thresholds_changed':False,'final_count_callback_used':True,
               'private_synthetic_training_batch_override':True,'real_data_used':False,
               'GPU_context_created':False,'quality_measured':False,
               'source_SHA256':{str(p):sha(p) for p in [Path(__file__).resolve(),ROOT/'analysis/test_routed_model.py',*(ROOT/'routed_model').glob('*.py')]},
               'scope':'The same synthetic2000gene/128cell challenge previously used for primitive feasibility, now through the final callback and full native update. Private fixture fixes only synthetic fit/guard inputs and initialization; effective63bit challenge seeds are losslessly encoded in final64bit format. No row-control threshold or mechanism disabled; actual native fit-selected proposal rejected on protected contexts. No accepted move, real fitting/selection, native deployment or quality claim.'}
        output.write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof,indent=2))


if __name__=='__main__':main()
