"""Actual frozen native checkpoint restore and fullwidth count replay on CPU."""
import datetime,json,os,sys,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import binding as b,preset_scope as scope
def main():
    method=sys.argv[1];assert method in ('pg','scvi') and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    value=scope.cell(0,method,0)
    sys.path.extend(map(str,b.native_site_roots(value,{str(b.PG_SHARED_PTH):b.PG_SHARED_PTH_SHA})))
    sys.path[:0]=[str(scope.MODEL_DIR),str(HERE.parent/'model'),str(b.SCIENCE),str(HERE.parent)]
    import torch,numpy as np
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    from state_fingerprint import fingerprint
    from generation import trace,iter_pg,iter_scvi
    cache=b.SCIENCE/'caches/fold0';checkpoint=scope.model_input(value) if method=='scvi' else Path(scope.read(scope.PROTOCOL)['PG']['historical095_checkpoint'])
    if method=='pg':
        import released_update as update,released_adapter as adapter
        update.configure(scope.CFG);update.configure_hyperparameters(scope.HYPER)
        ctx=adapter.Context(cache,'cpu');q=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert q['fold']==q['seed']==0 and q['completed_updates']==q['policy']['completed_steps']==20000
        assert q['released_hyperparameters']==scope.HYPER and q['variant_config']==scope.CFG
        bundle,*_=adapter.build(ctx.mu,ctx.theta,'cpu',0,0)
        assert q['recipe']==bundle[0].to_dict()
        for name,module in zip(('generator','critic','prior'),bundle[1:4]):
            module.load_state_dict(q['policy']['models'][name],strict=True)
            assert fingerprint(module.state_dict())==fingerprint(q['policy']['models'][name])
        state=fingerprint(q['policy']['models'])
        iterator=lambda schedule,c,s:iter_pg(ctx,bundle[1],bundle[3],schedule,c,s)
    else:
        from native_scvi_reload import load_native
        from sampling_context import Context
        model,receipt,state=load_native(cache,checkpoint.parent.parent,'cpu')
        assert receipt['fold']==receipt['seed']==0 and receipt['completed_epochs']==400
        ctx=Context(cache,'cpu');iterator=lambda schedule,c,s:iter_scvi(ctx,model,schedule,c,s)
    s,c,schedule=trace(ctx,0,0,0)
    first=next(iterator(schedule,c,s));second=next(iterator(schedule,c,s))
    assert first[0]==second[0]==0
    for x,y in zip(first[1:],second[1:]):np.testing.assert_array_equal(x,y)
    pcs,counts=first[1:3]
    assert pcs.shape==(5,400,50) and counts.shape==(1,400,2000)
    assert np.isfinite(pcs).all() and np.isfinite(counts).all() and (counts>=0).all() and np.equal(counts,np.floor(counts)).all()
    assert not torch.cuda.is_initialized()
    result=dict(status='ACTUAL_NATIVE_CPU_RESTORE_AND_FULLWIDTH_COUNT_REPLAY_PASS',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),method=method,
                checkpoint_path=str(checkpoint),checkpoint_SHA256=b.sha(checkpoint),native_state_fingerprint=state,
                PC_shape=list(pcs.shape),count_shape=list(counts.shape),count_SHA256=hashlib.sha256(counts.tobytes()).hexdigest(),
                exact_repeated_native_count_and_PC_replay=True,optimization_donors=33,evaluation_target_values_read=False,
                native_CUDA_policy_device_metadata_relabelled=False,GPU_allocation_created=False,CUDA_initialized=False,
                source_SHA256={str(p):b.sha(p) for p in (Path(__file__),HERE/'worker_gpu0.py',HERE/'native_scvi_reload.py',scope.PROTOCOL)})
    path=HERE.parent/'analysis'/f'external_core_{method}_model_v096_CPU.json';assert not path.exists();path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
if __name__=='__main__':main()
