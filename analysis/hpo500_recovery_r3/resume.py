"""Continue the interrupted500step trial without resetting any scientific state."""
import json,shutil,signal,time
from pathlib import Path
import preset_scope as scope
def model_module():
    import released_adapter as model
    assert Path(model.__file__).resolve()==scope.MODEL_DIR/'released_adapter.py'
    return model
def run_remaining(ctx,b,g,r,u,seed,streams,out,*,end_monotonic,target=500):
    import torch
    model=model_module();out=Path(out)
    if ctx.device.type=='cuda':
        assert target==500 and b[-1].completed_steps==178 and seed==0
    assert b[0].total_steps==20000 and b[-1].completed_steps<target<=500
    stopped=[];handlers={s:signal.signal(s,lambda sig,frame:stopped.append(sig)) for s in (signal.SIGINT,signal.SIGTERM)}
    def save(status):
        path=out/('final.pt' if status=='completed' else 'recovery.pt')
        state=model.snapshot(ctx,b,g,r,u,seed,streams,status if target==500 else 'synthetic')
        temp=path.with_suffix('.partial.pt');torch.save(state,temp);temp.replace(path)
        model.dump(out/'checkpoint_receipt.json',dict(path=path.name,SHA256=scope.sha(path),status=state['status'],completed_updates=b[-1].completed_steps))
        return path
    try:
        while b[-1].completed_steps<target:
            if stopped or time.monotonic()>=end_monotonic:raise InterruptedError('Preserve native state at complete optimizer pair; finite continuation ending')
            metrics=model.advance(ctx,b,g,r,u,execution_limit=target)
            if metrics['step']%250 in (0,1):
                metrics['particle_usage']=model.passive_particle_diagnostics(b,u);model.dump(out/'progress.json',metrics)
                with (out/'learning_curve.jsonl').open('a') as stream:stream.write(json.dumps(metrics,allow_nan=False)+'\n')
            if metrics['step']%100==0:save('incomplete')
        return b,save('completed')
    except BaseException as exc:
        if b[-1]._phase=='ready':save('incomplete')
        model.dump(out/'failure.json',dict(error=repr(exc),completed_updates=b[-1].completed_steps,phase=b[-1]._phase,primary_endpoint_complete=False));raise
    finally:
        for s,h in handlers.items():signal.signal(s,h)
def fit(ctx,out,seed,auth,*,end_monotonic):
    if ctx.device.type!='cuda':raise RuntimeError('Real continuation requires separately guarded CUDA admission')
    import torch
    from state_fingerprint import fingerprint
    model=model_module();p=scope.read(scope.PARENT_PROOF)
    assert auth['is_continuation'] and seed==0 and auth['source_model_path']==p['checkpoint_path']
    assert scope.sha(p['checkpoint_path'])==auth['source_model_SHA256']==p['checkpoint_SHA256']
    q=torch.load(p['checkpoint_path'],map_location='cpu',weights_only=False)
    assert q['completed_updates']==q['policy']['completed_steps']==q['usage']['steps']==178 and q['status']=='incomplete'
    assert q['fixed_endpoint']==500 and q['policy']['device']=='cuda:0' and q['particlegan_version']=='0.9.6'
    assert q['released_hyperparameters']==auth['released_hyperparameters']==p['hyperparameters']
    assert q['variant_config']==auth['variant_config']==scope.CFG and q['cache_manifest_SHA256']==scope.sha(ctx.cache/'manifest.json')
    b,g,r,u,streams=model.restore(ctx,q)
    actual=model.snapshot(ctx,b,g,r,u,seed,streams,q['status'])
    assert fingerprint(q)==fingerprint(actual)==p['whole_parent_scientific_fingerprint'],'Full CUDA scientific/RNG restore differs'
    assert fingerprint(b[-1].state_dict())==p['native_policy_fingerprint']
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(p['checkpoint_path'],out/'parent178.pt');assert scope.sha(out/'parent178.pt')==p['checkpoint_SHA256']
    assert scope.sha(p['learning_curve_path'])==p['learning_curve_SHA256']
    shutil.copyfile(p['learning_curve_path'],out/'learning_curve.jsonl')
    model.dump(out/'native_parent_restore.json',dict(status='EXACT_NATIVE178_ALL_SCIENTIFIC_AND_RANDOM_STATE_CUDA_RESTORE_PASS',completed_updates=178,whole_parent_scientific_fingerprint=fingerprint(actual),all_native_scientific_and_random_state_byte_exact=True,recipe_horizon=20000,endpoint=500,new_model_updates=322,native_clock_reset=False,native_device_metadata_relabelled=False,source_parent_checkpoint_SHA256=p['checkpoint_SHA256']))
    del q,actual
    return run_remaining(ctx,b,g,r,u,seed,streams,out,end_monotonic=end_monotonic)
