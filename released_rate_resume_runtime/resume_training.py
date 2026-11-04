"""Exact public native checkpoint and every original scientific random stream."""
import json,shutil,signal,time
from pathlib import Path
import binding as b

def validate_parent(q,auth):
    p=auth['continuation'];n=p['completed_updates']
    b.require(n==10842 and q.get('schema')==1 and q.get('method')=='pg','Exact10842 released parent required')
    b.require(q.get('native_recipe')=='bispf_optimized' and q.get('status')=='incomplete','Incomplete native fixed-family parent required')
    b.require(q.get('completed_updates')==q.get('policy',{}).get('completed_steps')==n and q.get('fixed_endpoint')==20000,'Parent native update clock differs')
    b.require(q.get('fold')==auth['cell']['fold']==0 and q.get('seed')==auth['cell']['seed']==0,'Same parent fold/seed required')
    b.require(q['policy']['device']=='cuda:0','Original native CUDA metadata required')
    b.require(q.get('cache_manifest_SHA256')==b.sha(Path(auth['cache_dir'])/'manifest.json'),'Parent cache differs')
    b.require(q.get('released_hyperparameters')==auth['released_hyperparameters'] and q.get('variant_config')==auth['variant_config'],'Parent parameters or architecture differ')
    b.require(q.get('usage',{}).get('steps')==n,'Parent usage clock differs')

def _run_remaining(ctx,bundle,generators,rng,usage,seed,streams,out,*,end_monotonic,target_updates=20000):
    import torch
    import released_adapter as model
    if ctx.device.type=='cuda' and target_updates!=20000:raise ValueError('Native continuation retains original20k endpoint')
    out=Path(out);stops=[];handlers={s:signal.signal(s,lambda sig,frame:stops.append(sig)) for s in (signal.SIGINT,signal.SIGTERM)}
    def save(status):
        path=out/('final.pt' if status=='completed' else 'recovery.pt');tmp=path.with_suffix('.partial.pt')
        torch.save(model.snapshot(ctx,bundle,generators,rng,usage,seed,streams,status),tmp);tmp.replace(path)
        model.dump(out/'checkpoint_receipt.json',{'path':path.name,'SHA256':model.sha(path),'status':status,'completed_updates':bundle[-1].completed_steps})
        return path
    try:
        while bundle[-1].completed_steps<target_updates:
            if stops or time.monotonic()>=end_monotonic:raise InterruptedError('Original finite allocation ending; preserve exact continuation')
            metrics=model.advance(ctx,bundle,generators,rng,usage,execution_limit=20000)
            if metrics['step']%250 in (0,1):
                metrics['particle_usage']=model.passive_particle_diagnostics(bundle,usage)
                model.dump(out/'progress.json',metrics)
                with (out/'learning_curve.jsonl').open('a') as stream:stream.write(json.dumps(metrics,allow_nan=False)+'\n')
            if metrics['step']%1000==0:save('recovery')
        return bundle,save('completed' if target_updates==20000 else 'synthetic')
    except BaseException as exc:
        if bundle[-1]._phase=='ready':save('incomplete')
        model.dump(out/'failure.json',{'error':repr(exc),'completed_updates':bundle[-1].completed_steps,'phase':bundle[-1]._phase,'primary_endpoint_complete':False})
        raise
    finally:
        for sig,handler in handlers.items():signal.signal(sig,handler)

def resume_fit(ctx,out,seed,auth,*,end_monotonic):
    if ctx.device.type!='cuda':raise RuntimeError('Native continuation requires guarded GPU admission; CPU fixtures use synthetic helper only')
    import torch
    import released_adapter as model
    from state_fingerprint import fingerprint
    parent=auth['continuation'];path=Path(parent['checkpoint_path'])
    b.require(seed==0 and b.sha(path)==parent['checkpoint_SHA256'],'Parent checkpoint or seed differs')
    q=torch.load(path,map_location='cpu',weights_only=False);validate_parent(q,auth)
    bundle,generators,rng,usage,streams=model.restore(ctx,q)
    b.require(fingerprint(q['policy'])==fingerprint(bundle[-1].state_dict()),'Full original CUDA policy restore differs')
    restored=model.snapshot(ctx,bundle,generators,rng,usage,seed,streams,q['status'])
    fp=fingerprint(q)
    b.require(fp==fingerprint(restored)==parent['whole_parent_scientific_fingerprint'],
        'Exact models/optimizers/usage/cell/count/numpy/CPU-CUDA Gamma/native streams restore differs')
    b.require(bundle[-1].completed_steps==usage.steps==10842,'Parent native or usage clock reset')
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    for diagnostic in parent['diagnostic_checkpoints']:
        source=Path(diagnostic['immutable_copy_path']);b.require(b.sha(source)==diagnostic['SHA256'],'Original diagnostic checkpoint changed')
        destination=out/('checkpoint%05d.pt'%diagnostic['completed_updates']);shutil.copyfile(source,destination)
        b.require(b.sha(destination)==diagnostic['SHA256'],'Diagnostic checkpoint copy differs')
    history=parent['training_log'];source=Path(history['immutable_copy_path']);b.require(b.sha(source)==history['SHA256'],'Original training history changed')
    shutil.copyfile(source,out/'learning_curve.jsonl')
    model.dump(out/'native_parent_restore.json',{'status':'EXACT_FULL_NATIVE_AND_ALL_SCIENTIFIC_PARENT_STATE_RESTORED_ON_CUDA',
        'parent_checkpoint_path':str(path),'parent_checkpoint_SHA256':b.sha(path),'parent_policy_fingerprint':fingerprint(q['policy']),
        'whole_parent_scientific_fingerprint':fp,'initial_completed_updates':10842,'remaining_model_updates':9158,'fixed_endpoint':20000,
        'logical_native_device':q['policy']['device'],'physical_gpu_uuid':auth['gpu_uuid'],
        'original_absolute_deadline_unix':parent['original_absolute_deadline_unix'],'native_or_usage_or_data_clock_reset':False,
        'both_original_diagnostic_checkpoints_and_log_prefix_byte_exact':True})
    del q,restored
    return _run_remaining(ctx,bundle,generators,rng,usage,seed,streams,out,end_monotonic=end_monotonic)
