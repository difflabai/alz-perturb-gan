"""Continue the exact native policy; preserve parent clocks and original deadline."""
import signal
import time
from pathlib import Path
import binding as b


def validate_parent(q,auth):
    c=auth['continuation'];n=c['completed_updates'];cell=b.cell(auth)
    b.require(q.get('schema')==1 and q.get('method')=='pg' and q.get('native_recipe')=='e22_routed','Native routed parent required')
    b.require(q.get('fold')==cell['fold'] and q.get('seed')==cell['seed'],'Parent fold/seed differs')
    b.require(q.get('status')=='incomplete' and q.get('fixed_endpoint')==20000,'Incomplete original20k parent required')
    b.require(q.get('completed_updates')==q.get('policy',{}).get('completed_steps')==n==7013,'Exact7013 parent boundary required')
    b.require(q['policy']['device']=='cuda:0','Original native CUDA device metadata must be retained')
    b.require(q.get('cache_manifest_SHA256')==b.sha(Path(auth['cache_dir'])/'manifest.json'),'Parent cache differs')
    b.require(q.get('variant_config')==auth['variant_config'] and q.get('routed_config')==auth['routed_config'],'Parent model configuration differs')
    b.require(q.get('fitting_contexts_seen')==q.get('protected_contexts_seen')==32*n,'Parent data sampler clocks differ')


def _run_remaining(ctx,runtime,out,*,end_monotonic,target_updates=20000):
    import torch
    import routed_pg_adapter as model
    if ctx.device.type=='cuda' and target_updates!=20000:raise ValueError('Native continuation retains original20k endpoint')
    out=Path(out);stops=[]
    handlers={s:signal.signal(s,lambda sig,frame:stops.append(sig)) for s in (signal.SIGINT,signal.SIGTERM)}
    def save(status):
        path=out/('final.pt' if status=='completed' else 'recovery.pt');temp=path.with_suffix('.partial.pt')
        torch.save(model.snapshot(ctx,runtime,status),temp);temp.replace(path)
        model.dump(out/'checkpoint_receipt.json',{'path':path.name,'SHA256':model.sha(path),'status':status,'completed_updates':runtime.policy.completed_steps})
        return path
    try:
        while runtime.policy.completed_steps<target_updates:
            if stops or time.monotonic()>=end_monotonic:raise InterruptedError('Original finite allocation ending; retain completed continuation checkpoint')
            metrics=model.advance(ctx,runtime)
            if metrics['step']%250==0:model.dump(out/'progress.json',metrics)
            if metrics['step']%1000==0:save('recovery')
        return runtime,save('completed' if target_updates==20000 else 'synthetic')
    except BaseException as exc:
        if runtime.policy._phase=='ready':save('incomplete')
        model.dump(out/'failure.json',{'error':repr(exc),'completed_updates':runtime.policy.completed_steps,'phase':runtime.policy._phase,'primary_endpoint_complete':False})
        raise
    finally:
        for sig,handler in handlers.items():signal.signal(sig,handler)


def resume_fit(ctx,out,seed,auth,*,end_monotonic):
    if ctx.device.type!='cuda':raise RuntimeError('Native continuation requires separately guarded GPU admission; CPU fixtures use synthetic helper only')
    import torch
    import routed_pg_adapter as model
    from state_fingerprint import fingerprint
    parent=auth['continuation'];path=Path(parent['checkpoint_path'])
    b.require(seed==0 and b.sha(path)==parent['checkpoint_SHA256'],'Frozen parent seed/checkpoint differs')
    q=torch.load(path,map_location='cpu',weights_only=False);validate_parent(q,auth)
    runtime=model.restore(ctx,q)
    b.require(fingerprint(q['policy'])==fingerprint(runtime.policy.state_dict()),'Original native CUDA policy restore is not byte-identical')
    for name,rng in [('fitting_rng',runtime.fitting_rng),('protected_rng',runtime.protected_rng)]:
        b.require(fingerprint(q[name])==fingerprint(rng.bit_generator.state),'Parent data sampler state differs')
    b.require(runtime.policy.completed_steps==7013 and runtime.fitting_contexts_seen==runtime.protected_contexts_seen==32*7013,'Parent native/data clocks reset')
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    model.dump(out/'native_parent_restore.json',{'status':'EXACT_NATIVE_PARENT_CUDA_POLICY_AND_BOTH_DATA_SAMPLERS_RESTORED',
        'parent_checkpoint_path':str(path),'parent_checkpoint_SHA256':b.sha(path),'parent_policy_fingerprint':fingerprint(q['policy']),
        'initial_completed_updates':7013,'remaining_model_updates':12987,'fixed_endpoint':20000,
        'logical_native_device':q['policy']['device'],'physical_gpu_uuid':auth['gpu_uuid'],
        'original_absolute_deadline_unix':parent['original_absolute_deadline_unix'],'native_or_data_clock_reset':False})
    del q
    return _run_remaining(ctx,runtime,out,end_monotonic=end_monotonic)
