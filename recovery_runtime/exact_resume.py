"""Exact interrupted PG8 state restoration; no source/model/seed substitution."""
from pathlib import Path
import signal,time
import torch
import variant_pg_adapter as variant
import variant_pg_update as update
from continuation import same
from contracts import dump,sha

def restore_exact(ctx,checkpoint,seed,*,expected_initial):
    assert type(expected_initial) is int and 0<expected_initial<20000
    assert checkpoint['schema']==1 and checkpoint['method']=='pg'
    assert checkpoint['status'] in ('incomplete','recovery')
    assert checkpoint['seed']==seed and checkpoint['fold']==ctx.fold
    assert checkpoint['completed_updates']==checkpoint['policy']['completed_steps']==expected_initial
    assert checkpoint['fixed_endpoint']==20000 and checkpoint['recipe']['total_steps']==20000
    assert checkpoint['variant_config']==dict(update.CONFIG)
    assert checkpoint['cache_manifest_SHA256']==sha(ctx.cache/'manifest.json')
    b,g,r,u,streams=variant.restore(ctx,checkpoint)
    same(checkpoint,variant.snapshot(ctx,b,g,r,u,seed,streams,checkpoint['status']))
    assert b[-1]._phase=='ready'
    return b,g,r,u,streams

def qualify_exact(ctx,checkpoint,*,expected_initial,branch_directory=None):
    seed=checkpoint['seed'];b,g,r,u,streams=restore_exact(ctx,checkpoint,seed,expected_initial=expected_initial)
    expected_metrics=variant.advance(ctx,b,g,r,u,execution_limit=20000)
    expected=variant.snapshot(ctx,b,g,r,u,seed,streams,'qualification_only')
    rb,rg,rr,ru,rs=restore_exact(ctx,checkpoint,seed,expected_initial=expected_initial)
    observed_metrics=variant.advance(ctx,rb,rg,rr,ru,execution_limit=20000)
    observed=variant.snapshot(ctx,rb,rg,rr,ru,seed,rs,'qualification_only')
    same(expected_metrics,observed_metrics);same(expected,observed)
    assert observed['completed_updates']==expected_initial+1 and observed['fixed_endpoint']==20000
    if branch_directory is not None:
        folder=Path(branch_directory);folder.mkdir(exist_ok=False,parents=True)
        torch.save(expected,folder/'branch0.pt');torch.save(observed,folder/'branch1.pt')
    return {'status':'EXACT_INTERRUPTED_STATE_AND_NEXT_UPDATE_BITWISE_VERIFIED','qualification_only':True,
            'parent_completed_updates':expected_initial,'qualified_completed_updates':expected_initial+1,
            'complete_initial_state_restored_exactly':True,'complete_next_state_and_metrics_bitwise_equal':True,
            'executed_optimizer_pairs':2,'replay_branches':2,'updates_per_branch':1,
            'native_recipe_schedule_horizon':20000,'caller_execution_limit':20000,
            'calibration_or_evaluation_scores_generated':False,'automatic_successor':False},observed

def fit_resume(ctx,out,seed,*,checkpoint,checkpoint_SHA256,expected_initial,end_monotonic):
    """Continue only the exact interrupted model to its original20k endpoint."""
    if ctx.device.type!='cuda':raise RuntimeError('Actual recovery requires separately qualified guarded CUDA admission')
    out=Path(out);out.mkdir(exist_ok=False,parents=True)
    b,g,r,u,streams=restore_exact(ctx,checkpoint,seed,expected_initial=expected_initial)
    stops=[];handlers={s:signal.signal(s,lambda sig,frame:stops.append(sig)) for s in [signal.SIGINT,signal.SIGTERM]}
    def save(status):
        path=out/('final.pt' if status=='completed' else 'recovery.pt');temporary=path.with_suffix('.partial.pt')
        torch.save(variant.snapshot(ctx,b,g,r,u,seed,streams,status),temporary);temporary.replace(path)
        dump(out/'checkpoint_receipt.json',{'path':path.name,'SHA256':sha(path),'status':status,'completed_updates':b[-1].completed_steps,
             'initial_completed_updates':expected_initial,'new_model_updates':b[-1].completed_steps-expected_initial,
             'resume_parent_checkpoint_SHA256':checkpoint_SHA256,'native_recipe_schedule_horizon':20000})
        return path
    try:
        while b[-1].completed_steps<20000:
            if stops or time.monotonic()>=end_monotonic:raise InterruptedError('Finite exact recovery allocation ending')
            metrics=variant.advance(ctx,b,g,r,u,execution_limit=20000)
            if metrics['step']%250==0 or metrics['step']==expected_initial+1:dump(out/'progress.json',metrics)
            if metrics['step']%1000==0:save('recovery')
        return b,save('completed')
    except BaseException as exc:
        if b[-1]._phase=='ready':save('incomplete')
        dump(out/'failure.json',{'error':repr(exc),'completed_updates':b[-1].completed_steps,'phase':b[-1]._phase,'primary_endpoint_complete':False});raise
    finally:
        for s,h in handlers.items():signal.signal(s,h)
