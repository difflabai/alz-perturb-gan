"""Bounded TRAIN-only native parity and 300-call candidate smoke verification.

No screen metric, model selection, or calibration is computed. Candidate
advances200 fresh steps and repeats100 after restoring its step100 snapshot,
so there are exactly300 native candidate update calls, horizon20k throughout.
"""
import argparse
import copy
from datetime import datetime,timezone
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import cov_common as C

SCR=C.MODES/'training_verification'
ENDPOINT=200


def snapshot(ctx,b,g,r,u,s,status='incomplete'):
    return C.T.snapshot(ctx,b,g,r,u,0,s,status,endpoint=ENDPOINT,
                        stream_root=C.T.STREAM_ROOT,wall_seconds=0.)


def ordinary():
    ctx=C.T.Context(C.RUNS/'caches/hvg2000/fold0','cpu',C.T.parse_arm('pbulk'))
    b,g,r,u,s=C.T.build(ctx,0);initial=snapshot(ctx,b,g,r,u,s)
    metrics=[C.T.advance(ctx,b,g,r,u,execution_limit=ENDPOINT) for _ in range(5)]
    return {'initial':initial,'final':snapshot(ctx,b,g,r,u,s),'metrics':metrics}


def reference():
    C.cpu();opened=set()
    C.B.require(C.verification_sources()==C.B.P.read(C.HERE/'verification_source_manifest_prelaunch.json'),'reference sources changed')
    cache,_=C.train_scope(0,opened)
    out=SCR/'ordinary_without_plugin.pt'
    C.B.require(not out.exists(),'existing reference scratch; inspect before resuming')
    C.torch.save(ordinary(),out);C.check_train_scope(cache,opened)
    print(json.dumps({'reference_saved':str(out),'cache_files_opened':sorted(opened)}),flush=True)


def main():
    C.cpu();started=time.monotonic();opened=set()
    C.ROOT.mkdir(parents=True,exist_ok=True);SCR.mkdir(parents=True,exist_ok=True)
    out=C.ROOT/'training_verification.json'
    C.B.require(not out.exists() and not any(p.is_file() for p in SCR.rglob('*')),'existing verification artifacts; inspect before resuming')
    expected=C.B.P.read(C.HERE/'verification_source_manifest_prelaunch.json')
    C.B.require(C.verification_sources()==expected and len(expected)==213,'wrong verification freeze')
    cache,anchor=C.train_scope(0,opened)
    C.prerequisite();C.B.require(C.disk_bytes()<2850000000,'insufficient verification headroom')
    peak={'peak_logical_task_bytes':C.disk_bytes()};hashes={}
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='3',OPENBLAS_NUM_THREADS='3',MKL_NUM_THREADS='3')
    result=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--reference'],
                          env=env,check=True,capture_output=True,text=True)
    reference_scope=json.loads(result.stdout.strip().splitlines()[-1])
    C.B.require(reference_scope['cache_files_opened']==sorted(str((cache/n).resolve())
        for n in ('manifest.json','space.npz','train/counts.npz','train/metadata.npz')),'reference readscope differs')
    original_path=SCR/'ordinary_without_plugin.pt'
    hashes[original_path.name]=C.B.sha(original_path)
    old=C.torch.load(original_path,map_location='cpu',weights_only=False)
    import covmom  # import occurs only AFTER the unmodified reference process
    import train_covariance as TRAIN
    peak['peak_logical_task_bytes']=max(peak['peak_logical_task_bytes'],C.disk_bytes())
    with TRAIN.watch_checkpoint_peak(peak):
        current=ordinary()
        C.B.require(C.T._same(current,old),'ordinary initialization/first5updates changed by plug-in import')
        original_path.unlink();del old,current;gc.collect()
        print('verified exact ordinary CPU initialization/first5updates parity',flush=True)
        ctx=C.T.Context(cache,'cpu',C.T.parse_arm(C.ARM))
        C.B.require(ctx.features.n_features==4650 and ctx.topt.kappa==C.KAPPA,'wrong candidate observation/cap')
        cfgargs=C.T.parser().parse_args(['--cache',str(cache),'--arm',C.ARM,'--seed','0','--updates','200',
             '--device','cpu','--threads','3','--out-root',str(SCR)])
        config=C.T.make_config(ctx,cfgargs,C.T.STREAM_ROOT)
        C.B.require('d.covmom' in config['departures'] and 't.kappa@0.679975' in config['departures'],
                    'named observation/cap departures missing')
        b,g,r,u,s=C.T.build(ctx,0);split=None;first=None;last=None
        for step in range(1,201):
            metrics=C.T.advance(ctx,b,g,r,u,execution_limit=ENDPOINT)
            if step==1:first=copy.deepcopy(metrics)
            if step==100:
                split=snapshot(ctx,b,g,r,u,s)
                p=SCR/'split.partial.pt';C.torch.save(split,p);p.replace(SCR/'split.pt')
                hashes['split.pt']=C.B.sha(SCR/'split.pt')
            if step in (50,100,150,200):
                C.B.require(C.verification_sources()==expected and C.disk_bytes()<3000000000,'smoke source/cap changed')
                C.B.write(C.ROOT/'training_verification_progress.json',{'status':'running','phase':'fresh_candidate',
                    'completed_updates':step,'elapsed_seconds':time.monotonic()-started,**peak})
                print('candidate smoke completed',step,flush=True)
            last=metrics
        final=snapshot(ctx,b,g,r,u,s)
        p=SCR/'continuous_final.partial.pt';C.torch.save(final,p);p.replace(SCR/'continuous_final.pt')
        hashes['continuous_final.pt']=C.B.sha(SCR/'continuous_final.pt')
        del b,g,r,u,s,split;gc.collect()
        saved=C.torch.load(SCR/'split.pt',map_location='cpu',weights_only=False)
        ctx=C.T.Context(cache,'cpu',C.T.parse_arm(C.ARM))
        b,g,r,u,s=C.T.restore(ctx,saved,0);del saved;gc.collect()
        for step in range(101,201):
            resumed=C.T.advance(ctx,b,g,r,u,execution_limit=ENDPOINT)
            if step in (150,200):
                C.B.write(C.ROOT/'training_verification_progress.json',{'status':'running','phase':'resumed_candidate',
                    'completed_updates':step,'elapsed_seconds':time.monotonic()-started,**peak})
        actual=snapshot(ctx,b,g,r,u,s)
        C.B.require(C.T._same(actual,final) and C.T._same(last,resumed),'candidate resume is not bitwise identical')
        C.B.require(first['G_gradient_norm']>0 and first['D_gradient_norm']>0
            and first['prior_gradient_norm']>0,'native finite/nonzero gradient check failed')
        # The native step0 assertions additionally cover G mean/theta/P and
        # nonzero critic gradients on both covmom and pbulk columns.
        for name in ('split.pt','continuous_final.pt'):(SCR/name).unlink()
    C.check_train_scope(cache,opened)
    C.B.require(C.verification_sources()==expected and C.disk_bytes()<3000000000,'verification final source/cap check failed')
    C.B.require(not any(p.is_file() for p in SCR.rglob('*')),'smoke weights not pruned')
    arm=next(a for a in ctx.plugins if a.name=='d.covmom')
    info={'status':'passed_native_TRAIN_parity_and_candidate_smoke','recorded_utc':datetime.now(timezone.utc).isoformat(),
        'sources':expected,'verification_manifest_sha256':C.B.sha(C.HERE/'verification_source_manifest_prelaunch.json'),
        'ordinary_initial_and_5updates_full_snapshot_bitwise_equal':True,'ordinary_imported_reference_update_calls':10,
        'candidate_fresh_steps':200,'candidate_repeated_after_resume_steps':100,'candidate_total_native_update_calls':300,
        'candidate_declared_recipe_horizon':20000,'candidate_resume_split_step':100,'candidate_full_snapshot_resume_bitwise_equal':True,
        'candidate_context_recomputed_before_restore':True,
        'candidate_last_metrics_resume_bitwise_equal':True,'candidate_first_native_gradient_metrics':first,
        'feature_width':4650,'native_kappa':C.KAPPA,'config_departures':config['departures'],'arm_info':arm.info,
        'cache_files_opened':sorted(opened),'reference_cache_files_opened':reference_scope['cache_files_opened'],
        'cache_manifest_sha256':anchor['manifest_sha256'],'heldout_payloads_opened':False,
        'pruned_weight_sha256':hashes,**peak,'logical_task_bytes':C.disk_bytes(),
        'runtime':{'python':sys.version,'numpy':C.np.__version__,'torch':str(C.torch.__version__),
                   'particlegan':str(C.T.VERSION),'compute_threads':C.torch.get_num_threads(),
                   'cpu_affinity':sorted(os.sched_getaffinity(0))},
        'seconds_total':time.monotonic()-started,'default_changed':False,'screen_fits_started':False,
        'interpretation':'Finite/gradient/parity/resume/resource verification only; no quality or eligibility inference'}
    C.B.write(out,info)
    print(json.dumps({'status':info['status'],'seconds':info['seconds_total'],**peak}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference',action='store_true');a=p.parse_args()
    if a.reference:reference()
    else:
        with C.leases():main()
