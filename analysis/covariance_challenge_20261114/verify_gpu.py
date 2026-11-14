"""Frozen, bounded native GPU parity/resume/resource proof; no held scores."""
import os
SELECTOR=os.environ.get('ALZ_COVARIANCE_GPU','1')
import argparse
import copy
from datetime import datetime,timezone
import gc
import io
import json
from pathlib import Path
import subprocess
import sys
import time
import cov_common as C
import resource_common as U

SCR=C.MODES/'gpu_training_verification'
ENDPOINT=200


def snapshot(ctx,b,g,r,u,s):
    return C.T.snapshot(ctx,b,g,r,u,0,s,'incomplete',endpoint=ENDPOINT,
                        stream_root=C.T.STREAM_ROOT,wall_seconds=0.)


def ordinary():
    ctx=C.T.Context(C.RUNS/'caches/hvg2000/fold0',U.TRAIN_DEVICE,C.T.parse_arm('pbulk'))
    b,g,r,u,s=C.T.build(ctx,0);initial=snapshot(ctx,b,g,r,u,s)
    metrics=[C.T.advance(ctx,b,g,r,u,execution_limit=ENDPOINT) for _ in range(5)]
    return {'initial':initial,'final':snapshot(ctx,b,g,r,u,s),'metrics':metrics}


def reference():
    U.verification_ready();U.setup_gpu(SELECTOR);opened=set()
    cache,_=C.train_scope(0,opened);out=SCR/'ordinary_without_plugin.pt'
    C.B.require(not out.exists(),'existing reference scratch')
    # Do not import train_covariance here: it registers covmom. Use the
    # same exact-byte checkpoint preflight directly for this one archive.
    buffer=io.BytesIO();C.torch.save(ordinary(),buffer);size=buffer.getbuffer().nbytes
    C.B.require(C.disk_bytes()+size<3000000000,'reference checkpoint would exceed3GB')
    with out.open('wb') as stream:stream.write(buffer.getbuffer())
    peak={'max_serialized_checkpoint_bytes':size,'peak_logical_task_bytes':C.disk_bytes()}
    C.check_train_scope(cache,opened)
    print(json.dumps({'cache_files_opened':sorted(opened),'gpu_resource':U.gpu_resource(),**peak}),flush=True)


def main():
    started=time.monotonic();expected=U.verification_ready();opened=set()
    C.ROOT.mkdir(parents=True,exist_ok=True);SCR.mkdir(parents=True,exist_ok=True)
    out=C.ROOT/'gpu_training_verification.json'
    C.B.require(not out.exists() and not any(p.is_file() for p in SCR.rglob('*')),'existing GPU verification artifacts')
    cache,anchor=C.train_scope(0,opened)
    C.prerequisite();C.B.require(C.disk_bytes()<2850000000,'insufficient GPU verification headroom')
    peak={'peak_logical_task_bytes':C.disk_bytes()};hashes={}
    # Reference process finishes before parent CUDA context: one GPU worker.
    env=dict(os.environ,ALZ_COVARIANCE_GPU=SELECTOR,CUDA_VISIBLE_DEVICES=SELECTOR,
        CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='3',OPENBLAS_NUM_THREADS='3',MKL_NUM_THREADS='3')
    result=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--reference'],
        env=env,check=True,capture_output=True,text=True)
    reference_scope=json.loads(result.stdout.strip().splitlines()[-1])
    C.B.require(reference_scope['cache_files_opened']==sorted(str((cache/n).resolve())
        for n in ('manifest.json','space.npz','train/counts.npz','train/metadata.npz')),'GPU reference readscope differs')
    for key in ('peak_logical_task_bytes','max_serialized_checkpoint_bytes'):
        peak[key]=max(peak.get(key,0),reference_scope[key])
    original_path=SCR/'ordinary_without_plugin.pt';hashes[original_path.name]=C.B.sha(original_path)
    old=C.torch.load(original_path,map_location='cpu',weights_only=False)
    import covmom_gpu  # registration only after reference process completes
    import train_covariance as TRAIN
    U.setup_gpu(SELECTOR)
    with TRAIN.watch_checkpoint_peak(peak):
        current=ordinary()
        C.B.require(C.T._same(current,old),'ordinary GPU initialization/first5updates changed')
        original_path.unlink();del old,current;gc.collect();C.torch.cuda.empty_cache()
        print('verified exact ordinary GPU initialization/first5updates parity',flush=True)
        ctx=C.T.Context(cache,U.TRAIN_DEVICE,C.T.parse_arm(C.ARM))
        arm=next(a for a in ctx.plugins if a.name=='d.covmom')
        cpu_context=C.T.Context(cache,'cpu',ctx.flags)
        cpu_arm=next(a for a in cpu_context.plugins if a.name=='d.covmom')
        C.B.require(C.T._same(arm.state(),cpu_arm.state()) and arm.info==cpu_arm.info,'GPU saved arm does not equal CPU recomputation')
        del cpu_context,cpu_arm;gc.collect()
        C.B.require(ctx.features.n_features==4650 and ctx.topt.kappa==C.KAPPA,'wrong GPU observation/cap')
        cfgargs=C.T.parser().parse_args(['--cache',str(cache),'--arm',C.ARM,'--seed','0','--updates','200',
            '--device',U.TRAIN_DEVICE,'--threads','3','--gpu-mem-fraction','.04','--out-root',str(SCR)])
        config=C.T.make_config(ctx,cfgargs,C.T.STREAM_ROOT)
        C.B.require('d.covmom' in config['departures'] and 't.kappa@0.679975' in config['departures'],'GPU departures missing')
        b,g,r,u,s=C.T.build(ctx,0);first=last=None
        for step in range(1,201):
            metrics=C.T.advance(ctx,b,g,r,u,execution_limit=ENDPOINT)
            if step==1:first=copy.deepcopy(metrics)
            if step==100:
                p=SCR/'split.partial.pt';C.torch.save(snapshot(ctx,b,g,r,u,s),p);p.replace(SCR/'split.pt')
                hashes['split.pt']=C.B.sha(SCR/'split.pt')
            if step in (50,100,150,200):
                C.B.require(C.sources()==expected and C.disk_bytes()<3000000000,'GPU smoke source/cap changed')
                C.B.write(C.ROOT/'gpu_training_verification_progress.json',{'status':'running','phase':'fresh_candidate',
                    'completed_updates':step,'elapsed_seconds':time.monotonic()-started,**peak})
                U.gpu_resource();print('GPU candidate smoke completed',step,flush=True)
            last=metrics
        final=snapshot(ctx,b,g,r,u,s)
        p=SCR/'continuous_final.partial.pt';C.torch.save(final,p);p.replace(SCR/'continuous_final.pt')
        hashes['continuous_final.pt']=C.B.sha(SCR/'continuous_final.pt')
        del b,g,r,u,s,ctx;gc.collect();C.torch.cuda.empty_cache()
        saved=C.torch.load(SCR/'split.pt',map_location='cpu',weights_only=False)
        ctx=C.T.Context(cache,U.TRAIN_DEVICE,C.T.parse_arm(C.ARM))
        b,g,r,u,s=C.T.restore(ctx,saved,0);del saved;gc.collect()
        for step in range(101,201):resumed=C.T.advance(ctx,b,g,r,u,execution_limit=ENDPOINT)
        actual=snapshot(ctx,b,g,r,u,s)
        C.B.require(C.T._same(actual,final) and C.T._same(last,resumed),'GPU resume is not bitwise identical')
        for key in ('G_gradient_norm','D_gradient_norm','prior_gradient_norm'):
            C.B.require(C.np.isfinite(first[key]) and first[key]>0,'GPU finite/nonzero gradient check failed')
        C.B.require(final['torch_device_rng'] is not None,'CUDA RNG missing from native snapshot')
        for name in ('split.pt','continuous_final.pt'):(SCR/name).unlink()
    C.check_train_scope(cache,opened)
    C.B.require(C.sources()==expected and C.disk_bytes()<3000000000,'GPU final source/cap check failed')
    C.B.require(not any(p.is_file() for p in SCR.rglob('*')),'GPU smoke weights not pruned')
    info={'status':U.GPU_STATUS,'recorded_utc':datetime.now(timezone.utc).isoformat(),'sources':expected,
        'gpu_verification_manifest_sha256':C.B.sha(C.HERE/'gpu_verification_source_manifest_prelaunch.json'),
        'cpu_training_verification_sha256':U.cpu_proof(),
        'ordinary_initial_and_5updates_full_snapshot_bitwise_equal':True,'ordinary_reference_and_imported_update_calls':10,
        'candidate_fresh_steps':200,'candidate_repeated_after_resume_steps':100,'candidate_total_native_update_calls':300,
        'candidate_declared_recipe_horizon':20000,'candidate_resume_split_step':100,
        'candidate_full_snapshot_resume_bitwise_equal':True,'candidate_context_recomputed_before_restore':True,
        'candidate_last_metrics_resume_bitwise_equal':True,'candidate_first_native_gradient_metrics':first,
        'CPU_reference_state_recomputation_exact':True,'feature_width':4650,'native_kappa':C.KAPPA,
        'config_departures':config['departures'],'arm_info':arm.info,'gpu_resource':U.gpu_resource(),
        'reference_gpu_resource':reference_scope['gpu_resource'],'cache_files_opened':sorted(opened),
        'reference_cache_files_opened':reference_scope['cache_files_opened'],
        'cache_manifest_sha256':anchor['manifest_sha256'],'heldout_payloads_opened':False,
        'pruned_weight_sha256':hashes,**peak,'logical_task_bytes':C.disk_bytes(),
        'runtime':{'python':sys.version,'numpy':C.np.__version__,'torch':str(C.torch.__version__),
            'particlegan':str(C.T.VERSION),'compute_threads':C.torch.get_num_threads(),
            'cpu_affinity':sorted(os.sched_getaffinity(0))},
        'seconds_total':time.monotonic()-started,'default_changed':False,'screen_fits_started':False,
        'interpretation':'Native GPU parity/gradient/resume/resource proof only; no quality or eligibility inference'}
    C.B.write(out,info);print(json.dumps({'status':info['status'],'seconds':info['seconds_total'],**peak}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference',action='store_true');a=p.parse_args()
    if a.reference:reference()
    else:
        with C.leases():main()
