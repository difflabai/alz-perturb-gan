"""Serial matched-GPU training / CPU inference; unchanged confirmation gates."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
import cov_common as C
import resource_common as U
import generate_covariance as D
import compare_covariance as COMPARE
import audit_covariance as AUDIT


def call(label,module,args):
    sources=C.full_launch_ready()
    C.B.require(C.disk_bytes()<2850000000,'less than150MB worker headroom')
    command=[sys.executable,'-B',str(C.HERE/module),*map(str,args)]
    env=dict(os.environ,OMP_NUM_THREADS='3',OPENBLAS_NUM_THREADS='3',MKL_NUM_THREADS='3')
    if module=='train_gpu.py':env['ALZ_COVARIANCE_GPU']='1'
    else:env['CUDA_VISIBLE_DEVICES']=''
    with (C.LOG/(label+'.log')).open('a') as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=C.HERE.parents[1])
        C.B.write(C.ROOT/'status.json',{'status':'running','phase':label,'runner_pid':os.getpid(),
            'worker_pid':child.pid,'command':command,'time':time.time(),'default_changed':False})
        code=child.wait()
    C.B.require(code==0,f'{label} exited{code}; inspect before resuming')
    C.B.require(C.sources()==sources and C.disk_bytes()<3000000000,'post-worker source/cap differs')
    print('completed',label,flush=True)


def inference(recipe,fold,seed):
    sources=C.B.P.read(C.ROOT/'source_manifest.json')
    D.require_confirmation_eligible(fold,sources)
    if D.finalized(recipe,fold,seed,sources):return
    call(f'{recipe}_f{fold}r{seed}','generate_covariance.py',
        ['--recipe',recipe,'--fold',fold,'--replicate',seed])
    C.B.require(D.finalized(recipe,fold,seed,sources),'inference worker did not finalize')


def fit(role,fold,seed):
    D.require_confirmation_eligible(fold,C.sources())
    path=C.fit_path(role,fold,seed)
    if (path/'study_training_receipt.json').exists():D.new_fit(role,fold,seed,allow_pruned=True);return
    call(f'train_{role}_f{fold}r{seed}','train_gpu.py',['--fold',fold,'--seed',seed,'--role',role])
    D.new_fit(role,fold,seed)


def phase(folds):
    for fold in folds:
        C.B.write(C.ROOT/'cache_verification'/f'fold{fold}.json',C.B.P.verify_cache(fold))
        for seed in (0,1):
            for recipe in ('original32','current32','empirical32'):inference(recipe,fold,seed)
            fit('control',fold,seed)
            if (fold,seed)==(0,0):
                if not (C.ROOT/'control_orchestration_identity.json').exists():
                    call('full_real_control_identity','generate_covariance.py',['--recipe','identity'])
                D.verify_identity(C.sources())
            inference('training_control32',fold,seed)
            if not (C.fit_path('control',fold,seed)/'control_weight_pruning.json').exists():
                call(f'prune_control_f{fold}r{seed}','generate_covariance.py',
                    ['--recipe','prune_control','--fold',fold,'--replicate',seed])
            fit('candidate',fold,seed);inference('covariance32',fold,seed)


def main():
    argparse.ArgumentParser(description=__doc__).parse_args();C.cpu()
    C.ROOT.mkdir(parents=True,exist_ok=True);C.LOG.mkdir(parents=True,exist_ok=True)
    # Compute children take all nine leases. Holding these same locks here
    # would prevent the child from acquiring them; the parent owns this
    # exclusive orchestration lock instead and waits for one child at a time.
    with (C.ROOT/'orchestration.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        expected=C.full_launch_ready();C.prerequisite()
        manifest=C.ROOT/'source_manifest.json'
        C.B.require(not manifest.exists() or C.B.P.read(manifest)==expected,'existing queue source differs')
        C.B.write(manifest,expected)
        try:
            phase(range(3));screen=COMPARE.report(tuple(range(3)));C.B.write(C.ROOT/'screen_comparison.json',screen)
            with C.leases():C.B.write(C.ROOT/'artifact_audit_screen_complete.json',AUDIT.audit(tuple(range(3))))
            confirmation=None
            if screen['passes']:
                phase(range(3,9));confirmation=COMPARE.report(tuple(range(3,9)),True)
                C.B.write(C.ROOT/'confirmation_comparison.json',confirmation)
                with C.leases():C.B.write(C.ROOT/'artifact_audit_full_complete.json',AUDIT.audit(tuple(range(9))))
            C.B.write(C.ROOT/'status.json',{'status':'completed','screen_passes':screen['passes'],
                'confirmation_passes':None if confirmation is None else confirmation['passes'],
                'default_changed':False,'logical_task_bytes':C.disk_bytes(),'time':time.time()})
        except BaseException as exc:
            C.B.write(C.ROOT/'status.json',{'status':'failed','error':repr(exc),'time':time.time(),'default_changed':False})
            raise


if __name__=='__main__':main()
