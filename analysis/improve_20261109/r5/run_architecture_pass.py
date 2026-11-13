"""Resumable serial GPU queue with explicit artifacts, scoring and provenance.

Run intervals first; train schedules named, predeclared 20k ablations. Never
changes the default or touches sealed donors. Each subprocess has <=3 threads
and a .04 GPU memory cap. Completed scored generated modes are hash-pruned.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import fcntl
import hashlib

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
RUNS=ROOT/'runs/improve_20261109'
LOG=RUNS/'logs_r5/architecture_pass'
SCORES=RUNS/'results_r5/architecture_pass'
MODES=RUNS/'smoke_r5/architecture_pass'
FITS=RUNS/'fits_r5/architecture_pass'
ARMS=('d.randproj,pbulk','g.split,pbulk','g.hier,pbulk')


def atomic(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.partial.json')
    tmp.write_text(json.dumps(value,indent=2)+'\n')
    tmp.replace(path)


def call(label,module,args,gpu=False):
    env=dict(os.environ,OMP_NUM_THREADS='3',OPENBLAS_NUM_THREADS='3',MKL_NUM_THREADS='3',PYTHONDONTWRITEBYTECODE='1')
    if gpu and not env.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('set CUDA_VISIBLE_DEVICES explicitly')
    if not gpu:
        env['CUDA_VISIBLE_DEVICES']=''
    command=[sys.executable,'-B',str(HERE/module),*map(str,args)]
    log=LOG/(label+'.log')
    LOG.mkdir(parents=True,exist_ok=True)
    atomic(SCORES/'queue_status.json',{'status':'running','phase':label,'command':command,'pid':os.getpid(),'time':time.time()})
    with log.open('a') as stream:
        stream.write('\nCOMMAND '+json.dumps(command)+'\n');stream.flush()
        subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
    print('completed',label,flush=True)


def score(label,generated,cache,out):
    if out.exists() and not generated.exists():
        # Pruning is allowed only after a successful score, with its source hash.
        old=json.loads(out.read_text())
        if not old.get('pruned_source_sha256'):
            raise FileNotFoundError(generated)
        return
    call(label,'architecture_score.py',['--generated',generated,'--cache',cache,'--out',out])


def prune(generated,score_path):
    result=json.loads(score_path.read_text())
    result['pruned_source_sha256']=result['fingerprint']['generated_sha256']
    atomic(score_path,result)
    generated.unlink(missing_ok=True)


def intervals(folds,seeds):
    for f in folds:
        cache=RUNS/f'caches/hvg2000/fold{f}'
        for s in seeds:
            fit=RUNS/f'fits_r3/pbulk/hvg2000/fold{f}/seed{s}'
            old=fit/'generated_biascal@stagehalf.npz'
            score(f'baseline32_f{f}s{s}',old,cache,SCORES/f'baseline32/fold{f}/seed{s}.json')
            out=MODES/f'baseline128/fold{f}/seed{s}.npz'
            dest=SCORES/f'baseline128/fold{f}/seed{s}.json'
            if not dest.exists() and not out.exists():
                call(f'baseline128_generate_f{f}s{s}','generate_more.py',
                     ['--fit',fit,'--cache',cache,'--calibration-receipt',old.with_suffix('.receipt.json'),
                      '--out',out,'--draws','128','--device','cuda','--gpu-mem-fraction','.04'],gpu=True)
            score(f'baseline128_score_f{f}s{s}',out,cache,dest)
            prune(out,dest)


def train(folds,seeds,arms):
    for f in folds:
        cache=RUNS/f'caches/hvg2000/fold{f}'
        for s in seeds:
            for arm in arms:
                fit=FITS/arm/f'hvg2000/fold{f}/seed{s}'
                out=MODES/arm/f'hvg2000/fold{f}/seed{s}.npz'
                dest=SCORES/arm/f'fold{f}/seed{s}.json'
                if dest.exists():
                    continue
                label=f'{arm}_f{f}s{s}'
                provenance=fit/'architecture_source.json'
                sources={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in
                         ('trainer.py','arms.py','arm_d_randproj.py','arm_g_hier.py','arm_pbulk.py')}
                if provenance.exists():
                    if json.loads(provenance.read_text()) != sources:
                        raise ValueError(f'architecture source changed for existing fit {fit}')
                else:atomic(provenance,sources)
                if not (fit/'final.pt').exists() or not (fit/'generated.npz').exists():
                    call(label+'_train','trainer.py',['--cache',cache,'--arm',arm,'--seed',s,'--updates',20000,
                         '--out-root',FITS,'--device','cuda','--threads',3,'--gpu-mem-fraction','.04'],gpu=True)
                if not out.exists():
                    call(label+'_generate','generate.py',['--fit',fit,'--cache',cache,'--mode','biascal@stagehalf',
                         '--out',out,'--device','cuda','--gpu-mem-fraction','.04'],gpu=True)
                score(label+'_score',out,cache,dest)
                prune(out,dest)


def smoke():
    cache=RUNS/'caches/hvg2000/fold0'
    for arm in ARMS:
        root=MODES/'gpu_smoke'
        fit=root/arm/'hvg2000/fold0/seed0'
        if (fit/'generated.npz').exists():continue
        call(arm+'_gpu_smoke','trainer.py',['--cache',cache,'--arm',arm,'--seed',0,'--updates',300,
             '--out-root',root,'--device','cuda','--threads',3,'--gpu-mem-fraction','.04'],gpu=True)


def encoder():
    for seed in (0,1):
        if (MODES/f'encoder/fold0/seed{seed}/evaluation.json').exists():continue
        call(f'encoder_f0s{seed}','donor_encoder.py',['--fold',0,'--seed',seed,'--steps',2000,
             '--device','cuda','--gpu-mem-fraction','.04'],gpu=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['intervals','train','smoke','encoder','full'])
    p.add_argument('--folds',default='0,1,2')
    p.add_argument('--seeds',default='0,1')
    p.add_argument('--arms',nargs='+',choices=ARMS,default=list(ARMS))
    a=p.parse_args()
    folds=[int(x) for x in a.folds.split(',')]
    seeds=[int(x) for x in a.seeds.split(',')]
    if any(f not in range(9) for f in folds) or any(s<0 for s in seeds):
        p.error('invalid development fold or seed')
    LOG.mkdir(parents=True,exist_ok=True)
    lock=(LOG/'queue.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:p.error('another architecture queue owns the task GPU lease')
    try:
        if a.phase=='intervals':intervals(folds,seeds)
        elif a.phase=='smoke':smoke()
        elif a.phase=='encoder':encoder()
        elif a.phase=='train':train(folds,seeds,a.arms)
        else:
            smoke()
            encoder()
            intervals(list(range(9)),seeds)
            train(folds,seeds,a.arms)
        atomic(SCORES/'queue_status.json',{'status':'completed','phase':a.phase,'time':time.time()})
    except BaseException as exc:
        atomic(SCORES/'queue_status.json',{'status':'failed','phase':a.phase,'error':repr(exc),'time':time.time()})
        raise


if __name__=='__main__':main()
