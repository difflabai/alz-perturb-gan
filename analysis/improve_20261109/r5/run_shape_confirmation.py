"""CPU matched confirmation of the fixed count-shape read-out, no retraining."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='3'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import fcntl

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
RUNS=ROOT/'runs/improve_20261109'
BASE=RUNS/'results_r5/architecture_pass/shape_confirmation'
MODES=RUNS/'smoke_r5/architecture_pass/shape_confirmation'
LOGS=RUNS/'logs_r5/architecture_pass/shape_confirmation'
RECIPES={'default':'biascal@stagehalf','count_shape':'cellshape@0.5:biascal:stagehalf'}


def atomic(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.partial.json')
    temp.write_text(json.dumps(value,indent=2)+'\n')
    temp.replace(path)


def call(label,module,args):
    command=[sys.executable,'-B',str(HERE/module),*map(str,args)]
    atomic(BASE/'status.json',{'phase':label,'status':'running','pid':os.getpid(),'command':command,'time':time.time()})
    LOGS.mkdir(parents=True,exist_ok=True)
    with (LOGS/(label+'.log')).open('a') as stream:
        stream.write('\nCOMMAND '+json.dumps(command)+'\n');stream.flush()
        subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,env=dict(os.environ),check=True)
    print('completed',label,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folds',default='3,4,5,6,7,8')
    p.add_argument('--seeds',default='0,1')
    a=p.parse_args()
    folds=[int(x) for x in a.folds.split(',')]
    seeds=[int(x) for x in a.seeds.split(',')]
    if any(f not in range(3,9) for f in folds) or any(s not in (0,1) for s in seeds):
        p.error('this confirmation is folds3-8,seeds0-1 only')
    BASE.mkdir(parents=True,exist_ok=True)
    lock=(BASE/'queue.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:p.error('shape confirmation already running')
    try:
        for f in folds:
            cache=RUNS/f'caches/hvg2000/fold{f}'
            for s in seeds:
                fit=RUNS/f'fits_r3/pbulk/hvg2000/fold{f}/seed{s}'
                for recipe,mode in RECIPES.items():
                    output=MODES/recipe/f'fold{f}/seed{s}.npz'
                    score=BASE/recipe/f'fold{f}/seed{s}.json'
                    label=f'{recipe}_f{f}s{s}'
                    if score.exists():
                        old=json.loads(score.read_text())
                        if old.get('pruned_source_sha256'):continue
                    if not output.exists():
                        module='mode_cellshape.py' if recipe=='count_shape' else 'generate.py'
                        call(label+'_generate',module,['--fit',fit,'--cache',cache,'--out',output,'--mode',mode,'--device','cpu'])
                    call(label+'_score','architecture_score.py',['--generated',output,'--cache',cache,'--out',score])
                    value=json.loads(score.read_text())
                    value['pruned_source_sha256']=value['fingerprint']['generated_sha256']
                    value['generation_device']='cpu'
                    value['generation_recipe']=mode
                    atomic(score,value)
                    output.unlink()
        atomic(BASE/'status.json',{'status':'completed','folds':folds,'seeds':seeds,'time':time.time()})
    except BaseException as exc:
        atomic(BASE/'status.json',{'status':'failed','error':repr(exc),'time':time.time()})
        raise


if __name__=='__main__':main()
