"""Serial, resumable CPU challenge queue; never selects a default."""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import generate_recipe as W
import compare as C
import provenance as PV

ROOT = W.ROOT
LOGS = W.RUNS/'logs_r5/empirical_challenge_20261114'


def disk_bytes():
    roots = [W.RUNS/k/name for k in ('fits_r5','smoke_r5','results_r5','logs_r5')
             for name in ('architecture_pass','empirical_challenge_20261114')]
    output = subprocess.check_output(['du','-sb',*[str(p) for p in roots if p.exists()]], text=True)
    return sum(int(line.split('\t')[0]) for line in output.splitlines())


def job(recipe, fold, replicate):
    sources = W.source_manifest()
    if sources != json.loads((ROOT/'source_manifest.json').read_text()):
        raise RuntimeError('source changed between queue jobs; inspect before resuming')
    score = ROOT/recipe/f'fold{fold}/seed{replicate}.json'
    if W.check_finished(score, recipe, fold, replicate, sources):
        return
    if disk_bytes()>2_800_000_000:
        raise RuntimeError('insufficient headroom below3GB task cap; inspect owned outputs')
    label = f'{recipe}_f{fold}r{replicate}'
    command = [sys.executable,'-B',str(HERE/'generate_recipe.py'),'--recipe',recipe,
               '--fold',str(fold),'--replicate',str(replicate)]
    LOGS.mkdir(parents=True, exist_ok=True)
    with (LOGS/(label+'.log')).open('a') as stream:
        stream.write('\nCOMMAND '+json.dumps(command)+'\n');stream.flush()
        process = subprocess.Popen(command, cwd=HERE.parents[1], stdout=stream, stderr=subprocess.STDOUT)
        W.write(ROOT/'status.json', {'status':'running','recipe':recipe,'fold':fold,
                 'replicate':replicate,'runner_pid':os.getpid(),'worker_pid':process.pid,'time':time.time()})
        code = process.wait()
    if code:
        raise RuntimeError(f'{label} failed with exit{code}; inspect {LOGS/(label+".log")}')
    if not W.check_finished(score, recipe, fold, replicate, sources):
        raise RuntimeError('worker exited without finalized score')
    print('completed',label,flush=True)


def phase(folds, recipes):
    for fold in folds:
        if fold not in VERIFIED_CACHES:
            proof=PV.verify_cache(fold)
            W.write(ROOT/'cache_verification'/f'fold{fold}.json',proof)
            VERIFIED_CACHES.add(fold)
        for replicate in (0,1):
            for recipe in recipes:
                job(recipe,fold,replicate)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('screen','controls','full'),default='full',nargs='?')
    a=p.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'queue.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:p.error('another challenge queue owns this lease')
        sources=W.source_manifest()
        if sources != json.loads((HERE/'source_manifest_prelaunch.json').read_text()):
            p.error('source differs from committed prelaunch manifest')
        provenance=ROOT/'source_manifest.json'
        if provenance.exists() and json.loads(provenance.read_text()) != sources:
            p.error('challenge source differs from queue manifest; inspect before resuming')
        W.write(provenance,sources)
        try:
            if a.phase in ('screen','full'):
                phase((0,1,2),('original32','current32','blind32','radial32','empirical32'))
                screen=C.report((0,1,2))
                W.write(ROOT/'screen_comparison.json',screen)
            if a.phase in ('controls','full'):
                phase((0,1,2),('replay32','current128','empirical128'))
                W.write(ROOT/'screen_comparison.json',C.report((0,1,2)))
            eligible=[]
            if a.phase=='full':
                eligible=screen['passing_recipes_ranked']
                if eligible:
                    phase(range(3,9),('original32','current32','empirical32',*eligible))
                    W.write(ROOT/'confirmation_comparison.json',C.report(tuple(range(3,9)),True,eligible))
            W.write(ROOT/'status.json',{'status':'completed','phase':a.phase,
                    'screen_eligible':eligible,'default_changed':False,
                    'logical_task_bytes':disk_bytes(),'time':time.time()})
        except BaseException as exc:
            W.write(ROOT/'status.json',{'status':'failed','error':repr(exc),'time':time.time()})
            raise


VERIFIED_CACHES=set()
if __name__=='__main__':main()
