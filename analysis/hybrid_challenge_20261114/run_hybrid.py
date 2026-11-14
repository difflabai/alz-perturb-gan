"""One serial CPU queue, fixed screen and conditional confirmation, no promotion."""
import argparse
import fcntl
import os
import subprocess
import sys
import time
import generate_hybrid as H
import compare_hybrid as C

LOGS=H.W.RUNS/'logs_r5/hybrid_challenge_20261114'
PRIOR_FILES={'status.json','artifact_audit.json','screen_comparison.json'}


def disk_bytes():
    roots=[H.W.RUNS/k/name for k in ('fits_r5','smoke_r5','results_r5','logs_r5')
           for name in ('architecture_pass','empirical_challenge_20261114','hybrid_challenge_20261114')]
    output=subprocess.check_output(['du','-sb',*[str(p) for p in roots if p.exists()]],text=True)
    return sum(int(line.split('\t')[0]) for line in output.splitlines())


def validate_previous(status,proof,comparison,pinned):
    H.require(status['status']=='completed' and status['phase']=='full' and status['screen_eligible']==[],
              'previous fixed queue must finish with no eligible candidates before new launch')
    H.require(proof['status']=='passed' and proof['scope']=='full_conditional_study'
              and proof['verified_artifacts']==proof['required_artifacts']==48
              and proof['complete'] is True and proof['missing']==[]
              and proof['screen_eligible']==[] and proof['confirmation_disposition']=='not_triggered'
              and proof['prelaunch_manifest_sha256']==H.A.PRELAUNCH_SHA256,
              'complete prior48-artifact proof required')
    expected={(name,f,r) for name in (*H.A.PRIMARY,*H.A.CONTROLS) for f in range(3) for r in (0,1)}
    artifacts=proof['artifacts']
    H.require(len(artifacts)==48 and {(a['recipe'],a['fold'],a['replicate']) for a in artifacts}==expected,
              'exact prior48-artifact identities required')
    H.require(comparison['phase']=='screen' and comparison['folds']==[0,1,2]
              and comparison['passing_recipes_ranked']==[] and
              set(comparison['arms'])=={'blind32','radial32'} and
              all(row['status']=='complete' and row['passes'] is False for row in comparison['arms'].values()),
              'complete failed prior screen required')
    H.require(set(pinned['runtime_sha256'])==PRIOR_FILES,'exact prior evidence links required')


def previous_completed():
    status=H.P.read(H.W.ROOT/'status.json')
    proof=H.P.read(H.W.ROOT/'artifact_audit.json')
    comparison=H.P.read(H.W.ROOT/'screen_comparison.json')
    pinned=H.P.read(H.HERE/'prior_study_complete_evidence.json')
    validate_previous(status,proof,comparison,pinned)
    for name,digest in pinned['runtime_sha256'].items():
        H.require(H.sha(H.W.ROOT/name)==digest,'previous complete evidence changed')
    for artifact in proof['artifacts']:
        name=artifact['recipe'];fold=artifact['fold'];rep=artifact['replicate']
        H.require(H.sha(H.W.ROOT/name/f'fold{fold}/seed{rep}.json')==artifact['score_sha256'] and
                  H.sha(H.W.MODES/name/f'fold{fold}/seed{rep}.receipt.json')==artifact['receipt_sha256'],
                  'previous artifact changed after complete audit')


def phase(folds):
    for fold in folds:
        H.write(H.ROOT/'cache_verification'/f'fold{fold}.json',H.P.verify_cache(fold))
        for replicate in (0,1):
            for recipe in H.RECIPES:
                sources=H.P.read(H.ROOT/'source_manifest.json')
                H.require(H.source_manifest()==sources,'source changed between jobs')
                if H.finalized(recipe,fold,replicate,sources):continue
                H.require(disk_bytes()<2_800_000_000,'insufficient headroom below combined3GB cap')
                label=f'{recipe}_f{fold}r{replicate}'
                command=[sys.executable,'-B',str(H.HERE/'generate_hybrid.py'),
                         '--recipe',recipe,'--fold',str(fold),'--replicate',str(replicate)]
                with (LOGS/(label+'.log')).open('a') as log:
                    child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=H.HERE.parents[1])
                    H.write(H.ROOT/'status.json',{'status':'running','recipe':recipe,'fold':fold,
                            'replicate':replicate,'runner_pid':os.getpid(),'worker_pid':child.pid,'time':time.time()})
                    code=child.wait()
                H.require(code==0,f'{label} exited{code}; inspect its log before resuming')
                H.require(H.finalized(recipe,fold,replicate,sources),'worker did not finalize')
                H.require(disk_bytes()<3_000_000_000,'combined task cap exceeded')
                print('completed',label,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.parse_args()
    H.ROOT.mkdir(parents=True,exist_ok=True);LOGS.mkdir(parents=True,exist_ok=True)
    with (H.ROOT/'queue.lock').open('a') as lock,(H.W.ROOT/'queue.lock').open('a') as previous_lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        fcntl.flock(previous_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        previous_completed()
        expected=H.P.read(H.HERE/'source_manifest_prelaunch.json')
        H.require(H.source_manifest()==expected,'source differs from committed prelaunch manifest')
        path=H.ROOT/'source_manifest.json'
        H.require(not path.exists() or H.P.read(path)==expected,'existing queue has different sources')
        H.write(path,expected)
        try:
            phase(range(3));screen=C.report(tuple(range(3)))
            H.write(H.ROOT/'screen_comparison.json',screen)
            if screen['verdict']['passes']:
                phase(range(3,9))
                H.write(H.ROOT/'confirmation_comparison.json',C.report(tuple(range(3,9)),True))
            H.write(H.ROOT/'status.json',{'status':'completed','screen_passes':screen['verdict']['passes'],
                    'default_changed':False,'logical_task_bytes':disk_bytes(),'time':time.time()})
        except BaseException as exc:
            H.write(H.ROOT/'status.json',{'status':'failed','error':repr(exc),'time':time.time()});raise


if __name__=='__main__':main()
