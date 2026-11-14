"""Serial paired-cell probe with conditional confirmation and identity control."""
import argparse
from contextlib import ExitStack
import fcntl
import os
import subprocess
import sys
import time
import pair_common as B
import generate_pair as D
import compare_pair as C
import run_factor as R

LOGS=B.W.RUNS/'logs_r5/paired_challenge_20261114'
PRIOR_FILES={'status.json','artifact_audit.json','screen_comparison.json'}


def disk_bytes():
    roots=[B.W.RUNS/k/name for k in ('fits_r5','smoke_r5','results_r5','logs_r5')
           for name in ('architecture_pass','empirical_challenge_20261114','hybrid_challenge_20261114',
                        'dispersion_challenge_20261114','paired_challenge_20261114')]
    output=subprocess.check_output(['du','-sb',*[str(p) for p in roots if p.exists()]],text=True)
    return sum(int(line.split('\t')[0]) for line in output.splitlines())


def validate_previous(status,proof,comparison,pinned):
    B.require(status['status']=='completed' and status['screen_passes'] is False,'prior dispersion queue not complete/failed')
    B.require(proof['status']=='passed' and proof['complete'] is True and proof['scope']=='full_conditional_study'
              and proof['verified_artifacts']==proof['required_artifacts']==24 and proof['missing']==[]
              and proof['zero_factor_fidelity_verified'] is True and proof['screen_passes'] is False and proof['confirmation_disposition']=='not_triggered'
              and proof['prelaunch_manifest_sha256']==B.PRIOR_MANIFEST_SHA,'complete prior24-artifact proof required')
    expected={(name,f,r) for name in B.D.RECIPES for f in range(3) for r in (0,1)}
    artifacts=proof['artifacts']
    B.require(len(artifacts)==24 and {(a['recipe'],a['fold'],a['replicate']) for a in artifacts}==expected,
              'prior artifact identities differ')
    B.require(comparison['phase']=='screen' and comparison['folds']==[0,1,2]
              and comparison['candidate']=='factor32' and comparison['verdict']['status']=='complete'
              and comparison['verdict']['passes'] is False,'complete failed dispersion screen required')
    B.require(set(pinned['runtime_sha256'])==PRIOR_FILES,'exact prior evidence links required')


def previous_completed():
    R.previous_completed()
    status=B.P.read(B.D.ROOT/'status.json');proof=B.P.read(B.D.ROOT/'artifact_audit.json')
    comparison=B.P.read(B.D.ROOT/'screen_comparison.json');pinned=B.P.read(B.HERE/'prior_study_complete_evidence.json')
    validate_previous(status,proof,comparison,pinned)
    R.D.verify_fidelity(B.P.read(B.PREVIOUS/'source_manifest_prelaunch.json'))
    B.require(B.sha(B.D.ROOT/'zero_factor_fidelity.json')==proof['zero_factor_fidelity_sha256'],
              'prior full zero-factor proof changed after complete audit')
    for name,digest in pinned['runtime_sha256'].items():
        B.require(B.sha(B.D.ROOT/name)==digest,'prior dispersion completion evidence changed')
    for item in proof['artifacts']:
        name=item['recipe'];fold=item['fold'];rep=item['replicate']
        B.require(B.sha(B.D.ROOT/name/f'fold{fold}/seed{rep}.json')==item['score_sha256']
                  and B.sha(B.D.MODES/name/f'fold{fold}/seed{rep}.receipt.json')==item['receipt_sha256'],
                  'prior dispersion score/receipt changed after complete audit')


def phase(folds):
    for fold in folds:
        B.write(B.ROOT/'cache_verification'/f'fold{fold}.json',B.P.verify_cache(fold))
        for replicate in (0,1):
            for recipe in B.RECIPES:
                sources=B.P.read(B.ROOT/'source_manifest.json')
                B.require(B.source_manifest()==sources,'source changed between jobs')
                if D.finalized(recipe,fold,replicate,sources):continue
                B.require(disk_bytes()<2_800_000_000,'insufficient headroom below combined3GB cap')
                label=f'{recipe}_f{fold}r{replicate}'
                command=[sys.executable,'-B',str(B.HERE/'generate_pair.py'),
                         '--recipe',recipe,'--fold',str(fold),'--replicate',str(replicate)]
                with (LOGS/(label+'.log')).open('a') as log:
                    child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=B.HERE.parents[1])
                    B.write(B.ROOT/'status.json',{'status':'running','recipe':recipe,'fold':fold,'replicate':replicate,
                            'runner_pid':os.getpid(),'worker_pid':child.pid,'time':time.time()})
                    code=child.wait()
                B.require(code==0,f'{label} exited{code}; inspect its log before resuming')
                B.require(D.finalized(recipe,fold,replicate,sources),'worker did not finalize')
                B.require(disk_bytes()<3_000_000_000,'combined task cap exceeded')
                print('completed',label,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.parse_args()
    B.ROOT.mkdir(parents=True,exist_ok=True);LOGS.mkdir(parents=True,exist_ok=True)
    with ExitStack() as stack:
        for root in (B.ROOT,B.D.ROOT,B.H.ROOT,B.W.ROOT):
            lock=stack.enter_context((root/'queue.lock').open('a'))
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        previous_completed()
        expected=B.P.read(B.HERE/'source_manifest_prelaunch.json')
        B.require(B.source_manifest()==expected,'source differs from committed prelaunch manifest')
        path=B.ROOT/'source_manifest.json'
        B.require(not path.exists() or B.P.read(path)==expected,'existing queue has different sources')
        B.write(path,expected)
        try:
            phase(range(3));screen=C.report(tuple(range(3)))
            B.write(B.ROOT/'screen_comparison.json',screen)
            if screen['verdict']['passes']:
                phase(range(3,9))
                B.write(B.ROOT/'confirmation_comparison.json',C.report(tuple(range(3,9)),True))
            B.write(B.ROOT/'status.json',{'status':'completed','screen_passes':screen['verdict']['passes'],
                    'default_changed':False,'logical_task_bytes':disk_bytes(),'time':time.time()})
        except BaseException as exc:
            B.write(B.ROOT/'status.json',{'status':'failed','error':repr(exc),'time':time.time()});raise


if __name__=='__main__':main()
