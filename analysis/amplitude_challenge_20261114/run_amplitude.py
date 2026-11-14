"""Serial fixed amplitude probe; prospective gates never promote a default."""
import argparse
from contextlib import ExitStack
import fcntl
import os
import subprocess
import sys
import time
import amp_common as B
import generate_amplitude as D
import compare_amplitude as C
import run_pair as PRIOR
LOGS=B.W.RUNS/'logs_r5/amplitude_challenge_20261114'
TASKS=(*B.Q.TASKS,'amplitude_challenge_20261114')


def disk_bytes():
    return sum(p.stat().st_size for root in ('fits_r5','smoke_r5','results_r5','logs_r5')
               for name in TASKS for p in (B.W.RUNS/root/name).rglob('*') if p.is_file())


def previous_completed():
    PRIOR.previous_completed();B.Q.prerequisite()
    pin=B.P.read(B.HERE/'predecessor_evidence.json');path=B.Q.ROOT/'report.json'
    report=B.P.read(path)
    B.require(B.sha(path)==pin['report_sha256']
              and B.sha(B.Q.ROOT/'compact_moments.npz')==pin['compact_moments_sha256']==report['compact_moments_sha256']
              and B.sha(B.PREVIOUS/'diagnostic_complete_20261114.json')==pin['report_sha256']
              and report['status']=='completed_training_diagnostic_only'
              and report['fold']==report['replicate']==0 and report['heldout_payloads_opened'] is False
              and report['default_changed'] is False
              and report['sources']==B.P.read(B.PREVIOUS/'source_manifest_prelaunch.json'),
              'committed TRAIN geometry diagnostic changed')


def phase(folds):
    for fold in folds:
        B.write(B.ROOT/'cache_verification'/f'fold{fold}.json',B.P.verify_cache(fold))
        for replicate in (0,1):
            for recipe in B.RECIPES:
                sources=B.P.read(B.ROOT/'source_manifest.json')
                B.require(B.source_manifest()==sources,'source changed between jobs')
                if D.finalized(recipe,fold,replicate,sources):continue
                B.require(disk_bytes()<2_800_000_000,'insufficient headroom under combined3GB cap')
                label=f'{recipe}_f{fold}r{replicate}'
                command=[sys.executable,'-B',str(B.HERE/'generate_amplitude.py'),
                         '--recipe',recipe,'--fold',str(fold),'--replicate',str(replicate)]
                with (LOGS/(label+'.log')).open('a') as log:
                    child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=B.HERE.parents[1])
                    B.write(B.ROOT/'status.json',{'status':'running','recipe':recipe,'fold':fold,'replicate':replicate,
                            'runner_pid':os.getpid(),'worker_pid':child.pid,'time':time.time()})
                    code=child.wait()
                if code==2 and recipe=='amplitude32':
                    path=B.ROOT/'raw_laws'/f'fold{fold}/seed{replicate}.json';record=B.P.read(path)
                    law=record['law'];cache=B.A.cache_metadata(fold,B.P.verify_cache(fold,payloads=False))
                    calibration=B.MODES/'current32'/f'fold{fold}/seed{replicate}.receipt.json'
                    values=D.validate_raw_record(record,cache,fold,replicate,sources,
                                                 B.sha(calibration),B.P.verify_fit(fold,replicate))
                    B.require(values is None and law['feasible'] is False
                              and not (B.ROOT/recipe/f'fold{fold}/seed{replicate}.json').exists()
                              and not (B.MODES/recipe/f'fold{fold}/seed{replicate}.npz').exists(),
                              'infeasibility exit is not proven')
                    D.verify_fidelity(sources)
                    B.write(B.ROOT/'status.json',{'status':'scientifically_infeasible','recipe':recipe,'fold':fold,
                            'replicate':replicate,'raw_law_sha256':B.sha(path),'default_changed':False,
                            'confirmation_disposition':'not_triggered','logical_task_bytes':disk_bytes(),'time':time.time()})
                    return False
                B.require(code==0,f'{label} exited{code}; inspect log before resuming')
                B.require(D.finalized(recipe,fold,replicate,sources),'worker did not finalize')
                B.require(disk_bytes()<3_000_000_000,'combined task output cap exceeded')
                print('completed',label,flush=True)
    return True


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    B.ROOT.mkdir(parents=True,exist_ok=True);LOGS.mkdir(parents=True,exist_ok=True)
    with ExitStack() as stack:
        for root in (B.ROOT,B.Q.ROOT,B.Q.B.ROOT,B.D.ROOT,B.H.ROOT,B.W.ROOT):
            lock=stack.enter_context((root/'queue.lock').open('a'));fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        previous_completed();expected=B.P.read(B.HERE/'source_manifest_prelaunch.json')
        B.require(B.source_manifest()==expected,'source differs from committed prelaunch manifest')
        path=B.ROOT/'source_manifest.json'
        B.require(not path.exists() or B.P.read(path)==expected,'existing queue uses different sources')
        B.write(path,expected)
        try:
            if not phase(range(3)):return
            screen=C.report(tuple(range(3)));B.write(B.ROOT/'screen_comparison.json',screen)
            if screen['verdict']['passes']:
                if not phase(range(3,9)):return
                B.write(B.ROOT/'confirmation_comparison.json',C.report(tuple(range(3,9)),True))
            B.write(B.ROOT/'status.json',{'status':'completed','screen_passes':screen['verdict']['passes'],
                    'default_changed':False,'logical_task_bytes':disk_bytes(),'time':time.time()})
        except BaseException as exc:
            B.write(B.ROOT/'status.json',{'status':'failed','error':repr(exc),'time':time.time()});raise


if __name__=='__main__':main()
