"""Finite CPU-only observer for the predeclared post-benchmark supplement."""
import datetime
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path('/ml2/alz')
STATE=ROOT/'artifacts/benchmark097-donor-matched-supplement-observer'
PROOF=ROOT/'analysis/benchmark097_donor_matched_supplement_CPU_preparation_v2_20261108.json'
PG=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark/current_base_20261029/runtime/bin/python')


def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def identity(pid):
    try:
        p=Path('/proc')/str(pid);s=(p/'stat').read_text().rsplit(')',1)[1].split()
        if s[0]=='Z':return None
        return dict(pid=pid,birth=int(s[19]),boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    except (FileNotFoundError,ProcessLookupError):return None


def write(path,value):
    p=Path(path);tmp=p.with_suffix('.partial');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(p)


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True)
    proof=read(PROOF);assert proof['status']=='DONOR_MATCHED_SUPPLEMENT_ESTIMATOR_SAMPLING_AND_OPAQUE_BARRIER_CPU_PASS'
    assert proof['actual_tests']==14 and proof['clinical_reference_allocations']==6 and proof['generated_row_plans_checked']==60
    for p,pin in proof['source_SHA256'].items():assert sha(p)==pin,p
    parent=ROOT/'artifacts/released097-final-independent-audits';up=read(parent/'identity.json');expected={k:up[k] for k in ('pid','birth','boot')}
    assert identity(up['pid'])==expected
    expiry=read(ROOT/'analysis/research_window_20261104_five_days.json')['expires_unix']+7200
    STATE.mkdir(exist_ok=False)
    with (STATE/'observer.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write(STATE/'identity.json',dict(**identity(os.getpid()),upstream=expected,source_SHA256=sha(__file__),qualification_SHA256=sha(PROOF),CPU_observation_expiry_unix=expiry,new_GPU_allocation=False))
        try:
            while identity(up['pid'])==expected:
                assert time.time()<expiry,'Finite CPU observation window ended; upstream is not inferred stopped'
                assert not(parent/'failure.json').exists(),'Upstream incomplete; no secondary target reads'
                write(STATE/'progress.json',dict(status='WAITING_ON_FINAL_NATIVE_QUEUES_AND_INDEPENDENT_AUDIT_OBSERVER',new_GPU_allocation=False,expression_or_quality_values_read=False,full_goal_complete=False))
                time.sleep(30)
            assert not(parent/'failure.json').exists()
            marker=parent/('completed.json' if (parent/'completed.json').is_file() else 'incomplete.json')
            assert read(marker)['status'] in ('AVAILABLE_COMPLETED_STUDY_AUDITS_FINISHED','AUDITS_UNAVAILABLE_UPSTREAM_STUDIES_INCOMPLETE')
            for p,pin in proof['source_SHA256'].items():assert sha(p)==pin,p
            assert time.time()<expiry
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
            command=[str(PG),'-I','-S','-B',str(ROOT/'analysis/score_benchmark097_donor_matched_supplement.py'),'--run']
            result=subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True,timeout=max(1,min(3600,expiry-time.time())))
            (STATE/'scorer.stdout').write_text(result.stdout);(STATE/'scorer.stderr').write_text(result.stderr)
            assert result.returncode==0,'Supplementary analysis incomplete; retain outputs without changing native studies'
            report=ROOT/'analysis/benchmark097_donor_matched_cell_distance_supplement_20261108.json'
            value=read(report);assert value['status']=='COMPLETE_DESCRIPTIVE_DONOR_MATCHED_CELL_DISTANCE_SUPPLEMENT_SAME45_MODELS' and not value['SOTA_claim']
            subprocess.run(['git','add','--',str(report)],cwd=ROOT,check=True)
            subprocess.run(['git','commit','--only','-m','Retain donor-matched descriptive supplement alongside original frozen scores','--',str(report)],cwd=ROOT,check=True,capture_output=True,text=True)
            write(STATE/'completed.json',dict(status='DONOR_MATCHED_SUPPLEMENT_COMPLETE_ORIGINAL_PRIMARY_PRESERVED',report_path=str(report),report_SHA256=sha(report),full_goal_complete=False,SOTA_claim=False))
            sys.path.insert(0,str(ROOT/'benchmark097_balanced_runtime'))
            from babel_reporting import report as send
            send('completed','benchmark097-donor-matched-supplement-complete','Descriptive donor-matched cell-distance supplement completed for the same45 frozen models and both57/53 cohorts. Original reports preserved; no SOTA promotion.',dict(report_path=str(report),report_SHA256=sha(report)),log_root=STATE)
        except BaseException as exc:
            write(STATE/'failure.json',dict(status='SUPPLEMENT_OBSERVER_OR_SCORER_INCOMPLETE_NO_NATIVE_STUDY_CHANGE',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),error=repr(exc),new_GPU_allocation=False,upstream_or_foreign_signals=False,full_goal_complete=False,SOTA_claim=False));raise


if __name__=='__main__':main()
