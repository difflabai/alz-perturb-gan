"""Finite read-only wait on the exact CPU handoff, then launch qualified pipeline.

A vanished handle alone is never success. No model/guard signal or restart is
performed. The original scientific/runtime gates remain in the launched code.
"""
import argparse,datetime,fcntl,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path('/ml2/alz');RUNTIME=ROOT/'benchmark097_balanced_runtime';STATE=ROOT/'artifacts/after-balanced-handoff-pipeline'
sys.path.insert(0,str(RUNTIME))
import binding as b,preset_scope as s
from dual_coordination import identity
ENV=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
def read(p):return json.loads(Path(p).read_bytes())
def write(p,q):
 p=Path(p);tmp=p.with_suffix('.partial');tmp.write_text(json.dumps(q,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');args=parser.parse_args()
 upstream=ROOT/'artifacts/benchmark097-balanced-handoff';parent=read(upstream/'identity.json');expected={k:parent[k] for k in ('pid','birth','boot')}
 assert identity(expected['pid'])==expected,'Bind the actual currently live handoff CPU process'
 proof=read(s.RUNTIME_PROOF);assert proof['status']=='EXTERNAL097_SAME45_BALANCED_RESOURCE_SCOPE_CPU_PASS'
 for path,pin in proof['source_SHA256'].items():assert b.sha(path)==pin
 expiry=read(s.WINDOW)['expires_unix']
 if args.resume:
  assert STATE.exists() and not (STATE/'launch_receipt.json').exists(),'Never relaunch an already started qualified pipeline'
 else:STATE.mkdir(exist_ok=False)
 with (STATE/'boundary.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if args.resume:
   history=STATE/'CPU_observer_history';history.mkdir(exist_ok=True)
   for name in ('failure.json','identity.json'):
    old=STATE/name
    if old.exists():old.rename(history/(str(time.time_ns())+'-'+name))
  write(STATE/'identity.json',dict(**identity(os.getpid()),source_SHA256=b.sha(__file__),upstream=expected,training_window_not_extended=True,new_GPU_allocation=False))
  try:
   while identity(expected['pid'])==expected:
    assert time.time()<expiry-150,'Owner training window ended before the qualified handoff'
    assert not (upstream/'failure.json').exists(),'Handoff requires reconciliation; no successor admission'
    write(STATE/'progress.json',dict(status='WAITING_FOR_EXACT_ORIGINAL_GUARDS_TO_FINISH_AND_HANDOFF_TO_CLOSE',upstream=expected,new_GPU_allocation=False));time.sleep(30)
   assert not (upstream/'failure.json').exists() and (upstream/'completed.json').exists()
   assert read(s.HANDOFF)['status']=='ORIGINAL_ADMITTED_CELLS_CLOSED_AND_RESOURCE_HANDOFF_READY'
   for path,pin in proof['source_SHA256'].items():assert b.sha(path)==pin
   while subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True):
    assert time.time()<expiry-150,'No clean committed source boundary before deadline';time.sleep(5)
   assert time.time()<expiry-150
   exe=b.runtime_path(b.NATIVE_LAYOUT,s.cell(0,'pg',0))
   with (STATE/'pipeline.stdout').open('xb') as out,(STATE/'pipeline.stderr').open('xb') as err:
    child=subprocess.Popen([str(exe),'-I','-S','-B',str(ROOT/'analysis/run_released097_balanced_pipeline.py')],cwd=ROOT,env=ENV,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
   handle=identity(child.pid);assert handle is not None;write(STATE/'launch_receipt.json',dict(**handle,source_SHA256=b.sha(ROOT/'analysis/run_released097_balanced_pipeline.py'),not_a_GPU_fit_claim=True))
   while child.poll() is None:
    assert time.time()<expiry+7200,'Finite CPU observation ended; original child handle retained without signals';time.sleep(30)
   assert child.returncode==0,'Qualified pipeline stopped: inspect original handles; do not retry models'
   completed=ROOT/'artifacts/released097-balanced-research-pipeline/completed.json';assert completed.exists()
   write(STATE/'completed.json',dict(status='QUALIFIED_BALANCED_RESEARCH_PIPELINE_COMPLETED',source_completion_path=str(completed),full_goal_complete=False,SOTA_claim=False))
  except BaseException as exc:
   write(STATE/'failure.json',dict(status='READONLY_HANDOFF_BOUNDARY_STOPPED_OR_CHILD_REQUIRES_RECONCILIATION',error=repr(exc),upstream_and_model_signals_sent=False,automatic_model_retry=False,full_goal_complete=False));raise
if __name__=='__main__':main()
