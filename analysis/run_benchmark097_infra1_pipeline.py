"""Observe untouched originals, then exactly two fresh infrastructure replacements."""
import datetime,fcntl,importlib.util,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path('/ml2/alz');RUNTIME=ROOT/'benchmark097_infra1_runtime';STATE=ROOT/'artifacts/benchmark097-infra1-pipeline';UPSTREAM=ROOT/'artifacts/benchmark097-after-capacity-continuation-v2'
sys.path[:0]=[str(RUNTIME),str(ROOT/'analysis')]
import binding as b,preset_scope as s,resource_profile as profile,prepare
from dual_coordination import GPUS,identity
import benchmark097_recovered_lineage as lineage
import run_benchmark097_infra1 as driver
ENV=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
PG=b.PROJECT/'current_base_20261029/runtime/bin/python'
def read(p):return json.loads(Path(p).read_bytes())
def write(p,q):
 p=Path(p);t=p.with_suffix('.partial');t.write_text(json.dumps(q,indent=2,allow_nan=False)+'\n');t.replace(p)
def command(args,timeout=120):
 p=subprocess.run(list(map(str,args)),cwd=ROOT,env=ENV,capture_output=True,text=True,timeout=timeout)
 if p.returncode:raise RuntimeError(p.stderr[-3000:])
 return p.stdout
def commit(paths,message):command(['git','add','--',*paths]);command(['git','commit','--only','-m',message,'--',*paths])
def main():
 assert not command(['git','status','--porcelain']) and not read(s.PLAN)['enabled']
 proof=read(s.RUNTIME_PROOF);assert proof['status']=='ONE_FRESH_INFRASTRUCTURE_REPLACEMENT_NATIVE_SCOPE_CPU_PASS' and proof['native_cell_device_bindings']==4
 for p,pin in proof['source_SHA256'].items():assert b.sha(p)==pin,p
 parent=read(UPSTREAM/'identity.json');parent_id={k:parent[k] for k in ('pid','birth','boot')};assert parent_id['pid']==962924 and identity(parent_id['pid'])==parent_id
 expiry=read(s.WINDOW)['expires_unix'];STATE.mkdir(exist_ok=False)
 with (STATE/'observer.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);write(STATE/'identity.json',dict(**identity(os.getpid()),source_SHA256=b.sha(__file__),upstream=parent_id,runtime_qualification_SHA256=b.sha(s.RUNTIME_PROOF),training_expiry_unix=expiry,new_model_fits=0))
  try:
   while identity(parent_id['pid'])==parent_id:
    assert not (UPSTREAM/'failure.json').exists(),'Untouched continuation stopped; retain all evidence without automatic retry'
    assert time.time()<expiry-150
    write(STATE/'progress.json',dict(status='WAITING_FOR_ALL45_ORIGINAL_ATTEMPTS_TO_CLOSE',new_model_fits=0,upstream=parent_id,full_goal_complete=False));time.sleep(30)
   assert not (UPSTREAM/'failure.json').exists() and (UPSTREAM/'completed.json').exists()
   original=read(s.HANDOFF);assert original['successful_original_models']==43 and original['failed_original_attempts']==list(lineage.FAILED)
   assert not read(b.PROJECT/'project_gpu_budget.json')['open_reservations'] and not command(['git','status','--porcelain'])
   for p,pin in proof['source_SHA256'].items():assert b.sha(p)==pin,p
   activation=read(ROOT/'artifacts/benchmark097-balanced-fullmatrix/identity.json')['activation'];command(['git','revert','--no-edit',activation]);s.closed_originals()
   review=dict(status='OWNER_AUTHORIZED_SOURCE_AND_RESOURCE_REVIEW_PASS',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),reviewer_session='01a1a22b-f310-7190-a7c6-8697ecabae6c',independent_review_claimed=False,owner_authorization=read(ROOT/'analysis/faithful097_owner_execution_review.json')['owner_authorization'],scope=profile.SCOPE,policy_SHA256=profile.policy_sha(),source_SHA256=profile.review_sources(),handoff_SHA256=b.sha(s.HANDOFF),failed_weights_never_loaded=True)
   assert not profile.REVIEW.exists();write(profile.REVIEW,review);commit([str(profile.REVIEW)],'Review two fresh infrastructure replacement attempts against closed original47-attempt lineage')
   write(s.PLAN,s.ENABLED);commit([str(s.PLAN)],'Activate exactly two from-zero infrastructure replacements under unchanged scientific endpoints')
   activation=command(['git','rev-parse','HEAD']).strip()
   for method in ('pg','cfgen'):
    for gpu in GPUS:
     auth,_=prepare.build_authorization(fold=2,method=method,seed=1,gpu=gpu,check_only=True);b.source_closure(auth,b.NATIVE_LAYOUT)
   stable={};polls={};amendment=read(s.AMENDMENT)
   def select(pending,active,gpu):
    if not pending:return None
    if gpu not in polls or time.time()-polls[gpu][0]>=30:
     try:
      a,_=prepare.build_authorization(fold=2,method='pg',seed=1,gpu=gpu,check_only=True);_,admission,_=b.modules(a);q=profile.capacity(admission.snapshot(),admission,gpu);profile.free_floor(gpu);enough=q['GPU_total_bytes']-q['GPU_required_bytes']>=amendment['scheduling_only_extra_GPU_capacity_bytes']
     except AssertionError:enough=False
     if not enough:stable.pop(gpu,None)
     else:stable.setdefault(gpu,time.time())
     polls[gpu]=(time.time(),enough and time.time()-stable[gpu]>=amendment['scheduling_only_stability_seconds'])
    if not polls[gpu][1]:return None
    preferred='cfgen' if gpu==GPUS[1] else 'pg'
    return next((v for v in pending if v['method']==preferred),pending[0])
   outcome=driver.run_queue([s.cell(method=m) for m in ('cfgen','pg')],initial_rows=[],start=driver.start_cell,ready=driver.ready_cell,finish=driver.finish_cell,observe=lambda rows,pending,active,errors,unsettled:write(STATE/'progress.json',dict(status='TWO_FROM_ZERO_INFRASTRUCTURE_REPLACEMENTS_RUNNING' if not errors else 'STOP_NEW_ADMISSIONS_DRAIN_HEALTHY_HANDLES',required_replacement_fits=2,completed_replacement_fits=sum(r['complete'] for r in rows),pending_replacement_fits=len(pending),active=[driver.public_handle(h) for h in active.values()],errors=errors,unsettled=unsettled,full_goal_complete=False,SOTA_claim=False)),sleep=time.sleep,now=time.time,training_expiry=expiry,select=select)
   assert not outcome['errors'] and not outcome['unsettled'] and len(outcome['rows'])==2 and all(r['complete'] for r in outcome['rows']),'Fresh replacement incomplete; retain without retry'
   matrix=lineage.build_matrix();path=ROOT/'analysis/benchmark097/matrix_complete.json';assert not path.exists();write(path,matrix);commit([str(path)],'Retain45 logical successful models and all47 physical attempt charges without hiding capacity failures')
   command(['git','revert','--no-edit',activation])
   assert not read(b.PROJECT/'project_gpu_budget.json')['open_reservations']
   for sensitivity in (False,True):
    args=[PG,'-I','-S','-B',ROOT/'analysis/aggregate_benchmark097_recovered.py']+(['--sensitivity'] if sensitivity else [])
    assert time.time()<expiry+7200
    output=command(args,timeout=min(3600,max(1,int(expiry+7200-time.time()))));(STATE/('sensitivity.stdout' if sensitivity else 'primary.stdout')).write_text(output)
    target=ROOT/('analysis/benchmark097_calibration_excluded53_report.json' if sensitivity else 'analysis/benchmark097/comparison.json');commit([str(target)],'Retain prespecified53 sensitivity with infrastructure provenance' if sensitivity else 'Retain original57 primary comparison with both infrastructure failures disclosed')
   paths=[ROOT/'analysis/benchmark097/matrix_complete.json',ROOT/'analysis/benchmark097/comparison.json',ROOT/'analysis/benchmark097_calibration_excluded53_report.json']
   barrier=s.STUDY/'scientific_reports_completed.json';record=dict(status='RECOVERED_LOGICAL45_ORIGINAL57_PRIMARY_AND_PRESPECIFIED53_COMPLETE',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),complete_fresh_models=45,physical_attempts=47,failed_attempts=list(lineage.FAILED),original57_primary_preserved=True,prespecified53_complete=True,new_generated_counts_for53=0,report_SHA256={str(p):b.sha(p) for p in paths},source_pipeline_path=str(Path(__file__).resolve()),source_pipeline_SHA256=b.sha(__file__),runtime_qualification_path=str(s.RUNTIME_PROOF),runtime_qualification_SHA256=b.sha(s.RUNTIME_PROOF),full_goal_complete=False,SOTA_claim=False)
   write(barrier,record);commit([str(barrier)],'Record complete logical45/57/53 recovery boundary for independent fresh scalar calibration')
   write(STATE/'completed.json',dict(status='RECOVERED45_57_53_READY_FRESH_SCALAR_CALIBRATION_AND_FINAL_AUDIT_REMAIN',barrier_path=str(barrier),full_goal_complete=False,SOTA_claim=False))
  except BaseException as exc:
   write(STATE/'failure.json',dict(status='RECOVERY_PIPELINE_STOPPED_RETAIN_EXACT_HANDLES',error=repr(exc),automatic_retry=False,foreign_processes_signalled=False,full_goal_complete=False,SOTA_claim=False));raise
if __name__=='__main__':main()
