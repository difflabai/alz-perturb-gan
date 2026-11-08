"""Bounded same45 balanced dispatch followed by unchanged57/53 numerical analyses."""
import datetime,fcntl,importlib.util,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path('/ml2/alz');RUNTIME=ROOT/'benchmark097_balanced_runtime';STATE=ROOT/'artifacts/released097-balanced-research-pipeline'
sys.path.insert(0,str(RUNTIME))
import binding as b,preset_scope as scope,resource_profile as profile
from dual_coordination import GPUS,identity
ENV=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
PG=b.runtime_path(b.NATIVE_LAYOUT,scope.cell(0,'pg',0))
def read(p):return json.loads(Path(p).read_bytes())
def write(p,q):
 p=Path(p);tmp=p.with_suffix('.partial');tmp.write_text(json.dumps(q,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def command(args,timeout=120):
 p=subprocess.run(list(map(str,args)),cwd=ROOT,env=ENV,capture_output=True,text=True,timeout=timeout)
 if p.returncode:raise RuntimeError(p.stderr[-3000:])
 return p.stdout
def commit(paths,message):command(['git','add','--',*map(str,paths)]);command(['git','commit','--only','-m',message,'--',*map(str,paths)])
def report(kind,event,summary,details):
 from babel_reporting import report as send
 return send(kind,event,summary,details,log_root=STATE)

def main():
 assert not command(['git','status','--porcelain'])
 handoff=read(scope.HANDOFF);assert handoff['status']=='ORIGINAL_ADMITTED_CELLS_CLOSED_AND_RESOURCE_HANDOFF_READY'
 proof=read(scope.RUNTIME_PROOF);assert proof['status']=='EXTERNAL097_SAME45_BALANCED_RESOURCE_SCOPE_CPU_PASS'
 for path,pin in proof['source_SHA256'].items():assert b.sha(path)==pin,path
 assert not read(scope.PLAN)['enabled'] and not read(b.PROJECT/'project_gpu_budget.json')['open_reservations']
 STATE.mkdir(exist_ok=False)
 with (STATE/'pipeline.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);expiry=read(scope.WINDOW)['expires_unix']
  write(STATE/'identity.json',dict(**identity(os.getpid()),source_SHA256=b.sha(__file__),handoff_SHA256=b.sha(scope.HANDOFF),training_expiry_unix=expiry,finite_CPU_analysis_expiry_unix=expiry+7200))
  try:
   assert time.time()<expiry-150
   review=dict(status='OWNER_AUTHORIZED_SOURCE_AND_RESOURCE_REVIEW_PASS',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),reviewer_session='01a1a22b-f310-7190-a7c6-8697ecabae6c',independent_review_claimed=False,owner_authorization=read(ROOT/'analysis/faithful097_owner_execution_review.json')['owner_authorization'],scope=profile.SCOPE,policy_SHA256=profile.policy_sha(),source_SHA256=profile.review_sources(),handoff_SHA256=b.sha(scope.HANDOFF),current_shared_capacity_checked_by_each_native_prepare=True)
   assert not profile.REVIEW.exists();write(profile.REVIEW,review);commit([profile.REVIEW],'Review source and shared profiles for unchanged45-model balanced dispatch')
   write(scope.PLAN,scope.ENABLED);commit([scope.PLAN],'Activate resource-only balanced dispatch for the remaining frozen45-model cells')
   activation=command(['git','rev-parse','HEAD']).strip()
   command([PG,'-I','-S','-B',RUNTIME/'post_handoff_check.py'],timeout=600)
   commit([scope.AUX/'post_handoff_source_closure.json'],'Verify all90 real cell/device source closures at the enabled clean handoff boundary')
   with (STATE/'driver.stdout').open('xb') as out,(STATE/'driver.stderr').open('xb') as err:
    child=subprocess.Popen([str(PG),'-I','-S','-B',str(ROOT/'analysis/run_benchmark097_balanced.py'),'--activation',activation],cwd=ROOT,env=ENV,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
   handle=identity(child.pid);assert handle is not None;write(STATE/'launch_receipt.json',dict(**handle,activation=activation,not_a_GPU_fit_claim=True))
   report('experiment_transition','benchmark097-balanced-dispatch-launched','Resource-only balanced CPU dispatcher launched for the same frozen45 cells. Original completed models retained; each untouched model starts at0.',dict(from_state='ORIGINAL_ADMITTED_MODELS_CLEANLY_RETAINED',to_state='BALANCED_CPU_DISPATCHER_LAUNCHED',child=handle))
   while child.poll() is None:
    assert time.time()<expiry+7200,'Finite CPU observation expired; exact child/model handles retained, no signals or restart';time.sleep(30)
   assert child.returncode==0,'Balanced driver incomplete; preserve every outcome and source handle'
   assert not read(scope.PLAN)['enabled'] and not read(b.PROJECT/'project_gpu_budget.json')['open_reservations']
   for name,limit in [('aggregate_benchmark097_balanced.py',1800),('score_calibration_excluded53_balanced.py',3600)]:
    assert time.time()<expiry+7200
    p=subprocess.run([str(PG),'-I','-S','-B',str(ROOT/'analysis'/name)],cwd=ROOT,env=ENV,capture_output=True,text=True,timeout=min(limit,max(1,expiry+7200-time.time())))
    (STATE/(name+'.stdout')).write_text(p.stdout);(STATE/(name+'.stderr')).write_text(p.stderr);assert p.returncode==0,name
    target=ROOT/('analysis/benchmark097/comparison.json' if name.startswith('aggregate') else 'analysis/benchmark097_calibration_excluded53_report.json');assert target.exists()
    commit([target],'Retain frozen45 scientific comparison after verified resource-only dispatch' if name.startswith('aggregate') else 'Retain original prespecified53-donor sensitivity after balanced-resource frozen45 study')
   primary=ROOT/'analysis/benchmark097/comparison.json';sensitivity=ROOT/'analysis/benchmark097_calibration_excluded53_report.json';matrix=scope.STUDY/'matrix_complete.json'
   q=read(primary);z=read(sensitivity);assert q['complete_cells']==45 and z['complete_fresh_generative_models']==45 and z['original57_primary_preserved']
   barrier=scope.AUX/'scientific_reports_completed.json';record=dict(status='SAME_FROZEN45_ORIGINAL57_PRIMARY_AND_PRESPECIFIED53_COMPLETE',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),complete_fresh_models=45,original57_primary_preserved=True,prespecified53_complete=True,new_generated_counts_for53=0,source_pipeline_path=str(Path(__file__)),source_pipeline_SHA256=b.sha(__file__),runtime_qualification_path=str(scope.RUNTIME_PROOF),runtime_qualification_SHA256=b.sha(scope.RUNTIME_PROOF),report_SHA256={str(p):b.sha(p) for p in (matrix,primary,sensitivity)},handoff_SHA256=b.sha(scope.HANDOFF),full_goal_complete=False,SOTA_claim=False);write(barrier,record)
   gp=ROOT/'analysis/current_research_goal.json';g=read(gp);g.update(study_status='SAME45_FROZEN_COMPARISON_AND53_COMPLETE_AFTER_RESOURCE_HANDOFF_FINAL_AUDIT_PENDING',frozen_external_matrix_path=str(matrix),study_completed_attempts=45,calibration_excluded53_completed=True,calibration_excluded53_report_path=str(sensitivity),balanced_scientific_completion_barrier_path=str(barrier),native_goal_status='active',full_goal_complete=False);write(gp,g)
   commit([barrier,gp],'Record completed original45/57/53 scientific scope with explicit resource provenance')
   write(STATE/'completed.json',dict(status='SAME45_SCIENTIFIC_REPORTS_READY_FINAL_AGENT_AUDIT_AND_SCALAR_CALIBRATION_REMAIN',barrier_path=str(barrier),full_goal_complete=False,SOTA_claim=False))
   report('completed','benchmark097-balanced-full45-57-53-reports-complete','All45 frozen models, original57 comparison and prespecified53 sensitivity complete after resource-only handoff. Final research audit and separate LR calibration remain.',record)
  except BaseException as exc:
   failure=dict(status='BALANCED_RESEARCH_PIPELINE_STOPPED_RETAINING_EVIDENCE',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),error=repr(exc),automatic_model_retry=False,GPU_jobs_signalled=False,full_goal_complete=False,SOTA_claim=False);write(STATE/'failure.json',failure);report('failed','benchmark097-balanced-pipeline-stopped','Balanced resource continuation stopped with original models, costs, source and exact handles retained. No restart or success claim.',failure);raise
if __name__=='__main__':main()
