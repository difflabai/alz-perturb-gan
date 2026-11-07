"""Finite CPU-only closure audit; benchmark freeze was deferred on evidence."""
import datetime,fcntl,hashlib,importlib.util,json,os,subprocess,time
from pathlib import Path
ROOT=Path('/ml2/alz');STATE=ROOT/'artifacts/confirmation097-audit-boundary'
PG=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark/current_base_20261029/runtime/bin/python')
ENV=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,q):Path(p).write_text(json.dumps(q,indent=2,allow_nan=False)+'\n')
def alive(i):
 p=Path('/proc',str(i['pid']),'stat')
 if not p.exists():return False
 s=p.read_text().rsplit(')',1)[1].split()
 return s[0]!='Z' and int(s[19])==i['birth'] and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==i['boot']
def command(name,args,timeout=120):
 p=subprocess.run(list(map(str,args)),cwd=ROOT,env=ENV,capture_output=True,text=True,timeout=timeout)
 (STATE/(name+'.stdout')).write_text(p.stdout);(STATE/(name+'.stderr')).write_text(p.stderr)
 assert p.returncode==0,name+' failed; retain evidence, no automatic retry'
def report(kind,event,summary,details):
 spec=importlib.util.spec_from_file_location('existing_reporter',ROOT/'analysis/external_benchmark_driver.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.report(kind,event,summary,json.dumps(details))
def main():
 STATE.mkdir(exist_ok=True)
 with (STATE/'lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (STATE/'identity.json').exists(),'One-use audit boundary consumed'
  old=read(ROOT/'artifacts/benchmark097-boundary/cancelled_by_review.json');assert not alive(old['identity']) and not old['owned_GPU_driver_or_worker_signaled']
  parent=read(ROOT/'artifacts/confirmation097-matched-seeds/identity.json');assert alive(parent)
  paths=[Path(__file__),ROOT/'analysis/audit_confirmation097.py',ROOT/'analysis/run_confirmation097.py',ROOT/'analysis/confirmation097/protocol.json',
    ROOT/'analysis/confirmation097_selection.py',ROOT/'analysis/confirmation097_selection_rule_20261106.json',
    ROOT/'analysis/distribution097_conditional_next_search_design_20261107.json',ROOT/'analysis/distribution097_conditional_continuation_CPU_model_qualification_20261107.json',
    *list((ROOT/'confirmation097_runtime').glob('*.py')),*list((ROOT/'analysis/confirmation097/model').glob('*.py'))]
  pins={str(p):sha(p) for p in paths};expiry=min(time.time()+4*3600,read(ROOT/'analysis/research_window_20261104_five_days.json')['expires_unix']-600)
  write(STATE/'identity.json',dict(pid=os.getpid(),birth=int(Path('/proc',str(os.getpid()),'stat').read_text().rsplit(')',1)[1].split()[19]),boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),parent=parent,
    finite_expiry_unix=expiry,source_SHA256=pins,activation_permitted=False,GPU_allocation_permitted=False,benchmark_freeze_permitted=False))
  try:
   while alive(parent):
    assert time.time()<expiry,'Finite observation expired; inspect original driver without replacement'
    time.sleep(15)
   for path,pin in pins.items():assert sha(path)==pin,'Frozen audit-boundary source changed'
   assert not (ROOT/'artifacts/confirmation097-matched-seeds/failure.json').exists()
   progress=read(ROOT/'artifacts/confirmation097-matched-seeds/progress.json');assert progress['activation_reverted'] and progress['goal_still_active']
   assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True)
   audit=ROOT/'analysis/confirmation097_final_independent_audit_20261107.json'
   command('audit-confirmation',[PG,'-I','-S','-B',ROOT/'analysis/audit_confirmation097.py','--output',audit],timeout=600)
   q=read(audit);assert q['status']=='INDEPENDENT_FRESH_NATIVE097_MATCHED_SEED_CONFIRMATION_AUDIT_PASS'
   candidate=next(c for c in q['summary']['candidates'] if c['candidate_id']=='d000')
   required=(not candidate['all_four_fresh_seeds_finite'] or candidate['fresh_seed_mean']['cell_energy_distance']>=candidate['mean_matched_resampling']['cell_energy_distance']
     or any(not .7<=row['metrics']['spread_ratio']<=1.3 for row in candidate.get('per_seed',[])))
   status='FOUR_FRESH_SEEDS_AUDITED_FURTHER_CALIBRATION_REFINEMENT_REQUIRED' if required else 'FOUR_FRESH_SEEDS_AUDITED_CANDIDATE_REQUIRES_SELECTION_REVIEW'
   decision=ROOT/'analysis/confirmation097_next_decision_20261107.json'
   result=dict(status=status,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),complete_audit_path=str(audit),complete_audit_SHA256=sha(audit),
     reference_summary=candidate,calibration_refinement_triggered=required,benchmark_candidate_frozen=False,benchmark_activated=False,
     new_search_design_path=str(ROOT/'analysis/distribution097_conditional_next_search_design_20261107.json'),
     new_search_CPU_model_qualification_path=str(ROOT/'analysis/distribution097_conditional_continuation_CPU_model_qualification_20261107.json'),
     new_search_runtime_qualified=False,GPU_allocation_created=False,full_goal_complete=False,SOTA_claim=False)
   assert not decision.exists();write(decision,result)
   gp=ROOT/'analysis/current_research_goal.json';g=read(gp);g.update(study_status=status,fresh_seed_final_audit_path=str(audit),fresh_seed_next_decision_path=str(decision),
     external_benchmark_candidate_freeze_deferred=True,external_benchmark_real_frozen_candidate_qualified=False,native_goal_status='active',full_goal_complete=False);write(gp,g)
   command('git-add',['git','add','--',audit,decision,gp]);command('git-commit',['git','commit','--only','-m','Audit all four fresh seeds and record calibration continuation decision without external freeze','--',audit,decision,gp])
   write(STATE/'completed.json',result)
   report('completed','confirmation097-all-four-seeds-independent-audit-calibration-decision-20261107','All four fresh20k seeds independently audited and original activation reverted; calibration continuation decision recorded. External benchmark remains unfrozen and disabled.',result)
  except BaseException as exc:
   failure=dict(status='CONFIRMATION097_AUDIT_ONLY_BOUNDARY_STOPPED',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),error=repr(exc),automatic_retry=False,activation=False,GPU_allocation_created=False)
   write(STATE/'failure.json',failure);report('failed','confirmation097-audit-only-boundary-stop-20261107','CPU closure audit stopped; evidence retained, no new activation or allocation.',failure);raise
if __name__=='__main__':main()
