"""Finite CPU-only closure audit and selected-setting followup preparation."""
import datetime,fcntl,hashlib,importlib.util,json,os,subprocess,time
from pathlib import Path
ROOT=Path('/ml2/alz');STATE=ROOT/'artifacts/followup097-preparation-boundary'
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
 assert p.returncode==0,name+' failed; preserve evidence, no automatic retry or activation'
def commit(paths,message):
 command('git-add',['git','add','--',*map(str,paths)]);command('git-commit',['git','commit','--only','-m',message,'--',*map(str,paths)])
def report(kind,event,summary,details):
 spec=importlib.util.spec_from_file_location('existing_reporter',ROOT/'analysis/external_benchmark_driver.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.report(kind,event,summary,json.dumps(details))
def main():
 STATE.mkdir(exist_ok=True)
 with (STATE/'lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (STATE/'identity.json').exists(),'One-use preparation boundary consumed'
  parent=read(ROOT/'artifacts/refinement097-paired-search/identity.json');assert alive(parent)
  runtime=ROOT/'followup097_runtime';study=ROOT/'analysis/followup097'
  paths=[Path(__file__),ROOT/'analysis/audit_refinement097.py',ROOT/'analysis/prepare_followup097.py',ROOT/'analysis/run_followup097.py',ROOT/'analysis/refinement097_followup_selection.py',ROOT/'analysis/refinement097_followup_selection_rule_20261107.json',
    *list(runtime.glob('*.py')),runtime/'policy.prepared.json',study/'protocol.json',*list((study/'model').glob('*.py')),
    ROOT/'analysis/run_refinement097.py',ROOT/'analysis/refinement097/protocol.json',*list((ROOT/'refinement097_runtime').glob('*.py'))]
  pins={str(p):sha(p) for p in paths};expiry=min(time.time()+12*3600,read(ROOT/'analysis/research_window_20261104_five_days.json')['expires_unix']-600)
  write(STATE/'identity.json',dict(pid=os.getpid(),birth=int(Path('/proc',str(os.getpid()),'stat').read_text().rsplit(')',1)[1].split()[19]),boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),parent=parent,
    finite_expiry_unix=expiry,source_SHA256=pins,activation_permitted=False,GPU_allocation_permitted=False,external_freeze_permitted=False))
  try:
   while alive(parent):
    assert time.time()<expiry,'Finite observation expired; inspect the same original driver, do not replace or restart'
    time.sleep(20)
   for path,pin in pins.items():assert sha(path)==pin,'Frozen followup preparation source changed'
   assert not (ROOT/'artifacts/refinement097-paired-search/failure.json').exists()
   progress=read(ROOT/'artifacts/refinement097-paired-search/progress.json');assert progress['activation_reverted'] and progress['goal_still_active']
   assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True)
   audit=ROOT/'analysis/refinement097_final_independent_audit_20261107.json'
   command('audit-all20-paired-fits',[PG,'-I','-S','-B',ROOT/'analysis/audit_refinement097.py','--output',audit],timeout=900)
   commit([audit],'Independently verify all20 paired full20k native states, counts, scores, reference labels and closed resources')
   command('materialize-followup',[PG,'-I','-S','-B',ROOT/'analysis/prepare_followup097.py','--paired-audit',audit])
   p=read(study/'protocol.json');n=len(p['selected_configurations'])
   gp=ROOT/'analysis/current_research_goal.json';g=read(gp)
   if n:
    command('qualify-actual-followup-CPU',[PG,'-I','-S','-B',runtime/'qualify.py'],timeout=900)
    q=read(study/'runtime_CPU_qualification.json');assert q['actual_runtime_tests']==23 and q['actual_cells_checked']==3*n and q['fullwidth_native_optimizer_pairs']==12*n
    assert not q['GPU_allocation_created'] and not q['CUDA_initialized']
    status='PAIRED_SELECTION_AUDITED_SELECTED_FOLLOWUP_CPU_QUALIFIED_DISABLED'
    g.update(study_status=status,matched_followup_actual_selected_scope_qualified=True,matched_followup_CPU_qualification_path=str(study/'runtime_CPU_qualification.json'))
   else:
    assert p['status']=='NO_NEW_FOLLOWUP_CONFIGURATIONS_SELECTED_REFERENCE_ONLY_DECISION_PENDING'
    status='PAIRED_SELECTION_AUDITED_REFERENCE_ONLY_DECISION_REQUIRES_REVIEW'
    g.update(study_status=status,matched_followup_actual_selected_scope_qualified=False,matched_followup_reference_only_decision_path=str(study/'reference_only_decision.json'))
   assert not read(runtime/'plan.json')['enabled']
   g.update(paired_refinement_final_audit_path=str(audit),matched_followup_protocol_path=str(study/'protocol.json'),native_goal_status='active',full_goal_complete=False);write(gp,g)
   commit([study,gp],'Freeze audited paired-selection followup choices and preserve measured reference labels; keep plan disabled')
   result=dict(status=status,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),paired_audit_path=str(audit),followup_protocol_path=str(study/'protocol.json'),
     selected_new_configurations=p['selected_configurations'],required_new_fits=3*n,training_seeds=[2,3,4],reference_new_fits=0,activation=False,GPU_allocation_created=False,external_candidate_frozen=False,full_goal_complete=False,SOTA_claim=False)
   write(STATE/'completed.json',result)
   report('completed','followup097-all20-paired-audited-selected-followup-prepared-disabled-20261107','All20 paired selection outcomes independently audited and activation reverted; followup choices prepared with prior-reference labels. GPU plan remains disabled; exact execution review or reference-only decision remains.',result)
  except BaseException as exc:
   failure=dict(status='FOLLOWUP097_CPU_PREPARATION_BOUNDARY_STOPPED',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),error=repr(exc),parent_currently_alive=alive(parent),automatic_retry=False,activation=False,GPU_allocation_created=False)
   write(STATE/'failure.json',failure);report('failed','followup097-CPU-preparation-boundary-stop-20261107','CPU closure/followup preparation stopped; evidence retained, no new activation or GPU allocation.',failure);raise
if __name__=='__main__':main()
