"""Finite CPU-only materialization/qualification after the exact closure auditor.

No plan activation, GPU allocation, training or external candidate freeze.
"""
import datetime,fcntl,hashlib,importlib.util,json,os,subprocess,time
from pathlib import Path
ROOT=Path('/ml2/alz');STATE=ROOT/'artifacts/refinement097-preparation-boundary'
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
 assert p.returncode==0,name+' failed; preserve one-use evidence, no automatic retry'
def report(kind,event,summary,details):
 spec=importlib.util.spec_from_file_location('existing_reporter',ROOT/'analysis/external_benchmark_driver.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.report(kind,event,summary,json.dumps(details))
def main():
 STATE.mkdir(exist_ok=True)
 with (STATE/'lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (STATE/'identity.json').exists(),'One-use preparation boundary consumed'
  parent=read(ROOT/'artifacts/confirmation097-audit-boundary/identity.json');assert alive(parent)
  study=ROOT/'analysis/refinement097';runtime=ROOT/'refinement097_runtime'
  paths=[Path(__file__),ROOT/'analysis/prepare_refinement097.py',ROOT/'analysis/refinement097_selection.py',ROOT/'analysis/test_refinement097_selection.py',ROOT/'analysis/run_refinement097.py',
    ROOT/'analysis/distribution097_conditional_next_search_design_20261107.json',ROOT/'analysis/distribution097_conditional_continuation_CPU_model_qualification_20261107.json',
    *list(runtime.glob('*.py')),runtime/'policy.prepared.json',study/'protocol.json',study/'resampling_controls_CPU.json',*list((study/'model').glob('*.py'))]
  pins={str(p):sha(p) for p in paths};expiry=min(time.time()+4*3600,read(ROOT/'analysis/research_window_20261104_five_days.json')['expires_unix']-600)
  write(STATE/'identity.json',dict(pid=os.getpid(),birth=int(Path('/proc',str(os.getpid()),'stat').read_text().rsplit(')',1)[1].split()[19]),boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),parent=parent,
    finite_expiry_unix=expiry,source_SHA256=pins,activation_permitted=False,GPU_allocation_permitted=False))
  try:
   while alive(parent):
    assert time.time()<expiry,'Finite wait expired; inspect the existing original auditor'
    time.sleep(15)
   for path,pin in pins.items():assert sha(path)==pin,'Frozen paired-study preparation source changed'
   assert not (ROOT/'artifacts/confirmation097-audit-boundary/failure.json').exists()
   closed=read(ROOT/'artifacts/confirmation097-audit-boundary/completed.json')
   assert not closed['GPU_allocation_created'] and not closed['benchmark_candidate_frozen']
   if not closed['calibration_refinement_triggered']:
    write(STATE/'completed.json',dict(status='REFERENCE_REFINEMENT_NOT_TRIGGERED_REQUIRES_SELECTION_REVIEW',activation=False,GPU_allocation_created=False));return
   assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True)
   audit=ROOT/'analysis/confirmation097_final_independent_audit_20261107.json';decision=ROOT/'analysis/confirmation097_next_decision_20261107.json'
   command('materialize-paired-refinement',[PG,'-I','-S','-B',ROOT/'analysis/prepare_refinement097.py','--reference-audit',audit,'--decision',decision])
   command('qualify-actual-paired-study-CPU',[PG,'-I','-S','-B',runtime/'qualify.py'],timeout=900)
   q=read(study/'runtime_CPU_qualification.json');assert q['actual_runtime_tests']==23 and q['actual_cells_checked']==20 and q['fullwidth_native_optimizer_pairs']==40
   assert not q['GPU_allocation_created'] and not q['CUDA_initialized'] and not read(runtime/'plan.json')['enabled']
   gp=ROOT/'analysis/current_research_goal.json';g=read(gp);g.update(study_status='ALL_REFERENCE_SEEDS_AUDITED_TWENTY_PAIRED_REFINEMENT_CELLS_CPU_QUALIFIED_DISABLED',
     paired_refinement_protocol_path=str(study/'protocol.json'),paired_refinement_CPU_qualification_path=str(study/'runtime_CPU_qualification.json'),
     new_hyperparameter_search_status='FROZEN_TWENTY_FRESH20K_CELLS_FULL_CPU_QUALIFIED_DISABLED',conditional_next_search_activated=False,native_goal_status='active');write(gp,g)
   command('git-add',['git','add','--',study,gp]);command('git-commit',['git','commit','--only','-m','Freeze paired full20k scalar refinement from complete negative reference audit and qualify all20 cells','--',study,gp])
   result=dict(status='TWENTY_PAIRED_REFINEMENT_CELLS_FROZEN_AND_CPU_QUALIFIED_DISABLED',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
     protocol_path=str(study/'protocol.json'),CPU_qualification_path=str(study/'runtime_CPU_qualification.json'),new_fits=20,selection_seeds=[0,1],endpoint=20000,
     every_new_model_from_update_zero=True,reference_models_refitted=False,activation=False,GPU_allocation_created=False,full_goal_complete=False,SOTA_claim=False)
   write(STATE/'completed.json',result)
   report('completed','refinement097-all20-paired-cells-frozen-CPU-qualified-disabled-20261107','All reference seeds audited; ten native scalar settings at both matched seeds0/1 frozen and CPU-qualified. Plan disabled; exact current source/resource execution review remains.',result)
  except BaseException as exc:
   failure=dict(status='REFINEMENT097_CPU_PREPARATION_BOUNDARY_STOPPED',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),error=repr(exc),automatic_retry=False,activation=False,GPU_allocation_created=False)
   write(STATE/'failure.json',failure);report('failed','refinement097-CPU-preparation-boundary-stop-20261107','CPU paired-study preparation stopped; evidence retained, no activation or GPU allocation.',failure);raise
if __name__=='__main__':main()
