"""At the exact closed stage2 boundary, freeze and qualify fresh-seed trials.

CPU only, one use and finite; this coordinator cannot activate or allocate.
"""
import datetime,fcntl,hashlib,importlib.util,json,os,subprocess,time
from pathlib import Path
ROOT=Path('/ml2/alz');STATE=ROOT/'artifacts/confirmation097-boundary'
PG=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark/current_base_20261029/runtime/bin/python')
ENV=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,q):Path(p).write_text(json.dumps(q,indent=2,allow_nan=False)+'\n')
def alive(i):
 p=Path('/proc',str(i['pid']),'stat')
 if not p.exists():return False
 s=p.read_text().rsplit(')',1)[1].split();return s[0]!='Z' and int(s[19])==i['birth'] and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==i['boot']
def command(name,args,timeout=120):
 p=subprocess.run(list(map(str,args)),cwd=ROOT,env=ENV,capture_output=True,text=True,timeout=timeout)
 (STATE/(name+'.stdout')).write_text(p.stdout);(STATE/(name+'.stderr')).write_text(p.stderr)
 assert p.returncode==0,name+' failed; retain evidence, no automatic retry or activation'
 return p.stdout
def commit(paths,message):
 command('git-add',['git','add','--',*map(str,paths)]);command('git-commit',['git','commit','--only','-m',message,'--',*map(str,paths)])
def report(kind,event,summary,details):
 spec=importlib.util.spec_from_file_location('existing_reporter',ROOT/'analysis/external_benchmark_driver.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.report(kind,event,summary,json.dumps(details))
def main():
 STATE.mkdir(exist_ok=True)
 with (STATE/'lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (STATE/'identity.json').exists(),'One-use coordinator consumed'
  parent=read(ROOT/'artifacts/distribution097b-stage2/driver_identity.json');assert alive(parent)
  paths=[Path(__file__),ROOT/'analysis/audit_distribution097b.py',ROOT/'analysis/prepare_confirmation097.py',ROOT/'analysis/confirmation097_selection.py',
         ROOT/'analysis/confirmation097_selection_rule_20261106.json',ROOT/'analysis/confirmation097_resampling_controls_CPU_20261106.json',ROOT/'analysis/run_confirmation097.py',
         *list((ROOT/'confirmation097_runtime').glob('*.py')),*list((ROOT/'analysis/confirmation097/model').glob('*.py')),
         *list((ROOT/'distribution097b_runtime').glob('*.py')),ROOT/'analysis/distribution097b/protocol.json']
  pins={str(p):sha(p) for p in paths};expiry=time.time()+3600
  write(STATE/'identity.json',dict(pid=os.getpid(),birth=int(Path('/proc',str(os.getpid()),'stat').read_text().rsplit(')',1)[1].split()[19]),boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),parent=parent,finite_expiry_unix=expiry,source_SHA256=pins,activation_permitted=False,GPU_allocation_permitted=False))
  try:
   while alive(parent):
    assert time.time()<expiry,'Finite wait expired; inspect the same live parent, no restart'
    time.sleep(10)
   for path,pin in pins.items():assert sha(path)==pin,'Frozen boundary source changed'
   assert not (ROOT/'artifacts/distribution097b-stage2/failure.json').exists()
   status=read(ROOT/'artifacts/distribution097b-stage2/progress.json');assert status['activation_reverted'] and status['goal_still_active']
   assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True)
   audit1=ROOT/'analysis/distribution097_stage1_final_independent_audit_20261106.json'
   audit2=ROOT/'analysis/distribution097b_stage2_final_independent_audit_20261106.json'
   command('complete-stage2-audit',[PG,'-I','-S','-B',ROOT/'analysis/audit_distribution097b.py','--complete','--output',audit2],timeout=600)
   commit([audit2],'Independently verify all stage2 fresh20k counts, scores and actual closed-resource rollback')
   command('freeze-confirmation',[PG,'-I','-S','-B',ROOT/'analysis/prepare_confirmation097.py','--stage1-audit',audit1,'--stage2-audit',audit2])
   runtime=ROOT/'confirmation097_runtime';study=ROOT/'analysis/confirmation097'
   command('qualify-confirmation-CPU',[PG,'-I','-S','-B',runtime/'qualify.py'],timeout=600)
   q=read(study/'runtime_CPU_qualification.json');p=read(study/'protocol.json');n=len(p['selected_configurations'])
   assert q['actual_runtime_tests']==22 and q['fullwidth_native_optimizer_pairs']==16*n and not q['GPU_allocation_created'] and not q['CUDA_initialized']
   assert not read(runtime/'plan.json')['enabled']
   gp=ROOT/'analysis/current_research_goal.json';g=read(gp);g.update(study_status='ALL_SIX_SELECTION_ENDPOINTS_AUDITED_MATCHED_SEED_CONFIRMATION_CPU_QUALIFIED_DISABLED',
       completed_distribution_stage2_audit_path=str(audit2),fresh_seed_confirmation_protocol_path=str(study/'protocol.json'),fresh_seed_confirmation_CPU_qualification_path=str(study/'runtime_CPU_qualification.json'),native_goal_status='active');write(gp,g)
   commit([study,gp],'Freeze distinct full20k confirmation choices and qualify all four fresh seed initializations; keep plan disabled')
   result=dict(status='ALL_SIX_SELECTION_ENDPOINTS_AUDITED_CONFIRMATION_FROZEN_CPU_QUALIFIED_DISABLED',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
      stage2_audit_path=str(audit2),confirmation_protocol_path=str(study/'protocol.json'),CPU_qualification_path=str(study/'runtime_CPU_qualification.json'),
      selected_configurations=p['selected_configurations'],required_fresh_fits=4*n,training_seeds=[1,2,3,4],activation=False,GPU_allocation_created=False,full_goal_complete=False,SOTA_claim=False)
   write(STATE/'completed.json',result)
   report('completed','confirmation097-configurations-frozen-all-seed-cpu-qualified-disabled-20261106','All six seed0 selection endpoints audited; distinct fresh20k confirmation choices and all four seeds CPU-qualified. Plan remains disabled; exact execution review still required.',result)
  except BaseException as exc:
   q=dict(status='CONFIRMATION097_BOUNDARY_PREPARATION_STOPPED',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),error=repr(exc),automatic_retry=False,activation=False,GPU_allocation_created=False)
   write(STATE/'failure.json',q);report('failed','confirmation097-boundary-preparation-stop-20261106','CPU confirmation boundary preparation stopped; evidence retained, no activation or GPU allocation.',q);raise
if __name__=='__main__':main()
