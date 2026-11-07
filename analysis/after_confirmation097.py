"""Finite CPU-only boundary: audit all fresh seeds, freeze and qualify benchmark.

This cannot enable a plan, create a GPU allocation or restart the training driver.
"""
import datetime,fcntl,hashlib,importlib.util,json,os,subprocess,time
from pathlib import Path
ROOT=Path('/ml2/alz');STATE=ROOT/'artifacts/benchmark097-boundary'
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
 assert p.returncode==0,name+' failed; retain evidence, no automatic retry or activation'
 return p.stdout
def commit(paths,message):
 command('git-add',['git','add','--',*map(str,paths)]);command('git-commit',['git','commit','--only','-m',message,'--',*map(str,paths)])
def report(kind,event,summary,details):
 spec=importlib.util.spec_from_file_location('existing_reporter',ROOT/'analysis/external_benchmark_driver.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.report(kind,event,summary,json.dumps(details))
def main():
 STATE.mkdir(exist_ok=True)
 with (STATE/'lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (STATE/'identity.json').exists(),'One-use boundary consumed'
  parent=read(ROOT/'artifacts/confirmation097-matched-seeds/identity.json');assert alive(parent)
  runtime=ROOT/'benchmark097_runtime';study=ROOT/'analysis/benchmark097'
  paths=[Path(__file__),ROOT/'analysis/audit_confirmation097.py',ROOT/'analysis/prepare_benchmark097.py',ROOT/'analysis/run_benchmark097.py',
    ROOT/'analysis/confirmation097_selection.py',ROOT/'analysis/confirmation097_selection_rule_20261106.json',
    *list(runtime.glob('*.py')),runtime/'policy.prepared.json',study/'protocol.json',
    *list((ROOT/'analysis/benchmark097_cfgen').glob('*.py')),
    *list((ROOT/'confirmation097_runtime').glob('*.py')),ROOT/'analysis/run_confirmation097.py',ROOT/'analysis/confirmation097/protocol.json',
    *list((ROOT/'analysis/distribution097/model').glob('*.py'))]
  pins={str(p):sha(p) for p in paths};expiry=min(time.time()+4*3600,read(ROOT/'analysis/research_window_20261104_five_days.json')['expires_unix']-600)
  write(STATE/'identity.json',dict(pid=os.getpid(),birth=int(Path('/proc',str(os.getpid()),'stat').read_text().rsplit(')',1)[1].split()[19]),
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),parent=parent,finite_expiry_unix=expiry,source_SHA256=pins,activation_permitted=False,GPU_allocation_permitted=False))
  try:
   while alive(parent):
    assert time.time()<expiry,'Finite boundary wait expired; inspect existing parent, do not replace'
    time.sleep(15)
   for path,pin in pins.items():assert sha(path)==pin,'Frozen boundary source changed'
   assert not (ROOT/'artifacts/confirmation097-matched-seeds/failure.json').exists()
   status=read(ROOT/'artifacts/confirmation097-matched-seeds/progress.json');assert status['activation_reverted'] and status['goal_still_active']
   assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True)
   audit=ROOT/'analysis/confirmation097_final_independent_audit_20261107.json'
   command('complete-confirmation-audit',[PG,'-I','-S','-B',ROOT/'analysis/audit_confirmation097.py','--output',audit],timeout=600)
   commit([audit],'Independently audit all fresh20k seeds, native counts, matched resampling and actual closed resources')
   command('freeze-benchmark',[PG,'-I','-S','-B',ROOT/'analysis/prepare_benchmark097.py','--confirmation-audit',audit])
   command('qualify-actual-benchmark-CPU',[PG,'-I','-S','-B',runtime/'qualify.py'],timeout=900)
   p=read(study/'protocol.json');q=read(study/'runtime_CPU_qualification.json')
   assert q['actual_tests']==24 and q['fullwidth_native_optimizer_pairs']==30 and not q['GPU_allocation_created'] and not q['CUDA_initialized']
   assert not read(runtime/'plan.json')['enabled']
   gp=ROOT/'analysis/current_research_goal.json';g=read(gp);g.update(study_status='FOUR_FRESH_SEEDS_AUDITED_EXTERNAL_CANDIDATE_FROZEN_ALL45_CPU_QUALIFIED_DISABLED',
      fresh_seed_final_audit_path=str(audit),frozen_external_protocol_path=str(study/'protocol.json'),external_actual_CPU_qualification_path=str(study/'runtime_CPU_qualification.json'),native_goal_status='active');write(gp,g)
   commit([study/'protocol.json',study/'runtime_CPU_qualification.json',gp],'Freeze confirmed PG candidate and qualify all45 fresh external cells; keep benchmark disabled')
   outcome=dict(status='CONFIRMATION_AUDITED_BENCHMARK_FROZEN_ALL45_CPU_QUALIFIED_DISABLED',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
      confirmation_audit_path=str(audit),frozen_protocol_path=str(study/'protocol.json'),CPU_qualification_path=str(study/'runtime_CPU_qualification.json'),
      PG_candidate_id=p['PG_candidate_id'],required_fresh_models=45,activation=False,GPU_allocation_created=False,full_goal_complete=False,SOTA_claim=False)
   write(STATE/'completed.json',outcome)
   report('completed','benchmark097-actual-confirmed-winner-frozen-all45-CPU-qualified-disabled-20261107','Fresh-seed confirmation independently audited; actual PG candidate frozen and all45 new external cells CPU-qualified. Plan disabled; exact current source/resource execution review remains.',outcome)
  except BaseException as exc:
   failure=dict(status='BENCHMARK097_BOUNDARY_PREPARATION_STOPPED',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),error=repr(exc),automatic_retry=False,activation=False,GPU_allocation_created=False,full_goal_complete=False)
   write(STATE/'failure.json',failure);report('failed','benchmark097-confirmation-boundary-preparation-stop-20261107','CPU boundary preparation stopped; evidence retained, no benchmark activation or GPU allocation.',failure);raise
if __name__=='__main__':main()
