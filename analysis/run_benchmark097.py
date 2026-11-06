"""Run all45 frozen fresh model fits serially; no test-driven reselection.

Incomplete cells stop the queue with their costs and evidence retained. The
original scientific and physical GPU locks and canonical budgets remain in use.
"""
import argparse,datetime,fcntl,importlib.util,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path('/ml2/alz');RUNTIME=ROOT/'benchmark097_runtime';STUDY=ROOT/'analysis/benchmark097';STATE=ROOT/'artifacts/benchmark097-fullmatrix'
sys.path.insert(0,str(RUNTIME))
import binding as b,preset_scope as scope
ENV=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
def read(p):return json.loads(Path(p).read_bytes())
def write(p,q):
 p=Path(p);tmp=p.with_suffix('.partial');tmp.write_text(json.dumps(q,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def command(args,timeout=120):
 p=subprocess.run(list(map(str,args)),cwd=ROOT,env=ENV,capture_output=True,text=True,timeout=timeout)
 if p.returncode:raise RuntimeError(p.stderr[-3000:])
 return p.stdout
def matrix():return [scope.cell(f,m,s) for s in range(5) for f in range(3) for m in ('pg','scvi','cfgen')]
def commit(paths,message):
 command(['git','add','--',*map(str,paths)]);command(['git','commit','--only','-m',message,'--',*map(str,paths)])
def report(kind,event,summary,details):
 spec=importlib.util.spec_from_file_location('existing_reporter',ROOT/'analysis/external_benchmark_driver.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.report(kind,event,summary,json.dumps(details))
def fit(value,completed_count):
 attempt=b.attempt_key(value);folder=ROOT/'artifacts/allocations'/attempt;result=STUDY/'results'/(attempt+'.json')
 assert not folder.exists() and not result.exists(),'One-use cell already exists; inspect rather than retry/reuse'
 assert not command(['git','status','--porcelain']),'Commit completed research outputs before the next cell'
 exe=b.runtime_path(b.NATIVE_LAYOUT,value)
 prepare=[exe,'-I','-S','-B',RUNTIME/'prepare.py','--fold',value['fold'],'--method',value['method'],'--seed',value['seed']]
 while True:
  assert time.time()<read(scope.WINDOW)['expires_unix']-150,'Owner research window ended; preserve incomplete matrix'
  p=subprocess.run(list(map(str,[*prepare,'--check-only'])),cwd=ROOT,env=ENV,capture_output=True,text=True,timeout=120)
  if p.returncode==0:break
  if not any(x in p.stderr for x in ('Exact parallel GPU growth forecasts do not fit','Exact parallel host growth forecasts do not fit','ALZ early stop: actual GPU free below3.5GiB','ALZ early stop: host available below12GiB')):raise RuntimeError(p.stderr[-3000:])
  write(STATE/'progress.json',dict(status='WAITING_FOR_SHARED_CAPACITY_NO_NEW_ALLOCATION',cell=value,completed_cells=completed_count));time.sleep(15)
 (STATE/(attempt+'-prepare.stdout')).write_text(command(prepare))
 dep=read(folder/'deployment.json');assert b.sha(dep['authorization_path'])==dep['authorization_SHA256'];auth=read(dep['authorization_path'])
 assert not Path(auth['ledger_path']).exists() and not Path(auth['output_dir']).exists()
 with (folder/'owner.stdout').open('xb') as out,(folder/'owner.stderr').open('xb') as err:
  owner=subprocess.Popen([str(exe),'-I','-S','-B',str(RUNTIME/'launch_core_gpu0.py'),'--authorization',dep['authorization_path']],cwd=ROOT,env=ENV,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
 write(folder/'launch_receipt.json',dict(pid=owner.pid,birth=int(Path('/proc',str(owner.pid),'stat').read_text().rsplit(')',1)[1].split()[19]),boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),cell=value))
 write(STATE/'progress.json',dict(status='FRESH_FROZEN_EXTERNAL_BENCHMARK_CELL_RUNNING',cell=value,attempt=attempt,completed_cells=completed_count,initial_completed_updates=0,from_scratch=True,endpoint=auth['fixed_endpoint'],endpoint_or_update_unit=auth['new_model_update_unit']))
 settlement=Path(auth['ledger_path']).with_name(attempt+'-settlement.json')
 while not settlement.exists():
  assert time.time()<auth['expires_unix']+660,'Cell did not settle by finite deadline; inspect exact identities'
  if owner.poll() is not None and not Path(auth['ledger_path']).exists():raise RuntimeError('Owner terminated without a ledger; no automatic retry')
  time.sleep(10)
 owner.wait(timeout=60)
 (STATE/(attempt+'-finish.stdout')).write_text(command([exe,'-I','-S','-B',RUNTIME/'finish.py','--attempt',attempt],timeout=600))
 q=read(result);assert q['cell']==value and q['benchmark_protocol_SHA256']==b.sha(scope.PROTOCOL)
 assert q['own_processes_absent'] and q['owned_registry_lease_absent'] and q['all_new_training_from_scratch']
 commit([result],f'Retain fresh external097 f{value["fold"]} {value["method"]} seed{value["seed"]}: native counts, scores, actual cost and outcome')
 assert q['complete'],'Incomplete model cell retained; full matrix queue stopped without retry or success claim'
 return q
def main():
 p=argparse.ArgumentParser();p.add_argument('--activation',required=True);args=p.parse_args()
 protocol=scope.protocol();assert read(scope.PLAN)==scope.ENABLED
 disk=os.statvfs(ROOT);assert disk.f_bavail*disk.f_frsize>=protocol['retained_PC_disk_bytes']+10*2**30,'Insufficient local output space for declared full-PC retention'
 cells=matrix();assert len(cells)==45 and len({b.allocation_key(c) for c in cells})==45
 STATE.mkdir(exist_ok=True)
 with (STATE/'driver.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  assert not (STATE/'identity.json').exists(),'One-use benchmark driver consumed; inspect rather than replace/restart'
  write(STATE/'identity.json',dict(pid=os.getpid(),birth=int(Path('/proc',str(os.getpid()),'stat').read_text().rsplit(')',1)[1].split()[19]),boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),source_SHA256=b.sha(__file__),protocol_SHA256=b.sha(scope.PROTOCOL),activation=args.activation,required_cells=45))
  rows=[]
  try:
   for value in cells:
    rows.append(fit(value,len(rows)));write(STATE/'partial_results.json',dict(completed_cells=len(rows),required_cells=45,full_goal_complete=False,rows=rows))
   result=dict(status='ALL45_FRESH_NATIVE_MODELS_AND_ORIGINAL_METRICS_AUDITED_AGGREGATION_PENDING',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               complete_cells=45,required_cells=45,all_new_training_from_scratch=True,initial_completed_updates=0,protocol_SHA256=b.sha(scope.PROTOCOL),
               result_SHA256={b.attempt_key(q['cell']):b.sha(STUDY/'results'/(b.attempt_key(q['cell'])+'.json')) for q in rows},
               actual_GPU_seconds=sum(q['charged_GPU_seconds'] for q in rows),
               matrix_complete=True,independent_full_matrix_audit_complete=False,paired_distribution_statistics_complete=False,
               no_test_driven_parameter_reselection=True,conditional_historically_exposed_cohort=True,full_goal_complete=False,SOTA_claim=False)
   path=STUDY/'matrix_complete.json';assert not path.exists();write(path,result)
   gp=ROOT/'analysis/current_research_goal.json';g=read(gp);g.update(study_status=result['status'],frozen_external_matrix_path=str(path),native_goal_status='active');write(gp,g)
   commit([path,gp],'Record all45 audited fresh benchmark cells; independent aggregate comparison still pending')
   command(['git','revert','--no-edit',args.activation]);write(STATE/'progress.json',{**result,'activation_reverted':True})
   report('completed','benchmark097-all45-fresh-native-cells-audited-aggregate-pending-20261106','All45 fresh native PG/scVI/CFGen fits and original scores audited; plan actually reverted. Full paired statistics and independent comparison review remain; no SOTA claim.',result)
  except BaseException as exc:
   q=dict(status='BENCHMARK097_STOPPED_RETAINING_ALL_OUTCOMES',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),error=repr(exc),completed_cells=len(rows),required_cells=45,automatic_retry=False,full_goal_complete=False,SOTA_claim=False)
   write(STATE/'failure.json',q);report('failed','benchmark097-fresh-matrix-stopped-20261106','Fresh benchmark stopped on an incomplete or operational cell; costs and evidence retained, no retry or full-matrix success claim.',q);raise
if __name__=='__main__':main()
