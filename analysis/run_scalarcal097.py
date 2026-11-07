"""Run six separate fresh scalar-calibration fits in two fixed-device shared lanes.

All admission, source mutation and Git writes occur in this one CPU parent.
Only the two independently guarded finite model processes execute in parallel.
An incomplete cell stops new admissions while other admitted work settles.
"""
import argparse,datetime,fcntl,importlib.util,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path('/ml2/alz');RUNTIME=ROOT/'scalarcal097_runtime';STUDY=ROOT/'analysis/scalarcal097';STATE=ROOT/'artifacts/scalarcal097-fresh-six'
sys.path.insert(0,str(RUNTIME))
import binding as b,preset_scope as scope
from dual_coordination import GPUS,RESOURCE_PROTOCOL,assigned_gpu,validate_resource_protocol
ENV=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
def read(p):return json.loads(Path(p).read_bytes())
def write(p,q):
 p=Path(p);tmp=p.with_suffix('.partial');tmp.write_text(json.dumps(q,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def command(args,timeout=120):
 p=subprocess.run(list(map(str,args)),cwd=ROOT,env=ENV,capture_output=True,text=True,timeout=timeout)
 if p.returncode:raise RuntimeError(p.stderr[-3000:])
 return p.stdout
def matrix():return [scope.cell(arm,seed) for seed in (5,6,7) for arm in ('fixed','adaptive')]
def commit(paths,message):
 command(['git','add','--',*map(str,paths)]);command(['git','commit','--only','-m',message,'--',*map(str,paths)])
def report(kind,event,summary,details):
 spec=importlib.util.spec_from_file_location('existing_reporter',ROOT/'analysis/external_benchmark_driver.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.report(kind,event,summary,json.dumps(details))
def public_handle(h):
 return {k:h[k] for k in ('cell','gpu','identity','attempt') if k in h}

def run_queue(cells,*,start,ready,finish,observe,sleep,now,training_expiry):
 """The actual scheduler, with test-only callbacks supplied by imported CPU tests."""
 pending=list(cells);active={};rows=[];errors=[];unsettled=[]
 while pending or active:
  if not errors and pending and now()>=training_expiry-150:
   errors.append(dict(error='Owner training admission window ended',pending_cells=list(pending)))
  if not errors:
   for gpu in reversed(GPUS):
    if gpu in active:continue
    value=next((v for v in pending if assigned_gpu(v)==gpu),None)
    if value is None:continue
    try:
     h=start(value,len(rows))
     if h is not None:
      assert h['gpu']==gpu and h['cell']==value
      active[gpu]=h;pending.remove(value)
    except BaseException as exc:
     errors.append(dict(cell=value,error=repr(exc),stage='admission'));break
  for gpu,h in list(active.items()):
   try:
    if not ready(h):continue
    q=finish(h);rows.append(q);del active[gpu]
    if not q['complete']:errors.append(dict(cell=h['cell'],error='Incomplete native model cell retained',stage='endpoint'))
   except BaseException as exc:
    # An observation/audit error never authorizes restarting or killing a job.
    errors.append(dict(**public_handle(h),error=repr(exc),stage='observation_or_audit'))
    unsettled.append(public_handle(h));del active[gpu]
  observe(rows,pending,active,errors,unsettled)
  if errors and not active:break
  if now()>training_expiry+7200:
   errors.append(dict(error='Finite CPU observation allowance ended; exact handles retained'))
   unsettled.extend(public_handle(h) for h in active.values());break
  if pending or active:sleep(2 if active else 15)
 return dict(rows=rows,errors=errors,unsettled=unsettled,pending=pending)

def start_cell(value,completed):
 attempt=b.attempt_key(value);folder=ROOT/'artifacts/allocations'/attempt;result=STUDY/'results'/(attempt+'.json')
 assert not folder.exists() and not result.exists(),'One-use cell already exists; inspect rather than retry/reuse'
 assert not command(['git','status','--porcelain']),'Commit completed research outputs before next admission'
 exe=b.runtime_path(b.NATIVE_LAYOUT,value);prepare=[exe,'-I','-S','-B',RUNTIME/'prepare.py','--arm',scope.arm_of(value),'--seed',value['seed']]
 check=subprocess.run(list(map(str,[*prepare,'--check-only'])),cwd=ROOT,env=ENV,capture_output=True,text=True,timeout=120)
 if check.returncode:
  if any(text in check.stderr for text in ('Exact parallel GPU growth forecasts do not fit','Exact parallel host growth forecasts do not fit','ALZ early stop: actual GPU free below3.5GiB','ALZ early stop: host available below12GiB')):return None
  raise RuntimeError(check.stderr[-3000:])
 command(prepare);dep=read(folder/'deployment.json');auth=read(dep['authorization_path'])
 assert auth['gpu_uuid']==assigned_gpu(value) and not Path(auth['ledger_path']).exists() and not Path(auth['output_dir']).exists()
 with (folder/'owner.stdout').open('xb') as out,(folder/'owner.stderr').open('xb') as err:
  owner=subprocess.Popen([str(exe),'-I','-S','-B',str(RUNTIME/'launch_core_gpu0.py'),'--authorization',dep['authorization_path']],cwd=ROOT,env=ENV,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
 identity=dict(pid=owner.pid,birth=int(Path('/proc',str(owner.pid),'stat').read_text().rsplit(')',1)[1].split()[19]),boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
 write(folder/'launch_receipt.json',dict(**identity,cell=value,gpu_uuid=auth['gpu_uuid']))
 return dict(cell=value,gpu=auth['gpu_uuid'],identity=identity,attempt=attempt,owner=owner,auth=auth,exe=exe,settlement=Path(auth['ledger_path']).with_name(attempt+'-settlement.json'))
def ready_cell(h):
 if h['owner'].poll() is not None and not Path(h['auth']['ledger_path']).exists():raise RuntimeError('Owner exited without a ledger; exact attempt retained')
 if time.time()>h['auth']['expires_unix']+660:raise TimeoutError('Cell observation allowance ended; inspect original handles')
 return h['settlement'].exists() and h['owner'].poll() is not None
def finish_cell(h):
 h['owner'].wait(timeout=30)
 (STATE/(h['attempt']+'-finish.stdout')).write_text(command([h['exe'],'-I','-S','-B',RUNTIME/'finish.py','--attempt',h['attempt']],timeout=600))
 result=STUDY/'results'/(h['attempt']+'.json');q=read(result)
 assert q['cell']==h['cell'] and q['benchmark_protocol_SHA256']==b.sha(scope.PROTOCOL)
 assert q['own_processes_absent'] and q['owned_registry_lease_absent'] and q['all_new_training_from_scratch']
 commit([result],f'Retain fresh dual-device scalarcal097 {h["attempt"]}: native counts, scores, closure and actual cost')
 return q
def observe_queue(rows,pending,active,errors,unsettled):
 q=dict(status='DUAL_SHARED_FRESH6_RUNNING' if active else 'WAITING_FOR_SHARED_CAPACITY_NO_NEW_ALLOCATION',completed_cells=sum(bool(x['complete']) for x in rows),resolved_cells=len(rows),required_cells=6,pending_cells=len(pending),active=[public_handle(h) for h in active.values()],errors=errors,unsettled=unsettled,full_goal_complete=False,SOTA_claim=False)
 if errors:q['status']='NO_NEW_ADMISSIONS_DRAINING_OR_RETAINING_ORIGINAL_HANDLES'
 write(STATE/'progress.json',q)
 if rows:write(STATE/'partial_results.json',dict(rows=rows,required_cells=6,full_goal_complete=False))

def main():
 p=argparse.ArgumentParser();p.add_argument('--activation',required=True);args=p.parse_args()
 protocol=scope.protocol();scope.prior_study_ready();validate_resource_protocol();assert read(scope.PLAN)==scope.ENABLED
 disk=os.statvfs(ROOT);assert disk.f_bavail*disk.f_frsize>=2*2**30
 cells=matrix();assert len(cells)==len({b.allocation_key(v) for v in cells})==6
 STATE.mkdir(exist_ok=True)
 with (STATE/'driver.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (STATE/'identity.json').exists(),'One-use driver consumed'
  identity=dict(pid=os.getpid(),birth=int(Path('/proc',str(os.getpid()),'stat').read_text().rsplit(')',1)[1].split()[19]),boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),source_SHA256=b.sha(__file__),protocol_SHA256=b.sha(scope.PROTOCOL),resource_protocol_SHA256=b.sha(RESOURCE_PROTOCOL),activation=args.activation,required_cells=6)
  write(STATE/'identity.json',identity)
  try:
   outcome=run_queue(cells,start=start_cell,ready=ready_cell,finish=finish_cell,observe=observe_queue,sleep=time.sleep,now=time.time,training_expiry=read(scope.WINDOW)['expires_unix'])
   rows=outcome['rows']
   if outcome['errors'] or outcome['unsettled'] or len(rows)!=6 or not all(q['complete'] for q in rows):
    write(STATE/'failure.json',dict(status='DUAL6_STOPPED_RETAINING_ALL_OUTCOMES_AND_EXACT_HANDLES',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),**outcome,automatic_retry=False,full_goal_complete=False,SOTA_claim=False))
    raise RuntimeError('Full6 incomplete; original outcomes/handles retained without retry')
   result=dict(status='ALL45_FRESH_NATIVE_MODELS_AND_ORIGINAL_METRICS_AUDITED_AGGREGATION_PENDING',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),complete_cells=6,required_cells=6,all_new_training_from_scratch=True,initial_completed_updates=0,protocol_SHA256=b.sha(scope.PROTOCOL),resource_protocol_SHA256=b.sha(RESOURCE_PROTOCOL),result_SHA256={b.attempt_key(q['cell']):b.sha(STUDY/'results'/(b.attempt_key(q['cell'])+'.json')) for q in rows},actual_GPU_seconds=sum(q['charged_GPU_seconds'] for q in rows),matrix_complete=True,independent_full_matrix_audit_complete=False,paired_distribution_statistics_complete=False,calibration_only=True,external_matrix_reselected=False,full_goal_complete=False,SOTA_claim=False)
   path=STUDY/'matrix_complete.json';assert not path.exists();write(path,result)
   gp=ROOT/'analysis/current_research_goal.json';g=read(gp);g.update(scalar_calibration_matrix_path=str(path),scalar_calibration_fresh_fits_completed=6,native_goal_status='active');write(gp,g)
   commit([path,gp],'Retain six fresh scalar-LR calibration outcomes; paired calibration report remains')
   command(['git','revert','--no-edit',args.activation]);write(STATE/'progress.json',{**result,'activation_reverted':True})
   report('completed','benchmark097-dual-all6-closed-20261107','All6 fresh native fits and original scores audited; dual plan reverted. Calibration-only paired report and final audit remain.',result)
  except BaseException as exc:
   if not (STATE/'failure.json').exists():write(STATE/'failure.json',dict(error=repr(exc),status='DUAL6_OPERATOR_RECONCILIATION_REQUIRED',automatic_retry=False,full_goal_complete=False,SOTA_claim=False))
   report('failed','benchmark097-dual-stop-20261107','Dual benchmark stopped with original outcomes and exact handles retained. No retry or full-comparison claim.',read(STATE/'failure.json'));raise
if __name__=='__main__':main()
