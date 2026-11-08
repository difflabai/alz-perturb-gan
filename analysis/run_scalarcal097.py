"""Resource-only dispatch for the six fresh scalar-calibration models.

Previously completed native fits are immutable references. Remaining scientific
cells keep their original canonical keys and start once at update0. No result
quality or target array is read by the scheduling decisions.
"""
import argparse,datetime,fcntl,importlib.util,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path('/ml2/alz');RUNTIME=ROOT/'scalarcal097_runtime';STUDY=ROOT/'analysis/scalarcal097';STATE=ROOT/'artifacts/scalarcal097-fresh-six'
sys.path.insert(0,str(RUNTIME))
import binding as b,preset_scope as scope
from dual_coordination import GPUS,RESOURCE_PROTOCOL,identity,validate_resource_protocol
from runtime_dispatch import choose_next,settled_cost_rows,cost_table
from cell_metadata import completed_metadata
ENV=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
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

def matrix():return [scope.cell(arm,seed) for seed in (5,6,7) for arm in ('fixed','adaptive')]
def public_handle(h):return {k:h[k] for k in ('cell','gpu','identity','attempt') if k in h}
def run_queue(cells,*,initial_rows,start,ready,finish,observe,sleep,now,training_expiry,select):
 pending=list(cells);active={};rows=list(initial_rows);errors=[];unsettled=[]
 seen={b.allocation_key(r['cell']) for r in rows}
 assert len(seen)==len(rows) and not seen.intersection(b.allocation_key(v) for v in pending)
 while pending or active:
  if not errors and pending and now()>=training_expiry-150:errors.append(dict(error='Owner training admission window ended',pending_cells=list(pending)))
  if not errors:
   for gpu in reversed(GPUS):
    if gpu in active:continue
    value=select(pending,active,gpu)
    if value is None:continue
    assert value in pending and b.allocation_key(value) not in seen
    try:
     h=start(value,gpu,len(rows))
     if h is not None:
      assert h['gpu']==gpu and h['cell']==value
      active[gpu]=h;pending.remove(value);seen.add(b.allocation_key(value))
    except BaseException as exc:errors.append(dict(cell=value,error=repr(exc),stage='admission'));break
  for gpu,h in list(active.items()):
   try:
    if not ready(h):continue
    q=finish(h);rows.append(q);del active[gpu]
    if not q['complete']:errors.append(dict(cell=h['cell'],error='Incomplete native model cell retained',stage='endpoint'))
   except BaseException as exc:
    errors.append(dict(**public_handle(h),error=repr(exc),stage='observation_or_audit'));unsettled.append(public_handle(h));del active[gpu]
  observe(rows,pending,active,errors,unsettled)
  if errors and not active:break
  if now()>training_expiry+7200:
   errors.append(dict(error='Finite CPU observation allowance ended; exact handles retained'));unsettled.extend(public_handle(h) for h in active.values());break
  if pending or active:sleep(2 if active else 15)
 return dict(rows=rows,errors=errors,unsettled=unsettled,pending=pending)

_CAPACITY={}
def capacity_now(value,gpu):
 import prepare,resource_profile as profile
 if gpu in _CAPACITY and _CAPACITY[gpu][0]>time.time():return _CAPACITY[gpu][1]
 auth,_=prepare.build_authorization(arm=scope.arm_of(value),seed=value['seed'],gpu=gpu,check_only=True)
 try:
  _,admission,_=b.modules(auth);profile.capacity(admission.snapshot(),admission,gpu);profile.free_floor(gpu)
  _CAPACITY[gpu]=(time.time()+30,True);return True
 except AssertionError as exc:
  if any(text in str(exc) for text in ('growth forecasts do not fit','ALZ early stop:')):
   _CAPACITY[gpu]=(time.time()+15,False);return False
  raise

def select_native(pending,active,gpu):
 if not pending:return None
 costs,evidence=cost_table(settled_cost_rows())
 eligible=set(active)
 for card in GPUS:
  if card not in active and capacity_now(pending[0],card):eligible.add(card)
 timed={g:dict(cell=h['cell'],elapsed_seconds=max(0.,time.time()-h['ledger_started_unix'])) for g,h in active.items()}
 chosen,allocation=choose_next(pending,timed,gpu,costs,eligible=eligible)
 write(STATE/'latest_cost_dispatch.json',dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),requested_gpu=gpu,selected_cell=chosen,eligible_cards=sorted(eligible),method_card_cost_seconds=costs,cost_source_attempts=evidence,allocation=allocation,quality_values_used=False,target_arrays_loaded=False,benchmark_derived_cost_overhead_multiplier=1.15))
 return chosen

def start_cell(value,gpu,completed):
 attempt=b.attempt_key(value);folder=ROOT/'artifacts/allocations'/attempt;result=STUDY/'results'/(attempt+'.json')
 assert not folder.exists() and not result.exists(),'One-use scientific key already prepared/consumed; inspect instead of retry'
 assert not command(['git','status','--porcelain']),'Commit completed research outputs before new admission'
 exe=b.runtime_path(b.NATIVE_LAYOUT,value);prepare=[exe,'-I','-S','-B',RUNTIME/'prepare.py','--arm',scope.arm_of(value),'--seed',value['seed'],'--gpu',gpu]
 check=subprocess.run(list(map(str,[*prepare,'--check-only'])),cwd=ROOT,env=ENV,capture_output=True,text=True,timeout=120)
 if check.returncode:
  if any(text in check.stderr for text in ('Exact parallel GPU growth forecasts do not fit','Exact parallel host growth forecasts do not fit','ALZ early stop: actual GPU free below3.5GiB','ALZ early stop: host available below12GiB')):
   _CAPACITY[gpu]=(time.time()+15,False);return None
  raise RuntimeError(check.stderr[-3000:])
 command(prepare);dep=read(folder/'deployment.json');auth=read(dep['authorization_path']);assert auth['gpu_uuid']==gpu and not Path(auth['ledger_path']).exists() and not Path(auth['output_dir']).exists()
 with (folder/'owner.stdout').open('xb') as out,(folder/'owner.stderr').open('xb') as err:
  owner=subprocess.Popen([str(exe),'-I','-S','-B',str(RUNTIME/'launch_core_gpu0.py'),'--authorization',dep['authorization_path']],cwd=ROOT,env=ENV,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
 ident=identity(owner.pid);assert ident is not None;write(folder/'launch_receipt.json',dict(**ident,cell=value,gpu_uuid=gpu))
 return dict(cell=value,gpu=gpu,identity=ident,attempt=attempt,owner=owner,auth=auth,exe=exe,ledger_started_unix=time.time(),settlement=Path(auth['ledger_path']).with_name(attempt+'-settlement.json'))
def ready_cell(h):
 if h['owner'].poll() is not None and not Path(h['auth']['ledger_path']).exists():raise RuntimeError('Owner exited without a ledger; original attempt retained')
 if time.time()>h['auth']['expires_unix']+660:raise TimeoutError('Cell observation allowance ended; inspect exact handles')
 ledger=Path(h['auth']['ledger_path'])
 if ledger.exists():h['ledger_started_unix']=read(ledger)['jobs'][0]['started_unix']
 return h['settlement'].exists() and h['owner'].poll() is not None

def finish_cell(h):
 h['owner'].wait(timeout=30)
 (STATE/(h['attempt']+'-finish.stdout')).write_text(command([h['exe'],'-I','-S','-B',RUNTIME/'finish.py','--attempt',h['attempt']],timeout=600))
 result=STUDY/'results'/(h['attempt']+'.json')
 # The unchanged scientific auditor wrote this result only after native state,
 # count, saved-score and clean settlement checks. Cost scheduling uses metadata.
 row=completed_metadata(h['cell'],require_full=False);commit([result],f'Retain fresh balanced-resource {h["attempt"]}: full native endpoint, counts, provenance and cost')
 return row

def observe_queue(rows,pending,active,errors,unsettled):
 q=dict(status='SCALARCAL097_FRESH_SIX_COST_AND_CAPACITY_RUNNING' if active else 'WAITING_FOR_SHARED_CAPACITY',completed_cells=sum(bool(r['complete']) for r in rows),resolved_cells=len(rows),required_cells=6,pending_cells=len(pending),active=[public_handle(h) for h in active.values()],errors=errors,unsettled=unsettled,scientific_protocol_unchanged=True,full_goal_complete=False,SOTA_claim=False)
 if errors:q['status']='NO_NEW_ADMISSIONS_DRAINING_OR_RETAINING_EXACT_HANDLES'
 write(STATE/'progress.json',q)
 if rows:write(STATE/'partial_results.json',dict(rows=rows,required_cells=6,full_goal_complete=False))

def main():
 p=argparse.ArgumentParser();p.add_argument('--activation',required=True);args=p.parse_args()
 protocol=scope.protocol();scope.prior_study_ready()
 validate_resource_protocol();assert read(scope.PLAN)==scope.ENABLED
 disk=os.statvfs(ROOT);assert disk.f_bavail*disk.f_frsize>=2*2**30
 cells=matrix();assert len(cells)==len({b.allocation_key(v) for v in cells})==6
 initial=[];pending=cells
 assert not list((STUDY/'results').glob('*.json')),'Fresh six-fit study cannot adopt previous outcomes or weights'
 STATE.mkdir(exist_ok=True)
 with (STATE/'driver.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (STATE/'identity.json').exists(),'One-use replacement driver consumed'
  ident=dict(**identity(os.getpid()),source_SHA256=b.sha(__file__),protocol_SHA256=b.sha(scope.PROTOCOL),resource_protocol_SHA256=b.sha(RESOURCE_PROTOCOL),activation=args.activation,required_cells=6,prior_scientific_completion_SHA256=scope.prior_study_ready(),immutable_initial_completed_cells=len(initial));write(STATE/'identity.json',ident)
  try:
   outcome=run_queue(pending,initial_rows=initial,start=start_cell,ready=ready_cell,finish=finish_cell,observe=observe_queue,sleep=time.sleep,now=time.time,training_expiry=read(scope.WINDOW)['expires_unix'],select=select_native)
   rows=outcome['rows']
   if outcome['errors'] or outcome['unsettled'] or len(rows)!=6 or not all(r['complete'] for r in rows):
    write(STATE/'failure.json',dict(status='SCALARCAL6_INCOMPLETE_ORIGINAL_OUTCOMES_AND_HANDLES_RETAINED',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),**outcome,automatic_retry=False,full_goal_complete=False,SOTA_claim=False));raise RuntimeError('Full6 incomplete; retain original scientific keys without retries')
   result=dict(status='ALL_SIX_FRESH_SCALAR_LR_NATIVE_MODELS_AND_CALIBRATION_AUDITS_COMPLETE',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),complete_cells=6,required_cells=6,all_new_training_from_scratch=True,initial_completed_updates=0,protocol_SHA256=b.sha(scope.PROTOCOL),resource_protocol_SHA256=b.sha(RESOURCE_PROTOCOL),result_SHA256={r['attempt']:r['source_result_SHA256'] for r in rows},resource_runtime_by_attempt={r['attempt']:r['runtime_directory'] for r in rows},actual_GPU_seconds=sum(r['charged_GPU_seconds'] for r in rows),matrix_complete=True,calibration_only=True,external45_candidate_unchanged=True,quality_or_target_values_used_for_scheduling=False,full_goal_complete=False,SOTA_claim=False)
   path=STUDY/'matrix_complete.json';assert not path.exists();write(path,result);commit([path],'Retain all6 frozen native outcomes across the verified resource-only handoff')
   command(['git','revert','--no-edit',args.activation]);write(STATE/'progress.json',{**result,'activation_reverted':True})
   report('completed','scalarcal097-cost-dispatch-all6-closed','All6 frozen fresh native fits complete across the resource handoff; final57/53 analysis remains.',result)
  except BaseException as exc:
   if not (STATE/'failure.json').exists():write(STATE/'failure.json',dict(status='SCALARCAL6_ORIGINAL_HANDLES_REQUIRE_RECONCILIATION',error=repr(exc),automatic_retry=False,full_goal_complete=False,SOTA_claim=False))
   report('failed','scalarcal097-cost-dispatch-stopped','Balanced resource driver stopped; original outcomes/handles retained without model retry.',read(STATE/'failure.json'));raise
if __name__=='__main__':main()
