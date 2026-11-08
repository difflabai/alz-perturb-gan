"""Finish untouched original cells under unchanged native guards; never retry here."""
import datetime,fcntl,hashlib,importlib.util,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path('/ml2/alz');STATE=ROOT/'artifacts/benchmark097-after-capacity-continuation-v2';PROTOCOL=ROOT/'analysis/benchmark097_capacity_recovery_continuation_v2_protocol_20261108.json';PROOF=ROOT/'analysis/benchmark097_capacity_continuation_v2_CPU_check_20261108.json';OLD=ROOT/'artifacts/benchmark097-balanced-fullmatrix';FAILED='CAL_GPU0_V1_external097_f2_cfgen_s1'
sys.path.insert(0,str(ROOT/'benchmark097_balanced_runtime'))
import binding as b,preset_scope as scope
from dual_coordination import GPUS,identity
from cell_metadata import completed_metadata
from runtime_dispatch import cost_table,choose_next
spec=importlib.util.spec_from_file_location('unchanged_balanced_native_execution',ROOT/'analysis/run_benchmark097_balanced.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,q):
 p=Path(p);t=p.with_suffix('.partial');t.write_text(json.dumps(q,indent=2,allow_nan=False)+'\n');t.replace(p)
def stability_ready(cache,gpu,now,enough,seconds):
 if not enough:cache.pop(gpu,None);return False
 cache.setdefault(gpu,now);return now-cache[gpu]>=seconds
def drive(pending,rows,active,*,key,start,ready,finish,select,observe,sleep,now,expiry):
 pending=list(pending);rows=list(rows);active=dict(active);seen={key(r['cell']) for r in rows}|{key(h['cell']) for h in active.values()};assert len(seen)==len(rows)+len(active) and not seen.intersection(key(v) for v in pending);errors=[];unsettled=[]
 while pending or active:
  if pending and not errors and now()>=expiry-150:errors.append(dict(error='Authorized training admission window ended',pending=list(pending)))
  if not errors:
   for gpu in reversed(GPUS):
    if gpu in active:continue
    v=select(pending,active,gpu)
    if v is None:continue
    assert v in pending and key(v) not in seen
    try:
     h=start(v,gpu,len(rows))
     if h is not None:assert h['cell']==v and h['gpu']==gpu;active[gpu]=h;pending.remove(v);seen.add(key(v))
    except BaseException as exc:errors.append(dict(cell=v,stage='admission',error=repr(exc)));break
  for gpu,h in list(active.items()):
   try:
    if not ready(h):continue
    r=finish(h);rows.append(r);del active[gpu]
    if not r['complete']:errors.append(dict(cell=h['cell'],stage='endpoint',error='Additional incomplete cell retained; no retry or further admission'))
   except BaseException as exc:errors.append(dict(cell=h['cell'],stage='observation_or_audit',error=repr(exc)));unsettled.append({k:h[k] for k in ['cell','gpu','identity','attempt']});del active[gpu]
  observe(rows,pending,active,errors,unsettled)
  if errors and not active:break
  if now()>expiry+7200:errors.append(dict(error='Finite observation expired; existing handles retained'));unsettled.extend({k:h[k] for k in ['cell','gpu','identity','attempt']} for h in active.values());break
  if pending or active:sleep(2 if active else 15)
 return dict(rows=rows,errors=errors,unsettled=unsettled,pending=pending)

def successful_costs():
 rows=[]
 for p in sorted((scope.STUDY/'results').glob('*.json')):
  a=read(ROOT/'artifacts/allocations'/p.stem/'authorization.json');ledger=Path(a['ledger_path']);j=read(ledger)['jobs'][0];s=read(ledger.with_name(p.stem+'-settlement.json'));assert j['reservation_released'] and s['uncertain'] is None and s['settlement']['status']=='settled'
  if p.stem in read(PROTOCOL)['known_failed_attempts']:
   assert sha(p)==read(PROTOCOL)['known_failed_attempts'][p.stem] and j['status']=='failed' and 'Protected/registered GPU growth no longer fits' in j['error'];continue
  assert j['status']=='completed' and j['exit_code']==0
  rows.append(dict(method=a['cell']['method'],gpu_uuid=a['gpu_uuid'],seconds=s['settlement']['charged_GPU_seconds'],checked_unix=s['checked_unix'],attempt=p.stem))
 return rows

def main():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not old.command(['git','status','--porcelain'])
 plan=read(PROTOCOL);assert plan['known_failed_attempt']==FAILED and plan['continue_only_original_never_started_scientific_keys']
 proof=read(PROOF);assert proof['status']=='UNTOUCHED_CELL_CONTINUATION_CPU_CHECK_PASS' and proof['source_SHA256']==sha(__file__) and proof['protocol_SHA256']==sha(PROTOCOL)
 inherited=read(scope.RUNTIME_PROOF)
 for p,pin in inherited['source_SHA256'].items():assert sha(p)==pin,p
 assert read(scope.PLAN)==scope.ENABLED and sha(ROOT/'analysis/run_benchmark097_balanced.py')==plan['original_driver_source_SHA256'];scope.protocol()
 progress=read(OLD/'progress.json');assert len(progress['errors'])==1 and progress['errors'][0]['cell']==scope.cell(2,'cfgen',1) and not progress['unsettled']
 original=read(OLD/'identity.json');expected={k:original[k] for k in ['pid','birth','boot']};assert original['pid']==600656 and original['birth']==25866551
 cells=old.matrix();rows=[];active={};legacy_live=identity(expected['pid'])==expected;legacy_cell=scope.cell(0,'cfgen',1)
 for v in cells:
  if legacy_live and v==legacy_cell:continue
  p=scope.STUDY/'results'/(b.attempt_key(v)+'.json')
  if p.exists():rows.append(completed_metadata(v,require_full=False))
 if legacy_live:
  if progress['active']:
   assert len(progress['active'])==1;h=progress['active'][0];assert h['cell']==legacy_cell
  else:
   key=b.attempt_key(legacy_cell);a=read(ROOT/'artifacts/allocations'/key/'authorization.json');j=read(a['ledger_path'])['jobs'][0];h=dict(cell=legacy_cell,gpu=a['gpu_uuid'],identity=j['guard_identity'],attempt=key)
  active[h['gpu']]=dict(h,legacy=True,legacy_driver=expected)
 else:assert (scope.STUDY/'results/CAL_GPU0_V1_external097_f0_cfgen_s1.json').exists()
 assert {r['attempt']:r['source_result_SHA256'] for r in rows if not r['complete']}==plan['known_failed_attempts']
 seen={b.attempt_key(r['cell']) for r in rows}|{h['attempt'] for h in active.values()};pending=[v for v in cells if b.attempt_key(v) not in seen]
 STATE.mkdir(exist_ok=False)
 with (STATE/'driver.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);write(STATE/'identity.json',dict(**identity(os.getpid()),source_SHA256=sha(__file__),protocol_SHA256=sha(PROTOCOL),inherited_native_qualification_SHA256=sha(scope.RUNTIME_PROOF),activation=original['activation'],new_model_retries=False))
  stable={};polls={}
  def capacity_stable(gpu):
   import resource_profile as profile
   if gpu in polls and time.time()-polls[gpu][0]<30:return polls[gpu][1]
   try:
    auth,_=__import__('prepare').build_authorization(fold=2,method='scvi',seed=1,gpu=gpu,check_only=True);_,admission,_=b.modules(auth);observation=admission.snapshot();forecast=profile.capacity(observation,admission,gpu);profile.free_floor(gpu)
    enough=forecast['GPU_total_bytes']-forecast['GPU_required_bytes']>=plan['scheduling_only_extra_GPU_capacity_bytes']
   except AssertionError:enough=False
   result=stability_ready(stable,gpu,time.time(),enough,plan['scheduling_only_stability_seconds']);polls[gpu]=(time.time(),result);return result
  def select(pending,active,gpu):
   if not pending or not capacity_stable(gpu):return None
   costs,evidence=cost_table(successful_costs());timed={}
   for g,h in active.items():
    auth=h.get('auth') or read(ROOT/'artifacts/allocations'/h['attempt']/'authorization.json');job=read(auth['ledger_path'])['jobs'][0];timed[g]=dict(cell=h['cell'],elapsed_seconds=max(0.,time.time()-job['started_unix']))
   choice,forecast=choose_next(pending,timed,gpu,costs);write(STATE/'latest_cost_choice.json',dict(selected=choice,gpu=gpu,forecast=forecast,quality_values_read=False));return choice
  def start(v,gpu,n):
   h=old.start_cell(v,gpu,n)
   if h is not None:
    write(STATE/(h['attempt']+'-start.json'),dict(cell=v,gpu=gpu,identity=h['identity'],source_SHA256=sha(__file__),from_update_zero=True));print(json.dumps(dict(event='native_guard_launched',attempt=h['attempt'],gpu=gpu,not_yet_a_native_progress_claim=True)),flush=True)
   return h
  def ready(h):
   if not h.get('legacy'):return old.ready_cell(h)
   return identity(h['legacy_driver']['pid'])!=h['legacy_driver']
  def finish(h):
   if not h.get('legacy'):return old.finish_cell(h)
   p=scope.STUDY/'results'/(h['attempt']+'.json');assert p.exists(),'Original parent exited without its audited result; retain handles'
   return completed_metadata(h['cell'],require_full=False)
  def observe(rows,pending,active,errors,unsettled):
   write(STATE/'progress.json',dict(status='UNTOUCHED_ORIGINAL_CELLS_CONTINUE_KNOWN_INFRA_FAILURE_RETAINED' if not errors else 'STOP_NEW_ADMISSIONS_DRAIN_HEALTHY_HANDLES',resolved_original_cells=len(rows),complete_original_cells=sum(r['complete'] for r in rows),known_infra_failed_attempts=plan['known_failed_attempts'],pending_original_cells=len(pending),active=[{k:h[k] for k in ['cell','gpu','identity','attempt']} for h in active.values()],errors=errors,unsettled=unsettled,full_goal_complete=False,SOTA_claim=False))
  try:
   out=drive(pending,rows,active,key=b.attempt_key,start=start,ready=ready,finish=finish,select=select,observe=observe,sleep=time.sleep,now=time.time,expiry=read(scope.WINDOW)['expires_unix'])
   if out['errors'] or out['unsettled'] or len(out['rows'])!=45:write(STATE/'failure.json',dict(status='ADDITIONAL_ISSUE_RETAIN_ORIGINAL_HANDLES',**out));raise RuntimeError('Original attempt continuation incomplete; no model restart')
   assert sum(r['complete'] for r in out['rows'])==43
   record=dict(status='ALL45_ORIGINAL_ATTEMPTS_CLOSED_43_COMPLETE_TWO_INFRA_FAILURES_FRESH_REPLACEMENTS_REMAIN',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),original_scientific_cells=45,successful_original_models=43,failed_original_attempts=sorted(plan['known_failed_attempts']),result_SHA256={r['attempt']:r['source_result_SHA256'] for r in out['rows']},actual_original_GPU_seconds=sum(r['charged_GPU_seconds'] for r in out['rows']),all_new_models_from_zero=True,failed_weights_not_resumed=True,fresh_infrastructure_replacement_required=True,original57_and53_reports_pending=True,scalar6_fresh_fits_pending=True,full_goal_complete=False,SOTA_claim=False)
   p=ROOT/'analysis/benchmark097_original_attempts_postcapacity_closed.json';assert not p.exists();write(p,record);old.commit([p],'Retain all45 original attempt outcomes including infrastructure failure before fresh replacement');write(STATE/'completed.json',record)
  except BaseException as exc:
   if not (STATE/'failure.json').exists():write(STATE/'failure.json',dict(error=repr(exc),automatic_retry=False,existing_handles_retained=True))
   raise
if __name__=='__main__':main()
