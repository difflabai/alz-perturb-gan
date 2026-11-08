"""Finish untouched original cells under unchanged native guards; never retry here."""
import datetime,traceback,errno,re,fcntl,hashlib,importlib.util,json,os,subprocess,sys,time
from itertools import product
from pathlib import Path
ROOT=Path('/ml2/alz');STATE=ROOT/'artifacts/benchmark097-after-capacity-continuation-v8';PROTOCOL=ROOT/'analysis/benchmark097_capacity_recovery_continuation_v8_protocol_20261108.json';PROOF=ROOT/'analysis/benchmark097_capacity_continuation_v8_CPU_check_20261108.json';OLD=ROOT/'artifacts/benchmark097-balanced-fullmatrix';FAILED='CAL_GPU0_V1_external097_f2_cfgen_s1'
sys.path.insert(0,str(ROOT/'benchmark097_balanced_runtime'))
import binding as b,preset_scope as scope
from dual_coordination import GPUS,identity
from cell_metadata import completed_metadata
from runtime_dispatch import cost_table,choose_next
import verify_cfgen_postfit_cap_failure as cap_failure
CAP_FAILURE=cap_failure.ATTEMPT
spec=importlib.util.spec_from_file_location('unchanged_balanced_native_execution',ROOT/'analysis/run_benchmark097_balanced.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,q):
 p=Path(p);t=p.with_suffix('.partial');t.write_text(json.dumps(q,indent=2,allow_nan=False)+'\n');t.replace(p)
def active_timing(h,*,clock=time.time,get_identity=identity,read_json=read,grace_seconds=30):
 """Defer a peer choice only while its exact newly launched guard publishes accounting."""
 auth=h.get('auth') or read_json(ROOT/'artifacts/allocations'/h['attempt']/'authorization.json')
 ledger=Path(auth['ledger_path'])
 try:state=read_json(ledger)
 except FileNotFoundError as exc:
  launched=h.get('admission_requested_unix');expected=h['identity']
  if exc.filename!=str(ledger) or type(launched) not in (int,float) or not 0<=clock()-launched<grace_seconds or get_identity(expected['pid'])!=expected:
   raise RuntimeError('Active guard accounting ledger absent outside exact bounded startup; retain handle without retry') from exc
  return None
 assert len(state['jobs'])==1 and state['jobs'][0]['guard_identity']==h['identity'],'Active timing ledger owner differs from exact guard'
 job=state['jobs'][0]
 return dict(cell=h['cell'],elapsed_seconds=max(0.,clock()-job['started_unix']))
def transient_proc_exit(exc):
 """Only an exited numeric process observation can reset capacity stability."""
 return isinstance(exc,FileNotFoundError) and exc.errno==errno.ENOENT and isinstance(exc.filename,str) and re.fullmatch(r'/proc/[1-9][0-9]*/(?:stat|status|cmdline|exe)',exc.filename) is not None
def stability_ready(cache,gpu,now,enough,seconds):
 if not enough:cache.pop(gpu,None);return False
 cache.setdefault(gpu,now);return now-cache[gpu]>=seconds
class ObservedOwner:
 """Observe a retained guard in its own session; never signal or restart it."""
 def __init__(self,expected,ledger,settlement,*,get_identity=identity,clock=time.monotonic,sleep=time.sleep):
  self.expected=expected;self.ledger=Path(ledger);self.settlement=Path(settlement);self.get_identity=get_identity;self.clock=clock;self.sleep=sleep;self.missing_since=None
 def poll(self):
  if self.get_identity(self.expected['pid'])==self.expected:self.missing_since=None;return None
  if self.ledger.exists() and self.settlement.exists():
   j=read(self.ledger)['jobs'][0];s=read(self.settlement)
   if j['status'] in ('completed','failed') and j.get('reservation_released') and s['uncertain'] is None and s['settlement']['status']=='settled':return j['exit_code']
  if self.missing_since is None:self.missing_since=self.clock()
  if self.clock()-self.missing_since>=30:raise RuntimeError('Retained guard absent without clean terminal settlement; preserve exact handles, never restart')
  return None
 def wait(self,timeout=30):
  end=self.clock()+timeout
  while self.clock()<end:
   code=self.poll()
   if code is not None:return code
   self.sleep(.1)
  raise TimeoutError('Retained guard observation timed out; native model state is not inferred terminal')
def choose_available(pending,active,gpu,costs,eligible,*,bootstrap_scvi=False):
 assert set(eligible)<=set(GPUS) and gpu not in active
 if not pending or gpu not in eligible:return None,dict(reason='no_pending_or_ineligible_card')
 if bootstrap_scvi and gpu==GPUS[0]:
  selected=next((v for v in pending if v['method']=='scvi'),None)
  if selected is not None:return selected,dict(reason='one_short_untouched_GPU0_native_scVI_first')
 if len(eligible)==1:
  value=max(pending,key=lambda v:costs[v['method']][GPUS.index(gpu)])
  return value,dict(reason='only_eligible_card_uses_pending_work',gpu=gpu)
 return choose_next(pending,active,gpu,costs)
DIRTY_ADMISSION='Commit completed research outputs before new admission'
def admit_when_clean(start,clean,untouched):
 if not untouched():raise RuntimeError('Original scientific key already prepared or consumed')
 if not clean():return None
 try:return start()
 except AssertionError as exc:
  if str(exc)==DIRTY_ADMISSION and untouched():return None
  raise
def drive(pending,rows,active,*,key,start,ready,finish,select,observe,sleep,now,expiry):
 pending=list(pending);rows=list(rows);active=dict(active);seen={key(r['cell']) for r in rows}|{key(h['cell']) for h in active.values()};assert len(seen)==len(rows)+len(active) and not seen.intersection(key(v) for v in pending);errors=[];unsettled=[]
 while pending or active:
  if pending and not errors and now()>=expiry-150:errors.append(dict(error='Authorized training admission window ended',pending=list(pending)))
  if not errors:
   for gpu in reversed(GPUS):
    if gpu in active:continue
    try:v=select(pending,active,gpu)
    except BaseException as exc:errors.append(dict(stage='choice',error=repr(exc),filename=getattr(exc,'filename',None),traceback=traceback.format_exc()));break
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

def choose_with_exclusions(pending,active,gpu,costs,eligible,exclusions=()):
 """Resource-only choice; every excluded scientific cell remains pending."""
 assert gpu not in active and set(eligible)<=set(GPUS)
 exclusions=set(exclusions)
 if not exclusions:return choose_available(pending,active,gpu,costs,eligible)
 assert exclusions<={(g,m) for g in GPUS for m in costs}
 if not pending or gpu not in eligible:return None,dict(reason='no_pending_or_ineligible_card')
 legal=[v for v in pending if (gpu,v['method']) not in exclusions]
 if not legal:return None,dict(reason='no_pending_method_eligible_on_card')
 if len(eligible)==1:return max(legal,key=lambda v:costs[v['method']][GPUS.index(gpu)]),dict(reason='only_eligible_card_uses_permitted_pending_work')
 methods=sorted(costs);counts={m:sum(v['method']==m for v in pending) for m in methods};initial=[]
 for g in GPUS:
  h=active.get(g);initial.append(0. if h is None else max(0.,costs[h['cell']['method']][GPUS.index(g)]-h['elapsed_seconds']))
 candidates=[]
 for n0 in product(*(range(counts[m]+1) for m in methods)):
  n1=tuple(counts[m]-n for m,n in zip(methods,n0))
  if any(n and (g,m) in exclusions for g,ns in zip(GPUS,(n0,n1)) for m,n in zip(methods,ns)):continue
  loads=[initial[i]+sum(n*costs[m][i] for m,n in zip(methods,ns)) for i,ns in enumerate((n0,n1))]
  candidates.append((max(loads),sum(loads),n0,n1,loads))
 assert candidates,'No legal two-card allocation; preserve all pending cells'
 _,_,n0,n1,loads=min(candidates);assigned=(n0,n1)[GPUS.index(gpu)]
 choices=[v for v in legal if assigned[methods.index(v['method'])]]
 if not choices:return None,dict(reason='no_remaining_count_assigned_to_card')
 return max(choices,key=lambda v:costs[v['method']][GPUS.index(gpu)]),dict(reason='declared_method_device_exclusions_respected',counts_per_card={m:[n0[i],n1[i]] for i,m in enumerate(methods)},predicted_card_seconds=loads,quality_values_read=False)
def verify_known_failure(attempt,path,auth,job,settlement,plan):
 assert attempt in plan['known_failed_attempts'] and sha(path)==plan['known_failed_attempts'][attempt]
 if attempt!=CAP_FAILURE:
  assert attempt in {'CAL_GPU0_V1_external097_f2_cfgen_s1','CAL_GPU0_V1_external097_f2_pg_s1'}
  assert job['status']=='failed' and 'Protected/registered GPU growth no longer fits' in job['error'];return
 out=Path(auth['output_dir'])
 absent=all(identity(job[k]['pid'])!=job[k] for k in ('guard_identity','child_identity'))
 lease_absent=job['lease_token'] not in read(Path(auth['registry_path'])/'registry.json')['jobs']
 return cap_failure.classify(auth=auth,auth_sha256=sha(ROOT/'artifacts/allocations'/attempt/'authorization.json'),job=job,settlement=settlement,result=read(path),result_sha256=sha(path),expected_result_sha256=plan['known_failed_attempts'][attempt],fit=read(out/'fit/fit_receipt.json'),restore=read(out/'native_restore.json'),processes_absent=absent,lease_absent=lease_absent)
def successful_costs():
 rows=[];seen_failures=set();known=read(PROTOCOL)['known_failed_attempts']
 for p in sorted((scope.STUDY/'results').glob('*.json')):
  a=read(ROOT/'artifacts/allocations'/p.stem/'authorization.json');ledger=Path(a['ledger_path']);j=read(ledger)['jobs'][0];s=read(ledger.with_name(p.stem+'-settlement.json'));assert j['reservation_released'] and s['uncertain'] is None and s['settlement']['status']=='settled'
  if p.stem in known:
   verify_known_failure(p.stem,p,a,j,s,read(PROTOCOL));seen_failures.add(p.stem);continue
  assert j['status']=='completed' and j['exit_code']==0
  rows.append(dict(method=a['cell']['method'],gpu_uuid=a['gpu_uuid'],seconds=s['settlement']['charged_GPU_seconds'],checked_unix=s['checked_unix'],attempt=p.stem))
 assert seen_failures==set(known),'Every declared failure needs its actual closed result; forecasts cannot enable V6'
 return rows

def main():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not old.command(['git','status','--porcelain'])
 plan=read(PROTOCOL);assert plan['known_failed_attempt']==FAILED and plan['continue_only_original_never_started_scientific_keys']
 assert set(plan['known_failed_attempts'])=={FAILED,'CAL_GPU0_V1_external097_f2_pg_s1',CAP_FAILURE},'Only an actual third retained cap failure enables V6'
 assert plan['exclude_CFGen_GPU0_after_verified_cap_failure'] is True
 successful_costs()
 proof=read(PROOF);assert proof['status']=='UNTOUCHED_CELL_CONTINUATION_CPU_CHECK_PASS' and proof['source_SHA256']==sha(__file__) and proof['protocol_SHA256']==sha(PROTOCOL)
 for path,pin in proof['supporting_source_SHA256'].items():assert sha(path)==pin,path
 inherited=read(scope.RUNTIME_PROOF)
 for p,pin in inherited['source_SHA256'].items():assert sha(p)==pin,p
 assert read(scope.PLAN)==scope.ENABLED and sha(ROOT/'analysis/run_benchmark097_balanced.py')==plan['original_driver_source_SHA256'];scope.protocol()
 handoff=read(plan['CPU_handoff_path']);assert handoff['status']=='CPU_ONLY_SCHEDULER_HANDOFF_ALL_NATIVE_OUTCOMES_DRAINED'
 for old_cpu in handoff['stopped_CPU_observers']:
  ident=old_cpu['identity'];assert identity(ident['pid'])!=ident
 original=read(OLD/'identity.json');cells=old.matrix();rows=[];active={}
 for v in cells:
  p=scope.STUDY/'results'/(b.attempt_key(v)+'.json')
  if p.exists():rows.append(completed_metadata(v,require_full=False))
 assert {r['attempt']:r['source_result_SHA256'] for r in rows}==handoff['retained_result_SHA256']
 for h in handoff['active_native_handles']:
  assert h['attempt']==b.attempt_key(h['cell']) and not (scope.STUDY/'results'/(h['attempt']+'.json')).exists()
  a=read(h['authorization_path']);assert sha(h['authorization_path'])==h['authorization_SHA256'] and a['cell']==h['cell'] and a['gpu_uuid']==h['gpu']
  ledger=Path(a['ledger_path']);settlement=ledger.with_name(h['attempt']+'-settlement.json')
  owner=ObservedOwner(h['identity'],ledger,settlement)
  active[h['gpu']]=dict(h,owner=owner,auth=a,exe=b.runtime_path(b.NATIVE_LAYOUT,h['cell']),ledger_started_unix=read(ledger)['jobs'][0]['started_unix'],settlement=settlement)
 assert {r['attempt']:r['source_result_SHA256'] for r in rows if not r['complete']}==plan['known_failed_attempts']
 seen={b.attempt_key(r['cell']) for r in rows}|{h['attempt'] for h in active.values()};pending=[v for v in cells if b.attempt_key(v) not in seen]
 STATE.mkdir(exist_ok=False)
 with (STATE/'driver.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);write(STATE/'identity.json',dict(**identity(os.getpid()),source_SHA256=sha(__file__),protocol_SHA256=sha(PROTOCOL),inherited_native_qualification_SHA256=sha(scope.RUNTIME_PROOF),activation=original['activation'],new_model_retries=False))
  stable={};polls={};bootstrap=[False]
  def capacity_stable(gpu):
   import resource_profile as profile
   if gpu in polls and time.time()-polls[gpu][0]<30:return polls[gpu][1]
   try:
    auth,_=__import__('prepare').build_authorization(fold=2,method='scvi',seed=1,gpu=gpu,check_only=True);_,admission,_=b.modules(auth);observation=admission.snapshot();forecast=profile.capacity(observation,admission,gpu);profile.free_floor(gpu)
    enough=forecast['GPU_total_bytes']-forecast['GPU_required_bytes']>=plan['scheduling_only_extra_GPU_capacity_bytes']
   except FileNotFoundError as exc:
    if not transient_proc_exit(exc):raise
    enough=False
   except AssertionError as exc:
    if not any(value in str(exc) for value in ('growth forecasts do not fit','ALZ early stop:')):raise
    enough=False
   result=stability_ready(stable,gpu,time.time(),enough,plan['scheduling_only_stability_seconds']);polls[gpu]=(time.time(),result);write(STATE/('capacity-'+str(GPUS.index(gpu))+'.json'),dict(checked_unix=time.time(),capacity_fits=enough,stable_since=stable.get(gpu),ready=result,extra_GPU_scheduling_bytes=plan['scheduling_only_extra_GPU_capacity_bytes']));return result
  def select(pending,active,gpu):
   if not pending or not capacity_stable(gpu):return None
   costs,evidence=cost_table(successful_costs());timed={}
   for g,h in active.items():
    timing=active_timing(h)
    if timing is None:return None
    timed[g]=timing
   eligible=set(active)|{g for g in GPUS if g not in active and capacity_stable(g)}
   choice,forecast=choose_with_exclusions(pending,timed,gpu,costs,eligible,exclusions={(GPUS[0],'cfgen')});write(STATE/'latest_cost_choice.json',dict(selected=choice,gpu=gpu,forecast=forecast,quality_values_read=False));return choice
  def start(v,gpu,n):
   attempt=b.attempt_key(v)
   def untouched():
    if any(p.exists() for p in (ROOT/'artifacts/allocations'/attempt,scope.STUDY/'results'/(attempt+'.json'),ROOT/'runs'/attempt,b.ledger_path(b.NATIVE_LAYOUT,v))):return False
    budget=read(b.PROJECT/'project_gpu_budget.json');key=b.allocation_key(v)
    if any(r['allocation_id']==key for r in budget.get('allocation_history',[])+budget.get('open_reservations',[])):return False
    return key not in budget.get('execution_controller',{}).get('calibration_parallel_backends',{})
   h=admit_when_clean(lambda:old.start_cell(v,gpu,n),lambda:not old.command(['git','status','--porcelain']),untouched)
   if h is not None:
    h['admission_requested_unix']=time.time()
    if gpu==GPUS[0]:bootstrap[0]=False
    write(STATE/(h['attempt']+'-start.json'),dict(cell=v,gpu=gpu,identity=h['identity'],source_SHA256=sha(__file__),from_update_zero=True));print(json.dumps(dict(event='native_guard_launched',attempt=h['attempt'],gpu=gpu,not_yet_a_native_progress_claim=True)),flush=True)
   return h
  def ready(h):return old.ready_cell(h)
  def finish(h):return old.finish_cell(h)
  def observe(rows,pending,active,errors,unsettled):
   write(STATE/'progress.json',dict(status='UNTOUCHED_ORIGINAL_CELLS_CONTINUE_KNOWN_INFRA_FAILURE_RETAINED' if not errors else 'STOP_NEW_ADMISSIONS_DRAIN_HEALTHY_HANDLES',resolved_original_cells=len(rows),complete_original_cells=sum(r['complete'] for r in rows),known_infra_failed_attempts=plan['known_failed_attempts'],pending_original_cells=len(pending),active=[{k:h[k] for k in ['cell','gpu','identity','attempt']} for h in active.values()],errors=errors,unsettled=unsettled,full_goal_complete=False,SOTA_claim=False))
  try:
   out=drive(pending,rows,active,key=b.attempt_key,start=start,ready=ready,finish=finish,select=select,observe=observe,sleep=time.sleep,now=time.time,expiry=read(scope.WINDOW)['expires_unix'])
   if out['errors'] or out['unsettled'] or len(out['rows'])!=45:write(STATE/'failure.json',dict(status='ADDITIONAL_ISSUE_RETAIN_ORIGINAL_HANDLES',**out));raise RuntimeError('Original attempt continuation incomplete; no model restart')
   assert sum(r['complete'] for r in out['rows'])==42
   record=dict(status='ALL45_ORIGINAL_ATTEMPTS_CLOSED_42_COMPLETE_THREE_INFRA_FAILURES_FRESH_REPLACEMENTS_REMAIN',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),original_scientific_cells=45,successful_original_models=42,failed_original_attempts=sorted(plan['known_failed_attempts']),result_SHA256={r['attempt']:r['source_result_SHA256'] for r in out['rows']},actual_original_GPU_seconds=sum(r['charged_GPU_seconds'] for r in out['rows']),all_new_models_from_zero=True,failed_weights_not_resumed=True,fresh_infrastructure_replacement_required=True,original57_and53_reports_pending=True,scalar6_fresh_fits_pending=True,full_goal_complete=False,SOTA_claim=False)
   p=ROOT/'analysis/benchmark097_original_attempts_postcapacity_closed.json';assert not p.exists();write(p,record);old.commit([p],'Retain all45 original outcomes and allthree infrastructure failures before fresh replacements');write(STATE/'completed.json',record)
  except BaseException as exc:
   if not (STATE/'failure.json').exists():write(STATE/'failure.json',dict(error=repr(exc),automatic_retry=False,existing_handles_retained=True))
   raise
if __name__=='__main__':main()
