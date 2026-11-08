"""Qualified CPU-dispatcher handoff; admitted native models finish unchanged."""
import argparse,datetime,fcntl,importlib.util,json,os,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path('/ml2/alz');RUNTIME=ROOT/'benchmark097_balanced_runtime';OLD=ROOT/'benchmark097_dual_runtime';OLDSTATE=ROOT/'artifacts/benchmark097-dual-fullmatrix';STATE=ROOT/'artifacts/benchmark097-balanced-handoff'
sys.path.insert(0,str(RUNTIME))
import binding as b,preset_scope as s,resource_profile as profile
from dual_coordination import identity,GPUS
from handoff_support import matching_processes,signal_dispatcher_only,classify
from cell_metadata import completed_metadata
ENV=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
SCRIPTS=[OLD/n for n in ('prepare.py','launch_core_gpu0.py','worker_gpu0.py','finish.py')]
def read(p):return json.loads(Path(p).read_bytes())
def write(p,q):
 p=Path(p);tmp=p.with_suffix('.partial');tmp.write_text(json.dumps(q,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def command(args,timeout=120):
 p=subprocess.run(list(map(str,args)),cwd=ROOT,env=ENV,capture_output=True,text=True,timeout=timeout)
 if p.returncode:raise RuntimeError(p.stderr[-3000:])
 return p.stdout
def commit(paths,message):command(['git','add','--',*map(str,paths)]);command(['git','commit','--only','-m',message,'--',*map(str,paths)])
def cells():return [s.cell(f,m,seed) for seed in range(5) for f in range(3) for m in ('pg','scvi','cfgen')]
def old_identity():
 q=read(OLDSTATE/'identity.json');expected={k:q[k] for k in ('pid','birth','boot')}
 assert b.sha(ROOT/'analysis/run_benchmark097_dual.py')==q['source_SHA256']
 assert q['protocol_SHA256']==b.sha(s.PROTOCOL)
 return q,expected

def verify_preparation():
 q=read(s.RUNTIME_PROOF);assert q['status']=='EXTERNAL097_SAME45_BALANCED_RESOURCE_SCOPE_CPU_PASS' and q['native_cell_device_bindings']==90 and q['actual_CPU_dispatcher_signal_isolation_test']
 for path,pin in q['source_SHA256'].items():assert b.sha(path)==pin,path
 assert not read(s.PLAN)['enabled'];s.protocol();profile.validate_proposal()
 return q

def stop_original(original,expected):
 controller=b.load_pinned(b.CONTROLLER,b.CONTROLLER_SHA,'balanced_handoff_unchanged_controller');control=controller.Controller(b.PROJECT)
 deadline=min(time.time()+120,read(s.WINDOW)['expires_unix']-150)
 while time.time()<deadline:
  assert identity(expected['pid'])==expected,'Original CPU dispatcher ended: inspect authoritative completion before proceeding'
  try:
   with control.transaction() as state:
    progress=read(OLDSTATE/'progress.json');assert not progress['errors'] and not progress['unsettled']
    active={h['identity']['pid']:h for h in progress['active']};rows=state['open_reservations']
    if len(rows)!=len(active) or not rows:raise ValueError('Wait for a stable original active set')
    for row in rows:
     owner=row['owner'];assert row['status']=='reserved' and owner['pid'] in active and owner==active[owner['pid']]['identity'] and identity(owner['pid'])==owner
     h=active[owner['pid']];a=read(ROOT/'artifacts/allocations'/h['attempt']/'authorization.json');assert a['operation']=='external097_benchmark_cell'
     job=read(a['ledger_path'])['jobs'][0];assert job['status']=='active' and job['guard_identity']==owner
    write(STATE/'stop_request.json',dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),original_driver=original,active_original_guards=list(active.values()),only_CPU_dispatcher_to_be_signalled=True,existing_native_caps_and_authorizations_unchanged=True))
    receipt=signal_dispatcher_only(expected,ROOT/'analysis/run_benchmark097_dual.py')
    write(STATE/'stop_signal_receipt.json',receipt);return
  except ValueError:time.sleep(.2)
 raise TimeoutError('No verified quiescent CPU dispatch boundary observed; no model was signalled')

def inventory():
 budget=read(b.PROJECT/'project_gpu_budget.json');reservations={r['allocation_id']:r for r in budget['open_reservations']};history={r['allocation_id']:r for r in budget['allocation_history']};items=[]
 for v in cells():
  key=b.attempt_key(v);allocation=b.allocation_key(v);folder=ROOT/'artifacts/allocations'/key;authpath=folder/'authorization.json';result=s.STUDY/'results'/(key+'.json');ledger=b.ledger_path(b.NATIVE_LAYOUT,v)
  auth=None
  if authpath.exists():
   try:auth=read(authpath)
   except json.JSONDecodeError:pass
  job=read(ledger)['jobs'][0] if ledger.exists() else None
  settlement_path=ledger.with_name(key+'-settlement.json');settlement=read(settlement_path) if settlement_path.exists() else None
  matches=matching_processes(SCRIPTS,authpath)+matching_processes([OLD/'finish.py'],key)
  guard_live=bool(job and identity(job['guard_identity']['pid'])=={k:job['guard_identity'][k] for k in ('pid','birth','boot')})
  status=classify(has_result=result.exists(),has_authorization=folder.exists(),has_ledger=ledger.exists(),job_status=job['status'] if job else None,settled=bool(settlement and settlement.get('settlement',{}).get('status')=='settled'),uncertain=bool(settlement and settlement.get('uncertain')),guard_live=guard_live,matching_process_live=bool(matches),reservation=allocation in reservations,history_consumed=allocation in history)
  items.append(dict(cell=v,attempt=key,allocation_id=allocation,status=status,auth=auth,authorization_path=str(authpath),folder=str(folder),ledger_path=str(ledger),result_path=str(result),matching_processes=matches))
 return items,budget

def finish_old(item):
 v=item['cell'];auth=item['auth'];assert auth and auth['cell']==v and auth['operation']=='external097_benchmark_cell'
 # Original auditor runs before the original one-file activation is reverted.
 exe=b.runtime_path(b.NATIVE_LAYOUT,v)
 output=command([exe,'-I','-S','-B',OLD/'finish.py','--attempt',item['attempt']],timeout=600)
 (STATE/(item['attempt']+'-finish.stdout')).write_text(output)
 metadata=completed_metadata(v);commit([Path(item['result_path'])],f'Retain unchanged admitted original {item["attempt"]} after CPU dispatcher handoff')
 return metadata

def verify_existing_result(item):
 # Audit metadata/source closure; saved quality fields are never used for a
 # resource choice. No target arrays are opened by this handoff process.
 q=read(item['result_path']);assert q['complete'] and q['status']=='EXTERNAL097_NATIVE_CHECKPOINT_COUNTS_SCORES_CLEAN_SETTLEMENT_AUDIT_PASS'
 assert q['all_new_training_from_scratch'] and q['initial_completed_updates']==0
 row=completed_metadata(item['cell']);assert q['cell']==row['cell'] and q['charged_GPU_seconds']==row['charged_GPU_seconds']
 return row

def reconcile(original):
 limit=read(s.WINDOW)['expires_unix']+7200;launched={};archived=[]
 while time.time()<limit:
  for child in launched.values():
   if child.poll() is not None:child.wait(timeout=5)
  items,budget=inventory();waiting=[];unsafe=[]
  for item in items:
   status=item['status']
   if status=='FINISH_ORIGINAL_READONLY_AUDIT':finish_old(item)
   elif status.startswith('WAIT_'):waiting.append(item)
   elif status.startswith('RECONCILE_') or status=='RETAIN_INCOMPLETE_NO_MODEL_RETRY':unsafe.append(dict(cell=item['cell'],status=status))
  if unsafe:raise RuntimeError('Retain original uncertain/incomplete scientific cells without retry: '+json.dumps(unsafe))
  operators=matching_processes(SCRIPTS)
  if not waiting and operators:
   write(STATE/'progress.json',dict(status='WAITING_FOR_ORIGINAL_CPU_HELPERS',processes=operators,native_models_signalled=False));time.sleep(2);continue
  if waiting:
   write(STATE/'progress.json',dict(status='DRAINING_UNCHANGED_ORIGINAL_NATIVE_GUARDS',waiting=[{k:item[k] for k in ('cell','status','matching_processes')} for item in waiting],native_models_signalled=False));time.sleep(15);continue
  # No guard, worker, old CLI audit, reservation or lease may be hidden by a
  # missing state file. This is the first point at which zero-work preparations
  # can be reconciled or the old activation can be reverted.
  assert not budget['open_reservations'] and not matching_processes(SCRIPTS)
  for index in (0,1):
   reg=ROOT/f'artifacts/gpu{index}-benchmark097-dual-20261107/live_registry/registry.json';assert not reg.exists() or not read(reg)['jobs']
  prepared=[item for item in items if item['status']=='PREPARED_UNSTARTED_ZERO_WORK_RECONCILIATION']
  if prepared:
   item=prepared[0];auth=item['auth'];key=item['attempt']
   if auth and auth['expires_unix']>time.time()+150:
    # This is the very first launch of an already prepared original scientific
    # key, not a refit or a new authorization. Keep its original device/cap.
    assert key not in launched and not Path(auth['ledger_path']).exists() and not Path(auth['output_dir']).exists()
    probe_code='import sys;sys.path.insert(0,sys.argv[1]);import binding as b,resource_profile as p;a,_=b.verify(sys.argv[2]);_,ad,_=b.modules(a);p.capacity(ad.snapshot(),ad,a["gpu_uuid"]);p.free_floor(a["gpu_uuid"])'
    probe=subprocess.run([str(b.runtime_path(b.NATIVE_LAYOUT,item['cell'])),'-I','-S','-B','-c',probe_code,str(OLD),item['authorization_path']],cwd=ROOT,env=ENV,capture_output=True,text=True,timeout=120)
    if probe.returncode:
     if any(text in probe.stderr for text in ('growth forecasts do not fit','ALZ early stop:')):
      write(STATE/'progress.json',dict(status='WAITING_FOR_SHARED_CAPACITY_FIRST_PREPARED_NATIVE_LAUNCH',cell=item['cell'],new_GPU_allocation=False));time.sleep(15);continue
     raise RuntimeError(probe.stderr[-3000:])
    with (Path(item['folder'])/'handoff-owner.stdout').open('xb') as out,(Path(item['folder'])/'handoff-owner.stderr').open('xb') as err:
     p=subprocess.Popen([str(b.runtime_path(b.NATIVE_LAYOUT,item['cell'])),'-I','-S','-B',str(OLD/'launch_core_gpu0.py'),'--authorization',item['authorization_path']],cwd=ROOT,env=ENV,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
    launched[key]=p;write(STATE/(key+'-first-launch.json'),dict(**identity(p.pid),first_native_launch_of_prepared_key=True,original_authorization_preserved=True));time.sleep(2);continue
   # An expired or incomplete preparation with no reservation history, no
   # ledger, no process and empty registries has provably never released a
   # scientific worker through require_live. Preserve it before first admission.
   assert item['allocation_id'] not in {r['allocation_id'] for r in budget['allocation_history']}
   src=Path(item['folder']);dest=STATE/'unstarted_archive'/key;dest.parent.mkdir(exist_ok=True)
   assert not dest.exists();pins={str(p.relative_to(src)):b.sha(p) for p in src.rglob('*') if p.is_file()};src.rename(dest)
   archived.append(dict(attempt=key,archived_path=str(dest),files_SHA256=pins,proof='No ledger, reservation/history, matching process or registry lease; original CPU dispatcher gone before archive; zero native worker starts.'))
   continue
  for child in launched.values():
   assert child.poll() is not None;child.wait(timeout=5)
  items,budget=inventory();assert all(item['status'] in ('VERIFY_RETAINED_RESULT','UNTOUCHED_CELL') for item in items)
  retained=[verify_existing_result(item) for item in items if item['status']=='VERIFY_RETAINED_RESULT']
  # Commit any result whose original parent was interrupted between its audit
  # and Git commit. Never stage unrelated source or overwrite scientific output.
  dirty=command(['git','status','--porcelain']).splitlines();allowed={str(Path(item['result_path']).relative_to(ROOT)) for item in items if item['status']=='VERIFY_RETAINED_RESULT'}
  assert all(line[3:] in allowed for line in dirty),'Unexpected dirty source: reconcile before changing activation'
  if dirty:commit([ROOT/line[3:] for line in dirty],'Retain already audited original outputs after CPU-only dispatcher drain')
  command(['git','revert','--no-edit',original['activation']])
  record=dict(status='ORIGINAL_ADMITTED_CELLS_CLOSED_AND_RESOURCE_HANDOFF_READY',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),original_driver_identity=original,original_CPU_signal_receipt=read(STATE/'stop_signal_receipt.json'),preserved_result_SHA256={row['attempt']:row['source_result_SHA256'] for row in retained},preserved_result_metadata=retained,untouched_cells=[item['cell'] for item in items if item['status']=='UNTOUCHED_CELL'],archived_zero_work_preparations=archived,original_activation_reverted=True,original_activation_revert_commit=command(['git','rev-parse','HEAD']).strip(),all_original_guards_and_workers_absent=True,canonical_reservations_empty=True,original_physical_registries_empty=True,models_interrupted_or_restarted=False,target_arrays_loaded=False,quality_used_for_resource_choices=False,full_goal_complete=False,SOTA_claim=False)
  write(s.HANDOFF,record);commit([s.HANDOFF],'Preserve original admitted native fits and clean resource-only dispatcher handoff');write(STATE/'completed.json',dict(status=record['status'],record_path=str(s.HANDOFF),full_goal_complete=False));return record
 raise TimeoutError('Finite CPU handoff observation ended; original handles retained without signals or restarts')

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--drain-original',action='store_true');parser.add_argument('--resume',action='store_true');args=parser.parse_args()
 verify_preparation();original,expected=old_identity()
 if not args.drain_original:
  assert identity(expected['pid'])==expected
  print(json.dumps(dict(status='QUALIFIED_REPLACEMENT_SOURCE_CHECK_ONLY_ORIGINAL_CPU_STILL_LIVE',GPU_allocations=0,signals_sent=False)));return
 assert not command(['git','status','--porcelain'])
 if args.resume:assert STATE.exists() and (STATE/'stop_signal_receipt.json').exists()
 else:STATE.mkdir(exist_ok=False)
 with (STATE/'handoff.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if args.resume:
   history=STATE/'CPU_observer_history';history.mkdir(exist_ok=True)
   for name in ('failure.json','identity.json'):
    old=STATE/name
    if old.exists():
     saved=history/(str(time.time_ns())+'-'+name);old.rename(saved)
  write(STATE/'identity.json',dict(**identity(os.getpid()),source_SHA256=b.sha(__file__),original_driver=original,CPU_observation_resumed=args.resume,existing_models_restarted=False))
  try:
   if not args.resume:stop_original(original,expected)
   while identity(expected['pid'])==expected:
    assert time.time()<read(s.WINDOW)['expires_unix']+7200,'Original CPU handle retained after finite observation expiry';time.sleep(2)
   reconcile(original)
  except BaseException as exc:
   write(STATE/'failure.json',dict(status='RESOURCE_HANDOFF_RECONCILIATION_REQUIRED',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),error=repr(exc),native_jobs_signalled=False,automatic_model_retry=False,full_goal_complete=False));raise
if __name__=='__main__':main()
