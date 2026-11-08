"""One-use exact-PID CPU handoff; detached admitted native guard is retained."""
import datetime,hashlib,json,os,signal,time
from pathlib import Path
ROOT=Path('/ml2/alz');BOOT=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
OUT=ROOT/'analysis/benchmark097_capacity_CPU_handoff_v4_20261108.json'
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def identity(pid):
 try:
  stat=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()
  return None if stat[0]=='Z' else dict(pid=pid,birth=int(stat[19]),boot=BOOT)
 except (FileNotFoundError,ProcessLookupError):return None
def main():
 assert not OUT.exists()
 p=read(ROOT/'artifacts/benchmark097-after-capacity-continuation-v2/progress.json')
 assert p['complete_original_cells']==15 and len(p['active'])==1 and len(p['errors'])==1 and not p['unsettled']
 assert p['errors'][0]['stage']=='admission' and p['errors'][0]['error']=="AssertionError('Commit completed research outputs before new admission')"
 h=p['active'][0];assert h['attempt']=='CAL_GPU0_V1_external097_f1_cfgen_s2'
 authpath=ROOT/'artifacts/allocations'/h['attempt']/'authorization.json';a=read(authpath);job=read(a['ledger_path'])['jobs'][0]
 lease=read(Path(a['registry_path'])/'registry.json')['jobs'][job['lease_token']]
 guard=job['guard_identity'];worker=job['child_identity']
 assert identity(guard['pid'])==guard and identity(worker['pid'])==worker and lease['owner']==guard and lease['worker']==worker and lease['status']=='running' and time.time()-lease['heartbeat_unix']<30
 assert os.getsid(guard['pid'])==guard['pid']
 specs=[('dispersion097-after-scalarcal097',1410636,27283404,'after_scalarcal097_dispersion.py'),('scalarcal097-after-recovered-science-v5',1020103,26586743,'after_benchmark097_scalarcal.py'),('benchmark097-infra1-pipeline',977389,26513482,'run_benchmark097_infra1_pipeline.py'),('benchmark097-after-capacity-continuation-v2',962924,26487633,'continue_benchmark097_after_capacity_v2.py')]
 subjects=[]
 for name,pid,birth,filename in specs:
  path=ROOT/'artifacts'/name;record=read(path/'identity.json');expected=dict(pid=pid,birth=birth,boot=BOOT)
  assert {k:record[k] for k in expected}==expected==identity(pid)
  source=ROOT/'analysis'/filename;proc=Path('/proc',str(pid));cwd=(proc/'cwd').resolve()
  args=(proc/'cmdline').read_bytes().split(b'\0')
  scripts=[(cwd/os.fsdecode(arg)).resolve() for arg in args if arg.endswith(b'.py')]
  assert source.resolve() in scripts and record['source_SHA256']==sha(source)
  assert pid not in (guard['pid'],worker['pid']) and os.getpgid(pid)!=os.getpgid(guard['pid'])
  archive=None if name.endswith('continuation-v2') else path.with_name(path.name+'-superseded-CPU-handoff-v4-20261108')
  if archive:assert not archive.exists()
  subjects.append(dict(state_path=str(path),archived_state_path=str(archive) if archive else None,identity=expected,source_path=str(source),source_SHA256=sha(source)))
 for name in ('scalarcal097_runtime','dispersion097_runtime','benchmark097_infra1_runtime'):assert not read(ROOT/name/'plan.json')['enabled']
 key='CAL_GPU0_V1_external097_f2_cfgen_s2'
 assert p['errors'][0]['cell']==dict(experiment='external097',fold=2,method='cfgen',seed=2)
 for path in (ROOT/'artifacts/allocations'/key,ROOT/'runs'/key,ROOT/'analysis/benchmark097/results'/(key+'.json')):assert not path.exists()
 proofpath=ROOT/'analysis/benchmark097_capacity_continuation_v4_CPU_check_20261108.json';proof=read(proofpath)
 assert proof['actual_tests']==22 and proof['source_SHA256']==sha(ROOT/'analysis/continue_benchmark097_after_capacity_v4.py')
 retained={path.stem:sha(path) for path in (ROOT/'analysis/benchmark097/results').glob('*.json')};assert len(retained)==17
 # Every precondition passes before any signal. Only exact own CPU PIDs follow.
 for subject in subjects:
  ident=subject['identity'];assert identity(ident['pid'])==ident
  os.kill(ident['pid'],signal.SIGTERM);end=time.monotonic()+8
  while identity(ident['pid'])==ident and time.monotonic()<end:time.sleep(.05)
  assert identity(ident['pid'])!=ident,'CPU observer did not stop'
  assert identity(guard['pid'])==guard and identity(worker['pid'])==worker,'Retain and inspect native handles'
  if subject['archived_state_path']:Path(subject['state_path']).rename(subject['archived_state_path'])
 lease=read(Path(a['registry_path'])/'registry.json')['jobs'][job['lease_token']]
 assert lease['status']=='running' and lease['owner']==guard and lease['worker']==worker
 result=dict(status='CPU_ONLY_SCHEDULER_HANDOFF_LIVE_NATIVE_MODEL_RETAINED',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),cause='Diagnostic source/commit briefly dirtied Git at the old next-admission check. The exact assertion occurred before prepare/backend/allocation/model operations. No extra scientific fit failed.',stopped_CPU_observers=subjects,active_native_handles=[dict(cell=h['cell'],gpu=h['gpu'],identity=guard,worker_identity=worker,attempt=h['attempt'],authorization_path=str(authpath),authorization_SHA256=sha(authpath))],retained_result_SHA256=retained,successful_models_retained=15,known_infrastructure_failed_models_retained=2,unstarted_admission_error_key=key,unstarted_original_key_not_consumed=True,native_models_never_signalled_or_restarted=True,native_reservation_and_lease_preserved=True,foreign_processes_signalled=False,qualification_path=str(proofpath),qualification_SHA256=sha(proofpath),handoff_source_SHA256=sha(__file__),full_goal_complete=False,SOTA_claim=False)
 OUT.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(dict(status=result['status'],stopped_CPU_PIDs=[s['identity']['pid'] for s in subjects],retained_guard=guard,retained_worker=worker,retained_closed_outcomes=len(retained))))
if __name__=='__main__':main()
