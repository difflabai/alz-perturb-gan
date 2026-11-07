"""Finite CPU closure audit of the four fresh released-objective baselines.

No benchmark freeze, model training, GPU allocation, or automatic continuation.
The original exact live queue is observed without replacement or signaling.
"""
import datetime,fcntl,hashlib,importlib.util,json,os,subprocess,time
from pathlib import Path
ROOT=Path('/ml2/alz');STATE=ROOT/'artifacts/faithful097-audit-boundary'
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
 assert p.returncode==0,name+' failed; retain evidence, no automatic retry'
def report(kind,event,summary,details):
 spec=importlib.util.spec_from_file_location('existing_reporter',ROOT/'analysis/external_benchmark_driver.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.report(kind,event,summary,json.dumps(details))
def main():
 STATE.mkdir(exist_ok=True)
 with (STATE/'lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (STATE/'identity.json').exists(),'One-use audit observer consumed; inspect instead of replacing'
  parent=read(ROOT/'artifacts/faithful097-paired-search/identity.json');assert alive(parent)
  paths=[Path(__file__),ROOT/'analysis/audit_faithful097.py',ROOT/'analysis/run_faithful097.py',ROOT/'analysis/faithful097/protocol.json',
    ROOT/'analysis/released097_selection.py',ROOT/'analysis/faithful097/native_trainer_parity_CPU.json',
    *list((ROOT/'faithful097_runtime').glob('*.py')),*list((ROOT/'analysis/faithful097/model').glob('*.py'))]
  pins={str(p):sha(p) for p in paths};expiry=min(time.time()+26*3600,read(ROOT/'analysis/research_window_20261104_five_days.json')['expires_unix']-600)
  write(STATE/'identity.json',dict(pid=os.getpid(),birth=int(Path('/proc',str(os.getpid()),'stat').read_text().rsplit(')',1)[1].split()[19]),boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),parent=parent,
    finite_expiry_unix=expiry,source_SHA256=pins,activation_permitted=False,GPU_allocation_permitted=False,benchmark_freeze_permitted=False))
  try:
   while alive(parent):
    assert time.time()<expiry,'Observation allowance expired; inspect exact original job without replacement'
    time.sleep(15)
   for path,pin in pins.items():assert sha(path)==pin,'Pinned observer/audit source changed'
   assert not (ROOT/'artifacts/faithful097-paired-search/failure.json').exists()
   progress=read(ROOT/'artifacts/faithful097-paired-search/progress.json');assert progress['activation_reverted'] and progress['goal_still_active']
   assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True)
   audit=ROOT/'analysis/faithful097_final_independent_audit_20261107.json'
   command('audit-baselines',[PG,'-I','-S','-B',ROOT/'analysis/audit_faithful097.py','--output',audit],timeout=900)
   q=read(audit);assert q['status']=='INDEPENDENT_RELEASED097_FRESH_BASELINE4_AUDIT_PASS'
   summary=q['summary'];finite=[c for c in summary['candidates'] if c['all_seeds_finite']]
   beats_baseline=any(c['mean']['cell_energy_distance']<c['mean_matched_resampling']['cell_energy_distance'] for c in finite)
   result=dict(status='RELEASED_OBJECTIVE_BASELINES_AUDITED_NEXT_SCIENTIFIC_DECISION_REQUIRED',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    complete_audit_path=str(audit),complete_audit_SHA256=sha(audit),summary=summary,any_two_seed_mean_beats_matched_resampling=beats_baseline,
    pending_decision='Review full20k means, seed variation and spread; then freeze calibration refinement or fresh seed2..4 confirmation. No selection is automatically frozen by this observer.',
    fresh_confirmation_needed=True,benchmark_candidate_frozen=False,benchmark_activated=False,GPU_allocation_created=False,SOTA_claim=False,full_goal_complete=False)
   decision=ROOT/'analysis/faithful097_next_decision_20261107.json';assert not decision.exists();write(decision,result)
   gp=ROOT/'analysis/current_research_goal.json';g=read(gp);g.update(study_status=result['status'],faithful_restart_final_audit_path=str(audit),faithful_restart_next_decision_path=str(decision),external_benchmark_candidate_freeze_deferred=True,native_goal_status='active',full_goal_complete=False);write(gp,g)
   command('git-add',['git','add','--',audit,decision,gp]);command('git-commit',['git','commit','--only','-m','Audit four fresh released-objective endpoints and retain full calibration tradeoffs','--',audit,decision,gp])
   write(STATE/'completed.json',result)
   report('completed','faithful097-all-four-fresh-baselines-independent-audit-20261107','Four fresh released-objective20k baseline attempts independently audited with controls, negative outcomes and actual costs. Further calibration/confirmation decision and external matrix remain.',result)
  except BaseException as exc:
   failure=dict(status='RELEASED097_BASELINE_AUDIT_OBSERVER_STOPPED',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),error=repr(exc),automatic_retry=False,activation=False,GPU_allocation_created=False)
   write(STATE/'failure.json',failure);report('failed','faithful097-audit-observer-stop-20261107','CPU closure audit stopped; evidence retained, no new activation or allocation.',failure);raise
if __name__=='__main__':main()
