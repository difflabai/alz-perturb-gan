"""Resolve every prespecified fresh-seed trial and matched resampling control."""
import argparse,datetime,fcntl,importlib.util,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path('/ml2/alz');RUNTIME=ROOT/'confirmation097_runtime';STUDY=ROOT/'analysis/confirmation097';STATE=ROOT/'artifacts/confirmation097-matched-seeds'
PG=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark/current_base_20261029/runtime/bin/python')
sys.path.insert(0,str(RUNTIME));import preset_scope as scope,binding as b
ENV=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
def read(p):return json.loads(Path(p).read_bytes())
def write(p,q):
 p=Path(p);tmp=p.with_suffix('.partial');tmp.write_text(json.dumps(q,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def command(args,timeout=120):
 p=subprocess.run(list(map(str,args)),cwd=ROOT,env=ENV,capture_output=True,text=True,timeout=timeout)
 if p.returncode:raise RuntimeError(p.stderr[-3000:])
 return p.stdout
def commit(paths,message):
 command(['git','add','--',*map(str,paths)]);command(['git','commit','--only','-m',message,'--',*map(str,paths)])
def report(kind,event,summary,details):
 spec=importlib.util.spec_from_file_location('existing_reporter',ROOT/'analysis/external_benchmark_driver.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.report(kind,event,summary,json.dumps(details))
def fit(cid,seed,completed):
 value=scope.cell(cid,seed,'confirm');attempt=b.attempt_key(value);folder=ROOT/'artifacts/allocations'/attempt;result=STUDY/'results'/(attempt+'.json')
 assert not folder.exists() and not result.exists(),'One-use confirmation cell already consumed; inspect, do not retry'
 assert not command(['git','status','--porcelain'])
 prepare=[PG,'-I','-S','-B',RUNTIME/'prepare.py','--candidate',cid,'--seed',seed,'--phase','confirm']
 while True:
  assert time.time()<read(scope.WINDOW)['expires_unix']-150
  p=subprocess.run(list(map(str,[*prepare,'--check-only'])),cwd=ROOT,env=ENV,capture_output=True,text=True,timeout=120)
  if p.returncode==0:break
  if not any(x in p.stderr for x in ('Exact parallel GPU growth forecasts do not fit','Exact parallel host growth forecasts do not fit','ALZ early stop: actual GPU free below3.5GiB','ALZ early stop: host available below12GiB')):raise RuntimeError(p.stderr[-3000:])
  write(STATE/'progress.json',dict(status='WAITING_FOR_SHARED_CAPACITY_NO_NEW_ALLOCATION',candidate=cid,seed=seed,resolved_trials=completed));time.sleep(15)
 (STATE/(attempt+'-prepare.stdout')).write_text(command(prepare))
 dep=read(folder/'deployment.json');assert b.sha(dep['authorization_path'])==dep['authorization_SHA256'];auth=read(dep['authorization_path'])
 assert not Path(auth['ledger_path']).exists() and not Path(auth['output_dir']).exists()
 with (folder/'owner.stdout').open('xb') as out,(folder/'owner.stderr').open('xb') as err:
  owner=subprocess.Popen([str(PG),'-I','-S','-B',str(RUNTIME/'launch_core_gpu0.py'),'--authorization',dep['authorization_path']],cwd=ROOT,env=ENV,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
 write(folder/'launch_receipt.json',dict(pid=owner.pid,birth=int(Path('/proc',str(owner.pid),'stat').read_text().rsplit(')',1)[1].split()[19]),boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()))
 progress=dict(status='FRESH20K_MATCHED_INITIALIZATION_CONFIRMATION_RUNNING',candidate=cid,seed=seed,attempt=attempt,endpoint=20000,initial_completed_updates=0,from_scratch=True,resolved_trials=completed)
 write(STATE/'progress.json',progress);settlement=Path(auth['ledger_path']).with_name(attempt+'-settlement.json');milestone=0
 while not settlement.exists():
  assert time.time()<auth['expires_unix']+660
  p=Path(auth['output_dir'])/'fit/progress.json'
  if p.exists():
   step=read(p)['step'];bucket=step//5000
   if bucket>milestone:milestone=bucket;write(STATE/'progress.json',{**progress,'last_verified_logged_update':step,'milestone':bucket*5000})
  if owner.poll() is not None and not Path(auth['ledger_path']).exists():raise RuntimeError('Owner terminated without ledger; preserve attempt, no retry')
  time.sleep(10)
 owner.wait(timeout=60);(STATE/(attempt+'-finish.stdout')).write_text(command([PG,'-I','-S','-B',RUNTIME/'finish.py','--attempt',attempt],timeout=600))
 q=read(result);assert q['candidate_id']==cid and q['seed']==seed and q['endpoint']==20000 and q['protocol_SHA256']==b.sha(scope.PROTOCOL)
 assert q['clean_settlement'] and q['owned_processes_and_lease_absent'] and q['from_scratch'] and q['initial_completed_updates']==0
 commit([result],f'Retain fresh20k confirmation {cid} seed{seed}: native counts, matched controls, outcome and actual cost')
 return q
def main():
 p=argparse.ArgumentParser();p.add_argument('--activation',required=True);args=p.parse_args();protocol=scope.validate_protocol();assert read(scope.PLAN)==scope.ENABLED
 cells=[(q['candidate_id'],seed) for seed in range(1,5) for q in protocol['selected_configurations']];assert len(cells)==len(set(cells))<=12
 STATE.mkdir(exist_ok=True)
 with (STATE/'driver.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (STATE/'identity.json').exists(),'One-use queue consumed; inspect rather than replace'
  write(STATE/'identity.json',dict(pid=os.getpid(),birth=int(Path('/proc',str(os.getpid()),'stat').read_text().rsplit(')',1)[1].split()[19]),boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),source_SHA256=b.sha(__file__),protocol_SHA256=b.sha(scope.PROTOCOL),activation=args.activation,required_trials=len(cells)))
  rows=[]
  try:
   for cid,seed in cells:
    rows.append(fit(cid,seed,len(rows)));write(STATE/'partial_results.json',dict(rows=rows,all_planned_attempts_resolved=False))
   summary=scope.selector().summarize_confirmation(protocol['selected_configurations'],rows)
   result=dict(status='ALL_PLANNED_FRESH20K_CONFIRMATION_ATTEMPTS_AUDITED_REQUIRES_INDEPENDENT_REVIEW',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               **summary,actual_GPU_seconds=sum(q['charged_GPU_seconds'] for q in rows),protocol_SHA256=b.sha(scope.PROTOCOL),
               result_SHA256={q['attempt']:b.sha(STUDY/'results'/(q['attempt']+'.json')) for q in rows},external_candidate_actually_frozen=False)
   path=STUDY/'report.json';assert not path.exists();write(path,result)
   gp=ROOT/'analysis/current_research_goal.json';g=read(gp);g.update(study_status=result['status'],fresh_seed_confirmation_report_path=str(path),native_goal_status='active');write(gp,g)
   commit([path,gp],'Record matched fresh-seed robustness and all negative outcomes; external freeze and benchmark remain')
   command(['git','revert','--no-edit',args.activation]);write(STATE/'progress.json',dict(status=result['status'],report_path=str(path),activation_reverted=True,goal_still_active=True))
   report('completed','confirmation097-all-fresh-seed-attempts-audited-external-freeze-pending-20261106','Prespecified fresh20k initialization trials and matched controls audited; plan actually reverted, independent review and frozen external comparison remain.',dict(report_path=str(path),proposed_candidate=summary['proposed_external_candidate'],actual_GPU_seconds=result['actual_GPU_seconds'],SOTA_claim=False))
  except BaseException as exc:
   q=dict(status='CONFIRMATION097_OPERATIONAL_STOP_RETAINING_EVIDENCE',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),error=repr(exc),resolved_trials=len(rows),automatic_retry=False,full_goal_complete=False)
   write(STATE/'failure.json',q);report('failed','confirmation097-operational-stop-20261106','Fresh-seed confirmation stopped on an operational or audit issue; evidence retained, no automatic retry.',q);raise
if __name__=='__main__':main()
