"""Run three frozen fresh20k distribution trials; retain failures and audit all."""
import argparse,datetime,fcntl,hashlib,importlib.util,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path('/ml2/alz');RUNTIME=ROOT/'distribution097_runtime';STUDY=ROOT/'analysis/distribution097';STATE=ROOT/'artifacts/distribution097-stage1'
PG=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark/current_base_20261029/runtime/bin/python')
ENV=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
 p=Path(p);tmp=p.with_suffix('.partial');tmp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def command(args,timeout=120):
 p=subprocess.run(list(map(str,args)),cwd=ROOT,env=ENV,capture_output=True,text=True,timeout=timeout)
 if p.returncode:raise RuntimeError(p.stderr[-3000:])
 return p.stdout
def report(kind,event,summary,details):
 spec=importlib.util.spec_from_file_location('core_reporter',ROOT/'analysis/external_benchmark_driver.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.report(kind,event,summary,json.dumps(details))
def commit(paths,message):
 command(['git','add','--',*paths]);command(['git','commit','--only','-m',message,'--',*paths])
def run(cid):
 attempt='CAL_GPU0_V1_dist097_'+cid+'_f0_pg_s0';result=STUDY/'results'/(attempt+'.json')
 assert not result.exists() and not (ROOT/'artifacts/allocations'/attempt).exists(),'One-use run already exists; inspect rather than restart'
 assert not command(['git','status','--porcelain']),'Commit research outputs before the next queued arm'
 prepare=[PG,'-I','-S','-B',RUNTIME/'prepare.py','--candidate',cid]
 while True:
  assert time.time()<read(read(STUDY/'protocol.json')['research_window_path'])['expires_unix']-150
  p=subprocess.run(list(map(str,[*prepare,'--check-only'])),cwd=ROOT,env=ENV,capture_output=True,text=True,timeout=120)
  if p.returncode==0:break
  if not any(t in p.stderr for t in ('Exact parallel GPU growth forecasts do not fit','Exact parallel host growth forecasts do not fit','ALZ early stop: actual GPU free below3.5GiB','ALZ early stop: host available below12GiB')):raise RuntimeError(p.stderr[-3000:])
  write(STATE/'progress.json',dict(status='WAITING_FOR_SHARED_CAPACITY_NO_ALLOCATION',candidate=cid,attempt=attempt));time.sleep(15)
 (STATE/(cid+'-prepare.stdout')).write_text(command(prepare));folder=ROOT/'artifacts/allocations'/attempt;dep=read(folder/'deployment.json');auth=read(dep['authorization_path']);assert sha(dep['authorization_path'])==dep['authorization_SHA256']
 assert not Path(auth['ledger_path']).exists() and not Path(auth['output_dir']).exists()
 with (folder/'owner.stdout').open('xb') as out,(folder/'owner.stderr').open('xb') as err:
  owner=subprocess.Popen([str(PG),'-I','-S','-B',str(RUNTIME/'launch_core_gpu0.py'),'--authorization',dep['authorization_path']],cwd=ROOT,env=ENV,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
 write(folder/'launch_receipt.json',dict(pid=owner.pid,birth=int(Path('/proc',str(owner.pid),'stat').read_text().rsplit(')',1)[1].split()[19]),boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()))
 progress=dict(status='FRESH_NATIVE09720K_DISTRIBUTION_ARM_RUNNING',candidate=cid,attempt=attempt,endpoint=20000,seed=0,from_scratch=True,initial_completed_updates=0,hyperparameters=auth['released_hyperparameters'],GPU=auth['gpu_uuid'],audited_arms=len(list((STUDY/'results').glob('*.json'))));write(STATE/'progress.json',progress)
 settlement=Path(auth['ledger_path']).with_name(attempt+'-settlement.json');milestone=0
 while not settlement.exists():
  assert time.time()<auth['expires_unix']+660,'Owned phase failed to settle; no automatic retry'
  cp=Path(auth['output_dir'])/'fit/progress.json'
  if cp.exists():
   step=read(cp)['step'];bucket=min(step//5000,4)
   if bucket>milestone:milestone=bucket;write(STATE/'progress.json',{**progress,'last_verified_logged_update':step,'milestone':bucket*5000})
  if owner.poll() is not None and not Path(auth['ledger_path']).exists():raise RuntimeError('Owner terminated before ledger creation; inspect exact launch, no retry')
  time.sleep(10)
 owner.wait(timeout=60);(STATE/(cid+'-finish.stdout')).write_text(command([PG,'-I','-S','-B',RUNTIME/'finish.py','--attempt',attempt],timeout=600));q=read(result)
 assert q['candidate_id']==cid and q['endpoint']==20000 and q['protocol_SHA256']==sha(STUDY/'protocol.json') and q['clean_settlement'] and q['owned_processes_and_lease_absent']
 commit([str(result)],f'Retain native097 fresh20k distribution arm {cid}: actual counts, scores and settlement')
 return q

def main():
 p=argparse.ArgumentParser();p.add_argument('--activation',required=True);args=p.parse_args();STATE.mkdir(exist_ok=True)
 with (STATE/'driver.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  assert not (STATE/'failure.json').exists() and not (STATE/'driver_identity.json').exists(),'One-use queue already consumed; no restart'
  write(STATE/'driver_identity.json',dict(pid=os.getpid(),birth=int(Path('/proc',str(os.getpid()),'stat').read_text().rsplit(')',1)[1].split()[19]),boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),source_SHA256=sha(__file__),activation=args.activation))
  try:
   rows=[]
   for cid in ('d000','d001','d002'):
    rows.append(run(cid));write(STATE/'partial_results.json',dict(rows=rows,complete_stage1=False))
   finite=sorted([q for q in rows if q['finite']],key=lambda q:(q['metrics']['cell_energy_distance'],q['metrics']['energy_score'],q['candidate_id']))
   feasible=[q for q in finite if .7<=q['metrics']['spread_ratio']<=1.3 and q['metrics']['energy_score']<=read(ROOT/'analysis/baseline_calibration_seed0.json')['energy_score']]
   result=dict(status='COMPLETE_STAGE1_THREE_NATIVE097_FRESH20K_DISTRIBUTION_TRIALS',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),rows=[{k:q[k] for k in ('candidate_id','finite','metrics','hyperparameters','charged_GPU_seconds')} for q in rows],ranking_by_distribution=[q['candidate_id'] for q in finite],best_feasible_candidate=feasible[0]['candidate_id'] if feasible else None,total_actual_GPU_seconds=sum(q['charged_GPU_seconds'] for q in rows),all_new_fits_from_scratch=True,initial_completed_updates=0,native_version='0.9.7',formulation_changed=False,heldout_used=False,initialization_confirmation_complete=False,external_benchmark_complete=False,full_goal_complete=False,SOTA_claim=False,automatic_promotion=False,protocol_SHA256=sha(STUDY/'protocol.json'),result_SHA256={q['attempt']:sha(STUDY/'results'/(q['attempt']+'.json')) for q in rows},next='Independent stage1 review and evidence-led adaptation/seed confirmation remain; then frozen complete external comparison.')
   path=STUDY/'stage1_report.json';assert not path.exists();write(path,result)
   gp=ROOT/'analysis/current_research_goal.json';g=read(gp);g.update(study_status=result['status'],distribution_stage1_result_path=str(path),native_goal_status='active');write(gp,g);commit([str(path),str(gp)],'Record prospective fresh20k distribution stage1 without declaring broader goal complete')
   command(['git','revert','--no-edit',args.activation]);write(STATE/'progress.json',dict(status=result['status'],result_path=str(path),activation_reverted=True,goal_still_active=True))
   report('completed','distribution097-stage1-three-fresh20k-arms-audited-20261106','Three prospective fresh native0.9.7 distribution-focused20k arms are audited; stage1 only, broader SOTA goal remains active.',dict(result_path=str(path),best_feasible=result['best_feasible_candidate'],actual_GPU_seconds=result['total_actual_GPU_seconds']))
  except BaseException as exc:
   write(STATE/'failure.json',dict(status='DISTRIBUTION097_STAGE1_STOPPED_RETAINING_EVIDENCE',error=repr(exc),automatic_retry=False,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()))
   report('failed','distribution097-stage1-operational-stop-20261106','Native0.9.7 stage1 stopped on an operational/audit issue; all evidence retained, no automatic retry.',dict(error=repr(exc),failure_path=str(STATE/'failure.json')));raise
if __name__=='__main__':main()
