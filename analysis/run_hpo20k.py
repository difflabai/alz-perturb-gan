"""Run one frozen fresh20k c014 fit with finite guards, audit and disable."""
import argparse,datetime,fcntl,hashlib,importlib.util,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path('/ml2/alz');RUNTIME=ROOT/'hpo20k_runtime';STUDY=ROOT/'analysis/hpo20k';STATE=ROOT/'artifacts/hpo20k-c014-batch'
PG=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark/current_base_20261029/runtime/bin/python')
ENV=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
ATTEMPT='CAL_GPU0_V1_h20_c014_f0_pg_s0'
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
def main():
 p=argparse.ArgumentParser();p.add_argument('--activation',required=True);args=p.parse_args();STATE.mkdir(exist_ok=True)
 with (STATE/'driver.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  assert not (STATE/'failure.json').exists() and not (STATE/'driver_identity.json').exists(),'Inspect existing one-use run, never relaunch blindly'
  write(STATE/'driver_identity.json',dict(pid=os.getpid(),birth=int(Path('/proc',str(os.getpid()),'stat').read_text().rsplit(')',1)[1].split()[19]),boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),source_SHA256=sha(__file__),activation=args.activation))
  try:
   assert not command(['git','status','--porcelain'])
   prepare=[PG,'-I','-S','-B',RUNTIME/'prepare.py']
   while True:
    assert time.time()<read(read(STUDY/'protocol.json')['research_window_path'])['expires_unix']-150
    p=subprocess.run(list(map(str,[*prepare,'--check-only'])),cwd=ROOT,env=ENV,capture_output=True,text=True,timeout=120)
    if p.returncode==0:break
    if not any(t in p.stderr for t in ('Exact parallel GPU growth forecasts do not fit','Exact parallel host growth forecasts do not fit','ALZ early stop: actual GPU free below3.5GiB','ALZ early stop: host available below12GiB')):raise RuntimeError(p.stderr[-3000:])
    write(STATE/'progress.json',dict(status='WAITING_FOR_SHARED_CAPACITY_NO_ALLOCATION',attempt=ATTEMPT));time.sleep(15)
   (STATE/'prepare.stdout').write_text(command(prepare));folder=ROOT/'artifacts/allocations'/ATTEMPT;dep=read(folder/'deployment.json');auth=read(dep['authorization_path']);assert sha(dep['authorization_path'])==dep['authorization_SHA256']
   assert not Path(auth['ledger_path']).exists() and not Path(auth['output_dir']).exists()
   with (folder/'owner.stdout').open('xb') as out,(folder/'owner.stderr').open('xb') as err:
    owner=subprocess.Popen([str(PG),'-I','-S','-B',str(RUNTIME/'launch_core_gpu0.py'),'--authorization',dep['authorization_path']],cwd=ROOT,env=ENV,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
   write(folder/'launch_receipt.json',dict(pid=owner.pid,birth=int(Path('/proc',str(owner.pid),'stat').read_text().rsplit(')',1)[1].split()[19]),boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()))
   progress=dict(status='FRESH20K_NATIVE_FIT_OR_CALIBRATION_RUNNING',attempt=ATTEMPT,endpoint=20000,seed=0,hyperparameters=auth['released_hyperparameters'],GPU=auth['gpu_uuid']);write(STATE/'progress.json',progress)
   settlement=Path(auth['ledger_path']).with_name(ATTEMPT+'-settlement.json');milestone=0
   while not settlement.exists():
    assert time.time()<auth['expires_unix']+660,'Owned phase failed to settle; no automatic retry'
    cp=Path(auth['output_dir'])/'fit/progress.json'
    if cp.exists():
     step=read(cp)['step'];bucket=min(step//5000,4)
     if bucket>milestone:milestone=bucket;write(STATE/'progress.json',{**progress,'last_verified_logged_update':step,'milestone':bucket*5000})
    if owner.poll() is not None and not Path(auth['ledger_path']).exists():raise RuntimeError('Owner terminated before ledger creation; inspect exact launch and no retry')
    time.sleep(10)
   owner.wait(timeout=60);(STATE/'finish.stdout').write_text(command([PG,'-I','-S','-B',RUNTIME/'finish.py','--attempt',ATTEMPT],timeout=600))
   result=STUDY/'results'/(ATTEMPT+'.json');q=read(result)
   command(['git','add','--',result]);command(['git','commit','--only','-m','Record fresh20k c014 endpoint, counts audit and actual resource settlement','--',result])
   command(['git','revert','--no-edit',args.activation])
   write(STATE/'progress.json',dict(status='COMPLETE_FRESH20K_C014_AUDITED' if q['finite'] else 'SCIENTIFIC_NONFINITE20K_C014_RETAINED',result_path=str(result),endpoint=20000,finite=q['finite'],activation_reverted=True))
   report('completed' if q['finite'] else 'failed','hpo20k-c014-endpoint-audited-20261106','Fresh c014 twenty-thousand-update run settled and independently audited; activation reverted.',dict(result_path=str(result),finite=q['finite'],metrics=q['metrics'],actual_GPU_seconds=q['charged_GPU_seconds']))
  except BaseException as exc:
   write(STATE/'failure.json',dict(status='FRESH20K_C014_STOPPED_RETAINING_EVIDENCE',error=repr(exc),automatic_retry=False,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()))
   report('failed','hpo20k-c014-operational-stop-20261106','Fresh c01420k run stopped on an operational or audit issue. All evidence retained; no automatic retry.',dict(error=repr(exc),failure_path=str(STATE/'failure.json')));raise
if __name__=='__main__':main()
