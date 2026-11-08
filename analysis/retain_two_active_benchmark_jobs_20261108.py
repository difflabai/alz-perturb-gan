"""Drain and audit exactly two existing jobs; no admission or training entry point."""
import json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path('/ml2/alz');sys.path.insert(0,str(ROOT/'benchmark097_infra1_runtime'))
import binding as b
STATE=ROOT/'artifacts/benchmark-retained-two-hackathon-20261108'
KEYS=('CAL_GPU0_V1_external097infra1_f2_cfgen_s1','CAL_GPU0_V1_external097infra1_f2_pg_s1')
def read(p):return json.loads(Path(p).read_bytes())
def write(p,q):
 t=p.with_suffix('.partial');t.write_text(json.dumps(q,indent=2)+'\n');t.replace(p)
def present(ident):
 p=Path('/proc')/str(ident['pid'])/'stat'
 if not p.exists():return False
 v=p.read_text().rsplit(')',1)[1].split()
 return int(v[19])==ident['birth'] and v[0]!='Z' and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==ident['boot']
def main():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 prior=read(ROOT/'artifacts/benchmark097-infra1-pipeline/identity.json');assert not present(prior)
 STATE.mkdir();me=os.getpid();write(STATE/'identity.json',dict(pid=me,birth=int(Path('/proc',str(me),'stat').read_text().rsplit(')',1)[1].split()[19]),boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),source_SHA256=b.sha(__file__),admissions_allowed=False,retained_attempts=list(KEYS),superseded_CPU_owner=prior))
 results={};pending=list(KEYS);deadline=time.time()+12*3600
 try:
  while pending:
   assert time.time()<deadline,'Finite retained-job observation cap reached'
   for key in list(pending):
    auth=read(ROOT/'artifacts/allocations'/key/'authorization.json');ledger=Path(auth['ledger_path']);job=read(ledger)['jobs'][0];settlement=ledger.with_name(key+'-settlement.json')
    result=ROOT/'analysis/benchmark097_infra1/results'/(key+'.json')
    if not settlement.exists() or any(present(job[k]) for k in ['guard_identity','child_identity']):continue
    if not result.exists():
     exe=b.runtime_path(b.NATIVE_LAYOUT,auth['cell'])
     p=subprocess.run([str(exe),'-I','-S','-B',str(ROOT/'benchmark097_infra1_runtime/finish.py'),'--attempt',key],cwd=ROOT,env=os.environ.copy(),capture_output=True,text=True,timeout=900)
     (STATE/(key+'.stdout')).write_text(p.stdout);(STATE/(key+'.stderr')).write_text(p.stderr)
     assert p.returncode==0,f'Independent retained-job audit failed: {key}'
    q=read(result);assert q['attempt']==key
    results[key]=dict(complete=q['complete'],path=str(result),SHA256=b.sha(result),charged_GPU_seconds=q['charged_GPU_seconds']);pending.remove(key)
   write(STATE/'progress.json',dict(status='DRAINING_ONLY_EXISTING_NATIVE_JOBS',pending=pending,audited=results,new_admissions=0,third_replacement_held_for_hackathon_priority=True))
   if pending:time.sleep(30)
  write(STATE/'completed.json',dict(status='TWO_RETAINED_JOBS_INDEPENDENTLY_AUDITED',results=results,new_admissions=0,third_replacement_held=True,full_benchmark_matrix_complete=False))
 except BaseException as exc:
  write(STATE/'failure.json',dict(error=repr(exc),pending=pending,audited=results,foreign_processes_signalled=False,new_admissions=0));raise
if __name__=='__main__':main()
