"""Finite CPU-only sensitivity after the exact full research pipeline closes."""
import datetime,fcntl,hashlib,json,os,subprocess,time
from pathlib import Path
ROOT=Path('/ml2/alz');STATE=ROOT/'artifacts/released097-dual-calibration53-boundary-v2'
PG=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark/current_base_20261029/runtime/bin/python')
ENV=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,q):Path(p).write_text(json.dumps(q,indent=2,allow_nan=False)+'\n')
def alive(i):
 p=Path('/proc',str(i['pid']),'stat')
 if not p.exists():return False
 try:s=p.read_text().rsplit(')',1)[1].split()
 except FileNotFoundError:return False
 return s[0]!='Z' and int(s[19])==i['birth'] and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==i['boot']
def main():
 STATE.mkdir(exist_ok=True)
 with (STATE/'lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (STATE/'identity.json').exists(),'One-use CPU sensitivity observer consumed'
  parent=read(ROOT/'artifacts/released097-dual-research-pipeline-v2/identity.json');assert alive(parent)
  files=[Path(__file__).resolve(),ROOT/'analysis/score_calibration_excluded53_dual.py',ROOT/'analysis/benchmark097_calibration_excluded53_plan.json',ROOT/'analysis/aggregate_benchmark097_dual.py',ROOT/'analysis/distribution_primary_statistics.py']
  pins={str(p):sha(p) for p in files};expiry=read(ROOT/'analysis/research_window_20261104_five_days.json')['expires_unix']+7200
  write(STATE/'identity.json',dict(pid=os.getpid(),birth=int(Path('/proc',str(os.getpid()),'stat').read_text().rsplit(')',1)[1].split()[19]),boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),parent={k:parent[k] for k in ('pid','birth','boot')},finite_CPU_analysis_expiry_unix=expiry,training_window_extended=False,GPU_allocation_permitted=False,source_SHA256=pins))
  try:
   while alive(parent):
    assert time.time()<expiry,'CPU observation allowance expired; original handles retained, no restart'
    time.sleep(30)
   for p,pin in pins.items():assert sha(p)==pin,'Prespecified sensitivity source changed'
   pstate=ROOT/'artifacts/released097-dual-research-pipeline-v2';assert not (pstate/'failure.json').exists() and (pstate/'completed.json').exists()
   assert read(pstate/'completed.json')['status']=='FULL_RESEARCH_PIPELINE_COMPARISON_READY_FOR_FINAL_AGENT_AUDIT'
   assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True)
   p=subprocess.run([str(PG),'-I','-S','-B',str(ROOT/'analysis/score_calibration_excluded53_dual.py')],cwd=ROOT,env=ENV,capture_output=True,text=True,timeout=3600)
   (STATE/'score.stdout').write_text(p.stdout);(STATE/'score.stderr').write_text(p.stderr);assert p.returncode==0,'CPU sensitivity stopped; no model restart or automatic retry'
   report=ROOT/'analysis/benchmark097_calibration_excluded53_report.json';q=read(report);assert q['status']=='COMPLETE_PRESPECIFIED_CALIBRATION_EXCLUDED53_SENSITIVITY' and q['complete_fresh_generative_models']==45 and q['new_model_fits']==q['new_generated_RNA_counts']==0
   gp=ROOT/'analysis/current_research_goal.json';g=read(gp);g.update(study_status='FULL57_COMPARISON_AND_CALIBRATION_EXCLUDED53_SENSITIVITY_READY_FINAL_AUDIT_PENDING',calibration_excluded53_report_path=str(report),calibration_excluded53_completed=True,native_goal_status='active',full_goal_complete=False);write(gp,g)
   subprocess.run(['git','add','--',str(report),str(gp)],cwd=ROOT,check=True,capture_output=True,text=True)
   subprocess.run(['git','commit','--only','-m','Retain prespecified53-donor sensitivity while preserving full57primary and all45fresh models','--',str(report),str(gp)],cwd=ROOT,check=True,capture_output=True,text=True)
   write(STATE/'completed.json',dict(status='CALIBRATION_EXCLUDED53_SENSITIVITY_COMPLETE_FINAL_AGENT_AUDIT_REMAINS',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),report_path=str(report),original57_primary_preserved=True,new_model_fits=0,new_generated_counts=0,SOTA_claim=False,full_goal_complete=False))
  except BaseException as exc:
   write(STATE/'failure.json',dict(status='CPU_CALIBRATION53_OBSERVER_STOPPED_RETAINING_EVIDENCE',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),error=repr(exc),automatic_retry=False,GPU_jobs_signaled=False,GPU_allocation_created=False,full_goal_complete=False));raise
if __name__=='__main__':main()
