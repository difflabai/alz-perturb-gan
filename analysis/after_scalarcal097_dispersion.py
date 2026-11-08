"""Finite CPU handoff observer; retain original upstream, then six fresh fits.

No current model is interrupted, no foreign process is controlled, and a missing
upstream handle is never taken as proof of successful completion.
"""
import datetime,fcntl,hashlib,importlib.util,json,os,subprocess,sys,time,urllib.request
from pathlib import Path
ROOT=Path('/ml2/alz');RUNTIME=ROOT/'dispersion097_runtime';STATE=ROOT/'artifacts/dispersion097-after-scalarcal097'
sys.path.insert(0,str(RUNTIME))
import binding as b,preset_scope as s,resource_profile as profile
from dual_coordination import GPUS,identity
ENV=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
PG=b.runtime_path(b.NATIVE_LAYOUT,s.cell('theta1',8))
def write(p,q):
 p=Path(p);tmp=p.with_suffix('.partial');tmp.write_text(json.dumps(q,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def command(args,timeout=120):
 p=subprocess.run(list(map(str,args)),cwd=ROOT,env=ENV,capture_output=True,text=True,timeout=timeout)
 if p.returncode:raise RuntimeError(p.stderr[-3000:])
 return p.stdout
def commit(paths,message):command(['git','add','--',*paths]);command(['git','commit','--only','-m',message,'--',*paths])
def report(kind,event,summary,details):
 from babel_reporting import report as send
 return send(kind,event,summary,details,log_root=STATE)

def main():
 assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True) and not s.read(s.PLAN)['enabled']
 proof=s.read(s.RUNTIME_PROOF);assert proof['status']=='DISPERSION097_SIX_FRESH_CELLS_SCOPE_ACCOUNTING_CPU_PASS'
 pins={**proof['source_SHA256'],str(s.RUNTIME_PROOF):s.sha(s.RUNTIME_PROOF)}
 for path,pin in pins.items():assert s.sha(path)==pin
 upstream=ROOT/'artifacts/scalarcal097-after-recovered-science-v5';parent=s.read(upstream/'identity.json')
 assert identity(parent['pid'])=={k:parent[k] for k in ('pid','birth','boot')},'Verify the exact currently live balanced-science continuation before waiting'
 expiry=s.read(s.WINDOW)['expires_unix'];STATE.mkdir(exist_ok=False)
 with (STATE/'observer.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  write(STATE/'identity.json',dict(**identity(os.getpid()),source_SHA256=s.sha(__file__),upstream={k:parent[k] for k in ('pid','birth','boot')},finite_training_expiry_unix=expiry,new_GPU_allocation=False))
  try:
   while identity(parent['pid'])=={k:parent[k] for k in ('pid','birth','boot')}:
    assert time.time()<expiry-150,'Owner training window ended before the independent handoff'
    assert not (upstream/'failure.json').exists(),'Original upstream failed; no restart or successor admission'
    write(STATE/'progress.json',dict(status='WAITING_ON_COMPLETED_BENCHMARK_AND_SIX_SCALAR_LR_FITS',new_GPU_allocation=False,upstream_pid=parent['pid'],full_goal_complete=False));time.sleep(30)
   assert not (upstream/'failure.json').exists();handoff=s.prior_study_ready()
   for path,pin in pins.items():assert s.sha(path)==pin
   assert time.time()<expiry-150 and not s.read(b.PROJECT/'project_gpu_budget.json')['open_reservations']
   assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True)
   with urllib.request.urlopen('https://pypi.org/pypi/particlegan/json',timeout=30) as stream:release=json.load(stream)
   check=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),official_PyPI_latest=release['info']['version'],qualified_version='0.9.7',old_results_invalidated=False)
   write(STATE/'release_check.json',check)
   assert release['info']['version']=='0.9.7','New release observed: retain current evidence and qualify the separate phase against the new release before activation'
   wheels=[row for row in release['urls'] if row['filename']=='particlegan-0.9.7-py3-none-any.whl']
   assert len(wheels)==1 and wheels[0]['digests']['sha256']=='8e6e748c84984e9fe979261c262e0a1300fe1225d1f3bc63c6d6a2e2f94c8189'
   from runtime_dispatch import settled_cost_rows,cost_table
   from released097_remaining_runtime_balance import balance_counts
   costs,evidence=cost_table(settled_cost_rows());forecast=balance_counts({'pg':6},costs)
   allowance=1.25*forecast['predicted_makespan_seconds']+1200
   viability=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),costs_seconds=costs,cost_source_attempts=evidence,forecast=forecast,safety_multiplier=1.25,additional_report_cleanup_seconds=1200,required_seconds=allowance,remaining_window_seconds=expiry-time.time(),quality_values_read=False,not_a_completion_guarantee=True)
   write(STATE/'window_cost_preflight.json',viability)
   if allowance>expiry-time.time():
    deferred=dict(status='ALL_SIX_THETA_FITS_DEFERRED_UNSTARTED_INSUFFICIENT_REMAINING_WINDOW',new_GPU_allocations=0,actual_fresh_fits=0,viability=viability,full_goal_complete=False,SOTA_claim=False)
    write(STATE/'deferred.json',deferred)
    report('blocked','dispersion097-insufficient-window-all-six-unstarted','Current cost/window preflight cannot accommodate all six fresh theta fits; none admitted and full goal remains incomplete.',deferred)
    return
   observations={}
   import prepare
   while time.time()<expiry-150:
    for gpu in GPUS:
     auth,_=prepare.build_authorization(arm='theta1',seed=8,gpu=gpu,check_only=True)
     try:
      _,admission,_=b.modules(auth);observations[gpu]=profile.capacity(admission.snapshot(),admission,gpu);profile.free_floor(gpu)
     except AssertionError as exc:
      if not any(text in str(exc) for text in ('growth forecasts do not fit','ALZ early stop:')):raise
      observations[gpu]=dict(GPU_fits=False,host_fits=False,temporarily_unavailable=True)
    if any(q['GPU_fits'] and q['host_fits'] for q in observations.values()):break
    write(STATE/'progress.json',dict(status='WAITING_FOR_SHARED_CAPACITY_BEFORE_FRESH_CALIBRATION',new_GPU_allocation=False));time.sleep(30)
   else:raise TimeoutError('Owner training window ended before shared capacity became available')
   review=dict(status='OWNER_AUTHORIZED_SOURCE_AND_RESOURCE_REVIEW_PASS',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),reviewer_session='01a1a22b-f310-7190-a7c6-8697ecabae6c',independent_review_claimed=False,owner_authorization=s.read(ROOT/'analysis/faithful097_owner_execution_review.json')['owner_authorization'],scope=profile.SCOPE,policy_SHA256=profile.policy_sha(),source_SHA256=profile.review_sources(),handoff_SHA256=handoff,shared_capacity=observations)
   assert not profile.REVIEW.exists();write(profile.REVIEW,review);commit([str(profile.REVIEW)],'Review separate theta initialization runtime against clean frozen-study handoff')
   write(s.PLAN,s.ENABLED);commit([str(s.PLAN)],'Activate separate six fresh theta initialization fits within existing budget/window')
   activation=command(['git','rev-parse','HEAD']).strip()
   command([PG,'-I','-S','-B',RUNTIME/'post_handoff_check.py'],timeout=300)
   commit([str(s.STUDY/'post_handoff_source_closure.json')],'Verify all12 current scalar cell/card source closures before fresh native admission')
   with (STATE/'driver.stdout').open('xb') as out,(STATE/'driver.stderr').open('xb') as err:
    child=subprocess.Popen([str(PG),'-I','-S','-B',str(ROOT/'analysis/run_dispersion097.py'),'--activation',activation],cwd=ROOT,env=ENV,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
   handle=identity(child.pid);assert handle is not None
   write(STATE/'launch_receipt.json',dict(**handle,activation=activation,source_SHA256=s.sha(ROOT/'analysis/run_dispersion097.py'),not_a_GPU_fit_claim=True))
   while child.poll() is None:
    assert time.time()<expiry+7200,'Finite CPU observation ended; exact child handle retained without signals'
    time.sleep(30)
   assert child.returncode==0,'Separate theta driver incomplete; retain outcomes and exact handles without retry'
   assert not s.read(s.PLAN)['enabled']
   result=subprocess.run([str(PG),'-I','-S','-B',str(ROOT/'analysis/aggregate_dispersion097.py')],cwd=ROOT,env=ENV,capture_output=True,text=True,timeout=1800)
   (STATE/'aggregate.stdout').write_text(result.stdout);(STATE/'aggregate.stderr').write_text(result.stderr);assert result.returncode==0
   path=s.STUDY/'comparison.json';q=s.read(path);assert q['complete_fresh_fits']==6
   gp=ROOT/'analysis/current_research_goal.json';g=s.read(gp);g.update(dispersion_calibration_report_path=str(path),dispersion_fresh_fits_completed=6,native_goal_status='active',full_goal_complete=False);g.update(dispersion_followup_real_ALZ_fits=6,dispersion_followup_status='ALL_SIX_FRESH_THETA_FITS_AND_CALIBRATION_REPORT_COMPLETE_FINAL_RESEARCH_REVIEW_REMAINS');write(gp,g)
   commit([str(path),str(gp)],'Retain complete fresh theta-initialization calibration comparison without SOTA promotion')
   write(STATE/'completed.json',dict(status='DISPERSION097_COMPLETE_CALIBRATION_REPORT_FINAL_RESEARCH_AUDIT_REMAINS',report_path=str(path),full_goal_complete=False,SOTA_claim=False))
   report('completed','dispersion097-six-fresh-calibration-complete','All six fresh20k theta1/theta5 calibration fits and independent audits complete. No SOTA or old45 candidate change.',dict(report_path=str(path),calibration_preferred_arm=q['calibration_preferred_arm']))
  except BaseException as exc:
   failure=dict(status='DISPERSION097_HANDOFF_OR_DRIVER_STOPPED_RETAINING_EVIDENCE',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),error=repr(exc),automatic_retry=False,foreign_or_upstream_jobs_signaled=False,full_goal_complete=False,SOTA_claim=False);write(STATE/'failure.json',failure)
   report('failed','dispersion097-handoff-or-driver-stopped','Separate theta initialization continuation stopped with evidence retained; no upstream restart, foreign signals or success claim.',failure);raise
if __name__=='__main__':main()
