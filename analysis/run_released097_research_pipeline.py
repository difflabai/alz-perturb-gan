"""Finite owner-authorized handoff from audited fresh baselines to comparison.

Each scientific phase has its own fixed protocol, real CPU qualification,
source/resource review, one-file activation, one-use guarded queue, settlement
and independent raw-artifact audit. No checkpoint warm start, foreign signal,
evaluation-driven parameter choice, automatic retry, or SOTA claim is allowed.
"""
import argparse,ast,datetime,fcntl,hashlib,json,os,subprocess,sys,time,urllib.request
from pathlib import Path
ROOT=Path('/ml2/alz');STATE=ROOT/'artifacts/released097-research-pipeline'
PG=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark/current_base_20261029/runtime/bin/python')
WINDOW=ROOT/'analysis/research_window_20261104_five_days.json'
ENV=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
STAGES={
 'local':dict(runtime='localfaith097_runtime',study='analysis/localfaith097',driver='analysis/run_localfaith097.py',state='artifacts/localfaith097-paired-search',review_scope='RELEASED097_FRESH20K_LOCAL_COORDINATE_SEARCH'),
 'confirmation':dict(runtime='confirmfaith097_runtime',study='analysis/confirmfaith097',driver='analysis/run_confirmfaith097.py',state='artifacts/confirmfaith097-paired-search',review_scope='RELEASED097_FRESH20K_MATCHED_CONFIRMATION'),
 'benchmark':dict(runtime='benchmark097_runtime',study='analysis/benchmark097',driver='analysis/run_benchmark097.py',state='artifacts/benchmark097-fullmatrix',review_scope='NATIVE097_FULL_MATRIX_ALL_FRESH_THREE_METHODS')}
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,q):
 p=Path(p);tmp=p.with_suffix('.partial');tmp.write_text(json.dumps(q,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def alive(i):
 p=Path('/proc',str(i['pid']),'stat')
 if not p.exists():return False
 try:s=p.read_text().rsplit(')',1)[1].split()
 except FileNotFoundError:return False
 return s[0]!='Z' and int(s[19])==i['birth'] and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==i['boot']
def require_sources(pins):
 for path,pin in pins.items():assert sha(path)==pin,'Pinned pipeline source changed: '+path

def command(name,args,timeout=180):
 p=subprocess.run(list(map(str,args)),cwd=ROOT,env=ENV,capture_output=True,text=True,timeout=timeout)
 (STATE/(name+'.stdout')).write_text(p.stdout);(STATE/(name+'.stderr')).write_text(p.stderr)
 assert p.returncode==0,name+' failed; retain evidence, no automatic retry: '+p.stderr[-1000:]
 return p.stdout

def commit(name,paths,message):
 command(name+'-add',['git','add','--',*map(str,paths)])
 command(name+'-commit',['git','commit','--only','-m',message,'--',*map(str,paths)])

def report(kind,event,summary,details,from_state=None,to_state=None):
 request=dict(recipient='bits255.dots.ada_20261002',conversation_id='report_auto_a20a79ed947ac6c51b74dbd821b796e2_ember',task_ref='01a1a22b-f310-7190-a7c6-8697ecabae6c',event_ref=event,kind=kind,summary=summary,details=json.dumps(details))
 if kind=='experiment_transition':request.update(from_state=from_state,to_state=to_state)
 helper=['/mnt/nvme/ai/conda/anaconda3/bin/python3','/home/mikkel/.local/lib/agent-babel/account_session_helper.py','tool','--client','codex','--account-file','/home/mikkel/.config/255bits/agent-babel/trusted-account.json','--native-session','01a1a22b-f310-7190-a7c6-8697ecabae6c','--tool','stage_report']
 p=subprocess.run(helper,input=json.dumps(request),cwd=ROOT,capture_output=True,text=True,timeout=30)
 (STATE/(event+'-report.stdout')).write_text(p.stdout);(STATE/(event+'-report.stderr')).write_text(p.stderr)
 # Transport failure is recorded separately; it never changes scientific state.

def immutable_sources():
 paths=[Path(__file__).resolve(),ROOT/'analysis/faithful097/protocol.json',ROOT/'analysis/faithful097/native_trainer_parity_CPU.json',ROOT/'analysis/faithful097_owner_execution_review.json',ROOT/'analysis/localfaith097/possible_settings_native_CPU_qualification.json',WINDOW]
 for s in STAGES.values():paths.extend((ROOT/s['runtime']).glob('*.py'));paths.append(ROOT/s['runtime']/'policy.prepared.json')
 paths.extend((ROOT/'analysis/faithful097/model').glob('*.py'))
 paths.extend((ROOT/'analysis/localfaith097').glob('*.py'))
 paths.extend(ROOT/'analysis'/n for n in ['prepare_localfaith097.py','run_localfaith097.py','audit_localfaith097.py','released097_local_search_design.py','released097_local_search_design_CPU_qualification_v2.json','released097_selection.py','released097_external_selection.py','prepare_confirmfaith097.py','run_confirmfaith097.py','audit_confirmfaith097.py','prepare_benchmark097.py','run_benchmark097.py','aggregate_benchmark097.py','distribution_primary_statistics.py'])
 return {str(p):sha(p) for p in paths}

def wait_exact(identity,expiry):
 while alive(identity):
  assert time.time()<expiry,'Observation allowance ended; original handles retained, no replacement'
  time.sleep(30)

def current_release(study):
 p=json.load(urllib.request.urlopen('https://pypi.org/pypi/particlegan/json',timeout=20))
 g=json.load(urllib.request.urlopen(urllib.request.Request('https://api.github.com/repos/255BITS/ParticleGAN/releases/latest',headers={'User-Agent':'ALZ-release-check'}),timeout=20))
 wheel=next(x for x in p['urls'] if x['filename'].endswith('.whl'))
 assert p['info']['version']=='0.9.7' and g['tag_name']=='v0.9.7' and wheel['digests']['sha256']=='8e6e748c84984e9fe979261c262e0a1300fe1225d1f3bc63c6d6a2e2f94c8189','Current release changed; requalification required before further launch'
 q=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),PyPI_version=p['info']['version'],GitHub_tag=g['tag_name'],wheel_SHA256=wheel['digests']['sha256'],PyPI_url='https://pypi.org/pypi/particlegan/json',GitHub_url=g['html_url'])
 path=study/'current_release_check_before_activation.json';assert not path.exists();write(path,q);return path

def owner_review(stage,release):
 s=STAGES[stage];runtime=ROOT/s['runtime']
 code='''import sys,json,datetime
from pathlib import Path
sys.path.insert(0,sys.argv[1]);import resource_profile as p,preset_scope as s
p.validate_proposal();p.previous_trial_released();q=s.read(s.RUNTIME_PROOF)
template=s.read(s.ROOT/'analysis/faithful097_owner_execution_review.json')
review=dict(status='OWNER_AUTHORIZED_SOURCE_AND_RESOURCE_REVIEW_PASS',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),reviewer_session='01a1a22b-f310-7190-a7c6-8697ecabae6c',independent_review_claimed=False,owner_authorization=template['owner_authorization'],scope=sys.argv[2],policy_SHA256=p.policy_sha(),source_SHA256=p.review_sources(),all_new_training_from_scratch=True,actual_runtime_CPU_tests=q.get('actual_runtime_tests',q.get('actual_tests')),current_release_check_path=sys.argv[3],current_release_check_SHA256=p.sha(sys.argv[3]),automatic_SOTA_claim=False)
assert not p.REVIEW.exists();p.REVIEW.write_text(json.dumps(review,indent=2)+'\\n');p.reviewed();print(p.REVIEW)
'''
 return Path(command(stage+'-owner-review',[PG,'-I','-S','-B','-c',code,runtime,s['review_scope'],release]).strip())

def run_stage(stage,pins,expiry):
 s=STAGES[stage];runtime=ROOT/s['runtime'];study=ROOT/s['study'];state=ROOT/s['state']
 require_sources(pins);assert time.time()<expiry-150
 assert not read(runtime/'plan.json')['enabled'] and not (state/'identity.json').exists()
 command(stage+'-qualify-real-frozen-scope',[PG,'-I','-S','-B',runtime/'qualify.py'],timeout=900)
 require_sources(pins)
 release=current_release(study);review=owner_review(stage,release);require_sources(pins)
 gp=ROOT/'analysis/current_research_goal.json';g=read(gp);g.update(study_status='QUALIFIED_'+stage.upper()+'_READY',study_protocol_path=str(study/'protocol.json'),study_progress_path=str(state/'progress.json'),full_goal_complete=False,native_goal_status='active');write(gp,g)
 # Every changed tracked study artifact is now concrete and independently
 # testable. Commit all exact phase evidence before the one-file activation.
 commit(stage+'-qualified',[study,review,gp],'Qualify and review frozen '+stage+' source, native state and complete declared scope')
 code="import sys,json;sys.path.insert(0,sys.argv[1]);import preset_scope as s,resource_profile as p;p.reviewed();assert not s.read(s.PLAN)['enabled'];s.PLAN.write_text(json.dumps(s.ENABLED,indent=2)+'\\n')"
 command(stage+'-activate-plan',[PG,'-I','-S','-B','-c',code,runtime])
 commit(stage+'-activation',[runtime/'plan.json'],'Activate qualified owner-authorized released097 '+stage+' phase')
 activation=command(stage+'-activation-id',['git','rev-parse','HEAD']).strip()
 assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True)
 state.mkdir(exist_ok=True)
 with (state/'driver.stdout').open('xb') as out,(state/'driver.stderr').open('xb') as err:
  child=subprocess.Popen([str(PG),'-I','-S','-B',str(ROOT/s['driver']),'--activation',activation],cwd=ROOT,env=ENV,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
 ident=dict(pid=child.pid,birth=int(Path('/proc',str(child.pid),'stat').read_text().rsplit(')',1)[1].split()[19]),boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),activation=activation)
 write(state/'launch_receipt.json',ident);write(STATE/'progress.json',dict(status='QUALIFIED_PHASE_QUEUE_LAUNCHED',phase=stage,driver=ident,actual_first_GPU_fit_admission_not_yet_claimed=True,full_goal_complete=False))
 report('experiment_transition','released097-pipeline-'+stage+'-queue-launched-20261107','Qualified fresh released097 '+stage+' queue launched under existing owner authorization; all endpoints and costs still require their audits.',dict(driver=ident,protocol_path=str(study/'protocol.json'),runtime_proof_path=str(study/'runtime_CPU_qualification.json'),GPU_fit_admission_not_yet_claimed=True),from_state='PREVIOUS_PHASE_AUDITED',to_state=stage.upper()+'_QUALIFIED_QUEUE_LAUNCHED')
 wait_exact(ident,expiry);child.wait(timeout=30)
 assert child.returncode==0 and not (state/'failure.json').exists(),stage+' driver stopped; preserve exact outputs without retry'
 assert read(state/'progress.json')['activation_reverted'] and not read(runtime/'plan.json')['enabled']
 require_sources(pins);return study

def audit_phase(stage,script,status):
 dest=ROOT/'analysis'/(stage+'097_pipeline_independent_audit_20261107.json');assert not dest.exists()
 command(stage+'-independent-raw-audit',[PG,'-I','-S','-B',ROOT/'analysis'/script,'--output',dest],timeout=900)
 assert read(dest)['status']==status
 commit(stage+'-audit',[dest],'Independently verify '+stage+' full native endpoints, matched controls, scores, costs and closure');return dest

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--check-only',action='store_true');args=parser.parse_args()
 pins=immutable_sources()
 for path in pins:
  if path.endswith('.py'):ast.parse(Path(path).read_text())
 parent=read(ROOT/'artifacts/faithful097-audit-boundary/identity.json')
 if args.check_only:
  assert alive(parent);print(json.dumps(dict(status='PIPELINE_READ_ONLY_SOURCE_CHECK_PARENT_LIVE',pinned_files=len(pins),GPU_allocation_created=False,activation=False,full_pipeline_execution_qualified=False)));return
 STATE.mkdir(exist_ok=True)
 with (STATE/'lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (STATE/'identity.json').exists(),'One-use pipeline consumed; inspect rather than replace'
  assert alive(parent);expiry=read(WINDOW)['expires_unix']+90
  write(STATE/'identity.json',dict(pid=os.getpid(),birth=int(Path('/proc',str(os.getpid()),'stat').read_text().rsplit(')',1)[1].split()[19]),boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),source_SHA256=pins,parent=parent,finite_expiry_unix=expiry,foreign_signals_permitted=False,checkpoint_warmstarts_permitted=False,full_goal_complete=False))
  try:
   wait_exact(parent,expiry);require_sources(pins)
   boundary=ROOT/'artifacts/faithful097-audit-boundary';assert not (boundary/'failure.json').exists() and (boundary/'completed.json').exists()
   selection_audit=Path(read(boundary/'completed.json')['complete_audit_path']);base=read(selection_audit)
   assert base['status']=='INDEPENDENT_RELEASED097_FRESH_BASELINE4_AUDIT_PASS'
   assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True)
   sys.path.insert(0,str(ROOT/'analysis'));from released097_local_search_design import design
   all_finite=all(c['all_seeds_finite'] for c in base['summary']['candidates'])
   proposed=design(base,time.time(),read(WINDOW)['expires_unix'],read(ROOT/'analysis/benchmark097_historical_cost_forecast_20261107.json')['estimated_matrix_GPU_hours']) if all_finite else None
   if proposed is not None and proposed['candidates']:
    assert read(ROOT/'analysis/localfaith097/possible_settings_native_CPU_qualification.json')['status']=='ALL24_POSSIBLE_LOCAL_SETTING_SEED_NATIVE_TRAINER_PREFIX_CASES_PASS'
    command('freeze-local-design',[PG,'-I','-S','-B',ROOT/'analysis/prepare_localfaith097.py','--baseline-audit',selection_audit])
    run_stage('local',pins,expiry)
    selection_audit=audit_phase('local','audit_localfaith097.py','INDEPENDENT_RELEASED097_PAIRED_SELECTION_AUDIT_PASS')
   else:
    write(STATE/'local_search_skipped.json',dict(reason='Baseline scientific failures need finite-choice confirmation' if not all_finite else 'No forecast slack for complete paired coordinate search',full45_model_scope_preserved=True,all_new_models_from_scratch=True,design=proposed))
   command('freeze-fresh-confirmation',[PG,'-I','-S','-B',ROOT/'analysis/prepare_confirmfaith097.py','--selection-audit',selection_audit])
   run_stage('confirmation',pins,expiry)
   confirmation_audit=audit_phase('confirmation','audit_confirmfaith097.py','INDEPENDENT_RELEASED097_MATCHED_CONFIRMATION_AUDIT_PASS')
   command('freeze-external-PG-before-evaluation',[PG,'-I','-S','-B',ROOT/'analysis/prepare_benchmark097.py','--selection-audit',selection_audit,'--confirmation-audit',confirmation_audit,'--confirmation-protocol',ROOT/'analysis/confirmfaith097/protocol.json','--confirmation-report',ROOT/'analysis/confirmfaith097/report.json'])
   run_stage('benchmark',pins,expiry)
   command('full45-paired-distribution-comparison',[PG,'-I','-S','-B',ROOT/'analysis/aggregate_benchmark097.py'],timeout=1800)
   comparison=ROOT/'analysis/benchmark097/comparison.json';assert read(comparison)['complete_cells']==45
   gp=ROOT/'analysis/current_research_goal.json';g=read(gp);g.update(study_status='FULL45_COMPARISON_READY_FINAL_REQUIREMENT_AUDIT_PENDING',external_comparison_path=str(comparison),full_goal_complete=False,native_goal_status='active');write(gp,g)
   commit('comparison',[comparison,gp],'Retain complete45fresh-model distribution comparison and uncertainty; final scope review remains')
   completed=dict(status='FULL_RESEARCH_PIPELINE_COMPARISON_READY_FOR_FINAL_AGENT_AUDIT',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),selection_audit_path=str(selection_audit),confirmation_audit_path=str(confirmation_audit),comparison_path=str(comparison),full_goal_complete=False,SOTA_claim=False,required_final_review='Verify full objective, every original endpoint, real artifacts and statistics, selection/evaluation separation limits, actual costs and properly bounded conclusions before marking native goal complete.')
   write(STATE/'completed.json',completed);report('completed','released097-full45-pipeline-comparison-ready-20261107','Complete fresh45-model comparison is ready for final requirement-by-requirement review; no automatic SOTA or goal-complete claim.',completed)
  except BaseException as exc:
   failure=dict(status='RESEARCH_PIPELINE_STOPPED_RETAINING_ORIGINAL_HANDLES',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),error=repr(exc),automatic_retry=False,original_GPU_jobs_not_signaled=True,full_goal_complete=False,SOTA_claim=False)
   write(STATE/'failure.json',failure);report('failed','released097-research-pipeline-stop-20261107','Research handoff stopped on a concrete qualification, source, endpoint or audit issue; original handles/evidence retained without retry.',failure);raise
if __name__=='__main__':main()
