"""Preserve the live9-fit queue, then qualify the unchanged45-cell dual queue."""
import argparse,ast,datetime,fcntl,json,os,subprocess,time
from pathlib import Path
import sys
ROOT=Path('/ml2/alz');sys.path.insert(0,str(ROOT/'analysis'))
import run_released097_research_pipeline as engine
STATE=ROOT/'artifacts/released097-dual-research-pipeline-v2';engine.STATE=STATE
engine.STAGES['benchmark']=dict(runtime='benchmark097_dual_runtime',study='analysis/benchmark097',driver='analysis/run_benchmark097_dual.py',state='artifacts/benchmark097-dual-fullmatrix',review_scope='NATIVE097_FULL_MATRIX_ALL_FRESH_THREE_METHODS_DUAL_SHARED_GPUS')
OLD_STATE=ROOT/'artifacts/released097-adaptive-research-pipeline'
SELECTION=ROOT/'analysis/local097_pipeline_independent_audit_20261107.json'
PREPARED=ROOT/'analysis/benchmark097/dual_prepared_CPU_qualification_v5.json'
SOURCE_REPAIR=ROOT/'analysis/released097_dual_source_closure_repair_20261107.json'
ANALYSIS=ROOT/'analysis/benchmark097_dual_analysis_CPU_preparation.json'
EXTRA=['run_released097_dual_pipeline.py','after_released097_dual_calibration53.py','run_benchmark097_dual.py','aggregate_benchmark097_dual.py','score_calibration_excluded53_dual.py','test_aggregate_benchmark097_dual.py','test_calibration_excluded53_dual.py','test_benchmark097_dual_analysis_gates.py','benchmark097_dual_resource_protocol.json','benchmark097_dual_analysis_CPU_preparation.json']
def validate_preparation():
 old=engine.read(OLD_STATE/'identity.json');engine.require_sources(old['source_SHA256'])
 repair=engine.read(SOURCE_REPAIR);prior=engine.read(ROOT/'artifacts/released097-dual-research-pipeline/identity.json');prior_observer=engine.read(ROOT/'artifacts/released097-dual-calibration53-boundary/identity.json')
 assert repair['status']=='DUAL_NATIVE_SOURCE_CLOSURE_REPAIR_CPU_PARENTS_RECONCILED_GPU9_PRESERVED' and repair['GPU_processes_signalled'] is False
 assert repair['prior_pipeline']=={k:prior[k] for k in ('pid','birth','boot')} and repair['prior53_observer']=={k:prior_observer[k] for k in ('pid','birth','boot')}
 assert not engine.alive(prior) and not engine.alive(prior_observer)
 transitions=repair['source_transition'];assert transitions
 for path,pin in prior['source_SHA256'].items():
  if path in transitions:assert transitions[path]['before']==pin and engine.sha(path)==transitions[path]['after'] and path not in old['source_SHA256']
  else:assert engine.sha(path)==pin
 assert set(transitions)<=set(prior['source_SHA256'])
 observer=engine.read(ROOT/'artifacts/released097-adaptive-calibration53-boundary/identity.json');assert observer['parent']=={k:old[k] for k in ('pid','birth','boot')};engine.require_sources(observer['source_SHA256'])
 parent=engine.read(ROOT/'artifacts/confirmfaith097-paired-search/identity.json');assert engine.alive(parent) and parent['activation']=='81efc12bfe4cb22dddb318ffbe38874010523e1e'
 assert engine.read(OLD_STATE/'progress.json')['phase']=='confirmation'
 assert not engine.read(ROOT/'benchmark097_runtime/plan.json')['enabled'] and not engine.read(ROOT/'benchmark097_dual_runtime/plan.json')['enabled']
 assert engine.read(ROOT/'analysis/benchmark097/protocol.json')['status']=='PREPARED_NOT_FROZEN_PG_SELECTION_AND_CONFIRMATION_PENDING'
 assert not list((ROOT/'analysis/benchmark097/results').glob('*.json')) and not (ROOT/'artifacts/benchmark097-dual-fullmatrix/identity.json').exists() and not (ROOT/'artifacts/benchmark097-fullmatrix/identity.json').exists()
 p=engine.read(PREPARED);assert p['status']=='PREPARED_ACCOUNTING_AND_SYNTHETIC_SELECTION_SCOPE_CPU_TESTS_PASS' and p['actual_tests']==42 and p['actual_all45_native_preimport_source_closures_checked'] and not p['actual_frozen_PG_candidate_qualified'] and not p['live_runtime_deployment_qualified']
 engine.require_sources(p['source_SHA256']);a=engine.read(ANALYSIS);assert a['actual_CPU_tests']==14 and not a['real_target_values_read'];engine.require_sources(a['source_SHA256'])
 assert engine.read(SELECTION)['status']=='INDEPENDENT_RELEASED097_PAIRED_SELECTION_AUDIT_PASS'
 readiness=ROOT/'analysis/benchmark097_dual_metadata_readiness_v2_20261107.json';r=engine.read(readiness);assert r['status']=='ALL45_NEW_CELLS_FRESH_METADATA_AND_JOINT_CORE_BUDGET_PASS' and r['required_cells']==45 and not r['collisions'];engine.require_sources(r['source_SHA256'])
 pins=dict(old['source_SHA256']);pins.update(observer['source_SHA256']);pins.update(engine.immutable_sources())
 paths=[ROOT/'analysis'/x for x in EXTRA]+[PREPARED,SELECTION,readiness,SOURCE_REPAIR,ROOT/'benchmark097_dual_runtime/policy.gpu0.prepared.json',ROOT/'benchmark097_dual_runtime/policy.gpu1.prepared.json']
 pins.update({str(p):engine.sha(p) for p in paths})
 for p in pins:
  if p.endswith('.py'):ast.parse(Path(p).read_text())
 return old,observer,parent,pins
def main():
 p=argparse.ArgumentParser();p.add_argument('--check-only',action='store_true');args=p.parse_args();old,observer,parent,pins=validate_preparation()
 if args.check_only:
  assert not engine.alive(old) and not engine.alive(observer),'Earlier owned CPU handles must remain reconciled'
  print(json.dumps(dict(status='DUAL_PIPELINE_PREPARATION_SOURCE_CHECK_PASS',original_confirmation_driver=parent['pid'],pinned_files=len(pins),CPU_handoff_performed=False,GPU_allocation_created=False)));return
 handoff=engine.read(ROOT/'analysis/released097_dual_CPU_handoff_20261107.json')
 assert handoff['status']=='ADAPTIVE_CPU_PARENT_SUPERSEDED_ORIGINAL_CONFIRMATION_GPU_HANDLES_PRESERVED'
 assert handoff['original_pipeline']=={k:old[k] for k in ('pid','birth','boot')} and handoff['original_sensitivity_observer']=={k:observer[k] for k in ('pid','birth','boot')}
 assert not engine.alive(old) and not engine.alive(observer) and handoff['GPU_processes_signalled'] is False
 STATE.mkdir(exist_ok=True)
 with (STATE/'lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (STATE/'identity.json').exists(),'One-use CPU continuation consumed'
  training_expiry=engine.read(engine.WINDOW)['expires_unix'];CPU_expiry=training_expiry+7200
  engine.write(STATE/'identity.json',dict(pid=os.getpid(),birth=int(Path('/proc',str(os.getpid()),'stat').read_text().rsplit(')',1)[1].split()[19]),boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),source_SHA256=pins,parent={k:parent[k] for k in ('pid','birth','boot')},original_pipeline=handoff['original_pipeline'],training_expiry_unix=training_expiry,finite_CPU_observation_expiry_unix=CPU_expiry,training_window_extended=False,foreign_signals_permitted=False,checkpoint_warmstarts_permitted=False,full45_required=True,full_goal_complete=False))
  try:
   engine.write(STATE/'progress.json',dict(status='WAITING_EXACT_ORIGINAL9_DRIVER_BEFORE_DUAL45',driver=parent,full_goal_complete=False))
   while engine.alive(parent):
    assert time.time()<CPU_expiry,'Finite observation allowance ended; original handles retained'
    engine.require_sources(pins);time.sleep(30)
   engine.require_sources(pins);boundary=ROOT/'artifacts/confirmfaith097-paired-search'
   assert not (boundary/'failure.json').exists() and engine.read(boundary/'progress.json')['activation_reverted']
   assert not engine.read(ROOT/'confirmfaith097_runtime/plan.json')['enabled']
   confirmation=engine.audit_phase('confirmation','audit_confirmfaith097.py','INDEPENDENT_RELEASED097_MATCHED_CONFIRMATION_AUDIT_PASS')
   engine.command('freeze-external-PG-before-values',[engine.PG,'-I','-S','-B',ROOT/'analysis/prepare_benchmark097.py','--selection-audit',SELECTION,'--confirmation-audit',confirmation,'--confirmation-protocol',ROOT/'analysis/confirmfaith097/protocol.json','--confirmation-report',ROOT/'analysis/confirmfaith097/report.json'])
   assert time.time()<training_expiry-150,'Owner training admission window ended; retain full uncompleted scope'
   engine.run_stage('benchmark',pins,CPU_expiry)
   engine.command('full45-dual-paired-distribution-comparison',[engine.PG,'-I','-S','-B',ROOT/'analysis/aggregate_benchmark097_dual.py'],timeout=1800)
   comparison=ROOT/'analysis/benchmark097/comparison.json';assert engine.read(comparison)['complete_cells']==45
   gp=ROOT/'analysis/current_research_goal.json';g=engine.read(gp);g.update(study_status='FULL45_COMPARISON_READY_FINAL_REQUIREMENT_AUDIT_PENDING',external_comparison_path=str(comparison),full_goal_complete=False,native_goal_status='active');engine.write(gp,g);engine.commit('comparison',[comparison,gp],'Retain full45 paired comparison from the same fresh two-card native matrix')
   completed=dict(status='FULL_RESEARCH_PIPELINE_COMPARISON_READY_FOR_FINAL_AGENT_AUDIT',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),selection_audit_path=str(SELECTION),confirmation_audit_path=str(confirmation),comparison_path=str(comparison),full_goal_complete=False,SOTA_claim=False,required_final_review='Original objective, every fresh endpoint/count/state/artifact,57paired statistics,53prespecified sensitivity, historical/calibration exposure, actual costs and bounded conclusions.')
   engine.write(STATE/'completed.json',completed);engine.report('completed','released097-dual-full45-comparison-ready-20261107','Full45 fresh-model paired comparison is ready for final original-objective audit;53sensitivity and bounded conclusions remain.',completed)
  except BaseException as exc:
   failure=dict(status='DUAL_RESEARCH_PIPELINE_STOPPED_RETAINING_ORIGINAL_HANDLES',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),error=repr(exc),automatic_retry=False,GPU_jobs_not_signalled=True,full_goal_complete=False,SOTA_claim=False);engine.write(STATE/'failure.json',failure);engine.report('failed','released097-dual-pipeline-stop-20261107','Dual continuation stopped on an actual source, qualification, endpoint or audit issue; original handles retained.',failure);raise
if __name__=='__main__':main()
