"""Frozen six-fit calibration experiment, isolated from external evaluation."""
import hashlib,json,math,sys
from pathlib import Path
ROOT=Path('/ml2/alz');HERE=Path(__file__).resolve().parent;STUDY=ROOT/'analysis/scalarcal097'
sys.path.append(str(ROOT/'analysis'))
PROTOCOL=STUDY/'protocol.json';PLAN=HERE/'plan.json';RESOURCE_PROTOCOL=STUDY/'resource_protocol_v3.json'
MODEL_DIR=ROOT/'analysis/adaptive_lr097/model';CONTROLLER_DIR=ROOT/'analysis/adaptive_lr097'
MODEL_PROOF=CONTROLLER_DIR/'ALZ_integration_CPU_qualification.json'
NATIVE_PROOF=ROOT/'analysis/distribution097/native_release.json';ENGINE_PROOF=ROOT/'analysis/shared_gpu0_candidate_CPU_qualification.json'
ENDPOINT_PROOF=STUDY/'endpoint_CPU_qualification.json'
RUNTIME_PROOF=STUDY/'runtime_CPU_qualification_v7.json';WINDOW=ROOT/'analysis/research_window_20261104_five_days.json'
CFG=dict(z_dim=8,stage_balance=False,cell_z_dim=8,theta_init_scale=1.,fourier_feature_scale=1.)
ENABLED=dict(enabled=True,experiment='scalarcal097',fold=0,training_seeds=[5,6,7],arms=['fixed','adaptive'],endpoint=20000,all_training_from_scratch=True,calibration_only=True)
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def cell(arm='fixed',seed=5):
 assert arm in ('fixed','adaptive') and type(seed) is int and seed in (5,6,7),'unregistered fresh scalar calibration cell'
 return dict(experiment='scalarcal097_'+arm,fold=0,method='pg',seed=seed)
def arm_of(value):
 arm=value['experiment'].removeprefix('scalarcal097_');assert value==cell(arm,value['seed']);return arm
def endpoint(method):assert method=='pg';return 20000
def update_count(value):arm_of(value);return 20000
def update_unit(method):assert method=='pg';return 'optimizer_pairs'
def protocol():
 p=read(PROTOCOL)
 assert p['status']=='FROZEN_SEPARATE_SCALAR_CALIBRATION_COMPARISON_BEFORE_REAL_VALUES'
 assert p['native_version']=='0.9.7' and p['training_seeds']==[5,6,7] and p['arms']==['fixed','adaptive']
 assert p['folds']==[0] and p['updates_per_fit']==20000 and p['required_fresh_fits']==6 and p['cap_seconds']==21600
 assert p['all_from_scratch'] and p['initial_completed_updates']==0 and p['source_checkpoint'] is None and p['calibration_only']
 assert p['decision_metric']=='calibration_cell_energy_distance' and not p['existing45_reselection'] and not p['SOTA_claim']
 assert p['controller']==dict(interval=1000,warmup=2000,window=3,patience=3,cooldown=1,factor=.5,min_scale=.125,relative_improvement=.01)
 assert p['hyperparameters']==dict(generator_lr=.012,critic_lr=.018,particle_lr=.06,reg_coeff=1.,reg_kappa=1.,prior_reg=.01)
 for path,pin in p['fidelity_evidence_SHA256'].items():assert sha(path)==pin
 q=read(MODEL_PROOF);assert q['qualified_sources_unchanged_during_execution'] and q['tests_run']==6 and q['parity_actual_optimizer_pairs_total']==30 and not q['CUDA_initialized'] and q['real_ALZ_adaptive_fits']==0
 for path,pin in q['source_SHA256'].items():assert sha(path)==pin
 endpoint=read(ENDPOINT_PROOF)
 assert endpoint['status']=='FULL2000GENE_SYNTHETIC_CPU_SCALAR_AND_ENDPOINT_POOL_PARITY_PASS' and endpoint['endpoint_primary_reservoir_bitwise_equal_to_scalar_replicate0'] and not endpoint['real_ALZ_target_values_read'] and not endpoint['CUDA_initialized']
 for path,pin in endpoint['source_SHA256'].items():assert sha(path)==pin
 return p
def hyperparameters():return protocol()['hyperparameters']
def proof_source_pins():
 pins={}
 for proof in (MODEL_PROOF,ENGINE_PROOF):
  for path,pin in read(proof)['source_SHA256'].items():
   if path in pins:assert pins[path]==pin
   pins[path]=pin
 return pins
def handoff_paths(root=ROOT):
 root=Path(root)
 if (root/'analysis/benchmark097_infra1/scientific_reports_completed.json').exists():
  from scalarcal097_recovered_handoff import paths
  return paths(root)
 common=[root/'analysis/benchmark097/matrix_complete.json',root/'analysis/benchmark097/comparison.json',root/'analysis/benchmark097_calibration_excluded53_report.json']
 barrier=root/'analysis/benchmark097_balanced/scientific_reports_completed.json'
 if barrier.exists():return [*common,barrier,root/'artifacts/released097-balanced-research-pipeline/completed.json']
 return [*common,root/'artifacts/released097-dual-research-pipeline-v2/completed.json',root/'artifacts/released097-dual-calibration53-boundary-v2/completed.json']
def prior_study_ready(root=ROOT):
 """Accept either qualified resource lineage; hash, never parse target reports."""
 root=Path(root)
 if (root/'analysis/benchmark097_infra1/scientific_reports_completed.json').exists():
  from scalarcal097_recovered_handoff import ready
  return ready(root)
 matrix,primary,sensitivity,parent,child=handoff_paths(root)
 paths=(matrix,primary,sensitivity,parent,child)
 assert all(p.is_file() for p in paths),'All45 and both scientific reports must finish before scalar calibration'
 q=read(matrix);assert q['matrix_complete'] and q['complete_cells']==q['required_cells']==45 and len(q['result_SHA256'])==45
 if parent==root/'analysis/benchmark097_balanced/scientific_reports_completed.json':
  barrier=read(parent);assert barrier['status']=='SAME_FROZEN45_ORIGINAL57_PRIMARY_AND_PRESPECIFIED53_COMPLETE' and barrier['complete_fresh_models']==45 and barrier['original57_primary_preserved'] and barrier['prespecified53_complete'] and barrier['new_generated_counts_for53']==0
  source=root/'analysis/run_released097_balanced_pipeline.py';proof_path=Path(barrier.get('runtime_qualification_path',str(root/'analysis/benchmark097_balanced/runtime_CPU_qualification.json')))
  assert proof_path in (root/'analysis/benchmark097_balanced/runtime_CPU_qualification.json',root/'analysis/benchmark097_balanced/runtime_CPU_qualification_v2.json')
  proof=read(proof_path)
  assert proof['status']=='EXTERNAL097_SAME45_BALANCED_RESOURCE_SCOPE_CPU_PASS' and proof['native_cell_device_bindings']==90
  assert barrier['source_pipeline_path']==str(source) and barrier['source_pipeline_SHA256']==proof['source_SHA256'][str(source)]==sha(source)
  assert barrier['runtime_qualification_SHA256']==sha(proof_path) and barrier['handoff_SHA256']==sha(root/'analysis/benchmark097_balanced/handoff_completed.json')
  assert barrier['report_SHA256']=={str(p):sha(p) for p in (matrix,primary,sensitivity)}
  assert read(child)['status']=='SAME45_SCIENTIFIC_REPORTS_READY_FINAL_AGENT_AUDIT_AND_SCALAR_CALIBRATION_REMAIN'
  assert not read(root/'benchmark097_balanced_runtime/plan.json')['enabled']
  paths=(*paths,source,proof_path,root/'analysis/benchmark097_balanced/handoff_completed.json')
 else:
  assert read(parent)['status']=='FULL_RESEARCH_PIPELINE_COMPARISON_READY_FOR_FINAL_AGENT_AUDIT'
  c=read(child);assert c['status']=='CALIBRATION_EXCLUDED53_SENSITIVITY_COMPLETE_FINAL_AGENT_AUDIT_REMAINS' and c['original57_primary_preserved']
 assert not read(root/'benchmark097_dual_runtime/plan.json')['enabled']
 for namespace in ('benchmark097-dual','benchmark097-balanced'):
  for index in (0,1):
   registry=root/f'artifacts/gpu{index}-{namespace}-20261107/live_registry/registry.json'
   assert not registry.exists() or not read(registry)['jobs'],'Previous physical registry is still active'
 return {str(p):sha(p) for p in paths}

def protocol_paths():
 return [ROOT/'analysis/scalarcal097_recovered_handoff.py',ROOT/'analysis/released097_remaining_runtime_balance.py',ROOT/'analysis/run_scalarcal097.py',ROOT/'analysis/after_benchmark097_scalarcal.py',ROOT/'analysis/aggregate_scalarcal097.py',PROTOCOL,RESOURCE_PROTOCOL,PLAN,MODEL_PROOF,ENDPOINT_PROOF,NATIVE_PROOF,ENGINE_PROOF,WINDOW,CONTROLLER_DIR/'published_integration_sources.json',*map(Path,protocol()['fidelity_evidence_SHA256']),ROOT/'benchmark097_runtime/resource_profile.py',*[ROOT/f'benchmark097_dual_runtime/policy.gpu{i}.prepared.json' for i in (0,1)]]
def validate_authorization(auth,*,require_runtime_proof=True,require_activation=True,require_handoff=True):
 p=protocol();v=auth['cell'];assert v==cell(auth['arm'],v['seed'])
 assert auth['operation']=='scalarcal097_cell' and auth['metric_role']=='calibration_selection' and auth['role']=='alz_core_pg'
 assert auth['fixed_endpoint']==auth['new_model_updates']==20000 and auth['initial_completed_updates']==0 and auth['new_model_update_unit']=='optimizer_pairs'
 assert not auth['reuse_completed_model'] and not auth['is_continuation'] and auth['source_model_path'] is auth['source_model_SHA256'] is None
 assert auth['cap_seconds']==21600 and auth['benchmark_protocol_SHA256']==sha(PROTOCOL)
 assert auth['released_hyperparameters']==p['hyperparameters'] and auth['variant_config']==CFG
 assert auth['particlegan_version']=='0.9.7' and auth['penalty_options']=={'input_domain':'continuous'} and not auth['cpu_fixture']
 assert auth['independent_current_device_review_claimed'] is False and auth['source_review_type']=='OWNER_AUTHORIZED_VERIFIED_SOURCE_AND_RESOURCE_CHECKS'
 if require_activation:assert read(PLAN)==ENABLED,'Separate experiment plan remains disabled'
 assert sha(WINDOW)=='2dffe7ce5956cac37fd212e5b0b02b145acff98e35b75700855c98cda57dfbc1'
 assert math.isfinite(auth['issued_unix']) and auth['expires_unix']<=min(read(WINDOW)['expires_unix'],auth['issued_unix']+21630)
 from dual_coordination import GPUS
 assert auth['gpu_uuid'] in GPUS and auth['dynamic_shared_gpu0']==(auth['gpu_uuid']=='GPU-72c1b506-891d-b8bc-b353-e020585e1c47')
 pins=auth['files_SHA256'];research=auth['research_source_SHA256']
 for path in protocol_paths():assert pins[str(path)]==sha(path),str(path)
 for path,pin in proof_source_pins().items():assert pins[path]==pin==sha(path)
 for path,pin in read(NATIVE_PROOF)['official_wheel_source_SHA256'].items():assert pins[path]==pin==sha(path)
 for folder in (ROOT,ROOT/'model',HERE,MODEL_DIR,CONTROLLER_DIR,ROOT/'analysis/faithful097/model'):
  for path in folder.glob('*.py'):assert research[str(path)]==pins[str(path)]==sha(path)
 if require_handoff:
  for path,pin in prior_study_ready().items():assert pins[path]==pin
 if require_runtime_proof:
  assert auth['runtime_qualification_SHA256']==pins[str(RUNTIME_PROOF)]==sha(RUNTIME_PROOF)
  q=read(RUNTIME_PROOF);assert q['status']=='SCALARCAL097_SIX_FRESH_CELLS_PREPARED_SCOPE_ACCOUNTING_CPU_PASS' and q['native_cells_checked']==6 and q['native_cell_device_bindings']==12 and not q['GPU_allocation_created']
  for path,pin in q['source_SHA256'].items():assert pins[path]==pin==sha(path)
 return p
