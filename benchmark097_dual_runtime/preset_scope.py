"""All45 benchmark fits are fresh; an unfrozen PG candidate cannot be admitted."""
import hashlib,json,math
from pathlib import Path
import sys
ROOT=Path('/ml2/alz');HERE=Path(__file__).resolve().parent;STUDY=ROOT/'analysis/benchmark097'
PROTOCOL=STUDY/'protocol.json';PLAN=HERE/'plan.json';MODEL_DIR=ROOT/'analysis/faithful097/model'
CFGEN_DIR=ROOT/'analysis/benchmark097_cfgen'
MODEL_PROOF=ROOT/'analysis/faithful097/native_trainer_parity_CPU.json'
NATIVE_PROOF=ROOT/'analysis/distribution097/native_release.json'
ENGINE_PROOF=ROOT/'analysis/shared_gpu0_candidate_CPU_qualification.json'
RUNTIME_PROOF=STUDY/'dual_runtime_CPU_qualification.json'
WINDOW=ROOT/'analysis/research_window_20261104_five_days.json'
sys.path.insert(0,str(ROOT/'analysis'))
from released097_selection import FORMULATION
from released097_external_selection import resolve as resolve_selection
SELECTION_LOGIC=ROOT/'analysis/released097_selection.py';SELECTION_VALIDATOR=ROOT/'analysis/released097_external_selection.py'
CFG={'z_dim':8,'stage_balance':False,'cell_z_dim':8,'theta_init_scale':1.,'fourier_feature_scale':1.}
ENABLED={'enabled':True,'benchmark':'external097','folds':[0,1,2],'training_seeds':[0,1,2,3,4],
         'methods':['pg','scvi','cfgen'],'PG_updates':20000,'scVI_epochs':400,'CFGen_AE_epochs':300,'CFGen_FM_epochs':1500,'all_training_from_scratch':True}
PROOFS=[MODEL_PROOF,ENGINE_PROOF,
        ROOT/'analysis/matched_scvi/CPU_model_qualification.json',
        ROOT/'analysis/external_core_scvi_model_v096_CPU.json',
        CFGEN_DIR/'CPU_model_qualification.json',CFGEN_DIR/'CPU_saved_sampling_qualification.json',CFGEN_DIR/'CPU_exact_reload_qualification.json',
        STUDY/'fit_evidence_CPU_qualification_v2.json']
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def cell(fold,method,seed):
 assert type(fold) is int and fold in range(3) and method in ('pg','scvi','cfgen') and type(seed) is int and seed in range(5)
 return dict(experiment='external097',fold=fold,method=method,seed=seed)
def endpoint(method):return {'pg':20000,'scvi':400,'cfgen':1800}[method]
def update_count(value):
 if value['method']!='cfgen':return endpoint(value['method'])
 p=ROOT/'analysis/benchmark097_cfgen/CPU_model_qualification.json'
 fold=next(q for q in read(p)['all_frozen_fold_training_inputs'] if q['fold']==value['fold'])
 return fold['AE300_updates']+fold['FM1500_updates']
def update_unit(method):return {'pg':'optimizer_pairs','scvi':'epochs','cfgen':'optimizer_steps'}[method]
def protocol():
 p=read(PROTOCOL)
 assert p['status']=='FROZEN_NATIVE097_ALL45_FRESH_MODELS_BEFORE_NEW_EVALUATION','PG selection/confirmation must be frozen before benchmark admission'
 assert p['all_new_training_from_scratch'] and p['initial_completed_updates']==0 and not p['reuse_historical_models']
 assert p['native_version']=='0.9.7' and p['folds']==list(range(3)) and p['training_seeds']==list(range(5)) and p['methods']==['pg','scvi','cfgen']
 assert p['PG_fresh_seed_confirmation_complete'] and p['PG_selected_before_new_heldout_evaluation'] and not p['PG_heldout_results_used_for_selection']
 assert p['PG_formulation']==dict(recipe='bispf_optimized',EMA_decay=0,generator_gradient_response='none',penalty_input_domain='continuous',architecture='unchanged PG8 count generator and150D VectorCritic',prior_regularizer_domain='all512rows',critic_thermal='explicitly_bound')
 assert p['endpoints']==dict(pg=20000,scvi=400,cfgen=dict(AE_epochs=300,FM_epochs=1500))
 assert p['cap_seconds']==21600 and not p['automatic_promotion'] and not p['SOTA_claim']
 assert p['retain_full_generated_stage_PCs'] is True
 assert p['PG_frozen_selection_evidence_SHA256']
 for path,pin in p['PG_frozen_selection_evidence_SHA256'].items():assert Path(path).is_absolute() and sha(path)==pin
 audit=read(p['PG_confirmation_audit_path']);confirmation=read(p['PG_confirmation_protocol_path']);report=read(p['PG_confirmation_report_path'])
 selection=read(p['PG_selection_audit_path'])
 choice=resolve_selection(selection,audit,confirmation)
 assert audit['report_SHA256']==sha(p['PG_confirmation_report_path']) and audit['protocol_SHA256']==sha(p['PG_confirmation_protocol_path'])
 for q in (selection,audit):
  assert q['model_parity_proof_path']==str(MODEL_PROOF) and q['model_parity_proof_SHA256']==sha(MODEL_PROOF)
  for path,pin in q['source_SHA256'].items():assert sha(path)==pin
 for key in ('PG_selection_audit_path','PG_confirmation_audit_path','PG_confirmation_protocol_path','PG_confirmation_report_path'):
  assert p['PG_frozen_selection_evidence_SHA256'][p[key]]==sha(p[key])
 assert p['PG_candidate_id']==choice['candidate_id'] and p['PG_hyperparameters']==choice['hyperparameters']

 return p
def hyperparameters():
 h=protocol()['PG_hyperparameters']
 assert set(h)=={'generator_lr','critic_lr','particle_lr','reg_coeff','reg_kappa','prior_reg'}
 assert all(type(v) in (int,float) and math.isfinite(v) and v>0 for v in h.values())
 return h
def protocol_paths(p):
 return [ROOT/'analysis/run_released097_dual_pipeline.py',ROOT/'analysis/after_released097_dual_calibration53.py',ROOT/'analysis/run_benchmark097_dual.py',ROOT/'analysis/aggregate_benchmark097_dual.py',ROOT/'analysis/score_calibration_excluded53_dual.py',ROOT/'analysis/benchmark097_dual_analysis_CPU_preparation.json',ROOT/'benchmark097_runtime/resource_profile.py',HERE/'policy.prepared.json',HERE/'policy.gpu0.prepared.json',HERE/'policy.gpu1.prepared.json',ROOT/'analysis/benchmark097_dual_resource_protocol.json',PROTOCOL,NATIVE_PROOF,WINDOW,PLAN,SELECTION_LOGIC,SELECTION_VALIDATOR,*PROOFS,CFGEN_DIR/'protocol.json',
         ROOT/'analysis/particlegan097_runtime_equivalence_20261106.json',
         ROOT/'analysis/distribution097_external_evaluation_plan_20261106.json',
         ROOT/'analysis/distribution097_split_isolation_audit_20261106.json',
         ROOT/'analysis/distribution_primary_statistics.py',
         *[Path(x) for x in p['PG_frozen_selection_evidence_SHA256']]]
def proof_source_pins():
 pins={}
 for proof in PROOFS:
  for path,pin in read(proof)['source_SHA256'].items():
   if path in pins:assert pins[path]==pin
   pins[path]=pin
 return pins
def validate_authorization(auth,*,require_runtime_proof=True):
 p=protocol();v=auth['cell'];assert v==cell(v['fold'],v['method'],v['seed'])
 assert auth['role']=='alz_core_'+v['method']
 assert auth['operation']=='external097_benchmark_cell' and auth['metric_role']=='final_evaluation'
 assert auth['fixed_endpoint']==endpoint(v['method']) and auth['new_model_updates']==update_count(v)
 assert auth['new_model_update_unit']==update_unit(v['method']) and auth['initial_completed_updates']==0
 assert not auth['reuse_completed_model'] and not auth['is_continuation'] and auth['source_model_path'] is auth['source_model_SHA256'] is None
 assert auth['cap_seconds']==21600 and auth['benchmark_protocol_SHA256']==sha(PROTOCOL)
 assert read(PLAN)==ENABLED and sha(WINDOW)=='2dffe7ce5956cac37fd212e5b0b02b145acff98e35b75700855c98cda57dfbc1'
 assert auth['particlegan_version']=='0.9.7' and auth['variant_config']==CFG and auth['released_hyperparameters']==hyperparameters()
 assert auth['penalty_options']=={'input_domain':'continuous'} and not auth['cpu_fixture']
 assert auth['independent_current_device_review_claimed'] is False and auth['source_review_type']=='OWNER_AUTHORIZED_VERIFIED_SOURCE_AND_RESOURCE_CHECKS'
 from dual_coordination import assigned_gpu
 assert auth['gpu_uuid']==assigned_gpu(v) and auth['dynamic_shared_gpu0']==(auth['gpu_uuid']=='GPU-72c1b506-891d-b8bc-b353-e020585e1c47')
 assert math.isfinite(auth['issued_unix']) and auth['expires_unix']<=min(read(WINDOW)['expires_unix'],auth['issued_unix']+21630)
 pins=auth['files_SHA256'];research=auth['research_source_SHA256']
 for path in protocol_paths(p):assert pins[str(path)]==sha(path)
 proof=read(MODEL_PROOF);assert proof['status']=='EXACT_RELEASED_GANTRAINER_UPDATE_PARITY_ON_FULLWIDTH_CONDITIONAL_FIXTURE_PASS' and proof['actual_cases']==4 and proof['actual_optimizer_pairs_per_implementation']==12 and not proof['native_package_modified'] and proof['prior_regularizer_domain']=='all512rows' and proof['explicit_bound_critic_thermal']
 assert read(NATIVE_PROOF)['version']=='0.9.7' and len(read(NATIVE_PROOF)['official_wheel_source_SHA256'])==69
 eq=read(ROOT/'analysis/particlegan097_runtime_equivalence_20261106.json');assert eq['all_core_runtime_files_byte_identical'] and eq['dependency_requirements_unchanged']
 for name in ('released_update.py','released_adapter.py','release_env.py'):assert sha(MODEL_DIR/name)==proof['source_SHA256'][str(MODEL_DIR/name)]
 for path,pin in proof_source_pins().items():assert pins[path]==pin==sha(path)
 for path,pin in read(NATIVE_PROOF)['official_wheel_source_SHA256'].items():assert pins[path]==pin==sha(path)
 for folder in (ROOT,ROOT/'model',HERE,MODEL_DIR,CFGEN_DIR):
  for path in folder.glob('*.py'):assert research[str(path)]==pins[str(path)]==sha(path)
 if require_runtime_proof:
  assert auth['runtime_qualification_SHA256']==pins[str(RUNTIME_PROOF)]==sha(RUNTIME_PROOF)
  q=read(RUNTIME_PROOF);assert q['status']=='EXTERNAL097_ALL45_FRESH_CELLS_SCOPE_ACCOUNTING_CPU_PASS' and not q['GPU_allocation_created']
  assert q['actual_tests']==43 and q['two_physical_policy_profiles'] and q['actual_two_concurrent_confined_CPU_guard_workers'] and q['actual_driver_failure_drain_and_observation_timeout_tests']
  for path,pin in q['source_SHA256'].items():assert pins[path]==pin==sha(path)
 return p
