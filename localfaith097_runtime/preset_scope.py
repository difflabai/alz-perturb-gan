"""Four fresh20k calibration fits after exact released trainer objective parity."""
import hashlib,json,math
from pathlib import Path
ROOT=Path('/ml2/alz');HERE=Path(__file__).resolve().parent;STUDY=ROOT/'analysis/localfaith097'
PROTOCOL=STUDY/'protocol.json';PLAN=HERE/'plan.json'
MODEL_PROOF=ROOT/'analysis/faithful097/native_trainer_parity_CPU.json';HPO_PROOF=MODEL_PROOF;GRID_MODEL_PROOF=STUDY/'possible_settings_native_CPU_qualification.json';SCALAR_PROOF=GRID_MODEL_PROOF
NATIVE_PROOF=STUDY/'native_release.json';ENGINE_PROOF=ROOT/'analysis/shared_gpu0_candidate_CPU_qualification.json'
RUNTIME_PROOF=STUDY/'runtime_CPU_qualification.json';WINDOW=ROOT/'analysis/research_window_20261104_five_days.json'
DESIGN=ROOT/'analysis/released097_local_search_design.py';SELECTOR=STUDY/'selection.py';BASELINE_PROOF=ROOT/'analysis/faithful097/resampling_controls_CPU.json'
CFG={'z_dim':8,'stage_balance':False,'cell_z_dim':8,'theta_init_scale':1.,'fourier_feature_scale':1.}
MODEL_DIR=ROOT/'analysis/faithful097/model'
import sys
sys.path.insert(0,str(ROOT/'analysis'))
from released097_selection import FORMULATION
from released097_local_search_design import design as resolve_design
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
ENABLED=dict(enabled=True,experiment='localfaith097',native_version='0.9.7',formulation='released097_full_table_sbnp_nonEMA',endpoint=20000,selection_seeds=[0,1],candidate_ids=[q['id'] for q in read(PROTOCOL)['candidates']],owned_GPU_GiB=2,calibration_only=True,all_training_from_scratch=True)
def validate_protocol():
 p=read(PROTOCOL)
 assert p['status']=='FROZEN_RELEASED097_FULL20K_LOCAL_COORDINATE_SEARCH','Complete corrected baseline audit and frozen local design required before admission'
 assert p['native_version']=='0.9.7' and p['endpoint']==20000 and p['selection_seeds']==[0,1] and p['fresh_confirmation_seeds']==[2,3,4]
 assert p['all_new_training_from_scratch'] and p['initial_completed_updates']==0 and p['source_checkpoint'] is None and p['calibration_only'] and p['metric_role']=='calibration_selection'
 assert p['formulation']==FORMULATION and not p['automatic_promotion'] and not p['SOTA_claim'] and not p['full_goal_complete']
 assert p['design_evidence_SHA256']
 for path,pin in p['design_evidence_SHA256'].items():assert sha(path)==pin
 audit=read(p['baseline_audit_path']);assert audit['status']=='INDEPENDENT_RELEASED097_FRESH_BASELINE4_AUDIT_PASS'
 assert audit['formulation']==FORMULATION and audit['all_planned_attempts_resolved'] and audit['all_new_models_from_update_zero']
 assert audit['training_seeds']==[0,1] and audit['activation_actually_reverted'] and audit['owned_processes_and_leases_absent'] and not audit['heldout_values_read']
 assert audit['model_parity_proof_path']==str(MODEL_PROOF) and audit['model_parity_proof_SHA256']==sha(MODEL_PROOF)
 frozen=read(STUDY/'frozen_design.json')
 expected=resolve_design(audit,p['design_now_unix'],read(WINDOW)['expires_unix'],read(p['matrix_cost_forecast_path'])['estimated_matrix_GPU_hours'])
 assert frozen==expected and frozen['status']=='PROPOSED_WINDOW_BOUNDED_LOCAL_COORDINATE_SEARCH'
 assert p['candidates']==frozen['candidates'] and 2<=len(p['candidates'])<=6 and p['maximum_new_fits']==2*len(p['candidates'])
 assert [q['id'] for q in p['candidates']]==ENABLED['candidate_ids']
 controls=p['reference_controls'];assert len(controls)==4 and {(q['candidate_id'],q['seed']) for q in controls}=={(c,s) for c in ('f000','f001') for s in (0,1)}
 for q in controls:
  original=read(q['source_result_path']);assert sha(q['source_result_path'])==q['source_result_SHA256']
  assert q==dict(original,read_only_previously_measured_reference=True,new_model_fitted=False,new_GPU_charge=0,source_result_path=q['source_result_path'],source_result_SHA256=q['source_result_SHA256'])
  assert audit['result_SHA256'][q['attempt']]==q['source_result_SHA256'] and q['from_scratch'] and q['physical_start_update']==0 and q['finite']
 grid=read(GRID_MODEL_PROOF);assert grid['status']=='ALL24_POSSIBLE_LOCAL_SETTING_SEED_NATIVE_TRAINER_PREFIX_CASES_PASS' and grid['actual_cases']==24 and grid['actual_optimizer_pairs_per_implementation']==72 and not grid['CUDA_initialized'] and not grid['real_ALZ_data_read'] and not grid['native_package_modified']
 for path,pin in grid['source_SHA256'].items():assert sha(path)==pin
 for candidate in p['candidates']:
  for seed in (0,1):
   entry=next(q for q in grid['rows'] if q['anchor_id']==frozen['anchor_candidate_id'] and q['candidate_id']==candidate['id'] and q['seed']==seed)
   assert entry['hyperparameters']==candidate['hyperparameters'] and entry['optimizer_pairs_per_loop']==3 and entry['exact_models_optimizers_rng_and_update_clock'] and entry['exact_all_losses_and_penalty'] and entry['from_scratch']
 proof=read(MODEL_PROOF);assert proof['status']=='EXACT_RELEASED_GANTRAINER_UPDATE_PARITY_ON_FULLWIDTH_CONDITIONAL_FIXTURE_PASS' and proof['prior_regularizer_domain']=='all512rows' and proof['explicit_bound_critic_thermal'] and not proof['native_package_modified']
 for path,pin in proof['source_SHA256'].items():assert sha(path)==pin
 return p
def cell(candidate='t000',seed=0,phase='search'):
 validate_protocol();assert candidate in ENABLED['candidate_ids'] and type(seed) is int and seed in (0,1) and phase=='search'
 return dict(experiment='lf097_'+candidate,fold=0,method='pg',seed=seed)
def proposal_path(cid):assert cid in ENABLED['candidate_ids'];return STUDY/'proposals'/(cid+'.json')
def validate_proposal(q):
 p=validate_protocol();assert q['id'] in ENABLED['candidate_ids'] and q['native_version']=='0.9.7' and q['endpoint']==20000 and q['protocol_SHA256']==sha(PROTOCOL)
 assert q['hyperparameters']==next(c['hyperparameters'] for c in p['candidates'] if c['id']==q['id'])
 assert q['fold']==q['initial_completed_updates']==0 and q['training_seeds']==[0,1] and q['from_scratch'] and q['source_checkpoint'] is None and not q['heldout_results_used']
 return q['hyperparameters']
def protocol_paths(auth):
 p=read(PROTOCOL)
 return [PROTOCOL,PLAN,MODEL_PROOF,GRID_MODEL_PROOF,NATIVE_PROOF,ENGINE_PROOF,BASELINE_PROOF,WINDOW,DESIGN,SELECTOR,STUDY/'frozen_design.json',proposal_path(auth['candidate_id']),ROOT/'analysis/particlegan097_runtime_equivalence_20261106.json',*[Path(x) for x in p['design_evidence_SHA256']]]
def validate_authorization(auth,*,require_runtime_proof=True):
 p=validate_protocol();v=auth['cell'];assert v==cell(auth['candidate_id'],v['seed'],auth['phase'])
 assert auth['released_hyperparameters']==validate_proposal(read(proposal_path(auth['candidate_id']))) and auth['variant_config']==CFG
 assert auth['proposal_SHA256']==sha(proposal_path(auth['candidate_id'])) and auth['benchmark_protocol_SHA256']==sha(PROTOCOL)
 assert auth['operation']=='localfaith097_cell' and auth['metric_role']=='calibration_selection' and auth['role']=='alz_core_pg'
 assert auth['fixed_endpoint']==auth['new_model_updates']==20000 and auth['initial_completed_updates']==0
 assert auth['is_continuation'] is False and auth['reuse_completed_model'] is False and auth['source_model_path'] is auth['source_model_SHA256'] is None
 assert auth['cap_seconds']==21600 and read(PLAN)==ENABLED
 assert auth['particlegan_version']=='0.9.7' and auth['penalty_options']=={'input_domain':'continuous'}
 assert not auth['cpu_fixture'] and not auth['independent_current_device_review_claimed'] and auth['source_review_type']=='OWNER_AUTHORIZED_VERIFIED_SOURCE_AND_RESOURCE_CHECKS'
 assert auth['gpu_uuid']=='GPU-cb4ce47d-d968-bffd-5646-e830a9fa1c69' and auth['dynamic_shared_gpu0'] is False
 assert sha(WINDOW)=='2dffe7ce5956cac37fd212e5b0b02b145acff98e35b75700855c98cda57dfbc1'
 assert math.isfinite(auth['issued_unix']) and auth['expires_unix']<=min(read(WINDOW)['expires_unix'],auth['issued_unix']+21630)
 pins=auth['files_SHA256'];research=auth['research_source_SHA256']
 for path in protocol_paths(auth):assert pins[str(path)]==sha(path)
 eq=read(ROOT/'analysis/particlegan097_runtime_equivalence_20261106.json');assert eq['all_core_runtime_files_byte_identical'] and eq['dependency_requirements_unchanged']
 baseline=read(BASELINE_PROOF);assert baseline['status']=='TWO_MATCHED_SELECTION_SEED_RESAMPLING_CONTROLS_CPU_VERIFIED' and set(baseline['metrics_by_seed'])=={'0','1'} and not baseline['CUDA_initialized']
 for evidence in (read(MODEL_PROOF)['source_SHA256'],read(GRID_MODEL_PROOF)['source_SHA256'],read(NATIVE_PROOF)['official_wheel_source_SHA256'],read(ENGINE_PROOF)['source_SHA256'],baseline['source_SHA256'],baseline['files_SHA256']):
  for path,pin in evidence.items():assert pins[path]==pin==sha(path)
 for folder in (ROOT,ROOT/'model',HERE,MODEL_DIR,STUDY):
  for path in folder.glob('*.py'):assert research[str(path)]==pins[str(path)]==sha(path)
 if require_runtime_proof:
  assert auth['runtime_qualification_SHA256']==pins[str(RUNTIME_PROOF)]==sha(RUNTIME_PROOF)
  q=read(RUNTIME_PROOF);assert q['status']=='RELEASED097_LOCAL_COORDINATE_SCOPE_ACCOUNTING_CPU_PASS' and q['actual_cells_checked']==2*len(p['candidates']) and not q['GPU_allocation_created']
  for path,pin in q['source_SHA256'].items():assert pins[path]==pin==sha(path)
 return p
