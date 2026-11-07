"""Four fresh20k calibration fits after exact released trainer objective parity."""
import hashlib,json,math
from pathlib import Path
ROOT=Path('/ml2/alz');HERE=Path(__file__).resolve().parent;STUDY=ROOT/'analysis/faithful097'
PROTOCOL=STUDY/'protocol.json';PLAN=HERE/'plan.json';MODEL_DIR=STUDY/'model'
MODEL_PROOF=STUDY/'native_trainer_parity_CPU.json';HPO_PROOF=MODEL_PROOF;SCALAR_PROOF=MODEL_PROOF
NATIVE_PROOF=STUDY/'native_release.json';ENGINE_PROOF=ROOT/'analysis/shared_gpu0_candidate_CPU_qualification.json'
RUNTIME_PROOF=STUDY/'runtime_CPU_qualification.json';WINDOW=ROOT/'analysis/research_window_20261104_five_days.json'
DESIGN=PROTOCOL;SELECTOR=STUDY/'native_trainer_parity.py';BASELINE_PROOF=STUDY/'resampling_controls_CPU.json'
CFG={'z_dim':8,'stage_balance':False,'cell_z_dim':8,'theta_init_scale':1.,'fourier_feature_scale':1.}
IDS=['f000','f001']
ENABLED=dict(enabled=True,experiment='faithful097',native_version='0.9.7',formulation='released097_full_table_sbnp_nonEMA',endpoint=20000,selection_seeds=[0,1],candidate_ids=IDS,owned_GPU_GiB=2,calibration_only=True,all_training_from_scratch=True)
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def validate_protocol():
 p=read(PROTOCOL)
 assert p['status']=='FROZEN_FRESH_RELEASED_OBJECTIVE_RESTART_BASELINE'
 assert p['native_version']=='0.9.7' and p['endpoint']==20000 and p['selection_seeds']==[0,1] and p['maximum_new_fits']==4
 assert p['all_new_training_from_scratch'] and p['initial_completed_updates']==0 and p['source_checkpoint'] is None and p['calibration_only'] and p['metric_role']=='calibration_selection'
 assert not p['automatic_promotion'] and not p['SOTA_claim'] and not p['full_goal_complete']
 assert p['prior_regularizer_domain']=='all512rows' and p['thermal_source']=='bound_critic_optimizer_record'
 assert [c['id'] for c in p['candidates']]==IDS
 for path,pin in p['design_evidence_SHA256'].items():assert sha(path)==pin
 proof=read(MODEL_PROOF)
 assert proof['status']=='EXACT_RELEASED_GANTRAINER_UPDATE_PARITY_ON_FULLWIDTH_CONDITIONAL_FIXTURE_PASS' and proof['native_version']=='0.9.7'
 assert proof['actual_cases']==4 and proof['actual_optimizer_pairs_per_implementation']==12 and proof['prior_regularizer_domain']=='all512rows' and proof['explicit_bound_critic_thermal']
 assert not proof['CUDA_initialized'] and not proof['native_package_modified'] and not proof['real_ALZ_data_read']
 for path,pin in proof['source_SHA256'].items():assert sha(path)==pin
 expected=[dict(generator_lr=.012,critic_lr=.018,particle_lr=.03,reg_coeff=1.,reg_kappa=1.,prior_reg=.01),dict(generator_lr=.006,critic_lr=.009,particle_lr=.015,reg_coeff=1.,reg_kappa=.5,prior_reg=.01)]
 assert [c['hyperparameters'] for c in p['candidates']]==expected
 for h in expected:
  for seed in (0,1):
   q=next(r for r in proof['rows'] if r['seed']==seed and r['hyperparameters']==h)
   assert q['from_scratch'] and q['exact_models_optimizers_rng_and_update_clock'] and q['exact_all_losses_and_penalty']
 return p
def cell(candidate='f000',seed=0,phase='search'):
 validate_protocol();assert candidate in IDS and type(seed) is int and seed in (0,1) and phase=='search'
 return dict(experiment='faith097_'+candidate,fold=0,method='pg',seed=seed)
def proposal_path(cid):assert cid in IDS;return STUDY/'proposals'/(cid+'.json')
def validate_proposal(q):
 p=validate_protocol();assert q['id'] in IDS and q['native_version']=='0.9.7' and q['endpoint']==20000 and q['protocol_SHA256']==sha(PROTOCOL)
 assert q['hyperparameters']==next(c['hyperparameters'] for c in p['candidates'] if c['id']==q['id'])
 assert q['fold']==q['initial_completed_updates']==0 and q['training_seeds']==[0,1] and q['from_scratch'] and q['source_checkpoint'] is None and not q['heldout_results_used']
 return q['hyperparameters']
def protocol_paths(auth):
 p=read(PROTOCOL)
 return [PROTOCOL,PLAN,MODEL_PROOF,NATIVE_PROOF,ENGINE_PROOF,BASELINE_PROOF,WINDOW,SELECTOR,proposal_path(auth['candidate_id']),ROOT/'analysis/particlegan097_runtime_equivalence_20261106.json',*[Path(x) for x in p['design_evidence_SHA256']]]
def validate_authorization(auth,*,require_runtime_proof=True):
 p=validate_protocol();v=auth['cell'];assert v==cell(auth['candidate_id'],v['seed'],auth['phase'])
 assert auth['released_hyperparameters']==validate_proposal(read(proposal_path(auth['candidate_id']))) and auth['variant_config']==CFG
 assert auth['proposal_SHA256']==sha(proposal_path(auth['candidate_id'])) and auth['benchmark_protocol_SHA256']==sha(PROTOCOL)
 assert auth['operation']=='faithful097_cell' and auth['metric_role']=='calibration_selection' and auth['role']=='alz_core_pg'
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
 for evidence in (read(MODEL_PROOF)['source_SHA256'],read(NATIVE_PROOF)['official_wheel_source_SHA256'],read(ENGINE_PROOF)['source_SHA256'],baseline['source_SHA256'],baseline['files_SHA256']):
  for path,pin in evidence.items():assert pins[path]==pin==sha(path)
 for folder in (ROOT,ROOT/'model',HERE,MODEL_DIR,STUDY):
  for path in folder.glob('*.py'):assert research[str(path)]==pins[str(path)]==sha(path)
 if require_runtime_proof:
  assert auth['runtime_qualification_SHA256']==pins[str(RUNTIME_PROOF)]==sha(RUNTIME_PROOF)
  q=read(RUNTIME_PROOF);assert q['status']=='RELEASED097_FRESH_BASELINE_SCOPE_ACCOUNTING_CPU_PASS' and q['actual_cells_checked']==4 and not q['GPU_allocation_created']
  for path,pin in q['source_SHA256'].items():assert pins[path]==pin==sha(path)
 return p
