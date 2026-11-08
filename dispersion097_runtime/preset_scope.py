"""Isolated six fresh theta-initialization cells; existing studies stay frozen."""
import hashlib,json,math,sys
from pathlib import Path
ROOT=Path('/ml2/alz');HERE=Path(__file__).resolve().parent;STUDY=ROOT/'analysis/dispersion097'
sys.path.append(str(ROOT/'analysis'))
PROTOCOL=STUDY/'protocol.json';PLAN=HERE/'plan.json';RESOURCE_PROTOCOL=STUDY/'resource_protocol_v1.json'
MODEL_DIR=STUDY/'model';CONTROLLER_DIR=STUDY
MODEL_PROOF=STUDY/'fresh_seed_model_CPU_qualification.json';BASE_MODEL_PROOF=STUDY/'model_CPU_qualification.json'
NATIVE_PROOF=ROOT/'analysis/distribution097/native_release.json';ENGINE_PROOF=ROOT/'analysis/shared_gpu0_candidate_CPU_qualification.json'
RUNTIME_PROOF=STUDY/'runtime_CPU_qualification_v7.json';WINDOW=ROOT/'analysis/research_window_20261104_five_days.json'
ENABLED=dict(enabled=True,experiment='dispersion097',fold=0,training_seeds=[8,9,10],arms=['theta1','theta5'],endpoint=20000,all_training_from_scratch=True,calibration_only=True)
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def cell(arm='theta1',seed=8):
 assert arm in ('theta1','theta5') and type(seed) is int and seed in (8,9,10),'unregistered fresh theta cell'
 return dict(experiment='dispersion097_'+arm,fold=0,method='pg',seed=seed)
def arm_of(value):
 arm=value['experiment'].removeprefix('dispersion097_');assert value==cell(arm,value['seed']);return arm
def configuration(arm):
 assert arm in ('theta1','theta5')
 return dict(z_dim=8,stage_balance=False,cell_z_dim=8,theta_init_scale=1. if arm=='theta1' else 5.,fourier_feature_scale=1.)
def endpoint(method):assert method=='pg';return 20000
def update_count(value):arm_of(value);return 20000
def update_unit(method):assert method=='pg';return 'optimizer_pairs'
def protocol():
 p=read(PROTOCOL)
 assert p['status']=='FROZEN_SEPARATE_THETA_INIT_SIX_FRESH_MODELS' and p['native_version']=='0.9.7'
 assert p['training_seeds']==[8,9,10] and p['arms']==['theta1','theta5'] and p['initialization_scales']=={'theta1':1.,'theta5':5.}
 assert p['required_fresh_fits']==6 and p['updates_per_fit']==20000 and p['cap_seconds']==21600
 assert p['all_from_scratch'] and p['initial_completed_updates']==0 and p['source_checkpoint'] is None and p['calibration_only']
 assert p['hyperparameters']==dict(generator_lr=.012,critic_lr=.018,particle_lr=.06,reg_coeff=1.,reg_kappa=1.,prior_reg=.01)
 assert p['no_old45_reselection'] and p['no_interaction_with_pending_LR_arm_selection'] and not p['formulation_change'] and not p['SOTA_claim']
 for path,pin in p['fidelity_evidence_SHA256'].items():assert sha(path)==pin,path
 proof=read(MODEL_PROOF)
 assert proof['status']=='ALL_SIX_FRESH_THETA1_THETA5_SEED8TO10_FULLWIDTH_NATIVE_TRAINER_PREFIX_CASES_PASS'
 assert proof['actual_cases']==6 and proof['actual_optimizer_pairs_per_implementation']==18 and proof['fullwidth_genes']==2000
 assert proof['fresh_training_seed_namespaces_qualified']==[8,9,10] and proof['new_initialization_streams_unseen_in_original_seed0to4']
 assert not proof['real_ALZ_data_read'] and not proof['CUDA_initialized'] and proof['real_new_ALZ_fits']==0
 for proofpath in (MODEL_PROOF,BASE_MODEL_PROOF):
  for path,pin in read(proofpath)['source_SHA256'].items():assert sha(path)==pin,path
 return p
def hyperparameters():return protocol()['hyperparameters']
def proof_source_pins():
 pins={}
 for proof in (MODEL_PROOF,BASE_MODEL_PROOF,ENGINE_PROOF):
  for path,pin in read(proof)['source_SHA256'].items():
   if path in pins:assert pins[path]==pin
   pins[path]=pin
 return pins
def handoff_paths(root=ROOT):
 from prior_handoff import paths
 return paths(root)
def prior_study_ready(root=ROOT):
 from prior_handoff import ready
 return ready(root)
def protocol_paths():
 return [PROTOCOL,RESOURCE_PROTOCOL,PLAN,MODEL_PROOF,BASE_MODEL_PROOF,NATIVE_PROOF,ENGINE_PROOF,WINDOW,STUDY/'proposed_fresh_six_design.json',HERE/'prior_handoff.py',ROOT/'analysis/scalarcal097_recovered_handoff.py',ROOT/'analysis/released097_remaining_runtime_balance.py',ROOT/'analysis/run_dispersion097.py',ROOT/'analysis/after_scalarcal097_dispersion.py',ROOT/'analysis/aggregate_dispersion097.py',ROOT/'benchmark097_runtime/resource_profile.py',*[ROOT/f'benchmark097_dual_runtime/policy.gpu{i}.prepared.json' for i in (0,1)],*map(Path,protocol()['fidelity_evidence_SHA256'])]
def validate_authorization(auth,*,require_runtime_proof=True,require_activation=True,require_handoff=True):
 p=protocol();v=auth['cell'];assert v==cell(auth['arm'],v['seed'])
 assert auth['operation']=='dispersion097_cell' and auth['metric_role']=='calibration_selection' and auth['role']=='alz_core_pg'
 assert auth['fixed_endpoint']==auth['new_model_updates']==20000 and auth['initial_completed_updates']==0 and auth['new_model_update_unit']=='optimizer_pairs'
 assert not auth['reuse_completed_model'] and not auth['is_continuation'] and auth['source_model_path'] is auth['source_model_SHA256'] is None
 assert auth['cap_seconds']==21600 and auth['benchmark_protocol_SHA256']==sha(PROTOCOL)
 assert auth['released_hyperparameters']==p['hyperparameters'] and auth['variant_config']==configuration(auth['arm'])
 assert auth['particlegan_version']=='0.9.7' and auth['penalty_options']=={'input_domain':'continuous'} and not auth['cpu_fixture']
 assert auth['independent_current_device_review_claimed'] is False and auth['source_review_type']=='OWNER_AUTHORIZED_VERIFIED_SOURCE_AND_RESOURCE_CHECKS'
 if require_activation:assert read(PLAN)==ENABLED,'Separate theta study remains disabled'
 assert sha(WINDOW)=='2dffe7ce5956cac37fd212e5b0b02b145acff98e35b75700855c98cda57dfbc1'
 assert math.isfinite(auth['issued_unix']) and auth['expires_unix']<=min(read(WINDOW)['expires_unix'],auth['issued_unix']+21630)
 from dual_coordination import GPUS
 assert auth['gpu_uuid'] in GPUS and auth['dynamic_shared_gpu0']==(auth['gpu_uuid']==GPUS[0])
 pins=auth['files_SHA256'];research=auth['research_source_SHA256']
 for path in protocol_paths():assert pins[str(path)]==sha(path),str(path)
 for path,pin in proof_source_pins().items():assert pins[path]==pin==sha(path),path
 for path,pin in read(NATIVE_PROOF)['official_wheel_source_SHA256'].items():assert pins[path]==pin==sha(path),path
 for folder in (ROOT,ROOT/'model',HERE,MODEL_DIR,STUDY,ROOT/'analysis/faithful097/model'):
  for path in folder.glob('*.py'):assert research[str(path)]==pins[str(path)]==sha(path),str(path)
 if require_handoff:
  for path,pin in prior_study_ready().items():assert pins[path]==pin
 if require_runtime_proof:
  assert auth['runtime_qualification_SHA256']==pins[str(RUNTIME_PROOF)]==sha(RUNTIME_PROOF)
  proof=read(RUNTIME_PROOF)
  assert proof['status']=='DISPERSION097_SIX_FRESH_CELLS_SCOPE_ACCOUNTING_CPU_PASS' and proof['native_cells_checked']==6 and proof['native_cell_device_bindings']==12 and not proof['GPU_allocation_created']
  for path,pin in proof['source_SHA256'].items():assert pins[path]==pin==sha(path),path
 return p
