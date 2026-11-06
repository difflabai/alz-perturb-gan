"""One frozen c014 fresh20k fit; same released native API and science."""
import hashlib,json,math
from pathlib import Path
ROOT=Path('/ml2/alz');HERE=Path(__file__).resolve().parent;STUDY=ROOT/'analysis/hpo20k'
PROTOCOL=STUDY/'protocol.json';SETTINGS=ROOT/'analysis/hpo500_best_measured_settings_20261106.json'
MODEL_DIR=ROOT/'analysis/released_tuning_v096';MODEL_PROOF=MODEL_DIR/'CPU_qualification.json';HPO_PROOF=ROOT/'analysis/hpo500/model_CPU_qualification.json'
NATIVE_PROOF=MODEL_DIR/'native_release.json';ENGINE_PROOF=ROOT/'analysis/shared_gpu0_candidate_CPU_qualification.json'
RUNTIME_PROOF=STUDY/'runtime_CPU_qualification.json';WINDOW=ROOT/'analysis/research_window_20261104_five_days.json';PLAN=HERE/'plan.json'
CFG={'z_dim':8,'stage_balance':False,'cell_z_dim':8,'theta_init_scale':1.,'fourier_feature_scale':1.}
ENABLED={'enabled':True,'experiment':'hpo20k_c014','native_version':'0.9.6','formulation':'native096_bispf_optimized_nonEMA','endpoint':20000,'training_seed':0,'candidate_id':'c014','owned_GPU_GiB':2,'calibration_only':True}
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def cell(candidate='c014',seed=0,phase='duration'):
 assert candidate=='c014' and type(seed) is int and seed==0 and phase=='duration'
 return dict(experiment='h20_c014',fold=0,method='pg',seed=0)
def proposal_path(candidate):assert candidate=='c014';return STUDY/'proposals/c014.json'
def validate_proposal(q):
 s=read(SETTINGS);assert q['id']==s['selected_candidate']=='c014' and q['native_version']==s['native_version']=='0.9.6' and q['endpoint']==20000
 assert q['hyperparameters']==s['hyperparameters'] and q['source_settings_SHA256']==sha(SETTINGS) and not q['heldout_results_used']
 assert all(type(v) in (int,float) and math.isfinite(v) and v>0 for v in q['hyperparameters'].values())
 return q['hyperparameters']
def protocol_paths(auth):return [PROTOCOL,SETTINGS,MODEL_PROOF,HPO_PROOF,NATIVE_PROOF,ENGINE_PROOF,WINDOW,PLAN,proposal_path(auth['candidate_id']),ROOT/'analysis/hpo500_final_independent_audit_20261106.json']
def validate_authorization(auth,*,require_runtime_proof=True):
 assert auth['cell']==cell(auth['candidate_id'],auth['cell']['seed'],auth['phase'])
 p=read(PROTOCOL);assert p['status']=='FROZEN_OWNER_AUTHORIZED_HPO500_C014_FRESH20K' and p['hyperparameters']==read(SETTINGS)['hyperparameters'] and p['endpoint']==20000
 assert auth['released_hyperparameters']==p['hyperparameters']==validate_proposal(read(proposal_path('c014'))) and auth['variant_config']==CFG
 assert auth['proposal_SHA256']==sha(proposal_path('c014')) and auth['benchmark_protocol_SHA256']==sha(PROTOCOL)
 assert auth['operation']=='hpo20k_cell' and auth['metric_role']=='calibration_selection'
 assert auth['fixed_endpoint']==auth['new_model_updates']==20000 and auth['initial_completed_updates']==0
 assert auth['is_continuation'] is False and auth['reuse_completed_model'] is False and auth['source_model_path'] is auth['source_model_SHA256'] is None
 assert auth['cap_seconds']==21600 and read(PLAN)==ENABLED
 assert auth['particlegan_version']=='0.9.6' and auth['penalty_options']=={'input_domain':'continuous'}
 assert not auth['cpu_fixture'] and not auth['independent_current_device_review_claimed'] and auth['source_review_type']=='OWNER_AUTHORIZED_VERIFIED_SOURCE_AND_RESOURCE_CHECKS'
 assert auth['gpu_uuid']=='GPU-cb4ce47d-d968-bffd-5646-e830a9fa1c69' and auth['dynamic_shared_gpu0'] is False
 assert sha(WINDOW)=='2dffe7ce5956cac37fd212e5b0b02b145acff98e35b75700855c98cda57dfbc1'
 assert math.isfinite(auth['issued_unix']) and auth['expires_unix']<=min(read(WINDOW)['expires_unix'],auth['issued_unix']+21630)
 pins=auth['files_SHA256'];research=auth['research_source_SHA256']
 for path in protocol_paths(auth):assert pins[str(path)]==sha(path)
 assert read(MODEL_PROOF)['status']=='PASSED_ACTUAL_RELEASED_PARAMETER_SYNTHETIC_CPU_QUALIFICATION' and read(MODEL_PROOF)['native_version']=='0.9.6'
 assert read(HPO_PROOF)['native_update_byte_identical'] and not read(HPO_PROOF)['CUDA_initialized']
 assert (MODEL_DIR/'released_update.py').read_bytes()==(ROOT/'analysis/hpo500/model/released_update.py').read_bytes()
 for evidence in (read(MODEL_PROOF)['source_SHA256'],read(HPO_PROOF)['source_SHA256'],read(NATIVE_PROOF)['official_wheel_source_SHA256'],read(ENGINE_PROOF)['source_SHA256']):
  for path,pin in evidence.items():assert pins[path]==pin==sha(path)
 for folder in (ROOT,ROOT/'model',HERE,MODEL_DIR,STUDY):
  for path in folder.glob('*.py'):assert research[str(path)]==pins[str(path)]==sha(path)
 if require_runtime_proof:
  assert auth['runtime_qualification_SHA256']==pins[str(RUNTIME_PROOF)]==sha(RUNTIME_PROOF)
  q=read(RUNTIME_PROOF);assert q['status']=='FROZEN_C014_FRESH20K_NATIVE_SCOPE_ACCOUNTING_CPU_PASS' and not q['GPU_allocation_created']
  for path,pin in q['source_SHA256'].items():assert pins[path]==pin==sha(path)
 return p
