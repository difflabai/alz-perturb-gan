"""Frozen full20k distribution-primary arms, all fresh0 and same nativePG."""
import hashlib,json,math,re
from pathlib import Path
ROOT=Path('/ml2/alz');HERE=Path(__file__).resolve().parent;STUDY=ROOT/'analysis/distribution097'
PROTOCOL=STUDY/'protocol.json';MODEL_DIR=STUDY/'model';MODEL_PROOF=ROOT/'analysis/released_tuning_v096/CPU_qualification.json';HPO_PROOF=ROOT/'analysis/hpo500/model_CPU_qualification.json'
NATIVE_PROOF=STUDY/'native_release.json';ENGINE_PROOF=ROOT/'analysis/shared_gpu0_candidate_CPU_qualification.json'
RUNTIME_PROOF=STUDY/'runtime_CPU_qualification.json';WINDOW=ROOT/'analysis/research_window_20261104_five_days.json';PLAN=HERE/'plan.json'
CFG={'z_dim':8,'stage_balance':False,'cell_z_dim':8,'theta_init_scale':1.,'fourier_feature_scale':1.}
ENABLED={'enabled':True,'experiment':'distribution097_stage1','native_version':'0.9.7','formulation':'native097_bispf_optimized_nonEMA','endpoint':20000,'selection_seed':0,'candidate_ids':['d000','d001','d002'],'owned_GPU_GiB':2,'calibration_only':True,'all_training_from_scratch':True}
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def cell(candidate='d000',seed=0,phase='search'):
 assert candidate in ENABLED['candidate_ids'] and type(seed) is int and seed==0 and phase=='search'
 return dict(experiment='dist097_'+candidate,fold=0,method='pg',seed=0)
def proposal_path(candidate):assert candidate in ENABLED['candidate_ids'];return STUDY/'proposals'/(candidate+'.json')
def validate_proposal(q):
 assert q['id'] in ENABLED['candidate_ids'] and q['native_version']=='0.9.7' and q['endpoint']==20000 and q['protocol_SHA256']==sha(PROTOCOL)
 p=read(PROTOCOL);expected=next(c['parameters'] for c in p['selection_stage1']['configurations'] if c['id']==q['id']);assert q['hyperparameters']==expected
 assert q['fold']==q['seed']==q['initial_completed_updates']==0 and q['from_scratch'] and q['source_checkpoint'] is None and not q['heldout_results_used']
 assert all(type(v) in (int,float) and math.isfinite(v) and v>0 for v in q['hyperparameters'].values())
 return q['hyperparameters']
def protocol_paths(auth):return [PROTOCOL,MODEL_PROOF,HPO_PROOF,NATIVE_PROOF,ENGINE_PROOF,WINDOW,PLAN,proposal_path(auth['candidate_id']),ROOT/'analysis/particlegan097_runtime_equivalence_20261106.json']
def validate_authorization(auth,*,require_runtime_proof=True):
 assert auth['cell']==cell(auth['candidate_id'],auth['cell']['seed'],auth['phase'])
 p=read(PROTOCOL);assert p['status']=='PROSPECTIVE_FULL20K_DISTRIBUTION_PRIMARY_FRESH_NATIVE097_SEARCH' and p['all_new_training_from_scratch'] and p['endpoint']==20000
 assert auth['released_hyperparameters']==validate_proposal(read(proposal_path(auth['candidate_id']))) and auth['variant_config']==CFG
 assert auth['proposal_SHA256']==sha(proposal_path(auth['candidate_id'])) and auth['benchmark_protocol_SHA256']==sha(PROTOCOL)
 assert auth['operation']=='distribution097_cell' and auth['metric_role']=='calibration_selection'
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
 assert read(MODEL_PROOF)['status']=='PASSED_ACTUAL_RELEASED_PARAMETER_SYNTHETIC_CPU_QUALIFICATION' and read(MODEL_PROOF)['native_version']=='0.9.6'
 assert read(HPO_PROOF)['native_update_byte_identical'] and not read(HPO_PROOF)['CUDA_initialized']
 for name in ('released_update.py','released_adapter.py'):assert (MODEL_DIR/name).read_bytes()==(ROOT/'analysis/released_tuning_v096'/name).read_bytes()
 eq=read(ROOT/'analysis/particlegan097_runtime_equivalence_20261106.json');assert eq['all_core_runtime_files_byte_identical'] and eq['dependency_requirements_unchanged']
 for evidence in (read(MODEL_PROOF)['source_SHA256'],read(HPO_PROOF)['source_SHA256'],read(NATIVE_PROOF)['official_wheel_source_SHA256'],read(ENGINE_PROOF)['source_SHA256']):
  for path,pin in evidence.items():assert pins[path]==pin==sha(path)
 for folder in (ROOT,ROOT/'model',HERE,MODEL_DIR,STUDY):
  for path in folder.glob('*.py'):assert research[str(path)]==pins[str(path)]==sha(path)
 if require_runtime_proof:
  assert auth['runtime_qualification_SHA256']==pins[str(RUNTIME_PROOF)]==sha(RUNTIME_PROOF)
  q=read(RUNTIME_PROOF);assert q['status']=='NATIVE097_FRESH20K_DISTRIBUTION_SCOPE_ACCOUNTING_CPU_PASS' and not q['GPU_allocation_created']
  for path,pin in q['source_SHA256'].items():assert pins[path]==pin==sha(path)
 return p
