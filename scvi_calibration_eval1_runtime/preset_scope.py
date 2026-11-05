"""Fixed stage-only scVI comparator, calibration only; no held-out scoring."""
import hashlib,json,math
from pathlib import Path
ROOT=Path('/ml2/alz');HERE=ROOT/'scvi_calibration_eval1_runtime'
MODEL_DIR=ROOT/'analysis/matched_scvi'
PROTOCOL=MODEL_DIR/'protocol.json'
MODEL_PROOF=MODEL_DIR/'CPU_model_qualification.json'
PARENT_FIT=ROOT/'runs/CAL_GPU0_V1_matched_scvi_calibration_f0_scvi_s0/fit'
PARENT_AUTH=ROOT/'artifacts/allocations/CAL_GPU0_V1_matched_scvi_calibration_f0_scvi_s0/authorization.json'
PARENT_FAILURE=ROOT/'analysis/matched_scvi_400fit_preserved_reload_failure.json'
RELOAD_PROOF=MODEL_DIR/'eval1_preserved_model_reload_CPU.json'
PARITY_PROOF=MODEL_DIR/'CPU_CUDA_source_parity.json'
RUNTIME_PROOF=MODEL_DIR/'eval1_runtime_CPU_qualification.json'
ENGINE_PROOF=ROOT/'analysis/shared_gpu0_candidate_CPU_qualification.json'
WINDOW=ROOT/'analysis/research_window_20261104_five_days.json'
PLAN=HERE/'plan.json'
CELL={'experiment':'matched_scvi_eval1','fold':0,'method':'scvi','seed':0}
ENABLED={'enabled':True,'method':'scvi','fold':0,'training_seed':0,'fixed_endpoint_epochs':400,'experiment':CELL['experiment'],'query_information':['stage'],'evaluation_used':False}
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def protocol_paths(_p=None):
 return [PARENT_AUTH,PARENT_FAILURE,PARENT_FIT/'fit_receipt.json',RELOAD_PROOF,PROTOCOL,MODEL_PROOF,PARITY_PROOF,ENGINE_PROOF,WINDOW,PLAN,ROOT/'pure_incumbent.json',ROOT/'incumbent.json',ROOT/'analysis/matched_scvi/initial_resource_observation.json']
def validate_authorization(auth,*,require_runtime_proof=True):
 p=read(PROTOCOL);pins=auth['files_SHA256'];research=auth['research_source_SHA256']
 assert auth['operation']=='evaluate_matched_scvi_calibration' and auth['cell']==CELL
 assert auth['fixed_endpoint']==400 and auth['cap_seconds']==900 and auth['role']=='alz_core_scvi'
 assert auth['scvi_version']=='1.3.3' and auth['scvi_architecture']==p['architecture'] and auth['scvi_training_plan']==p['training_plan']
 assert auth['query_information']==['stage'] and auth['evaluation_used'] is False
 assert auth['cpu_fixture'] is False and auth['independent_current_device_review_claimed'] is False
 assert auth['source_review_type']=='OWNER_AUTHORIZED_VERIFIED_SOURCE_AND_RESOURCE_CHECKS'
 assert auth['gpu_uuid']=='GPU-72c1b506-891d-b8bc-b353-e020585e1c47' and auth['dynamic_shared_gpu0'] is True
 assert read(PLAN)==ENABLED,'matched scVI plan remains disabled or changed'
 for key in ('checkpoint_path','checkpoint_SHA256','released_hyperparameters','particlegan_version','continuation','hybrid_config'):
  assert key not in auth,'comparator must be fresh native scVI, not a PG continuation'
 assert p['fixed_epochs']==400 and p['validation_cells']==0 and p['checkpoint_selection']=='fixed final400epochs'
 assert p['generation_replicates']==2 and p['calibration_manifest_SHA256']==auth['calibration_manifest_SHA256']
 assert auth['comparator_protocol_SHA256']==sha(PROTOCOL)
 w=read(WINDOW)
 assert w['project_GPU_hour_cap']==1024 and w['expires_unix']-w['effective_unix']==432000
 assert math.isfinite(auth['issued_unix']) and math.isfinite(auth['expires_unix'])
 assert auth['expires_unix']<=min(w['expires_unix'],auth['issued_unix']+930,read(PARENT_AUTH)['expires_unix'])
 assert auth['source_review_passed'] is True
 assert auth['parent_fit_path']==str(PARENT_FIT) and auth['new_model_updates']==0
 assert auth['parent_fit_receipt_SHA256']==sha(PARENT_FIT/'fit_receipt.json')
 parent=read(PARENT_AUTH);failed=read(PARENT_FAILURE);receipt=read(PARENT_FIT/'fit_receipt.json')
 assert failed['completed_native_fit_epochs']==receipt['completed_epochs']==400 and failed['calibration_quality_available'] is False
 assert auth['parent_authorization_SHA256']==sha(PARENT_AUTH) and auth['parent_absolute_expiry_unix']==parent['expires_unix']
 assert not (PARENT_FIT.parent/'r0').exists() and not (PARENT_FIT.parent/'r1').exists()
 for path,pin in receipt['model_files_SHA256'].items():assert pins[str(PARENT_FIT/path)]==pin==sha(PARENT_FIT/path)
 reload=read(RELOAD_PROOF)
 assert reload['status']=='PRESERVED_SCVI400_NATIVE_CPU_RELOAD_AND_AUTO_DEVICE_API_VERIFIED' and reload['CUDA_initialized'] is False
 for path,pin in reload['source_SHA256'].items():assert pins[path]==pin==sha(path)

 proof=read(MODEL_PROOF)
 assert proof['status']=='MATCHED_SCVI_NATIVE_CPU_MODEL_AND_FULLWIDTH_SAMPLER_QUALIFIED'
 assert proof['CUDA_initialized'] is False and proof['GPU_job_created'] is False
 assert proof['native_scvi_tests_passed']==2 and proof['full2000gene_prior_sampling_replay_passed'] is True
 parity=read(PARITY_PROOF)
 assert parity['Python_files_compared']==210 and parity['all_SHA256_equal'] is True
 for q in (p,proof,parity,read(ENGINE_PROOF)):
  for path,pin in q['source_SHA256'].items():assert pins[path]==pin==sha(path)
 for path in protocol_paths():assert pins[str(path)]==sha(path)
 for folder in (ROOT,ROOT/'model',HERE,MODEL_DIR):
  for path in folder.glob('*.py'):assert research[str(path)]==pins[str(path)]==sha(path)
 if require_runtime_proof:
  q=read(RUNTIME_PROOF)
  assert q['status']=='MATCHED_SCVI_ISOLATED_RUNTIME_CPU_QUALIFIED' and q['GPU_allocation_created'] is False
  assert auth['runtime_qualification_SHA256']==pins[str(RUNTIME_PROOF)]==sha(RUNTIME_PROOF)
  for path,pin in q['source_SHA256'].items():assert pins[path]==pin==sha(path)
 return p
