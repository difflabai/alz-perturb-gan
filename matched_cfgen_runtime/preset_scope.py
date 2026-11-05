"""One fresh native CFGen matched comparator, fixed300AE/1500FMepochs."""
import hashlib,json,math
from pathlib import Path
ROOT=Path('/ml2/alz');HERE=ROOT/'matched_cfgen_runtime';MODEL_DIR=ROOT/'analysis/matched_cfgen';PROTOCOL=MODEL_DIR/'protocol.json'
MODEL_PROOF=MODEL_DIR/'CPU_model_qualification.json';RUNTIME_PROOF=MODEL_DIR/'runtime_CPU_qualification.json'
GPU_PROOF=ROOT/'analysis/CAL_GPU0_V1_cfgen_gpu_reload_qual_f0_cfgen_s0-qualification.json'
ENGINE_PROOF=ROOT/'analysis/shared_gpu0_candidate_CPU_qualification.json';WINDOW=ROOT/'analysis/research_window_20261104_five_days.json';PLAN=HERE/'plan.json'
CELL={'experiment':'matched_cfgen','fold':0,'method':'cfgen','seed':0}
ENABLED={'enabled':True,'method':'cfgen','fold':0,'training_seed':0,'experiment':CELL['experiment'],'operation':'train_matched_cfgen','AE_epochs':300,'FM_epochs':1500,'fixed_endpoint':1800}
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def protocol_paths(_p=None):return [PROTOCOL,MODEL_PROOF,GPU_PROOF,ENGINE_PROOF,WINDOW,PLAN,ROOT/'pure_incumbent.json',ROOT/'incumbent.json',MODEL_DIR/'initial_resource_observation.json']
def validate_authorization(auth,*,require_runtime_proof=True):
 p=read(PROTOCOL);pins=auth['files_SHA256'];research=auth['research_source_SHA256']
 assert auth['operation']=='train_matched_cfgen' and auth['cell']==CELL and auth['role']=='alz_core_cfgen'
 assert auth['AE_epochs']==300 and auth['FM_epochs']==1500 and auth['fixed_endpoint']==1800 and auth['new_model_updates']==102600
 assert auth['cap_seconds']==21600 and auth['qualification_only'] is False and auth['quality_scoring'] is True
 assert auth['CFGen_source_commit']==p['source_commit']=='ee5ca0ccdd97a85a3d6bab9344063b77a8fc7af3'
 for key in ['checkpoint_path','checkpoint_SHA256','parent_fit_path','continuation']:assert key not in auth,'Fresh matched initialization required'
 assert read(PLAN)==ENABLED,'Matched CFGen plan remains disabled or changed'
 assert auth['cpu_fixture'] is False and auth['independent_current_device_review_claimed'] is False
 assert auth['source_review_type']=='OWNER_AUTHORIZED_VERIFIED_SOURCE_AND_RESOURCE_CHECKS'
 assert auth['gpu_uuid']=='GPU-72c1b506-891d-b8bc-b353-e020585e1c47' and auth['dynamic_shared_gpu0'] is True
 assert auth['comparator_protocol_SHA256']==sha(PROTOCOL)
 assert p['optimization_donors']==33 and p['optimization_cells']==14762 and p['evaluation_donors_used'] is False
 assert p['AE_epochs']==300 and p['FM_epochs']==1500 and p['total_optimizer_updates']==102600
 assert p['use_ot'] is False and p['validation_cells']==0 and p['early_stopping'] is False
 w=read(WINDOW);assert w['project_GPU_hour_cap']==1024
 assert math.isfinite(auth['issued_unix']) and math.isfinite(auth['expires_unix']) and auth['expires_unix']<=min(w['expires_unix'],auth['issued_unix']+21630)
 proof=read(MODEL_PROOF);assert proof['status']=='MATCHED_CFGEN_NATIVE_FULL2000GENE_TWO_STAGE_CPU_FIT_REPLAY_PASS'
 assert proof['checks']['AE_optimizer_updates']==proof['checks']['FM_optimizer_updates']==2
 for key in ['frozen_encoder_exact','native_saved_model_reloaded_exact','native_count_replay_exact','native_trace_chunk_to_bag_replay_passed','synthetic_endpoint_and_CPU_realfit_refused']:assert proof['checks'][key] is True
 assert proof['CUDA_initialized'] is False and proof['GPU_job_created'] is False
 gpu=read(GPU_PROOF)
 assert gpu['status']=='NATIVE_CFGEN_GPU_RELOAD_FRESH_PROCESS_STATE_AND_COUNTS_REPLAY_AUDITED_PASS'
 assert gpu['owned_release_and_clean_settlement_verified'] and gpu['native_two_fresh_GPU_processes_count_replay_passed']
 for q in (p,proof,gpu,read(ENGINE_PROOF)):
  for path,pin in q['source_SHA256'].items():assert pins[path]==pin==sha(path)
 for path in protocol_paths():assert pins[str(path)]==sha(path)
 for folder in (ROOT,ROOT/'model',HERE,MODEL_DIR):
  for path in folder.glob('*.py'):assert research[str(path)]==pins[str(path)]==sha(path)
 if require_runtime_proof:
  q=read(RUNTIME_PROOF);assert q['status']=='MATCHED_CFGEN_ISOLATED_RUNTIME_CPU_QUALIFIED' and q['GPU_allocation_created'] is False
  assert auth['runtime_qualification_SHA256']==pins[str(RUNTIME_PROOF)]==sha(RUNTIME_PROOF)
  for path,pin in q['source_SHA256'].items():assert pins[path]==pin==sha(path)
 return p
