"""One preserved-checkpoint CFGen GPU qualification; no fitting or quality."""
import hashlib,json,math
from pathlib import Path
ROOT=Path('/ml2/alz');HERE=ROOT/'cfgen_gpu_qualification_runtime'
MODEL_DIR=ROOT/'analysis/cfgen_gpu_reload_qualification';PROTOCOL=MODEL_DIR/'protocol.json'
MODEL_PROOF=ROOT/'analysis/cfgen_device_explicit_CPU_replay_20261105.json'
RUNTIME_PROOF=MODEL_DIR/'runtime_CPU_qualification.json'
ENGINE_PROOF=ROOT/'analysis/shared_gpu0_candidate_CPU_qualification.json'
WINDOW=ROOT/'analysis/research_window_20261104_five_days.json';PLAN=HERE/'plan.json'
CHECKPOINT=Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/execution-control-20261103/target-pilot-output/sealed.ckpt')
CELL={'experiment':'cfgen_gpu_reload_qual','fold':0,'method':'cfgen','seed':0}
ENABLED={'enabled':True,'method':'cfgen','fold':0,'training_seed':0,'experiment':CELL['experiment'],'operation':'qualify_cfgen_gpu_reload','new_model_updates':0,'quality_scoring':False}
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def protocol_paths(_p=None):
 return [PROTOCOL,MODEL_PROOF,ENGINE_PROOF,WINDOW,PLAN,ROOT/'pure_incumbent.json',ROOT/'incumbent.json',MODEL_DIR/'initial_resource_observation.json']
def validate_authorization(auth,*,require_runtime_proof=True):
 p=read(PROTOCOL);pins=auth['files_SHA256'];research=auth['research_source_SHA256']
 assert auth['operation']=='qualify_cfgen_gpu_reload' and auth['cell']==CELL
 assert auth['fixed_endpoint']==0 and auth['cap_seconds']==900 and auth['role']=='alz_core_cfgen'
 assert auth['CFGen_source_commit']==p['source_commit']=='ee5ca0ccdd97a85a3d6bab9344063b77a8fc7af3'
 assert auth['qualification_only'] is True and auth['quality_scoring'] is False and auth['new_model_updates']==0
 assert auth['checkpoint_path']==str(CHECKPOINT) and auth['checkpoint_SHA256']==sha(CHECKPOINT)==p['checkpoint_SHA256']
 assert read(PLAN)==ENABLED,'CFGen qualification plan remains disabled or changed'
 assert auth['cpu_fixture'] is False and auth['independent_current_device_review_claimed'] is False
 assert auth['source_review_type']=='OWNER_AUTHORIZED_VERIFIED_SOURCE_AND_RESOURCE_CHECKS'
 assert auth['gpu_uuid']=='GPU-72c1b506-891d-b8bc-b353-e020585e1c47' and auth['dynamic_shared_gpu0'] is True
 assert auth['comparator_protocol_SHA256']==sha(PROTOCOL)
 w=read(WINDOW)
 assert w['project_GPU_hour_cap']==1024 and math.isfinite(auth['issued_unix']) and math.isfinite(auth['expires_unix'])
 assert auth['expires_unix']<=min(w['expires_unix'],auth['issued_unix']+930)
 proof=read(MODEL_PROOF)
 assert proof['status']=='CFGEN_DEVICE_EXPLICIT_FRESH_CPU_RESTORE_AND_COUNT_REPLAY_VERIFIED'
 assert proof['actual_fresh_processes']==2 and proof['GPU_contexts_created']==0
 assert proof['registered_and_external_state_and_counts_equal_across_fresh_processes'] is True
 assert p['sample_shape']==[256,2000] and p['new_model_updates']==0 and p['quality_metrics_computed'] is False
 for q in (p,proof,read(ENGINE_PROOF)):
  for path,pin in q['source_SHA256'].items():assert pins[path]==pin==sha(path)
 for path in protocol_paths():assert pins[str(path)]==sha(path)
 for folder in (ROOT,ROOT/'model',HERE,MODEL_DIR):
  for path in folder.glob('*.py'):assert research[str(path)]==pins[str(path)]==sha(path)
 if require_runtime_proof:
  q=read(RUNTIME_PROOF)
  assert q['status']=='CFGEN_GPU_RELOAD_ISOLATED_RUNTIME_CPU_QUALIFIED' and q['GPU_allocation_created'] is False
  assert auth['runtime_qualification_SHA256']==pins[str(RUNTIME_PROOF)]==sha(RUNTIME_PROOF)
  for path,pin in q['source_SHA256'].items():assert pins[path]==pin==sha(path)
 return p
