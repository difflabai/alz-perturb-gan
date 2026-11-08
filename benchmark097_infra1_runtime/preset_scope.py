"""Exactly one new from-zero attempt for a certified infrastructure preemption."""
import hashlib,importlib.util,json,math
from pathlib import Path
ROOT=Path('/ml2/alz');HERE=Path(__file__).resolve().parent;STUDY=ROOT/'analysis/benchmark097_infra1'
spec=importlib.util.spec_from_file_location('unchanged_original45_scientific_scope',ROOT/'benchmark097_balanced_runtime/preset_scope.py');original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
PROTOCOL=original.PROTOCOL;AMENDMENT=STUDY/'protocol_v2.json';PLAN=HERE/'plan.json'
MODEL_DIR=original.MODEL_DIR;CFGEN_DIR=original.CFGEN_DIR;MODEL_PROOF=original.MODEL_PROOF;NATIVE_PROOF=original.NATIVE_PROOF;ENGINE_PROOF=original.ENGINE_PROOF;WINDOW=original.WINDOW
RUNTIME_PROOF=STUDY/'runtime_CPU_qualification_v3.json';HANDOFF=ROOT/'analysis/benchmark097_original_attempts_postcapacity_closed.json'
CFG=original.CFG;ENABLED=dict(enabled=True,experiment='external097infra1',fold=2,methods=['pg','cfgen'],seed=1,initial_completed_updates=0,all_training_from_scratch=True)
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def cell(fold=2,method='cfgen',seed=1):
 assert type(fold) is type(seed) is int and fold==2 and seed==1 and method in ('pg','cfgen'),'Only certified infrastructure replacement is registered'
 return dict(experiment='external097infra1',fold=fold,method=method,seed=seed)
def endpoint(method):assert method in ('pg','cfgen');return 20000 if method=='pg' else 1800
def update_count(v):assert v==cell(v['fold'],v['method'],v['seed']);return original.update_count(v)
def update_unit(method):assert method in ('pg','cfgen');return 'optimizer_pairs' if method=='pg' else 'optimizer_steps'
def protocol():
 p=original.protocol();a=read(AMENDMENT)
 assert a['status']=='PRESPECIFIED_TWO_FRESH_INFRASTRUCTURE_REPLACEMENTS_NO_SCIENTIFIC_RESELECTION'
 assert a['physical_cells']==[cell(method=m) for m in ('pg','cfgen')] and a['logical_cells']==[original.cell(2,m,1) for m in ('pg','cfgen')]
 assert a['original_scientific_protocol_SHA256']==sha(PROTOCOL)
 for key,pin in a['known_failed_attempts'].items():assert sha(ROOT/'analysis/benchmark097/results'/(key+'.json'))==pin
 assert a['new_attempt_limit']==2 and a['initial_completed_updates']==0 and not a['reuse_failed_weights'] and not a['selection_quality_values_read']
 return p
def hyperparameters():return original.hyperparameters()
def proof_source_pins():return original.proof_source_pins()
def protocol_paths(p):
 return [*original.protocol_paths(p),AMENDMENT,PLAN,*[HERE/f'policy.gpu{i}.prepared.json' for i in (0,1)],ROOT/'analysis/continue_benchmark097_after_capacity_v5.py',ROOT/'analysis/benchmark097_capacity_recovery_continuation_v5_protocol_20261108.json',ROOT/'analysis/benchmark097_capacity_continuation_v5_CPU_check_20261108.json',ROOT/'analysis/benchmark097_capacity_CPU_handoff_v5_20261108.json']
def closed_originals():
 q=read(HANDOFF)
 assert q['status']=='ALL45_ORIGINAL_ATTEMPTS_CLOSED_43_COMPLETE_TWO_INFRA_FAILURES_FRESH_REPLACEMENTS_REMAIN'
 assert q['original_scientific_cells']==45 and q['successful_original_models']==43 and q['failed_original_attempts']==sorted(read(AMENDMENT)['known_failed_attempts'])
 assert len(q['result_SHA256'])==45
 for key,pin in q['result_SHA256'].items():assert sha(ROOT/'analysis/benchmark097/results'/(key+'.json'))==pin
 assert not read(ROOT/'benchmark097_balanced_runtime/plan.json')['enabled']
 return q
def validate_authorization(auth,*,require_runtime_proof=True,require_activation=True,require_handoff=True):
 p=protocol();v=auth['cell'];assert v==cell(v['fold'],v['method'],v['seed'])
 assert auth['role']=='alz_core_'+v['method'] and auth['operation']=='external097_infra1_cell' and auth['metric_role']=='final_evaluation'
 assert auth['fixed_endpoint']==endpoint(v['method']) and auth['new_model_updates']==update_count(v) and auth['new_model_update_unit']==update_unit(v['method'])
 assert auth['initial_completed_updates']==0 and not auth['reuse_completed_model'] and not auth['is_continuation'] and auth['source_model_path'] is auth['source_model_SHA256'] is None
 assert auth['cap_seconds']==21600 and auth['benchmark_protocol_SHA256']==sha(PROTOCOL)
 assert auth['particlegan_version']=='0.9.7' and auth['variant_config']==CFG and auth['released_hyperparameters']==hyperparameters()
 assert auth['penalty_options']=={'input_domain':'continuous'} and not auth['cpu_fixture']
 assert auth['source_review_type']=='OWNER_AUTHORIZED_VERIFIED_SOURCE_AND_RESOURCE_CHECKS' and auth['independent_current_device_review_claimed'] is False
 assert math.isfinite(auth['issued_unix']) and auth['expires_unix']<=min(read(WINDOW)['expires_unix'],auth['issued_unix']+21630)
 from dual_coordination import GPUS
 assert auth['gpu_uuid'] in GPUS and auth['dynamic_shared_gpu0']==(auth['gpu_uuid']==GPUS[0])
 if require_activation:assert read(PLAN)==ENABLED
 pins=auth['files_SHA256'];research=auth['research_source_SHA256']
 for path in protocol_paths(p):assert pins[str(path)]==sha(path),str(path)
 for path,pin in proof_source_pins().items():assert pins[path]==pin==sha(path)
 for path,pin in read(NATIVE_PROOF)['official_wheel_source_SHA256'].items():assert pins[path]==pin==sha(path)
 for folder in (ROOT,ROOT/'model',HERE,MODEL_DIR,CFGEN_DIR):
  for path in folder.glob('*.py'):assert research[str(path)]==pins[str(path)]==sha(path)
 if require_handoff:closed_originals();assert pins[str(HANDOFF)]==sha(HANDOFF)
 if require_runtime_proof:
  q=read(RUNTIME_PROOF);assert q['status']=='ONE_FRESH_INFRASTRUCTURE_REPLACEMENT_NATIVE_SCOPE_CPU_PASS' and q['native_cell_device_bindings']==4 and not q['GPU_allocation_created']
  assert auth['runtime_qualification_SHA256']==pins[str(RUNTIME_PROOF)]==sha(RUNTIME_PROOF)
  for path,pin in q['source_SHA256'].items():assert pins[path]==pin==sha(path)
 return p
