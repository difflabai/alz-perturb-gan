"""Verified cost/identity references without parsing quality-result JSON."""
import hashlib,json
from pathlib import Path
from dual_coordination import identity,valid_cell
ROOT=Path('/ml2/alz')
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def key(v):return f"CAL_GPU0_V1_{v['experiment']}_f{v['fold']}_{v['method']}_s{v['seed']}"
def completed_metadata(value,root=ROOT,*,require_full=True):
 valid_cell(value);root=Path(root);attempt=key(value);folder=root/'artifacts/allocations'/attempt
 dep=read(folder/'deployment.json');assert sha(dep['authorization_path'])==dep['authorization_SHA256'];auth=read(dep['authorization_path'])
 assert auth['cell']==value and auth['initial_completed_updates']==0 and not auth['is_continuation'] and not auth['reuse_completed_model'] and auth['source_model_path'] is auth['source_model_SHA256'] is None
 ledger=Path(auth['ledger_path']);data=read(ledger);assert data['gpu_uuid']==auth['gpu_uuid'] and len(data['jobs'])==1;job=data['jobs'][0]
 assert job['reservation_released']
 complete=job['status']=='completed' and job['exit_code']==0
 if require_full:assert complete
 settlement=read(ledger.with_name(attempt+'-settlement.json'));assert settlement['uncertain'] is None and settlement['settlement']['status']=='settled'
 for name in ('guard_identity','child_identity'):
  old=job[name];assert identity(old['pid'])!={k:old[k] for k in ('pid','birth','boot')}
 path=root/'analysis/dispersion097/results'/(attempt+'.json');assert path.is_file()
 return dict(attempt=attempt,cell=value,complete=complete,all_new_training_from_scratch=True,initial_completed_updates=0,source_checkpoint=None,charged_GPU_seconds=settlement['settlement']['charged_GPU_seconds'],gpu_uuid=auth['gpu_uuid'],runtime_directory=str(Path(auth['command'][4]).parent),benchmark_protocol_SHA256=auth['benchmark_protocol_SHA256'],source_result_path=str(path),source_result_SHA256=sha(path),authorization_path=dep['authorization_path'],authorization_SHA256=dep['authorization_SHA256'],own_processes_absent=True,owned_registry_lease_absent=True,result_quality_JSON_parsed=False)
