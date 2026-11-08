"""Exact logical45 mapping; all three infrastructure failures and charges stay visible."""
import hashlib,json,math,os
from pathlib import Path
ROOT=Path('/ml2/alz');STUDY=ROOT/'analysis/benchmark097';RECOVERY=ROOT/'analysis/benchmark097_infra1'
FAILED=('CAL_GPU0_V1_external097_f1_cfgen_s2','CAL_GPU0_V1_external097_f2_cfgen_s1','CAL_GPU0_V1_external097_f2_pg_s1')
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def identity(pid):
 try:
  p=Path('/proc',str(pid));v=(p/'stat').read_text().rsplit(')',1)[1].split()
  if v[0]=='Z' or p.stat().st_uid!=os.getuid():return None
  return dict(pid=pid,birth=int(v[19]),boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
 except (FileNotFoundError,ProcessLookupError,PermissionError):return None
def physical_key(fold,method,seed):
 prefix='external097infra1' if (fold,method,seed) in ((2,'pg',1),(2,'cfgen',1),(1,'cfgen',2)) else 'external097'
 return f'CAL_GPU0_V1_{prefix}_f{fold}_{method}_s{seed}'
def metadata(attempt,*,complete):
 folder=ROOT/'artifacts/allocations'/attempt;dep=read(folder/'deployment.json');assert sha(dep['authorization_path'])==dep['authorization_SHA256'];a=read(dep['authorization_path'])
 assert a['attempt_id']==attempt and a['initial_completed_updates']==0 and not a['is_continuation'] and not a['reuse_completed_model'] and a['source_model_path'] is a['source_model_SHA256'] is None
 ledger=Path(a['ledger_path']);data=read(ledger);assert len(data['jobs'])==1 and data['gpu_uuid']==a['gpu_uuid'];j=data['jobs'][0];s=read(ledger.with_name(attempt+'-settlement.json'))
 assert j['reservation_released'] and s['uncertain'] is None and s['settlement']['status']=='settled'
 assert (j['status']=='completed' and j['exit_code']==0)==complete
 if not complete:assert attempt in FAILED
 for key in ('guard_identity','child_identity'):
  old=j[key];assert identity(old['pid'])!={k:old[k] for k in ('pid','birth','boot')}
 assert j['lease_token'] not in read(Path(a['registry_path'])/'registry.json')['jobs']
 base=RECOVERY if a['cell']['experiment']=='external097infra1' else STUDY
 result=base/'results'/(attempt+'.json');assert result.is_file()
 if not complete:
  if attempt=='CAL_GPU0_V1_external097_f1_cfgen_s2':
   from verify_cfgen_postfit_cap_failure import classify
   expected=read(RECOVERY/'protocol_v3.json')['known_failed_attempts'][attempt];out=Path(a['output_dir'])
   classify(auth=a,auth_sha256=sha(dep['authorization_path']),job=j,settlement=s,result=read(result),result_sha256=sha(result),expected_result_sha256=expected,fit=read(out/'fit/fit_receipt.json'),restore=read(out/'native_restore.json'),processes_absent=True,lease_absent=True)
  else:assert 'Protected/registered GPU growth no longer fits' in j['error']
 return dict(attempt=attempt,cell=a['cell'],complete=complete,result_path=str(result),result_SHA256=sha(result),runtime_directory=str(Path(a['command'][4]).parent),authorization_path=dep['authorization_path'],authorization_SHA256=dep['authorization_SHA256'],charged_GPU_seconds=s['settlement']['charged_GPU_seconds'])
def build_matrix():
 closure=read(ROOT/'analysis/benchmark097_original_attempts_postcapacity_closed.json');assert closure['successful_original_models']==42 and closure['failed_original_attempts']==list(FAILED)
 amendment=read(RECOVERY/'protocol_v3.json');rows={};logical={}
 for f in range(3):
  for seed in range(5):
   for method in ('pg','scvi','cfgen'):
    key=physical_key(f,method,seed);r=metadata(key,complete=True);expected=dict(experiment='external097infra1' if key.startswith('CAL_GPU0_V1_external097infra1_') else 'external097',fold=f,method=method,seed=seed);assert r['cell']==expected
    rows[key]=r;logical[f'f{f}_{method}_s{seed}']=key
 failures={key:metadata(key,complete=False) for key in FAILED}
 assert {k:v['result_SHA256'] for k,v in failures.items()}==amendment['known_failed_attempts']
 assert len(rows)==45 and len(set(logical.values()))==45
 for key,pin in closure['result_SHA256'].items():assert sha(STUDY/'results'/(key+'.json'))==pin
 return dict(status='LOGICAL45_FROZEN_MODELS_COMPLETE_WITH_THREE_DISCLOSED_FRESH_INFRASTRUCTURE_REPLACEMENTS',matrix_complete=True,complete_cells=45,required_cells=45,physical_attempts=48,successful_fresh_models=45,infrastructure_failed_attempts=list(FAILED),logical_to_physical_attempt=logical,result_SHA256={k:v['result_SHA256'] for k,v in rows.items()},resource_runtime_by_attempt={k:v['runtime_directory'] for k,v in rows.items()},failed_result_SHA256={k:v['result_SHA256'] for k,v in failures.items()},successful_model_GPU_seconds=sum(r['charged_GPU_seconds'] for r in rows.values()),failed_infrastructure_GPU_seconds=sum(r['charged_GPU_seconds'] for r in failures.values()),actual_GPU_seconds=sum(r['charged_GPU_seconds'] for r in [*rows.values(),*failures.values()]),all_new_training_from_scratch=True,initial_completed_updates=0,failed_weights_resumed=False,scientific_protocol_unchanged=True,protocol_SHA256=sha(STUDY/'protocol.json'),recovery_protocol_SHA256=sha(RECOVERY/'protocol_v3.json'),original_attempt_closure_SHA256=sha(ROOT/'analysis/benchmark097_original_attempts_postcapacity_closed.json'),source_SHA256={str(Path(__file__).resolve()):sha(__file__)},full_goal_complete=False,SOTA_claim=False)
def load_complete_audits():
 matrix=read(STUDY/'matrix_complete.json');assert matrix==build_matrix(),'All45 successful logical models and all three retained failed costs are mandatory before quality reads'
 project=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark');assert not read(project/'project_gpu_budget.json')['open_reservations']
 for name in ('benchmark097_dual_runtime','benchmark097_balanced_runtime','benchmark097_infra1_runtime'):assert not read(ROOT/name/'plan.json')['enabled']
 rows={};allowed={ROOT/x for x in ('benchmark097_dual_runtime','benchmark097_balanced_runtime','benchmark097_infra1_runtime')}
 for f in range(3):
  for seed in range(5):
   for method in ('pg','scvi','cfgen'):
    key=physical_key(f,method,seed);meta=metadata(key,complete=True);a=read(meta['authorization_path']);runtime=Path(meta['runtime_directory']);assert runtime in allowed
    assert a['benchmark_protocol_SHA256']==matrix['protocol_SHA256'] and a['fixed_endpoint']==dict(pg=20000,scvi=400,cfgen=1800)[method]
    assert (runtime.name=='benchmark097_infra1_runtime')==(key.startswith('CAL_GPU0_V1_external097infra1_'))
    for file,pin in a['files_SHA256'].items():
     if Path(file).name=='plan.json' and Path(file).parent in allowed:assert not read(file)['enabled']
     else:assert sha(file)==pin,file
    q=read(meta['result_path']);assert q['attempt']==key and q['cell']==a['cell'] and q['complete'] and q['status']=='EXTERNAL097_NATIVE_CHECKPOINT_COUNTS_SCORES_CLEAN_SETTLEMENT_AUDIT_PASS'
    assert q['all_new_training_from_scratch'] and q['initial_completed_updates']==0 and not q['reuse_completed_model'] and q['benchmark_protocol_SHA256']==matrix['protocol_SHA256']
    assert math.isclose(q['charged_GPU_seconds'],meta['charged_GPU_seconds'],rel_tol=0,abs_tol=1e-8)
    for file,pin in q['files_SHA256'].items():assert sha(file)==pin,file
    rows[f,method,seed]=q
 return rows,matrix
