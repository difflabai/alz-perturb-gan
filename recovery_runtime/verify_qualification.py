"""Independent CPU check of actual native replay branch artifacts and clean settlement."""
import argparse,datetime,json,os,sys
from pathlib import Path
ROOT=Path('/ml2/alz');B=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark');S=Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/core-comparison-20261103')
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
sys.path[:0]=list(map(str,[Path(__file__).parent,ROOT,ROOT/'model',S,B,B/'current_base_20261029/runtime/lib/python3.12/site-packages',B/'.venv312/lib/python3.12/site-packages']))
import recovery_scope as scope
import torch
from continuation import same

def verify():
 p,f=scope.lineage();attempt=scope.attempt(p['qualification_cell']);dep=scope.read(ROOT/'artifacts/allocations'/attempt/'deployment.json');auth=scope.read(dep['authorization_path']);scope.validate_authorization(auth)
 assert scope.sha(dep['authorization_path'])==dep['authorization_SHA256']
 ledger=Path(dep['ledger_path']);job=scope.read(ledger)['jobs'][0];settlement=scope.read(ledger.with_name(attempt+'-settlement.json'))
 assert job['status']=='completed' and job['exit_code']==0 and job['reservation_released'] and settlement['uncertain'] is None and settlement['settlement']['status']=='settled'
 out=Path(dep['output_dir']);result=scope.read(out/'qualification.json');assert not (out/'completed.json').exists()
 assert result['status']=='ACTUAL_CUDA_EXACT_INTERRUPTED_STATE_AND_NEXT_UPDATE_BITWISE_VERIFIED' and result['qualification_only'] is True
 assert result['checkpoint_SHA256']==p['checkpoint_SHA256'] and result['recovery_protocol_SHA256']==scope.sha(scope.PROTOCOL)
 assert result['parent_completed_updates']==scope.PARENT_INITIAL and result['qualified_completed_updates']==scope.PARENT_INITIAL+1 and result['executed_optimizer_pairs']==2
 assert result['complete_initial_state_restored_exactly'] and result['complete_next_state_and_metrics_bitwise_equal']
 assert result['completed_parent_restored_exactly'] and result['completed_parent_checkpoint_SHA256']==p['completed_parent_checkpoint']['SHA256']
 assert result['calibration_or_evaluation_scores_generated'] is False
 files=[out/'qualification_branches'/f'branch{i}.pt' for i in (0,1)];branches=[]
 for path in files:
  assert scope.sha(path)==result['branch_SHA256'][str(path)]
  q=torch.load(path,map_location='cpu',weights_only=False);assert q['seed']==1 and q['fold']==0 and q['status']=='qualification_only'
  assert q['completed_updates']==q['policy']['completed_steps']==scope.PARENT_INITIAL+1 and q['fixed_endpoint']==q['recipe']['total_steps']==20000
  for model in q['policy']['models'].values():
   for v in model.values():
    if v.is_floating_point():assert torch.isfinite(v).all()
  branches.append(q)
 same(branches[0],branches[1]);assert not torch.cuda.is_initialized()
 proof={'status':'VERIFIED_ACTUAL_CUDA_INTERRUPTED_STATE_AND_NEXT_UPDATE','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'attempt':attempt,
        'checkpoint_SHA256':p['checkpoint_SHA256'],'recovery_protocol_SHA256':scope.sha(scope.PROTOCOL),'complete_initial_state_restored_exactly':True,
        'independent_saved_branch_states_bitwise_equal':True,'completed_parent_restored_exactly':True,'completed_parent_checkpoint_SHA256':p['completed_parent_checkpoint']['SHA256'],'complete_clean_settlement_verified':True,'qualified_execution_source_SHA256':auth['research_source_SHA256'],
        'qualification_branch_SHA256':result['branch_SHA256'],'charged_GPU_seconds':settlement['settlement']['charged_GPU_seconds'],'authorization_SHA256':scope.sha(dep['authorization_path']),
        'verification_source_SHA256':{str(Path(__file__).resolve()):scope.sha(Path(__file__).resolve())},
        'scope':'Actual guarded sameGPU learned9237 fullstate restoration and two one-update branches, plus exact completedPG8 restoration. Independent CPU equality of both complete saved9238 states after native success/release/settlement. Throwaway qualification states MUST NOT initialize scientific recovery; use newest learned9237 state. No quality generation/modelselection.'}
 assert not scope.NATIVE_PROOF.exists();scope.NATIVE_PROOF.write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps({k:v for k,v in proof.items() if k!='qualified_execution_source_SHA256'},indent=2))
if __name__=='__main__':verify()
