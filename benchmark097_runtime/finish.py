"""Retain every outcome; independently verify native state, counts and scores."""
import argparse,datetime,os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import binding as b,preset_scope as scope

def fit_evidence(out,method):
 """Pin final fitting receipts/history as well as the sampled model weights."""
 fit=Path(out)/'fit';paths=[]
 if method=='pg':
  receipt=b.read(fit/'checkpoint_receipt.json')
  assert receipt['status']=='completed' and receipt['completed_updates']==20000 and receipt['path']=='final.pt'
  assert receipt['SHA256']==b.sha(fit/'final.pt')
  curve=[__import__('json').loads(x) for x in (fit/'learning_curve.jsonl').read_text().splitlines()]
  steps=[x['step'] for x in curve];assert steps==sorted(set(steps)) and steps[0]==1 and steps[-1]==20000
  paths=[fit/'checkpoint_receipt.json',fit/'learning_curve.jsonl',fit/'progress.json',fit/'final.pt']
 else:
  receipt=b.read(fit/'fit_receipt.json');assert receipt['status']=='completed'
  paths=[fit/'fit_receipt.json']
  if method=='scvi':
   assert receipt['completed_epochs']==400 and receipt['validation_cells']==0 and not receipt['early_stopping']
   for name,pin in receipt['model_files_SHA256'].items():
    p=(fit/name).resolve();assert p.is_relative_to(fit.resolve()) and b.sha(p)==pin;paths.append(p)
  else:
   assert method=='cfgen' and receipt['AE_epochs']==300 and receipt['FM_epochs']==1500 and not receipt['synthetic']
   for phase in ('AE','FM'):
    p=fit/phase/'receipt.json';assert b.sha(p)==receipt[phase+'_receipt_SHA256'];q=b.read(p)
    checkpoint=fit/phase/'final.ckpt';assert q['checkpoint_path']==str(checkpoint) and q['checkpoint_SHA256']==b.sha(checkpoint)
    paths.extend([p,checkpoint])
 return {str(p):b.sha(p) for p in paths}

def finish(attempt):
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 folder=HERE.parent/'artifacts/allocations'/attempt;dep=b.read(folder/'deployment.json');auth=b.read(dep['authorization_path']);v=auth['cell']
 assert b.sha(dep['authorization_path'])==dep['authorization_SHA256'] and attempt==b.attempt_key(v)
 scope.validate_authorization(auth)
 for path,pin in auth['files_SHA256'].items():assert b.sha(path)==pin,path
 job=b.read(auth['ledger_path'])['jobs'][0];settlement=b.read(Path(auth['ledger_path']).with_name(attempt+'-settlement.json'))
 assert job['reservation_released'] and settlement['uncertain'] is None and settlement['settlement']['status']=='settled'
 for key in ('child_identity','guard_identity'):assert not Path('/proc',str(job[key]['pid'])).exists(),'Owned process still present'
 assert job['lease_token'] not in b.read(Path(auth['registry_path'])/'registry.json')['jobs']
 out=Path(auth['output_dir'])
 result=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),attempt=attempt,cell=v,
             benchmark_protocol_SHA256=b.sha(scope.PROTOCOL),own_processes_absent=True,owned_registry_lease_absent=True,
             charged_GPU_seconds=settlement['settlement']['charged_GPU_seconds'],all_new_training_from_scratch=True,
             initial_completed_updates=0,reuse_completed_model=False,automatic_retry=False,
             full_matrix_required_before_decision=True,SOTA_claim=False)
 if job['status']!='completed' or job['exit_code']!=0:
  result.update(status='INCOMPLETE_BENCHMARK097_CELL_CLEAN_SETTLEMENT_AUDIT',complete=False,
                job_status=job['status'],exit_code=job['exit_code'],
                failure=b.read(out/'failure.json') if (out/'failure.json').exists() else {'reason':'No worker failure receipt; retain the closed guardian outcome.'})
 else:
  sys.path.extend(map(str,b.native_site_roots(v,auth['files_SHA256'])))
  sys.path[:0]=[str(scope.MODEL_DIR),str(HERE.parent/'model'),str(b.SCIENCE),str(HERE.parent)]
  if v['method']=='cfgen':
   sys.path.insert(0,str(scope.CFGEN_DIR));import fit_adapter
   from native_reload import load_complete
   torch=fit_adapter.torch
  else:
   import torch
   torch.set_num_threads(1);torch.set_num_interop_threads(1)
  import numpy as np
  from scoring import score_summary_arrays
  from bench_core import cell_energy
  from state_fingerprint import fingerprint
  completed=b.read(out/'completed.json');native=b.read(out/'native_restore.json')
  assert completed['status']=='completed' and completed['metric_role']=='final_evaluation'
  assert (completed['fold'],completed['method'],completed['seed'])==(v['fold'],v['method'],v['seed'])
  assert completed['endpoint']==scope.endpoint(v['method']) and completed['benchmark_protocol_SHA256']==b.sha(scope.PROTOCOL)
  assert completed['from_scratch'] and completed['initial_completed_updates']==0 and not completed['reuse_completed_model']
  assert completed['new_model_updates']==scope.update_count(v) and completed['new_model_update_unit']==scope.update_unit(v['method'])
  assert completed['checkpoint_SHA256']==b.sha(completed['checkpoint_path'])==native['checkpoint_SHA256']
  assert completed['native_restore_SHA256']==b.sha(out/'native_restore.json')
  if v['method']=='pg':
   from worker_gpu0 import validate_pg
   import released_update as update,released_adapter as adapter
   update.configure(scope.CFG);update.configure_hyperparameters(auth['released_hyperparameters'])
   q=torch.load(completed['checkpoint_path'],map_location='cpu',weights_only=False);validate_pg(q,auth)
   assert fingerprint(q['policy'])==native['policy_fingerprint']
   assert q['training_streams']==adapter.training_streams(v['fold'],v['seed'])
   assert q['recipe']==update.recipe().to_dict()
   for module in q['policy']['models'].values():
    assert all(torch.isfinite(t).all() for t in module.values())
  elif v['method']=='scvi':
   from native_scvi_reload import load_native
   model,receipt,state=load_native(auth['cache_dir'],out/'fit','cpu')
   assert state==native['model_fingerprint'] and receipt['completed_epochs']==400
   assert receipt['seed']==v['seed'] and receipt['fold']==v['fold']
  else:
   model,receipt,state=load_complete(auth['cache_dir'],out/'fit',v['seed'],'cpu')
   assert receipt['fold']==v['fold'] and receipt['total_optimizer_updates']==auth['new_model_updates']
   for key in ('native_state_fingerprint','external_feature_fingerprint','checkpoint_SHA256','native_global_step','native_epoch'):assert native[key]==state[key]
  pins=fit_evidence(out,v['method']);records={}
  for name in ('evaluation_r0','evaluation_r1','resampling_r0','resampling_r1','stage_swap_r0'):
   dest=out/name;m=b.read(dest/'metrics.json');rep=0 if name.endswith('r0') else 1
   assert m['status']=='completed' and m['fold']==v['fold'] and m['training_seed']==v['seed'] and m['generation_replicate']==rep
   assert m['diagnostic_only'] is name.startswith('stage_swap')
   assert m['query_stage_permutation']==([2,3,0,1] if name.startswith('stage_swap') else [0,1,2,3])
   assert m['method']==('hierarchical_resampling' if name.startswith('resampling') else v['method'])
   assert m['generated_cells']==320000 and m['arrays_SHA256']==b.sha(dest/'arrays.npz')
   with np.load(dest/'arrays.npz',allow_pickle=False) as a:
    recomputed=score_summary_arrays(a['stage_synthetic_summaries'],a['real_summaries'],a['stage'])
    assert a['donors'].tolist()==m['donors']
    assert a['generated_cell_reservoir'].shape==(1500,50) and a['real_cell_reservoir'].shape==(1500,50)
    recomputed['cell_energy_distance']=cell_energy(a['generated_cell_reservoir'],a['real_cell_reservoir'])
   for key,value in recomputed.items():
    if isinstance(value,(int,float,list)):np.testing.assert_allclose(m[key],value,rtol=0,atol=1e-10)
   if not name.startswith('resampling'):
    for stage in range(4):
     p=dest/f'stage{stage}_count_exemplar.npz'
     with np.load(p,allow_pickle=False) as a:
      c=a['counts'];assert c.shape==(1,400,2000) and np.isfinite(c).all() and (c>=0).all() and np.equal(c,np.floor(c)).all()
   for path in dest.iterdir():
    if path.is_file():pins[str(path)]=b.sha(path)
   records[name]=m
  for path in (out/'completed.json',out/'native_restore.json'):
   pins[str(path)]=b.sha(path)
  assert not torch.cuda.is_initialized()
  result.update(status='EXTERNAL097_NATIVE_CHECKPOINT_COUNTS_SCORES_CLEAN_SETTLEMENT_AUDIT_PASS',complete=True,completed=completed,
                summary={name:{key:m[key] for key in ('energy_score','cell_energy_distance','spread_ratio','coverage80')} for name,m in records.items()},
                independent_CPU_saved_score_audit=True,native_saved_state_matches_GPU_restore=True,files_SHA256=pins)
 path=scope.STUDY/'results'/(attempt+'.json');assert not path.exists();path.parent.mkdir(exist_ok=True)
 path.write_text(__import__('json').dumps(result,indent=2,allow_nan=False)+'\n')
 print(__import__('json').dumps({k:result[k] for k in ('status','cell','complete','charged_GPU_seconds')},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);finish(p.parse_args().attempt)
