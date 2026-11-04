"""Verify all confirmation scores/counts and full parent proofs; never select/revert a seed."""
import argparse,datetime,json,sys
from pathlib import Path
ROOT=Path('/ml2/alz');sys.path[:0]=[str(Path(__file__).parent),str(ROOT/'analysis'),str(ROOT)]
import recovery_scope as scope
from aggregate_validation import verified_deployment,scientific_sources,IDENTITY
from aggregate_pure_confirmation import frozen_input_pins
from calibration import score,average
import numpy as np

def verify(attempt):
 f=scope.frozen();incumbent_pin=scope.sha(ROOT/'pure_incumbent.json');hybrid_pin=scope.sha(ROOT/'incumbent.json')
 dep,auth,candidate=verified_deployment(attempt)
 assert scope.sha(dep['authorization_path'])==dep['authorization_SHA256']
 scope.validate_authorization(auth);reference=scope.verify_reference(candidate);seed=candidate['training_seed']
 assert type(seed) is int and seed==auth['cell']['seed'] and candidate['operation']==('resume_pure_ensemble' if seed==1 else 'confirm_frozen_ensemble_seed2') and candidate['qualification_only'] is False
 assert candidate['pure_only'] is True and candidate['real_bag_replacements']==0 and candidate['hybrid_config'] is None
 assert candidate['method']=='pg_pure_ensemble75' and candidate['generation']=='iter_pg_whole_bag_ensemble'
 assert candidate['native_recipe']=='bispf_optimized' and candidate['parent_completed_updates']==[20000,20000] and candidate['initial_completed_updates']==(8942 if seed==1 else 0)
 assert candidate['completed_updates']==candidate['fixed_endpoint']==40000
 assert candidate['original_confirmation_freeze_SHA256']==scope.lineage()[0]['original_confirmation_freeze_SHA256'] and candidate['recovery_protocol_SHA256']==scope.sha(scope.PROTOCOL)
 assert scientific_sources(auth)==f['scientific_source_SHA256'];frozen_input_pins(auth,candidate)
 baseline=scope.read(f['matched_baselines'][str(seed)]['path'])
 for key in IDENTITY:assert candidate[key]==baseline[key]
 records=[];array_pins={}
 out=Path(dep['output_dir'])
 for rep in (0,1):
  folder=out/f'r{rep}';metrics=scope.read(folder/'metrics.json');assert metrics['arrays_SHA256']==scope.sha(folder/'arrays.npz')
  with np.load(folder/'arrays.npz',allow_pickle=False) as arrays:recomputed=score(arrays['stage_synthetic_summaries'],arrays['generated_cells_by_stage'])
  assert metrics['donors']==recomputed['donors']==candidate['donors'] and metrics['stage']==recomputed['stage']
  for key,value in recomputed.items():
   if isinstance(value,(int,float,list)) and key not in ('donors','stage'):np.testing.assert_allclose(metrics[key],value,rtol=0,atol=1e-10)
  for stage in range(4):
   path=folder/f'stage{stage}_exemplar.npz'
   with np.load(path,allow_pickle=False) as a:
    x=a['counts'];assert x.shape==(1,400,2000) and np.isfinite(x).all() and (x>=0).all() and np.equal(x,np.floor(x)).all()
   array_pins[str(path)]=scope.sha(path)
  array_pins[str(folder/'arrays.npz')]=scope.sha(folder/'arrays.npz');records.append(metrics)
 avg=average(records)
 assert all(candidate[k]==avg[k] for k in ['donors','training_seed','calibration_manifest_SHA256','trace_manifest_SHA256','sampling_contract'])
 for key,value in avg.items():
  if isinstance(value,(int,float,list)) and key not in ('donors',):np.testing.assert_allclose(candidate[key],value,rtol=0,atol=1e-10)
 audit_path=ROOT/'analysis'/(attempt+'-checkpoint-audit.json');audit=scope.read(audit_path)
 assert audit['status']=='VERIFIED_EXACT_RECOVERED_OR_FROZEN_SEED2_CHECKPOINTS' and audit['checkpoint_SHA256']==candidate['checkpoint_SHA256']
 assert audit['training_seed']==seed and audit['recovery_protocol_SHA256']==scope.sha(scope.PROTOCOL)
 for key in ['recipe_exactly_matches_frozen_seed0','complete_model_keys_shapes_dtypes_finite_verified','policy_update_count_and_seed_streams_verified']:assert audit[key] is True
 assert len(audit['parents'])==2
 for record,parent in zip(audit['parents'],reference['parents']):assert record['checkpoint_path']==parent['checkpoint_path'] and record['checkpoint_SHA256']==parent['checkpoint_SHA256']
 for path,pin in audit['source_SHA256'].items():assert scope.sha(path)==pin==auth['files_SHA256'][path]
 assert scope.sha(ROOT/'pure_incumbent.json')==incumbent_pin and scope.sha(ROOT/'incumbent.json')==hybrid_pin
 ledger=Path(dep['ledger_path']);settlement=scope.read(ledger.with_name(attempt+'-settlement.json'))
 result={'status':'EXACT_LINEAGE_ENSEMBLE_SEED_CONFIRMATION_RECORDED_NO_MODEL_SELECTION','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'attempt':attempt,'candidate':candidate,'baseline':baseline,'original_confirmation_freeze_SHA256':scope.lineage()[0]['original_confirmation_freeze_SHA256'],'recovery_protocol_SHA256':scope.sha(scope.PROTOCOL),
         'ES_relative_improvement':1-candidate['energy_score']/baseline['energy_score'],'cell_ED_difference':candidate['cell_energy_distance']-baseline['cell_energy_distance'],
         'joint_improvement':candidate['energy_score']<baseline['energy_score'] and candidate['cell_energy_distance']<=baseline['cell_energy_distance'] and .7<=candidate['spread_ratio']<=1.3,
         'independently_verified_saved_scores_counts_and_checkpoint':True,'full_checkpoint_and_native_recipe_verified':True,
         'owned_release_and_clean_settlement_verified':True,'training_seed_was_not_ranked':True,'incumbent_unchanged':True,
         'checkpoint_audit':{'path':str(audit_path),'SHA256':scope.sha(audit_path)},'array_and_exemplar_SHA256':array_pins,
         'charged_GPU_seconds':settlement['settlement']['charged_GPU_seconds'],'scope':'All results retained for frozen three-seed confirmation, no selection or seed replacement. Four adaptive calibration donors only.'}
 dest=ROOT/'analysis'/(attempt+'-confirmation.json');assert not dest.exists();dest.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k not in ('candidate','baseline','array_and_exemplar_SHA256')},indent=2))
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--attempt',required=True);verify(parser.parse_args().attempt)
