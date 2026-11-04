"""All frozen ensemble seeds, both models per seed, and no seed ranking."""
import argparse,json,sys
from pathlib import Path
ROOT=Path('/ml2/alz');sys.path[:0]=[str(Path(__file__).parent),str(ROOT/'analysis'),str(ROOT)]
import ensemble_confirmation_scope as scope
from aggregate_validation import verified_deployment,scientific_sources,IDENTITY
from aggregate_pure_confirmation import stable_target,frozen_input_pins

def aggregate(output):
 assert not output.exists(),'Preserve existing all-seed evidence'
 f=scope.frozen();records=[];evidence={str(scope.FREEZE):scope.sha(scope.FREEZE)};sources=None
 immutable=None
 for seed in (0,1,2):
  attempt=f['seed0_attempt'] if seed==0 else f['confirmation_identities'][str(seed)]['attempt_id']
  dep,auth,candidate=verified_deployment(attempt)
  assert auth['cpu_fixture'] is False and auth['status']=='APPROVED_FOR_BOUNDED_EXECUTION'
  assert scope.sha(dep['authorization_path'])==dep['authorization_SHA256']
  assert scientific_sources(auth)==f['scientific_source_SHA256']
  inputs=frozen_input_pins(auth,candidate)
  if immutable is None:immutable=inputs
  else:assert inputs==immutable
  baseline=scope.read(f['matched_baselines'][str(seed)]['path'])
  for key in IDENTITY:assert candidate[key]==baseline[key]
  assert candidate['training_seed']==seed and candidate['ensemble_config']==f['seed0_candidate']['ensemble_config']
  assert candidate['pure_only'] is True and candidate['real_bag_replacements']==0 and candidate['hybrid_config'] is None
  assert candidate['parent_training_updates_total']==40000 and candidate['parent_completed_updates']==[20000,20000]
  assert candidate['method']=='pg_pure_ensemble75' and candidate['native_recipe']=='bispf_optimized'
  if seed==0:
   assert candidate==f['seed0_candidate'] and scope.sha(dep['authorization_path'])==f['seed0_authorization_SHA256']
   decision=scope.read(ROOT/'analysis'/(attempt+'-result.json'));assert decision['candidate']==candidate and decision['decision']['accept']
  else:
   scope.validate_authorization(auth);reference=scope.verify_reference(candidate)
   if sources is None:sources=auth['research_source_SHA256']
   else:assert sources==auth['research_source_SHA256'],'Confirmation execution sources changed between seeds'
   path=ROOT/'analysis'/(attempt+'-confirmation.json');proof=scope.read(path)
   assert proof['status']=='PURE_ENSEMBLE_SEED_CONFIRMATION_RECORDED_NO_MODEL_SELECTION' and proof['candidate']==candidate and proof['baseline']==baseline
   assert proof['confirmation_freeze_SHA256']==scope.sha(scope.FREEZE)
   for key in ['independently_verified_saved_scores_counts_and_checkpoint','full_checkpoint_and_native_recipe_verified','owned_release_and_clean_settlement_verified','training_seed_was_not_ranked','incumbent_unchanged']:assert proof[key] is True
   audit=proof['checkpoint_audit'];assert scope.sha(audit['path'])==audit['SHA256']
   q=scope.read(audit['path']);assert q['status']=='VERIFIED_FROZEN_PURE_ENSEMBLE_CONFIRMATION_CHECKPOINTS' and q['training_seed']==seed and q['checkpoint_SHA256']==candidate['checkpoint_SHA256']
   assert q['confirmation_freeze_SHA256']==scope.sha(scope.FREEZE) and q['seed0_checkpoint_SHA256']==f['seed0_candidate']['checkpoint_SHA256']
   for key in ['recipe_exactly_matches_frozen_seed0','complete_model_keys_shapes_dtypes_finite_verified','policy_update_count_and_seed_streams_verified']:assert q[key] is True
   assert len(q['parents'])==2
   for a,b in zip(q['parents'],reference['parents']):assert a['checkpoint_SHA256']==b['checkpoint_SHA256'] and a['checkpoint_path']==b['checkpoint_path']
   for source,pin in q['source_SHA256'].items():assert scope.sha(source)==pin==auth['files_SHA256'][source]
   for source,pin in proof['array_and_exemplar_SHA256'].items():assert scope.sha(source)==pin
   evidence.update({str(path):scope.sha(path),audit['path']:audit['SHA256']})
  records.append({'seed':seed,'attempt':attempt,'energy_score':candidate['energy_score'],'matched_baseline_energy_score':baseline['energy_score'],
                  'cell_energy_distance':candidate['cell_energy_distance'],'matched_baseline_cell_energy_distance':baseline['cell_energy_distance'],'spread_ratio':candidate['spread_ratio']})
 result={**stable_target(records),'records':records,'evidence_SHA256':evidence,'pure_incumbent_SHA256':f['pure_incumbent_SHA256'],
         'incumbent_unchanged':True,'training_seeds_were_not_ranked':True,'each_seed_parent_updates':[20000,20000],
         'scope':'Same four adaptive calibration donors, all three predeclared training seeds, two full models per seed. Not twelve independent donors, independent biological confirmation or external SOTA. No evaluation donors used.'}
 assert scope.sha(ROOT/'pure_incumbent.json')==f['pure_incumbent_SHA256']
 output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--output',required=True,type=Path);aggregate(parser.parse_args().output)
