"""Frozen0/1/2 target with original failedseed1 plus exact recovery retained."""
import argparse,json,sys
from pathlib import Path
ROOT=Path('/ml2/alz');sys.path[:0]=[str(Path(__file__).parent),str(ROOT/'analysis'),str(ROOT)]
import recovery_scope as scope
from aggregate_validation import verified_deployment,scientific_sources,IDENTITY
from aggregate_pure_confirmation import stable_target,frozen_input_pins

def aggregate(output):
 assert not output.exists(),'Never overwrite all-seed evidence'
 p,f=scope.lineage();records=[];evidence={str(scope.PROTOCOL):scope.sha(scope.PROTOCOL),p['original_interruption_evidence']:scope.sha(p['original_interruption_evidence']),str(scope.NATIVE_PROOF):scope.sha(scope.NATIVE_PROOF)};sources=None;inputs0=None;seed1_total_charge=None
 for seed in (0,1,2):
  name=f['seed0_attempt'] if seed==0 else scope.attempt(scope.cell_for('resume' if seed==1 else 'seed2'))
  dep,auth,candidate=verified_deployment(name);assert scope.sha(dep['authorization_path'])==dep['authorization_SHA256']
  assert auth['cpu_fixture'] is False and auth['status']=='APPROVED_FOR_BOUNDED_EXECUTION'
  assert scientific_sources(auth)==f['scientific_source_SHA256'];inputs=frozen_input_pins(auth,candidate)
  if inputs0 is None:inputs0=inputs
  else:assert inputs==inputs0
  baseline=scope.read(f['matched_baselines'][str(seed)]['path'])
  for key in IDENTITY:assert candidate[key]==baseline[key]
  assert type(candidate['training_seed']) is int and candidate['training_seed']==seed and candidate['ensemble_config']==p['frozen_ensemble_config']
  assert candidate['pure_only'] is True and candidate['real_bag_replacements']==0 and candidate['hybrid_config'] is None
  assert candidate['parent_training_updates_total']==40000 and candidate['parent_completed_updates']==[20000,20000]
  assert candidate['method']=='pg_pure_ensemble75' and candidate['native_recipe']=='bispf_optimized'
  if seed==0:
   assert candidate==f['seed0_candidate'] and scope.sha(dep['authorization_path'])==f['seed0_authorization_SHA256']
   decision=scope.read(ROOT/'analysis'/(name+'-result.json'));assert decision['candidate']==candidate and decision['decision']['accept']
  else:
   scope.validate_authorization(auth);reference=scope.verify_reference(candidate)
   if sources is None:sources=auth['research_source_SHA256']
   else:assert sources==auth['research_source_SHA256']
   path=ROOT/'analysis'/(name+'-confirmation.json');proof=scope.read(path)
   assert proof['status']=='EXACT_LINEAGE_ENSEMBLE_SEED_CONFIRMATION_RECORDED_NO_MODEL_SELECTION' and proof['candidate']==candidate and proof['baseline']==baseline
   assert proof['recovery_protocol_SHA256']==scope.sha(scope.PROTOCOL)
   for key in ['independently_verified_saved_scores_counts_and_checkpoint','full_checkpoint_and_native_recipe_verified','owned_release_and_clean_settlement_verified','training_seed_was_not_ranked','incumbent_unchanged']:assert proof[key] is True
   audit=proof['checkpoint_audit'];assert scope.sha(audit['path'])==audit['SHA256'];q=scope.read(audit['path'])
   assert q['status']=='VERIFIED_EXACT_RECOVERED_OR_FROZEN_SEED2_CHECKPOINTS' and q['training_seed']==seed and q['checkpoint_SHA256']==candidate['checkpoint_SHA256']
   assert q['recovery_protocol_SHA256']==scope.sha(scope.PROTOCOL)
   for key in ['recipe_exactly_matches_frozen_seed0','complete_model_keys_shapes_dtypes_finite_verified','policy_update_count_and_seed_streams_verified']:assert q[key] is True
   assert len(q['parents'])==2
   for a,b in zip(q['parents'],reference['parents']):assert a['checkpoint_SHA256']==b['checkpoint_SHA256'] and a['checkpoint_path']==b['checkpoint_path']
   for source,pin in q['source_SHA256'].items():assert scope.sha(source)==pin==auth['files_SHA256'][source]
   assert len(proof['array_and_exemplar_SHA256'])==10
   for source,pin in proof['array_and_exemplar_SHA256'].items():assert scope.sha(source)==pin
   evidence.update({str(path):scope.sha(path),audit['path']:audit['SHA256']})
   if seed==1:
    assert candidate['new_model_updates']==40000-scope.INITIAL and candidate['initial_completed_updates']==scope.INITIAL
    assert candidate['resume_parent_checkpoint_SHA256']==p['checkpoint_SHA256'] and candidate['original_interrupted_attempt']==p['original_attempt']
    qualification=scope.read(scope.NATIVE_PROOF)
    ledger=Path(dep['ledger_path']);settlement=scope.read(ledger.with_name(name+'-settlement.json'))
    assert proof['charged_GPU_seconds']==settlement['settlement']['charged_GPU_seconds'],'Confirmation charge differs from canonical settlement'
    previous=scope.read(p['latest_interruption_evidence']);previous_qual=scope.read(p['previous_native_qualification_path'])
    seed1_total_charge=p['wall_clock_envelope']['original_charged_GPU_seconds']+previous['charged_GPU_seconds']+previous_qual['charged_GPU_seconds']+settlement['settlement']['charged_GPU_seconds']+qualification['charged_GPU_seconds']
    evidence.update({p['latest_interruption_evidence']:scope.sha(p['latest_interruption_evidence']),p['previous_native_qualification_path']:scope.sha(p['previous_native_qualification_path'])})
    assert seed1_total_charge<=21600 and scope.read(ledger)['jobs'][0]['ended_unix']<=p['wall_clock_envelope']['absolute_recovery_deadline_unix']
   else:assert candidate['new_model_updates']==40000 and candidate['initial_completed_updates']==0
  records.append({'seed':seed,'attempt':name,'energy_score':candidate['energy_score'],'matched_baseline_energy_score':baseline['energy_score'],
                  'cell_energy_distance':candidate['cell_energy_distance'],'matched_baseline_cell_energy_distance':baseline['cell_energy_distance'],'spread_ratio':candidate['spread_ratio']})
 result={**stable_target(records),'records':records,'evidence_SHA256':evidence,'seed1_total_charged_GPU_seconds_including_failure_and_qualification':seed1_total_charge,'original_failed_seed1_attempt_retained':p['original_attempt'],'previous_failed_recovery_attempt_retained':scope.read(p['latest_interruption_evidence'])['attempt'],'original_confirmation_freeze_SHA256':p['original_confirmation_freeze_SHA256'],
         'pure_incumbent_SHA256':f['pure_incumbent_SHA256'],'incumbent_unchanged':True,'training_seeds_were_not_ranked':True,'each_seed_parent_training_updates':[20000,20000],
         'scope':'All three frozen training seeds and same four adaptive calibration donors, not twelve independent donors or external SOTA. Interrupted seed1 preserved and exactly continued; no replacement seed, fromscratch rerun or outcome dropping. Evaluation donors excluded.'}
 assert scope.sha(ROOT/'pure_incumbent.json')==f['pure_incumbent_SHA256']
 output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--output',required=True,type=Path);aggregate(parser.parse_args().output)
