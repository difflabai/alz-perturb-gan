"""Confined counterfactual ensemble wire fixtures; no fitted models or native state."""
import contextlib,copy,importlib.util,io,json,sys
from pathlib import Path
from unittest.mock import patch
import unittest
ROOT=Path('/ml2/alz');sys.path[:0]=[str(ROOT/'analysis'),str(ROOT/'confirmation_runtime')]
import test_pure_confirmation_artifacts as fixture_base
import aggregate_validation as common
import aggregate_pure_confirmation as base_aggregate
import ensemble_confirmation_scope as scope
spec=importlib.util.spec_from_file_location('ensemble_aggregate_fixture_target',ROOT/'confirmation_runtime/aggregate.py')
aggregate=importlib.util.module_from_spec(spec);spec.loader.exec_module(aggregate)

class EnsembleArtifacts(fixture_base.ArtifactBindings):
 def setUp(self):
  super().setUp()
  self.extension=self.root/'confirmation_runtime';self.extension.mkdir()
  self.proof=self.root/'analysis/PRIVATE-qualification.json'
  self.write(self.root/'ensemble_parents.json',{'fixture':'COUNTERFACTUAL_METADATA_NOT_MODELS'})
  self.write(self.root/'ensemble_protocol.json',{'fixture':'COUNTERFACTUAL_METADATA_NOT_PROTOCOL'})
  config={'learned_weight':.25,'mask_seed':20261108,'bags_per_stage':200}
  for seed,candidate in enumerate(self.candidates):
   candidate.update(pure_formulation='pg8_learned_ensemble75',method='pg_pure_ensemble75',completed_updates=40000,fixed_endpoint=40000,
                    operation='evaluate_pure_ensemble' if seed==0 else 'confirm_pure_ensemble',ensemble_config=config,
                    parent_training_updates_total=40000,parent_completed_updates=[20000,20000],new_model_updates=0 if seed==0 else 40000,
                    initial_completed_updates=40000 if seed==0 else 0)
   candidate.pop('network_rate_config')
   self.deployments[seed]['endpoint']=40000
  self.write(self.root/'pure_incumbent.json',{**self.candidates[0],'candidate_commit':'PRIVATE_METADATA_FIXTURE'})
  decision=self.root/'analysis'/(self.attempt0+'-result.json')
  self.write(decision,{'candidate':self.candidates[0],'decision':{'accept':True},'independent_saved_scores_checkpoint_and_count_exemplars_verified':True})
  # Seed0 fake checkpoint bytes remain unchanged, so the predeclared seed IDs remain stable.
  f=common.read(self.freeze);f.update(seed0_candidate=self.candidates[0],pure_incumbent_SHA256=common.sha(self.root/'pure_incumbent.json'),seed0_decision_SHA256=common.sha(decision))
  # Existing seed0 wire authorization has no future confirmation pins.
  self.deployments[0]['authorization_SHA256']=common.sha(self.deployments[0]['authorization_path']);self.write(Path(self.deployments[0]['authorization_path']).with_name('deployment.json'),self.deployments[0])
  f['seed0_authorization_SHA256']=self.deployments[0]['authorization_SHA256'];self.write(self.freeze,f)
  self.write(self.proof,{'status':'PASSED_CPU_FROZEN_TWO_PARENT_CONFIRMATION','freeze_SHA256':common.sha(self.freeze),'GPU_allocation_created':False,'native_confirmation_completed':False,'source_SHA256':{},'fixture':'PRIVATE WIRE ONLY, NOT A REAL QUALIFICATION'})
  for seed in (0,1,2):
   candidate=self.candidates[seed];dep=self.deployments[seed];auth=self.auths[seed]
   if seed:
    parents=[]
    for name,label in [('PG8','pg8'),('learned_observation','learned')]:
     p=Path(dep['output_dir'])/'fit'/label/'final.pt';p.parent.mkdir(parents=True);p.write_bytes(f'NOT_A_TORCH_MODEL_{seed}_{name}'.encode())
     self.write(p.with_name('checkpoint_receipt.json'),{'status':'completed','path':'final.pt','completed_updates':20000,'SHA256':common.sha(p)})
     parents.append({'name':name,'checkpoint_path':str(p),'checkpoint_SHA256':common.sha(p),'completed_updates':20000})
    reference={'schema':1,'type':'confirmed_pure_ensemble_complete_parent_references','status':'completed','fold':0,'seed':seed,'completed_updates':40000,'fixed_endpoint':40000,'parent_training_updates_total':40000,'new_model_updates':40000,'ensemble_config':config,'confirmation_freeze_SHA256':common.sha(self.freeze),'parents':parents}
    cp=Path(dep['output_dir'])/'fit/ensemble.json';self.write(cp,reference);candidate.update(checkpoint_path=str(cp),checkpoint_SHA256=common.sha(cp))
    auth.update(cell=self.ids[str(seed)]['cell'],allocation_id=self.ids[str(seed)]['allocation_id'],attempt_id=self.ids[str(seed)]['attempt_id'],label=self.ids[str(seed)]['attempt_id'],operation='confirm_pure_ensemble',qualification_only=False,
      cap_seconds=21600,fixed_endpoint=40000,initial_completed_updates=0,new_model_updates=40000,parent_updates=[20000,20000],hybrid_config=None,variant_config=candidate['variant_config'],pure_formulation='pg8_learned_ensemble75',ensemble_config=config,ensemble_weight=.25,
      confirmation_freeze_SHA256=common.sha(self.freeze),confirmation_qualification_SHA256=common.sha(self.proof),source_review_passed=True)
    for p in [self.freeze,self.proof,self.protocol,self.root/'pure_incumbent.json',self.root/'ensemble_parents.json',self.root/'ensemble_protocol.json',*[Path(q['path']) for q in f['matched_baselines'].values()]]:auth['files_SHA256'][str(p)]=common.sha(p)
    self.write(dep['authorization_path'],auth);dep['authorization_SHA256']=common.sha(dep['authorization_path'])
    audit=self.root/'analysis'/f'checkpoint-seed{seed}-fixture.json'
    self.write(audit,{'status':'VERIFIED_FROZEN_PURE_ENSEMBLE_CONFIRMATION_CHECKPOINTS','training_seed':seed,'checkpoint_SHA256':candidate['checkpoint_SHA256'],'seed0_checkpoint_SHA256':self.candidates[0]['checkpoint_SHA256'],
       'confirmation_freeze_SHA256':common.sha(self.freeze),'recipe_exactly_matches_frozen_seed0':True,'complete_model_keys_shapes_dtypes_finite_verified':True,'policy_update_count_and_seed_streams_verified':True,'parents':parents,
       'source_SHA256':{str(self.root/'scorer.py'):common.sha(self.root/'scorer.py')},'fixture':'PRIVATE METADATA ONLY. NO CHECKPOINT AUDIT EXECUTED.'})
    self.write(self.root/'analysis'/(self.ids[str(seed)]['attempt_id']+'-confirmation.json'),{
       'status':'PURE_ENSEMBLE_SEED_CONFIRMATION_RECORDED_NO_MODEL_SELECTION','candidate':candidate,'baseline':self.baselines[seed],'confirmation_freeze_SHA256':common.sha(self.freeze),
       'independently_verified_saved_scores_counts_and_checkpoint':True,'full_checkpoint_and_native_recipe_verified':True,'owned_release_and_clean_settlement_verified':True,'training_seed_was_not_ranked':True,'incumbent_unchanged':True,
       'checkpoint_audit':{'path':str(audit),'SHA256':common.sha(audit)},'array_and_exemplar_SHA256':{},'fixture':'PRIVATE WIRE ONLY. NOT MODEL EVIDENCE.'})
   self.write(Path(dep['authorization_path']).with_name('deployment.json'),dep)
   self.write(Path(dep['output_dir'])/'completed.json',candidate)
  self.write(self.root/'incumbent.json',{'fixture':'UNCHANGED_HYBRID_PRIVATE_METADATA'})
 def run_aggregate(self):
  with patch.object(aggregate,'ROOT',self.root),patch.object(common,'ROOT',self.root),patch.object(base_aggregate,'ROOT',self.root), \
       patch.object(scope,'ROOT',self.root),patch.object(scope,'HERE',self.extension),patch.object(scope,'FREEZE',self.freeze),patch.object(scope,'PROOF',self.proof),contextlib.redirect_stdout(io.StringIO()):
   aggregate.aggregate(self.output)
 def test_unplanned_repeat_attempt_refuses_without_output(self):
  auth=self.auths[1];auth['cell']={**auth['cell'],'experiment':'UNPLANNED_REPEAT'};self.write(self.deployments[1]['authorization_path'],auth)
  self.deployments[1]['authorization_SHA256']=common.sha(self.deployments[1]['authorization_path']);self.write(Path(self.deployments[1]['authorization_path']).with_name('deployment.json'),self.deployments[1])
  with self.assertRaises(AssertionError):self.run_aggregate()
  self.assertFalse(self.output.exists())
 def test_missing_one_of_two_parent_checkpoints_refuses(self):
  reference=common.read(self.candidates[1]['checkpoint_path']);Path(reference['parents'][1]['checkpoint_path']).unlink()
  with self.assertRaises(FileNotFoundError):self.run_aggregate()
  self.assertFalse(self.output.exists())
 def test_changed_parent_receipt_refuses(self):
  reference=common.read(self.candidates[1]['checkpoint_path']);p=Path(reference['parents'][0]['checkpoint_path']).with_name('checkpoint_receipt.json');q=common.read(p);q['completed_updates']=19999;self.write(p,q)
  with self.assertRaises(AssertionError):self.run_aggregate()
  self.assertFalse(self.output.exists())
if __name__=='__main__':unittest.main()
