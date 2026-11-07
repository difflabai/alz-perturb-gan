"""Synthetic full-chain refusal tests; no actual ALZ targets or GPU models."""
import copy,unittest
from benchmark097_selection_validation import resolve
import test_refinement097_followup_selection as fixtures
class FreezeValidation(unittest.TestCase):
 def setUp(self):
  f=fixtures.Followup();f.setUp();self.report=f.run_followup();self.paired=copy.deepcopy(f.selection)
  common=dict(all_planned_attempts_resolved=True,all_new_models_from_update_zero=True,activation_actually_reverted=True,owned_processes_and_leases_absent=True,heldout_values_read=False,native_version='0.9.7',EMA_decay=0,formulation_changed=False)
  self.a=dict(**common,status='INDEPENDENT_NATIVE097_PAIRED_REFINEMENT20_FITS_AUDIT_PASS',required_attempts=20,training_seeds=[0,1],summary=copy.deepcopy(self.paired))
  self.b=dict(**common,status='INDEPENDENT_NATIVE097_MATCHED_REFINEMENT_FOLLOWUP_AUDIT_PASS',required_attempts=3,reference_new_fits=0,training_seeds=[2,3,4],summary=copy.deepcopy(self.report))
  self.p=dict(status='FROZEN_MATCHED_SEED234_FOLLOWUP_AFTER_PAIRED_REFINEMENT',native_version='0.9.7',endpoint=20000,calibration_only=True,metric_role='calibration_selection',training_seeds=[2,3,4],selection_seeds=[0,1],all_new_training_from_scratch=True,initial_completed_updates=0,source_checkpoint=None,reference_models_refitted=False,reference_measurements_previously_observed=True,all_configurations_for_ranking=copy.deepcopy(f.selection['proposed_followup_configurations']),selected_configurations=[copy.deepcopy(f.selection['proposed_followup_configurations'][0])])
 def call(self):return resolve(self.a,self.paired,self.b,self.report,self.p)
 def sync(self):self.b['summary']=copy.deepcopy(self.report)
 def test_complete_chain_resolves_canonical_candidate_and_three_seed_mean(self):
  q=self.call();self.assertEqual(q['candidate_id'],'r000');self.assertAlmostEqual(q['followup_mean']['cell_energy_distance'],.2);self.assertTrue(q['all45_new_external_models_required']);self.assertFalse(q['SOTA_claim'])
 def test_missing_paired_fit_or_unclean_resource_closure_refuses(self):
  for key,value in [('required_attempts',19),('all_planned_attempts_resolved',False),('all_new_models_from_update_zero',False),('activation_actually_reverted',False),('owned_processes_and_leases_absent',False),('heldout_values_read',True)]:
   a=copy.deepcopy(self.a);a[key]=value
   with self.assertRaises(ValueError):resolve(a,self.paired,self.b,self.report,self.p)
 def test_missing_followup_seed_or_replacement_with_selection_seed_refuses(self):
  for rows in [self.report['candidates'][0]['per_seed'][:-1],[{**r,'seed':0} if r['seed']==4 else r for r in self.report['candidates'][0]['per_seed']]]:
   old=copy.deepcopy(self.report);self.report['candidates'][0]['per_seed']=rows;self.sync()
   with self.assertRaises(ValueError):self.call()
   self.report=old
  self.sync()
 def test_audited_false_winner_or_fabricated_mean_still_refuses(self):
  self.report['proposed_external_candidate']='d000';self.sync()
  with self.assertRaises(ValueError):self.call()
  self.report['proposed_external_candidate']='r000';self.report['candidates'][0]['followup_mean']['cell_energy_distance']=0;self.sync()
  with self.assertRaises(ValueError):self.call()
 def test_changed_native_formulation_or_horizon_refuses(self):
  for key,value in [('EMA_decay',.9),('native_version','0.9.6'),('formulation_changed',True)]:
   b=copy.deepcopy(self.b);b[key]=value
   with self.assertRaises(ValueError):resolve(self.a,self.paired,b,self.report,self.p)
  self.p['endpoint']=500
  with self.assertRaises(ValueError):self.call()
 def test_relabeling_known_reference_or_changing_hyperparameters_refuses(self):
  self.report['reference_new_fits']=3;self.sync()
  with self.assertRaises(ValueError):self.call()
  self.report['reference_new_fits']=0;self.report['candidates'][0]['hyperparameters']['generator_lr']=9;self.sync()
  with self.assertRaises(ValueError):self.call()
 def test_failed_seed_cannot_be_dropped_from_candidate_mean(self):
  self.report['candidates'][0]['per_seed'][0].update(finite=False,metrics=None);self.sync()
  with self.assertRaises(ValueError):self.call()
 def test_different_control_or_target_trace_refuses(self):
  self.report['candidates'][0]['per_seed'][0]['baseline_metrics']['energy_score']=9;self.sync()
  with self.assertRaises(ValueError):self.call()
  self.setUp();self.report['candidates'][0]['per_seed'][0]['metrics']['trace_manifest_SHA256']='OTHER';self.sync()
  with self.assertRaises(ValueError):self.call()
if __name__=='__main__':unittest.main(verbosity=2)
