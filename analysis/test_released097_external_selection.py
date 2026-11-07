"""Synthetic source-fidelity, incomplete-seed and candidate-freeze refusal tests."""
import copy,json,unittest
from released097_selection import summarize,proposed_confirmation,FORMULATION
from released097_external_selection import resolve
from test_released097_selection import fixture

def chain():
 candidates,rows=fixture();summary=summarize(candidates,rows,[0,1]);summary['reference_id']='b';summary['proposed_confirmation_configurations']=proposed_confirmation(summary,'b')
 shared=dict(formulation=FORMULATION,all_planned_attempts_resolved=True,all_new_models_from_update_zero=True,activation_actually_reverted=True,owned_processes_and_leases_absent=True,heldout_values_read=False,CUDA_initialized=False,model_parity_proof_path='/fixture/proof.json',model_parity_proof_SHA256='fixtureproof')
 selection=dict(shared,status='INDEPENDENT_RELEASED097_FRESH_BASELINE4_AUDIT_PASS',training_seeds=[0,1],required_attempts=4,summary=summary)
 c,rows=fixture((2,3,4));confirmation=dict(shared,status='INDEPENDENT_RELEASED097_MATCHED_CONFIRMATION_AUDIT_PASS',training_seeds=[2,3,4],required_attempts=6,summary=summarize(c,rows,[2,3,4]))
 protocol=dict(status='FROZEN_RELEASED097_MATCHED_SEED234_CONFIRMATION',native_version='0.9.7',endpoint=20000,training_seeds=[2,3,4],selection_seeds=[0,1],all_new_training_from_scratch=True,initial_completed_updates=0,source_checkpoint=None,calibration_only=True,metric_role='calibration_selection',formulation=FORMULATION,automatic_promotion=False,SOTA_claim=False,selected_configurations=summary['proposed_confirmation_configurations'])
 return json.loads(json.dumps((selection,confirmation,protocol)))
class Freeze(unittest.TestCase):
 def test_fully_finite_declared_chain_yields_only_fresh_confirmation_winner(self):
  q=resolve(*chain());self.assertEqual(q['candidate_id'],'a');self.assertTrue(q['all45_new_external_models_required']);self.assertFalse(q['SOTA_claim'])
 def test_historical_sampled_row_or_unbound_thermal_refused(self):
  for key,val in [('prior_regularizer_domain','sampled_rows'),('critic_thermal','unbound'),('EMA_decay',.99)]:
   s,c,p=chain();c['formulation'][key]=val
   with self.assertRaises(ValueError):resolve(s,c,p)
 def test_old_audit_status_cannot_qualify_by_relabeling_a_version(self):
  s,c,p=chain();s['status']='INDEPENDENT_FRESH_NATIVE097_MATCHED_SEED_CONFIRMATION_AUDIT_PASS'
  with self.assertRaises(ValueError):resolve(s,c,p)
 def test_missing_confirmation_seed_or_exploratory_seed_rescue_refused(self):
  for seed in (0,1,5):
   s,c,p=chain();c['summary']['candidates'][0]['per_seed'][-1]['seed']=seed
   with self.assertRaises(ValueError):resolve(s,c,p)
 def test_wrong_frozen_hyperparameters_and_proof_and_freshness_refused(self):
  for key,val in [('all_new_models_from_update_zero',False),('model_parity_proof_SHA256','oldproof'),('owned_processes_and_leases_absent',False)]:
   s,c,p=chain();c[key]=val
   with self.assertRaises(ValueError):resolve(s,c,p)
  s,c,p=chain();p['selected_configurations'][0]['hyperparameters']['prior_reg']*=2
  with self.assertRaises(ValueError):resolve(s,c,p)
 def test_arithmetic_rank_or_failed_seed_omission_refused(self):
  s,c,p=chain();c['summary']['candidates'][0]['mean']['cell_energy_distance']=.001
  with self.assertRaises(ValueError):resolve(s,c,p)
  s,c,p=chain();c['summary']['candidates'][0]['per_seed'][0]['finite']=False
  with self.assertRaises(ValueError):resolve(s,c,p)
 def test_calibration_target_leak_or_checkpoint_reuse_refused(self):
  for key,val in [('source_checkpoint','old.pt'),('calibration_only',False)]:
   s,c,p=chain();p[key]=val
   with self.assertRaises(ValueError):resolve(s,c,p)
  s,c,p=chain();c['heldout_values_read']=True
  with self.assertRaises(ValueError):resolve(s,c,p)
if __name__=='__main__':unittest.main(verbosity=2)
