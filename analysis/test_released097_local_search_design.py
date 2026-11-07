"""Synthetic endpoint, parameter-direction and full-comparison time-budget checks."""
import copy,unittest
from released097_selection import summarize,FORMULATION
from released097_local_search_design import design
from test_released097_selection import fixture,H

def audit():
 c,r=fixture();mapping={'a':'f000','b':'f001'}
 for x in c:
  x['candidate_id']=mapping[x['candidate_id']];x['hyperparameters']=dict(H) if x['candidate_id']=='f000' else dict(H,generator_lr=.006,critic_lr=.009,particle_lr=.015,reg_kappa=.5)
 for x in r:
  x['candidate_id']=mapping[x['candidate_id']];x['hyperparameters']=next(q['hyperparameters'] for q in c if q['candidate_id']==x['candidate_id'])
 return dict(status='INDEPENDENT_RELEASED097_FRESH_BASELINE4_AUDIT_PASS',formulation=FORMULATION,training_seeds=[0,1],required_attempts=4,all_planned_attempts_resolved=True,all_new_models_from_update_zero=True,activation_actually_reverted=True,owned_processes_and_leases_absent=True,heldout_values_read=False,summary=summarize(c,r,[0,1]),total_actual_GPU_seconds=4*1400.)
class LocalDesign(unittest.TestCase):
 def test_full_grid_is_twelve_new_fresh_full_endpoint_fits(self):
  q=design(audit(),0,60*3600,34.3);self.assertEqual(q['required_new_fits'],12);self.assertEqual(q['anchor_candidate_id'],'f000');self.assertEqual(q['directional_factor_for_rates_and_kappa'],2.)
  for c in q['candidates']:
   self.assertEqual(c['endpoint'],20000);self.assertEqual(c['training_seeds'],[0,1]);self.assertIsNone(c['source_checkpoint']);self.assertEqual(sum(v!=q['anchor_hyperparameters'][k] for k,v in c['hyperparameters'].items()),1)
 def test_full_external_and_confirmation_reserves_bound_search_size(self):
  a=audit();q=design(a,0,45*3600,34.3);f=q['forecast'];self.assertLessEqual(f['new_search_seconds']+f['external45_models_seconds']+f['fresh_confirmation_up_to9fits_seconds']+f['handoff_and_cleanup_seconds'],f['remaining_window_seconds']);self.assertEqual(q['required_new_fits']%2,0);self.assertEqual(q['external_required_fresh_models'],45)
 def test_too_little_search_slack_prioritizes_full_confirmation_and_matrix(self):
  q=design(audit(),0,42*3600,34.3);self.assertEqual(q['required_new_fits'],0);self.assertFalse(q['external_scope_reduced'])
 def test_half_rate_anchor_uses_lower_direction_without_formulation_change(self):
  a=audit();c,r=fixture();
  for x in c:
   x['candidate_id']='f000' if x['candidate_id']=='a' else 'f001';x['hyperparameters']=dict(H) if x['candidate_id']=='f000' else dict(H,generator_lr=.006,critic_lr=.009,particle_lr=.015,reg_kappa=.5)
  for x in r:
   old=x['candidate_id'];x['candidate_id']='f000' if old=='a' else 'f001';x['metrics']['cell_energy_distance']=.4 if old=='a' else .2;x['hyperparameters']=next(q['hyperparameters'] for q in c if q['candidate_id']==x['candidate_id'])
  a['summary']=summarize(c,r,[0,1]);q=design(a,0,60*3600,34.3);self.assertEqual(q['anchor_candidate_id'],'f001');self.assertEqual(q['directional_factor_for_rates_and_kappa'],.5);self.assertFalse(q['formulation_changed'])
 def test_incomplete_or_old_variant_audit_refused(self):
  for key,val in [('status','OLD_ADAPTER'),('all_planned_attempts_resolved',False),('heldout_values_read',True),('required_attempts',3)]:
   a=audit();a[key]=val
   with self.assertRaises(ValueError):design(a,0,60*3600,34.3)
 def test_no_automatic_activation_or_sota_claim(self):
  q=design(audit(),0,60*3600,34.3);self.assertFalse(q['automatic_activation']);self.assertFalse(q['SOTA_claim']);self.assertTrue(q['measured_baseline_controls_previously_observed'])
if __name__=='__main__':unittest.main(verbosity=2)
