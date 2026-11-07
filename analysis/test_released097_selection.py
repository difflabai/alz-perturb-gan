"""Synthetic failure, draw-contract, fresh-start and ranking checks."""
import copy,unittest
from released097_selection import summarize,proposed_confirmation
H=dict(generator_lr=.012,critic_lr=.018,particle_lr=.03,reg_coeff=1.,reg_kappa=1.,prior_reg=.01)
def fixture(seeds=(0,1)):
 candidates=[dict(candidate_id='a',hyperparameters=H),dict(candidate_id='b',hyperparameters=dict(H,prior_reg=.03))];rows=[]
 for c in candidates:
  for seed in seeds:
   m=dict(training_seed=seed,metric_role='calibration_selection',generation_replicates=2,donors=['d0','d1'],calibration_manifest_SHA256='cal',trace_manifest_SHA256='trace',sampling_contract='fixed',cell_energy_distance=.2 if c['candidate_id']=='a' else .3,energy_score=2.,spread_ratio=1.,within_condition_spread_ratio=1.,coverage80=.7)
   rows.append(dict(candidate_id=c['candidate_id'],hyperparameters=c['hyperparameters'],seed=seed,endpoint=20000,from_scratch=True,initial_completed_updates=0,calibration_only=True,clean_settlement=True,owned_processes_and_lease_absent=True,finite=True,physical_start_update=0,new_phase_model_updates=20000,metrics=m,baseline_metrics=dict(m,cell_energy_distance=.4,energy_score=2.1)))
 return candidates,rows
class Selection(unittest.TestCase):
 def test_complete_paired_selection_roles(self):
  c,r=fixture();q=summarize(c,r,[0,1]);self.assertEqual(q['ranking'],['a','b']);self.assertIsNone(q['proposed_external_candidate']);self.assertEqual([x['candidate_id'] for x in proposed_confirmation(q,'b')],['a','b'])
 def test_three_fresh_confirmation_seeds_rank_without_selection_seed_rescue(self):
  c,r=fixture((2,3,4));q=summarize(c,r,[2,3,4]);self.assertEqual(q['proposed_external_candidate'],'a');self.assertAlmostEqual(q['candidates'][0]['SD']['cell_energy_distance'],0.)
 def test_missing_duplicate_or_wrong_seed_refused(self):
  c,r=fixture()
  for rows in [r[:-1],r+[r[0]],r[:3]+[dict(r[3],seed=4)]]:
   with self.assertRaises(ValueError):summarize(c,rows,[0,1])
 def test_nonfinite_failure_retained_and_cannot_win(self):
  c,r=fixture();r[0].update(finite=False,metrics=None,failed_candidate_not_retried=True,failure=dict(error='FloatingPointError(nonfinite)'));q=summarize(c,r,[0,1]);self.assertEqual(q['ranking'],['b']);self.assertEqual(q['candidates'][0]['failed_seeds'],[0])
 def test_warmstart_partial_and_unsettled_rejected(self):
  for key,val in [('initial_completed_updates',500),('from_scratch',False),('new_phase_model_updates',19500),('clean_settlement',False),('resumed_same_trial',True)]:
   c,r=fixture();r[0][key]=val
   with self.assertRaises(ValueError):summarize(c,r,[0,1])
 def test_changed_targets_baseline_or_hyperparameters_refused(self):
  for key in ('donors','trace_manifest_SHA256','sampling_contract'):
   c,r=fixture();r[0]['metrics'][key]='different'
   with self.assertRaises(ValueError):summarize(c,r,[0,1])
  c,r=fixture();r[2]['baseline_metrics']['energy_score']=4.
  with self.assertRaises(ValueError):summarize(c,r,[0,1])
if __name__=='__main__':unittest.main(verbosity=2)
