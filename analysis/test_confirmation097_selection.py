"""Prospective rules tested without real calibration or evaluation values."""
import copy,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from confirmation097_selection import choose_configurations,summarize_confirmation
H=dict(generator_lr=.006,critic_lr=.009,particle_lr=.015,reg_coeff=1.,reg_kappa=.5,prior_reg=.01)
def selection():
 return [dict(candidate_id=f'd00{i}',finite=True,hyperparameters={**H,'generator_lr':.006+i*.001},
              metrics=dict(cell_energy_distance=.2+i*.01,energy_score=1.9,spread_ratio=1.)) for i in range(6)]
def metric(seed,distance,energy=1.8,spread=1.):
 return dict(metric_role='calibration_selection',generation_replicates=2,training_seed=seed,cell_energy_distance=distance,
             energy_score=energy,spread_ratio=spread,within_condition_spread_ratio=.9,coverage80=.8,
             donors=['a','b','c','d'],calibration_manifest_SHA256='calibration',trace_manifest_SHA256='trace',sampling_contract='shared')
def outcomes(selected):
 return [dict(candidate_id=c['candidate_id'],hyperparameters=c['hyperparameters'],seed=seed,finite=True,
              endpoint=20000,initial_completed_updates=0,from_scratch=True,
              metrics=metric(seed,.2+.005*seed),baseline_metrics=metric(seed,.3,2.)) for c in selected for seed in range(1,5)]
class Selection(unittest.TestCase):
 def test_reference_winner_is_one_configuration_with_three_roles(self):
  q=choose_configurations(selection(),2.)
  self.assertEqual(len(q),1);self.assertEqual(q[0]['candidate_id'],'d000')
  self.assertEqual(set(q[0]['roles']),{'primary_distance','joint_quality','reference'})
 def test_collapsed_distance_and_joint_quality_winners_remain_distinct(self):
  rows=selection();rows[1]['metrics'].update(cell_energy_distance=.1,spread_ratio=.1)
  rows[2]['metrics']['cell_energy_distance']=.15
  q=choose_configurations(rows,2.)
  self.assertEqual([x['candidate_id'] for x in q],['d001','d002','d000'])
  self.assertEqual([x['roles'] for x in q],[['primary_distance'],['joint_quality'],['reference']])
 def test_missing_sixth_endpoint_and_nonfinite_reference_refused(self):
  with self.assertRaises(ValueError):choose_configurations(selection()[:5],2.)
  rows=selection();rows[0]['finite']=False
  with self.assertRaises(ValueError):choose_configurations(rows,2.)
 def test_fresh_four_seed_means_and_initialization_sd_do_not_use_seed0(self):
  selected=choose_configurations(selection(),2.);q=summarize_confirmation(selected,outcomes(selected));c=q['candidates'][0]
  self.assertAlmostEqual(c['fresh_seed_mean']['cell_energy_distance'],.2125)
  self.assertGreater(c['fresh_seed_SD']['cell_energy_distance'],0)
  self.assertEqual(q['ranking_seed_set'],[1,2,3,4]);self.assertTrue(q['selection_seed0_excluded_from_confirmation_ranking'])
  self.assertEqual(q['proposed_external_candidate'],'d000');self.assertFalse(q['SOTA_claim'])
 def test_failed_seed_not_omitted_or_replaced_with_selection_seed(self):
  selected=choose_configurations(selection(),2.);rows=outcomes(selected);rows[2].update(finite=False,metrics=None,baseline_metrics=None)
  q=summarize_confirmation(selected,rows);self.assertIsNone(q['proposed_external_candidate'])
  self.assertEqual(q['candidates'][0]['failed_seeds'],[3])
  with self.assertRaises(ValueError):summarize_confirmation(selected,rows[:3])
  rows=outcomes(selected);rows[2]['seed']=0
  with self.assertRaises(ValueError):summarize_confirmation(selected,rows)
 def test_warmstart_wrong_endpoint_and_cross_configuration_baseline_drift_refused(self):
  rows=selection();rows[1]['metrics']['cell_energy_distance']=.1;selected=choose_configurations(rows,2.)
  for field,value in [('initial_completed_updates',500),('from_scratch',False),('endpoint',500)]:
   result=outcomes(selected);result[0][field]=value
   with self.assertRaises(ValueError):summarize_confirmation(selected,result)
  result=outcomes(selected);result[4]['baseline_metrics']['cell_energy_distance']+=.01
  with self.assertRaisesRegex(ValueError,'resampling differs'):summarize_confirmation(selected,result)

if __name__=='__main__':unittest.main(verbosity=2)
