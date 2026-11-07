import copy,unittest
from refinement097_selection import summarize
H=dict(generator_lr=.006,critic_lr=.009,particle_lr=.015,reg_coeff=1.,reg_kappa=.5,prior_reg=.01)
def metrics(seed,distance,energy=2.,spread=1.):
 return dict(training_seed=seed,donors=['SYNTH_A','SYNTH_B'],calibration_manifest_SHA256='SYNTH_TARGET',trace_manifest_SHA256='SYNTH_TRACE',sampling_contract='SYNTH_CONTRACT',
  cell_energy_distance=distance,energy_score=energy,spread_ratio=spread,within_condition_spread_ratio=spread,coverage80=.8)
class Selection(unittest.TestCase):
 def setUp(self):
  self.candidates=[dict(id='r000',hyperparameters={**H,'generator_lr':.003}),dict(id='r001',hyperparameters={**H,'generator_lr':.009})]
  self.baselines={str(s):metrics(s,.24,2.1) for s in (0,1)}
  def row(cid,h,seed,value):return dict(candidate_id=cid,hyperparameters=h,seed=seed,endpoint=20000,from_scratch=True,initial_completed_updates=0,finite=True,metrics=metrics(seed,value),baseline_metrics=self.baselines[str(seed)])
  self.rows=[row(c['id'],c['hyperparameters'],s,v) for c,values in zip(self.candidates,[(.1,.5),(.25,.25)]) for s,v in enumerate(values)]
  self.reference=[row('d000',H,s,.29) for s in (0,1)]
 def run_selection(self):return summarize(self.candidates,self.rows,self.reference,self.baselines)
 def test_paired_mean_beats_lucky_seed0_and_reference_is_not_a_new_fit(self):
  q=self.run_selection();self.assertEqual(q['ranking_by_two_seed_mean_distance'],['r001','d000','r000'])
  self.assertTrue(q['reference_models_not_refitted']);self.assertEqual(q['selection_seed_set'],[0,1]);self.assertFalse(q['automatic_external_freeze'])
  winner=next(c for c in q['candidates'] if c['candidate_id']=='r001');self.assertEqual(winner['mean']['cell_energy_distance'],.25)
 def test_failed_seed_retained_and_configuration_cannot_rank_from_one_seed(self):
  self.rows[1].update(finite=False,metrics=None);q=self.run_selection()
  self.assertNotIn('r000',q['ranking_by_two_seed_mean_distance']);bad=next(c for c in q['candidates'] if c['candidate_id']=='r000');self.assertEqual(bad['failed_seeds'],[1]);self.assertIsNone(bad['per_seed'][1]['metrics'])
 def test_missing_duplicate_wrong_seed_and_reference_missing_refuse(self):
  for rows in [self.rows[:-1],self.rows+[self.rows[0]],self.rows[:-1]+[self.rows[0]]]:
   with self.assertRaises(ValueError):summarize(self.candidates,rows,self.reference,self.baselines)
  with self.assertRaises(ValueError):summarize(self.candidates,self.rows,self.reference[:1],self.baselines)
 def test_collapse_does_not_hide_primary_distance_winner_or_quality_alternative(self):
  for row in self.rows[:2]:row['metrics']=metrics(row['seed'],.1,1.8,.2)
  q=self.run_selection();self.assertEqual(q['ranking_by_two_seed_mean_distance'][0],'r000')
  self.assertEqual([c['candidate_id'] for c in q['proposed_followup_configurations']],['r000','r001','d000'])
  self.assertEqual(q['proposed_followup_configurations'][0]['roles'],['primary_distance'])
 def test_warm_start_partial_horizon_changed_settings_and_nonfinite_metric_refuse(self):
  for key,value in [('from_scratch',False),('initial_completed_updates',500),('endpoint',500),('hyperparameters',{**H,'prior_reg':3})]:
   changed=copy.deepcopy(self.rows);changed[0][key]=value
   with self.assertRaises(ValueError):summarize(self.candidates,changed,self.reference,self.baselines)
  changed=copy.deepcopy(self.rows);changed[0]['metrics']['cell_energy_distance']=float('nan')
  with self.assertRaises(ValueError):summarize(self.candidates,changed,self.reference,self.baselines)
 def test_changed_resampling_or_target_contract_cannot_pass(self):
  changed=copy.deepcopy(self.rows);changed[0]['baseline_metrics']['cell_energy_distance']=.2
  with self.assertRaises(ValueError):summarize(self.candidates,changed,self.reference,self.baselines)
  changed=copy.deepcopy(self.rows);changed[0]['metrics']['trace_manifest_SHA256']='SOME_OTHER_TRACE'
  with self.assertRaises(ValueError):summarize(self.candidates,changed,self.reference,self.baselines)
if __name__=='__main__':unittest.main(verbosity=2)
