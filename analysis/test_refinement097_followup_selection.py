import copy,unittest
from refinement097_followup_selection import summarize
H=dict(generator_lr=.006,critic_lr=.009,particle_lr=.015,reg_coeff=1.,reg_kappa=.5,prior_reg=.01)
def metric(seed,d,energy=2.,spread=1.):
 return dict(training_seed=seed,metric_role='calibration_selection',generation_replicates=2,donors=['SYNTH_A'],calibration_manifest_SHA256='SYNTH_TARGET',trace_manifest_SHA256='SYNTH_TRACE',sampling_contract='SYNTH_TRACE_CONTRACT',
  cell_energy_distance=d,energy_score=energy,spread_ratio=spread,within_condition_spread_ratio=spread,coverage80=.8)
class Followup(unittest.TestCase):
 def setUp(self):
  self.selection=dict(all_planned_attempts_resolved=True,heldout_used=False,proposed_followup_configurations=[dict(candidate_id='r000',hyperparameters={**H,'generator_lr':.003},roles=['primary_distance']),dict(candidate_id='d000',hyperparameters=H,roles=['reference'])])
  self.baseline={str(s):metric(s,.24,2.1) for s in (2,3,4)}
  def row(cid,h,s,d):return dict(candidate_id=cid,hyperparameters=h,seed=s,endpoint=20000,initial_completed_updates=0,from_scratch=True,finite=True,metrics=metric(s,d),baseline_metrics=self.baseline[str(s)])
  self.new=[row('r000',self.selection['proposed_followup_configurations'][0]['hyperparameters'],s,d) for s,d in zip((2,3,4),(.1,.2,.3))]
  self.reference=[dict(**row('d000',H,s,.35),read_only_previously_measured_reference=True,new_model_fitted=False,new_GPU_charge=0) for s in (2,3,4)]
 def run_followup(self):return summarize(self.selection,self.new,self.reference,self.baseline)
 def test_three_new_seeds_mean_SD_and_prior_reference_labels(self):
  q=self.run_followup();self.assertEqual(q['proposed_external_candidate'],'r000');e=q['candidates'][0]
  self.assertAlmostEqual(e['followup_mean']['cell_energy_distance'],.2);self.assertAlmostEqual(e['followup_SD']['cell_energy_distance'],.1)
  self.assertEqual(q['required_new_fits'],3);self.assertEqual(q['reference_new_fits'],0);self.assertTrue(q['selection_seeds0_and1_excluded_from_followup_ranking']);self.assertFalse(q['automatic_external_freeze'])
 def test_missing_duplicate_or_selection_seed_cannot_replace_followup_seed(self):
  for rows in [self.new[:-1],self.new+[self.new[0]],self.new[:-1]+[self.new[0]],self.new[:-1]+[{**self.new[-1],'seed':0}]]:
   with self.assertRaises(ValueError):summarize(self.selection,rows,self.reference,self.baseline)
 def test_failed_new_seed_is_retained_and_cannot_rank(self):
  self.new[1].update(finite=False,metrics=None);q=self.run_followup();self.assertEqual(q['proposed_external_candidate'],'d000');self.assertEqual(q['candidates'][0]['failed_seeds'],[3]);self.assertIsNone(q['candidates'][0]['per_seed'][1]['metrics'])
 def test_reference_cannot_be_labeled_new_training_and_must_be_complete(self):
  for key,value in [('read_only_previously_measured_reference',False),('new_model_fitted',True),('new_GPU_charge',1)]:
   r=copy.deepcopy(self.reference);r[0][key]=value
   with self.assertRaises(ValueError):summarize(self.selection,self.new,r,self.baseline)
  with self.assertRaises(ValueError):summarize(self.selection,self.new,self.reference[:-1],self.baseline)
 def test_collapsed_primary_stays_visible_but_is_not_joint_quality(self):
  for r in self.new:r['metrics']['spread_ratio']=.2
  q=self.run_followup();self.assertEqual(q['proposed_external_candidate'],'r000');self.assertFalse(q['proposed_candidate_joint_quality']);self.assertEqual(q['joint_quality_candidate'],'d000')
 def test_wrong_horizon_warmstart_changed_control_or_target_refuses(self):
  for key,value in [('endpoint',500),('initial_completed_updates',500),('from_scratch',False),('hyperparameters',{**H,'generator_lr':9})]:
   r=copy.deepcopy(self.new);r[0][key]=value
   with self.assertRaises(ValueError):summarize(self.selection,r,self.reference,self.baseline)
  r=copy.deepcopy(self.new);r[0]['baseline_metrics']['cell_energy_distance']=0
  with self.assertRaises(ValueError):summarize(self.selection,r,self.reference,self.baseline)
  r=copy.deepcopy(self.new);r[0]['metrics']['trace_manifest_SHA256']='OTHER'
  with self.assertRaises(ValueError):summarize(self.selection,r,self.reference,self.baseline)
if __name__=='__main__':unittest.main(verbosity=2)
