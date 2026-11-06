"""Actual frozen configurations, all four seeds and strict fresh-start gates."""
import copy,sys,unittest
from unittest.mock import patch
import binding as b,preset_scope as s,prepare
class Scope(unittest.TestCase):
 def setUp(self):
  self.reader=s.read;self.mock=patch.object(s,'read',side_effect=lambda p:dict(s.ENABLED) if p==s.PLAN else self.reader(p));self.mock.start()
 def tearDown(self):self.mock.stop()
 def test_every_frozen_configuration_seed_binds_fresh20k(self):
  seen=set()
  for cid in s.ENABLED['candidate_ids']:
   for seed in range(1,5):
    a,_=prepare.build_authorization(candidate=cid,seed=seed,phase='confirm',check_only=True);s.validate_authorization(a,require_runtime_proof=False)
    self.assertEqual(a['cell'],s.cell(cid,seed,'confirm'));self.assertEqual((a['initial_completed_updates'],a['new_model_updates']),(0,20000))
    self.assertFalse(a['reuse_completed_model']);self.assertIsNone(a['source_model_path']);seen.add(a['allocation_id'])
  self.assertEqual(len(seen),4*len(s.ENABLED['candidate_ids']));self.assertNotIn('torch',sys.modules)
 def test_selection_seed_and_other_phase_not_admitted(self):
  for seed,phase in [(0,'confirm'),(5,'confirm'),(True,'confirm'),(1,'search')]:
   with self.assertRaises(AssertionError):s.cell('d000',seed,phase)
  with self.assertRaises(AssertionError):s.cell('d999',1,'confirm')
 def test_no_parameter_retuning_after_seed_confirmation_freeze(self):
  for cid in s.ENABLED['candidate_ids']:
   q=s.read(s.proposal_path(cid));s.validate_proposal(q)
   for k in q['hyperparameters']:
    changed=copy.deepcopy(q);changed['hyperparameters'][k]*=2
    with self.assertRaises(AssertionError):s.validate_proposal(changed)
 def test_old_weights_partial_horizon_version_and_reuse_refused(self):
  a,_=prepare.build_authorization(check_only=True)
  for k,v in [('fixed_endpoint',500),('new_model_updates',19500),('initial_completed_updates',500),('is_continuation',True),('reuse_completed_model',True),('source_model_path','old.pt'),('source_model_SHA256','old'),('phase','search'),('particlegan_version','0.9.6'),('metric_role','final_evaluation')]:
   changed=copy.deepcopy(a);changed[k]=v
   with self.assertRaises(AssertionError):s.validate_authorization(changed,require_runtime_proof=False)
 def test_matched_resampling_controls_and_unchanged_shared_profile(self):
  q=s.read(s.BASELINE_PROOF);self.assertEqual(set(q['metrics_by_seed']),{'1','2','3','4'});self.assertFalse(q['CUDA_initialized'])
  import resource_profile as p
  policy=p.validate_proposal();self.assertEqual(policy['roles']['alz_core_pg'],{'gpu_bytes':2*2**30,'ram_bytes':6*2**30});self.assertEqual(policy['host_free_floor_bytes'],12*2**30)
if __name__=='__main__':unittest.main(verbosity=2)
