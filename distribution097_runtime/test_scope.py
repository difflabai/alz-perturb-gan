"""Native0.9.7 stage1 cannot reuse trained states or change frozen arms."""
import copy,sys,unittest
from unittest.mock import patch
import binding as b,preset_scope as s,prepare
class Scope(unittest.TestCase):
 def setUp(self):
  self.reader=s.read;self.mock=patch.object(s,'read',side_effect=lambda p:dict(s.ENABLED) if p==s.PLAN else self.reader(p));self.mock.start()
 def tearDown(self):self.mock.stop()
 def test_all_three_arms_start_at_zero_without_model_import(self):
  for cid in s.ENABLED['candidate_ids']:
   a,_=prepare.build_authorization(candidate=cid,check_only=True);s.validate_authorization(a,require_runtime_proof=False)
   self.assertEqual((a['initial_completed_updates'],a['new_model_updates']),(0,20000));self.assertIsNone(a['source_model_path']);self.assertFalse(a['is_continuation']);self.assertFalse(a['reuse_completed_model'])
  self.assertNotIn('torch',sys.modules)
 def test_undeclared_trial_keys_refused(self):
  for cid in s.ENABLED['candidate_ids']:self.assertEqual(b.cell(dict(cell=s.cell(cid),cpu_fixture=False,operation='distribution097_cell')),s.cell(cid))
  for cid,seed,phase in [('d003',0,'search'),('d000',1,'search'),('d000',0,'confirm')]:
   with self.assertRaises(AssertionError):s.cell(cid,seed,phase)
 def test_no_parameter_retuning_after_freeze(self):
  for cid in s.ENABLED['candidate_ids']:
   q=s.read(s.proposal_path(cid));s.validate_proposal(q)
   for k in q['hyperparameters']:
    d=copy.deepcopy(q);d['hyperparameters'][k]*=2
    with self.assertRaises(AssertionError):s.validate_proposal(d)
 def test_old_weight_resume_partial_clock_version_and_deadline_refused(self):
  a,_=prepare.build_authorization(check_only=True)
  for k,v in [('operation','hpo20k_cell'),('fixed_endpoint',500),('metric_role','final_evaluation'),('reuse_completed_model',True),('is_continuation',True),('initial_completed_updates',500),('new_model_updates',19500),('particlegan_version','0.9.6'),('cap_seconds',3600),('source_model_path','existing.pt'),('source_model_SHA256','parent')]:
   d=copy.deepcopy(a);d[k]=v
   with self.assertRaises(AssertionError):s.validate_authorization(d,require_runtime_proof=False)
 def test_shared_GPU1_same_margins(self):
  import resource_profile as p
  q=p.validate_proposal();self.assertEqual(q['roles']['alz_core_pg'],{'gpu_bytes':2*2**30,'ram_bytes':6*2**30});self.assertEqual(q['host_free_floor_bytes'],12*2**30);self.assertEqual(q['driver_headroom_bytes'],3*2**30)
if __name__=='__main__':unittest.main(verbosity=2)
