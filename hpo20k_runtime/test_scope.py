"""Fixed endpoint, exact selected parameters and native next-pair equivalence."""
import copy,sys,unittest
from unittest.mock import patch
import binding as b,preset_scope as s,prepare
class Scope(unittest.TestCase):
 def setUp(self):
  self.reader=s.read;self.mock=patch.object(s,'read',side_effect=lambda p:dict(s.ENABLED) if p==s.PLAN else self.reader(p));self.mock.start()
 def tearDown(self):self.mock.stop()
 def test_exact_fresh20k_no_model_import(self):
  a,_=prepare.build_authorization(check_only=True);s.validate_authorization(a,require_runtime_proof=False)
  self.assertEqual((a['initial_completed_updates'],a['new_model_updates']),(0,20000));self.assertIsNone(a['source_model_path']);self.assertNotIn('torch',sys.modules)
 def test_other_trial_keys_refused(self):
  self.assertEqual(b.cell(dict(cell=s.cell(),cpu_fixture=False,operation='hpo20k_cell')),s.cell())
  for c,seed,phase in [('k002',0,'duration'),('c014',1,'duration'),('c014',0,'confirm')]:
   with self.assertRaises(AssertionError):s.cell(c,seed,phase)
 def test_no_parameter_retuning(self):
  q=s.read(s.proposal_path('c014'));s.validate_proposal(q)
  for k in q['hyperparameters']:
   d=copy.deepcopy(q);d['hyperparameters'][k]*=2
   with self.assertRaises(AssertionError):s.validate_proposal(d)
 def test_wrong_horizon_seed_warmstart_version_deadline_refused(self):
  a,_=prepare.build_authorization(check_only=True)
  for k,v in [('operation','hpo500_cell'),('fixed_endpoint',500),('metric_role','final_evaluation'),('reuse_completed_model',True),('is_continuation',True),('initial_completed_updates',500),('new_model_updates',19500),('particlegan_version','0.9.4'),('cap_seconds',3600),('source_model_path','existing.pt')]:
   d=copy.deepcopy(a);d[k]=v
   with self.assertRaises(AssertionError):s.validate_authorization(d,require_runtime_proof=False)
 def test_shared_GPU1_same_margins(self):
  import resource_profile as p
  q=p.validate_proposal();self.assertEqual(q['roles']['alz_core_pg'],{'gpu_bytes':2*2**30,'ram_bytes':6*2**30});self.assertEqual(q['host_free_floor_bytes'],12*2**30);self.assertEqual(q['driver_headroom_bytes'],3*2**30)
if __name__=='__main__':unittest.main(verbosity=2)
