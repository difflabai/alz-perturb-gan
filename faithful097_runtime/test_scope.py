"""Every real paired study cell after complete reference-audit materialization."""
import copy,sys,unittest
from unittest.mock import patch
import binding as b,preset_scope as s,prepare
class Scope(unittest.TestCase):
 def setUp(self):
  s.validate_protocol();self.reader=s.read;self.plan=patch.object(s,'read',side_effect=lambda p:dict(s.ENABLED) if p==s.PLAN else self.reader(p));self.plan.start()
 def tearDown(self):self.plan.stop()
 def test_all_twenty_real_frozen_fresh_authorizations(self):
  keys=set()
  for cid in s.IDS:
   for seed in (0,1):
    a,_=prepare.build_authorization(candidate=cid,seed=seed,phase='search',check_only=True);s.validate_authorization(a,require_runtime_proof=False)
    self.assertEqual(a['cell'],s.cell(cid,seed,'search'));self.assertFalse(a['reuse_completed_model']);self.assertEqual(a['initial_completed_updates'],0);self.assertIsNone(a['source_model_path']);keys.add(a['allocation_id'])
    for path,pin in a['files_SHA256'].items():self.assertEqual(b.sha(path),pin,path)
  self.assertEqual(len(keys),4);self.assertNotIn('torch',sys.modules)
 def test_wrong_seeds_phase_ids_and_boolean_seed_refused(self):
  for cid,seed,phase in [('f000',2,'search'),('f000',True,'search'),('f002',0,'search'),('f000',0,'confirm')]:
   with self.assertRaises(AssertionError):s.cell(cid,seed,phase)
 def test_hyperparameters_cannot_change_after_materialization(self):
  for cid in s.IDS:
   q=s.read(s.proposal_path(cid));s.validate_proposal(q)
   for key in q['hyperparameters']:
    changed=copy.deepcopy(q);changed['hyperparameters'][key]*=2
    with self.assertRaises(AssertionError):s.validate_proposal(changed)
 def test_old_weights_partial_horizon_and_wrong_formulation_refuse(self):
  a,_=prepare.build_authorization(check_only=True)
  for key,value in [('initial_completed_updates',500),('new_model_updates',19500),('fixed_endpoint',500),('reuse_completed_model',True),('is_continuation',True),('source_model_path','old.pt'),('source_model_SHA256','old'),('particlegan_version','0.9.6'),('metric_role','final_evaluation')]:
   changed=copy.deepcopy(a);changed[key]=value
   with self.assertRaises(AssertionError):s.validate_authorization(changed,require_runtime_proof=False)
 def test_false_native_parity_or_changed_fidelity_refuse(self):
  for path,key,value in [(s.MODEL_PROOF,'actual_cases',0),(s.PROTOCOL,'prior_regularizer_domain','sampled_rows')]:
   changed=copy.deepcopy(self.reader(path));changed[key]=value
   with patch.object(s,'read',side_effect=lambda p:changed if p==path else self.reader(p)):
    with self.assertRaises(AssertionError):s.validate_protocol()
 def test_controls_and_original_shared_resource_margins(self):
  self.assertEqual(set(s.read(s.BASELINE_PROOF)['metrics_by_seed']),{'0','1'})
  import resource_profile as p
  q=p.validate_proposal();self.assertEqual(q['roles']['alz_core_pg'],dict(gpu_bytes=2*2**30,ram_bytes=6*2**30));self.assertEqual(q['driver_headroom_bytes'],3*2**30);self.assertEqual(q['host_free_floor_bytes'],12*2**30)
if __name__=='__main__':unittest.main(verbosity=2)
