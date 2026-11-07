"""Every actual frozen fresh confirmation authorization, source and seed."""
import copy,sys,unittest
from unittest.mock import patch
import binding as b,preset_scope as s,prepare
class Scope(unittest.TestCase):
 def setUp(self):
  s.validate_protocol();self.reader=s.read;self.plan=patch.object(s,'read',side_effect=lambda p:dict(s.ENABLED) if p==s.PLAN else self.reader(p));self.plan.start()
 def tearDown(self):self.plan.stop()
 def test_all_actual_frozen_confirm_authorizations(self):
  keys=set()
  for cid in s.ENABLED['candidate_ids']:
   for seed in (2,3,4):
    a,_=prepare.build_authorization(candidate=cid,seed=seed,phase='confirm',check_only=True);s.validate_authorization(a,require_runtime_proof=False)
    self.assertEqual(a['cell'],s.cell(cid,seed,'confirm'));self.assertFalse(a['reuse_completed_model']);self.assertEqual(a['initial_completed_updates'],0);self.assertIsNone(a['source_model_path']);keys.add(a['allocation_id'])
    for path,pin in a['files_SHA256'].items():self.assertEqual(b.sha(path),pin,path)
  self.assertEqual(len(keys),3*len(s.ENABLED['candidate_ids']));self.assertNotIn('torch',sys.modules)
 def test_wrong_seeds_phase_ids_and_boolean_seed_refused(self):
  cid=s.ENABLED['candidate_ids'][0]
  for cid,seed,phase in [(cid,0,'confirm'),(cid,1,'confirm'),(cid,True,'confirm'),('undeclared',2,'confirm'),(cid,2,'search')]:
   with self.assertRaises(AssertionError):s.cell(cid,seed,phase)
 def test_hyperparameters_cannot_change_after_frozen_selection(self):
  for cid in s.ENABLED['candidate_ids']:
   q=s.read(s.proposal_path(cid));s.validate_proposal(q)
   for key in q['hyperparameters']:
    changed=copy.deepcopy(q);changed['hyperparameters'][key]*=2
    with self.assertRaises(AssertionError):s.validate_proposal(changed)
 def test_old_weights_partial_horizon_and_wrong_formulation_refuse(self):
  a,_=prepare.build_authorization(candidate=s.ENABLED['candidate_ids'][0],seed=2,phase='confirm',check_only=True)
  for key,value in [('initial_completed_updates',500),('new_model_updates',19500),('fixed_endpoint',500),('reuse_completed_model',True),('is_continuation',True),('source_model_path','old.pt'),('source_model_SHA256','old'),('particlegan_version','0.9.6'),('metric_role','final_evaluation')]:
   changed=copy.deepcopy(a);changed[key]=value
   with self.assertRaises(AssertionError):s.validate_authorization(changed,require_runtime_proof=False)
 def test_incomplete_selection_audit_refuses_confirmation(self):
  p=self.reader(s.PROTOCOL);path=p['selection_audit_path'];audit=copy.deepcopy(self.reader(path));audit['all_planned_attempts_resolved']=False
  with patch.object(s,'read',side_effect=lambda p:audit if str(p)==path else self.reader(p)):
   with self.assertRaises(AssertionError):s.validate_protocol()
 def test_controls_and_original_shared_resource_margins(self):
  self.assertEqual(set(s.read(s.BASELINE_PROOF)['metrics_by_seed']),{'1','2','3','4'})
  import resource_profile as p
  q=p.validate_proposal();self.assertEqual(q['roles']['alz_core_pg'],dict(gpu_bytes=2*2**30,ram_bytes=6*2**30));self.assertEqual(q['driver_headroom_bytes'],3*2**30);self.assertEqual(q['host_free_floor_bytes'],12*2**30)
if __name__=='__main__':unittest.main(verbosity=2)
