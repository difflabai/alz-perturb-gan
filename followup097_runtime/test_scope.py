"""Actual selected settings; reference-only values cannot be retrained here."""
import copy,sys,unittest
from unittest.mock import patch
import binding as b,preset_scope as s,prepare
class Scope(unittest.TestCase):
 def setUp(self):
  s.validate_protocol();self.reader=s.read;self.plan=patch.object(s,'read',side_effect=lambda p:dict(s.ENABLED) if p==s.PLAN else self.reader(p));self.plan.start()
 def tearDown(self):self.plan.stop()
 def test_every_selected_configuration_at_all_three_new_seeds(self):
  keys=set()
  for cid in s.ENABLED['candidate_ids']:
   for seed in (2,3,4):
    a,_=prepare.build_authorization(candidate=cid,seed=seed,phase='followup',check_only=True);s.validate_authorization(a,require_runtime_proof=False)
    self.assertEqual(a['cell'],s.cell(cid,seed,'followup'));self.assertEqual(a['initial_completed_updates'],0);self.assertFalse(a['reuse_completed_model']);self.assertIsNone(a['source_model_path']);keys.add(a['allocation_id'])
    for path,pin in a['files_SHA256'].items():self.assertEqual(b.sha(path),pin,path)
  self.assertEqual(len(keys),3*len(s.ENABLED['candidate_ids']));self.assertNotIn('torch',sys.modules)
 def test_reference_selection_seeds_other_phases_and_undeclared_ids_refuse(self):
  cid=s.ENABLED['candidate_ids'][0]
  for candidate,seed,phase in [(cid,0,'followup'),(cid,1,'followup'),(cid,True,'followup'),(cid,5,'followup'),('d000',2,'followup'),('r999',2,'followup'),(cid,2,'search')]:
   with self.assertRaises(AssertionError):s.cell(candidate,seed,phase)
 def test_changed_proposal_scalars_refuse(self):
  for cid in s.ENABLED['candidate_ids']:
   original=s.read(s.proposal_path(cid));s.validate_proposal(original)
   for key in original['hyperparameters']:
    q=copy.deepcopy(original);q['hyperparameters'][key]*=2
    with self.assertRaises(AssertionError):s.validate_proposal(q)
 def test_warm_start_partial_clock_and_wrong_native_release_refuse(self):
  a,_=prepare.build_authorization(check_only=True)
  for key,value in [('initial_completed_updates',500),('new_model_updates',19500),('fixed_endpoint',500),('is_continuation',True),('reuse_completed_model',True),('source_model_path','old.pt'),('source_model_SHA256','old'),('particlegan_version','0.9.6'),('role','alz_core_scvi')]:
   q=copy.deepcopy(a);q[key]=value
   with self.assertRaises(AssertionError):s.validate_authorization(q,require_runtime_proof=False)
 def test_incomplete_selection_and_reference_masquerading_as_new_fit_refuse(self):
  p=self.reader(s.PROTOCOL);path=p['paired_selection_audit_path'];audit=copy.deepcopy(self.reader(path));audit['all_planned_attempts_resolved']=False
  with patch.object(s,'read',side_effect=lambda f:audit if str(f)==path else self.reader(f)):
   with self.assertRaises(AssertionError):s.validate_protocol()
  p=copy.deepcopy(p);p['reference_controls'][0]['new_model_fitted']=True
  with patch.object(s,'read',side_effect=lambda f:p if f==s.PROTOCOL else self.reader(f)):
   with self.assertRaises(AssertionError):s.validate_protocol()
 def test_same_shared_profile_and_verified_matched_controls(self):
  self.assertEqual(set(s.read(s.BASELINE_PROOF)['metrics_by_seed']),{'1','2','3','4'})
  import resource_profile as p
  q=p.validate_proposal();self.assertEqual(q['roles']['alz_core_pg'],dict(gpu_bytes=2*2**30,ram_bytes=6*2**30));self.assertEqual(q['driver_headroom_bytes'],3*2**30);self.assertEqual(q['host_free_floor_bytes'],12*2**30)
if __name__=='__main__':unittest.main(verbosity=2)
