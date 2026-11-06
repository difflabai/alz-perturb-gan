"""Fresh500 boundary trials cannot change any other scalar, family or clock."""
import copy,sys,unittest
from unittest.mock import patch
import binding as b,preset_scope as s,prepare
class Scope(unittest.TestCase):
 def setUp(self):
  self.reader=s.read;self.mock=patch.object(s,'read',side_effect=lambda p:dict(s.ENABLED) if p==s.PLAN else self.reader(p));self.mock.start()
 def tearDown(self):self.mock.stop()
 def test_fresh500_has_no_warmstart_or_model_import(self):
  a,_=prepare.build_authorization(check_only=True);s.validate_authorization(a,require_runtime_proof=False)
  self.assertEqual((a['initial_completed_updates'],a['new_model_updates']),(0,500));self.assertFalse(a['is_continuation']);self.assertIsNone(a['source_model_path']);self.assertNotIn('torch',sys.modules)
 def test_all30_native_keys_and_undeclared_ids_refused(self):
  for i in range(6):
   for phase,seeds in [('search',[0]),('confirm',[1,2,3,4])]:
    for seed in seeds:
     v=s.cell(f'k{i:03d}',seed,phase);self.assertEqual(b.cell(dict(cell=v,cpu_fixture=False,operation='hpo500_cell')),v)
  for cid,seed,phase in [('k006',0,'search'),('c014',0,'search'),('k000',1,'search'),('k000',0,'confirm')]:
   with self.assertRaises(AssertionError):s.cell(cid,seed,phase)
 def test_only_positive_predeclared_kappa_varies(self):
  q=s.read(s.proposal_path('k000'));s.validate_proposal(q)
  for k,v in [('generator_lr',q['hyperparameters']['generator_lr']*2),('reg_kappa',0),('reg_kappa',s.read(s.ANCHOR)['anchor_hyperparameters']['reg_kappa']),('prior_reg',float('nan'))]:
   d=copy.deepcopy(q);d['hyperparameters'][k]=v
   with self.assertRaises(AssertionError):s.validate_proposal(d)
 def test_wrong_family_horizon_seed_reuse_or_deadline_refuses(self):
  a,_=prepare.build_authorization(check_only=True)
  for k,v in [('operation','resume_hpo500_cell'),('fixed_endpoint',20000),('metric_role','final_evaluation'),('reuse_completed_model',True),('is_continuation',True),('initial_completed_updates',420),('new_model_updates',80),('particlegan_version','0.9.4'),('cap_seconds',21600),('source_model_path','/existing.pt')]:
   d=copy.deepcopy(a);d[k]=v
   with self.assertRaises(AssertionError):s.validate_authorization(d,require_runtime_proof=False)
 def test_confirmation_requires_frozen_new_winner(self):
  a,_=prepare.build_authorization(check_only=True);a['phase']='confirm';a['cell']=s.cell('k000',1,'confirm');reader=s.read
  def missing(p):
   if p==s.FINALISTS:raise FileNotFoundError(p)
   return reader(p)
  with patch.object(s,'read',side_effect=missing),self.assertRaises(FileNotFoundError):s.validate_authorization(a,require_runtime_proof=False)
 def test_GPU1_shared_guard_and_resource_margins(self):
  import resource_profile as p
  q=p.validate_proposal();a,_=prepare.build_authorization(check_only=True);_,ad,guard=b.modules(a)
  self.assertEqual(b.device(a),b.GPU1);self.assertEqual(ad.GPU0,b.GPU1);self.assertEqual(ad.LOCKS,('/mnt/nvme/sliders/qualification/admission-gpu-1.lock','/mnt/nvme/sliders/locks/gpu-1.lock'))
  self.assertEqual(q['roles']['alz_core_pg'],{'gpu_bytes':2*2**30,'ram_bytes':6*2**30});self.assertEqual(q['driver_headroom_bytes'],3*2**30);self.assertEqual(q['host_free_floor_bytes'],12*2**30);self.assertEqual(q['foreign_growth_allowances']['compute_GPU_bytes'],2**29)
if __name__=='__main__':unittest.main(verbosity=2)
