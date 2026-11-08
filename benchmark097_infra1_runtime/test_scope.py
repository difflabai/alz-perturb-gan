import copy,sys,unittest
from unittest.mock import patch
import binding as b,preset_scope as s,prepare,resource_profile as profile
from dual_coordination import GPUS,valid_cell,validate_active_peers
ORIGINAL=s.validate_authorization
def prepared(a):return ORIGINAL(a,require_activation=False,require_handoff=False,require_runtime_proof=False)
class Scope(unittest.TestCase):
 def test_all_four_real_cell_card_preimport_source_closures(self):
  seen=set()
  for method in ('pg','cfgen'):
   for gpu in GPUS:
    a,_=prepare.build_authorization(fold=2,method=method,seed=1,gpu=gpu,check_only=True);prepared(a)
    with patch.object(s,'validate_authorization',side_effect=prepared):b.source_closure(a,b.NATIVE_LAYOUT)
    seen.add((method,gpu));self.assertEqual(a['initial_completed_updates'],0);self.assertIsNone(a['source_model_path'])
  self.assertEqual(len(seen),4);self.assertNotIn('torch',sys.modules)
 def test_other_original_cells_and_experiments_refused(self):
  for f,m,seed in ((0,'pg',1),(2,'scvi',1),(2,'cfgen',2)):
   with self.assertRaises(AssertionError):s.cell(f,m,seed)
  with self.assertRaises(AssertionError):valid_cell(dict(experiment='external097',fold=2,method='pg',seed=1))
 def test_weights_resume_endpoint_recipe_mutants_refused(self):
  a,_=prepare.build_authorization(fold=2,method='pg',seed=1,gpu=GPUS[1],check_only=True)
  for k,v in [('source_model_path','failed.pt'),('reuse_completed_model',True),('initial_completed_updates',1),('fixed_endpoint',500),('operation','external097_balanced_cell'),('is_continuation',True)]:
   c=copy.deepcopy(a);c[k]=v
   with self.subTest(k=k),self.assertRaises(AssertionError):prepared(c)
  c=copy.deepcopy(a);c['released_hyperparameters']['particle_lr']=.03
  with self.assertRaises(AssertionError):prepared(c)
 def test_plan_handoff_and_runtime_proof_required(self):
  a,_=prepare.build_authorization(gpu=GPUS[1],check_only=True)
  with self.assertRaises(AssertionError):ORIGINAL(a,require_runtime_proof=False,require_handoff=False)
  with self.assertRaises(FileNotFoundError):ORIGINAL(a,require_runtime_proof=False,require_activation=False)
 def test_scientific_workers_native_auditor_and_backend_byte_identical(self):
  for n in ('worker_gpu0.py','finish.py','retained_pcs.py','native_scvi_reload.py','state_fingerprint.py','project_runner_gpu0.py','launch_core_gpu0.py'):
   self.assertEqual((b.HERE/n).read_bytes(),(s.ROOT/'benchmark097_balanced_runtime'/n).read_bytes(),n)
 def test_exact_physical_bounds_and_foreign_controls_unchanged(self):
  for g,p in profile.validate_proposal().items():
   q=s.read(s.ROOT/f'benchmark097_balanced_runtime/policy.gpu{GPUS.index(g)}.prepared.json')
   for key in ('roles','protected_peers','driver_headroom_bytes','host_free_floor_bytes','locks','source_sha256'):self.assertEqual(p[key],q[key])
if __name__=='__main__':unittest.main(verbosity=2)
