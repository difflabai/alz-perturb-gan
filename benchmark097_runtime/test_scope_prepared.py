"""Prepared-frame tests use an explicit synthetic frozen-selection fixture.

The on-disk real protocol remains unfrozen and native admissions must refuse it.
No actual candidate or GPU deployment qualification is claimed by this suite.
"""
import copy,sys,unittest
from unittest.mock import patch
import binding as b,preset_scope as s,prepare

H={'generator_lr':.006,'critic_lr':.009,'particle_lr':.015,'reg_coeff':1.,'reg_kappa':.5,'prior_reg':.01}
class PreparedScope(unittest.TestCase):
 def setUp(self):
  self.reader=s.read;self.fixture={**self.reader(s.PROTOCOL),'PG_hyperparameters':H}
  self.plan=patch.object(s,'read',side_effect=lambda p:dict(s.ENABLED) if p==s.PLAN else self.reader(p));self.plan.start()
 def tearDown(self):self.plan.stop()
 def auth(self,fold=0,method='pg',seed=0):
  return prepare.build_authorization(fold=fold,method=method,seed=seed,check_only=True)[0]
 def test_real_unfrozen_candidate_refuses_without_models_or_allocation(self):
  with self.assertRaisesRegex(AssertionError,'PG selection/confirmation'):self.auth()
  self.assertNotIn('torch',sys.modules);self.assertNotIn('scvi',sys.modules)
 def test_all45_fresh_bindings_under_synthetic_selection_fixture(self):
  keys=set()
  with patch.object(s,'protocol',return_value=self.fixture):
   for fold in range(3):
    for method in ('pg','scvi','cfgen'):
     for seed in range(5):
      a=self.auth(fold,method,seed);s.validate_authorization(a,require_runtime_proof=False)
      self.assertEqual(b.cell(a),s.cell(fold,method,seed));keys.add(a['allocation_id'])
      self.assertFalse(a['reuse_completed_model']);self.assertEqual(a['initial_completed_updates'],0)
      self.assertIsNone(a['source_model_path']);self.assertIsNone(a['source_model_SHA256'])
  self.assertEqual(len(keys),45);self.assertNotIn('torch',sys.modules)
 def test_partial_or_parent_weights_and_wrong_role_cannot_pass_frozen_scope(self):
  with patch.object(s,'protocol',return_value=self.fixture):
   a=self.auth()
   for key,value in [('reuse_completed_model',True),('is_continuation',True),('source_model_path','old.pt'),('source_model_SHA256','old'),('initial_completed_updates',500),('new_model_updates',19500),('fixed_endpoint',500),('particlegan_version','0.9.6'),('role','alz_core_cfgen')]:
    changed=copy.deepcopy(a);changed[key]=value
    with self.assertRaises(AssertionError):s.validate_authorization(changed,require_runtime_proof=False)
 def test_cfgen_training_clocks_scale_with_actual_frozen_fold_cells(self):
  self.assertEqual([s.update_count(s.cell(f,'cfgen',0)) for f in range(3)],[102600,106200,102600])
  self.assertEqual(s.endpoint('cfgen'),1800);self.assertEqual(s.update_unit('cfgen'),'optimizer_steps')
  self.assertEqual(s.update_unit('scvi'),'epochs');self.assertEqual(s.update_unit('pg'),'optimizer_pairs')
 def test_new_native_gate_refuses_undeclared_methods_or_experiments(self):
  for args in [(3,'pg',0),(0,'ddgan',0),(0,'pg',5),(False,'pg',0),(0,'cfgen',True)]:
   with self.assertRaises(AssertionError):s.cell(*args)
  for method in ('pg','scvi','cfgen'):
   a=dict(cpu_fixture=False,cell=s.cell(2,method,4),operation='external097_benchmark_cell');self.assertEqual(b.cell(a),a['cell'])
   changed=copy.deepcopy(a);changed['cell']['experiment']='retuned_after_test'
   with self.assertRaises(b.Refusal):b.cell(changed)
 def test_declared_three_method_resource_profile_keeps_physical_locks_and_floors(self):
  import resource_profile as p
  q=p.validate_proposal();self.assertEqual(set(q['roles']),{'alz_core_pg','alz_core_scvi','alz_core_cfgen'})
  self.assertTrue(all(v=={'gpu_bytes':3*2**30,'ram_bytes':6*2**30} for v in q['roles'].values()))
  self.assertEqual(q['driver_headroom_bytes'],3*2**30);self.assertEqual(q['host_free_floor_bytes'],12*2**30)
  self.assertEqual(q['gpu'],b.GPU1);self.assertTrue(all('gpu-1.lock' in x for x in q['locks']))
 def test_actual_prepared_plan_is_disabled(self):
  self.assertFalse(self.reader(s.PLAN)['enabled'])

if __name__=='__main__':unittest.main(verbosity=2)
