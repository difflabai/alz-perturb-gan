"""Actual frozen selection and every real cell authorization, CPU only."""
import copy,sys,unittest
from unittest.mock import patch
import binding as b,preset_scope as s,prepare
class FrozenScope(unittest.TestCase):
 def setUp(self):
  self.protocol=s.protocol();self.reader=s.read
  self.plan=patch.object(s,'read',side_effect=lambda p:dict(s.ENABLED) if p==s.PLAN else self.reader(p));self.plan.start()
 def tearDown(self):self.plan.stop()
 def auth(self,fold=0,method='pg',seed=0):return prepare.build_authorization(fold=fold,method=method,seed=seed,check_only=True)[0]
 def test_all45_actual_frozen_fresh_authorizations_and_all_pins(self):
  ids=set()
  for fold in range(3):
   for method in ('pg','scvi','cfgen'):
    for seed in range(5):
     a=self.auth(fold,method,seed);s.validate_authorization(a,require_runtime_proof=False)
     self.assertEqual(b.cell(a),s.cell(fold,method,seed));ids.add(a['allocation_id'])
     self.assertEqual(a['initial_completed_updates'],0);self.assertFalse(a['is_continuation']);self.assertFalse(a['reuse_completed_model'])
     self.assertIsNone(a['source_model_path']);self.assertIsNone(a['source_model_SHA256'])
     self.assertEqual(a['released_hyperparameters'],self.protocol['PG_hyperparameters'])
     for path,pin in a['files_SHA256'].items():self.assertEqual(b.sha(path),pin,path)
  self.assertEqual(len(ids),45);self.assertNotIn('torch',sys.modules);self.assertNotIn('scvi',sys.modules)
 def test_warm_start_partial_or_wrong_formulation_and_clock_refused(self):
  a=self.auth()
  for key,value in [('initial_completed_updates',500),('reuse_completed_model',True),('is_continuation',True),('source_model_path','old.pt'),('source_model_SHA256','old'),('new_model_updates',19500),('fixed_endpoint',500),('particlegan_version','0.9.6'),('role','alz_core_cfgen')]:
   changed=copy.deepcopy(a);changed[key]=value
   with self.assertRaises(AssertionError):s.validate_authorization(changed,require_runtime_proof=False)
  changed=copy.deepcopy(a);changed['released_hyperparameters']['generator_lr']*=2
  with self.assertRaises(AssertionError):s.validate_authorization(changed,require_runtime_proof=False)
 def test_actual_frozen_hyperparameters_cannot_depart_from_confirmation_winner(self):
  changed=copy.deepcopy(self.protocol);changed['PG_hyperparameters']['reg_kappa']*=2
  with patch.object(s,'read',side_effect=lambda p:changed if p==s.PROTOCOL else self.reader(p)):
   with self.assertRaises(AssertionError):s.protocol()
 def test_complete_fresh_confirmation_is_required(self):
  changed=copy.deepcopy(self.protocol);changed['PG_fresh_seed_confirmation_complete']=False
  with patch.object(s,'read',side_effect=lambda p:changed if p==s.PROTOCOL else self.reader(p)):
   with self.assertRaises(AssertionError):s.protocol()
 def test_exact_units_and_undeclared_cells(self):
  self.assertEqual([s.update_count(s.cell(f,'cfgen',0)) for f in range(3)],[102600,106200,102600])
  self.assertEqual(s.update_unit('cfgen'),'optimizer_steps');self.assertEqual(s.update_unit('pg'),'optimizer_pairs');self.assertEqual(s.update_unit('scvi'),'epochs')
  for value in [(3,'pg',0),(0,'ddgan',0),(0,'pg',5),(False,'pg',0),(0,'cfgen',True)]:
   with self.assertRaises(AssertionError):s.cell(*value)
 def test_real_plan_stays_disabled_during_CPU_qualification(self):self.assertFalse(self.reader(s.PLAN)['enabled'])
if __name__=='__main__':unittest.main(verbosity=2)
