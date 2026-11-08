"""All90 actual scientific cell/device source closures, with deployment held."""
import ast,copy,sys,unittest
from pathlib import Path
from unittest.mock import patch
import binding as b,preset_scope as s,prepare,resource_profile as p
from dual_coordination import GPUS,validate_resource_protocol
ORIGINAL=s.validate_authorization
def prepared(a):return ORIGINAL(a,require_activation=False,require_handoff=False,require_runtime_proof=False)
class Scope(unittest.TestCase):
 def test_all90_actual_cell_device_authorizations_and_preimport_closures(self):
  seen=set()
  for seed in range(5):
   for fold in range(3):
    for method in ('pg','scvi','cfgen'):
     for gpu in GPUS:
      auth,_=prepare.build_authorization(fold=fold,method=method,seed=seed,gpu=gpu,check_only=True);prepared(auth)
      with patch.object(s,'validate_authorization',side_effect=prepared):b.source_closure(auth,b.NATIVE_LAYOUT)
      self.assertIsNone(auth['source_model_path']);self.assertEqual(auth['initial_completed_updates'],0)
      seen.add((fold,method,seed,gpu))
  self.assertEqual(len(seen),90);self.assertNotIn('torch',sys.modules)
 def test_disabled_or_missing_handoff_never_native_admission(self):
  auth,_=prepare.build_authorization(gpu=GPUS[0],check_only=True)
  with self.assertRaises(AssertionError):ORIGINAL(auth,require_handoff=False,require_runtime_proof=False)
  with self.assertRaises(FileNotFoundError):ORIGINAL(auth,require_activation=False,require_runtime_proof=False)
 def test_changed_recipe_seed_source_endpoint_or_card_refused(self):
  auth,_=prepare.build_authorization(gpu=GPUS[1],check_only=True)
  for key,value in [('gpu_uuid','OTHER'),('source_model_path','old.pt'),('is_continuation',True),('initial_completed_updates',500),('reuse_completed_model',True),('fixed_endpoint',500),('operation','scalarcal097_cell')]:
   changed=copy.deepcopy(auth);changed[key]=value
   with self.subTest(key=key),self.assertRaises(AssertionError):prepared(changed)
  changed=copy.deepcopy(auth);changed['released_hyperparameters']['particle_lr']=.03
  with self.assertRaises(AssertionError):prepared(changed)
 def test_scientific_worker_and_native_samplers_are_byte_identical(self):
  old=s.ROOT/'benchmark097_dual_runtime'
  for name in ('worker_gpu0.py','retained_pcs.py','native_scvi_reload.py','finish.py','state_fingerprint.py','project_runner_gpu0.py'):
   self.assertEqual((old/name).read_bytes(),(b.HERE/name).read_bytes(),name)
 def test_one_use_keys_and_core_live_bindings_are_unchanged(self):
  old=ast.parse((s.ROOT/'benchmark097_dual_runtime/binding.py').read_text());new=ast.parse((b.HERE/'binding.py').read_text())
  def node(tree,name):return ast.dump(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)
  for name in ('core_envelope','unseen','require_live','verify','modules','allocation_key','attempt_key'):self.assertEqual(node(old,name),node(new,name),name)
 def test_only_registry_and_resource_description_change_physical_profiles(self):
  policies=p.validate_proposal();validate_resource_protocol()
  for gpu,policy in policies.items():
   self.assertEqual(policy['gpu'],gpu);self.assertEqual(policy['roles'],{r:dict(gpu_bytes=3*2**30,ram_bytes=6*2**30) for r in ('alz_core_pg','alz_core_scvi','alz_core_cfgen')})
if __name__=='__main__':unittest.main(verbosity=2)
