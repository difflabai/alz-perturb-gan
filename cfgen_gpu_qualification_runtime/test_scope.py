"""Qualification-only scope, source and endpoint refusals; no model imports."""
import ast,hashlib,json,sys,unittest
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import binding as b,preset_scope as scope,prepare,resource_profile as profile
class Scope(unittest.TestCase):
 def auth(self):return prepare.build_authorization(check_only=True)[0]
 def active(self,a):
  rd=scope.read;sh=scope.sha;pin=hashlib.sha256((json.dumps(scope.ENABLED,indent=2)+'\n').encode()).hexdigest();a['files_SHA256'][str(scope.PLAN)]=pin
  return patch.object(scope,'read',side_effect=lambda p:dict(scope.ENABLED) if Path(p)==scope.PLAN else rd(p)),patch.object(scope,'sha',side_effect=lambda p:pin if Path(p)==scope.PLAN else sh(p))
 def test_fixed_preserved_checkpoint_qualification_scope(self):
  a=self.auth();rd,sh=self.active(a)
  with rd,sh:scope.validate_authorization(a,require_runtime_proof=False)
  self.assertEqual(a['role'],'alz_core_cfgen');self.assertNotIn('torch',sys.modules)
 def test_training_quality_endpoint_and_checkpoint_substitution_refused(self):
  for change in ({'operation':'train_cfgen'},{'new_model_updates':1},{'quality_scoring':True},{'qualification_only':False},{'fixed_endpoint':400},{'cap_seconds':21600},{'checkpoint_path':'/tmp/other'},{'checkpoint_SHA256':'0'*64},{'cell':{**scope.CELL,'method':'scvi'}},{'CFGen_source_commit':'main'}):
   a=self.auth();a.update(change);rd,sh=self.active(a)
   with rd,sh,self.subTest(change=change),self.assertRaises(AssertionError):scope.validate_authorization(a,require_runtime_proof=False)
 def test_disabled_plan_and_changed_native_source_refused(self):
  a=self.auth()
  with self.assertRaisesRegex(AssertionError,'disabled'):scope.validate_authorization(a,require_runtime_proof=False)
  source=next(p for p in scope.read(scope.MODEL_PROOF)['source_SHA256'] if p.endswith('/fm.py'));a['files_SHA256'][source]='0'*64;rd,sh=self.active(a)
  with rd,sh,self.assertRaises(AssertionError):scope.validate_authorization(a,require_runtime_proof=False)
 def test_native_cfgen_runtime_is_pinned_and_model_free(self):
  a=self.auth();self.assertEqual(b.runtime_path(b.NATIVE_LAYOUT,scope.CELL),b.SCIENCE.parent/'isolated/cfgen-cu124-py311/bin/python')
  self.assertEqual(b.native_site_roots(scope.CELL,a['files_SHA256']),[b.SCIENCE.parent/'isolated/cfgen-cu124-py311/lib/python3.11/site-packages'])
  self.assertNotIn('torch',sys.modules)
 def test_expiry_cannot_expand_or_nan(self):
  for change in ({'issued_unix':float('nan')},{'expires_unix':scope.read(scope.WINDOW)['expires_unix']+1}):
   a=self.auth();a.update(change);rd,sh=self.active(a)
   with rd,sh,self.assertRaises(AssertionError):scope.validate_authorization(a,require_runtime_proof=False)
 def test_no_training_calls_or_unowned_existing_output_bypass(self):
  tree=ast.parse((HERE/'worker_gpu0.py').read_text())
  self.assertFalse(any(isinstance(n,ast.Call) and ((isinstance(n.func,ast.Name) and n.func.id in ('fit','train_native','trainer')) or (isinstance(n.func,ast.Attribute) and n.func.attr in ('fit','backward','step'))) for n in ast.walk(tree)))
  authpath=scope.ROOT/'artifacts/allocations/CAL_GPU0_V1_matched_scvi_eval1_f0_scvi_s0/authorization.json'
  with self.assertRaises(b.Refusal):b.require_live(authpath,allow_existing_output=True)
  self.assertFalse(profile.DEPLOYED.exists());self.assertFalse((profile.REGISTRY/'registry.json').exists())
if __name__=='__main__':unittest.main(verbosity=2)
