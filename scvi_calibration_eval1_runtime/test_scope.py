"""Evaluation-only lineage, original deadline, immutable400epoch fit gates."""
import ast,hashlib,json,sys,unittest
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import binding as b,prepare,preset_scope as scope,resource_profile as profile
class Scope(unittest.TestCase):
 def auth(self):return prepare.build_authorization(check_only=True)[0]
 def active(self,a):
  rd=scope.read;sh=scope.sha;pin=hashlib.sha256((json.dumps(scope.ENABLED,indent=2)+'\n').encode()).hexdigest();a['files_SHA256'][str(scope.PLAN)]=pin
  return patch.object(scope,'read',side_effect=lambda p:dict(scope.ENABLED) if Path(p)==scope.PLAN else rd(p)),patch.object(scope,'sha',side_effect=lambda p:pin if Path(p)==scope.PLAN else sh(p))
 def test_valid_exact_parent_evaluation_scope(self):
  a=self.auth();rd,sh=self.active(a)
  with rd,sh:scope.validate_authorization(a,require_runtime_proof=False)
  self.assertEqual(a['new_model_updates'],0);self.assertNotIn('torch',sys.modules)
 def test_refit_endpoint_seed_model_or_lineage_changes_refuse(self):
  for change in ({'operation':'train_matched_scvi_calibration'},{'new_model_updates':1},{'fixed_endpoint':399},{'cap_seconds':21600},{'parent_fit_path':'/tmp/other'},{'parent_fit_receipt_SHA256':'0'*64},{'parent_authorization_SHA256':'0'*64},{'evaluation_used':True},{'cell':{**scope.CELL,'seed':1}}):
   a=self.auth();a.update(change);rd,sh=self.active(a)
   with rd,sh,self.subTest(change=change),self.assertRaises(AssertionError):scope.validate_authorization(a,require_runtime_proof=False)
 def test_parent_absolute_expiry_cannot_reset(self):
  a=self.auth();a['expires_unix']=scope.read(scope.PARENT_AUTH)['expires_unix']+1;rd,sh=self.active(a)
  with rd,sh,self.assertRaises(AssertionError):scope.validate_authorization(a,require_runtime_proof=False)
 def test_parent_model_bytes_required(self):
  a=self.auth();a['files_SHA256'][str(scope.PARENT_FIT/'model/model.pt')]='0'*64;rd,sh=self.active(a)
  with rd,sh,self.assertRaises(AssertionError):scope.validate_authorization(a,require_runtime_proof=False)
 def test_evaluation_worker_contains_no_training_call(self):
  tree=ast.parse((HERE/'worker_gpu0.py').read_text())
  self.assertFalse(any(isinstance(n,ast.Call) and ((isinstance(n.func,ast.Name) and n.func.id in ('fit','train_native')) or (isinstance(n.func,ast.Attribute) and n.func.attr in ('fit','train_native','backward','step'))) for n in ast.walk(tree)))
 def test_original_fit_actual_disabled_and_no_new_deployment(self):
  self.assertFalse(b.read(scope.ROOT/'scvi_calibration_runtime/plan.json')['enabled'])
  self.assertFalse(profile.DEPLOYED.exists());self.assertFalse((profile.REGISTRY/'registry.json').exists())
if __name__=='__main__':unittest.main(verbosity=2)
