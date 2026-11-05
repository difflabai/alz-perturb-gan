"""Fresh matchedCFGen endpoints, frozen-source and no-qualification-reuse gates."""
import hashlib,json,sys,unittest
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import binding as b,preset_scope as scope,prepare,resource_profile as profile
class Scope(unittest.TestCase):
 def auth(self):return prepare.build_authorization(check_only=True)[0]
 def active(self,a):
  rd=scope.read;sh=scope.sha;pin=hashlib.sha256((json.dumps(scope.ENABLED,indent=2)+'\n').encode()).hexdigest();a['files_SHA256'][str(scope.PLAN)]=pin
  return patch.object(scope,'read',side_effect=lambda p:dict(scope.ENABLED) if Path(p)==scope.PLAN else rd(p)),patch.object(scope,'sha',side_effect=lambda p:pin if Path(p)==scope.PLAN else sh(p))
 def test_fixed_fresh_matched_native_scope(self):
  a=self.auth();rd,sh=self.active(a)
  with rd,sh:scope.validate_authorization(a,require_runtime_proof=False)
  self.assertNotIn('checkpoint_path',a);self.assertNotIn('torch',sys.modules)
 def test_warmstart_endpoints_seeds_method_cap_or_quality_changes_refused(self):
  for change in ({'checkpoint_path':'/tmp/old57donor'},{'new_model_updates':0},{'AE_epochs':4},{'FM_epochs':8},{'fixed_endpoint':0},{'cap_seconds':900},{'qualification_only':True},{'quality_scoring':False},{'operation':'qualify_cfgen_gpu_reload'},{'CFGen_source_commit':'main'},{'cell':{**scope.CELL,'seed':1}}):
   a=self.auth();a.update(change);rd,sh=self.active(a)
   with rd,sh,self.subTest(change=change),self.assertRaises(AssertionError):scope.validate_authorization(a,require_runtime_proof=False)
 def test_disabled_plan_and_changed_native_source_refused(self):
  a=self.auth()
  with self.assertRaisesRegex(AssertionError,'disabled'):scope.validate_authorization(a,require_runtime_proof=False)
  source=next(p for p in scope.read(scope.PROTOCOL)['source_SHA256'] if p.endswith('/fm.py'));a['files_SHA256'][source]='0'*64;rd,sh=self.active(a)
  with rd,sh,self.assertRaises(AssertionError):scope.validate_authorization(a,require_runtime_proof=False)
 def test_deadline_cannot_expand(self):
  for change in ({'issued_unix':float('nan')},{'expires_unix':scope.read(scope.WINDOW)['expires_unix']+1}):
   a=self.auth();a.update(change);rd,sh=self.active(a)
   with rd,sh,self.assertRaises(AssertionError):scope.validate_authorization(a,require_runtime_proof=False)
 def test_GPU_qualification_and_model_checks_mandatory(self):
  original=scope.read
  for path,key in [(scope.GPU_PROOF,'native_two_fresh_GPU_processes_count_replay_passed'),(scope.MODEL_PROOF,'CUDA_initialized')]:
   a=self.auth();rd,sh=self.active(a);q=original(path);q[key]=not q[key]
   with rd,sh,patch.object(scope,'read',side_effect=lambda p:q if Path(p)==path else (dict(scope.ENABLED) if Path(p)==scope.PLAN else original(p))),self.assertRaises(AssertionError):scope.validate_authorization(a,require_runtime_proof=False)
 def test_old_qualification_and_new_deployment_disabled(self):
  self.assertFalse(b.read(scope.ROOT/'cfgen_gpu_qualification_runtime/plan.json')['enabled'])
  self.assertFalse(profile.DEPLOYED.exists());self.assertFalse((profile.REGISTRY/'registry.json').exists())
if __name__=='__main__':unittest.main(verbosity=2)
