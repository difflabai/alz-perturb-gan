"""Comparator identity, all-donor fit, held-out exclusion and live-source gates."""
import copy,hashlib,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import binding as b,prepare,preset_scope as scope,resource_profile as profile
from worker_gpu0 import validate_fit
class Scope(unittest.TestCase):
 def auth(self):return prepare.build_authorization(check_only=True)[0]
 def active(self,auth):
  rd=scope.read;sh=scope.sha;pin=hashlib.sha256((json.dumps(scope.ENABLED,indent=2)+'\n').encode()).hexdigest();auth['files_SHA256'][str(scope.PLAN)]=pin
  return patch.object(scope,'read',side_effect=lambda p:dict(scope.ENABLED) if Path(p)==scope.PLAN else rd(p)),patch.object(scope,'sha',side_effect=lambda p:pin if Path(p)==scope.PLAN else sh(p))
 def test_fixed_native_comparator_configuration(self):
  auth=self.auth();rd,sh=self.active(auth)
  with rd,sh:self.assertEqual(scope.validate_authorization(auth,require_runtime_proof=False)['method'],'scvi')
  self.assertNotIn('torch',sys.modules)
 def test_training_endpoint_information_and_version_changes_refused(self):
  for change in ({'fixed_endpoint':399},{'scvi_version':'1.5.1'},{'evaluation_used':True},{'query_information':['stage','donor']},{'cell':{**scope.CELL,'seed':1}},{'operation':'train_released_hyperparameters'},{'role':'alz_core_pg'},{'cap_seconds':900},{'checkpoint_path':'/tmp/unread'},{'released_hyperparameters':{}},{'scvi_architecture':{}},{'scvi_training_plan':{}},{'independent_current_device_review_claimed':True}):
   auth=self.auth();auth.update(change);rd,sh=self.active(auth)
   with rd,sh,self.subTest(change=change),self.assertRaises(AssertionError):scope.validate_authorization(auth,require_runtime_proof=False)
 def test_disabled_plan_and_unqualified_native_source_refused(self):
  auth=self.auth()
  with self.assertRaisesRegex(AssertionError,'disabled'):scope.validate_authorization(auth,require_runtime_proof=False)
  source=scope.read(scope.PROTOCOL)['source_SHA256'];path=next(p for p in source if p.endswith('/scvi_adapter.py'));auth['files_SHA256'][path]='0'*64;rd,sh=self.active(auth)
  with rd,sh,self.assertRaises(AssertionError):scope.validate_authorization(auth,require_runtime_proof=False)
 def test_active_predecessor_blocks_deployment(self):
  self.assertEqual(self.auth()['operation'],'train_matched_scvi_calibration')
  rd=profile.read
  with patch.object(profile,'read',side_effect=lambda p:{'jobs':{'fixture':'active'}} if Path(p)==profile.PREVIOUS_REGISTRY else rd(p)):
   with self.assertRaisesRegex(AssertionError,'must release'):profile.configure(self.auth(),check_only=False)
  self.assertFalse(profile.DEPLOYED.exists())
 def test_original_expiry_cannot_expand(self):
  for change in ({'issued_unix':float('inf')},{'issued_unix':float('nan')},{'expires_unix':scope.read(scope.WINDOW)['expires_unix']+1}):
   auth=self.auth();auth.update(change);rd,sh=self.active(auth)
   with rd,sh,self.assertRaises(AssertionError):scope.validate_authorization(auth,require_runtime_proof=False)
 def test_scvi_runtime_and_import_roots_are_distinct(self):
  auth=self.auth();self.assertEqual(b.runtime_path(b.NATIVE_LAYOUT,scope.CELL),b.SCIENCE.parent/'isolated/cfgen-cu124-py311/bin/python')
  self.assertEqual(b.native_site_roots(scope.CELL,auth['files_SHA256']),[b.SCIENCE.parent/'isolated/cfgen-cu124-py311/lib/python3.11/site-packages'])
  self.assertNotIn('torch',sys.modules)
 def test_partial_or_selected_fit_cannot_satisfy_endpoint(self):
  with tempfile.TemporaryDirectory() as d:
   cache=Path(d);(cache/'manifest.json').write_text('{}');p=scope.read(scope.PROTOCOL)
   auth={'fixed_endpoint':400,'cache_dir':str(cache),'scvi_architecture':p['architecture'],'scvi_training_plan':p['training_plan']}
   q={'status':'completed','method':'scvi','completed_epochs':400,'fixed_endpoint':400,'fold':0,'seed':0,'scvi_tools':'1.3.3','training_cells':14762,'training_donors':[str(i) for i in range(33)],'validation_cells':0,'early_stopping':False,'train_indices_exactly_all_optimization_cells':True,'architecture':p['architecture'],'training_plan':p['training_plan'],'query_keys':['stage'],'cache_manifest_SHA256':b.sha(cache/'manifest.json')}
   validate_fit(q,auth)
   for change in ({'completed_epochs':399},{'validation_cells':1},{'early_stopping':True},{'seed':1},{'training_cells':24676},{'query_keys':['stage','donor']},{'train_indices_exactly_all_optimization_cells':False}):
    with self.subTest(change=change),self.assertRaises(b.Refusal):validate_fit({**q,**change},auth)
 def test_current_PG_execution_pins_unchanged(self):
  auth=b.read(scope.ROOT/'artifacts/allocations/CAL_GPU0_V1_released_v095_regkappa050_f0_pg_s0/authorization.json')
  self.assertTrue(all(b.sha(p)==pin for p,pin in auth['research_source_SHA256'].items()))
  self.assertFalse(profile.DEPLOYED.exists());self.assertFalse((profile.REGISTRY/'registry.json').exists())
if __name__=='__main__':unittest.main(verbosity=2)
