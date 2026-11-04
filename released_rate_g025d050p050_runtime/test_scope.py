"""Private CPU source/policy wires and refusal tests, no canonical mutation."""
import copy,hashlib,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import binding as b
import prepare
import preset_scope as scope
import resource_profile as profile

class Scope(unittest.TestCase):
    def auth(self,**kwargs):return prepare.build_authorization(check_only=True,**kwargs)[0]
    def active(self,auth):
        old_read=scope.read;old_sha=scope.sha
        plan_pin=hashlib.sha256((json.dumps(scope.ENABLED,indent=2)+'\n').encode()).hexdigest()
        auth['files_SHA256'][str(scope.PLAN)]=plan_pin
        return (patch.object(scope,'read',side_effect=lambda p:dict(scope.ENABLED) if Path(p)==scope.PLAN else old_read(p)),
                patch.object(scope,'sha',side_effect=lambda p:plan_pin if Path(p)==scope.PLAN else old_sha(p)))
    def test_model_proof_whole_preset_scope_and_owner_window(self):
        auth=self.auth();rd,sh=self.active(auth)
        with rd,sh:self.assertEqual(scope.validate_authorization(auth,require_runtime_proof=False)['status'],'PROSPECTIVE_RELEASED_PARTICLEGAN_HYPERPARAMETER_SEARCH_NOT_LAUNCHED')
        self.assertNotIn('torch',sys.modules)
    def test_wrong_recipe_cell_updates_hybrid_or_warm_start_refuses(self):
        for change in ({'operation':'train_bcap'},{'cell':{**scope.CELL,'seed':1}},{'fixed_endpoint':40000},{'cap_seconds':900},
                       {'hybrid_config':{'pg_bag_fraction':.75}},{'variant_config':{**scope.CFG,'z_dim':2}},
                       {'initial_completed_updates':1},{'new_model_updates':19999},{'checkpoint_path':'/tmp/unused'},
                       {'released_hyperparameters':{**scope.HYPER,'particle_lr':scope.HYPER['particle_lr']*2}},{'tuning_candidate_id':'invalid_candidate'},{'qualification_only':True},{'pure_formulation':'bcap'},{'independent_current_device_review_claimed':True},{'native_recipe':'mog'},{'prior_update':'learned'},{'stage_prior_config':{}}):
            auth=self.auth();auth.update(change);rd,sh=self.active(auth)
            with rd,sh,self.subTest(change=change),self.assertRaises(AssertionError):scope.validate_authorization(auth,require_runtime_proof=False)
    def test_disabled_plan_refuses_and_does_not_import_model(self):
        auth=self.auth();original_read=scope.read
        with patch.object(scope,'read',side_effect=lambda p:{**scope.ENABLED,'enabled':False} if Path(p)==scope.PLAN else original_read(p)):
            with self.assertRaisesRegex(AssertionError,'disabled or changed'):scope.validate_authorization(auth,require_runtime_proof=False)
        self.assertNotIn('torch',sys.modules)
    def test_partial_diagnostic_checkpoint_cannot_satisfy_complete_endpoint(self):
        import worker_gpu0 as worker
        with tempfile.TemporaryDirectory() as d:
            cache=Path(d);(cache/'manifest.json').write_text('{}')
            auth={'cell':dict(scope.CELL),'cpu_fixture':False,'cache_dir':str(cache),
                  'released_hyperparameters':dict(scope.HYPER),'variant_config':dict(scope.CFG)}
            q={'schema':1,'method':'pg','native_recipe':'bispf_optimized','fold':0,'seed':0,
               'fixed_endpoint':20000,'completed_updates':5000,'policy':{'completed_steps':5000},
               'status':'diagnostic','cache_manifest_SHA256':b.sha(cache/'manifest.json'),
               'released_hyperparameters':dict(scope.HYPER),'variant_config':dict(scope.CFG)}
            worker.validate_curve_checkpoint(q,auth,5000)
            with self.assertRaises(b.Refusal):worker.validate_reference(q,auth)
            for updates in (0,7013,19999,20000):
                with self.assertRaises(b.Refusal):worker.validate_curve_checkpoint(q,auth,updates)
            for change in ({'status':'completed'},{'completed_updates':10000},
                    {'released_hyperparameters':{**scope.HYPER,'particle_lr':scope.HYPER['particle_lr']*2}},{'seed':1}):
                with self.assertRaises(b.Refusal):worker.validate_curve_checkpoint({**q,**change},auth,5000)
        self.assertNotIn('torch',sys.modules)
    def test_active_stage_trial_blocks_native_deployment_but_not_readonly_preparation(self):
        original_read=profile.read
        def active_stage(path):
            if Path(path)==profile.PREVIOUS_REGISTRY:
                return {'jobs':{'stage':{'status':'active'}},'fixture_only':'PRIVATE_ACTIVE_STAGE_COUNTERFACTUAL'}
            return original_read(path)
        with patch.object(profile,'read',side_effect=active_stage):
            auth=self.auth()
            self.assertEqual(auth['operation'],'train_released_hyperparameters')
            with self.assertRaisesRegex(AssertionError,'must release'):
                profile.configure(auth,check_only=False)
            with self.assertRaisesRegex(AssertionError,'must release'):
                profile.reviewed()
        self.assertFalse(profile.DEPLOYED.exists())
        self.assertFalse((profile.REGISTRY/'registry.json').exists())
    def test_changed_model_proof_source_refuses(self):
        auth=self.auth();auth['files_SHA256'][str(scope.ROOT/'analysis/released_tuning/released_update.py')]='0'*64;rd,sh=self.active(auth)
        with rd,sh,self.assertRaises(AssertionError):scope.validate_authorization(auth,require_runtime_proof=False)
    def test_deadline_cannot_reset_or_expand_and_issue_must_be_finite(self):
        for change in ({'expires_unix':scope.read(scope.WINDOW)['expires_unix']+1},{'expires_unix':1e20},{'issued_unix':float('inf')},{'issued_unix':float('nan')}):
            auth=self.auth();auth.update(change);rd,sh=self.active(auth)
            with rd,sh,self.assertRaises(AssertionError):scope.validate_authorization(auth,require_runtime_proof=False)
    def test_exact_old_execution_sources_retained(self):
        root=scope.ROOT;attempt='CAL_GPU0_V1_pure_confirm_0da2a09092c823377ca5_f0_pg_s2'
        auth=b.read(root/'artifacts/allocations'/attempt/'authorization.json')
        milestone=b.read(root/'analysis/seed2_PG8_completed_milestone.json');archive=Path(milestone['actual_execution_source_archive'])
        for path,pin in auth['research_source_SHA256'].items():
            self.assertEqual(b.sha(path),pin);self.assertEqual(b.sha(archive/path.lstrip('/')),pin)
        self.assertFalse(profile.DEPLOYED.exists());self.assertFalse((profile.REGISTRY/'registry.json').exists())
    def test_native_source_and_profile_wire_without_deployment(self):
        # All substitutions remain private in this process. Counterfactual wire
        # evidence is never written to the native proof or deployed policy path.
        with tempfile.TemporaryDirectory(prefix='alz-mog-prior-wire-') as temp:
            root=Path(temp);authpath=root/'auth.json';reviewpath=root/'review.json'
            runtimeproof={'status':'PASSED_ISOLATED_RELEASED_PARAMETER_RUNTIME_CPU_QUALIFICATION','native_profile_wire_and_refusals_verified':True,
                          'old_execution_sources_unchanged':True,'GPU_allocation_created':False,'prior_active_trial_native_launch_refusal_verified':True,'source_SHA256':{str(p):b.sha(p) for p in HERE.glob('*.py')},
                          'fixture_only':'COUNTERFACTUAL_PRIVATE_WIRE_NOT_ACTUAL_QUALIFICATION'}
            proof_bytes=(json.dumps(runtimeproof,indent=2)+'\n').encode();proof_pin=hashlib.sha256(proof_bytes).hexdigest()
            plan_bytes=(json.dumps(scope.ENABLED,indent=2)+'\n').encode();plan_pin=hashlib.sha256(plan_bytes).hexdigest()
            original_bread=b.read;original_bsha=b.sha;original_sread=scope.read;original_ssha=scope.sha;original_psha=profile.sha
            def fake_read(path):
                path=Path(path)
                if path==scope.RUNTIME_PROOF:return copy.deepcopy(runtimeproof)
                if path==scope.PLAN:return dict(scope.ENABLED)
                if path==profile.DEPLOYED:return json.loads(profile.deployed_bytes())
                if path==profile.PREVIOUS_REGISTRY:return {'jobs':{},'fixture_only':'PRIVATE_RELEASED_COUNTERFACTUAL_NOT_NATIVE_STATE'}
                return original_bread(path)
            def fake_sha(path,otherwise):
                path=Path(path)
                if path==scope.RUNTIME_PROOF:return proof_pin
                if path==scope.PLAN:return plan_pin
                if path==profile.DEPLOYED:return profile.policy_sha()
                return otherwise(path)
            with patch.object(b,'read',side_effect=fake_read),patch.object(scope,'read',side_effect=fake_read),patch.object(profile,'read',side_effect=fake_read),\
                 patch.object(b,'sha',side_effect=lambda p:fake_sha(p,original_bsha)),patch.object(scope,'sha',side_effect=lambda p:fake_sha(p,original_ssha)),\
                 patch.object(profile,'sha',side_effect=lambda p:fake_sha(p,original_psha)),patch.object(profile,'REVIEW',reviewpath):
                review={'status':'OWNER_AUTHORIZED_SOURCE_AND_RESOURCE_REVIEW_PASS','reviewer_session':'01a1a22b-f310-7190-a7c6-8697ecabae6c','independent_review_claimed':False,
                        'owner_authorization':{'available_capacity':'I authorize using the available GPU capacity.','both_cards':'You can use both gpus without permission.','sharing':'you dont need solo access','continuation':'contineue','deadline_extension':'extend it to 5 days'},
                        'scope':'ONE_RELEASED_LR_G025_D050_P050_SEED0_TRIAL_ONLY','policy_SHA256':profile.policy_sha(),'source_SHA256':profile.review_sources(),
                        'fixture_only':'COUNTERFACTUAL_PRIVATE_REVIEW_WIRE_NOT_AN_ACTUAL_REVIEW'}
                reviewpath.write_text(json.dumps(review))
                auth,_=prepare.build_authorization(check_only=False,auth_path=authpath)
                auth['runtime_qualification_SHA256']=proof_pin;auth['files_SHA256'][str(scope.RUNTIME_PROOF)]=proof_pin
                auth['files_SHA256'].update(runtimeproof['source_SHA256']);auth['files_SHA256'][str(scope.PLAN)]=plan_pin
                authpath.write_text(json.dumps(auth));self.assertEqual(b.verify(authpath)[0]['operation'],'train_released_hyperparameters')
                _,admission,_=b.modules(auth);self.assertEqual(Path(admission.__file__).name,'shared_gpu1_admission.py')
                bad=copy.deepcopy(auth);bad['operation']='train_bcap';authpath.write_text(json.dumps(bad))
                with self.assertRaises(b.Refusal):b.verify(authpath)
                bad=copy.deepcopy(auth);bad['dynamic_shared_foreign']=False;authpath.write_text(json.dumps(bad))
                with self.assertRaises(b.Refusal):b.verify(authpath)
            self.assertFalse(profile.DEPLOYED.exists());self.assertFalse((profile.REGISTRY/'registry.json').exists())
            self.assertNotIn('torch',sys.modules)

if __name__=='__main__':unittest.main()
