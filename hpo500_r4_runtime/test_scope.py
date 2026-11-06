"""Native matrix gates and exact500step scalar-only controls."""
import copy,sys,unittest
from unittest.mock import patch
import binding as b,preset_scope as scope,prepare
class Scope(unittest.TestCase):
    def setUp(self):
        self.reader=scope.read;self.mock=patch.object(scope,'read',side_effect=lambda p:dict(scope.ENABLED) if p==scope.PLAN else self.reader(p));self.mock.start()
    def tearDown(self):self.mock.stop()
    def test_control_authorization_is_fresh500_and_no_model_import(self):
        a,_=prepare.build_authorization(check_only=True);scope.validate_authorization(a,require_runtime_proof=False)
        self.assertEqual(a['released_hyperparameters'],scope.HYPER);self.assertEqual(a['new_model_updates'],500);self.assertNotIn('torch',sys.modules)
    def test_all25_search_ids_and_all_registered_confirmation_seeds_lower_native_gate(self):
        for index in range(25):
            for phase,seeds in [('search',[0]),('confirm',[1,2,3,4])]:
                for seed in seeds:
                    v=scope.cell(f'c{index:03d}',seed,phase)
                    self.assertEqual(b.cell(dict(cell=v,cpu_fixture=False,operation='hpo500_cell')),v)
    def test_other_formulations_horizons_reuse_roles_or_hyperparameters_refuse(self):
        a,_=prepare.build_authorization(check_only=True)
        for key,value in [('operation','external_benchmark_cell'),('fixed_endpoint',20000),('metric_role','final_evaluation'),('reuse_completed_model',True),('initial_completed_updates',20000),('new_model_updates',0),('particlegan_version','0.9.5'),('cap_seconds',21600),('released_hyperparameters',{**scope.HYPER,'reg_kappa':1.})]:
            q=copy.deepcopy(a);q[key]=value
            with self.assertRaises(AssertionError):scope.validate_authorization(q,require_runtime_proof=False)
    def test_undeclared_candidate_seed_or_fold_refused(self):
        for cid,seed,phase in [('c025',0,'search'),('c000',1,'search'),('c000',0,'confirm')]:
            with self.assertRaises(AssertionError):scope.cell(cid,seed,phase)
        a,_=prepare.build_authorization(check_only=True);a['cell']['fold']=1
        with self.assertRaises(b.Refusal):b.cell(a)
    def test_confirmation_cannot_run_before_frozen_shortlist(self):
        with self.assertRaises(FileNotFoundError):prepare.build_authorization(candidate='c000',seed=1,phase='confirm',check_only=True)
    def test_declared_resource_resume_has420plus80_same_parameters_and_original_expiry(self):
        a,_=prepare.build_authorization(candidate='c008',resume=True,check_only=True)
        scope.validate_authorization(a,require_runtime_proof=False)
        self.assertEqual(b.cell(a),scope.cell('c008',0,'search',True))
        self.assertEqual(a['initial_completed_updates'],420);self.assertEqual(a['new_model_updates'],80)
        self.assertEqual(a['cap_seconds'],3139)
        for key,value in [('initial_completed_updates',0),('new_model_updates',500),('cap_seconds',3600),('source_model_SHA256','wrong'),('released_hyperparameters',scope.HYPER)]:
            changed=copy.deepcopy(a);changed[key]=value
            with self.assertRaises(AssertionError):scope.validate_authorization(changed,require_runtime_proof=False)
    def test_smaller_own_GPU_request_keeps_foreign_growth_driver_and_host_envelopes(self):
        import resource_profile as p
        q=p.validate_proposal()
        self.assertEqual(q['roles']['alz_core_pg']['gpu_bytes'],2*2**30)
        self.assertEqual(q['roles']['alz_core_pg']['ram_bytes'],6*2**30)
        self.assertEqual(q['driver_headroom_bytes'],3*2**30)
        self.assertEqual(q['host_free_floor_bytes'],12*2**30)
        self.assertEqual(q['foreign_growth_allowances']['compute_GPU_bytes'],2**29)
    def test_GPU1_native_device_uses_matching_shared_guard_and_physical_lock_pair(self):
        a,_=prepare.build_authorization(check_only=True)
        self.assertEqual(b.device(a),b.GPU1)
        _,admission,guard=b.modules(a)
        self.assertEqual(admission.GPU0,b.GPU1)
        self.assertEqual(admission.LOCKS,('/mnt/nvme/sliders/qualification/admission-gpu-1.lock','/mnt/nvme/sliders/locks/gpu-1.lock'))
        self.assertEqual(__import__('pathlib').Path(guard.__file__).resolve(),b.HERE.parent/'recovery_runtime/shared_gpu1_runner.py')
        self.assertFalse(a['dynamic_shared_gpu0'])
if __name__=='__main__':unittest.main(verbosity=2)
