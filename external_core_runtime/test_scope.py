"""Exercise benchmark identity, checkpoint selection and all30 authorization cells."""
import copy,sys,unittest
from unittest.mock import patch
import binding as b,preset_scope as scope,prepare

class Scope(unittest.TestCase):
    def setUp(self):
        self.real_read=scope.read
        self.reader=patch.object(scope,'read',side_effect=lambda p:dict(scope.ENABLED) if p==scope.PLAN else self.real_read(p))
        self.reader.start()
    def tearDown(self):self.reader.stop()
    def auth(self,fold=0,method='pg',seed=0):
        return prepare.build_authorization(fold=fold,method=method,seed=seed,check_only=True)[0]
    def test_all30_matrix_cells_bind_exact_fold_seed_and_fixed_recipe(self):
        for fold in range(3):
            for method in ('pg','scvi'):
                for seed in range(5):
                    a=self.auth(fold,method,seed)
                    scope.validate_authorization(a,require_runtime_proof=False)
                    self.assertEqual(b.cell(a),scope.cell(fold,method,seed))
                    self.assertEqual(a['reuse_completed_model'],method=='scvi' and fold==seed==0)
                    self.assertEqual(a['fixed_endpoint'],20000 if method=='pg' else 400)
    def test_frozen_parameters_endpoint_and_role_cannot_be_changed(self):
        a=self.auth()
        for key,value in [('benchmark_protocol_SHA256','wrong'),('metric_role','calibration_selection'),('fixed_endpoint',10000),('reuse_completed_model',True),('new_model_updates',0),('particlegan_version','0.9.4')]:
            changed=copy.deepcopy(a);changed[key]=value
            with self.assertRaises(AssertionError):scope.validate_authorization(changed,require_runtime_proof=False)
        changed=copy.deepcopy(a);changed['released_hyperparameters']['reg_kappa']=1.
        with self.assertRaises(AssertionError):scope.validate_authorization(changed,require_runtime_proof=False)
    def test_foreign_seed_checkpoint_or_intermediate_cannot_be_reused(self):
        a=self.auth();a['source_model_SHA256']='wrong'
        with self.assertRaises(AssertionError):scope.validate_authorization(a,require_runtime_proof=False)
        a=self.auth(seed=1);a['source_model_path']=str(scope.model_input(scope.cell(0,'pg',0)))
        with self.assertRaises(AssertionError):scope.validate_authorization(a,require_runtime_proof=False)
    def test_unregistered_matrix_or_experiment_refuses(self):
        for args in [(3,'pg',0),(0,'cfgen',0),(0,'pg',5),(False,'pg',0)]:
            with self.assertRaises(AssertionError):scope.cell(*args)
        a=self.auth();a['cell']['experiment']='retuned_after_test'
        with self.assertRaises(AssertionError):scope.validate_authorization(a,require_runtime_proof=False)
    def test_preparation_imports_no_models_or_cuda(self):
        self.auth();self.assertNotIn('torch',sys.modules);self.assertNotIn('scvi',sys.modules)
    def test_lower_native_gate_refuses_unrelated_experiments_and_operations(self):
        for fold,method,seed in ((0,'pg',0),(2,'scvi',4)):
            a=self.auth(fold,method,seed)
            changed=copy.deepcopy(a);changed['cell']['experiment']='calibration_retune'
            with self.assertRaises(b.Refusal):b.cell(changed)
            changed=copy.deepcopy(a);changed['operation']='train_released_hyperparameters'
            with self.assertRaises(b.Refusal):b.cell(changed)

if __name__=='__main__':unittest.main(verbosity=2)
