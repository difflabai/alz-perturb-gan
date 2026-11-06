"""Real duration authorizations and lower native gates, no model import."""
import copy,sys,unittest
from unittest.mock import patch
import binding as b,preset_scope as scope,prepare
class Scope(unittest.TestCase):
    def setUp(self):
        self.reader=scope.read
        self.mock=patch.object(scope,'read',side_effect=lambda p:dict(scope.ENABLED) if p==scope.PLAN else self.reader(p));self.mock.start()
    def tearDown(self):self.mock.stop()
    def auth(self):return prepare.build_authorization(check_only=True)[0]
    def test_exact_parent_endpoint_native_version_and_calibration_binding(self):
        a=self.auth();scope.validate_authorization(a,require_runtime_proof=False)
        self.assertEqual(b.cell(a),scope.cell(0,'pg',0));self.assertNotIn('torch',sys.modules)
    def test_native_lower_gate_and_scope_refuse_external_test_or_other_cells(self):
        a=self.auth()
        for changes in ({'operation':'external_benchmark_cell'},{'cell':{**a['cell'],'seed':1}},{'cell':{**a['cell'],'fold':1}},{'cell':{**a['cell'],'experiment':'external_b1'}}):
            q=copy.deepcopy(a);q.update(changes)
            with self.assertRaises(b.Refusal):b.cell(q)
            with self.assertRaises(AssertionError):scope.validate_authorization(q,require_runtime_proof=False)
    def test_no_parameter_release_parent_clock_or_metric_switch(self):
        a=self.auth()
        for key,value in [('metric_role','final_evaluation'),('fixed_endpoint',60000),('initial_completed_updates',0),('new_model_updates',40000),('particlegan_version','0.9.5'),('source_model_SHA256','wrong'),('reuse_completed_model',False),('released_hyperparameters',{**scope.HYPER,'reg_kappa':1.})]:
            q=copy.deepcopy(a);q[key]=value
            with self.assertRaises(AssertionError):scope.validate_authorization(q,require_runtime_proof=False)
if __name__=='__main__':unittest.main(verbosity=2)
