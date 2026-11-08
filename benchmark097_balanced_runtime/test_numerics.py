"""Numerical analysis equivalence and mandatory gates before target arrays."""
import ast,importlib.util,os,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path('/ml2/alz')
def tree(path):return ast.parse(Path(path).read_text())
def node(t,name):return ast.dump(next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)
class Numerics(unittest.TestCase):
 def test_distribution_primary_numerical_functions_are_identical(self):
  old=tree(ROOT/'analysis/aggregate_benchmark097_dual.py');new=tree(ROOT/'analysis/aggregate_benchmark097_balanced.py')
  for name in ('summarize_points','contrast_interval','main'):self.assertEqual(node(old,name),node(new,name),name)
 def test_prespecified53_numerics_unchanged_except_source_provenance_path(self):
  old=(ROOT/'analysis/score_calibration_excluded53_dual.py').read_text();new=(ROOT/'analysis/score_calibration_excluded53_balanced.py').read_text().replace('benchmark097_balanced_runtime','benchmark097_dual_runtime').replace('aggregate_benchmark097_balanced','aggregate_benchmark097_dual')
  self.assertEqual(old,new)
 def test_incomplete_actual_matrix_refuses_before_numpy_target_load(self):
  path=ROOT/'analysis/aggregate_benchmark097_balanced.py';spec=importlib.util.spec_from_file_location('balanced_aggregate_under_test',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  with patch.object(m.np,'load',side_effect=AssertionError('Target values must not be read')) as loader:
   with self.assertRaises(FileNotFoundError):m.load_complete_audits()
   self.assertFalse(loader.called)
if __name__=='__main__':unittest.main(verbosity=2)
