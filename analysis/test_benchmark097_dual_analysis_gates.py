"""Resource adaptation cannot change metrics or weaken the full45 value gate."""
import ast,copy,json,os,unittest
from pathlib import Path
from unittest.mock import patch
import aggregate_benchmark097_dual as core
import score_calibration_excluded53_dual as sensitivity
R=Path('/ml2/alz')
class Gates(unittest.TestCase):
 def test_scientific_point_statistics_and_sensitivity_source_are_unchanged(self):
  def node(path,name):return next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
  for name in ('summarize_points','contrast_interval','main'):
   self.assertEqual(ast.dump(node(R/'analysis/aggregate_benchmark097.py',name)),ast.dump(node(R/'analysis/aggregate_benchmark097_dual.py',name)))
  original=(R/'analysis/score_calibration_excluded53.py').read_text()
  expected=original.replace('benchmark097_runtime','benchmark097_dual_runtime').replace('import aggregate_benchmark097 as core','import aggregate_benchmark097_dual as core').replace("ROOT/'analysis/aggregate_benchmark097.py'","ROOT/'analysis/aggregate_benchmark097_dual.py'")
  self.assertEqual(expected,(R/'analysis/score_calibration_excluded53_dual.py').read_text())
 def bad_matrix(self):
  return dict(matrix_complete=True,complete_cells=44,required_cells=45,protocol_SHA256=core.sha(core.scope.PROTOCOL),all_new_training_from_scratch=True)
 def test_incomplete_primary_matrix_refuses_before_target_values(self):
  real=core.read
  with patch.object(core.scope,'protocol',return_value={}),patch.object(core,'read',side_effect=lambda p:self.bad_matrix() if Path(p).name=='matrix_complete.json' else real(p)),patch.object(core.np,'load') as arrays:
   with self.assertRaises(AssertionError):core.main()
   self.assertEqual(arrays.call_count,0)
 def test_incomplete_sensitivity_matrix_refuses_before_target_values(self):
  real=core.read
  with patch.object(core.scope,'protocol',return_value={}),patch.object(core,'read',side_effect=lambda p:self.bad_matrix() if Path(p).name=='matrix_complete.json' else real(p)),patch.object(sensitivity.np,'load') as arrays,patch('sys.argv',['score_calibration_excluded53_dual.py']):
   with self.assertRaises(AssertionError):sensitivity.main()
   self.assertEqual(arrays.call_count,0)
if __name__=='__main__':unittest.main(verbosity=2)
