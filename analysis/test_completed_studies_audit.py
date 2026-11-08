"""Synthetic mathematical/gating checks; these are not completed ALZ outcomes."""
import copy,importlib.util,pathlib,sys,tempfile,unittest
ROOT=pathlib.Path('/ml2/alz');PROJECT=pathlib.Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark')
sys.path[:0]=[str(PROJECT/'current_base_20261029/runtime/lib/python3.12/site-packages'),str(PROJECT/'.venv312/lib/python3.12/site-packages')]
import numpy as np
spec=importlib.util.spec_from_file_location('completed_studies_audit',ROOT/'analysis/audit_released097_completed_studies.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class ScientificAudit(unittest.TestCase):
 def test_condensed_energy_statistic_has_known_analytic_value(self):
  a=np.array([[0.],[2.]]);b=np.array([[1.],[3.]])
  self.assertAlmostEqual(m.independent_cell_ED(a,b),1.,places=12);self.assertAlmostEqual(m.independent_cell_ED(a,a),0.,places=12)
 def test_distance_rejects_nonfinite_or_mismatched_dimensions(self):
  for a,b in [(np.array([[np.nan]]),np.ones((1,1))),(np.ones((1,2)),np.ones((1,3))),(np.empty((0,1)),np.ones((1,1)))]:
   with self.assertRaises(AssertionError):m.independent_cell_ED(a,b)
 def fixture_summary(self):
  donors={f:[f'D{f}_{i}' for i in range(f+1)] for f in range(3)}
  folds=[dict(fold=f,donors=donors[f],energy_score=float(f+1),per_donor_energy_score=[float(f+1)]*(f+1)) for f in range(3)]
  points=[dict(fold=f,seed=s,cell_energy_distance=.1,spread_ratio=.9,coverage80=.75) for f in range(3) for s in range(5)]
  return dict(folds=folds,fold_seed_rows=points,cell_energy_distance=.1,mean_spread_ratio=.9,mean_coverage80=.75,pooled_donor_energy_score=14/6,equal_fold_energy_score=2.),donors
 def test_donor_energy_pools_donors_and_retains_equal_fold_distinction(self):
  value,donors=self.fixture_summary();m.validate_summary(value,donors);value['pooled_donor_energy_score']=2.
  with self.assertRaises(AssertionError):m.validate_summary(value,donors)
 def test_missing_or_duplicate_model_points_cannot_pass(self):
  value,donors=self.fixture_summary();value['fold_seed_rows'][-1]=value['fold_seed_rows'][0]
  with self.assertRaises(AssertionError):m.validate_summary(value,donors)
 def contrast_fixture(self):
  summary={k:dict(cell_energy_distance=v) for k,v in [('pg',.1),('scvi',.2),('cfgen',.2),('hierarchical_resampling',.2)]};contrasts={k:dict(PG_minus_other_primary_distance=-.1,conditional_real_donor_cluster_interval=dict(mean_delta=-.1,lower=-.2,upper=-.01,interval_percent=98.33333333333333,draws=10000,model_and_generation_reservoirs_fixed=True,independent_population_confirmation=False,unit='real donor cluster, stratified by fold and stage')) for k in summary if k!='pg'};return summary,contrasts
 def test_point_win_does_not_replace_missing_or_crossing_uncertainty(self):
  s,c=self.contrast_fixture();self.assertTrue(m.tested_ED_win(s,c));c['scvi']['conditional_real_donor_cluster_interval']=None;self.assertFalse(m.tested_ED_win(s,c));s,c=self.contrast_fixture();c['cfgen']['conditional_real_donor_cluster_interval']['upper']=.01;self.assertFalse(m.tested_ED_win(s,c))
 def test_scalar_choice_is_energy_distance_with_fixed_tie_break(self):
  a=dict(fixed=dict(cell_energy_distance=dict(mean=.2),energy_score=dict(mean=2.)),adaptive=dict(cell_energy_distance=dict(mean=.3),energy_score=dict(mean=1.)));self.assertEqual(m.preferred_arm(a),'fixed');a['adaptive']['cell_energy_distance']['mean']=.2;self.assertEqual(m.preferred_arm(a),'fixed');a['adaptive']['cell_energy_distance']['mean']=.1;self.assertEqual(m.preferred_arm(a),'adaptive')
 def test_metadata_readiness_does_not_parse_partial_quality_reports(self):
  with tempfile.TemporaryDirectory() as folder:
   root=pathlib.Path(folder);paths=m.required_paths(root)
   for p in paths[:-1]:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'OPAQUE_REPORT_MUST_NOT_BE_PARSED')
   result=m.readiness(root);self.assertEqual(result['missing_paths'],[str(paths[-1])]);self.assertFalse(result['quality_values_read']);self.assertFalse(result['full_goal_complete'])
 def test_present_paths_still_do_not_claim_completed_audit_or_sota(self):
  with tempfile.TemporaryDirectory() as folder:
   root=pathlib.Path(folder)
   for p in m.required_paths(root):p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'OPAQUE')
   q=m.readiness(root);self.assertEqual(q['status'],'INPUT_PATHS_PRESENT_NOT_A_COMPLETION_CLAIM');self.assertFalse(q['full_goal_complete']);self.assertFalse(q['SOTA_claim']);self.assertFalse(q['quality_values_read'])
if __name__=='__main__':unittest.main()
