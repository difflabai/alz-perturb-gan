"""Bracket or extend deterministically from seed0; no fresh-seed feedback."""
import importlib.util,math,unittest
from pathlib import Path
import preset_scope as s
path=Path(__file__).resolve().parents[1]/'analysis/hpo500_boundary/search.py'
if not path.exists():path=s.BOUNDARY/'search.py'
spec=importlib.util.spec_from_file_location('boundary_search_tests',path);search=importlib.util.module_from_spec(spec);spec.loader.exec_module(search)
class Search(unittest.TestCase):
 def records(self,es):
  a=search.anchor();return [dict(candidate_id=f'k{i:03d}',attempt=f'trial{i}',hyperparameters=a['initial_proposals'][i]['hyperparameters'],metrics=dict(energy_score=e,cell_energy_distance=.3,spread_ratio=1.),finite=True,seed=0,phase='search',endpoint=500) for i,e in enumerate(es)]
 def test_four_initial_proposals_frozen(self):
  records=[]
  for i in range(4):
   q=search.proposal(i,records);self.assertEqual(q['hyperparameters'],search.anchor()['initial_proposals'][i]['hyperparameters']);self.assertFalse(q['fresh_confirmation_results_used']);records.append(self.records([2.,2.,2.,2.])[i])
 def test_interior_best_gets_geometric_neighbors(self):
  k=search.anchor()['anchor_hyperparameters']['reg_kappa'];values=search.refinements(self.records([1.,1.2,1.3,1.4]));self.assertAlmostEqual(values[0],k*.5/math.sqrt(2));self.assertAlmostEqual(values[1],k*.5*math.sqrt(2))
 def test_lower_edge_extends_and_fifth_outcome_cannot_change_sixth(self):
  records=self.records([1.4,1.3,1.2,1.]);k=search.anchor()['anchor_hyperparameters']['reg_kappa'];self.assertEqual(search.refinements(records),[k*.03125,k*.015625])
  fifth={**records[0],'candidate_id':'k004','attempt':'trial4','metrics':dict(energy_score=-100,cell_energy_distance=.1,spread_ratio=1.)};self.assertEqual(search.proposal(5,records+[fifth])['hyperparameters']['reg_kappa'],k*.015625)
 def test_negative_infeasible_and_failed_outcomes_do_not_displace_anchor(self):
  rows=self.records([9.,9.,9.,9.]);self.assertEqual(search.refinements(rows),[]);rows[0].update(finite=False,metrics=None);rows[1]['metrics'].update(energy_score=-99,cell_energy_distance=99);rows[2]['metrics'].update(energy_score=-99,spread_ratio=.1);self.assertEqual(search.best(rows)['candidate_id'],search.anchor()['anchor_candidate'])
 def test_fresh_seeds_and_extra_candidates_refused(self):
  rows=self.records([1.,1.2,1.3,1.4]);rows[0]['seed']=1
  with self.assertRaises(AssertionError):search.proposal(4,rows)
  with self.assertRaises(AssertionError):search.proposal(6,rows)
if __name__=='__main__':unittest.main(verbosity=2)
