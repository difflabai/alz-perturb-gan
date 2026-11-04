"""Scientific anchor rule retains tradeoffs and requires complete valid records."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from summarize_released_search import choose_anchor

class Anchor(unittest.TestCase):
    def row(self,name,es,ed,spread=1,state='audited_complete'):
        return dict(id=name,energy_score=es,cell_energy_distance=ed,spread_ratio=spread,state=state)
    def test_balanced_tradeoff_wins_over_better_energy_with_worse_distribution(self):
        rows=[self.row('energy_only',.8,1.2),self.row('balanced',.95,.95),self.row('dominated',1.1,1.1)]
        self.assertEqual(choose_anchor(rows,dict(energy_score=1,cell_energy_distance=1))['id'],'balanced')
    def test_early_or_collapsed_or_failed_record_cannot_choose_regularization_anchor(self):
        rows=[self.row('partial',.01,.01,state='allocated_quality_not_audited'),
            self.row('failure',.01,.01,state='incomplete_operational_failure'),self.row('collapsed',.01,.01,spread=.1)]
        self.assertIsNone(choose_anchor(rows,dict(energy_score=1,cell_energy_distance=1)))
    def test_stable_id_breaks_true_ties_without_dropping_tradeoff_rows(self):
        a=self.row('a',.9,1.1);b=self.row('b',1.1,.9)
        for rows in ([a,b],[b,a]):self.assertEqual(choose_anchor(rows,dict(energy_score=1,cell_energy_distance=1))['id'],'a')
    def test_nonfinite_audited_metric_refuses(self):
        with self.assertRaises(ValueError):choose_anchor([self.row('invalid',float('nan'),1)],dict(energy_score=1,cell_energy_distance=1))
if __name__=='__main__':unittest.main()
