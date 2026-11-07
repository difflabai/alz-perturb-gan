import itertools,math,unittest
from released097_remaining_runtime_balance import balance_counts,measured_costs
class Balance(unittest.TestCase):
 def test_exact_work_conservation_and_crosschecked_optimal_makespan(self):
  counts={'pg':3,'cfgen':2};costs={'pg':[5.,1.],'cfgen':[4.,3.]};q=balance_counts(counts,costs)
  expected=min(max(a*5+b*4,(3-a)+3*(2-b)) for a,b in itertools.product(range(4),range(3)))
  self.assertEqual(q['predicted_makespan_seconds'],expected)
  for method in counts:self.assertEqual(sum(q['counts_per_lane'][method]),counts[method])
 def test_empty_queue_and_existing_work_are_not_restarted(self):
  q=balance_counts({'pg':0},{'pg':[2.,1.]},initial=(7.,3.));self.assertEqual(q['predicted_lane_seconds'],[7.,3.]);self.assertEqual(q['counts_per_lane']['pg'],[0,0])
 def test_slow_card_need_not_get_half_the_work(self):
  q=balance_counts({'pg':6},{'pg':[4.,1.]});self.assertEqual(q['counts_per_lane']['pg'],[1,5]);self.assertEqual(q['predicted_makespan_seconds'],5.)
 def test_bad_costs_and_missing_card_samples_are_rejected(self):
  for values in ([0.,1.],[float('nan'),1.],[float('inf'),1.]):
   with self.assertRaises(ValueError):balance_counts({'pg':1},{'pg':values})
  with self.assertRaises(ValueError):measured_costs([],['a','b'])
if __name__=='__main__':unittest.main(verbosity=2)
