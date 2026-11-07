"""Equal-fold point weighting, donor weighting and paired interval integration."""
import copy,sys,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path('/ml2/alz');P=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark')
sys.path[:0]=[str(ROOT/'analysis'),str(P/'current_base_20261029/runtime/lib/python3.12/site-packages'),str(P/'.venv312/lib/python3.12/site-packages')]
import numpy as np
from scipy.spatial.distance import cdist
import aggregate_benchmark097_dual as a

def point_records():
 records={}
 for f,n in enumerate((20,20,17)):
  for seed in range(5):
   for rep in (0,1):
    offset=.1*seed+.01*rep;e=f+1+offset
    records[f,seed,rep]=dict(donors=[f'f{f}_donor{i}' for i in range(n)],per_donor_energy_score=[e]*n,energy_score=e,
      cell_energy_distance=(f+1)**2+offset,spread_ratio=1.,within_condition_spread_ratio=.9,coverage80=.8,within_stage_spread_ratio_fit_reference=.95)
 return records

def reservoirs():
 rng=np.random.default_rng(874);targets={};pg={};other={}
 for f in range(3):
  real=rng.normal(size=(16,3));donors=[f'f{f}_d{i}' for i in range(8)]
  targets[f]=dict(cells=real,donors=donors,labels=np.repeat(donors,2),stages=np.repeat(range(4),2))
  for s in range(5):
   for rep in (0,1):
    noise=rng.normal(0,.01,size=real.shape);pg[f,s,rep]=real+noise+.05;other[f,s,rep]=real+noise+5.
 return pg,other,targets

class Aggregation(unittest.TestCase):
 def test_primary_equal_fold_weight_and_separate_pooled57_donor_energy(self):
  q=a.summarize_points(point_records());offset=.205
  self.assertAlmostEqual(q['cell_energy_distance'],(1+4+9)/3+offset)
  self.assertAlmostEqual(q['pooled_donor_energy_score'],(20+40+51)/57+offset)
  self.assertAlmostEqual(q['equal_fold_energy_score'],2+offset)
  self.assertNotAlmostEqual(q['pooled_donor_energy_score'],q['equal_fold_energy_score'])
  self.assertEqual(len(q['fold_seed_rows']),15);self.assertEqual(q['models_outside_spread_band'],0)
 def test_bad_spread_retained_and_incomplete_or_corrupt_points_refused(self):
  r=point_records()
  for rep in (0,1):r[0,0,rep]['spread_ratio']=.1
  q=a.summarize_points(r);self.assertEqual(q['models_outside_spread_band'],1);self.assertEqual(len(q['fold_seed_rows']),15)
  incomplete=dict(r);incomplete.pop((2,4,1))
  with self.assertRaises(ValueError):a.summarize_points(incomplete)
  invalid=copy.deepcopy(r);invalid[0,0,0]['energy_score']+=1
  with self.assertRaises(ValueError):a.summarize_points(invalid)
 def test_interval_zero_for_identical_fitted_model_reservoirs(self):
  pg,_,targets=reservoirs();q=a.contrast_interval(pg,pg,targets,draws=400)
  self.assertEqual((q['mean_delta'],q['lower'],q['upper']),(0.,0.,0.))
  self.assertFalse(q['independent_population_confirmation'])
 def test_paired_fixed_model_interval_point_matches_original_pooled_distance(self):
  pg,other,targets=reservoirs();q=a.contrast_interval(pg,other,targets,draws=400)
  values=[]
  def distance(g,r):return 2*cdist(g,r).mean()-cdist(g,g).mean()-cdist(r,r).mean()
  for key in a.KEYS:
   real=targets[key[0]]['cells'];values.append(distance(pg[key],real)-distance(other[key],real))
  self.assertAlmostEqual(q['mean_delta'],float(np.mean(values)),places=12)
  self.assertLess(q['upper'],0)
 def test_missing_target_donor_is_refused_instead_of_silently_removed(self):
  pg,other,targets=reservoirs();targets[1]['donors'].append('DECLARED_MISSING_DONOR')
  targets[1]['stages']=np.r_[targets[1]['stages'],0]
  with self.assertRaisesRegex(ValueError,'every declared donor'):a.contrast_interval(pg,other,targets,draws=100)
 def test_unfrozen_protocol_stops_before_any_target_array_read(self):
  with patch.object(a.scope,'protocol',side_effect=AssertionError('TEST_UNFROZEN_PROTOCOL')),patch.object(a.np,'load',side_effect=AssertionError('TARGET_READ')) as loader:
   with self.assertRaisesRegex(AssertionError,'TEST_UNFROZEN_PROTOCOL'):a.load_complete_audits()
   loader.assert_not_called()

if __name__=='__main__':unittest.main(verbosity=2)
