"""Synthetic arithmetic and dependence tests; no real subject values read."""
import copy,unittest
import numpy as np
from supplementary25_statistics import shared_donor_interval
class SharedSubjects(unittest.TestCase):
 def setUp(self):
  self.donors=[f'SYNTH_{i}' for i in range(8)];self.stages=np.repeat(np.arange(4),2)
  self.base=dict(donors=self.donors,cross_sums=np.array([1,6,3,12,5,18,7,24],float),cell_counts=np.array([1,2,1,2,1,2,1,2]),generated_constant=-.5)
 def manual(self,components,draws=1000,seed=7,interval=95):
  rng=np.random.default_rng(seed);chosen=np.concatenate([rng.choice(np.where(self.stages==s)[0],size=(draws,2),replace=True) for s in range(4)],axis=1)
  values=np.mean([c['cross_sums'][chosen].sum(1)/c['cell_counts'][chosen].sum(1)+c['generated_constant'] for c in components],axis=0)
  a=(100-interval)/2;return np.percentile(values,[a,100-a])
 def test_identical_views_preserve_full_subject_uncertainty_without_tripling_sample_size(self):
  c=[copy.deepcopy(self.base) for _ in range(3)];r=shared_donor_interval(c,self.stages,draws=1000,seed=7,interval_percent=95)
  expected=self.manual(c);np.testing.assert_allclose([r['lower'],r['upper']],expected,rtol=0,atol=1e-15)
  self.assertEqual(r['biological_subject_count'],8);self.assertTrue(r['same_subject_draw_reused_across_all_three_views'])
  self.assertAlmostEqual(r['mean_delta'],self.base['cross_sums'].sum()/self.base['cell_counts'].sum()-.5)
 def test_unequal_cell_counts_use_view_specific_ratios_with_one_global_subject_draw(self):
  c=[copy.deepcopy(self.base) for _ in range(3)];c[1]['cross_sums']*=2;c[1]['cell_counts']*=3;c[2]['generated_constant']=.7;c[2]['cell_counts'][0]=11
  r=shared_donor_interval(c,self.stages,draws=1000,seed=7,interval_percent=95)
  np.testing.assert_allclose([r['lower'],r['upper']],self.manual(c),rtol=0,atol=1e-15)
  self.assertAlmostEqual(r['mean_delta'],np.mean([x['cross_sums'].sum()/x['cell_counts'].sum()+x['generated_constant'] for x in c]))
 def test_different_order_or_missing_subject_cannot_be_silently_paired(self):
  for key,value in [('donors',self.donors[::-1]),('cell_counts',np.array([0,2,1,2,1,2,1,2])),('cross_sums',np.arange(7))]:
   c=[copy.deepcopy(self.base) for _ in range(3)];c[1][key]=value
   with self.assertRaises(ValueError):shared_donor_interval(c,self.stages)
 def test_invalid_counts_contributions_or_constants_refuse(self):
  for key,value in [('cell_counts',np.full(8,1.5)),('cell_counts',np.full(8,np.nan)),('cross_sums',np.full(8,np.inf)),('generated_constant',float('nan'))]:
   c=[copy.deepcopy(self.base) for _ in range(3)];c[1][key]=value
   with self.assertRaises(ValueError):shared_donor_interval(c,self.stages)
 def test_missing_stage_duplicate_donors_and_incomplete_views_refuse(self):
  c=[copy.deepcopy(self.base) for _ in range(3)]
  for stages in [np.zeros(8),np.arange(7)]:
   with self.assertRaises(ValueError):shared_donor_interval(c,stages)
  with self.assertRaises(ValueError):shared_donor_interval(c[:2],self.stages)
  c[0]['donors']=['same']*8
  with self.assertRaises(ValueError):shared_donor_interval(c,self.stages)
 def test_identical_method_components_have_exact_zero_interval(self):
  c=[copy.deepcopy(self.base) for _ in range(3)]
  for x in c:x['cross_sums'][:]=0;x['generated_constant']=0
  r=shared_donor_interval(c,self.stages)
  self.assertEqual((r['mean_delta'],r['lower'],r['upper']),(0,0,0))
if __name__=='__main__':unittest.main(verbosity=2)
