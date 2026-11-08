import importlib.util,pathlib,sys,unittest
ROOT=pathlib.Path('/ml2/alz');BASE=pathlib.Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark');sys.path[:0]=[str(BASE/'current_base_20261029/runtime/lib/python3.12/site-packages'),str(BASE/'.venv312/lib/python3.12/site-packages')]
import numpy as np
spec=importlib.util.spec_from_file_location('nb_variance',ROOT/'analysis/t005_nb_variance_cpu.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Math(unittest.TestCase):
 def test_analytic_mixture_keeps_all_three_variance_terms(self):
  s=m.moments_class(np)(1);s.add([[1.],[3.]]);p=m.components(np,s.mean,s.variance(),[2.]);self.assertEqual(float(p['mean_variation'][0]),1.);self.assertEqual(float(p['gamma_excess'][0]),2.5);self.assertEqual(float(p['poisson'][0]),2.);self.assertEqual(float(p['total'][0]),5.5)
 def test_unequal_chunk_merges_match_population_moments(self):
  x=np.random.default_rng(15).normal(size=(23,7));s=m.moments_class(np)(7)
  for a in (x[:1],x[1:8],x[8:]):s.add(a)
  np.testing.assert_allclose(s.mean,x.mean(0),rtol=0,atol=1e-12);np.testing.assert_allclose(s.variance(),x.var(0),rtol=0,atol=1e-12)
 def test_singleton_chunk_has_zero_mean_variation(self):
  s=m.moments_class(np)(2);s.add([[1.,2.]]);np.testing.assert_array_equal(s.variance(),[0.,0.])
 def test_invalid_dispersion_or_negative_variance_refused(self):
  for mean,var,theta in [([1.],[0.],[0.]),([1.],[-.1],[1.]),([-1.],[0.],[1.]),([1.],[0.],[np.inf]),([1.],[0.,0.],[1.])]:
   with self.assertRaises(AssertionError):m.components(np,mean,var,theta)
 def test_stage_conditional_sum_does_not_invent_between_stage_variance(self):
  means=np.array([[1.],[3.],[5.],[7.]]);p=m.components(np,means,np.zeros_like(means),np.full_like(means,2.));self.assertEqual(float(p['total'].sum()),58.);self.assertNotEqual(float(p['total'].sum()),78.)
 def test_large_theta_leaves_poisson_and_mean_variance_terms(self):
  p=m.components(np,[2.],[1.],[1e12]);self.assertAlmostEqual(float(p['total'][0]),3.,places=10);self.assertEqual(float(p['poisson'][0]),2.)
if __name__=='__main__':unittest.main()
