"""Exact small-population checks for a proposed clustered energy estimator."""
import itertools,math,unittest
import torch
from estimator import pooled_cell_energy as energy,stage_importance_weights as weights

class Energy(unittest.TestCase):
    def panel(self):
        gen=torch.Generator().manual_seed(207)
        return (torch.randn(3,4,5,generator=gen,dtype=torch.float64),
                torch.randn(3,4,5,generator=gen,dtype=torch.float64),
                torch.tensor([.5,1.,2.],dtype=torch.float64))

    def test_explicit_crossbag_pair_enumeration(self):
        real,fake,w=self.panel();terms=[]
        for i,j in itertools.permutations(range(3),2):
            for a,b in itertools.product(range(4),repeat=2):
                terms.append(w[i]*w[j]*(2*(fake[i,a]-real[j,b]).norm()
                    -(fake[i,a]-fake[j,b]).norm()-(real[i,a]-real[j,b]).norm()))
        expected=torch.stack(terms).mean()/math.sqrt(5)
        torch.testing.assert_close(energy(real,fake,w),expected,rtol=1e-13,atol=1e-13)

    def test_exact_unequal_stage_expectation_matches_uniform_target_energy(self):
        real=torch.tensor([[0.,1.],[2.,1.],[2.,3.],[-1.,-2.]],dtype=torch.float64)
        fake=torch.tensor([[1.,2.],[0.,1.],[3.,4.],[-2.,-3.]],dtype=torch.float64)
        counts=torch.tensor([1,2,3,4]);prob=counts.double()/counts.sum()
        estimate=torch.tensor(0.,dtype=torch.float64)
        for i,j in itertools.product(range(4),repeat=2):
            stages=torch.tensor([i,j]);w=weights(stages,counts,dtype=torch.float64)
            estimate+=prob[i]*prob[j]*energy(real[stages,None,:],fake[stages,None,:],w)
        expected=(2*torch.cdist(fake,real).mean()-torch.cdist(fake,fake).mean()-torch.cdist(real,real).mean())/math.sqrt(2)
        torch.testing.assert_close(estimate,expected,rtol=1e-13,atol=1e-13)
        self.assertGreater(float(expected),0)

    def test_correlated_matched_bag_counterexample_unbiased_but_naive_V_biased(self):
        # Both marginals are uniform on {-1,+1}; real/fake in one bag are
        # perfectly anticorrelated. True marginal energy is exactly zero.
        estimates=[];naive=[]
        for a,b in itertools.product((-1.,1.),repeat=2):
            real=torch.tensor([a,b],dtype=torch.float64).reshape(2,1,1);fake=-real
            estimates.append(energy(real,fake,torch.ones(2,dtype=torch.float64)))
            x=fake.flatten(0,1);y=real.flatten(0,1)
            naive.append(2*torch.cdist(x,y).mean()-torch.cdist(x,x).mean()-torch.cdist(y,y).mean())
        self.assertEqual(float(torch.stack(estimates).mean()),0.)
        self.assertEqual(float(torch.stack(naive).mean()),2.)
        self.assertTrue(any(float(v)<0 for v in estimates),'Do not clamp a negative minibatch U-statistic')

    def test_symmetry_bag_and_selected_cell_permutation_replication(self):
        real,fake,w=self.panel();reference=energy(real,fake,w)
        torch.testing.assert_close(reference,energy(fake,real,w))
        bags=torch.tensor([2,0,1]);cells=torch.tensor([2,0,3,1])
        torch.testing.assert_close(reference,energy(real[bags][:,cells],fake[bags].flip(1),w[bags]))
        torch.testing.assert_close(reference,energy(real.repeat_interleave(2,1),fake.repeat_interleave(2,1),w))

    def test_finite_differentiable_gradient_and_invalid_inputs(self):
        real,fake,w=self.panel();fake.requires_grad_(True)
        value=energy(real,fake,w);grad=torch.autograd.grad(value,fake)[0]
        self.assertTrue(torch.isfinite(grad).all());self.assertGreater(float(grad.norm()),0)
        # Finite difference is independent of the autograd implementation.
        epsilon=1e-6;plus=fake.detach().clone();minus=plus.clone()
        plus[0,0,0]+=epsilon;minus[0,0,0]-=epsilon
        numeric=(energy(real,plus,w)-energy(real,minus,w))/(2*epsilon)
        torch.testing.assert_close(grad[0,0,0],numeric,rtol=1e-5,atol=1e-8)
        for bad in (torch.zeros_like(w),w*float('nan'),w.clone().requires_grad_(True),w.float()):
            with self.assertRaises(ValueError):energy(real,fake,bad)
        with self.assertRaises(ValueError):energy(real[:1],fake[:1],w[:1])
        with self.assertRaises(ValueError):energy(real,fake[...,:4],w)
        for counts in (torch.tensor([0,1,2,3]),torch.ones(3,dtype=torch.long),torch.ones(4)):
            with self.assertRaises(ValueError):weights(torch.tensor([0,1]),counts,dtype=torch.float64)

if __name__=='__main__':unittest.main()
