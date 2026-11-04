"""Qualify the final routed count adapter's boundaries and native continuation."""
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np
import torch
from particlegan import RoutedCandidate, get_recipe
from pg_model import critic_features
from critic import bag_features
from generation import trace
import variant_pg_adapter as original
import routed_pg_adapter as a
import routed_counts as rc
from partition import partition

ROOT=Path('/ml2/alz')


def same(x,y):
    if isinstance(x,torch.Tensor):
        assert x.dtype==y.dtype and x.shape==y.shape
        assert torch.equal(x.contiguous().reshape(-1).view(torch.uint8),y.contiguous().reshape(-1).view(torch.uint8))
    elif isinstance(x,np.ndarray):np.testing.assert_array_equal(x,y)
    elif isinstance(x,dict):
        assert x.keys()==y.keys()
        for k in x:same(x[k],y[k])
    elif isinstance(x,(tuple,list)):
        assert type(x)==type(y) and len(x)==len(y)
        for i,j in zip(x,y):same(i,j)
    elif isinstance(x,float):assert struct.pack('<d',x)==struct.pack('<d',y)
    else:assert x==y,(x,y)


def fixture(cache,genes=8):
    c=object.__new__(a.Context);c.cache=Path(cache);c.fold=0;c.device=torch.device('cpu')
    c.ids=np.asarray([f'd{i}' for i in range(6)]);c.fit_labels=np.repeat(c.ids,20)
    c.cell_names=np.asarray([f'cell{i}' for i in range(120)])
    c.X=torch.randint(1,8,(120,genes),generator=torch.Generator().manual_seed(20261104231)).float()
    c.partition=partition(c.fit_labels,c.cell_names,c.ids)
    c.by=c.partition.fitting_by_donor;c.guard_by=c.partition.protected_by_donor
    c.all_by=[np.flatnonzero(c.fit_labels==d) for d in c.ids]
    c.conditions=torch.zeros(6,7);c.conditions[:,3]=torch.tensor([0.,1/3,2/3,1.,1.,1.])
    c.fitc=c.conditions.numpy().copy();c.mean=torch.zeros(genes);c.sd=torch.ones(genes)
    c.W=torch.randn(genes,50,generator=torch.Generator().manual_seed(20261104232))/np.sqrt(genes)
    c.mu,c.theta=a.fitting_statistics(c.X,c.partition);c.fitlibs=c.X.sum(-1)
    return c


class RoutedModelChecks(unittest.TestCase):
    def test_native_recipe_and_original11seed_prefix(self):
        for seed in (0,1,2):
            observed=a.training_streams(0,seed);expected=original.training_streams(0,seed)
            self.assertEqual({k:observed[k] for k in expected},expected)
            self.assertEqual(set(observed)-set(expected),{'guard_sampler'})
        for fold,seed in [(1,0),(False,0),(0,True),(0,-1)]:
            with self.assertRaises(ValueError):a.training_streams(fold,seed)
        with tempfile.TemporaryDirectory() as d:
            ctx=fixture(d);r=a.build(ctx,0)
            self.assertEqual(r.recipe.to_dict(),get_recipe('e22_routed',total_steps=None,batch_size=32,num_particles=512,z_dim=8).to_dict())
            self.assertTrue(r.recipe.particle_birth_death and r.recipe.row_evidence_gate)
            self.assertEqual(r.policy.row_semantics,'dense_soft')

    def test_fitting_protected_sampling_is_disjoint_and_donor_local(self):
        with tempfile.TemporaryDirectory() as d:
            ctx=fixture(d)
            for role,pool in [('fitting',ctx.partition.fitting),('protected',ctx.partition.protected)]:
                rng=np.random.default_rng(20261104233)
                for _ in range(5):
                    counts,c,lib,rows=ctx.draw(role,rng)
                    self.assertTrue(np.isin(rows,pool).all())
                    same(counts,ctx.X[torch.as_tensor(rows)]);same(lib,counts.sum(-1))
                    for i,row in enumerate(rows):
                        donor=ctx.fit_labels[row[0]];self.assertEqual(set(ctx.fit_labels[row]),{donor})
                        index=ctx.ids.tolist().index(donor);same(c[i],ctx.conditions[index])
            with self.assertRaises(ValueError):ctx.draw('evaluation',np.random.default_rng(0))

    def test_protected_values_do_not_change_fitting_initial_statistics(self):
        with tempfile.TemporaryDirectory() as d:
            ctx=fixture(d);expected=a.fitting_statistics(ctx.X,ctx.partition)
            altered=ctx.X.clone();altered[torch.as_tensor(ctx.partition.protected.copy())]+=10000
            same(expected,a.fitting_statistics(altered,ctx.partition))

    def test_unsigned64bit_context_seed_round_trip(self):
        c=torch.zeros(1,7);lib=torch.full((1,8),40.)
        context=rc.make_context(c,lib,generator=torch.Generator().manual_seed(7))
        context[:,-12:]=65535
        self.assertEqual(rc.CountForward.unpack_seeds(context),[[2**64-1]*3])
        for seed in rc.CountForward.unpack_seeds(context)[0]:
            self.assertEqual(torch.Generator().manual_seed(seed).initial_seed(),seed)

    def test_invalid_context_and_target_domains_refused(self):
        with tempfile.TemporaryDirectory() as d:
            r=a.build(fixture(d),0)
            good=rc.make_context(torch.zeros(1,7),torch.full((1,8),40.),generator=torch.Generator().manual_seed(7))
            for col,value in [(0,float('nan')),(8,1.),(11,.5),(15,-1.),(-1,.5),(-1,65536.)]:
                bad=good.clone();bad[:,col]=value
                with self.assertRaises(ValueError):rc.validate_context(bad)
            for value in (-1.,.5,float('nan')):
                bad=torch.ones(1,8,8);bad[0,0,0]=value
                with self.assertRaises(ValueError):r.forward.targets(bad)

    def test_raw_forward_and_original_transformed_rate_gradient_exact(self):
        with tempfile.TemporaryDirectory() as d:
            ctx=fixture(d);r=a.build(ctx,0);p=r.policy
            context=rc.make_context(ctx.conditions[:1],torch.full((1,8),40.),generator=torch.Generator().manual_seed(20261104234))
            packed=p.routed_generate(context,sigma=0,perturb=False)
            raw=r.forward.raw_counts(packed)
            same(packed[...,8:],a.standardize_counts(raw,ctx.mean,ctx.sd))
            params=[p.G.out[-1].weight,p.G.log_theta,p.table]
            actual=torch.autograd.grad(r.forward.observation(packed).square().mean(),params)
            candidate=p.routed_control.candidate()
            weights=r.forward.route({},context,candidate);codes=weights@candidate.table
            gamma,cell,count=r.forward.unpack_seeds(context)[0]
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(gamma)
                rate=p.G.rates(codes,context[:,8:15],8,context[:,15:-12],torch.Generator().manual_seed(cell))
                counts=torch.poisson(rate.detach(),generator=torch.Generator().manual_seed(count))
                reference=bag_features(critic_features(counts,rate,ctx.mean,ctx.sd)@ctx.W)
            expected=torch.autograd.grad(reference.square().mean(),params)
            same(raw,counts);same(actual,expected)

    def test_candidate_codes_and_mass_are_obeyed(self):
        with tempfile.TemporaryDirectory() as d:
            r=a.build(fixture(d),0);p=r.policy
            context=rc.make_context(torch.zeros(1,7),torch.full((1,8),40.),generator=torch.Generator().manual_seed(7))
            candidate=p.routed_control.candidate();weights=r.forward.route({},context,candidate)
            from dataclasses import replace
            deleted=replace(candidate,log_mass=candidate.log_mass.detach().clone())
            deleted.log_mass[0]=-torch.inf
            after=r.forward.route({},context,deleted)
            self.assertEqual(float(after[0,0].detach()),0.)
            expected=weights.detach().clone();expected[:,0]=0;expected/=expected.sum(-1,keepdim=True)
            torch.testing.assert_close(after,expected,rtol=5e-3,atol=5e-5)
            models={'generator':p.G,'critic':p.D,'prior':p.prior,'router':p.router}
            one=replace(candidate,codes=(weights@candidate.table))
            other=replace(one,codes=one.codes+100)
            first=r.forward.generate(models,context,one,weights)
            second=r.forward.generate(models,context,other,weights)
            self.assertFalse(torch.equal(first,second))

    def test_protected_targets_cannot_change_first_optimizer_pair(self):
        with tempfile.TemporaryDirectory() as d:
            first=fixture(d);r=a.build(first,0);metrics=a.advance(first,r)
            state=copy.deepcopy(r.policy.state_dict())
            second=fixture(d);second.X[torch.as_tensor(second.partition.protected.copy())]+=1000
            second.mu,second.theta=a.fitting_statistics(second.X,second.partition)
            changed=a.build(second,0);same(metrics,a.advance(second,changed))
            same(state['models'],changed.policy.state_dict()['models'])
            same(state['optimizers'],changed.policy.state_dict()['optimizers'])
            same(state['output_noise'],changed.policy.state_dict()['output_noise'])

    def test_complete_dynamic_sampler_and_probe_boundary_replay(self):
        with tempfile.TemporaryDirectory() as d:
            cache=Path(d);(cache/'manifest.json').write_text('{}');ctx=fixture(cache);r=a.build(ctx,0)
            for _ in range(19):a.advance(ctx,r)
            torch.save(a.snapshot(ctx,r,'synthetic'),cache/'before19.pt')
            before=torch.load(cache/'before19.pt',map_location='cpu',weights_only=False)
            m20=a.advance(ctx,r);q20=a.snapshot(ctx,r,'synthetic')
            self.assertEqual(q20['policy']['routing']['probe_clock'],{'observed_updates':20,'last_probe_update':20})
            self.assertEqual(q20['policy']['routing']['counters']['probes'],8)
            m21=a.advance(ctx,r);q21=a.snapshot(ctx,r,'synthetic')
            restored=a.restore(ctx,before)
            same(m20,a.advance(ctx,restored));same(q20,a.snapshot(ctx,restored,'synthetic'))
            same(m21,a.advance(ctx,restored));same(q21,a.snapshot(ctx,restored,'synthetic'))
            self.assertEqual(restored.fitting_contexts_seen,672);self.assertEqual(restored.protected_contexts_seen,672)

    def test_checkpoint_cache_partition_recipe_and_clocks_refused(self):
        with tempfile.TemporaryDirectory() as d:
            cache=Path(d);(cache/'manifest.json').write_text('{}');ctx=fixture(cache);r=a.build(ctx,0);a.advance(ctx,r)
            q=a.snapshot(ctx,r,'synthetic')
            for field,value in [('cache_manifest_SHA256','0'*64),('native_recipe','e22'),
                                ('variant_config',{**a.CONFIG,'z_dim':2}),('fixed_endpoint',40000),
                                ('fitting_contexts_seen',0)]:
                bad=copy.deepcopy(q);bad[field]=value
                with self.assertRaises(ValueError):a.restore(ctx,bad)
            bad=copy.deepcopy(q);bad['partition']['membership_SHA256']['protected']='0'*64
            with self.assertRaises(ValueError):a.restore(ctx,bad)
            bad=copy.deepcopy(q);bad['recipe']['serve_average']=0
            with self.assertRaises(ValueError):a.restore(ctx,bad)

    def test_real_CPU_fit_refuses_before_output_or_model_creation(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(RuntimeError,'guarded GPU admission'):
                a.fit(fixture(d),Path(d)/'fit',0,end_monotonic=float('inf'))
            self.assertFalse((Path(d)/'fit').exists())
        self.assertFalse(torch.cuda.is_initialized())


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';torch.set_num_threads(1)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(RoutedModelChecks))
    if not result.wasSuccessful():raise SystemExit(1)
    output=ROOT/'analysis/routed_model_unit_CPU_qualification.json';assert not output.exists()
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    proof={'status':'ACTUAL_FINAL_ROUTED_COUNT_ADAPTER_NATIVE_CPU_UNIT_CHECKS_PASSED',
           'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'tests_passed':result.testsRun,
           'source_SHA256':{str(p):sha(p) for p in [Path(__file__).resolve(),*(ROOT/'routed_model').glob('*.py')]},
           'GPU_context_created':False,'real_data_accessed':False,'quality_measured':False,
           'scope':'Synthetic final-adapter tests: exact native recipe/seed prefix, disjoint samplers and fit-only moments, full unsigned64bit context seeds, integer forward/exact original transformed-rate gradient, candidate mass/codes, protected-target optimizer isolation, complete dynamic-sampler native probe/checkpoint replay and refusals. No fullwidth final-adapter/native runtime or quality claim.'}
    output.write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof,indent=2))


if __name__=='__main__':main()
