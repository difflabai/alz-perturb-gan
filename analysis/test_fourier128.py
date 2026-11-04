"""Synthetic fixed observation/native update/replay qualification; no donor data."""
import ast,copy,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
import fourier128_adapter as a
import fourier128_update as u
import fourier128_observation as obs
import variant_pg_adapter as baseline
import variant_pg_update as base_update
from critic import bag_features as base_features
from pg_model import critic_features,CountGenerator
from particlegan import get_recipe
from test_variant import fixture,same
CFG={'z_dim':8,'stage_balance':False,'cell_z_dim':8,'theta_init_scale':1.,'fourier_feature_scale':1.}

class Fourier128(unittest.TestCase):
    def context(self,path):
        ctx=fixture(path);ctx.__class__=a.Context;return ctx
    def test_exact_moments_permutation_invariance_and_unit_fourier_bound(self):
        x=torch.randn(7,129,50,generator=torch.Generator().manual_seed(77))
        y=obs.bag_features(x);self.assertEqual(y.shape,(7,356));self.assertTrue(torch.isfinite(y).all())
        same(y[:,:100],base_features(x)[:,:100])
        same(y,obs.bag_features(x[:,torch.randperm(129,generator=torch.Generator().manual_seed(78))]))
        self.assertTrue((y[:,100:].norm(dim=1)<=1+1e-6).all());self.assertFalse(obs.FREQUENCIES.requires_grad)
        expected=torch.randn(128,50,generator=torch.Generator().manual_seed(20261030),dtype=torch.float64)/np.sqrt(50.)
        same(obs.FREQUENCIES,expected)
    def test_exact_native_recipe_components_initial_G_prior_and_declared_critic(self):
        with tempfile.TemporaryDirectory() as d:
            ctx=self.context(Path(d));bundle,*_=a.build(ctx.mu,ctx.theta,'cpu',0,0)
            base_update.configure(CFG);original,*_=baseline.build(ctx.mu,ctx.theta,'cpu',0,0)
            self.assertEqual(bundle[0].to_dict(),get_recipe(total_steps=20000,batch_size=32,num_particles=512,z_dim=8).to_dict())
            for index in (4,5,6,7,8,9):self.assertIs(type(bundle[index]),type(original[index]))
            for index in (1,3):same(bundle[index].state_dict(),original[index].state_dict())
            self.assertEqual(bundle[2].phi[0].in_features,363);self.assertEqual(original[2].phi[0].in_features,157)
            rec=bundle[0]
            models=[CountGenerator(torch.ones(2000),torch.ones(2000),zd=8,zc=8),obs.VectorCritic(),rec.make_prior(generator=torch.Generator().manual_seed(0))]
            rec.apply_prior_policy(models[-1])
            count=sum(v.numel() for model in models for v in model.state_dict().values() if v.is_floating_point())
            self.assertEqual(count,397545)
    def test_update_observation_wrapper_input_adapter_and_fit_AST_unchanged(self):
        def function(path,name):return ast.dump(next(n for n in ast.parse(Path(path).read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)
        for name in ('advance','finite','gradnorm','bag_features','recipe'):self.assertEqual(function(u.__file__,name),function(base_update.__file__,name))
        for name in ('training_streams','fit'):self.assertEqual(function(a.__file__,name),function(baseline.__file__,name))
        def method(path,name):
            cls=next(n for n in ast.parse(Path(path).read_text()).body if isinstance(n,ast.ClassDef) and n.name=='Context')
            return ast.dump(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)
        for name in ('__init__','project','batch'):self.assertEqual(method(a.__file__,name),method(baseline.__file__,name))
    def test_isolated_new_fourier_block_reaches_G_dispersion_and_prior(self):
        with tempfile.TemporaryDirectory() as d:
            ctx=self.context(Path(d));b,g,r,usage,streams=a.build(ctx.mu,ctx.theta,'cpu',0,0)
            real,c,lib=ctx.batch(r);z,_=b[3].sample(32,generator=b[-1].latent_generator)
            rate=b[1].rates(z,c,128,lib,g['cell']);counts=torch.poisson(rate.detach(),generator=g['count'])
            pcs=critic_features(counts,rate,ctx.mean,ctx.sd)@ctx.W;features=obs.bag_features(pcs)
            weights=torch.linspace(-1,1,256);objective=(features[:,100:]*weights).sum()
            grads=torch.autograd.grad(objective,[b[1].out[-1].weight,b[1].log_theta,b[3].z])
            for grad in grads:self.assertTrue(torch.isfinite(grad).all());self.assertGreater(float(grad.norm()),0)
            expected=baseline.Context.batch(ctx,np.random.default_rng(101));actual=ctx.batch(np.random.default_rng(101))
            same(actual[0][:,:100],expected[0][:,:100]);same(actual[1:],expected[1:])
    def test_full_saved_state_next_update_bitwise_and_source_config_refusals(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d);(path/'manifest.json').write_text('{}');ctx=self.context(path)
            b,g,r,usage,streams=a.build(ctx.mu,ctx.theta,'cpu',0,0)
            metrics=a.advance(ctx,b,g,r,usage,execution_limit=20000)
            for key in ('D_total','G_total','D_gradient_norm','G_gradient_norm','prior_gradient_norm'):self.assertTrue(np.isfinite(metrics[key]));self.assertGreater(metrics[key],0)
            q=a.snapshot(ctx,b,g,r,usage,0,streams,'synthetic');torch.save(q,path/'q.pt');q=torch.load(path/'q.pt',map_location='cpu',weights_only=False)
            expected=a.advance(ctx,b,g,r,usage,execution_limit=20000);state=a.snapshot(ctx,b,g,r,usage,0,streams,'synthetic')
            rb,rg,rr,ru,rs=a.restore(ctx,q);same(expected,a.advance(ctx,rb,rg,rr,ru,execution_limit=20000));same(state,a.snapshot(ctx,rb,rg,rr,ru,0,rs,'synthetic'))
            for key,value in (('fourier_observation_config',{**obs.OPTIONS,'frequencies':25}),('native_recipe','bcap_adam'),('cache_manifest_SHA256','0'*64)):
                bad=copy.deepcopy(q);bad[key]=value
                with self.assertRaises(ValueError):a.restore(ctx,bad)
    def test_unregistered_architecture_feature_configuration_and_GPU_refused(self):
        for change in ({'z_dim':2},{'stage_balance':True},{'cell_z_dim':32},{'fourier_feature_scale':5.}):
            with self.assertRaises(ValueError):u.configure({**CFG,**change})
        for change in ({'frequencies':25},{'frequency_seed':20261031},{'features':150},{'hidden_width':256}):
            with self.assertRaises(ValueError):obs.normalized_options({**obs.OPTIONS,**change})
        u.configure(CFG);self.assertFalse(torch.cuda.is_initialized())

if __name__=='__main__':unittest.main()
