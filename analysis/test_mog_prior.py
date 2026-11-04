"""Synthetic CPU qualification of the native MoG prior count-model family."""
import ast,copy,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from particlegan import get_recipe
from particlegan.particle_prior import MoGParticlePrior,calibrate_mog_sigma
import mog_pg_adapter as a
import mog_pg_update as u
import variant_pg_adapter as base
import variant_pg_update as base_u
from test_variant import fixture,same
CFG={'z_dim':8,'stage_balance':False,'cell_z_dim':8,'theta_init_scale':1.,'fourier_feature_scale':1.}

class MoG(unittest.TestCase):
    def context(self,path):
        ctx=fixture(path);ctx.__class__=a.Context;return ctx

    def test_exact_public_recipe_components_initial_models_and_old_streams(self):
        base_u.configure(CFG)
        with tempfile.TemporaryDirectory() as d:
            ctx=self.context(Path(d));b,*_=a.build(ctx.mu,ctx.theta,'cpu',0,0)
            rb,*_=base.build(ctx.mu,ctx.theta,'cpu',0,0)
            same(b[0].to_dict(),get_recipe('mog',total_steps=20000,batch_size=32,num_particles=512,z_dim=8).to_dict())
            before=rb[0].to_dict();after=b[0].to_dict()
            self.assertEqual({k for k in set(before)|set(after) if before.get(k)!=after.get(k)},{'name','prior_kind','sigma_rel'})
            self.assertIs(type(b[3]),MoGParticlePrior)
            for i in (1,2):same(b[i].state_dict(),rb[i].state_dict())
            same(b[3].z,rb[3].z)
            for i in range(4,10):self.assertIs(type(b[i]),type(rb[i]))
            old=base.training_streams(0,0);new=a.training_streams(0,0)
            self.assertEqual({name:new[name] for name in old},old);self.assertEqual(set(new)-set(old),{'prior_noise'})
            self.assertEqual(b[-1].row_semantics,'conditional')

    def test_native_kernel_calibration_independent_noise_and_explicit_eps(self):
        with tempfile.TemporaryDirectory() as d:
            ctx=self.context(Path(d));b,*_=a.build(ctx.mu,ctx.theta,'cpu',0,0);P=b[3]
            sigma,d0=calibrate_mog_sigma(P.means(),.025)
            same(P.sigma,sigma);same(P.d0,d0);self.assertGreater(float(sigma),0)
            idxgen=torch.Generator().manual_seed(88);noise=torch.Generator().manual_seed(89)
            idxref=torch.Generator().manual_seed(88);noiseref=torch.Generator().manual_seed(89)
            latent,indices=P.sample(32,generator=idxgen,noise_generator=noise)
            expected_indices=torch.randint(0,512,(32,),generator=idxref)
            eps=torch.randn(32,8,generator=noiseref)
            same(indices,expected_indices);same(latent,P.means()[indices]+P.sigma*eps)
            same(latent,P(indices,eps=eps));same(idxgen.get_state(),idxref.get_state());same(noise.get_state(),noiseref.get_state())
            self.assertEqual(P.get_extra_state(),{'sigma_rel':.025,'standardize':True})

    def test_complete_native_update_gradients_fixed_kernel_and_row_ownership(self):
        with tempfile.TemporaryDirectory() as d:
            ctx=self.context(Path(d));b,g,r,usage,seeds=a.build(ctx.mu,ctx.theta,'cpu',0,0)
            sigma=b[3].sigma.clone();d0=b[3].d0.clone();state=g['prior_noise'].get_state().clone()
            metrics=a.advance(ctx,b,g,r,usage,execution_limit=20000)
            for key in ('D_total','G_total','G_gradient_norm','prior_gradient_norm'):self.assertTrue(np.isfinite(metrics[key]));self.assertGreater(metrics[key],0)
            same(sigma,b[3].sigma);same(d0,b[3].d0)
            self.assertFalse(torch.equal(state,g['prior_noise'].get_state()));self.assertEqual(usage.steps,1)
            self.assertEqual(int(usage.unsampled_moved_steps.sum()),0)

    def test_full_serialized_next_update_bitwise_and_kernel_noise_refusals(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d);(path/'manifest.json').write_text('{}');ctx=self.context(path)
            b,g,r,usage,seeds=a.build(ctx.mu,ctx.theta,'cpu',0,0)
            a.advance(ctx,b,g,r,usage,execution_limit=20000)
            q=a.snapshot(ctx,b,g,r,usage,0,seeds,'synthetic');torch.save(q,path/'q.pt');q=torch.load(path/'q.pt',weights_only=False,map_location='cpu')
            expected=a.advance(ctx,b,g,r,usage,execution_limit=20000);state=a.snapshot(ctx,b,g,r,usage,0,seeds,'synthetic')
            rb,rg,rr,ru,rs=a.restore(ctx,q);same(expected,a.advance(ctx,rb,rg,rr,ru,execution_limit=20000));same(state,a.snapshot(ctx,rb,rg,rr,ru,0,rs,'synthetic'))
            for key,value in (('native_recipe','bispf_optimized'),('mog_prior_config',{'prior_kind':'mog','sigma_rel':.05,'standardize':True}),('initial_kernel',{'sigma':0.,'d0':0.}),('cache_manifest_SHA256','0'*64)):
                bad=copy.deepcopy(q);bad[key]=value
                with self.assertRaises(ValueError):a.restore(ctx,bad)
            bad=copy.deepcopy(q);bad['generators'].pop('prior_noise')
            with self.assertRaises(ValueError):a.restore(ctx,bad)

    def test_only_two_sampling_noise_keywords_change_update_AST(self):
        def fun(path,name):return copy.deepcopy(next(n for n in ast.parse(Path(path).read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name))
        actual=fun(u.__file__,'advance');expected=fun(base_u.__file__,'advance');removed=0
        for call in ast.walk(actual):
            if isinstance(call,ast.Call):
                previous=len(call.keywords);call.keywords=[k for k in call.keywords if k.arg!='noise_generator'];removed+=previous-len(call.keywords)
        self.assertEqual(removed,2);self.assertEqual(ast.dump(actual,include_attributes=False),ast.dump(expected,include_attributes=False))
        for name in ('fit','training_streams'):self.assertEqual(ast.dump(fun(a.__file__,name),include_attributes=False),ast.dump(fun(base.__file__,name),include_attributes=False))
        def context(path):return next(n for n in ast.parse(Path(path).read_text()).body if isinstance(n,ast.ClassDef) and n.name=='Context')
        self.assertEqual(ast.dump(context(a.__file__),include_attributes=False),ast.dump(context(base.__file__),include_attributes=False))

    def test_unregistered_task_config_and_CPU_fitting_refused(self):
        for change in ({'z_dim':2},{'stage_balance':True},{'cell_z_dim':32},{'fourier_feature_scale':5.}):
            with self.assertRaises(ValueError):u.configure({**CFG,**change})
        with tempfile.TemporaryDirectory() as d:
            ctx=self.context(Path(d))
            with self.assertRaises(RuntimeError):a.fit(ctx,Path(d)/'fit',0,end_monotonic=0)
        self.assertFalse(torch.cuda.is_initialized())

if __name__=='__main__':unittest.main()
