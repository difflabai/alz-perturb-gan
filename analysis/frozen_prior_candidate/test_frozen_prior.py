"""Actual synthetic CPU qualification of an inactive native fixed-particle control."""
import ast,copy,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from particlegan import get_recipe
import frozen_pg_adapter as a
import frozen_pg_update as u
import variant_pg_adapter as base
import variant_pg_update as base_u
from test_variant import fixture,same
CFG={'z_dim':8,'stage_balance':False,'cell_z_dim':8,'theta_init_scale':1.,'fourier_feature_scale':1.}

class Frozen(unittest.TestCase):
    def setUp(self):base_u.configure(CFG);u.configure(CFG)
    def context(self,path):
        ctx=fixture(path);ctx.__class__=a.Context;return ctx
    def test_public_recipe_initial_models_streams_and_frozen_factory_ownership(self):
        with tempfile.TemporaryDirectory() as d:
            ctx=self.context(Path(d));b,*_=a.build(ctx.mu,ctx.theta,'cpu',0,0);rb,*_=base.build(ctx.mu,ctx.theta,'cpu',0,0)
            same(b[0].to_dict(),get_recipe(total_steps=20000,batch_size=32,num_particles=512,z_dim=8,prior_update='frozen').to_dict())
            before=rb[0].to_dict();after=b[0].to_dict();self.assertEqual({k for k in set(before)|set(after) if before.get(k)!=after.get(k)},{'prior_update'})
            for i in (1,2):same(b[i].state_dict(),rb[i].state_dict())
            same(b[3].z,rb[3].z);self.assertEqual(a.training_streams(0,0),base.training_streams(0,0))
            self.assertFalse(b[3].z.requires_grad);self.assertEqual(list(b[3].parameters()),[])
            self.assertEqual(list(b[3].named_buffers())[0][0],'z');self.assertEqual(len(b[4].param_groups),1)
            for group in b[4].param_groups:self.assertFalse(any(p is b[3].z for p in group['params']))
            for i in range(4,10):self.assertIs(type(b[i]),type(rb[i]))
    def test_common_initial_first_G_D_update_and_draw_stream_parity(self):
        with tempfile.TemporaryDirectory() as d:
            ctx=self.context(Path(d));b,g,r,usage,_=a.build(ctx.mu,ctx.theta,'cpu',0,0);state=torch.get_rng_state().clone()
            rb,rg,rr,ru,_=base.build(ctx.mu,ctx.theta,'cpu',0,0);same(state,torch.get_rng_state())
            initial=b[3].z.clone();torch.set_rng_state(state);expected=base.advance(ctx,rb,rg,rr,ru,execution_limit=20000)
            torch.set_rng_state(state);actual=a.advance(ctx,b,g,r,usage,execution_limit=20000)
            for key in ('D_adversarial','D_penalty','D_total','G_adversarial','D_gradient_norm','G_gradient_norm'):same(actual[key],expected[key])
            for i in (1,2):same(b[i].state_dict(),rb[i].state_dict())
            for name in g:same(g[name].get_state(),rg[name].get_state())
            same(r.bit_generator.state,rr.bit_generator.state);same(b[3].z,initial)
            self.assertFalse(torch.equal(rb[3].z,initial));self.assertEqual(actual['prior_regularization_unweighted'],0.)
    def test_repeated_native_updates_train_networks_dispersion_but_never_bank(self):
        with tempfile.TemporaryDirectory() as d:
            ctx=self.context(Path(d));b,g,r,usage,_=a.build(ctx.mu,ctx.theta,'cpu',0,0)
            z=b[3].z.clone();theta=b[1].log_theta.detach().clone();gw=b[1].out[-1].weight.detach().clone();dw=copy.deepcopy(b[2].state_dict())
            for _ in range(8):
                row=a.advance(ctx,b,g,r,usage,execution_limit=20000);same(z,b[3].z);self.assertIsNone(b[3].z.grad);self.assertEqual(row['prior_regularization_unweighted'],0.)
            self.assertFalse(torch.equal(theta,b[1].log_theta));self.assertFalse(torch.equal(gw,b[1].out[-1].weight));self.assertTrue(any(not torch.equal(v,b[2].state_dict()[k]) for k,v in dw.items()))
            self.assertEqual(int(usage.moved_steps.sum()),0);self.assertEqual(int(usage.unsampled_moved_steps.sum()),0);self.assertEqual(usage.steps,8)
    def test_complete_serialized_next_update_bitwise(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'manifest.json').write_text('{}');ctx=self.context(p);b,g,r,usage,seeds=a.build(ctx.mu,ctx.theta,'cpu',0,0)
            a.advance(ctx,b,g,r,usage,execution_limit=20000);q=a.snapshot(ctx,b,g,r,usage,0,seeds,'synthetic');torch.save(q,p/'q.pt');q=torch.load(p/'q.pt',weights_only=False,map_location='cpu')
            expected=a.advance(ctx,b,g,r,usage,execution_limit=20000);state=a.snapshot(ctx,b,g,r,usage,0,seeds,'synthetic')
            rb,rg,rr,ru,rs=a.restore(ctx,q);same(expected,a.advance(ctx,rb,rg,rr,ru,execution_limit=20000));same(state,a.snapshot(ctx,rb,rg,rr,ru,0,rs,'synthetic'))
    def test_corrupt_frozen_bank_recipe_cache_and_config_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'manifest.json').write_text('{}');ctx=self.context(p);b,g,r,usage,seeds=a.build(ctx.mu,ctx.theta,'cpu',0,0);q=a.snapshot(ctx,b,g,r,usage,0,seeds,'synthetic')
            for key,value in [('prior_update','learned'),('native_recipe','mog'),('cache_manifest_SHA256','0'*64)]:
                bad=copy.deepcopy(q);bad[key]=value
                with self.assertRaises(ValueError):a.restore(ctx,bad)
            bad=copy.deepcopy(q);bad['policy']['models']['prior']['z'][0,0]+=1
            with self.assertRaises(ValueError):a.restore(ctx,bad)
            bad=copy.deepcopy(q);bad['recipe']['prior_update']='learned'
            with self.assertRaises(ValueError):a.restore(ctx,bad)
        for change in ({'z_dim':2},{'stage_balance':True},{'cell_z_dim':32},{'fourier_feature_scale':5.}):
            with self.assertRaises(ValueError):u.configure({**CFG,**change})
    def test_original_Context_fit_streams_and_CPU_fitting_refusal(self):
        def node(path,name):return next(n for n in ast.parse(Path(path).read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
        for name in ('Context','fit','training_streams'):self.assertEqual(ast.dump(node(a.__file__,name),include_attributes=False),ast.dump(node(base.__file__,name),include_attributes=False))
        with tempfile.TemporaryDirectory() as d:
            ctx=self.context(Path(d))
            with self.assertRaises(RuntimeError):a.fit(ctx,Path(d)/'fit',0,end_monotonic=0)
        self.assertFalse(torch.cuda.is_initialized())

if __name__=='__main__':unittest.main()
