"""Synthetic complete K3P update tests; no donor data or GPU context."""
import ast,copy,inspect,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from particlegan import get_recipe
from particlegan.k3p import K3PGeneratorAdam
from particlegan.k3p import K3PCriticAdam
import k3p_adapter as a
import k3p_update as u
import variant_pg_adapter as baseline
import variant_pg_update as base_update
from test_variant import fixture,same
CFG={'z_dim':8,'stage_balance':False,'cell_z_dim':8,'theta_init_scale':1.,'fourier_feature_scale':1.}

class K3P(unittest.TestCase):
    def context(self,path):
        ctx=fixture(path);ctx.__class__=a.Context;return ctx

    def test_published_recipe_components_initial_models_and_ema(self):
        with tempfile.TemporaryDirectory() as d:
            ctx=self.context(Path(d));b,*_=a.build(ctx.mu,ctx.theta,'cpu',0,0)
            base_update.configure(CFG);original,*_=baseline.build(ctx.mu,ctx.theta,'cpu',0,0)
            rec=b[0];self.assertEqual(rec.to_dict(),get_recipe('k3p',total_steps=20000,batch_size=32,num_particles=512,z_dim=8).to_dict())
            self.assertIsInstance(b[4],K3PGeneratorAdam);self.assertIsInstance(b[5],K3PCriticAdam)
            for i in (1,2):same(b[i].state_dict(),original[i].state_dict())
            same(b[3].z,original[3].z)
            same(b[2].state_dict(),b[5].ema_critic.state_dict())
            self.assertFalse(any(p.requires_grad for p in b[5].ema_critic.parameters()))
            self.assertEqual(rec.ema_decay,.995);self.assertEqual(rec.input_noise_std,.5)
            self.assertEqual(rec.output_noise_std,.029);self.assertEqual(rec.network_lr_horizon_cap,1600)
            self.assertTrue(rec.direct_particle_gain);self.assertEqual(b[-1].row_semantics,'conditional')

    def test_native_noise_conditional_ema_gradients_and_average_updates(self):
        with tempfile.TemporaryDirectory() as d:
            ctx=self.context(Path(d));b,g,r,usage,seeds=a.build(ctx.mu,ctx.theta,'cpu',0,0)
            policy=b[-1];old=copy.deepcopy(b[1].state_dict());old_noise=policy.noise_generator.get_state().clone()
            calls=[]
            handles=[module.register_forward_pre_hook(lambda mod,args:calls.append((mod is b[2],len(args),args[1].shape if len(args)>1 else None))) for module in (b[2],b[5].ema_critic)]
            metrics=a.advance(ctx,b,g,r,usage,execution_limit=20000)
            for handle in handles:handle.remove()
            for key in ('D_total','G_total','D_gradient_norm','G_gradient_norm','prior_gradient_norm'):self.assertTrue(np.isfinite(metrics[key]));self.assertGreater(metrics[key],0)
            self.assertEqual(metrics['step'],1);self.assertEqual(metrics['input_noise_sigma'],.5)
            self.assertFalse(torch.equal(old_noise,policy.noise_generator.get_state()))
            self.assertFalse(b[5].record.anchor_started,'native initial warmup must remain intact')
            self.assertTrue(all(n==2 and shape==torch.Size([32,7]) for _,n,shape in calls))
            for key,value in policy.ema_G.state_dict().items():
                current=b[1].state_dict()[key]
                expected=old[key]*.995+current*(1-.995)
                torch.testing.assert_close(value,expected,rtol=1e-6,atol=1e-7)
            self.assertEqual(usage.steps,1);self.assertEqual(policy._phase,'ready')
            second=a.advance(ctx,b,g,r,usage,execution_limit=20000)
            self.assertGreater(second['output_noise_sigma'],0)
            self.assertLess(second['input_noise_sigma'],.5)

    def test_full_serialized_next_update_bitwise_including_native_ema_noise(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d);(path/'manifest.json').write_text('{}');ctx=self.context(path)
            b,g,r,usage,seeds=a.build(ctx.mu,ctx.theta,'cpu',0,0)
            a.advance(ctx,b,g,r,usage,execution_limit=20000)
            q=a.snapshot(ctx,b,g,r,usage,0,seeds,'synthetic');torch.save(q,path/'q.pt');q=torch.load(path/'q.pt',weights_only=False,map_location='cpu')
            expected=a.advance(ctx,b,g,r,usage,execution_limit=20000);state=a.snapshot(ctx,b,g,r,usage,0,seeds,'synthetic')
            rb,rg,rr,ru,rs=a.restore(ctx,q)
            same(expected,a.advance(ctx,rb,rg,rr,ru,execution_limit=20000));same(state,a.snapshot(ctx,rb,rg,rr,ru,0,rs,'synthetic'))
            for key,value in (('native_recipe','bcap_adam'),('serving_contract',{'weights':'ema','output_noise':False}),('cache_manifest_SHA256','0'*64)):
                bad=copy.deepcopy(q);bad[key]=value
                with self.assertRaises(ValueError):a.restore(ctx,bad)
            bad=copy.deepcopy(q);bad['policy']['optimizers'][1]['regularizer']['ema']={}
            with self.assertRaises((ValueError,RuntimeError,KeyError)):a.restore(ctx,bad)

    def test_actual_native_warmup_transition_and_late_state_replay(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d);(path/'manifest.json').write_text('{}');ctx=self.context(path)
            b,g,r,usage,seeds=a.build(ctx.mu,ctx.theta,'cpu',0,0)
            for _ in range(1400):a.advance(ctx,b,g,r,usage,execution_limit=20000)
            self.assertEqual(b[-1].completed_steps,1400)
            self.assertTrue(b[5].record.anchor_started)
            self.assertEqual(b[5].record.observed_steps,1400)
            q=a.snapshot(ctx,b,g,r,usage,0,seeds,'synthetic')
            calls=[]
            hook=b[5].ema_critic.register_forward_pre_hook(lambda module,args:calls.append(args[1].shape))
            expected=a.advance(ctx,b,g,r,usage,execution_limit=20000)
            hook.remove()
            self.assertTrue(calls,'actual post-warmup penalty must call conditional EMA critic')
            self.assertTrue(all(shape==torch.Size([32,7]) for shape in calls))
            state=a.snapshot(ctx,b,g,r,usage,0,seeds,'synthetic')
            rb,rg,rr,ru,rs=a.restore(ctx,q)
            same(expected,a.advance(ctx,rb,rg,rr,ru,execution_limit=20000))
            same(state,a.snapshot(ctx,rb,rg,rr,ru,0,rs,'synthetic'))

    def test_unchanged_task_sampling_fit_and_feature_bodies(self):
        def function(path,name):return ast.dump(next(n for n in ast.parse(Path(path).read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)
        for name in ('training_streams','fit'):self.assertEqual(function(a.__file__,name),function(baseline.__file__,name))
        for name in ('bag_features','finite','gradnorm'):self.assertEqual(function(inspect.getsourcefile(getattr(u,name)),name),function(base_update.__file__,name))
        def method(path,name):
            cls=next(n for n in ast.parse(Path(path).read_text()).body if isinstance(n,ast.ClassDef) and n.name=='Context')
            return ast.dump(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)
        for name in ('__init__','project','batch'):self.assertEqual(method(a.__file__,name),method(baseline.__file__,name))
        with tempfile.TemporaryDirectory() as d:
            ctx=self.context(Path(d));same(ctx.batch(np.random.default_rng(71)),baseline.Context.batch(ctx,np.random.default_rng(71)))

    def test_unregistered_config_and_cpu_fit_refused(self):
        for change in ({'z_dim':2},{'stage_balance':True},{'cell_z_dim':32},{'fourier_feature_scale':5.}):
            with self.assertRaises(ValueError):u.configure({**CFG,**change})
        with tempfile.TemporaryDirectory() as d:
            ctx=self.context(Path(d))
            with self.assertRaises(RuntimeError):a.fit(ctx,Path(d)/'fit',0,end_monotonic=0)
        self.assertFalse(torch.cuda.is_initialized())

if __name__=='__main__':unittest.main()
