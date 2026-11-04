"""Synthetic native-loop checks for an explicit inactive ALZ loss extension."""
import ast,copy,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
import cell_ed_adapter as a
import cell_ed_update as u
import variant_pg_adapter as base
import variant_pg_update as base_u
from particlegan import get_recipe
from pg_model import critic_features
from stage_law import stages
from test_variant import fixture,same
CFG={'z_dim':8,'stage_balance':False,'cell_z_dim':8,'theta_init_scale':1.,'fourier_feature_scale':1.}

class TaskLoop(unittest.TestCase):
    def context(self,path):
        ctx=fixture(path);ctx.__class__=a.Context
        ctx.stage_counts=torch.as_tensor(np.bincount(stages(ctx.fitc),minlength=4),dtype=torch.long)
        return ctx

    def test_same_initial_native_models_factories_and_fitting_draws(self):
        base_u.configure(CFG)
        with tempfile.TemporaryDirectory() as d:
            ctx=self.context(Path(d));b,*_=a.build(ctx.mu,ctx.theta,'cpu',0,0)
            rb,*_=base.build(ctx.mu,ctx.theta,'cpu',0,0)
            same(b[0].to_dict(),get_recipe(total_steps=20000,batch_size=32,num_particles=512,z_dim=8).to_dict())
            for i in (1,2,3):same(b[i].state_dict(),rb[i].state_dict())
            for i in range(4,10):self.assertIs(type(b[i]),type(rb[i]))
            r=np.random.default_rng(42);rr=np.random.default_rng(42)
            pcs,c,lib=ctx.batch(r);features,bc,blib=base.Context.batch(ctx,rr)
            same(u.bag_features(pcs),features);same(c,bc);same(lib,blib);same(r.bit_generator.state,rr.bit_generator.state)
            self.assertEqual(ctx.stage_counts.tolist(),[1,1,1,3])
            expected=torch.where(c[:,3]==1.,torch.tensor(.5),torch.tensor(1.5))
            same(ctx.cell_energy_weights(c),expected)

    def test_auxiliary_alone_reaches_generator_dispersion_and_prior(self):
        with tempfile.TemporaryDirectory() as d:
            ctx=self.context(Path(d));b,g,r,usage,seeds=a.build(ctx.mu,ctx.theta,'cpu',0,0)
            real,c,lib=ctx.batch(r);z,_=b[3].sample(32,generator=b[-1].latent_generator)
            rate=b[1].rates(z,c,128,lib,g['cell']);counts=torch.poisson(rate.detach(),generator=g['count'])
            fake=critic_features(counts,rate,ctx.mean,ctx.sd)@ctx.W
            objective=u.pooled_cell_energy(real[:,:4],fake[:,:4],ctx.cell_energy_weights(c))
            grads=torch.autograd.grad(objective,[b[1].out[-1].weight,b[1].log_theta,b[3].z])
            for grad in grads:self.assertTrue(torch.isfinite(grad).all());self.assertGreater(float(grad.norm()),0)

    def test_first_critic_update_and_all_draw_streams_match_baseline(self):
        base_u.configure(CFG)
        with tempfile.TemporaryDirectory() as d:
            ctx=self.context(Path(d));b,g,r,usage,seeds=base.build(ctx.mu,ctx.theta,'cpu',0,0)
            original_ctx=copy.copy(ctx);original_ctx.__class__=base.Context
            base.advance(original_ctx,b,g,r,usage,execution_limit=20000)
            expected_D=copy.deepcopy(b[2].state_dict());expected_od=copy.deepcopy(b[5].state_dict())
            expected_G=copy.deepcopy(b[1].state_dict());expected_global=torch.get_rng_state().clone()
            expected_streams={name:getattr(b[-1],name).get_state().clone() for name in b[-1]._STREAMS}
            expected_cell={name:gen.get_state().clone() for name,gen in g.items()};expected_np=copy.deepcopy(r.bit_generator.state)
            b,g,r,usage,seeds=a.build(ctx.mu,ctx.theta,'cpu',0,0)
            metrics=a.advance(ctx,b,g,r,usage,execution_limit=20000)
            same(expected_D,b[2].state_dict());same(expected_od,b[5].state_dict());same(expected_global,torch.get_rng_state())
            for name,state in expected_streams.items():same(state,getattr(b[-1],name).get_state())
            for name,state in expected_cell.items():same(state,g[name].get_state())
            same(expected_np,r.bit_generator.state)
            self.assertTrue(any(not torch.equal(value,b[1].state_dict()[name]) for name,value in expected_G.items()))
            self.assertTrue(np.isfinite(metrics['stage_balanced_pooled_cell_energy_auxiliary']))

    def test_complete_serialized_update_replay_and_task_population_refusals(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d);(path/'manifest.json').write_text('{}');ctx=self.context(path)
            b,g,r,usage,seeds=a.build(ctx.mu,ctx.theta,'cpu',0,0)
            metrics=a.advance(ctx,b,g,r,usage,execution_limit=20000)
            for key in ('D_total','G_total','G_gradient_norm','prior_gradient_norm'):self.assertTrue(np.isfinite(metrics[key]))
            q=a.snapshot(ctx,b,g,r,usage,0,seeds,'synthetic');torch.save(q,path/'q.pt');q=torch.load(path/'q.pt',weights_only=False,map_location='cpu')
            expected=a.advance(ctx,b,g,r,usage,execution_limit=20000);state=a.snapshot(ctx,b,g,r,usage,0,seeds,'synthetic')
            rb,rg,rr,ru,rs=a.restore(ctx,q);same(expected,a.advance(ctx,rb,rg,rr,ru,execution_limit=20000));same(state,a.snapshot(ctx,rb,rg,rr,ru,0,rs,'synthetic'))
            for key,value in (('task_objective','published_unmodified'),('native_recipe','ka2'),('cell_energy_config',{**u.CELL_ENERGY_CONFIG,'coefficient':2.}),('fitting_stage_donor_counts',[2,1,1,2]),('cache_manifest_SHA256','0'*64)):
                bad=copy.deepcopy(q);bad[key]=value
                with self.assertRaises(ValueError):a.restore(ctx,bad)

    def test_sampling_body_only_removes_feature_projection_and_fit_preserved(self):
        def fun(path,name):return next(n for n in ast.parse(Path(path).read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
        for name in ('training_streams','build','fit'):self.assertEqual(ast.dump(fun(a.__file__,name),include_attributes=False),ast.dump(fun(base.__file__,name),include_attributes=False))
        def batch(path):
            cls=next(n for n in ast.parse(Path(path).read_text()).body if isinstance(n,ast.ClassDef) and n.name=='Context')
            return copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='batch'))
        expected=batch(base.__file__);actual=batch(a.__file__)
        self.assertIsInstance(expected.body[-1].value.elts[0],ast.Call)
        expected.body[-1].value.elts[0]=expected.body[-1].value.elts[0].args[0]
        self.assertEqual(ast.dump(expected,include_attributes=False),ast.dump(actual,include_attributes=False))

    def test_unregistered_architecture_and_native_fitting_on_CPU_refused(self):
        for change in ({'z_dim':2},{'stage_balance':True},{'cell_z_dim':32},{'fourier_feature_scale':5.}):
            with self.assertRaises(ValueError):u.configure({**CFG,**change})
        with tempfile.TemporaryDirectory() as d:
            ctx=self.context(Path(d))
            with self.assertRaises(RuntimeError):a.fit(ctx,Path(d)/'fit',0,end_monotonic=0)
            invalid=torch.zeros(32,7);invalid[:,3]=.17
            with self.assertRaises(ValueError):ctx.cell_energy_weights(invalid)
        self.assertFalse(torch.cuda.is_initialized())

if __name__=='__main__':unittest.main()
