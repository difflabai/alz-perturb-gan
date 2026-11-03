"""CPU qualification of learned cell observations and exact native state recovery."""
import ast,copy,tempfile,unittest
from pathlib import Path
import numpy as np,torch
import learned_pg_adapter as adapter
import learned_pg_update as update
import variant_pg_adapter as original
import variant_pg_update as old_update
from learned_cell_observation import OPTIONS,make_critic,observations
from critic import moment_features
from particlegan import get_recipe
from test_variant import fixture,same

CFG={'z_dim':8,'stage_balance':False,'cell_z_dim':8,'theta_init_scale':1.,'fourier_feature_scale':1.}
ON={**OPTIONS,'enabled':True};OFF=dict(OPTIONS)

class LearnedObservation(unittest.TestCase):
 def setUp(self):update.configure(CFG);old_update.configure(CFG);update.configure_observation(ON)
 def tearDown(self):update.configure_observation(OFF);self.assertFalse(torch.cuda.is_initialized())
 def context(self,cache):
  ctx=fixture(cache);ctx.__class__=adapter.Context;return ctx
 def test_features_independent_math_fixed_moments_and_permutations(self):
  torch.manual_seed(391);d=make_critic(ON);x=torch.randn(3,11,50,requires_grad=True)
  y=d.features(x);cell=d.cell_encoder(x.reshape(-1,50)).reshape(3,11,25)
  independent=torch.cat([moment_features(x),torch.cat([cell.mean(1),(cell.var(1,unbiased=False)+1e-6).sqrt()],1)*.2],1)
  self.assertTrue(torch.equal(y[:,:100],moment_features(x)));torch.testing.assert_close(y,independent,rtol=1e-6,atol=1e-7)
  self.assertTrue(torch.equal(y,d.features(x.flip(1))));self.assertTrue(torch.equal(y.flip(0),d.features(x.flip(0))))
  self.assertLessEqual(float(y[:,100:].detach().square().sum(1).max()),1.0001)
  y.sum().backward();self.assertTrue(torch.isfinite(x.grad).all());self.assertGreater(float(x.grad.abs().sum()),0.)
  const=torch.ones(2,8,50,requires_grad=True);d.features(const).sum().backward();self.assertTrue(torch.isfinite(const.grad).all())
 def test_same_initial_generator_prior_and_original_head(self):
  with tempfile.TemporaryDirectory() as directory:
   ctx=self.context(Path(directory));b,*_=adapter.build(ctx.mu,ctx.theta,'cpu',0,0);old,*_=original.build(ctx.mu,ctx.theta,'cpu',0,0)
   self.assertEqual(b[0].to_dict(),get_recipe(total_steps=20000,batch_size=32,num_particles=512,z_dim=8).to_dict())
   same(b[1].state_dict(),old[1].state_dict());same(b[3].state_dict(),old[3].state_dict());same(b[2].head.state_dict(),old[2].state_dict())
   self.assertEqual(sum(p.numel() for p in b[2].cell_encoder.parameters()),4889)
 def test_real_and_fake_encoder_gradients_but_penalty_only_regularizes_head(self):
  with tempfile.TemporaryDirectory() as directory:
   ctx=self.context(Path(directory));b,*_=adapter.build(ctx.mu,ctx.theta,'cpu',0,0);d=b[2];params=tuple(d.cell_encoder.parameters());c=torch.zeros(32,7)
   real=torch.randn(32,17,50);fake=torch.randn(32,17,50)+.4
   for x in [real,fake]:
    grads=torch.autograd.grad(d(d.features(x),c).mean(),params)
    self.assertTrue(all(torch.isfinite(g).all() for g in grads));self.assertGreater(sum(float(g.abs().sum()) for g in grads),0)
   seen=[];hook=d.register_forward_pre_hook(lambda module,args:seen.append(tuple(args[0].shape)))
   b[6].collect_stats=True;pen=b[6](d,d.features(real),d.features(fake),c)
   grads=torch.autograd.grad(pen,params,allow_unused=True);self.assertTrue(all(g is None for g in grads))
   hook.remove();self.assertTrue(seen and all(shape==(32,150) for shape in seen));self.assertTrue(torch.isfinite(pen))
 def test_frozen_encoder_preserves_finite_generator_dispersion_and_prior_gradients(self):
  from pg_model import critic_features
  with tempfile.TemporaryDirectory() as directory:
   ctx=self.context(Path(directory));b,g,r,u,s=adapter.build(ctx.mu,ctx.theta,'cpu',0,0);rec,G,D,P,*_=b
   D.requires_grad_(False);_,c,lib=ctx.batch(r);z,idx=P.sample(32,generator=b[-1].latent_generator)
   rate=G.rates(z,c,128,lib,g['cell']);counts=torch.poisson(rate.detach(),generator=g['count']);pcs=critic_features(counts,rate,ctx.mean,ctx.sd)@ctx.W
   features=D.features(pcs);loss=b[7].g_loss(D(features,c));grads=torch.autograd.grad(loss,(G.out[-1].weight,G.log_theta,P.z),retain_graph=True)
   self.assertTrue(all(torch.isfinite(v).all() and v.abs().sum()>0 for v in grads))
   isolated=(features[:,100:]*torch.arange(1,51,dtype=features.dtype)).sum();grads=torch.autograd.grad(isolated,(G.out[-1].weight,G.log_theta,P.z))
   self.assertTrue(all(torch.isfinite(v).all() and v.abs().sum()>0 for v in grads));self.assertTrue(all(p.grad is None and not p.requires_grad for p in D.parameters()))
 def test_disabled_complete_state_and_next_update_match_PG8(self):
  update.configure_observation(OFF)
  with tempfile.TemporaryDirectory() as directory:
   cache=Path(directory);(cache/'manifest.json').write_text('{}');ctx=fixture(cache)
   b,g,r,u,s=original.build(ctx.mu,ctx.theta,'cpu',0,0);expected_m=original.advance(ctx,b,g,r,u,execution_limit=20000);expected=original.snapshot(ctx,b,g,r,u,0,s,'synthetic')
   ctx.__class__=adapter.Context;b,g,r,u,s=adapter.build(ctx.mu,ctx.theta,'cpu',0,0);observed_m=adapter.advance(ctx,b,g,r,u,execution_limit=20000);observed=adapter.snapshot(ctx,b,g,r,u,0,s,'synthetic');observed.pop('learned_observation_config')
   same(expected_m,observed_m);same(expected,observed)
 def test_enabled_complete_serialized_restore_next_update_and_mismatch_refusal(self):
  with tempfile.TemporaryDirectory() as directory:
   cache=Path(directory);(cache/'manifest.json').write_text('{}');ctx=self.context(cache)
   b,g,r,u,s=adapter.build(ctx.mu,ctx.theta,'cpu',0,0);adapter.advance(ctx,b,g,r,u,execution_limit=20000)
   q=adapter.snapshot(ctx,b,g,r,u,0,s,'synthetic');torch.save(q,cache/'q.pt');q=torch.load(cache/'q.pt',map_location='cpu',weights_only=False)
   expected_m=adapter.advance(ctx,b,g,r,u,execution_limit=20000);expected=adapter.snapshot(ctx,b,g,r,u,0,s,'synthetic')
   rb,rg,rr,ru,rs=adapter.restore(ctx,q);same(expected_m,adapter.advance(ctx,rb,rg,rr,ru,execution_limit=20000));same(expected,adapter.snapshot(ctx,rb,rg,rr,ru,0,rs,'synthetic'))
   update.configure_observation(OFF)
   with self.assertRaisesRegex(ValueError,'observation checkpoint configuration'):adapter.restore(ctx,q)
 def test_only_declared_observation_changes_in_native_update_AST(self):
  source=Path(old_update.__file__).read_text().replace('def fake(z,c,lib):','def fake_pcs(z,c,lib):').replace('return bag_features(critic_features(counts,rate,ctx.mean,ctx.sd)@ctx.W)','return critic_features(counts,rate,ctx.mean,ctx.sd)@ctx.W')
  source=source.replace('real,c,lib=ctx.batch(rng)','real_pcs,c,lib=ctx.batch(rng)\n        real=observations(D,real_pcs,OBSERVATION_CONFIG)')
  source=source.replace('with torch.no_grad():z,_=P.sample(32,generator=policy.latent_generator);f=fake(z,c,lib)','with torch.no_grad():z,_=P.sample(32,generator=policy.latent_generator);f_pcs=fake_pcs(z,c,lib)\n        f=observations(D,f_pcs,OBSERVATION_CONFIG)')
  source=source.replace('f=fake(z,c2,lib2);lg=loss.g_loss(D(f,c2))','f=observations(D,fake_pcs(z,c2,lib2),OBSERVATION_CONFIG);lg=loss.g_loss(D(f,c2))')
  def node(text):return ast.dump(next(q for q in ast.parse(text).body if isinstance(q,ast.FunctionDef) and q.name=='advance'),include_attributes=False)
  self.assertEqual(node(source),node(Path(update.__file__).read_text()))

if __name__=='__main__':unittest.main()
