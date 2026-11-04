"""Qualify native BCAP/Adam components and complete count-model continuation on CPU."""
import ast,copy,os,tempfile,unittest
from pathlib import Path
import numpy as np,torch
import bcap_adam_adapter as adapter
import bcap_adam_update as update
import variant_pg_adapter as variant
import variant_pg_update as vu
from particlegan import get_recipe
from torch.optim import Adam
from particlegan.k3p import CriticPenalty
from particlegan.gan_loss import GANLoss
from particlegan.vicreg_loss import ParticleRegularizer
from test_variant import fixture,same
CFG={'z_dim':8,'stage_balance':False,'cell_z_dim':8,'theta_init_scale':1.,'fourier_feature_scale':1.}

class NativeBCAPAdam(unittest.TestCase):
 def context(self,cache):
  ctx=fixture(cache);ctx.__class__=adapter.Context;return ctx
 def test_exact_official_components_and_matching_initial_models(self):
  with tempfile.TemporaryDirectory() as d:
   ctx=self.context(Path(d));b,*_=adapter.build(ctx.mu,ctx.theta,'cpu',0,0)
   self.assertEqual(b[0].to_dict(),get_recipe('bcap_adam',total_steps=20000,batch_size=32,num_particles=512,z_dim=8).to_dict())
   for obj,kind in [(b[4],Adam),(b[5],Adam),(b[6],CriticPenalty),(b[7],GANLoss),(b[8],ParticleRegularizer)]:self.assertIsInstance(obj,kind)
   vu.configure(CFG);vb,*_=variant.build(ctx.mu,ctx.theta,'cpu',0,0)
   for i in (1,2):same(b[i].state_dict(),vb[i].state_dict())
   same(dict(b[3].named_parameters()),dict(vb[3].named_parameters()))
   self.assertEqual(set(b[3].state_dict()),{'z'})
   self.assertEqual(set(vb[3].state_dict()),{'z','moment_tether_mean','moment_tether_covariance','moment_tether_count'})
   self.assertEqual(b[0].prior_reg,0.)
 def test_only_native_paired_generator_loss_argument_changes_advance_AST(self):
  def node(path,name):return ast.dump(next(n for n in ast.parse(Path(path).read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)
  for name in ['finite','gradnorm','bag_features']:self.assertEqual(node(update.__file__,name),node(vu.__file__,name))
  actual=next(n for n in ast.parse(Path(update.__file__).read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='advance')
  calls=[n for n in ast.walk(actual) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='g_loss']
  self.assertEqual(len(calls),1);self.assertEqual(len(calls[0].args),2)
  self.assertEqual(ast.dump(calls[0].args[1],include_attributes=False),ast.dump(ast.parse('D(real2,c2).detach()',mode='eval').body,include_attributes=False))
  calls[0].args.pop()
  self.assertEqual(ast.dump(actual,include_attributes=False),node(vu.__file__,'advance'))
  for name in ['training_streams','fit']:self.assertEqual(node(adapter.__file__,name),node(variant.__file__,name))
 def test_saved_complete_next_update_bitwise_and_recipe_cache_refusal(self):
  with tempfile.TemporaryDirectory() as d:
   cache=Path(d);(cache/'manifest.json').write_text('{}');ctx=self.context(cache)
   b,g,r,u,s=adapter.build(ctx.mu,ctx.theta,'cpu',0,0)
   m=adapter.advance(ctx,b,g,r,u,execution_limit=20000)
   for key in ['D_total','G_total','D_gradient_norm','G_gradient_norm','prior_gradient_norm']:self.assertTrue(np.isfinite(m[key]));self.assertGreater(m[key],0)
   self.assertEqual(m['G_total'],m['G_adversarial'])
   q=adapter.snapshot(ctx,b,g,r,u,0,s,'synthetic');torch.save(q,cache/'q.pt');q=torch.load(cache/'q.pt',weights_only=False,map_location='cpu')
   expected_m=adapter.advance(ctx,b,g,r,u,execution_limit=20000);expected=adapter.snapshot(ctx,b,g,r,u,0,s,'synthetic')
   rb,rg,rr,ru,rs=adapter.restore(ctx,q)
   same(expected_m,adapter.advance(ctx,rb,rg,rr,ru,execution_limit=20000));same(expected,adapter.snapshot(ctx,rb,rg,rr,ru,0,rs,'synthetic'))
   for field,value,message in [('native_recipe','bispf_optimized','native BCAP/Adam checkpoint'),('recipe',get_recipe(total_steps=20000,batch_size=32,num_particles=512,z_dim=8).to_dict(),'recipe/seeds differ'),('cache_manifest_SHA256','0'*64,'fold/cache/endpoint mismatch')]:
    bad=copy.deepcopy(q);bad[field]=value
    with self.assertRaisesRegex(ValueError,message):adapter.restore(ctx,bad)
 def test_config_refuses_changed_architecture_and_no_GPU(self):
  for override in [{'z_dim':2},{'stage_balance':True},{'cell_z_dim':32},{'theta_init_scale':10.},{'fourier_feature_scale':5.}]:
   with self.assertRaisesRegex(ValueError,'matched PG8'):update.configure({**CFG,**override})
  update.configure(CFG);self.assertFalse(torch.cuda.is_initialized())

if __name__=='__main__':unittest.main()
