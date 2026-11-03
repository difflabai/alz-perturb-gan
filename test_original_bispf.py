"""Synthetic CPU qualification of actual native original BISPF components."""
import ast,copy,json,os,tempfile,unittest
from pathlib import Path
import numpy as np,torch
import original_bispf_adapter as adapter
import original_bispf_update as update
import variant_pg_adapter as variant
from particlegan import get_recipe
from particlegan.bispf import BISPFCriticAdam,BISPFCriticPenalty,SinkhornRelativisticLoss,SBNPPriorRegularizer
from particlegan.optim.dualnorm import NormalizedOptimizer
from test_variant import fixture,same
CFG={'z_dim':8,'stage_balance':False,'cell_z_dim':8,'theta_init_scale':1.,'fourier_feature_scale':1.}
class OriginalBISPF(unittest.TestCase):
 def context(self,cache):
  ctx=fixture(cache);ctx.__class__=adapter.Context;return ctx
 def test_declared_native_components_and_same_initial_G_prior(self):
  with tempfile.TemporaryDirectory() as d:
   ctx=self.context(Path(d));b,g,r,u,s=adapter.build(ctx.mu,ctx.theta,'cpu',0,0)
   self.assertEqual(b[0].to_dict(),get_recipe('bispf',total_steps=20000,batch_size=32,num_particles=512,z_dim=8).to_dict())
   for obj,kind in [(b[5],BISPFCriticAdam),(b[6],BISPFCriticPenalty),(b[7],SinkhornRelativisticLoss),(b[8],SBNPPriorRegularizer)]:self.assertIsInstance(obj,kind)
   import variant_pg_update as vu
   vu.configure(CFG);vb,*_=variant.build(ctx.mu,ctx.theta,'cpu',0,0)
   same(b[1].state_dict(),vb[1].state_dict());same(b[2].state_dict(),vb[2].state_dict());same(b[3].state_dict(),vb[3].state_dict());self.assertIsInstance(vb[4],NormalizedOptimizer)
 def test_only_native_G_real_logits_change_in_update_AST(self):
  old=Path(variant.__file__).with_name('variant_pg_update.py').read_text().replace('lg=loss.g_loss(D(f,c2))','lg=loss.g_loss(D(f,c2),D(real2,c2))')
  def body(source):
   node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='advance');node.body=node.body[1:];return ast.dump(node,include_attributes=False)
  self.assertEqual(body(old),body(Path(update.__file__).read_text()))
 def test_full_serialized_restore_next_update_and_penalty_state_bitwise(self):
  with tempfile.TemporaryDirectory() as d:
   cache=Path(d);(cache/'manifest.json').write_text('{}');ctx=self.context(cache)
   b,g,r,u,s=adapter.build(ctx.mu,ctx.theta,'cpu',0,0)
   metrics=adapter.advance(ctx,b,g,r,u,execution_limit=20000)
   self.assertEqual(metrics['step'],1)
   for k in ['D_total','G_total','D_gradient_norm','G_gradient_norm','prior_gradient_norm']:
    self.assertTrue(np.isfinite(metrics[k]));self.assertGreater(metrics[k],0)
   checkpoint=adapter.snapshot(ctx,b,g,r,u,0,s,'synthetic');torch.save(checkpoint,cache/'q.pt');q=torch.load(cache/'q.pt',weights_only=False,map_location='cpu')
   expected_metrics=adapter.advance(ctx,b,g,r,u,execution_limit=20000);expected=adapter.snapshot(ctx,b,g,r,u,0,s,'synthetic')
   rb,rg,rr,ru,rs=adapter.restore(ctx,q);observed=adapter.advance(ctx,rb,rg,rr,ru,execution_limit=20000)
   same(expected_metrics,observed);same(expected,adapter.snapshot(ctx,rb,rg,rr,ru,0,rs,'synthetic'))
   q['native_recipe']='bispf_optimized'
   with self.assertRaisesRegex(ValueError,'original BISPF checkpoint'):adapter.restore(ctx,q)
 def test_invalid_architecture_refuses_and_no_cuda_initialized(self):
  for override in [{'z_dim':2},{'stage_balance':True},{'cell_z_dim':32},{'theta_init_scale':10.},{'fourier_feature_scale':5.}]:
   with self.assertRaisesRegex(ValueError,'matched PG8'):update.configure({**CFG,**override})
  update.configure(CFG);self.assertFalse(torch.cuda.is_initialized())
if __name__=='__main__':unittest.main()
