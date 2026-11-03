"""Synthetic CPU qualification of frozen subtype observations and PG state parity."""
import ast,copy,json,os,tempfile,unittest
from pathlib import Path
import numpy as np,torch
from bench_core import TypeClassifier
from critic import VectorCritic
from semantic_observation import SubtypeObservation,make_critic
import semantic_pg_adapter as adapter
import semantic_pg_update as update
import variant_pg_adapter as variant
import variant_pg_update as original_update
from test_variant import fixture,same
CFG={'z_dim':8,'stage_balance':False,'cell_z_dim':8,'theta_init_scale':1.,'fourier_feature_scale':1.}
SEM={'enabled':True,'temperature':1.,'epsilon':1e-4}
def context(cache,enabled=True):
 ctx=fixture(cache);ctx.__class__=adapter.Context
 with torch.random.fork_rng(devices=[]):
  torch.manual_seed(777);classifier=TypeClassifier(50)
 ctx.space=type('Space',(),{'clf':classifier})()
 torch.save(classifier.state_dict(),cache/'type_classifier.pt')
 update.configure(CFG);update.configure_semantic({**SEM,'enabled':enabled});ctx.bind_semantic()
 return ctx
class Semantic(unittest.TestCase):
 def tearDown(self):
  update.configure({'z_dim':2});update.configure_semantic({**SEM,'enabled':False});self.assertFalse(torch.cuda.is_initialized())
 def test_update_AST_and_native_recipe_unchanged(self):
  def update_ast(path):return ast.dump(next(n for n in ast.parse(Path(path).read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='advance'),include_attributes=False)
  self.assertEqual(update_ast(update.__file__),update_ast(original_update.__file__))
  update.configure(CFG);original_update.configure(CFG);self.assertEqual(update.recipe().to_dict(),original_update.recipe().to_dict())
 def test_hard_forward_independent_math_permutations_and_surrogate_gradient(self):
  with tempfile.TemporaryDirectory() as d:
   ctx=context(Path(d));obs=ctx.observation;x=torch.randn(3,17,50,generator=torch.Generator().manual_seed(778),requires_grad=True)
   with torch.no_grad():
    labels=obs.classifier(x.reshape(-1,50)).argmax(-1).reshape(3,17)
    expected=torch.stack([torch.bincount(row,minlength=6).to(x.dtype).div(17).sqrt() for row in labels])
   observed=obs.subtype_features(x);self.assertTrue(torch.equal(observed,expected));self.assertTrue(torch.equal(observed,obs.subtype_features(x.flip(1))))
   self.assertTrue(torch.equal(observed.flip(0),obs.subtype_features(x.flip(0))))
   weights=torch.arange(1,7,dtype=x.dtype);(observed*weights).sum().backward();self.assertTrue(torch.isfinite(x.grad).all());self.assertGreater(float(x.grad.abs().sum()),0)
   self.assertTrue(all(not p.requires_grad and p.grad is None for p in obs.classifier.parameters()));self.assertEqual(obs.state_digest(),obs.digest)
 def test_expanded_critic_initial_function_weights_and_RNG_match(self):
  torch.manual_seed(779);base=make_critic(150);state=torch.get_rng_state().clone()
  torch.manual_seed(779);expanded=make_critic(156);same(state,torch.get_rng_state())
  x=torch.randn(5,150,generator=torch.Generator().manual_seed(780));sem=torch.randn(5,6,generator=torch.Generator().manual_seed(781));c=torch.randn(5,7,generator=torch.Generator().manual_seed(782))
  # Matrix dimensions can choose different dot reduction kernels, so functional
  # values use tolerance; parameter values and all retained seeds remain exact.
  torch.testing.assert_close(base(x,c),expanded(torch.cat([x,sem],-1),c),rtol=1e-6,atol=1e-7)
  self.assertTrue(torch.count_nonzero(expanded.phi[0].weight[:,150:156])==0)
  same(base.phi[0].weight[:,:150],expanded.phi[0].weight[:,:150]);same(base.phi[0].weight[:,150:],expanded.phi[0].weight[:,156:])
  for name,value in base.state_dict().items():
   if name!='phi.0.weight':same(value,expanded.state_dict()[name])
 def test_disabled_feature_complete_state_and_next_update_match_PG8(self):
  with tempfile.TemporaryDirectory() as d:
   cache=Path(d);(cache/'manifest.json').write_text('{}');ctx=context(cache,enabled=False);original_update.configure(CFG)
   b,g,r,u,s=variant.build(ctx.mu,ctx.theta,'cpu',0,0);original_metrics=variant.advance(ctx,b,g,r,u,execution_limit=20000);expected=variant.snapshot(ctx,b,g,r,u,0,s,'synthetic')
   b,g,r,u,s=adapter.build(ctx.mu,ctx.theta,'cpu',0,0);metrics=adapter.advance(ctx,b,g,r,u,execution_limit=20000);actual=adapter.snapshot(ctx,b,g,r,u,0,s,'synthetic');actual.pop('semantic_observation');actual.pop('classifier_file_SHA256')
   same(original_metrics,metrics);same(expected,actual)
 def test_enabled_checkpoint_restore_next_update_and_classifier_immutability(self):
  with tempfile.TemporaryDirectory() as d:
   cache=Path(d);(cache/'manifest.json').write_text('{}');ctx=context(cache);before=copy.deepcopy(ctx.observation.classifier.state_dict())
   b,g,r,u,s=adapter.build(ctx.mu,ctx.theta,'cpu',0,0);adapter.advance(ctx,b,g,r,u,execution_limit=20000)
   self.assertGreater(float(b[2].phi[0].weight[:,150:156].detach().abs().sum()),0)
   q=adapter.snapshot(ctx,b,g,r,u,0,s,'synthetic');torch.save(q,cache/'q.pt');q=torch.load(cache/'q.pt',weights_only=False,map_location='cpu')
   expected_metrics=adapter.advance(ctx,b,g,r,u,execution_limit=20000);expected=adapter.snapshot(ctx,b,g,r,u,0,s,'synthetic')
   rb,rg,rr,ru,rs=adapter.restore(ctx,q);observed_metrics=adapter.advance(ctx,rb,rg,rr,ru,execution_limit=20000)
   same(expected_metrics,observed_metrics);same(expected,adapter.snapshot(ctx,rb,rg,rr,ru,0,rs,'synthetic'));same(before,ctx.observation.classifier.state_dict());self.assertTrue(all(p.grad is None for p in ctx.observation.classifier.parameters()))
   changed=copy.deepcopy(q);changed['semantic_observation']['enabled']=False
   with self.assertRaisesRegex(ValueError,'semantic observation/classifier differs'):adapter.restore(ctx,changed)
   update.configure_semantic({**SEM,'enabled':False})
   with self.assertRaisesRegex(ValueError,'semantic observation/classifier differs'):adapter.restore(ctx,q)
   update.configure_semantic(SEM)
   (cache/'type_classifier.pt').write_bytes(b'CHANGED_FROZEN_FILE')
   with self.assertRaisesRegex(ValueError,'semantic observation/classifier differs'):adapter.restore(ctx,q)
   torch.save(ctx.observation.classifier.state_dict(),cache/'type_classifier.pt')
   ctx.classifier_file_SHA256=adapter.sha(cache/'type_classifier.pt')
   with torch.no_grad():next(ctx.observation.classifier.parameters()).view(-1)[0].add_(1)
   with self.assertRaisesRegex(ValueError,'semantic observation/classifier differs'):adapter.restore(ctx,q)
if __name__=='__main__':unittest.main()
