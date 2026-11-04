"""Synthetic CPU qualification of fixed-capacity stage-conditioned native particles."""
import ast,copy,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
import stage_pg_adapter as a
import stage_pg_update as u
import stage_prior as sp
import variant_pg_adapter as base
import variant_pg_update as bu
from test_variant import fixture,same
CFG={'z_dim':8,'stage_balance':False,'cell_z_dim':8,'theta_init_scale':1.,'fourier_feature_scale':1.}
class Stage(unittest.TestCase):
 def setUp(self):u.configure(CFG);bu.configure(CFG)
 def context(self,p):ctx=fixture(p);ctx.__class__=a.Context;return ctx
 def test_exact_recipe_initial_models_old11streams_and_no_extra_capacity(self):
  with tempfile.TemporaryDirectory() as d:
   ctx=self.context(Path(d));b,*_=a.build(ctx.mu,ctx.theta,'cpu',0,0);r,*_=base.build(ctx.mu,ctx.theta,'cpu',0,0)
   same(b[0].to_dict(),r[0].to_dict())
   for i in (1,2,3):same(b[i].state_dict(),r[i].state_dict())
   self.assertEqual(a.training_streams(0,0),base.training_streams(0,0));self.assertEqual(tuple(b[3].z.shape),(512,8));self.assertTrue(b[3].z.requires_grad)
 def test_all_stage_supports_replacement_rng_and_generation_view_match(self):
  with tempfile.TemporaryDirectory() as d:
   ctx=self.context(Path(d));b,*_=a.build(ctx.mu,ctx.theta,'cpu',0,0);P=b[3];c=torch.zeros(512,7);c[:,3]=torch.arange(4).repeat_interleave(128)/3
   g=torch.Generator().manual_seed(33);ref=torch.Generator().manual_seed(33);z,idx=sp.sample_stage(P,c,generator=g);expected=torch.arange(4).repeat_interleave(128)*128+torch.randint(128,(512,),generator=ref)
   same(idx,expected);same(z,P.z[idx]);same(g.get_state(),ref.get_state());self.assertLess(len(idx.unique()),512)
   old=torch.Generator().manual_seed(33);P.sample(512,generator=old);same(g.get_state(),old.get_state())
   for stage in range(4):
    c=torch.zeros(5,7);c[:,3]=stage/3;view=sp.view_for_condition(P,c[0]);g=torch.Generator().manual_seed(88);rg=torch.Generator().manual_seed(88)
    x,xi=view.sample(5,generator=g);y,yi=sp.sample_stage(P,c,generator=rg);same(x,y);same(xi,yi)
    view.eval();self.assertFalse(P.training);view.train();self.assertTrue(P.training)
 def test_native_update_gradients_sampled_row_ownership_and_training_changes(self):
  with tempfile.TemporaryDirectory() as d:
   ctx=self.context(Path(d));b,g,r,usage,seeds=a.build(ctx.mu,ctx.theta,'cpu',0,0);before=b[3].z.detach().clone()
   row=a.advance(ctx,b,g,r,usage,execution_limit=20000)
   for key in ('G_gradient_norm','prior_gradient_norm','D_gradient_norm'):self.assertTrue(np.isfinite(row[key]));self.assertGreater(row[key],0)
   self.assertFalse(torch.equal(before,b[3].z));self.assertEqual(int(usage.unsampled_moved_steps.sum()),0);self.assertEqual(int(usage.draws.sum()),32)
 def test_full_serialized_next_update_bitwise_and_metadata_refusals(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/'manifest.json').write_text('{}');ctx=self.context(p);b,g,r,usage,seeds=a.build(ctx.mu,ctx.theta,'cpu',0,0);a.advance(ctx,b,g,r,usage,execution_limit=20000)
   q=a.snapshot(ctx,b,g,r,usage,0,seeds,'synthetic');torch.save(q,p/'q.pt');q=torch.load(p/'q.pt',weights_only=False,map_location='cpu');expected=a.advance(ctx,b,g,r,usage,execution_limit=20000);state=a.snapshot(ctx,b,g,r,usage,0,seeds,'synthetic')
   rb,rg,rr,ru,rs=a.restore(ctx,q);same(expected,a.advance(ctx,rb,rg,rr,ru,execution_limit=20000));same(state,a.snapshot(ctx,rb,rg,rr,ru,0,rs,'synthetic'))
   for key,value in [('native_recipe','ka2'),('stage_prior_config',{**sp.CONFIG,'particles_per_stage':64}),('cache_manifest_SHA256','0'*64)]:
    bad=copy.deepcopy(q);bad[key]=value
    with self.assertRaises(ValueError):a.restore(ctx,bad)
 def test_Context_fit_streams_and_update_AST_only_two_sampling_calls_change(self):
  def node(path,name):return copy.deepcopy(next(x for x in ast.parse(Path(path).read_text()).body if isinstance(x,(ast.FunctionDef,ast.ClassDef)) and x.name==name))
  for name in ('Context','fit','training_streams'):self.assertEqual(ast.dump(node(a.__file__,name),include_attributes=False),ast.dump(node(base.__file__,name),include_attributes=False))
  actual=node(u.__file__,'advance');expected=node(bu.__file__,'advance');count=0
  for call in ast.walk(actual):
   if isinstance(call,ast.Call) and isinstance(call.func,ast.Name) and call.func.id=='sample_stage':
    self.assertEqual(ast.unparse(call.args[0]),'P');self.assertIn(ast.unparse(call.args[1]),('c','c2'));call.func=ast.Attribute(value=ast.Name(id='P',ctx=ast.Load()),attr='sample',ctx=ast.Load());call.args=[ast.Constant(value=32)];count+=1
  self.assertEqual(count,2);self.assertEqual(ast.dump(actual,include_attributes=False),ast.dump(expected,include_attributes=False))
 def test_invalid_conditions_configs_and_CPU_fitting_refused(self):
  with tempfile.TemporaryDirectory() as d:
   ctx=self.context(Path(d));b,*_=a.build(ctx.mu,ctx.theta,'cpu',0,0)
   for value in (-1.,.2,4/3,float('nan')):
    c=torch.zeros(1,7);c[:,3]=value
    with self.assertRaises(ValueError):sp.sample_stage(b[3],c,generator=torch.Generator())
   c=torch.zeros(2,7);c[1,3]=1/3
   with self.assertRaises(ValueError):sp.view_for_condition(b[3],c)
   with self.assertRaises(ValueError):u.configure({'z_dim':2})
   with self.assertRaises(RuntimeError):a.fit(ctx,Path(d)/'fit',0,end_monotonic=0)
  self.assertFalse(torch.cuda.is_initialized())
if __name__=='__main__':unittest.main()
