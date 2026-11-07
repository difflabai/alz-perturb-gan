"""Actual selected native recipe at every fold/seed; synthetic fullwidth counts."""
import importlib.util,os,sys,tempfile,unittest
from pathlib import Path
import preset_scope as s
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
s.protocol()
P=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark');SCIENCE=Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/core-comparison-20261103')
sys.path[:0]=list(map(str,[s.MODEL_DIR,s.ROOT/'analysis/duration096',s.ROOT/'model',s.ROOT,SCIENCE,P,P/'current_base_20261029/runtime/lib/python3.12/site-packages',P/'.venv312/lib/python3.12/site-packages']))
from release_env import activate
activate()
import torch,released_adapter as model,released_update as update
from native_fixture import context
from state_fingerprint import fingerprint
class Factory(unittest.TestCase):
 def tearDown(self):self.assertFalse(torch.cuda.is_initialized())
 def test_selected_native_recipe_fixed_and_nonEMA(self):
  update.configure(s.CFG);update.configure_hyperparameters(s.hyperparameters());recipe=update.recipe()
  self.assertEqual((recipe.total_steps,recipe.ema_decay,recipe.generator_gradient_response),(20000,0,'none'))
  self.assertEqual(s.sha(s.MODEL_DIR/'released_update.py'),s.read(s.MODEL_PROOF)['source_SHA256'][str(s.MODEL_DIR/'released_update.py')])
 def test_all15_native_initializations_and_exact_next_pair_horizon_parity(self):
  update.configure(s.CFG);update.configure_hyperparameters(s.hyperparameters())
  for fold in range(3):
   for seed in range(5):
    with tempfile.TemporaryDirectory() as folder:
     root=Path(folder);(root/'manifest.json').write_text('{}');ctx=context(root,fullwidth=True);ctx.fold=fold;states=[]
     for adapter,limit in ((model,500),(model,20000)):
      bundle,g,r,u,streams=adapter.build(ctx.mu,ctx.theta,'cpu',fold,seed)
      self.assertEqual(bundle[-1].completed_steps,0);self.assertEqual(u.steps,0)
      adapter.advance(ctx,bundle,g,r,u,execution_limit=limit)
      q=adapter.snapshot(ctx,bundle,g,r,u,seed,streams,'synthetic');q.pop('fixed_endpoint');states.append(q)
      self.assertEqual(bundle[-1].completed_steps,1);self.assertEqual(int(u.draws.sum()),32)
      self.assertTrue(all(torch.isfinite(t).all() for m in bundle[1:4] for t in m.state_dict().values()))
     self.assertEqual(fingerprint(states[0]),fingerprint(states[1]))
if __name__=='__main__':unittest.main(verbosity=2)
