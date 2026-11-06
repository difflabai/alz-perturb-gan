"""Real native factories and full2000gene optimizer pairs at smallerkappa."""
import os,sys,tempfile,unittest
from pathlib import Path
import preset_scope as s
ROOT=s.ROOT;P=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark');SCIENCE=Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/core-comparison-20261103')
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
sys.path[:0]=list(map(str,[s.MODEL_DIR,ROOT/'analysis/duration096',ROOT/'model',ROOT,SCIENCE,P,P/'current_base_20261029/runtime/lib/python3.12/site-packages',P/'.venv312/lib/python3.12/site-packages']))
from release_env import activate
activate()
import torch,released_update as update,released_adapter as model
from native_fixture import context
class Factory(unittest.TestCase):
 def tearDown(self):self.assertFalse(torch.cuda.is_initialized())
 def test_public_recipes_change_only_kappa_through_the64th(self):
  h=s.read(s.ANCHOR)['anchor_hyperparameters'];update.configure(s.CFG);update.configure_hyperparameters(h);base=update.recipe().to_dict()
  for factor in (.5,.25,.125,.0625,.03125,.015625):
   update.configure_hyperparameters({**h,'reg_kappa':h['reg_kappa']*factor});rec=update.recipe();self.assertEqual({k for k in base if base[k]!=rec.to_dict()[k]},{'reg_kappa'});self.assertEqual(rec.ema_decay,0);self.assertEqual(rec.generator_gradient_response,'none');self.assertEqual(rec.total_steps,20000)
 def test_fullwidth_native_lowerkappa_pairs_are_finite_with_original_clocks(self):
  h=s.read(s.ANCHOR)['anchor_hyperparameters']
  for factor in (.0625,.015625):
   with tempfile.TemporaryDirectory() as folder:
    root=Path(folder);(root/'manifest.json').write_text('{}');ctx=context(root,fullwidth=True);update.configure(s.CFG);update.configure_hyperparameters({**h,'reg_kappa':h['reg_kappa']*factor});b,g,r,u,seeds=model.build(ctx.mu,ctx.theta,'cpu',0,0);q=model.advance(ctx,b,g,r,u,execution_limit=500);self.assertEqual(q['step'],1);self.assertEqual(u.steps,1);self.assertEqual(int(u.draws.sum()),32)
    for module in b[1:4]:self.assertTrue(all(torch.isfinite(x).all() for x in module.state_dict().values()))
if __name__=='__main__':unittest.main(verbosity=2)
