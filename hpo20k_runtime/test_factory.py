"""Actual fullwidth CPU equivalence to the same500-step recipe at c014."""
import importlib.util,os,sys,tempfile,unittest
from pathlib import Path
import preset_scope as s
P=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark');SCIENCE=Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/core-comparison-20261103')
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
sys.path[:0]=list(map(str,[s.MODEL_DIR,s.ROOT/'analysis/duration096',s.ROOT/'model',s.ROOT,SCIENCE,P,P/'current_base_20261029/runtime/lib/python3.12/site-packages',P/'.venv312/lib/python3.12/site-packages']))
from release_env import activate
activate()
import torch,released_adapter as model,released_update as update
from native_fixture import context
from state_fingerprint import fingerprint
spec=importlib.util.spec_from_file_location('unchanged_hpo500_adapter',s.ROOT/'analysis/hpo500/model/released_adapter.py');short=importlib.util.module_from_spec(spec);spec.loader.exec_module(short)
class Factory(unittest.TestCase):
 def tearDown(self):self.assertFalse(torch.cuda.is_initialized())
 def test_native_c014_recipe_and_unchanged_update(self):
  h=s.read(s.SETTINGS)['hyperparameters'];update.configure(s.CFG);update.configure_hyperparameters(h);rec=update.recipe()
  self.assertEqual(rec.to_dict(),short.recipe().to_dict());self.assertEqual((rec.total_steps,rec.ema_decay,rec.generator_gradient_response),(20000,0,'none'));self.assertEqual(model.PG_UPDATES,20000)
  self.assertEqual((s.MODEL_DIR/'released_update.py').read_bytes(),(s.ROOT/'analysis/hpo500/model/released_update.py').read_bytes())
 def test_two_complete_pairs_have_identical_native_state_to500step_adapter(self):
  h=s.read(s.SETTINGS)['hyperparameters'];update.configure(s.CFG);update.configure_hyperparameters(h)
  with tempfile.TemporaryDirectory() as folder:
   root=Path(folder);(root/'manifest.json').write_text('{}');ctx=context(root,fullwidth=True);snap=[]
   for adapter,limit in ((short,500),(model,20000)):
    bundle,g,r,u,seeds=adapter.build(ctx.mu,ctx.theta,'cpu',0,0)
    for _ in range(2):adapter.advance(ctx,bundle,g,r,u,execution_limit=limit)
    q=adapter.snapshot(ctx,bundle,g,r,u,0,seeds,'synthetic');q.pop('fixed_endpoint');snap.append(q)
    self.assertEqual(bundle[-1].completed_steps,2);self.assertEqual(int(u.draws.sum()),64)
    self.assertTrue(all(torch.isfinite(v).all() for m in bundle[1:4] for v in m.state_dict().values()))
   self.assertEqual(fingerprint(snap[0]),fingerprint(snap[1]))
if __name__=='__main__':unittest.main(verbosity=2)
