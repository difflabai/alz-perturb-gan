"""Actual copied-model initialization/update parity for all20 frozen cells."""
import importlib.util,os,sys,tempfile,unittest
from pathlib import Path
import preset_scope as s
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
s.validate_protocol()
P=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark');SCIENCE=Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/core-comparison-20261103')
sys.path[:0]=list(map(str,[s.MODEL_DIR,s.ROOT/'analysis/duration096',s.ROOT/'model',s.ROOT,SCIENCE,P,P/'current_base_20261029/runtime/lib/python3.12/site-packages',P/'.venv312/lib/python3.12/site-packages']))
from release_env import activate
activate()
import torch,released_adapter as model,released_update as update
from native_fixture import context
from state_fingerprint import fingerprint
spec=importlib.util.spec_from_file_location('unchanged_short_adapter',s.ROOT/'analysis/hpo500/model/released_adapter.py');short=importlib.util.module_from_spec(spec);spec.loader.exec_module(short)
class Factory(unittest.TestCase):
 def tearDown(self):self.assertFalse(torch.cuda.is_initialized())
 def test_exact_native_model_copy_and_nonEMA_recipe(self):
  self.assertEqual(Path(model.__file__).resolve(),s.MODEL_DIR/'released_adapter.py')
  for name in ('released_update.py','released_adapter.py'):self.assertEqual((s.MODEL_DIR/name).read_bytes(),(s.ROOT/'analysis/released_tuning_v096'/name).read_bytes())
  update.configure(s.CFG);update.configure_hyperparameters(s.read(s.proposal_path('r000'))['hyperparameters']);r=update.recipe();self.assertEqual((r.total_steps,r.ema_decay,r.generator_gradient_response),(20000,0,'none'))
 def test_twenty_fresh_native_cells_match_next_pair_across_execution_limits(self):
  for cid in s.IDS:
   update.configure(s.CFG);update.configure_hyperparameters(s.read(s.proposal_path(cid))['hyperparameters'])
   for seed in (0,1):
    with tempfile.TemporaryDirectory() as folder:
     path=Path(folder);(path/'manifest.json').write_text('{}');ctx=context(path,fullwidth=True);states=[]
     for adapter,horizon in ((short,500),(model,20000)):
      bundle,g,r,u,streams=adapter.build(ctx.mu,ctx.theta,'cpu',0,seed);self.assertEqual((bundle[-1].completed_steps,u.steps),(0,0))
      adapter.advance(ctx,bundle,g,r,u,execution_limit=horizon);q=adapter.snapshot(ctx,bundle,g,r,u,seed,streams,'synthetic');q.pop('fixed_endpoint');states.append(q)
      self.assertEqual(q['completed_updates'],1);self.assertEqual(int(q['usage']['draws'].sum()),32)
      self.assertTrue(all(torch.isfinite(v).all() for m in bundle[1:4] for v in m.state_dict().values()))
     self.assertEqual(fingerprint(states[0]),fingerprint(states[1]))
if __name__=='__main__':unittest.main(verbosity=2)
