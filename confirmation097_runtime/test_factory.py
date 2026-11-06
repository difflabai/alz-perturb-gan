"""Actual full2000gene prefix parity at all four new initialization seeds."""
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
 def test_native_formulation_and_horizon_unchanged(self):
  update.configure(s.CFG);update.configure_hyperparameters(s.read(s.proposal_path('d000'))['hyperparameters']);r=update.recipe()
  self.assertEqual((r.total_steps,r.ema_decay,r.generator_gradient_response),(20000,0,'none'));self.assertEqual(model.PG_UPDATES,20000)
  self.assertEqual((s.MODEL_DIR/'released_update.py').read_bytes(),(s.ROOT/'analysis/released_tuning_v096/released_update.py').read_bytes())
 def test_all_frozen_configurations_and_fresh_seeds_exact_next_pair_parity(self):
  for cid in s.ENABLED['candidate_ids']:
   update.configure(s.CFG);update.configure_hyperparameters(s.read(s.proposal_path(cid))['hyperparameters'])
   for seed in range(1,5):
    with tempfile.TemporaryDirectory() as folder:
     root=Path(folder);(root/'manifest.json').write_text('{}');ctx=context(root,fullwidth=True);states=[]
     for adapter,limit in ((short,500),(model,20000)):
      bundle,g,r,u,streams=adapter.build(ctx.mu,ctx.theta,'cpu',0,seed)
      self.assertEqual(bundle[-1].completed_steps,0);self.assertEqual(u.steps,0)
      for _ in range(2):adapter.advance(ctx,bundle,g,r,u,execution_limit=limit)
      q=adapter.snapshot(ctx,bundle,g,r,u,seed,streams,'synthetic');self.assertEqual(q['particlegan_version'],'0.9.7');q.pop('fixed_endpoint');states.append(q)
      self.assertEqual(bundle[-1].completed_steps,2);self.assertEqual(int(u.draws.sum()),64)
      self.assertTrue(all(torch.isfinite(t).all() for m in bundle[1:4] for t in m.state_dict().values()))
     self.assertEqual(fingerprint(states[0]),fingerprint(states[1]))
if __name__=='__main__':unittest.main(verbosity=2)
