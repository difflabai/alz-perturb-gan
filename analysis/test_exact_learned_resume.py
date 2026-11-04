"""Synthetic CPU checks for adapter-specific learned-observation continuation."""
import copy,io,os,sys,tempfile,unittest
from pathlib import Path
R=Path('/ml2/alz');B=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark');S=Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/core-comparison-20261103')
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
sys.path[:0]=list(map(str,[R/'recovery_runtime',R/'model',R,B,S,B/'current_base_20261029/runtime/lib/python3.12/site-packages',B/'.venv312/lib/python3.12/site-packages']))
import torch
from test_variant import fixture
from continuation import same
from ensemble_scope import CFG,OBS
import learned_pg_adapter as adapter
import learned_pg_update as update
import variant_pg_update
from exact_resume import restore_exact,qualify_exact,fit_resume

class LearnedResume(unittest.TestCase):
 def setUp(self):
  torch.set_num_threads(1);update.configure(CFG);variant_pg_update.configure(CFG);update.configure_observation(OBS)
  self.temp=tempfile.TemporaryDirectory();self.cache=Path(self.temp.name);(self.cache/'manifest.json').write_text('{}');self.ctx=fixture(self.cache);self.ctx.__class__=adapter.Context
  self.bundle=adapter.build(self.ctx.mu,self.ctx.theta,'cpu',0,1)
  b,g,r,u,s=self.bundle
  for _ in range(2):adapter.advance(self.ctx,b,g,r,u,execution_limit=20000)
  self.q=adapter.snapshot(self.ctx,b,g,r,u,1,s,'incomplete')
 def tearDown(self):self.assertFalse(torch.cuda.is_initialized());self.temp.cleanup()
 def test_full_serialized_restore_and_next_update_match_uninterrupted(self):
  stream=io.BytesIO();torch.save(self.q,stream);stream.seek(0);q=torch.load(stream,map_location='cpu',weights_only=False);same(self.q,q)
  b,g,r,u,s=self.bundle;expected_m=adapter.advance(self.ctx,b,g,r,u,execution_limit=20000);expected=adapter.snapshot(self.ctx,b,g,r,u,1,s,'qualification_only')
  b,g,r,u,s=restore_exact(self.ctx,q,1,expected_initial=2,adapter=adapter);observed_m=adapter.advance(self.ctx,b,g,r,u,execution_limit=20000);observed=adapter.snapshot(self.ctx,b,g,r,u,1,s,'qualification_only')
  same(expected_m,observed_m);same(expected,observed)
 def test_two_saved_replay_branches_equal_and_input_unchanged(self):
  before=copy.deepcopy(self.q);result,state=qualify_exact(self.ctx,self.q,expected_initial=2,adapter=adapter,branch_directory=self.cache/'branches');same(before,self.q)
  a=torch.load(self.cache/'branches/branch0.pt',map_location='cpu',weights_only=False);b=torch.load(self.cache/'branches/branch1.pt',map_location='cpu',weights_only=False)
  same(a,b);same(b,state);self.assertEqual(state['completed_updates'],3);self.assertTrue(result['complete_next_state_and_metrics_bitwise_equal']);self.assertFalse(result['calibration_or_evaluation_scores_generated'])
 def test_learned_config_mismatch_is_rejected(self):
  q=copy.deepcopy(self.q);q['learned_observation_config']['scale']=.3
  with self.assertRaises(ValueError):restore_exact(self.ctx,q,1,expected_initial=2,adapter=adapter)
 def test_partial_missing_encoder_state_is_rejected(self):
  q=copy.deepcopy(self.q);key=next(k for k in q['policy']['models']['critic'] if k.startswith('cell_encoder'));del q['policy']['models']['critic'][key]
  with self.assertRaises((RuntimeError,ValueError)):restore_exact(self.ctx,q,1,expected_initial=2,adapter=adapter)
 def test_wrong_seed_count_horizon_or_status_is_rejected(self):
  for k,v in [('seed',2),('completed_updates',3),('fixed_endpoint',40000),('status','completed')]:
   q=copy.deepcopy(self.q);q[k]=v
   with self.subTest(k=k),self.assertRaises((AssertionError,ValueError)):restore_exact(self.ctx,q,1,expected_initial=2,adapter=adapter)
 def test_cpu_cannot_enter_native_training_loop(self):
  with self.assertRaises(RuntimeError):fit_resume(self.ctx,self.cache/'not_native',1,checkpoint=self.q,checkpoint_SHA256='0'*64,expected_initial=2,end_monotonic=float('inf'),adapter=adapter)
  self.assertFalse((self.cache/'not_native').exists())
if __name__=='__main__':unittest.main()
