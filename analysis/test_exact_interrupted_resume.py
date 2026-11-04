"""Synthetic CPU restoration/next-update parity; never ALZ fitting or CUDA."""
import copy,io,tempfile,unittest
from pathlib import Path
import torch
from test_variant import fixture
import variant_pg_adapter as variant
import variant_pg_update as update
from continuation import same
from exact_resume import restore_exact,qualify_exact,fit_resume

class InterruptedResume(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory(prefix='alz-cpu-interrupted-resume-');self.cache=Path(self.temp.name);(self.cache/'manifest.json').write_text('{}')
  update.configure({'z_dim':8});self.ctx=fixture(self.cache)
  b,g,r,u,streams=variant.build(self.ctx.mu,self.ctx.theta,'cpu',0,1)
  for _ in range(2):variant.advance(self.ctx,b,g,r,u,execution_limit=20000)
  self.checkpoint=variant.snapshot(self.ctx,b,g,r,u,1,streams,'incomplete');self.bundle=(b,g,r,u,streams)
 def tearDown(self):
  self.assertFalse(torch.cuda.is_initialized());self.temp.cleanup()
 def test_serialized_full_initial_and_next_update_match_uninterrupted(self):
  payload=io.BytesIO();torch.save(self.checkpoint,payload);payload.seek(0);restored=torch.load(payload,map_location='cpu',weights_only=False)
  same(self.checkpoint,restored)
  b,g,r,u,streams=self.bundle
  expected_metrics=variant.advance(self.ctx,b,g,r,u,execution_limit=20000);expected=variant.snapshot(self.ctx,b,g,r,u,1,streams,'synthetic_fixture')
  rb,rg,rr,ru,rs=restore_exact(self.ctx,restored,1,expected_initial=2)
  actual_metrics=variant.advance(self.ctx,rb,rg,rr,ru,execution_limit=20000);actual=variant.snapshot(self.ctx,rb,rg,rr,ru,1,rs,'synthetic_fixture')
  same(expected_metrics,actual_metrics);same(expected,actual)
 def test_two_throwaway_replay_branches_match_and_parent_is_unchanged(self):
  before=copy.deepcopy(self.checkpoint);result,state=qualify_exact(self.ctx,self.checkpoint,expected_initial=2)
  same(before,self.checkpoint);self.assertTrue(result['complete_next_state_and_metrics_bitwise_equal']);self.assertEqual(state['completed_updates'],3)
  self.assertEqual(result['executed_optimizer_pairs'],2);self.assertFalse(result['calibration_or_evaluation_scores_generated'])
 def test_wrong_seed_endpoint_initial_step_or_architecture_refuses(self):
  cases=[('seed',2),('fixed_endpoint',40000),('completed_updates',3),('variant_config',{'z_dim':2}),('status','completed')]
  for key,value in cases:
   q=copy.deepcopy(self.checkpoint);q[key]=value
   with self.subTest(field=key),self.assertRaises((AssertionError,ValueError)):restore_exact(self.ctx,q,1,expected_initial=2)
  for initial in (False,0,3,20000):
   with self.assertRaises(AssertionError):restore_exact(self.ctx,self.checkpoint,1,expected_initial=initial)
 def test_changed_cache_or_native_recipe_refuses(self):
  for key,value in [('cache_manifest_SHA256','0'*64),('recipe',{**self.checkpoint['recipe'],'lr':.006})]:
   q=copy.deepcopy(self.checkpoint);q[key]=value
   with self.subTest(field=key),self.assertRaises((AssertionError,ValueError)):restore_exact(self.ctx,q,1,expected_initial=2)
 def test_CPU_fixture_cannot_start_actual_resume_loop(self):
  out=self.cache/'NOT-REAL-TRAINING'
  with self.assertRaises(RuntimeError):fit_resume(self.ctx,out,1,checkpoint=self.checkpoint,checkpoint_SHA256='0'*64,expected_initial=2,end_monotonic=float('inf'))
  self.assertFalse(out.exists())
if __name__=='__main__':unittest.main()
