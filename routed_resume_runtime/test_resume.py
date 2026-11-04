"""Actual native CPU recovery replay; private metadata refusal counterfactuals."""
import copy
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
import binding as b
import preset_scope as scope
import resume_training as resume
import routed_pg_adapter as model
from test_routed_model import fixture
from state_fingerprint import fingerprint


class Resume(unittest.TestCase):
    def test_exact_parent_metadata_and_wrong_boundary_or_reset_refuses(self):
        auth={'cell':dict(scope.CELL),'cpu_fixture':False,'cache_dir':str(b.SCIENCE/'caches/fold0'),
              'continuation':copy.deepcopy(scope.CONTINUATION),'variant_config':scope.CFG,'routed_config':scope.ROUTED}
        q={'schema':1,'method':'pg','native_recipe':'e22_routed','fold':0,'seed':0,'status':'incomplete',
           'fixed_endpoint':20000,'completed_updates':7013,'policy':{'completed_steps':7013,'device':'cuda:0'},
           'cache_manifest_SHA256':b.sha(Path(auth['cache_dir'])/'manifest.json'),'variant_config':scope.CFG,'routed_config':scope.ROUTED,
           'fitting_contexts_seen':32*7013,'protected_contexts_seen':32*7013}
        resume.validate_parent(q,auth)
        for key,value in [('status','completed'),('completed_updates',0),('seed',1),('native_recipe','bcap'),('protected_contexts_seen',0),('routed_config',{})]:
            changed=copy.deepcopy(q);changed[key]=value
            with self.subTest(key=key),self.assertRaises(b.Refusal):resume.validate_parent(changed,auth)

    def test_fullwidth_native_recovery_crosses_probe20_and_replays_all_state(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temp:
            cache=Path(temp);(cache/'manifest.json').write_text('{}');ctx=fixture(cache,genes=2000)
            path=scope.ROOT/'artifacts/diagnostics/routed_final_adapter_CPU_fullwidth_probe20_v1/before19.pt'
            q=torch.load(path,map_location='cpu',weights_only=False)
            original=model.restore(ctx,q)
            model.advance(ctx,original);model.advance(ctx,original)
            expected=model.snapshot(ctx,original,'synthetic')
            continued=model.restore(ctx,q);out=cache/'synthetic_continuation';out.mkdir()
            rt,cp=resume._run_remaining(ctx,continued,out,end_monotonic=time.monotonic()+60,target_updates=21)
            actual=torch.load(cp,map_location='cpu',weights_only=False)
            self.assertEqual(rt.policy.completed_steps,21)
            self.assertEqual(fingerprint(actual),fingerprint(expected))
            self.assertEqual(actual['policy']['routing']['counters']['probes'],8)
            self.assertEqual(actual['fitting_contexts_seen'],32*21)
            self.assertEqual(actual['protected_contexts_seen'],32*21)
            self.assertFalse(torch.cuda.is_initialized())

    def test_real_CPU_resume_refused_before_checkpoint_or_output(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/'must_not_exist'
            with self.assertRaisesRegex(RuntimeError,'guarded GPU'):
                resume.resume_fit(SimpleNamespace(device=torch.device('cpu')),out,0,{},end_monotonic=0)
            self.assertFalse(out.exists())


if __name__=='__main__':unittest.main(verbosity=2)
