"""Actual fullwidth public restore/replay for a resource-interrupted500step fit."""
import datetime,json,os,sys,tempfile,time,unittest
from pathlib import Path
ROOT=Path('/ml2/alz');HERE=Path(__file__).resolve().parent
PROJECT=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark')
SCIENCE=Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/core-comparison-20261103')
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
sys.path[:0]=list(map(str,[HERE,ROOT/'hpo500_r2_runtime',ROOT/'analysis/hpo500/model',ROOT/'analysis/duration096',ROOT/'model',ROOT,ROOT/'routed_resume_runtime',SCIENCE,PROJECT,PROJECT/'current_base_20261029/runtime/lib/python3.12/site-packages',PROJECT/'.venv312/lib/python3.12/site-packages']))
from release_env import activate
activate()
import torch,released_adapter as model,released_update as update
from native_fixture import context
from state_fingerprint import fingerprint
import preset_scope as scope,resume
class Continuation(unittest.TestCase):
    def setUp(self):update.configure(scope.CFG);update.configure_hyperparameters(scope.read(scope.PARENT_PROOF)['hyperparameters'])
    def tearDown(self):self.assertFalse(torch.cuda.is_initialized())
    def test_actual_full2000gene_native_replay_preserves_full_state_and_clock(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'manifest.json').write_text('{}');ctx=context(root,fullwidth=True)
            b,g,r,u,s=model.build(ctx.mu,ctx.theta,'cpu',0,0)
            for _ in range(2):model.advance(ctx,b,g,r,u,execution_limit=500)
            q=model.snapshot(ctx,b,g,r,u,0,s,'synthetic')
            for _ in range(2):model.advance(ctx,b,g,r,u,execution_limit=4)
            expected=model.snapshot(ctx,b,g,r,u,0,s,'synthetic')
            rb,rg,rr,ru,rs=model.restore(ctx,q)
            self.assertEqual(fingerprint(q),fingerprint(model.snapshot(ctx,rb,rg,rr,ru,0,rs,'synthetic')))
            out=root/'continued';out.mkdir()
            rb,path=resume.run_remaining(ctx,rb,rg,rr,ru,0,rs,out,end_monotonic=time.monotonic()+60,target=4)
            actual=torch.load(path,map_location='cpu',weights_only=False)
            self.assertEqual(fingerprint(expected),fingerprint(actual));self.assertEqual(actual['fixed_endpoint'],500);self.assertEqual(actual['recipe']['total_steps'],20000)
            self.assertEqual(actual['usage']['steps'],4)
    def test_real_CPU_fit_refuses_before_checkpoint_or_output(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'manifest.json').write_text('{}');ctx=context(root)
            with self.assertRaises(RuntimeError):resume.fit(ctx,root/'forbidden',0,{},end_monotonic=0)
            self.assertFalse((root/'forbidden').exists())
    def test_only_declared48_to500_native_continuation(self):
        from types import SimpleNamespace
        ctx=SimpleNamespace(device=SimpleNamespace(type='cuda'))
        b=[None]*9+[SimpleNamespace(completed_steps=48)]
        with self.assertRaises(AssertionError):resume.run_remaining(ctx,b,None,None,None,0,None,Path('/not-created'),end_monotonic=0,target=400)
        b[-1].completed_steps=47
        with self.assertRaises(AssertionError):resume.run_remaining(ctx,b,None,None,None,0,None,Path('/not-created'),end_monotonic=0)
if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Continuation));assert result.wasSuccessful() and result.testsRun==3
    paths=[*HERE.glob('*.py'),*scope.MODEL_DIR.glob('*.py'),scope.MODEL_PROOF,scope.PARENT_PROOF,scope.RECOVERY_PROTOCOL]
    q=dict(status='HPO500_RESOURCE_CONTINUATION_CPU_PASS',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),tests_passed=3,complete_fullwidth_native_state_next_pair_replay=True,CUDA_initialized=False,GPU_allocation_created=False,source_SHA256={str(p):scope.sha(p) for p in paths})
    scope.RESUME_PROOF.write_text(json.dumps(q,indent=2)+'\n')
