"""Actual native fullwidth CPU state/replay and exact-parent metadata refusals."""
import copy,json,sys,tempfile,time,unittest
from pathlib import Path
import binding as b
from types import SimpleNamespace
import preset_scope as scope
import resume_training as resume
from state_fingerprint import fingerprint
import torch
import released_update as update
import released_adapter as model
from qualify_cpu import context

class Resume(unittest.TestCase):
    def test_parent_metadata_clock_configuration_and_seed_refusals(self):
        with tempfile.TemporaryDirectory() as d:
            cache=Path(d);(cache/'manifest.json').write_text('{}')
            auth={'cell':dict(scope.CELL),'continuation':scope.CONTINUATION,'cache_dir':str(cache),
                'variant_config':dict(scope.CFG),'released_hyperparameters':dict(scope.HYPER)}
            q={'schema':1,'method':'pg','native_recipe':'bispf_optimized','status':'incomplete','fold':0,'seed':0,
                'completed_updates':2589,'fixed_endpoint':20000,'policy':{'completed_steps':2589,'device':'cuda:0'},
                'cache_manifest_SHA256':b.sha(cache/'manifest.json'),'variant_config':dict(scope.CFG),
                'released_hyperparameters':dict(scope.HYPER),'usage':{'steps':2589}}
            resume.validate_parent(q,auth)
            for key,value in [('status','completed'),('native_recipe','e22_routed'),('seed',1),
                    ('completed_updates',0),('usage',{'steps':0}),('fixed_endpoint',40000),
                    ('released_hyperparameters',{**scope.HYPER,'particle_lr':.03}),
                    ('policy',{'completed_steps':2589,'device':'cpu'})]:
                bad=copy.deepcopy(q);bad[key]=value
                with self.subTest(key=key),self.assertRaises(b.Refusal):resume.validate_parent(bad,auth)

    def test_full2000gene_whole_scientific_state_and_remaining_pairs_byte_exact(self):
        update.configure(scope.CFG);update.configure_hyperparameters(scope.HYPER)
        with tempfile.TemporaryDirectory() as d:
            cache=Path(d);(cache/'manifest.json').write_text('{}');ctx=context(cache,fullwidth=True)
            bundle,g,r,u,s=model.build(ctx.mu,ctx.theta,'cpu',0,0)
            for _ in range(2):model.advance(ctx,bundle,g,r,u,execution_limit=20000)
            q=model.snapshot(ctx,bundle,g,r,u,0,s,'incomplete')
            torch.save(q,cache/'q.pt');q=torch.load(cache/'q.pt',map_location='cpu',weights_only=False)
            rb,rg,rr,ru,rs=model.restore(ctx,q)
            self.assertEqual(fingerprint(q),fingerprint(model.snapshot(ctx,rb,rg,rr,ru,0,rs,'incomplete')))
            model.advance(ctx,rb,rg,rr,ru,execution_limit=20000)
            expected_diagnostic=model.snapshot(ctx,rb,rg,rr,ru,0,rs,'diagnostic')
            model.advance(ctx,rb,rg,rr,ru,execution_limit=20000)
            expected=model.snapshot(ctx,rb,rg,rr,ru,0,rs,'synthetic')
            rb,rg,rr,ru,rs=model.restore(ctx,q)
            out=cache/'continuation';out.mkdir()
            result,path=resume._run_remaining(ctx,rb,rg,rr,ru,0,rs,out,end_monotonic=time.monotonic()+60,target_updates=4,diagnostic_updates=(3,))
            self.assertEqual(fingerprint(expected),fingerprint(model.snapshot(ctx,result,rg,rr,ru,0,rs,'synthetic')))
            self.assertEqual(fingerprint(expected),fingerprint(torch.load(path,map_location='cpu',weights_only=False)))
            self.assertEqual(fingerprint(expected_diagnostic),fingerprint(torch.load(out/'checkpoint00003.pt',map_location='cpu',weights_only=False)))
            self.assertEqual(result[-1].completed_steps,4)
            self.assertFalse(torch.cuda.is_initialized())

    def test_native_endpoint_or_diagnostic_schedule_override_refused_before_GPU(self):
        ctx=SimpleNamespace(device=SimpleNamespace(type='cuda'))
        for options in ({'target_updates':4},{'diagnostic_updates':(3,)}):
            with self.subTest(options=options),self.assertRaises(ValueError):
                resume._run_remaining(ctx,None,None,None,None,0,None,Path('/not-created'),end_monotonic=0,**options)
        self.assertFalse(torch.cuda.is_initialized())

    def test_actual_CPU_resume_fit_refuses_before_checkpoint_or_output(self):
        with tempfile.TemporaryDirectory() as d:
            cache=Path(d);(cache/'manifest.json').write_text('{}');ctx=context(cache)
            with self.assertRaisesRegex(RuntimeError,'guarded GPU admission'):
                resume.resume_fit(ctx,cache/'refused',0,{},end_monotonic=time.monotonic()+60)
            self.assertFalse((cache/'refused').exists())
        self.assertFalse(torch.cuda.is_initialized())
if __name__=='__main__':unittest.main()
