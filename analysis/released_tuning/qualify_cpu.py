"""Actual finite synthetic released-API tests; no ALZ fitting or GPU allocation."""
import ast
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path('/ml2/alz')
HERE = Path(__file__).resolve().parent
PROJECT = Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark')
SCIENCE = Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/core-comparison-20261103')
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
sys.path[:0] = list(map(str, [HERE, ROOT/'model', ROOT, ROOT/'routed_resume_runtime', SCIENCE,
    PROJECT, PROJECT/'current_base_20261029/runtime/lib/python3.12/site-packages',
    PROJECT/'.venv312/lib/python3.12/site-packages']))
import numpy as np
import torch
from particlegan import get_recipe
from particlegan.recipes import learning_rate_scales
import released_adapter as adapter
import released_update as update
import variant_pg_adapter as original
import variant_pg_update as old_update
from state_fingerprint import fingerprint
from test_variant import fixture

PROTOCOL = json.loads((HERE/'protocol.json').read_text())
BASE = PROTOCOL['baseline_parameters']
CFG = dict(update.CONFIG)


def node(path, name):
    return ast.dump(next(n for n in ast.parse(Path(path).read_text()).body
        if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name), include_attributes=False)


def context(cache, fullwidth=False):
    ctx = fixture(cache)
    ctx.__class__ = adapter.Context
    if fullwidth:
        ctx.X = torch.randint(1, 8, (24, 2000), generator=torch.Generator().manual_seed(774)).float()
        ctx.mean, ctx.sd = torch.zeros(2000), torch.ones(2000)
        ctx.W = torch.randn(2000, 50, generator=torch.Generator().manual_seed(775))/2000
        ctx.trainpcs, ctx.mu = ctx.project(ctx.X), ctx.X.mean(0)
        ctx.theta = (ctx.mu.square()/(ctx.X.var(0, unbiased=False)-ctx.mu).clamp_min(1e-4)).clamp(.1,10000)
    return ctx


class ReleasedTuning(unittest.TestCase):
    def tearDown(self):
        update.configure_hyperparameters(BASE)
        old_update.configure(CFG)
        self.assertFalse(torch.cuda.is_initialized())

    def test_all27_recipes_and_actual_optimizer_rates_only_declared_fields(self):
        with tempfile.TemporaryDirectory() as d:
            ctx = context(Path(d))
            reference = get_recipe('gan', trainer='bispf_optimized', total_steps=20000,
                batch_size=32, num_particles=512, z_dim=8).to_dict()
            for setting in PROTOCOL['stage1']['configs']:
                with self.subTest(setting=setting['id']):
                    h = setting['parameters']
                    update.configure_hyperparameters(h)
                    b, *_ = adapter.build(ctx.mu,ctx.theta,'cpu',0,0)
                    rec = b[0]
                    self.assertEqual(rec.to_dict(), update.resolved_recipe(h).to_dict())
                    changed = {k for k in reference if rec.to_dict()[k] != reference[k]}
                    self.assertFalse(changed - {'lr','d_lr_mult','prior_lr_mult'})
                    for actual, expected in zip([b[4].param_groups[0]['lr'], b[5].param_groups[0]['lr'],
                            b[4].param_groups[1]['lr']], [h['generator_lr'],h['critic_lr'],h['particle_lr']]):
                        self.assertAlmostEqual(actual/expected, 1., places=14)
                    for step in (0,10000,19999):
                        self.assertEqual(learning_rate_scales(step,rec),(1.,1.))

    def test_baseline_initial_state_and_two_actual_updates_match_original_PG8(self):
        with tempfile.TemporaryDirectory() as d:
            cache=Path(d);(cache/'manifest.json').write_text('{}');ctx=context(cache)
            old_update.configure(CFG)
            ob,og,orr,ou,os = original.build(ctx.mu,ctx.theta,'cpu',0,0)
            for _ in range(2):original.advance(ctx,ob,og,orr,ou,execution_limit=20000)
            expected=original.snapshot(ctx,ob,og,orr,ou,0,os,'synthetic')
            update.configure_hyperparameters(BASE)
            b,g,r,u,s = adapter.build(ctx.mu,ctx.theta,'cpu',0,0)
            for _ in range(2):adapter.advance(ctx,b,g,r,u,execution_limit=20000)
            actual=adapter.snapshot(ctx,b,g,r,u,0,s,'synthetic')
            before=fingerprint(actual)
            diagnostics=adapter.passive_particle_diagnostics(b,u)
            self.assertGreater(diagnostics['rows_drawn'],0)
            self.assertEqual(diagnostics['total_draws'],64)
            self.assertEqual(diagnostics['total_unsampled_row_moves'],0)
            self.assertEqual(before,fingerprint(adapter.snapshot(ctx,b,g,r,u,0,s,'synthetic')))
            actual.pop('native_recipe');actual.pop('released_hyperparameters')
            self.assertEqual(fingerprint(actual),fingerprint(expected))

    def test_native_update_observation_context_and_seed_streams_preserved(self):
        for name in ('advance','bag_features','finite','gradnorm','normalized_config'):
            self.assertEqual(node(update.__file__,name),node(old_update.__file__,name))
        for name in ('Context','training_streams'):
            self.assertEqual(node(adapter.__file__,name),node(original.__file__,name))

    def test_extremes_and_regularization_serialized_next_pair_replay(self):
        cases=[PROTOCOL['stage1']['configs'][0]['parameters'],
            dict(BASE,generator_lr=.003,critic_lr=.0045,particle_lr=.0075),
            dict(BASE,reg_coeff=.25,reg_kappa=4.,prior_reg=.0025),
            dict(BASE,reg_coeff=4.,reg_kappa=.25,prior_reg=.04)]
        with tempfile.TemporaryDirectory() as d:
            cache=Path(d);(cache/'manifest.json').write_text('{}');ctx=context(cache)
            for h in cases:
                update.configure_hyperparameters(h)
                b,g,r,u,s=adapter.build(ctx.mu,ctx.theta,'cpu',0,0)
                metrics=adapter.advance(ctx,b,g,r,u,execution_limit=20000)
                for key in ('D_gradient_norm','G_gradient_norm','prior_gradient_norm'):
                    self.assertTrue(np.isfinite(metrics[key]) and metrics[key]>0)
                self.assertAlmostEqual(metrics['G_total'],metrics['G_adversarial']+h['prior_reg']*metrics['prior_regularization_unweighted'],places=5)
                self.assertEqual(b[6].regularizer.coeff,h['reg_coeff'])
                self.assertEqual(b[6].regularizer.kappa,h['reg_kappa'])
                q=adapter.snapshot(ctx,b,g,r,u,0,s,'diagnostic')
                torch.save(q,cache/'q.pt');q=torch.load(cache/'q.pt',map_location='cpu',weights_only=False)
                expected_metrics=adapter.advance(ctx,b,g,r,u,execution_limit=20000)
                expected=adapter.snapshot(ctx,b,g,r,u,0,s,'synthetic')
                rb,rg,rr,ru,rs=adapter.restore(ctx,q)
                self.assertEqual(fingerprint(expected_metrics),fingerprint(adapter.advance(ctx,rb,rg,rr,ru,execution_limit=20000)))
                self.assertEqual(fingerprint(expected),fingerprint(adapter.snapshot(ctx,rb,rg,rr,ru,0,rs,'synthetic')))

    def test_wrong_hyperparameters_or_clock_or_configuration_refused(self):
        with tempfile.TemporaryDirectory() as d:
            cache=Path(d);(cache/'manifest.json').write_text('{}');ctx=context(cache)
            update.configure_hyperparameters(BASE);b,g,r,u,s=adapter.build(ctx.mu,ctx.theta,'cpu',0,0)
            adapter.advance(ctx,b,g,r,u,execution_limit=20000)
            q=adapter.snapshot(ctx,b,g,r,u,0,s,'synthetic')
            for key,value in [('native_recipe','bispf'),('released_hyperparameters',dict(BASE,prior_reg=.02)),
                    ('cache_manifest_SHA256','0'*64),('completed_updates',2),('status','completed'),
                    ('fixed_endpoint',40000),('variant_config',dict(CFG,z_dim=2))]:
                bad=copy.deepcopy(q);bad[key]=value
                with self.subTest(key=key),self.assertRaises(ValueError):adapter.restore(ctx,bad)
            for bad in (dict(BASE,alpha=.5),dict(BASE,reg_coeff=0),dict(BASE,particle_lr=float('nan')),dict(BASE,generator_lr=True)):
                with self.assertRaises(ValueError):update.configure_hyperparameters(bad)
            for bad in (dict(CFG,z_dim=2),dict(CFG,stage_balance=True)):
                with self.assertRaises(ValueError):update.configure(bad)
            with self.assertRaisesRegex(RuntimeError,'separately reviewed guarded GPU admission'):
                adapter.fit(ctx,cache/'refused',0,end_monotonic=0)
            self.assertFalse((cache/'refused').exists())

    def test_full2000gene_count_model_connected_gradients_and_exact_replay(self):
        with tempfile.TemporaryDirectory() as d:
            cache=Path(d);(cache/'manifest.json').write_text('{}');ctx=context(cache,fullwidth=True)
            update.configure_hyperparameters(PROTOCOL['stage1']['configs'][0]['parameters'])
            b,g,r,u,s=adapter.build(ctx.mu,ctx.theta,'cpu',0,0)
            metrics=adapter.advance(ctx,b,g,r,u,execution_limit=20000)
            for key in ('D_gradient_norm','G_gradient_norm','prior_gradient_norm'):
                self.assertTrue(np.isfinite(metrics[key]) and metrics[key]>0)
            q=adapter.snapshot(ctx,b,g,r,u,0,s,'synthetic')
            expected_metrics=adapter.advance(ctx,b,g,r,u,execution_limit=20000)
            expected=adapter.snapshot(ctx,b,g,r,u,0,s,'synthetic')
            rb,rg,rr,ru,rs=adapter.restore(ctx,q)
            self.assertEqual(fingerprint(expected_metrics),fingerprint(adapter.advance(ctx,rb,rg,rr,ru,execution_limit=20000)))
            self.assertEqual(fingerprint(expected),fingerprint(adapter.snapshot(ctx,rb,rg,rr,ru,0,rs,'synthetic')))


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ReleasedTuning))
    if not result.wasSuccessful():raise SystemExit(1)
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    paths=[HERE/'released_adapter.py',HERE/'released_update.py',Path(__file__),HERE/'protocol.json',
        ROOT/'model/variant_pg_adapter.py',ROOT/'model/variant_pg_update.py',ROOT/'test_variant.py',
        ROOT/'routed_resume_runtime/state_fingerprint.py']
    native=json.loads((ROOT/'analysis/particlegan_release_reverification_20261104T1538.json').read_text())
    proof={'status':'PASSED_ACTUAL_RELEASED_PARAMETER_SYNTHETIC_CPU_QUALIFICATION',
        'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'tests_passed':result.testsRun,
        'all27_declared_native_recipes_and_optimizer_rates_verified':True,
        'baseline_PG8_two_update_complete_state_byte_parity':True,
        'same_native_update_observation_context_and_seed_streams_AST':True,
        'rate_and_regularization_extremes_connected_gradients_and_full_state_replay':True,
        'full2000gene_connected_gradients_and_exact_next_pair_replay':True,
        'wrong_hyperparameter_clock_status_recipe_cache_configuration_and_CPU_fit_refused':True,
        'CUDA_initialized':torch.cuda.is_initialized(),'GPU_job_created':False,
        'quality_or_convergence_known':False,'source_SHA256':{str(p):sha(p) for p in paths},
        'scope':'Finite synthetic CPU qualification only. No actual ALZ fitting, GPU admission or convergence/quality results. Original released native package and live E22 sources unchanged.'}
    (HERE/'CPU_qualification.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps(proof,indent=2))
