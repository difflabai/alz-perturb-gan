"""Actual full-width public-policy overrun/replay and native cap refusals."""
import copy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import time
import unittest
from unittest.mock import patch

import torch
from particlegan import get_recipe
import released_adapter as model
import released_update as update
from native_fixture import context
from state_fingerprint import fingerprint
import extension

CFG = {'z_dim': 8, 'stage_balance': False, 'cell_z_dim': 8, 'theta_init_scale': 1., 'fourier_feature_scale': 1.}
HYPER = {'generator_lr': .006, 'critic_lr': .009, 'particle_lr': .015, 'reg_coeff': 1., 'reg_kappa': .5, 'prior_reg': .01}


def short_recipe():
    return get_recipe('gan', trainer='bispf_optimized', total_steps=2, batch_size=32,
                      num_particles=512, z_dim=8, lr=.006, d_lr_mult=1.5, prior_lr_mult=2.5,
                      reg_coeff=1., reg_kappa=.5, prior_reg=.01)


class Extension(unittest.TestCase):
    def setUp(self):
        update.configure(CFG)
        update.configure_hyperparameters(HYPER)

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def test_full2000gene_public_execution_cap_overrun_and_complete_replay(self):
        with tempfile.TemporaryDirectory() as d, patch.object(model, 'recipe', short_recipe):
            root = Path(d)
            (root / 'manifest.json').write_text('{}')
            ctx = context(root, fullwidth=True)
            b, g, r, u, streams = model.build(ctx.mu, ctx.theta, 'cpu', 0, 0)
            for _ in range(2):
                model.advance(ctx, b, g, r, u, execution_limit=2)
            self.assertEqual(b[-1].completed_steps, b[0].total_steps)
            initial = extension.snapshot(ctx, b, g, r, u, 0, streams, 'incomplete', initial=2, target=4, horizon=2)
            # The native recipe's default cap is exhausted. Its documented
            # external execution_limit then continues without editing recipe/state.
            with self.assertRaisesRegex(RuntimeError, 'budget exhausted'):
                b[-1].begin_step(torch.zeros(32, 1))
            model.advance(ctx, b, g, r, u, execution_limit=4)
            expected3 = extension.snapshot(ctx, b, g, r, u, 0, streams, 'diagnostic', initial=2, target=4, horizon=2)
            model.advance(ctx, b, g, r, u, execution_limit=4)
            expected4 = extension.snapshot(ctx, b, g, r, u, 0, streams, 'completed', initial=2, target=4, horizon=2)
            rb, rg, rr, ru, rs = extension.restore(ctx, initial, initial=2, target=4, horizon=2)
            self.assertEqual(fingerprint(initial), fingerprint(extension.snapshot(ctx, rb, rg, rr, ru, 0, rs, 'incomplete', initial=2, target=4, horizon=2)))
            out = root / 'extension';out.mkdir()
            rb, final = extension.run_remaining(ctx, rb, rg, rr, ru, 0, rs, out,
                end_monotonic=time.monotonic() + 60, initial=2, target=4, horizon=2, diagnostics=(3,))
            self.assertEqual(fingerprint(expected3), fingerprint(torch.load(out / 'checkpoint00003.pt', map_location='cpu', weights_only=False)))
            actual4 = torch.load(final, map_location='cpu', weights_only=False)
            self.assertEqual(fingerprint(expected4), fingerprint(actual4))
            self.assertEqual(actual4['recipe'], initial['recipe'])
            self.assertEqual(actual4['policy']['recipe'], initial['policy']['recipe'])
            self.assertEqual(actual4['policy']['recipe']['total_steps'], 2)
            self.assertEqual(actual4['policy']['completed_steps'], 4)
            # Public load_state_dict accepts the saved clock beyond the unchanged
            # schedule horizon; all RNG/usage state also replays the next pair.
            rb, rg, rr, ru, rs = extension.restore(ctx, actual4, initial=2, target=4, horizon=2)
            self.assertEqual(fingerprint(actual4), fingerprint(extension.snapshot(ctx, rb, rg, rr, ru, 0, rs, 'completed', initial=2, target=4, horizon=2)))
            model.advance(ctx, rb, rg, rr, ru, execution_limit=6)
            expected5 = extension.snapshot(ctx, rb, rg, rr, ru, 0, rs, 'synthetic', initial=2, target=6, horizon=2)
            rb, rg, rr, ru, rs = extension.restore(ctx, actual4, initial=2, target=4, horizon=2)
            model.advance(ctx, rb, rg, rr, ru, execution_limit=6)
            self.assertEqual(fingerprint(expected5), fingerprint(extension.snapshot(ctx, rb, rg, rr, ru, 0, rs, 'synthetic', initial=2, target=6, horizon=2)))

    def test_completed_parent_metadata_identity_and_recipe_refusals(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d);(root / 'manifest.json').write_text('{}')
            recipe = model.recipe().to_dict()
            auth = {'extension': {'initial_completed_updates': 20000, 'fixed_endpoint': 40000},
                    'cell': {'fold': 0, 'seed': 0}, 'cache_dir': str(root),
                    'released_hyperparameters': dict(HYPER), 'variant_config': dict(CFG)}
            q = {'particlegan_version':'0.9.6','penalty_options':{'input_domain':'continuous'},'schema': 1, 'method': 'pg', 'native_recipe': 'bispf_optimized', 'status': 'completed',
                 'completed_updates': 20000, 'fixed_endpoint': 20000, 'fold': 0, 'seed': 0,
                 'policy': {'completed_steps': 20000, 'device': 'cuda:0', 'recipe': recipe},
                 'usage': {'steps': 20000}, 'released_hyperparameters': dict(HYPER),
                 'variant_config': dict(CFG), 'recipe': recipe,
                 'cache_manifest_SHA256': extension.sha(root / 'manifest.json'),
                 'training_streams': model.training_streams(0, 0)}
            extension.validate_parent(q, auth)
            for key, value in (('status', 'diagnostic'), ('status', 'incomplete'), ('completed_updates', 19999),
                    ('fixed_endpoint', 40000), ('seed', 1), ('cache_manifest_SHA256', '0' * 64),
                    ('released_hyperparameters', {**HYPER, 'particle_lr': .03}),
                    ('recipe', {**recipe, 'total_steps': 40000}),
                    ('policy', {**q['policy'], 'device': 'cpu'}), ('training_streams', {})):
                bad = copy.deepcopy(q);bad[key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    extension.validate_parent(bad, auth)

    def test_native_endpoint_or_horizon_override_refused_before_GPU(self):
        ctx = SimpleNamespace(device=SimpleNamespace(type='cuda'))
        for options in ({'initial': 19999}, {'target': 60000}, {'horizon': 40000}, {'diagnostics': (25000,)}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                extension.run_remaining(ctx, None, None, None, None, 0, None, Path('/not-created'), end_monotonic=0, **options)

    def test_CPU_real_extension_refuses_before_loading_checkpoint_or_output(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d);(root / 'manifest.json').write_text('{}');ctx = context(root)
            with self.assertRaisesRegex(RuntimeError, 'guarded GPU admission'):
                extension.fit(ctx, root / 'refused', 0, {}, end_monotonic=0)
            self.assertFalse((root / 'refused').exists())


if __name__ == '__main__':
    unittest.main()
