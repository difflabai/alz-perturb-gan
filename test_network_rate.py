"""Native rate-only qualification with original PG8 observations and count law."""
import ast
import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from particlegan import get_recipe
from particlegan.recipes import learning_rate_scales

import network_rate_adapter as adapter
import network_rate_update as update
import variant_pg_adapter as original
import variant_pg_update as baseline_update
from test_variant import fixture, same

CFG = {'z_dim': 8, 'stage_balance': False, 'cell_z_dim': 8,
       'theta_init_scale': 1., 'fourier_feature_scale': 1.}


class NetworkRate(unittest.TestCase):
    def context(self, cache):
        ctx = fixture(cache)
        ctx.__class__ = adapter.Context
        return ctx

    def test_official_recipe_only_two_fields_change_and_initial_models_match(self):
        with tempfile.TemporaryDirectory() as d:
            ctx = self.context(Path(d))
            b, *rest = adapter.build(ctx.mu, ctx.theta, 'cpu', 0, 0)
            baseline_update.configure(CFG)
            old, *old_rest = original.build(ctx.mu, ctx.theta, 'cpu', 0, 0)
            expected = get_recipe(total_steps=20000, batch_size=32, num_particles=512,
                                  z_dim=8, lr=.006, prior_lr_mult=5.)
            self.assertEqual(b[0].to_dict(), expected.to_dict())
            self.assertEqual({k for k in b[0].to_dict() if b[0].to_dict()[k] != old[0].to_dict()[k]},
                             {'lr', 'prior_lr_mult'})
            for i in (1, 2, 3):
                same(b[i].state_dict(), old[i].state_dict())
            for i in (4, 5, 6, 7, 8):
                self.assertIs(type(b[i]), type(old[i]))
            for i, rates in [(4, [.006, .006 * 5.]), (5, [.006 * 1.5])]:
                state = copy.deepcopy(old[i].state_dict())
                for group, rate in zip(state['param_groups'], rates):
                    group['lr'] = rate
                same(b[i].state_dict(), state)
            same(rest[-1], old_rest[-1])
            for step in (0, 10000, 19999):
                self.assertEqual(learning_rate_scales(step, b[0]), (1., 1.))

    def test_update_observations_context_and_fit_AST_unchanged(self):
        def node(path, name):
            return ast.dump(next(n for n in ast.parse(Path(path).read_text()).body
                                 if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name),
                            include_attributes=False)
        for name in ['advance', 'finite', 'gradnorm', 'bag_features', 'normalized_config']:
            self.assertEqual(node(update.__file__, name), node(baseline_update.__file__, name))
        for name in ['Context', 'training_streams', 'fit']:
            self.assertEqual(node(adapter.__file__, name), node(original.__file__, name))

    def test_complete_serialized_restore_and_next_update_bitwise_equal(self):
        with tempfile.TemporaryDirectory() as d:
            cache = Path(d)
            (cache / 'manifest.json').write_text('{}')
            ctx = self.context(cache)
            b, g, r, u, streams = adapter.build(ctx.mu, ctx.theta, 'cpu', 0, 0)
            metrics = adapter.advance(ctx, b, g, r, u, execution_limit=20000)
            for key in ['D_total', 'G_total', 'D_gradient_norm', 'G_gradient_norm', 'prior_gradient_norm']:
                self.assertTrue(np.isfinite(metrics[key]))
                if key.endswith('gradient_norm'):
                    self.assertGreater(metrics[key], 0)
            self.assertEqual(metrics['G_prior_LRs'], [.006, .03])
            self.assertEqual(metrics['D_LRs'], [.006 * 1.5])
            self.assertAlmostEqual(metrics['G_total'], metrics['G_adversarial'] + b[0].prior_reg * metrics['prior_regularization_unweighted'], places=5)
            q = adapter.snapshot(ctx, b, g, r, u, 0, streams, 'synthetic')
            torch.save(q, cache / 'q.pt')
            q = torch.load(cache / 'q.pt', weights_only=False, map_location='cpu')
            expected_metrics = adapter.advance(ctx, b, g, r, u, execution_limit=20000)
            expected = adapter.snapshot(ctx, b, g, r, u, 0, streams, 'synthetic')
            rb, rg, rr, ru, rs = adapter.restore(ctx, q)
            same(expected_metrics, adapter.advance(ctx, rb, rg, rr, ru, execution_limit=20000))
            same(expected, adapter.snapshot(ctx, rb, rg, rr, ru, 0, rs, 'synthetic'))
            for field, value in [('network_rate_config', {'lr': .012, 'prior_lr_mult': 2.5}),
                                 ('native_recipe', 'bcap'), ('cache_manifest_SHA256', '0' * 64),
                                 ('recipe', get_recipe(total_steps=20000, batch_size=32, num_particles=512, z_dim=8).to_dict())]:
                bad = copy.deepcopy(q)
                bad[field] = value
                with self.subTest(field=field), self.assertRaises(ValueError):
                    adapter.restore(ctx, bad)

    def test_changed_architecture_and_actual_CPU_fitting_refuse(self):
        for change in [{'z_dim': 2}, {'stage_balance': True}, {'cell_z_dim': 32},
                       {'theta_init_scale': 10.}, {'fourier_feature_scale': 5.}]:
            with self.assertRaisesRegex(ValueError, 'matched PG8'):
                update.configure({**CFG, **change})
        update.configure(CFG)
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(RuntimeError, 'real PG fitting'):
                adapter.fit(self.context(Path(d)), Path(d) / 'no-fit', 0, end_monotonic=0)
            self.assertFalse((Path(d) / 'no-fit').exists())
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main()
