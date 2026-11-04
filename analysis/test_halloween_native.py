"""Native Halloween count adapter: role wiring, complete replay and refusals."""
import ast
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import halloween_pg_adapter as a
import halloween_pg_update as u
import variant_pg_adapter as baseline
import variant_pg_update as original
from particlegan import get_recipe
from particlegan.tensorflow_adam import TensorFlowV1Adam
from particlegan.k3p import CriticPenalty
from particlegan.gan_loss import GANLoss
from particlegan.vicreg_loss import ParticleRegularizer
from test_variant import fixture, same

CFG = {'z_dim': 8, 'stage_balance': False, 'cell_z_dim': 8,
       'theta_init_scale': 1., 'fourier_feature_scale': 1.}


def node(path, name):
    return ast.dump(next(x for x in ast.parse(Path(path).read_text()).body
                         if isinstance(x, (ast.ClassDef, ast.FunctionDef)) and x.name == name),
                    include_attributes=False)


class NativeHalloween(unittest.TestCase):
    def context(self, cache):
        ctx = fixture(cache)
        ctx.__class__ = a.Context
        return ctx

    def test_exact_recipe_roles_labels_and_initial_state(self):
        with tempfile.TemporaryDirectory() as d:
            ctx = self.context(Path(d))
            b, *_ = a.build(ctx.mu, ctx.theta, 'cpu', 0, 0)
            rec = get_recipe('halloween', total_steps=20000, batch_size=32, num_particles=512, z_dim=8)
            self.assertEqual(b[0].to_dict(), rec.to_dict())
            for obj, kind in [(b[4], TensorFlowV1Adam), (b[5], TensorFlowV1Adam),
                              (b[6], CriticPenalty), (b[7], GANLoss), (b[8], ParticleRegularizer)]:
                self.assertIsInstance(obj, kind)
            self.assertEqual(b[7].loss, 'least_squares')
            self.assertEqual(b[7].labels, (-1., 1., 1.))
            for actual, betas, eps in [(b[4].param_groups[0], rec.betas, rec.eps),
                                       (b[4].param_groups[1], rec.prior_betas, rec.prior_eps),
                                       (b[5].param_groups[0], rec.d_betas, rec.d_eps)]:
                self.assertEqual(actual['betas'], betas)
                self.assertEqual(actual['eps'], eps)
                self.assertEqual(actual['_tf_initial_betas'], betas)
                self.assertEqual(actual['_tf_step'], 0)
            original.configure(CFG)
            vb, *_ = baseline.build(ctx.mu, ctx.theta, 'cpu', 0, 0)
            for i in (1, 2):
                same(b[i].state_dict(), vb[i].state_dict())
            same(dict(b[3].named_parameters()), dict(vb[3].named_parameters()))
            self.assertEqual(a.training_streams(0, 0), baseline.training_streams(0, 0))

    def test_preserved_full_update_data_and_fitting_AST(self):
        for name in ('finite', 'gradnorm', 'bag_features', 'advance'):
            self.assertEqual(node(u.__file__, name), node(original.__file__, name))
        for name in ('Context', 'training_streams', 'fit'):
            self.assertEqual(node(a.__file__, name), node(baseline.__file__, name))

    def test_actual_update_gradients_movement_and_native_clocks(self):
        with tempfile.TemporaryDirectory() as d:
            ctx = self.context(Path(d))
            b, g, r, usage, _ = a.build(ctx.mu, ctx.theta, 'cpu', 0, 0)
            before = [copy.deepcopy(m.state_dict()) for m in b[1:4]]
            m = a.advance(ctx, b, g, r, usage, execution_limit=20000)
            for key in ('D_total', 'G_total', 'D_gradient_norm', 'G_gradient_norm', 'prior_gradient_norm'):
                self.assertTrue(np.isfinite(m[key]))
                self.assertGreater(m[key], 0)
            self.assertEqual(m['D_penalty'], 0.)
            self.assertEqual(m['G_total'], m['G_adversarial'])
            for old, model in zip(before, b[1:4]):
                self.assertTrue(any(not torch.equal(v, model.state_dict()[k]) for k, v in old.items()))
            for opt in b[4:6]:
                self.assertEqual(opt.state_dict()['tensorflow_v1'], {'schema_version': 1, 'gradient_law': 'dense_real'})
                for group in opt.param_groups:
                    self.assertEqual(group['_tf_step'], 1)
                    for beta, key in zip(group['betas'], ('_tf_beta1_power', '_tf_beta2_power')):
                        self.assertEqual(group[key], beta)
            self.assertEqual(int(usage.draws.sum()), 32)
            self.assertGreater(int(usage.moved_steps.sum()), 0)
            self.assertEqual(int(usage.unsampled_moved_steps.sum()), 0)

    def test_complete_serialized_next_update_replay(self):
        with tempfile.TemporaryDirectory() as d:
            cache = Path(d)
            (cache / 'manifest.json').write_text('{}')
            ctx = self.context(cache)
            b, g, r, usage, seeds = a.build(ctx.mu, ctx.theta, 'cpu', 0, 0)
            a.advance(ctx, b, g, r, usage, execution_limit=20000)
            q = a.snapshot(ctx, b, g, r, usage, 0, seeds, 'synthetic')
            torch.save(q, cache / 'checkpoint.pt')
            q = torch.load(cache / 'checkpoint.pt', map_location='cpu', weights_only=False)
            expected_m = a.advance(ctx, b, g, r, usage, execution_limit=20000)
            expected = a.snapshot(ctx, b, g, r, usage, 0, seeds, 'synthetic')
            rb, rg, rr, ru, rs = a.restore(ctx, q)
            same(expected_m, a.advance(ctx, rb, rg, rr, ru, execution_limit=20000))
            same(expected, a.snapshot(ctx, rb, rg, rr, ru, 0, rs, 'synthetic'))

    def test_wrong_native_recipe_cache_and_optimizer_marker_refused(self):
        with tempfile.TemporaryDirectory() as d:
            cache = Path(d)
            (cache / 'manifest.json').write_text('{}')
            ctx = self.context(cache)
            b, g, r, usage, seeds = a.build(ctx.mu, ctx.theta, 'cpu', 0, 0)
            a.advance(ctx, b, g, r, usage, execution_limit=20000)
            q = a.snapshot(ctx, b, g, r, usage, 0, seeds, 'synthetic')
            for key, value in [('native_recipe', 'bcap_adam'),
                               ('cache_manifest_SHA256', '0' * 64),
                               ('recipe', get_recipe(total_steps=20000, batch_size=32, num_particles=512, z_dim=8).to_dict())]:
                bad = copy.deepcopy(q)
                bad[key] = value
                with self.assertRaises(ValueError):
                    a.restore(ctx, bad)
            bad = copy.deepcopy(q)
            bad['policy']['optimizers'][0].pop('tensorflow_v1')
            with self.assertRaisesRegex(ValueError, 'optimizer state') as caught:
                a.restore(ctx, bad)
            self.assertIn('tensorflow_v1', str(caught.exception.__cause__))

    def test_architecture_changes_and_real_CPU_fit_refused(self):
        for override in [{'z_dim': 2}, {'stage_balance': True}, {'cell_z_dim': 32},
                         {'theta_init_scale': 10.}, {'fourier_feature_scale': 5.}]:
            with self.assertRaisesRegex(ValueError, 'matched PG8'):
                u.configure({**CFG, **override})
        u.configure(CFG)
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(RuntimeError, 'guarded GPU admission'):
                a.fit(self.context(Path(d)), Path(d) / 'fit', 0, end_monotonic=float('inf'))
            self.assertFalse((Path(d) / 'fit').exists())
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main()
