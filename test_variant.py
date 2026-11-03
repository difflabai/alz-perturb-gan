"""Synthetic CPU parity, complete continuation and architecture checks."""
import ast
import copy
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

import pg_adapter as original
import variant_pg_adapter as variant
import variant_pg_update as update

assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def same(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            same(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            same(x, y)
    else:
        assert a == b, (a, b)


def fixture(cache):
    x = torch.randint(1, 8, (24, 8), generator=torch.Generator().manual_seed(774)).float()
    ctx = object.__new__(variant.Context)
    ctx.cache, ctx.fold, ctx.device = cache, 0, torch.device('cpu')
    ctx.ids = np.arange(6)
    ctx.by = [np.arange(4 * i, 4 * i + 4) for i in range(6)]
    ctx.X, ctx.conditions = x, torch.zeros(6, 7)
    ctx.conditions[:, 3] = torch.tensor([0., 1/3, 2/3, 1., 1., 1.])
    ctx.mean, ctx.sd = torch.zeros(8), torch.ones(8)
    ctx.W = torch.randn(8, 50, generator=torch.Generator().manual_seed(775)) / 8
    ctx.trainpcs, ctx.mu = ctx.project(x), x.mean(0)
    ctx.theta = (ctx.mu.square() / (x.var(0, unbiased=False) - ctx.mu).clamp_min(1e-4)).clamp(.1, 10000)
    return ctx


class Variant(unittest.TestCase):
    def tearDown(self):
        update.configure({'z_dim': 2})
        self.assertFalse(torch.cuda.is_initialized())

    def test_original_update_and_input_adapter_bodies_unchanged(self):
        source = Path(original.__file__).parent
        def node(path, name):
            return ast.dump(next(n for n in ast.parse(path.read_text()).body
                                 if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name), include_attributes=False)
        for name in ['advance', 'finite', 'gradnorm']:
            self.assertEqual(node(source / 'pg_update.py', name), node(Path(update.__file__), name))
        for name in ['Context', 'fit', 'snapshot', 'restore', 'training_streams']:
            self.assertEqual(node(Path(original.__file__), name), node(Path(variant.__file__), name))

    def test_latent2_reproduces_original_full_state_and_next_update(self):
        with tempfile.TemporaryDirectory() as d:
            cache = Path(d)
            (cache / 'manifest.json').write_text('{}')
            ctx = fixture(cache)
            b, g, r, u, streams = original.build(ctx.mu, ctx.theta, 'cpu', 0, 0)
            original.advance(ctx, b, g, r, u, execution_limit=20000)
            expected = original.snapshot(ctx, b, g, r, u, 0, streams, 'synthetic')
            update.configure({'z_dim': 2})
            b, g, r, u, streams = variant.build(ctx.mu, ctx.theta, 'cpu', 0, 0)
            variant.advance(ctx, b, g, r, u, execution_limit=20000)
            same(expected, variant.snapshot(ctx, b, g, r, u, 0, streams, 'synthetic'))

    def test_latent8_complete_restore_then_next_update_bitwise_equal(self):
        update.configure({'z_dim': 8})
        with tempfile.TemporaryDirectory() as d:
            cache = Path(d)
            (cache / 'manifest.json').write_text('{}')
            ctx = fixture(cache)
            b, g, r, u, streams = variant.build(ctx.mu, ctx.theta, 'cpu', 0, 0)
            self.assertEqual(tuple(b[3].z.shape), (512, 8))
            self.assertEqual(b[1].donor[0].in_features, 15)
            variant.advance(ctx, b, g, r, u, execution_limit=20000)
            checkpoint = variant.snapshot(ctx, b, g, r, u, 0, streams, 'synthetic')
            expected_metrics = variant.advance(ctx, b, g, r, u, execution_limit=20000)
            expected = variant.snapshot(ctx, b, g, r, u, 0, streams, 'synthetic')
            rb, rg, rr, ru, rs = variant.restore(ctx, checkpoint)
            observed_metrics = variant.advance(ctx, rb, rg, rr, ru, execution_limit=20000)
            same(expected_metrics, observed_metrics)
            same(expected, variant.snapshot(ctx, rb, rg, rr, ru, 0, rs, 'synthetic'))
            update.configure({'z_dim': 2})
            with self.assertRaisesRegex(ValueError, 'recipe/seeds differ'):
                variant.restore(ctx, checkpoint)


if __name__ == '__main__':
    unittest.main()
