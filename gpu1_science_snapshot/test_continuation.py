"""Complete-state comparison checks; real continuation qualification requires CUDA."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

import continuation
import variant_pg_adapter as variant
import variant_pg_update as update
from test_variant import fixture


class ContinuationComparison(unittest.TestCase):
    def tearDown(self):
        update.configure({'z_dim': 2})
        self.assertFalse(torch.cuda.is_initialized())

    def test_serialized_state_matches_and_changed_model_or_rng_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            (cache / 'manifest.json').write_text('{}')
            ctx = fixture(cache)
            update.configure({'z_dim': 8})
            b, g, r, u, streams = variant.build(ctx.mu, ctx.theta, 'cpu', 0, 0)
            variant.advance(ctx, b, g, r, u, execution_limit=20000)
            checkpoint = variant.snapshot(ctx, b, g, r, u, 0, streams, 'synthetic')
            path = cache / 'synthetic.pt'
            torch.save(checkpoint, path)
            loaded = torch.load(path, map_location='cpu', weights_only=False)
            continuation.same(checkpoint, loaded)
            loaded['torch_cpu_rng'][0] ^= 1
            with self.assertRaisesRegex(AssertionError, 'tensor mismatch'):
                continuation.same(checkpoint, loaded)
            loaded = torch.load(path, map_location='cpu', weights_only=False)
            module = next(iter(loaded['policy']['models'].values()))
            tensor = next(value for value in module.values() if isinstance(value, torch.Tensor) and value.numel())
            tensor.reshape(-1)[0] += 1
            with self.assertRaisesRegex(AssertionError, 'tensor mismatch'):
                continuation.same(checkpoint, loaded)

    def test_native_qualification_refuses_cpu_before_any_checkpoint_or_output_work(self):
        with self.assertRaises(AssertionError):
            continuation.qualify(SimpleNamespace(device=torch.device('cpu')), {}, Path('/not-created'))


if __name__ == '__main__':
    unittest.main()
