import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from hybrid import bag_mask, blend_bags


class Hybrid(unittest.TestCase):
    def fixture(self):
        raw = torch.stack([torch.full((2000,), float(i + 1)) for i in range(6)])
        rows = np.tile(np.arange(400) % 6, (200, 1))
        pcs = np.arange(6 * 50, dtype=np.float32).reshape(6, 50)
        ctx = SimpleNamespace(X=raw, device=torch.device('cpu'), baseline_pcs=pcs)
        schedule = SimpleNamespace(rows=rows)
        generated = (0, np.full((5, 400, 50), -7., dtype=np.float32),
                     np.full((1, 400, 2000), 9., dtype=np.float32), np.full(2000, 18000.))
        return ctx, schedule, generated

    def test_fixed_masks_exact_and_repeatable(self):
        for rep in [0, 1]:
            for stage in range(4):
                mask = bag_mask(.65, rep, stage)
                self.assertEqual(int(mask.sum()), 130)
                np.testing.assert_array_equal(mask, bag_mask(.65, rep, stage))
        self.assertFalse(np.array_equal(bag_mask(.65, 0, 0), bag_mask(.65, 1, 0)))

    def test_all_PG_is_exact_identity(self):
        ctx, schedule, original = self.fixture()
        observed = next(blend_bags(ctx, schedule, iter([original]), bag_mask(1., 0, 0)))
        for expected, value in zip(original[1:], observed[1:]):
            np.testing.assert_array_equal(expected, value)

    def test_all_resampling_is_raw_training_counts_and_PC_rows(self):
        ctx, schedule, original = self.fixture()
        start, pcs, exemplar, totals = next(blend_bags(ctx, schedule, iter([original]), bag_mask(0., 0, 0)))
        np.testing.assert_array_equal(pcs, ctx.baseline_pcs[schedule.rows[:5]])
        np.testing.assert_array_equal(exemplar, ctx.X[schedule.rows[:1]].numpy())
        np.testing.assert_array_equal(totals.reshape(5, 400), ctx.X[schedule.rows[:5]].sum(-1).numpy())

    def test_mixture_changes_only_chosen_whole_bags(self):
        ctx, schedule, original = self.fixture()
        mask = np.ones(200, dtype=bool)
        mask[[0, 2, 4]] = False
        _, pcs, exemplar, totals = next(blend_bags(ctx, schedule, iter([original]), mask))
        np.testing.assert_array_equal(pcs[[1, 3]], original[1][[1, 3]])
        np.testing.assert_array_equal(pcs[[0, 2, 4]], ctx.baseline_pcs[schedule.rows[[0, 2, 4]]])
        np.testing.assert_array_equal(exemplar, ctx.X[schedule.rows[:1]].numpy())
        np.testing.assert_array_equal(totals.reshape(5, 400)[[1, 3]], original[3].reshape(5, 400)[[1, 3]])

    def test_bad_or_silently_rounded_fraction_refused(self):
        for fraction in [True, float('nan'), -1., 1.1, .652]:
            with self.assertRaises(ValueError):
                bag_mask(fraction, 0, 0)


if __name__ == '__main__':
    unittest.main()
