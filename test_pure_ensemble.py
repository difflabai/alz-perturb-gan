"""Synthetic CPU streams validate whole-bag selection and global RNG isolation."""
import unittest
import numpy as np
from pure_ensemble_generation import model_mask, iter_ensemble


def factories(events, bad=None):
    def branch(index):
        events.append(('start', index)); np.random.seed(918 + index)
        for start in range(0, 200, 5):
            events.append(('chunk', index))
            pcs = np.broadcast_to((np.arange(start, start + 5) + 1000 * index + np.random.random())[:, None, None], (5, 400, 50)).astype(np.float32)
            counts = np.full((1, 400, 2000), index + 1, dtype=np.float32)
            totals = np.full(2000, 10 + index, dtype=np.float32)
            if bad == 'gap' and start == 5: start += 1
            if bad == 'nonfinite' and start == 0: pcs[0, 0, 0] = np.nan
            if bad == 'noninteger' and start == 0: counts[0, 0, 0] = .5
            if bad == 'short' and start == 195: return
            yield start, pcs, counts, totals
    return {i: lambda i=i: branch(i) for i in (0, 1)}


def arrays(stream):
    chunks = list(stream)
    return np.concatenate([q[1] for q in chunks]), np.concatenate([q[3] for q in chunks]).reshape(200, 400), chunks[0][2]


class PureEnsemble(unittest.TestCase):
    def test_exact_mask_counts_determinism_and_no_global_rng_use(self):
        before = np.random.get_state()
        for rep in (0, 1):
            for stage in range(4):
                a = model_mask(.25, rep, stage)
                self.assertEqual(int(a.sum()), 50)
                np.testing.assert_array_equal(a, model_mask(.25, rep, stage))
        after = np.random.get_state()
        for a, b in zip(before, after): np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(model_mask(.25, 0, 0), model_mask(.25, 1, 0)))

    def test_zero_one_controls_exactly_replay_parent_and_skip_unused_branch(self):
        for weight, index in [(0., 0), (1., 1)]:
            events = []
            actual = arrays(iter_ensemble(factories(events), weight=weight, replicate=0, stage=0, check_deadline=lambda: None))
            expected = arrays(factories([])[index]())
            for a, b in zip(actual, expected):
                self.assertEqual(a.dtype, b.dtype); self.assertEqual(a.tobytes(), b.tobytes())
            self.assertEqual([e for e in events if e[0] == 'start'], [('start', index)])

    def test_quarter_mixture_is_whole_bags_and_branches_are_fully_sequential(self):
        events = []
        actual = arrays(iter_ensemble(factories(events), weight=.25, replicate=1, stage=2, check_deadline=lambda: None))
        a, b = [arrays(factories([])[i]()) for i in (0, 1)]
        mask = model_mask(.25, 1, 2)
        for bag in range(200):
            expected = b if mask[bag] else a
            np.testing.assert_array_equal(actual[0][bag], expected[0][bag])
            np.testing.assert_array_equal(actual[1][bag], expected[1][bag])
        np.testing.assert_array_equal(actual[2], (b if mask[0] else a)[2])
        starts = [e for e in events if e[0] == 'start']
        self.assertEqual(starts, [('start', 0), ('start', 1)])
        first_b = events.index(('start', 1))
        self.assertEqual(events[1:first_b], [('chunk', 0)] * 40)
        self.assertEqual(events[first_b + 1:], [('chunk', 1)] * 40)

    def test_unregistered_weights_or_queries_refuse(self):
        for weight in [True, .1, .5, float('nan')]:
            with self.assertRaises(ValueError): model_mask(weight, 0, 0)
        for rep, stage in [(2, 0), (0, 4), (False, 0)]:
            with self.assertRaises(ValueError): model_mask(.25, rep, stage)

    def test_malformed_or_incomplete_parent_refuses_before_any_yield(self):
        for bad in ['gap', 'short', 'nonfinite', 'noninteger']:
            yielded = []
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                for q in iter_ensemble(factories([], bad), weight=.25, replicate=0, stage=0, check_deadline=lambda: None): yielded.append(q)
            self.assertEqual(yielded, [])

    def test_deadline_during_branch_collection_refuses_without_partial_yield(self):
        calls = []; yielded = []
        def check():
            calls.append(1)
            if len(calls) == 7: raise TimeoutError('synthetic deadline')
        with self.assertRaises(TimeoutError):
            for q in iter_ensemble(factories([]), weight=.25, replicate=0, stage=0, check_deadline=check): yielded.append(q)
        self.assertEqual(yielded, [])


if __name__ == '__main__': unittest.main()
