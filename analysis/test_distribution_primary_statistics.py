"""Check paired-distance algebra against direct distances, including clusters."""
import unittest
import numpy as np
from scipy.spatial.distance import cdist
from distribution_primary_statistics import (paired_components,
    average_fixed_models, donor_cluster_interval)


def direct(a, b):
    return 2 * cdist(a, b).mean() - cdist(a, a).mean() - cdist(b, b).mean()


class Statistics(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(124)
        self.pg = rng.normal(size=(35, 3))
        self.other = rng.normal(size=(41, 3)) + .8
        self.labels = np.repeat(np.arange(8), np.arange(2, 10))
        self.real = rng.normal(size=(len(self.labels), 3))
        self.order = np.arange(8)
        self.component = paired_components(self.pg, self.other, self.real,
                                           self.labels, self.order)

    def test_point_matches_independent_direct_energy_distances(self):
        self.assertAlmostEqual(self.component['point_delta'],
            direct(self.pg, self.real) - direct(self.other, self.real), places=12)

    def test_unequal_cluster_resampling_matches_direct_repeated_cell_pool(self):
        chosen = np.array([0, 0, 2, 3, 3, 5, 6, 7])
        repeated = np.concatenate([self.real[self.labels == d] for d in chosen])
        c = self.component
        observed = (c['cross_sums'][chosen].sum() / c['cell_counts'][chosen].sum()
                    + c['generated_constant'])
        self.assertAlmostEqual(observed,
            direct(self.pg, repeated) - direct(self.other, repeated), places=12)

    def test_fixed_model_average_preserves_mean_of_distances(self):
        second = paired_components(self.pg + .2, self.other, self.real,
                                   self.labels, self.order)
        result = average_fixed_models([self.component, second])
        self.assertAlmostEqual(result['point_delta'],
            np.mean([self.component['point_delta'], second['point_delta']]), places=12)

    def test_identical_generators_have_exact_zero_contrast_and_interval(self):
        c = paired_components(self.pg, self.pg, self.real, self.labels, self.order)
        q = donor_cluster_interval([c] * 3, [np.repeat(np.arange(4), 2)] * 3,
                                  draws=1000)
        self.assertEqual((q['mean_delta'], q['lower'], q['upper']), (0., 0., 0.))

    def test_resampling_is_reproducible_and_reports_real_donor_units(self):
        stages = [np.repeat(np.arange(4), 2)] * 3
        a = donor_cluster_interval([self.component] * 3, stages, draws=1000)
        b = donor_cluster_interval([self.component] * 3, stages, draws=1000)
        self.assertEqual(a, b)
        self.assertFalse(a['independent_population_confirmation'])

    def test_missing_donor_and_incomplete_fold_or_stage_are_refused(self):
        with self.assertRaises(ValueError):
            paired_components(self.pg, self.other, self.real, self.labels,
                              np.arange(9))
        with self.assertRaises(ValueError):
            donor_cluster_interval([self.component] * 2, [np.arange(8)] * 2)
        with self.assertRaises(ValueError):
            donor_cluster_interval([self.component] * 3, [np.zeros(8)] * 3)

    def test_real_cell_order_permutation_preserves_paired_delta(self):
        indices = np.random.default_rng(7).permutation(len(self.real))
        c = paired_components(self.pg, self.other, self.real[indices],
                              self.labels[indices], self.order)
        np.testing.assert_allclose(c['cross_sums'], self.component['cross_sums'],
                                   rtol=0, atol=1e-12)
        self.assertAlmostEqual(c['point_delta'], self.component['point_delta'], places=12)


if __name__ == '__main__':
    unittest.main()
