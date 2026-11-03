"""Validate analytic identities against direct enumeration on synthetic points."""
import itertools
import unittest
import numpy as np
from scipy.spatial.distance import cdist
from diagnose_pure_ensemble import donor_complementarity, expected_fixed_mask_score, cell_complementarity


class EnergyIdentities(unittest.TestCase):
    def test_fixed_mask_expectation_matches_every_synthetic_mask_directly(self):
        p = np.array([[0., 1.], [1., 3.], [2., -1.], [4., 2.]])
        q = np.array([[1., 2.], [-2., 1.], [3., 4.], [2., -3.]])
        target = np.array([[1.3, -.4]])
        def score(x):
            return cdist(x, target).mean() - .5 * cdist(x, x).sum() / (len(x) * (len(x) - 1))
        for count in range(5):
            scores = []
            for selected in itertools.combinations(range(4), count):
                mask = np.zeros(4, dtype=bool); mask[list(selected)] = True
                scores.append(score(np.where(mask[:, None], q, p)))
            predicted = expected_fixed_mask_score(score(p), score(q), donor_complementarity(p, q), 4, count)
            self.assertAlmostEqual(predicted, np.mean(scores), places=12)
        self.assertAlmostEqual(donor_complementarity(p, p), 0., places=12)

    def test_weighted_empirical_cell_identity_matches_direct_weighted_measure(self):
        p = np.array([[0.], [2.]]); q = np.array([[-1.], [4.], [5.]]); target = np.array([[1.], [3.]])
        def energy(a):
            return 2 * cdist(a, target).mean() - cdist(a, a).mean() - cdist(target, target).mean()
        points = np.concatenate([p, q]); alpha = .25
        weights = np.r_[np.full(len(p), (1 - alpha) / len(p)), np.full(len(q), alpha / len(q))]
        direct = 2 * (weights * cdist(points, target).mean(1)).sum() - weights @ cdist(points, points) @ weights - cdist(target, target).mean()
        predicted = (1 - alpha) * energy(p) + alpha * energy(q) - alpha * (1 - alpha) * cell_complementarity(p, q)
        self.assertAlmostEqual(predicted, direct, places=12)


if __name__ == '__main__':
    unittest.main()
