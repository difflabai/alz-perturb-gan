"""Prospective adaptive-selection behavior, independent of model fitting."""
import copy
import math
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_distribution097b import derive

H = dict(generator_lr=.006, critic_lr=.009, particle_lr=.015,
         reg_coeff=1., reg_kappa=.5, prior_reg=.01)
def rows():
    return [dict(candidate_id=cid, finite=True, hyperparameters={**H, 'reg_kappa': k},
                 metrics=dict(cell_energy_distance=d, energy_score=e, spread_ratio=s))
            for cid, k, d, e, s in [('d000', .5, .22, 1.95, 1.),
                                    ('d001', .5, .25, 1.89, .9),
                                    ('d002', .25, .3, 1.9, 1.)]]
HISTORY = [dict(finite=True, hyperparameters={**H, 'reg_kappa': 1.})]

class Selection(unittest.TestCase):
    def test_one_scalar_midpoint_and_joint_rate_changes_preserve_other_parameters(self):
        r = rows()
        original = copy.deepcopy(r)
        anchor, proposals = derive(r, HISTORY)
        self.assertEqual(anchor['candidate_id'], 'd000')
        self.assertEqual(r, original)
        for p, factor in zip(proposals[:2], (.5, 1.5)):
            for key in H:
                target = H[key] * factor if key.endswith('_lr') else H[key]
                self.assertEqual(p['hyperparameters'][key], target)
        self.assertEqual(proposals[2]['hyperparameters'], {**H, 'reg_kappa': math.sqrt(.5)})

    def test_primary_distance_winner_is_not_silently_replaced_by_better_energy(self):
        r = rows()
        r[1]['metrics'].update(cell_energy_distance=.18, energy_score=2.2)
        anchor, _ = derive(r, HISTORY)
        self.assertEqual(anchor['candidate_id'], 'd001')

    def test_nearest_larger_kappa_and_spread_guard_not_arbitrary_upper(self):
        r = rows()
        r[2]['metrics']['cell_energy_distance'] = .2
        anchor, proposals = derive(r, HISTORY)
        self.assertEqual(anchor['candidate_id'], 'd002')
        self.assertEqual(proposals[2]['hyperparameters']['reg_kappa'], math.sqrt(.25 * .5))
        r[2]['metrics']['spread_ratio'] = .1
        self.assertEqual(derive(r, HISTORY)[0]['candidate_id'], 'd000')

    def test_invalid_endpoints_failed_or_collapsed_search_cannot_make_blind_schedule(self):
        r = rows()
        r[0]['finite'] = False
        self.assertEqual(derive(r, HISTORY)[0]['candidate_id'], 'd001')
        for row in r:
            row['finite'] = False
        with self.assertRaises(AssertionError):
            derive(r, HISTORY)
        with self.assertRaises(AssertionError):
            derive(rows()[:2], HISTORY)
        r = rows()
        r[0]['metrics']['cell_energy_distance'] = float('nan')
        with self.assertRaises(AssertionError):
            derive(r, HISTORY)

    def test_exact_previously_measured_vector_is_refused(self):
        previous = copy.deepcopy(HISTORY)
        h = dict(H)
        for key in ('generator_lr', 'critic_lr', 'particle_lr'):
            h[key] *= .5
        previous.append(dict(finite=True, hyperparameters=h))
        with self.assertRaises(AssertionError):
            derive(rows(), previous)

if __name__ == '__main__':
    unittest.main(verbosity=2)
