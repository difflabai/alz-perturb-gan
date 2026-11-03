import copy
import unittest
from selection import decide


class SelectionContract(unittest.TestCase):
    def setUp(self):
        self.incumbent = dict(status='completed', completed_updates=20000,
                              metric_role='calibration_selection', donors=['a', 'b', 'c', 'd'],
                              generation_replicates=2, training_seed=0,
                              calibration_manifest_SHA256='a' * 64,
                              trace_manifest_SHA256='b' * 64,
                              sampling_contract='calibration_equal_stage375_priority_v1',
                              energy_score=2.0, cell_energy_distance=0.25,
                              spread_ratio=1.0, generated_zero_count_rate=0,
                              integer_count_contract_passed=True)
        self.candidate = copy.deepcopy(self.incumbent)
        self.candidate.update(energy_score=1.8, cell_energy_distance=0.2)

    def test_accepts_joint_improvement(self):
        self.assertTrue(decide(self.candidate, self.incumbent)['accept'])

    def test_mean_improvement_cannot_hide_distribution_regression(self):
        self.candidate['cell_energy_distance'] = 0.26
        self.assertFalse(decide(self.candidate, self.incumbent)['accept'])

    def test_better_scores_cannot_hide_donor_collapse(self):
        self.candidate['spread_ratio'] = 0.3
        self.assertFalse(decide(self.candidate, self.incumbent)['accept'])

    def test_incomplete_checkpoint_cannot_be_selected(self):
        self.candidate['completed_updates'] = 19000
        self.assertFalse(decide(self.candidate, self.incumbent)['accept'])

    def test_evaluation_scores_cannot_select_candidates(self):
        self.candidate['metric_role'] = 'evaluation'
        self.assertFalse(decide(self.candidate, self.incumbent)['accept'])

    def test_changed_draws_or_targets_cannot_rescue_a_candidate(self):
        for key in ['training_seed', 'donors', 'calibration_manifest_SHA256', 'trace_manifest_SHA256']:
            candidate = copy.deepcopy(self.candidate)
            candidate[key] = 'different'
            self.assertFalse(decide(candidate, self.incumbent)['accept'])

    def test_equal_result_is_not_an_ath(self):
        self.assertFalse(decide(self.incumbent, self.incumbent)['accept'])

    def test_nonfinite_or_unverified_counts_are_rejected(self):
        self.candidate['energy_score'] = float('nan')
        self.assertFalse(decide(self.candidate, self.incumbent)['accept'])
        self.candidate['energy_score'] = 1.8
        self.candidate['integer_count_contract_passed'] = False
        self.assertFalse(decide(self.candidate, self.incumbent)['accept'])


if __name__ == '__main__':
    unittest.main()
