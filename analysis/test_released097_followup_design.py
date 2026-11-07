"""Synthetic full-scope, direction, stopping and time-reserve checks."""
import copy
import unittest
from released097_selection import FORMULATION, summarize
from released097_followup_design import NOMINAL, design, neighbors
from released097_local_search_design import coordinate_neighbors
from test_released097_selection import fixture


def audit(joint=False):
    _, source = fixture()
    choices = [dict(candidate_id='f000', hyperparameters=dict(NOMINAL)),
               dict(candidate_id='f001', hyperparameters=dict(NOMINAL,
                    generator_lr=.006, critic_lr=.009, particle_lr=.015, reg_kappa=.5))]
    choices += [dict(candidate_id=c['id'], hyperparameters=c['hyperparameters'])
                for c in coordinate_neighbors('f000', NOMINAL)[1]]
    rows = []
    for c in choices:
        for original in source[:2]:
            r = copy.deepcopy(original)
            r.update(candidate_id=c['candidate_id'], hyperparameters=c['hyperparameters'])
            r['metrics']['cell_energy_distance'] = .30 if c['candidate_id']=='f000' else .31
            r['metrics']['energy_score'] = 2. if c['candidate_id']=='f000' else 2.05
            if joint and c['candidate_id']=='t005':
                r['metrics'].update(cell_energy_distance=.29, energy_score=1.99)
            rows.append(r)
    return dict(status='INDEPENDENT_RELEASED097_PAIRED_SELECTION_AUDIT_PASS',
        formulation=FORMULATION, training_seeds=[0,1], required_attempts=16,
        required_new_attempts=12, all_planned_attempts_resolved=True,
        all_new_models_from_update_zero=True, activation_actually_reverted=True,
        owned_processes_and_leases_absent=True, heldout_values_read=False,
        summary=summarize(choices,rows,[0,1]), total_actual_GPU_seconds=12*1450.)


class FollowupDesign(unittest.TestCase):
    def test_only_two_new_scalar_settings_from_fresh_zero(self):
        q=design(audit(),0,60*3600,34.3)
        self.assertEqual(q['required_new_fits'],4)
        self.assertEqual([c['changed_field'] for c in q['candidates']],['reg_coeff','generator_lr'])
        for c in q['candidates']:
            self.assertEqual(sum(c['hyperparameters'][k]!=v for k,v in NOMINAL.items()),1)
            self.assertEqual(c['endpoint'],20000);self.assertEqual(c['training_seeds'],[0,1])
            self.assertEqual(c['initial_completed_updates'],0);self.assertIsNone(c['source_checkpoint'])
        self.assertEqual(neighbors()[0]['hyperparameters']['reg_coeff'],.5)
        self.assertEqual(neighbors()[1]['hyperparameters']['generator_lr'],.006)
    def test_all45_and_worst9_confirmation_and_cleanup_reserved(self):
        q=design(audit(),0,46*3600,34.3);f=q['forecast']
        self.assertEqual(q['external_required_fresh_models'],45)
        self.assertFalse(q['external_scope_reduced'])
        self.assertLessEqual(sum(f[k] for k in ('external45_models_seconds',
            'fresh_confirmation_up_to9fits_seconds','handoff_and_cleanup_seconds',
            'scope_qualification_seconds','new_search_seconds')),f['remaining_window_seconds'])
    def test_one_setting_pair_allowed_but_partial_seed_pair_never_selected(self):
        a=audit();fit=1.1*1450;reserve=1.1*34.3*3600+9*fit+7200+120
        q=design(a,0,reserve+3*fit,34.3)
        self.assertEqual(q['required_new_fits'],2);self.assertEqual(q['candidates'][0]['id'],'u000')
    def test_insufficient_time_skips_instead_of_shrinking_external_scope(self):
        q=design(audit(),0,42*3600,34.3)
        self.assertEqual(q['required_new_fits'],0);self.assertFalse(q['external_scope_reduced'])
    def test_already_jointly_promising_result_prioritizes_confirmation(self):
        q=design(audit(joint=True),0,60*3600,34.3)
        self.assertEqual(q['required_new_fits'],0);self.assertIn('joint',q['reason'])
    def test_incomplete_contaminated_or_wrong_formulation_rejected(self):
        for k,v in [('status','OLD_ADAPTER'),('required_new_attempts',10),
                    ('all_planned_attempts_resolved',False),('heldout_values_read',True),
                    ('formulation','sampled_rows')]:
            a=audit();a[k]=v
            with self.assertRaises(ValueError):design(a,0,60*3600,34.3)
    def test_no_activation_refit_or_superiority_claim(self):
        q=design(audit(),0,60*3600,34.3)
        self.assertFalse(q['automatic_activation']);self.assertFalse(q['controls_refitted'])
        self.assertFalse(q['SOTA_claim']);self.assertFalse(q['full_goal_complete'])


if __name__=='__main__':unittest.main(verbosity=2)
