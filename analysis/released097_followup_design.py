"""Prospective lower-rate/penalty follow-up; reserve the full comparison first."""
import math
from released097_selection import FORMULATION, FIELDS, require
from released097_external_selection import validate_summary

NOMINAL = dict(generator_lr=.012, critic_lr=.018, particle_lr=.03,
               reg_coeff=1., reg_kappa=1., prior_reg=.01)


def neighbors():
    result = []
    for cid, field in [('u000', 'reg_coeff'), ('u001', 'generator_lr')]:
        h = dict(NOMINAL)
        h[field] *= .5
        result.append(dict(id=cid, hyperparameters=h, changed_field=field,
                           multiplier=.5, endpoint=20000, training_seeds=[0, 1],
                           from_scratch=True, initial_completed_updates=0,
                           source_checkpoint=None))
    return result


def design(audit, now_unix, expires_unix, matrix_estimated_hours):
    require(audit['status'] == 'INDEPENDENT_RELEASED097_PAIRED_SELECTION_AUDIT_PASS',
            'Complete independent original12-fit audit required')
    require(audit['formulation'] == FORMULATION and audit['all_planned_attempts_resolved']
            and audit['all_new_models_from_update_zero'] and audit['activation_actually_reverted']
            and audit['owned_processes_and_leases_absent'] and not audit['heldout_values_read'],
            'Closed, fresh, calibration-only source evidence required')
    require(audit['training_seeds'] == [0, 1] and audit['required_attempts'] == 16
            and audit['required_new_attempts'] == 12,
            'All12 original new endpoints and four previous controls required')
    ranking = validate_summary(audit['summary'], [0, 1])
    by_id = {c['candidate_id']: c for c in audit['summary']['candidates']}
    require(set(by_id) == {'f000', 'f001', *[f't{i:03}' for i in range(6)]},
            'Original complete candidate universe required')
    require(by_id['f000']['hyperparameters'] == NOMINAL and set(NOMINAL) == FIELDS,
            'Unchanged released-reference anchor required')
    require(all(math.isfinite(v) for v in (now_unix, expires_unix, matrix_estimated_hours,
                                         audit['total_actual_GPU_seconds']))
            and expires_unix > now_unix and matrix_estimated_hours > 0,
            'Finite current window and full-matrix forecast required')
    finite = len(ranking) == len(by_id)
    ref = by_id['f000']
    joint_gain = finite and any(c['candidate_id'] != 'f000'
        and c['mean']['cell_energy_distance'] < ref['mean']['cell_energy_distance']
        and c['mean']['energy_score'] <= ref['mean']['energy_score'] for c in ranking)
    # This scalar forecast includes completed generation/scoring and retains the
    # maximum9 confirmation fits, not an assumed deduplication of choices.
    fit_seconds = 1.1 * audit['total_actual_GPU_seconds'] / 12
    require(fit_seconds > 0, 'Measured positive full-fit cost required')
    external = 1.1 * matrix_estimated_hours * 3600
    confirmation = 9 * fit_seconds
    cleanup = 2 * 3600
    qualification = 120
    slack = expires_unix - now_unix - external - confirmation - cleanup - qualification
    slots = max(0, min(2, int(slack // (2 * fit_seconds)))) if finite and not joint_gain else 0
    candidates = neighbors()[:slots]
    reason = ('Complete paired evidence has a joint distance/energy improvement; prioritize confirmation'
              if joint_gain else 'Nonfinite original trajectory; prioritize diagnosis/confirmation'
              if not finite else 'Useful complete paired follow-up fits reserved window'
              if candidates else 'No safe forecast slack; preserve confirmation and all45fits')
    return dict(status='PROPOSED_BOUNDED_RELEASED097_LOWER_DIRECTION_FOLLOWUP' if candidates
                else 'FOLLOWUP_SKIPPED_CONFIRMATION_AND_FULL45_PRIORITY', reason=reason,
                formulation=FORMULATION, anchor_candidate_id='f000', anchor_hyperparameters=NOMINAL,
                candidates=candidates, required_new_fits=2*len(candidates), fixed_endpoint=20000,
                selection_seeds=[0, 1], fresh_confirmation_seeds=[2, 3, 4],
                previously_measured_control_fits=16, controls_refitted=False,
                all_new_training_from_scratch=True, formulation_changed=False,
                external_required_fresh_models=45, external_scope_reduced=False,
                forecast=dict(remaining_window_seconds=expires_unix-now_unix,
                    per_calibration_fit_seconds=fit_seconds, external45_models_seconds=external,
                    fresh_confirmation_up_to9fits_seconds=confirmation,
                    handoff_and_cleanup_seconds=cleanup, scope_qualification_seconds=qualification,
                    new_search_seconds=2*len(candidates)*fit_seconds,
                    full_proposed_plan_forecast_fits_window=slack >= 2*len(candidates)*fit_seconds,
                    forecast_is_not_guarantee=True),
                automatic_activation=False, SOTA_claim=False, full_goal_complete=False,
                limitation='Two prespecified scalar tests at most: halve previously unvaried reg_coeff, '
                'then halve generator_lr in isolation from the nominal reference. Current12fit results '
                'must all close before this decision; no heldout values or new formulation are used.')
