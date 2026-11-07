"""Prospective full20k coordinate search after both corrected baseline settings.

Every proposed fit starts zero. Existing baseline scores are labeled measured
controls. No stored model initializes a new trajectory. Time forecasts reserve the
entire45fresh-model external comparison and fresh seed2..4 confirmation first.
"""
import math
from released097_selection import FORMULATION,FIELDS,require
from released097_external_selection import validate_summary

def coordinate_neighbors(anchor_id,h):
 require(anchor_id in ('f000','f001') and set(h)==FIELDS,'Qualified baseline coordinate anchor required')
 factor=.5 if anchor_id=='f001' else 2.
 neighbors=[('prior_reg',.3),('prior_reg',3.),('reg_kappa',factor),('generator_lr',factor),('critic_lr',factor),('particle_lr',factor)]
 rows=[]
 for i,(field,multiplier) in enumerate(neighbors):
  value=dict(h);value[field]*=multiplier
  rows.append(dict(id=f't{i:03}',hyperparameters=value,changed_field=field,multiplier=multiplier,endpoint=20000,training_seeds=[0,1],from_scratch=True,initial_completed_updates=0,source_checkpoint=None))
 return factor,rows

def design(audit,now_unix,expires_unix,matrix_estimated_hours):
 require(audit['status']=='INDEPENDENT_RELEASED097_FRESH_BASELINE4_AUDIT_PASS','Complete independent corrected four-fit baseline audit required')
 require(audit['formulation']==FORMULATION and audit['all_planned_attempts_resolved'] and audit['all_new_models_from_update_zero'] and audit['activation_actually_reverted'] and audit['owned_processes_and_leases_absent'] and not audit['heldout_values_read'],'Fresh closed calibration evidence required')
 require(audit['training_seeds']==[0,1] and audit['required_attempts']==4,'Both initial settings and both exploratory seeds required')
 ranking=validate_summary(audit['summary'],[0,1]);require(ranking,'No finite baseline supports this local search; diagnose failures')
 by_id={c['candidate_id']:c for c in audit['summary']['candidates']};require(set(by_id)=={'f000','f001'},'Exact corrected nominal and half-rate/kappa controls required')
 expected_nominal=dict(generator_lr=.012,critic_lr=.018,particle_lr=.03,reg_coeff=1.,reg_kappa=1.,prior_reg=.01)
 expected_lower=dict(expected_nominal,generator_lr=.006,critic_lr=.009,particle_lr=.015,reg_kappa=.5)
 require(by_id['f000']['hyperparameters']==expected_nominal and by_id['f001']['hyperparameters']==expected_lower,'Qualified baseline scalar settings differ')
 require(all(c['all_seeds_finite'] for c in by_id.values()),'Diagnose incomplete or nonfinite baseline trajectories before extrapolating their average full-fit cost')
 anchor=ranking[0];h=anchor['hyperparameters'];require(set(h)==FIELDS,'Six-scalar baseline required')
 require(all(math.isfinite(x) for x in (now_unix,expires_unix,matrix_estimated_hours,audit['total_actual_GPU_seconds'])) and expires_unix>now_unix and matrix_estimated_hours>0,'Finite current window and forecast required')
 # Both baseline settings differ jointly in G/D/P rates and kappa. Extrapolation
 # is a testable pattern-search hypothesis, never a causal claim for each field.
 both_finite=all(c['all_seeds_finite'] for c in by_id.values())
 factor,neighbors=coordinate_neighbors(anchor['candidate_id'],h)
 # Measured per-fit costs include generation/scoring. This conservative mean
 # forecast is not admission, a reservation, or a runtime guarantee.
 cal_fit_seconds=1.1*audit['total_actual_GPU_seconds']/4
 require(cal_fit_seconds>0,'Positive measured GPU cost required')
 external_seconds=1.1*matrix_estimated_hours*3600
 confirmation_seconds=9*cal_fit_seconds
 handoff_and_cleanup_seconds=2*3600
 available=expires_unix-now_unix-external_seconds-confirmation_seconds-handoff_and_cleanup_seconds
 slots=max(0,min(6,int(available//(2*cal_fit_seconds))))
 # A prior-weight bracket is the minimum useful local study. Otherwise proceed
 # to baseline confirmation instead of shortening endpoints or external scope.
 if slots<2:slots=0
 proposals=neighbors[:slots]
 return dict(status='PROPOSED_WINDOW_BOUNDED_LOCAL_COORDINATE_SEARCH' if proposals else 'INSUFFICIENT_FORECAST_SLACK_FOR_LOCAL_SEARCH_BASELINE_CONFIRMATION_PRIORITY',formulation=FORMULATION,
   anchor_candidate_id=anchor['candidate_id'],anchor_hyperparameters=h,anchor_selection='Complete two-seed mean cell energy distance, then donor energy score',
   directional_factor_for_rates_and_kappa=factor,directional_hypothesis_not_causal=both_finite,candidates=proposals,required_new_fits=2*len(proposals),
   measured_baseline_controls_previously_observed=True,existing_baseline_models_refitted=False,all_new_training_from_scratch=True,
   selection_seeds=[0,1],fresh_confirmation_seeds=[2,3,4],fixed_endpoint=20000,formulation_changed=False,
   forecast=dict(base_plan_forecast_fits_window=available>=0,full_proposed_plan_forecast_fits_window=available>=2*len(proposals)*cal_fit_seconds,remaining_window_seconds=expires_unix-now_unix,per_calibration_fit_seconds=cal_fit_seconds,external45_models_seconds=external_seconds,fresh_confirmation_up_to9fits_seconds=confirmation_seconds,handoff_and_cleanup_seconds=handoff_and_cleanup_seconds,new_search_seconds=2*len(proposals)*cal_fit_seconds,forecast_is_not_guarantee=True),
   external_required_fresh_models=45,external_scope_reduced=False,automatic_activation=False,SOTA_claim=False,full_goal_complete=False,
   limitation='One local direction per G/D/P rate and kappa plus a two-sided prior-weight bracket. Critic reg_coeff stays fixed. Selection remains adaptive on the four calibration donors; all proposed settings require native CPU qualification and real endpoints.')
