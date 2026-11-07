"""Validate the full calibration selection chain before external PG freezing.

Pure metadata/arithmetic checks. The caller separately verifies artifact hashes
and closed resources; this function never loads evaluation arrays or models.
"""
import math
FIELDS={'generator_lr','critic_lr','particle_lr','reg_coeff','reg_kappa','prior_reg'}
METRICS=('cell_energy_distance','energy_score','spread_ratio','within_condition_spread_ratio','coverage80')
def require(condition,message):
 if not condition:raise ValueError(message)
def same_summary(audit,report):
 for key,value in audit['summary'].items():require(report[key]==value,'Audited and reported selection differ: '+key)
def resolve(paired_audit,paired_report,followup_audit,followup_report,followup_protocol):
 require(paired_audit['status']=='INDEPENDENT_NATIVE097_PAIRED_REFINEMENT20_FITS_AUDIT_PASS','Complete native paired audit required')
 require(paired_audit['required_attempts']==20 and paired_audit['training_seeds']==[0,1] and paired_audit['all_planned_attempts_resolved'],'All20 paired selection endpoints required')
 require(paired_audit['all_new_models_from_update_zero'] and paired_audit['activation_actually_reverted'] and paired_audit['owned_processes_and_leases_absent'] and not paired_audit['heldout_values_read'],'Fresh paired selection and closed resources required')
 same_summary(paired_audit,paired_report)
 require(followup_audit['status']=='INDEPENDENT_NATIVE097_MATCHED_REFINEMENT_FOLLOWUP_AUDIT_PASS','Complete native matched followup audit required')
 require(followup_audit['training_seeds']==[2,3,4] and followup_audit['all_planned_attempts_resolved'] and followup_audit['all_new_models_from_update_zero'],'All matched seed2..4 outcomes required')
 require(followup_audit['activation_actually_reverted'] and followup_audit['owned_processes_and_leases_absent'] and not followup_audit['heldout_values_read'],'Followup must close before external freeze')
 same_summary(followup_audit,followup_report)
 p=followup_protocol
 require(paired_audit['native_version']==followup_audit['native_version']==p['native_version']=='0.9.7' and paired_audit['EMA_decay']==followup_audit['EMA_decay']==0 and not paired_audit['formulation_changed'] and not followup_audit['formulation_changed'],'Released nonEMA formulation must stay fixed')
 require(p['endpoint']==20000 and p['calibration_only'] and p['metric_role']=='calibration_selection','Full20k calibration endpoint required')
 require(p['status']=='FROZEN_MATCHED_SEED234_FOLLOWUP_AFTER_PAIRED_REFINEMENT' and p['training_seeds']==[2,3,4] and p['selection_seeds']==[0,1],'Frozen paired-selection followup protocol required')
 require(p['all_new_training_from_scratch'] and p['initial_completed_updates']==0 and p['source_checkpoint'] is None and not p['reference_models_refitted'] and p['reference_measurements_previously_observed'],'No old-weight continuation or relabeled reference fits')
 choices=paired_report['proposed_followup_configurations'];require(p['all_configurations_for_ranking']==choices,'Followup choices differ from paired selection')
 new=[c for c in choices if c['candidate_id']!='d000'];require(p['selected_configurations']==new and 1<=len(new)<=2,'Exact nonreference followup choices required')
 require(followup_audit['required_attempts']==3*len(new)==followup_report['required_new_fits'],'All frozen new fits required')
 require(followup_report['reference_new_fits']==followup_audit['reference_new_fits']==0 and followup_report['reference_previously_observed'],'Previously observed reference must stay labeled')
 require(followup_report['ranking_seed_set']==[2,3,4] and followup_report['selection_seeds0_and1_excluded_from_followup_ranking'] and not followup_report['heldout_used'],'Only declared followup seeds may rank')
 expected={c['candidate_id']:c['hyperparameters'] for c in choices};entries=followup_report['candidates']
 require(len(entries)==len(expected) and {c['candidate_id'] for c in entries}==set(expected),'Every selected setting must be present once')
 ranked=[];baselines={}
 for c in entries:
  h=c['hyperparameters'];require(h==expected[c['candidate_id']] and set(h)==FIELDS and all(type(v) in (int,float) and math.isfinite(v) and v>0 for v in h.values()),'Frozen six-scalar settings differ')
  rows=sorted(c['per_seed'],key=lambda r:r['seed']);require(len(rows)==3 and [r['seed'] for r in rows]==[2,3,4],'Every followup seed must be retained once')
  finite=all(r['finite'] for r in rows);require(c['all_three_followup_seeds_finite']==finite,'Failed-seed eligibility differs')
  require(c['failed_seeds']==[r['seed'] for r in rows if not r['finite']],'Failed seeds cannot be dropped')
  require(c['reference_measurements_previously_observed']==(c['candidate_id']=='d000'),'Reference provenance differs')
  for r in rows:
   baseline=r['baseline_metrics']
   if r['seed'] in baselines:require(baselines[r['seed']]==baseline,'Resampling controls changed across configurations')
   baselines[r['seed']]=baseline
   if not r['finite']:continue
   for metric in (r['metrics'],baseline):
    require(metric['training_seed']==r['seed'] and metric['metric_role']=='calibration_selection' and metric['generation_replicates']==2 and all(math.isfinite(metric[k]) for k in METRICS),'Complete finite calibration records required')
   for key in ('donors','calibration_manifest_SHA256','trace_manifest_SHA256','sampling_contract'):require(r['metrics'][key]==baseline[key],'PG and comparator target/draw contracts differ')
  if not finite:continue
  mean={k:sum(r['metrics'][k] for r in rows)/3 for k in METRICS}
  sd={k:math.sqrt(sum((r['metrics'][k]-mean[k])**2 for r in rows)/2) for k in METRICS}
  base_mean={k:sum(r['baseline_metrics'][k] for r in rows)/3 for k in METRICS}
  for key in METRICS:
   require(math.isclose(c['followup_mean'][key],mean[key],rel_tol=0,abs_tol=1e-12) and math.isclose(c['followup_SD'][key],sd[key],rel_tol=0,abs_tol=1e-12),'Followup mean or sampleSD differs')
   require(math.isclose(c['mean_matched_resampling'][key],base_mean[key],rel_tol=0,abs_tol=1e-12),'Mean matched control differs')
  joint=all(.7<=r['metrics']['spread_ratio']<=1.3 for r in rows) and mean['energy_score']<=base_mean['energy_score'];require(c['joint_quality']==joint,'Joint-quality tradeoff differs')
  ranked.append(c)
 require(ranked,'No fully finite setting supports an external candidate')
 ranked.sort(key=lambda c:(c['followup_mean']['cell_energy_distance'],c['followup_mean']['energy_score'],c['candidate_id']))
 require(followup_report['ranking_by_followup_mean_distance']==[c['candidate_id'] for c in ranked],'Canonical followup ranking differs')
 require(followup_report['proposed_external_candidate']==ranked[0]['candidate_id'],'Candidate must minimize the declared followup mean')
 best=ranked[0]
 return dict(candidate_id=best['candidate_id'],hyperparameters=best['hyperparameters'],followup_mean=best['followup_mean'],followup_SD=best['followup_SD'],
   mean_matched_resampling=best['mean_matched_resampling'],joint_quality=best['joint_quality'],selection_seeds=[0,1],ranking_seeds=[2,3,4],
   reference_measurements_previously_observed=True,required_new_followup_fits=3*len(new),all45_new_external_models_required=True,
   independent_population_confirmation=False,SOTA_claim=False,full_goal_complete=False)
