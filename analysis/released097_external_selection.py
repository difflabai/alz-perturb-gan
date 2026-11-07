"""Reject historical sampled-row variants before freezing a released-PG benchmark.

This is a metadata validator. Callers must additionally verify hashes, immutable
source, clean closure and the independent raw-score/checkpoint audits.
"""
from released097_selection import FORMULATION,FIELDS,METRICS,require,proposed_confirmation
import math

def validate_summary(summary,seeds):
 require(summary['ranking_seeds']==seeds and summary['all_planned_attempts_resolved'] and summary['failed_seeds_not_dropped'] and not summary['heldout_used'],'Complete calibration selection required')
 require(not summary['SOTA_claim'] and not summary['full_goal_complete'],'Calibration is not external superiority')
 candidates=summary['candidates'];require(candidates and len({c['candidate_id'] for c in candidates})==len(candidates),'Every candidate must be present once')
 ranked=[];baselines={}
 for c in candidates:
  h=c['hyperparameters'];require(set(h)==FIELDS and all(type(v) in (int,float) and math.isfinite(v) and v>0 for v in h.values()),'Six frozen positive native scalars required')
  rows=c['per_seed'];require(len(rows)==len(seeds) and [r['seed'] for r in rows]==seeds and all(type(r['seed']) is int for r in rows),'All declared seeds retained exactly once')
  finite=all(r['finite'] for r in rows);require(c['all_seeds_finite']==finite and c['failed_seeds']==[r['seed'] for r in rows if not r['finite']],'No failed-seed dropping')
  if not finite:continue
  for r in rows:
   m,b=r['metrics'],r['baseline_metrics']
   for x in (m,b):require(x['training_seed']==r['seed'] and x['generation_replicates']==2 and x['metric_role']=='calibration_selection' and all(math.isfinite(x[k]) for k in METRICS),'Two finite matched calibration replicates required')
   for key in ('donors','calibration_manifest_SHA256','trace_manifest_SHA256','sampling_contract'):require(m[key]==b[key],'Target or draw contract differs')
   if r['seed'] in baselines:require(baselines[r['seed']]==b,'Controls differ between settings')
   baselines[r['seed']]=b
  means={k:sum(r['metrics'][k] for r in rows)/len(seeds) for k in METRICS}
  sd={k:math.sqrt(sum((r['metrics'][k]-means[k])**2 for r in rows)/(len(seeds)-1)) for k in METRICS}
  base={k:sum(r['baseline_metrics'][k] for r in rows)/len(seeds) for k in METRICS}
  for k in METRICS:
   for observed,want in ((c['mean'][k],means[k]),(c['SD'][k],sd[k]),(c['mean_matched_resampling'][k],base[k])):require(math.isclose(observed,want,rel_tol=0,abs_tol=1e-12),'Audited mean, SD or control differs')
  joint=all(.7<=r['metrics']['spread_ratio']<=1.3 for r in rows) and means['energy_score']<=base['energy_score'];require(c['joint_quality']==joint,'Joint quality differs')
  ranked.append(c)
 ranked.sort(key=lambda c:(c['mean']['cell_energy_distance'],c['mean']['energy_score'],c['candidate_id']))
 require(summary['ranking']==[c['candidate_id'] for c in ranked],'Ranking must use complete declared-seed distance, then energy score')
 return ranked

def resolve(selection_audit,confirmation_audit,confirmation_protocol):
 s,c,p=selection_audit,confirmation_audit,confirmation_protocol
 require(s['status'] in ('INDEPENDENT_RELEASED097_FRESH_BASELINE4_AUDIT_PASS','INDEPENDENT_RELEASED097_PAIRED_SELECTION_AUDIT_PASS'),'Released-objective exploratory audit required')
 require(c['status']=='INDEPENDENT_RELEASED097_MATCHED_CONFIRMATION_AUDIT_PASS','Fresh released-objective confirmation audit required')
 for audit,seeds in ((s,[0,1]),(c,[2,3,4])):
  require(audit['formulation']==FORMULATION,'Full-table SBNP released formulation and bound thermal record required')
  require(audit['training_seeds']==seeds and audit['all_planned_attempts_resolved'] and audit['all_new_models_from_update_zero'],'Every fresh initialization outcome required')
  require(audit['activation_actually_reverted'] and audit['owned_processes_and_leases_absent'] and not audit['heldout_values_read'] and not audit['CUDA_initialized'],'Closed independent calibration audits required')
  require(audit['model_parity_proof_path']==s['model_parity_proof_path'] and audit['model_parity_proof_SHA256']==s['model_parity_proof_SHA256'],'Same qualified native update source required')
  validate_summary(audit['summary'],seeds)
 require(p['status']=='FROZEN_RELEASED097_MATCHED_SEED234_CONFIRMATION' and p['native_version']=='0.9.7' and p['endpoint']==20000 and p['training_seeds']==[2,3,4] and p['selection_seeds']==[0,1],'Frozen full20k confirmation protocol required')
 require(p['all_new_training_from_scratch'] and p['initial_completed_updates']==0 and p['source_checkpoint'] is None and p['calibration_only'] and p['metric_role']=='calibration_selection','No warm starts or evaluation-driven confirmation')
 require(p['formulation']==FORMULATION and not p['automatic_promotion'] and not p['SOTA_claim'],'Released formulation must stay fixed')
 require(s['required_attempts']==2*len(s['summary']['candidates']),'All exploratory candidate/seed attempts required')
 choices=s['summary']['proposed_confirmation_configurations'];require(choices==proposed_confirmation(s['summary'],s['summary']['reference_id']),'Confirmation roles differ from declared distance, joint and released-reference rules');require(p['selected_configurations']==choices and choices,'Confirmation choices must be frozen from audited selection')
 require(c['required_attempts']==3*len(choices),'All selected settings need all three fresh confirmation seeds')
 actual={r['candidate_id']:r['hyperparameters'] for r in c['summary']['candidates']};require(actual=={r['candidate_id']:r['hyperparameters'] for r in choices},'No post hoc confirmation replacement or omissions')
 ranked=validate_summary(c['summary'],[2,3,4]);require(ranked,'No fully finite configuration can be frozen')
 best=ranked[0];require(c['summary']['proposed_external_candidate']==best['candidate_id'],'Freeze must follow fresh confirmation ranking')
 return dict(candidate_id=best['candidate_id'],hyperparameters=best['hyperparameters'],confirmation_mean=best['mean'],confirmation_SD=best['SD'],mean_matched_resampling=best['mean_matched_resampling'],joint_quality=best['joint_quality'],selection_seeds=[0,1],confirmation_seeds=[2,3,4],all45_new_external_models_required=True,independent_population_confirmation=False,SOTA_claim=False)
