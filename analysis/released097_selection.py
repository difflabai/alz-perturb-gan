"""Full-endpoint calibration ranking for corrected released PG, with no model I/O."""
import math
FIELDS={'generator_lr','critic_lr','particle_lr','reg_coeff','reg_kappa','prior_reg'}
METRICS=('cell_energy_distance','energy_score','spread_ratio','within_condition_spread_ratio','coverage80')
FORMULATION=dict(native_version='0.9.7',recipe='bispf_optimized',EMA_decay=0,generator_gradient_response='none',prior_regularizer_domain='all512rows',critic_thermal='explicitly_bound',penalty_input_domain='continuous')
def require(value,message):
 if not value:raise ValueError(message)
def summarize(candidates,results,seeds):
 require(seeds in ([0,1],[2,3,4]),'Declared selection or confirmation seeds required')
 require(candidates and len({c['candidate_id'] for c in candidates})==len(candidates),'Distinct declared candidates required')
 expected={(c['candidate_id'],seed) for c in candidates for seed in seeds}
 require(all(type(r['seed']) is int and r['seed'] in seeds for r in results),'Only declared integer seeds accepted')
 require(len(results)==len(expected) and {(r['candidate_id'],r['seed']) for r in results}==expected,'Every planned candidate and seed outcome required exactly once')
 baselines={};entries=[]
 for candidate in candidates:
  cid=candidate['candidate_id'];h=candidate['hyperparameters']
  require(set(h)==FIELDS and all(type(v) in (int,float) and math.isfinite(v) and v>0 for v in h.values()),'Six finite positive released scalar settings required')
  rows=sorted([r for r in results if r['candidate_id']==cid],key=lambda r:r['seed'])
  for r in rows:
   require(r['hyperparameters']==h and r['endpoint']==20000 and r['from_scratch'] and r['initial_completed_updates']==0 and not r.get('resumed_same_trial',False),'Full20k target and fresh initialization required')
   require(r['calibration_only'] and r['clean_settlement'] and r['owned_processes_and_lease_absent'],'Only audited closed calibration attempts may rank')
   if not r['finite']:
    require(r['failed_candidate_not_retried'] and r['failure']['error'].startswith('FloatingPointError('),'Preserve scientific failure; operational stops cannot complete selection');continue
   require(r['physical_start_update']==0 and r['new_phase_model_updates']==20000,'Every finite result must reach20k from zero')
   m,base=r['metrics'],r['baseline_metrics']
   for record in (m,base):
    require(record['training_seed']==r['seed'] and record['metric_role']=='calibration_selection' and record['generation_replicates']==2 and all(type(record[k]) in (int,float) and math.isfinite(record[k]) for k in METRICS),'Complete finite matched calibration metrics required')
   for key in ('donors','calibration_manifest_SHA256','trace_manifest_SHA256','sampling_contract'):require(m[key]==base[key],'Targets or generation contracts differ')
   if r['seed'] in baselines:require(baselines[r['seed']]==base,'Matched controls changed between settings')
   baselines[r['seed']]=base
  eligible=all(r['finite'] for r in rows)
  entry=dict(candidate_id=cid,hyperparameters=h,seeds=seeds,all_seeds_finite=eligible,failed_seeds=[r['seed'] for r in rows if not r['finite']],per_seed=[dict(seed=r['seed'],finite=r['finite'],metrics=r.get('metrics'),baseline_metrics=r.get('baseline_metrics')) for r in rows])
  if eligible:
   entry['mean']={k:sum(r['metrics'][k] for r in rows)/len(seeds) for k in METRICS}
   entry['SD']={k:math.sqrt(sum((r['metrics'][k]-entry['mean'][k])**2 for r in rows)/(len(seeds)-1)) for k in METRICS}
   entry['mean_matched_resampling']={k:sum(r['baseline_metrics'][k] for r in rows)/len(seeds) for k in METRICS}
   entry['joint_quality']=all(.7<=r['metrics']['spread_ratio']<=1.3 for r in rows) and entry['mean']['energy_score']<=entry['mean_matched_resampling']['energy_score']
  entries.append(entry)
 ranked=sorted([c for c in entries if c['all_seeds_finite']],key=lambda c:(c['mean']['cell_energy_distance'],c['mean']['energy_score'],c['candidate_id']))
 return dict(candidates=entries,ranking=[c['candidate_id'] for c in ranked],ranking_seeds=seeds,proposed_external_candidate=ranked[0]['candidate_id'] if ranked and seeds==[2,3,4] else None,all_planned_attempts_resolved=True,failed_seeds_not_dropped=True,heldout_used=False,independent_population_confirmation=False,SOTA_claim=False,full_goal_complete=False)
def proposed_confirmation(summary,reference_id):
 require(summary['ranking_seeds']==[0,1],'Confirmation choices must precede seed2..4 results')
 ranked=[next(c for c in summary['candidates'] if c['candidate_id']==cid) for cid in summary['ranking']]
 joint=[c for c in ranked if c['joint_quality']];ref=next((c for c in ranked if c['candidate_id']==reference_id),None);selected=[]
 for role,c in [('primary_distance',ranked[0] if ranked else None),('joint_quality',joint[0] if joint else None),('released_reference',ref)]:
  if c is None:continue
  q=next((x for x in selected if x['candidate_id']==c['candidate_id']),None)
  if q is None:q=dict(candidate_id=c['candidate_id'],hyperparameters=c['hyperparameters'],roles=[]);selected.append(q)
  q['roles'].append(role)
 return selected
