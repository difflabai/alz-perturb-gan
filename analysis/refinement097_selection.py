"""Paired initialization selection over existing PG scalars, no model I/O."""
import math
FIELDS={'generator_lr','critic_lr','particle_lr','reg_coeff','reg_kappa','prior_reg'}
METRICS=('cell_energy_distance','energy_score','spread_ratio','within_condition_spread_ratio','coverage80')
def validate_vector(h):
 if set(h)!=FIELDS or any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in h.values()):raise ValueError('Exact six positive finite PG scalars required')
def summarize(candidates,results,reference_rows,baseline_metrics):
 expected={(c['id'],seed) for c in candidates for seed in (0,1)}
 if len(results)!=len(expected) or {(r['candidate_id'],r['seed']) for r in results}!=expected:raise ValueError('All20 planned paired selection endpoints required exactly once')
 if len(reference_rows)!=2 or {r['seed'] for r in reference_rows}!={0,1}:raise ValueError('Exactly two labeled measured reference controls required')
 if set(baseline_metrics)!={'0','1'}:raise ValueError('Both matched resampling controls required')
 choices=[dict(id=c['id'],hyperparameters=c['hyperparameters'],read_only_reference=False) for c in candidates]
 if reference_rows[0]['hyperparameters']!=reference_rows[1]['hyperparameters']:raise ValueError('Reference settings differ')
 choices.append(dict(id='d000',hyperparameters=reference_rows[0]['hyperparameters'],read_only_reference=True))
 summary=[]
 for c in choices:
  validate_vector(c['hyperparameters']);rows=sorted(reference_rows if c['read_only_reference'] else [r for r in results if r['candidate_id']==c['id']],key=lambda r:r['seed'])
  for r in rows:
   if r['hyperparameters']!=c['hyperparameters'] or r['endpoint']!=20000 or not r['from_scratch'] or r['initial_completed_updates']!=0:raise ValueError('Full fresh trajectory provenance differs')
   if r['finite']:
    if any(not math.isfinite(r['metrics'][k]) for k in METRICS):raise ValueError('Finite selection metrics required')
    if r['baseline_metrics']!=baseline_metrics[str(r['seed'])]:raise ValueError('Matched resampling differs across settings')
    for key in ('training_seed','donors','calibration_manifest_SHA256','trace_manifest_SHA256','sampling_contract'):
     if r['metrics'][key]!=r['baseline_metrics'][key]:raise ValueError('Calibration draw/target contract differs')
  entry=dict(candidate_id=c['id'],hyperparameters=c['hyperparameters'],read_only_reference=c['read_only_reference'],all_two_selection_seeds_finite=all(r['finite'] for r in rows),
    failed_seeds=[r['seed'] for r in rows if not r['finite']],per_seed=[dict(seed=r['seed'],finite=r['finite'],metrics=r['metrics'],baseline_metrics=r.get('baseline_metrics')) for r in rows])
  if entry['all_two_selection_seeds_finite']:
   entry['mean']={k:sum(r['metrics'][k] for r in rows)/2 for k in METRICS}
   entry['SD']={k:math.sqrt(sum((r['metrics'][k]-entry['mean'][k])**2 for r in rows)) for k in METRICS}
   entry['mean_matched_resampling']={k:sum(baseline_metrics[str(seed)][k] for seed in (0,1))/2 for k in METRICS}
   entry['joint_quality']=all(.7<=r['metrics']['spread_ratio']<=1.3 for r in rows) and entry['mean']['energy_score']<=entry['mean_matched_resampling']['energy_score']
  summary.append(entry)
 rank=sorted([r for r in summary if r['all_two_selection_seeds_finite']],key=lambda r:(r['mean']['cell_energy_distance'],r['mean']['energy_score'],r['candidate_id']))
 joint=[r for r in rank if r['joint_quality']]
 selected=[]
 for role,row in [('primary_distance',rank[0] if rank else None),('joint_quality',joint[0] if joint else None),('reference',next(r for r in summary if r['candidate_id']=='d000'))]:
  if row is None:continue
  old=next((c for c in selected if c['candidate_id']==row['candidate_id']),None)
  if old is None:old=dict(candidate_id=row['candidate_id'],hyperparameters=row['hyperparameters'],roles=[]);selected.append(old)
  old['roles'].append(role)
 return dict(candidates=summary,ranking_by_two_seed_mean_distance=[r['candidate_id'] for r in rank],
    proposed_followup_configurations=selected,selection_seed_set=[0,1],followup_seed_set=[2,3,4],
    reference_measurements_previously_observed=True,reference_models_not_refitted=True,
    failed_seeds_not_dropped=True,all_planned_attempts_resolved=True,heldout_used=False,
    independent_biological_confirmation=False,automatic_external_freeze=False,SOTA_claim=False,full_goal_complete=False)
