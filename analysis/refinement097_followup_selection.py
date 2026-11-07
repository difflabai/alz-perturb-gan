"""Prespecified matched-seed followup after paired scalar selection.

New configurations use seeds2..4. Previously observed d000 values remain
labeled reference controls and are never presented as new independent fits.
"""
import math
METRICS=('cell_energy_distance','energy_score','spread_ratio','within_condition_spread_ratio','coverage80')
def summarize(selection,new_results,reference_rows,baseline_metrics):
 if not selection['all_planned_attempts_resolved'] or selection['heldout_used']:raise ValueError('Complete calibration-only paired selection required')
 choices=selection['proposed_followup_configurations'];ids=[c['candidate_id'] for c in choices]
 if len(ids)!=len(set(ids)) or 'd000' not in ids or not 1<=len(ids)<=3:raise ValueError('Distinct declared primary/joint/reference choices required')
 new_ids=set(ids)-{'d000'};expected={(cid,s) for cid in new_ids for s in (2,3,4)}
 if len(new_results)!=len(expected) or {(r['candidate_id'],r['seed']) for r in new_results}!=expected:raise ValueError('Every frozen new-configuration seed2..4 outcome required once')
 if len(reference_rows)!=3 or {r['seed'] for r in reference_rows}!={2,3,4} or any(r['candidate_id']!='d000' for r in reference_rows):raise ValueError('All three labeled prior d000 reference outcomes required')
 if set(baseline_metrics)!={'2','3','4'}:raise ValueError('Exact three matched resampling controls required')
 entries=[]
 for choice in choices:
  reference=choice['candidate_id']=='d000';rows=sorted(reference_rows if reference else [r for r in new_results if r['candidate_id']==choice['candidate_id']],key=lambda r:r['seed'])
  for r in rows:
   if r['hyperparameters']!=choice['hyperparameters'] or r['endpoint']!=20000 or not r['from_scratch'] or r['initial_completed_updates']!=0:raise ValueError('Fresh full-horizon configuration provenance differs')
   if reference and (not r['read_only_previously_measured_reference'] or r['new_model_fitted'] or r['new_GPU_charge']!=0):raise ValueError('Prior reference must not masquerade as a fresh independent fit')
   if r['finite']:
    if r['baseline_metrics']!=baseline_metrics[str(r['seed'])]:raise ValueError('Matched resampling differs')
    for m in (r['metrics'],r['baseline_metrics']):
     if m['training_seed']!=r['seed'] or m['metric_role']!='calibration_selection' or m['generation_replicates']!=2 or any(not math.isfinite(m[k]) for k in METRICS):raise ValueError('Complete finite matched calibration metrics required')
    for k in ('donors','calibration_manifest_SHA256','trace_manifest_SHA256','sampling_contract'):
     if r['metrics'][k]!=r['baseline_metrics'][k]:raise ValueError('Target or sampling contract differs')
  entry=dict(**choice,reference_measurements_previously_observed=reference,all_three_followup_seeds_finite=all(r['finite'] for r in rows),failed_seeds=[r['seed'] for r in rows if not r['finite']],
    per_seed=[dict(seed=r['seed'],finite=r['finite'],metrics=r['metrics'],baseline_metrics=baseline_metrics[str(r['seed'])]) for r in rows])
  if entry['all_three_followup_seeds_finite']:
   entry['followup_mean']={k:sum(r['metrics'][k] for r in rows)/3 for k in METRICS}
   entry['followup_SD']={k:math.sqrt(sum((r['metrics'][k]-entry['followup_mean'][k])**2 for r in rows)/2) for k in METRICS}
   entry['mean_matched_resampling']={k:sum(baseline_metrics[str(s)][k] for s in (2,3,4))/3 for k in METRICS}
   entry['joint_quality']=all(.7<=r['metrics']['spread_ratio']<=1.3 for r in rows) and entry['followup_mean']['energy_score']<=entry['mean_matched_resampling']['energy_score']
  entries.append(entry)
 ranked=sorted([e for e in entries if e['all_three_followup_seeds_finite']],key=lambda e:(e['followup_mean']['cell_energy_distance'],e['followup_mean']['energy_score'],e['candidate_id']))
 joint=[e for e in ranked if e['joint_quality']]
 return dict(candidates=entries,ranking_by_followup_mean_distance=[e['candidate_id'] for e in ranked],
   proposed_external_candidate=ranked[0]['candidate_id'] if ranked else None,joint_quality_candidate=joint[0]['candidate_id'] if joint else None,
   proposed_candidate_joint_quality=ranked[0]['joint_quality'] if ranked else False,ranking_seed_set=[2,3,4],selection_seeds0_and1_excluded_from_followup_ranking=True,
   all_planned_followup_attempts_resolved=True,required_new_fits=len(expected),reference_new_fits=0,reference_previously_observed=True,
   failed_seeds_not_dropped=True,heldout_used=False,independent_biological_confirmation=False,automatic_external_freeze=False,SOTA_claim=False,full_goal_complete=False)
