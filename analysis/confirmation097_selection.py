"""Prespecified selection and fresh-initialization confirmation, no model I/O."""
import json,math

FIELDS={'generator_lr','critic_lr','particle_lr','reg_coeff','reg_kappa','prior_reg'}
def vector(h):
 if set(h)!=FIELDS or any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in h.values()):raise ValueError('Exact six finite positive native PG scalars required')
 return json.dumps(h,sort_keys=True,separators=(',',':'))

def choose_configurations(rows,baseline_seed0_energy):
 """Keep the distance winner, joint-quality winner and reference, deduplicated."""
 if len(rows)!=6 or {r['candidate_id'] for r in rows}!={f'd00{i}' for i in range(6)}:raise ValueError('All six resolved full20k selection outcomes required')
 finite=[]
 for r in rows:
  vector(r['hyperparameters'])
  if r['finite']:
   if any(not math.isfinite(r['metrics'][k]) for k in ('cell_energy_distance','energy_score','spread_ratio')):raise ValueError('Finite endpoint metrics required')
   finite.append(r)
 if not finite:raise ValueError('No finite selection endpoint')
 reference=next(r for r in rows if r['candidate_id']=='d000')
 if not reference['finite']:raise ValueError('Finite reference required for matched initialization confirmation')
 rank=lambda r:(r['metrics']['cell_energy_distance'],r['metrics']['energy_score'],r['candidate_id'])
 primary=min(finite,key=rank)
 joint=[r for r in finite if .7<=r['metrics']['spread_ratio']<=1.3 and r['metrics']['energy_score']<=baseline_seed0_energy]
 roles=[('primary_distance',primary),('joint_quality',min(joint,key=rank) if joint else None),('reference',reference)]
 unique={}
 for role,row in roles:
  if row is None:continue
  key=vector(row['hyperparameters'])
  if key not in unique:unique[key]=dict(candidate_id=row['candidate_id'],hyperparameters=dict(row['hyperparameters']),roles=[])
  unique[key]['roles'].append(role)
 selected=list(unique.values())
 assert 1<=len(selected)<=3
 return selected

def summarize_confirmation(selected,results):
 """Use all four fresh seeds; never replace failed seeds with the selection seed.

The next external candidate minimizes mean fresh-seed1..4 distribution
distance among fully finite configurations. Energy and spread tradeoffs remain
visible. This is initialization robustness on reused calibration donors, not
independent biological confirmation or a final SOTA comparison.
"""
 ids=[q['candidate_id'] for q in selected]
 expected={(cid,seed) for cid in ids for seed in range(1,5)}
 if len(results)!=len(expected) or {(r['candidate_id'],r['seed']) for r in results}!=expected:raise ValueError('Every planned fresh-seed attempt must be resolved exactly once')
 candidates=[];baselines={};observed=[]
 for choice in selected:
  runs=sorted([r for r in results if r['candidate_id']==choice['candidate_id']],key=lambda r:r['seed'])
  for r in runs:
   if r['hyperparameters']!=choice['hyperparameters'] or r['endpoint']!=20000 or r['initial_completed_updates']!=0 or not r['from_scratch']:raise ValueError('Fresh complete-horizon configuration/seed provenance differs')
   if r['finite']:
    for m in (r['metrics'],r['baseline_metrics']):
     if m['training_seed']!=r['seed'] or any(not math.isfinite(m[k]) for k in ('cell_energy_distance','energy_score','spread_ratio')):raise ValueError('Finite matched-seed metrics required')
    for k in ('donors','calibration_manifest_SHA256','trace_manifest_SHA256','sampling_contract'):
     if r['metrics'][k]!=r['baseline_metrics'][k]:raise ValueError('PG and baseline target/trace contracts differ')
    if r['seed'] in baselines and baselines[r['seed']]!=r['baseline_metrics']:raise ValueError('Matched resampling differs across configurations')
    baselines[r['seed']]=r['baseline_metrics']
  finite=all(r['finite'] for r in runs)
  entry=dict(**choice,all_four_fresh_seeds_finite=finite,seeds=[r['seed'] for r in runs],failed_seeds=[r['seed'] for r in runs if not r['finite']])
  if finite:
   keys=('cell_energy_distance','energy_score','spread_ratio','within_condition_spread_ratio','coverage80')
   entry['fresh_seed_mean']={k:sum(r['metrics'][k] for r in runs)/4 for k in keys}
   entry['mean_matched_resampling']={k:sum(r['baseline_metrics'][k] for r in runs)/4 for k in keys}
   entry['per_seed']=[dict(seed=r['seed'],metrics=r['metrics'],baseline_metrics=r['baseline_metrics'],PG_minus_resampling={k:r['metrics'][k]-r['baseline_metrics'][k] for k in ('cell_energy_distance','energy_score')}) for r in runs]
   entry['joint_quality']=all(.7<=r['metrics']['spread_ratio']<=1.3 for r in runs) and entry['fresh_seed_mean']['energy_score']<=entry['mean_matched_resampling']['energy_score']
   observed.append(entry)
  candidates.append(entry)
 ranked=sorted(observed,key=lambda q:(q['fresh_seed_mean']['cell_energy_distance'],q['fresh_seed_mean']['energy_score'],q['candidate_id']))
 joint=[q for q in ranked if q['joint_quality']]
 return dict(candidates=candidates,ranking_by_fresh_seed_mean_distance=[q['candidate_id'] for q in ranked],
             proposed_external_candidate=ranked[0]['candidate_id'] if ranked else None,
             joint_quality_candidate=joint[0]['candidate_id'] if joint else None,
             proposed_candidate_joint_quality=ranked[0]['joint_quality'] if ranked else False,
             ranking_seed_set=[1,2,3,4],selection_seed0_excluded_from_confirmation_ranking=True,
             baseline_seeds_observed=sorted(baselines),all_planned_attempts_resolved=True,
             not_independent_biological_confirmation=True,heldout_used=False,SOTA_claim=False,full_goal_complete=False)
