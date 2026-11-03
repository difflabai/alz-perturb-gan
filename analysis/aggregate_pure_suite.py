"""Publish separate calibration-only pure PG leaderboard; never edit hybrid incumbent."""
import json,sys
from pathlib import Path
ROOT=Path('/ml2/alz');sys.path.insert(0,str(ROOT))
from pure_suite import read,sha
from calibration import dump
manifest=read(ROOT/'pure_formulations.json');rows=[read(ROOT/'analysis'/f'pure_{label}_verification.json') for label in manifest['entries']]
for q in rows:
 assert q['status']=='VERIFIED_PURE_PG_ONLY' and q['complete_clean_settlement_verified'] and q['real_bag_replacements']==0 and q['new_model_updates']==0
 for key in ['donors','training_seed','sampling_contract','calibration_manifest_SHA256','trace_manifest_SHA256','generation_replicates']:assert q[key]==rows[0][key]
 if manifest['entries'][q['formulation']]['prior_generation_is_pure']:assert q['control_arrays_and_count_exemplars_bitwise_equal']
base=read(ROOT/'baseline_calibration.json')
for row in rows:
 row['ES_improvement_vs_resampling_percent']=100*(1-row['energy_score']/base['energy_score'])
 row['cell_ED_difference_vs_resampling']=row['cell_energy_distance']-base['cell_energy_distance']
 row['joint_baseline_target_met']=row['ES_improvement_vs_resampling_percent']>=5 and row['cell_ED_difference_vs_resampling']<=0 and .7<=row['spread_ratio']<=1.3
rows.sort(key=lambda q:q['energy_score'])
result={'status':'ALL_SEVEN_RETAINED_FORMULATIONS_TESTED_PURE','protocol_SHA256':sha(ROOT/'pure_evaluation_protocol.json'),'manifest_SHA256':sha(ROOT/'pure_formulations.json'),'pure_only':True,'baseline':{k:base[k] for k in ['energy_score','cell_energy_distance','spread_ratio']},'rows':rows,'best_pure_by_energy_score':rows[0]['formulation'],'suite_charged_GPU_seconds':sum(q['charged_GPU_seconds'] for q in rows),'joint_target_met_any':any(q['joint_baseline_target_met'] for q in rows),'limitations':'Adaptive4calibrationdonors, firstfold,seed0,two generation replicates.40k uses extra training compute. Not evaluation or external SOTA.','hybrid_incumbent_SHA256':sha(ROOT/'incumbent.json'),'hybrid_incumbent_unchanged':True}
dump(ROOT/'analysis/pure_formulations_leaderboard.json',result)
best=rows[0];dump(ROOT/'pure_incumbent.json',{'status':'CALIBRATION_ONLY_PURE_ES_LEADER_NOT_VALIDATED',**best,'method':'pg_40k' if best['completed_updates']==40000 else 'pg','checkpoint_path':manifest['entries'][best['formulation']]['checkpoint_path']})
lines=['Pure ParticleGAN calibration comparison','', 'All seven retained trained formulations were generated directly through iter_pg, with zero real-bag replacements. Same4calibrationdonors,firstfold,seed0,two fixed generation replicates. Lower scores are better.','', '| Pure formulation | Updates | Energy score | Cell energy distance | Spread |','|---|---:|---:|---:|---:|',f"| Matched resampling | — | {base['energy_score']:.8f} | {base['cell_energy_distance']:.8f} | {base['spread_ratio']:.5f} |"]
for q in rows:lines.append(f"| {q['formulation']} | {q['completed_updates']} | {q['energy_score']:.8f} | {q['cell_energy_distance']:.8f} | {q['spread_ratio']:.5f} |")
lines+=['','Reference2, latent8 and balanced8 control arrays and integer count exemplars reproduced bitwise. Every run completed cleanly and saved scores/counts/checkpoint hashes were independently verified.','', 'The40000-update continuation uses extra compute. These are adaptive calibration results, not independent validation or an external SOTA claim. Pure and hybrid leaders are kept separately. No pure formulation meets the joint5%ES/cellED/spread target.' if not result['joint_target_met_any'] else 'Joint seed0 target reached; stable seed confirmation remains necessary.','',f"New training updates for this suite:0. Charged GPU time:{result['suite_charged_GPU_seconds']:.4f} seconds."]
(ROOT/'PURE_PARTICLEGAN_RESULTS.txt').write_text('\n'.join(lines)+'\n');print(json.dumps({'best':best['formulation'],'joint_target_met_any':result['joint_target_met_any'],'charged_GPU_seconds':result['suite_charged_GPU_seconds'],'rows':[{k:q[k] for k in ['formulation','energy_score','cell_energy_distance','spread_ratio']} for q in rows]},indent=2))
