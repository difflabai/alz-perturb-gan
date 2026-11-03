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
for row in rows:row['native_recipe']='bispf_optimized'
extra=ROOT/'analysis/CAL_GPU0_V1_original_bispf_v1_f0_pg_s0-result.json'
if extra.exists():
 evidence=read(extra);candidate=evidence['candidate'];audit=read(ROOT/'analysis/original_bispf_checkpoint_audit.json')
 assert evidence['independent_saved_scores_checkpoint_and_count_exemplars_verified']
 assert candidate['pure_only'] and candidate['real_bag_replacements']==0 and candidate['native_recipe']=='bispf'
 assert audit['status']=='VERIFIED_COMPLETE_NATIVE_ORIGINAL_BISPF_CHECKPOINT' and audit['checkpoint_SHA256']==candidate['checkpoint_SHA256'] and audit['native_recipe_exactly_matches_official_API_factory']
 for key in ['donors','training_seed','sampling_contract','calibration_manifest_SHA256','trace_manifest_SHA256','generation_replicates']:assert candidate[key]==rows[0][key]
 rows.append({**candidate,'status':'VERIFIED_PURE_PG_ONLY','formulation':'original_bispf','complete_clean_settlement_verified':True,'charged_GPU_seconds':evidence['charged_GPU_seconds'],'accepted':evidence['decision']['accept'],'decision_evidence':str(extra)})
extra=ROOT/'analysis/CAL_GPU0_V1_semantic_v1_f0_pg_s0-result.json'
if extra.exists():
 evidence=read(extra);candidate=evidence['candidate'];audit=read(ROOT/'analysis/semantic_checkpoint_audit.json')
 assert evidence['independent_saved_scores_checkpoint_and_count_exemplars_verified']
 assert candidate['pure_only'] and candidate['real_bag_replacements']==0 and candidate['native_recipe']=='bispf_optimized'
 assert audit['status']=='VERIFIED_COMPLETE_PURE_SEMANTIC_CHECKPOINT' and audit['checkpoint_SHA256']==candidate['checkpoint_SHA256'] and audit['native_recipe_exactly_matches_official_API_factory']
 for key in ['donors','training_seed','sampling_contract','calibration_manifest_SHA256','trace_manifest_SHA256','generation_replicates']:assert candidate[key]==rows[0][key]
 rows.append({**candidate,'status':'VERIFIED_PURE_PG_ONLY','formulation':'semantic_observation','complete_clean_settlement_verified':True,'charged_GPU_seconds':evidence['charged_GPU_seconds'],'accepted':evidence['decision']['accept'],'decision_evidence':str(extra)})
extra=ROOT/'analysis/CAL_GPU0_V1_bcap_v1_f0_pg_s0-result.json'
if extra.exists() and 'candidate' in read(extra):
 evidence=read(extra);candidate=evidence['candidate'];audit=read(ROOT/'analysis/bcap_checkpoint_audit.json')
 assert evidence['independent_saved_scores_checkpoint_and_count_exemplars_verified']
 assert candidate['pure_only'] and candidate['real_bag_replacements']==0 and candidate['native_recipe']=='bcap'
 assert audit['status']=='VERIFIED_COMPLETE_NATIVE_BCAP_CHECKPOINT' and audit['checkpoint_SHA256']==candidate['checkpoint_SHA256'] and audit['native_recipe_exactly_matches_official_API_factory']
 for key in ['donors','training_seed','sampling_contract','calibration_manifest_SHA256','trace_manifest_SHA256','generation_replicates']:assert candidate[key]==rows[0][key]
 rows.append({**candidate,'status':'VERIFIED_PURE_PG_ONLY','formulation':'bcap','complete_clean_settlement_verified':True,'charged_GPU_seconds':evidence['charged_GPU_seconds'],'accepted':evidence['decision']['accept'],'decision_evidence':str(extra)})
extra=ROOT/'analysis/CAL_GPU0_V1_learned_obs_v1_f0_pg_s0-result.json'
if extra.exists() and 'candidate' in read(extra):
 evidence=read(extra);candidate=evidence['candidate'];audit=read(ROOT/'analysis/learned_observation_checkpoint_audit.json')
 assert evidence['independent_saved_scores_checkpoint_and_count_exemplars_verified']
 assert candidate['pure_only'] and candidate['real_bag_replacements']==0 and candidate['native_recipe']=='bispf_optimized' and candidate['operation']=='train_learned'
 assert audit['status']=='VERIFIED_COMPLETE_PURE_LEARNED_OBSERVATION_CHECKPOINT' and audit['checkpoint_SHA256']==candidate['checkpoint_SHA256'] and audit['native_recipe_exactly_matches_official_API_factory']
 assert audit['all_model_keys_shapes_and_dtypes_match_declared_architecture'] and audit['finite_model_floating_point_values']==376066 and audit['learned_observation_config']==candidate['learned_observation_config']
 for key in ['donors','training_seed','sampling_contract','calibration_manifest_SHA256','trace_manifest_SHA256','generation_replicates']:assert candidate[key]==rows[0][key]
 rows.append({**candidate,'status':'VERIFIED_PURE_PG_ONLY','formulation':'learned_observation','complete_clean_settlement_verified':True,'charged_GPU_seconds':evidence['charged_GPU_seconds'],'accepted':evidence['decision']['accept'],'decision_evidence':str(extra)})
extra=ROOT/'analysis/CAL_GPU0_V1_network_rate_half_v1_f0_pg_s0-result.json'
if extra.exists() and 'candidate' in read(extra):
 evidence=read(extra);candidate=evidence['candidate'];audit=read(ROOT/'analysis/network_rate_checkpoint_audit.json')
 assert evidence['independent_saved_scores_checkpoint_and_count_exemplars_verified']
 assert candidate['pure_only'] and candidate['real_bag_replacements']==0 and candidate['native_recipe']=='bispf_optimized' and candidate['operation']=='train_network_rate'
 assert audit['status']=='VERIFIED_COMPLETE_NATIVE_NETWORK_RATE_CHECKPOINT' and audit['checkpoint_SHA256']==candidate['checkpoint_SHA256'] and audit['native_recipe_exactly_matches_official_API_factory']
 assert audit['all_model_keys_shapes_and_dtypes_match_declared_architecture'] and audit['finite_model_floating_point_values']==371177 and audit['network_rate_config']==candidate['network_rate_config']=={'lr':.006,'prior_lr_mult':5.}
 for key in ['donors','training_seed','sampling_contract','calibration_manifest_SHA256','trace_manifest_SHA256','generation_replicates']:assert candidate[key]==rows[0][key]
 rows.append({**candidate,'status':'VERIFIED_PURE_PG_ONLY','formulation':'network_rate_half','complete_clean_settlement_verified':True,'charged_GPU_seconds':evidence['charged_GPU_seconds'],'accepted':evidence['decision']['accept'],'decision_evidence':str(extra)})
base=read(ROOT/'baseline_calibration.json')
for row in rows:
 row['ES_improvement_vs_resampling_percent']=100*(1-row['energy_score']/base['energy_score'])
 row['cell_ED_difference_vs_resampling']=row['cell_energy_distance']-base['cell_energy_distance']
 row['joint_baseline_target_met']=row['ES_improvement_vs_resampling_percent']>=5 and row['cell_ED_difference_vs_resampling']<=0 and .7<=row['spread_ratio']<=1.3
rows.sort(key=lambda q:q['energy_score'])
result={'status':'PURE_RETAINED_VARIANTS_AND_NATIVE_RECIPE_COMPARISON_VERIFIED','native_recipe_families_tested':sorted(set(q['native_recipe'] for q in rows)),'new_training_updates':sum(q['new_model_updates'] for q in rows),'protocol_SHA256':sha(ROOT/'pure_evaluation_protocol.json'),'manifest_SHA256':sha(ROOT/'pure_formulations.json'),'pure_only':True,'baseline':{k:base[k] for k in ['energy_score','cell_energy_distance','spread_ratio']},'rows':rows,'best_pure_by_energy_score':rows[0]['formulation'],'suite_charged_GPU_seconds':sum(q['charged_GPU_seconds'] for q in rows),'joint_target_met_any':any(q['joint_baseline_target_met'] for q in rows),'limitations':'Adaptive4calibrationdonors, firstfold,seed0,two generation replicates.40k uses extra training compute. Not evaluation or external SOTA.','hybrid_incumbent_SHA256':sha(ROOT/'incumbent.json'),'hybrid_incumbent_unchanged':True}
dump(ROOT/'analysis/pure_formulations_leaderboard.json',result)
best=rows[0]  # Reporting does not mutate the selected pure incumbent.
lines=['Pure ParticleGAN calibration comparison','', f"{len(rows)} completed pure comparisons. Native recipe families: {', '.join(result['native_recipe_families_tested'])}. Seven retained architecture/training variants, native-preset comparisons, and observation variants. Direct iter_pg, zero real-bag replacements. Same four calibration donors, first fold, seed 0, two fixed generation replicates. Lower scores are better. Running or failed attempts are not quality results.",'', '| Pure formulation | Updates | Energy score | Cell energy distance | Spread |','|---|---:|---:|---:|---:|',f"| Matched resampling | — | {base['energy_score']:.8f} | {base['cell_energy_distance']:.8f} | {base['spread_ratio']:.5f} |"]
for q in rows:lines.append(f"| {q['formulation']} | {q['completed_updates']} | {q['energy_score']:.8f} | {q['cell_energy_distance']:.8f} | {q['spread_ratio']:.5f} |")
lines+=['','Reference2, latent8 and balanced8 control arrays and integer count exemplars reproduced bitwise. Every run completed cleanly and saved scores/counts/checkpoint hashes were independently verified.','', 'The40000-update continuation uses extra compute. These are adaptive calibration results, not independent validation or an external SOTA claim. Pure and hybrid leaders are kept separately. No pure formulation meets the joint5%ES/cellED/spread target.' if not result['joint_target_met_any'] else 'Joint seed0 target reached; stable seed confirmation remains necessary.','',f"New training updates in additional recipe/observation comparisons:{result['new_training_updates']}. Charged GPU time:{result['suite_charged_GPU_seconds']:.4f} seconds."]
(ROOT/'PURE_PARTICLEGAN_RESULTS.txt').write_text('\n'.join(lines)+'\n');print(json.dumps({'best':best['formulation'],'joint_target_met_any':result['joint_target_met_any'],'charged_GPU_seconds':result['suite_charged_GPU_seconds'],'rows':[{k:q[k] for k in ['formulation','energy_score','cell_energy_distance','spread_ratio']} for q in rows]},indent=2))
