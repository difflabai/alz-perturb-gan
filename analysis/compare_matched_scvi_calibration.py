"""Completed matched external calibration comparison; no SOTA or promotion."""
import datetime,hashlib,json
from pathlib import Path
ROOT=Path('/ml2/alz')
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def compare():
 protocol_path=ROOT/'analysis/matched_scvi/protocol.json';protocol=read(protocol_path)
 pg_path=Path(protocol['PG_reference_result']);assert sha(pg_path)==protocol['PG_reference_result_SHA256']
 scvi_path=ROOT/'analysis/CAL_GPU0_V1_matched_scvi_calibration_f0_scvi_s0-result.json'
 baseline_path=ROOT/'analysis/baseline_calibration_seed0.json'
 pg_audit=read(pg_path);scvi_audit=read(scvi_path)
 assert pg_audit['owned_release_and_clean_settlement_verified'] and pg_audit['independent_saved_scores_checkpoint_and_count_exemplars_verified']
 assert scvi_audit['owned_release_and_clean_settlement_verified'] and scvi_audit['independent_saved_scores_and_count_exemplars_verified']
 assert scvi_audit['native_checkpoint_CPU_reload_matches_saved_CUDA_state']
 baseline=read(baseline_path);pg=pg_audit['candidate'];scvi=scvi_audit['candidate']
 assert pg['status']==scvi['status']=='completed'
 assert pg['completed_updates']==20000 and scvi['completed_epochs']==400
 assert scvi['comparator_protocol_SHA256']==sha(protocol_path) and scvi['evaluation_used'] is False
 for key in ('donors','training_seed','generation_replicates','calibration_manifest_SHA256','trace_manifest_SHA256','sampling_contract'):
  assert pg[key]==scvi[key]==baseline[key],key
 assert pg['training_seed']==0 and pg['generation_replicates']==2 and len(pg['donors'])==4
 assert pg['calibration_manifest_SHA256']==protocol['calibration_manifest_SHA256']
 assert pg['trace_manifest_SHA256']==protocol['trace_manifest_SHA256']
 assert pg['real_bag_replacements']==scvi['real_bag_replacements']==0
 keys=('energy_score','cell_energy_distance','spread_ratio','per_donor_energy_score','per_stage_cell_energy_distance')
 result={'status':'AUDITED_EXTERNAL_SCVI_VS_RELEASED_PG_CALIBRATION_HEAD_TO_HEAD','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
  'methods':{'resampling':{k:baseline[k] for k in keys},'released_PG095':{k:pg[k] for k in keys},'scVI133':{k:scvi[k] for k in keys}},
  'PG_minus_scvi':{k:pg[k]-scvi[k] for k in ('energy_score','cell_energy_distance','spread_ratio')},
  'paired_donor_ES_PG_minus_scvi':[{'donor':donor,'difference':p-s} for donor,p,s in zip(pg['donors'],pg['per_donor_energy_score'],scvi['per_donor_energy_score'])],
  'PG_has_lower_ES_than_scvi':pg['energy_score']<scvi['energy_score'],
  'PG_has_no_higher_cell_ED_than_scvi':pg['cell_energy_distance']<=scvi['cell_energy_distance'],
  'comparison_population':'Four adaptive calibration donors, first fold, one training seed each, two generation replicates; replicates do not add biological donors.',
  'tuning_fairness':protocol['asymmetric_tuning_effort_disclosed'],
  'source_SHA256':{str(p):sha(p) for p in (Path(__file__),protocol_path,pg_path,scvi_path,baseline_path)},
  'full_final_3fold_5seed_benchmark_complete':False,'evaluation_used':False,'SOTA_claim':False,'incumbent_changed':False,
  'scope':'Actual completed external fixed-recipe calibration comparison, not independent final evaluation, equal adaptive tuning, or general SOTA superiority. All failed PG settings and comparator qualification remain retained. No automatic selection or promotion.'}
 destination=ROOT/'analysis/matched_scvi_vs_released095_calibration.json';assert not destination.exists();destination.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({'status':result['status'],'methods':result['methods'],'PG_minus_scvi':result['PG_minus_scvi']},indent=2))
if __name__=='__main__':compare()
