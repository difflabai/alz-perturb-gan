"""Fixed20k matched-seed calibration report; no new heldout model selection."""
import datetime,hashlib,json,statistics,sys
from pathlib import Path
ROOT=Path('/ml2/alz');sys.path.insert(0,str(ROOT/'scalarcal097_runtime'))
import preset_scope as s,binding as b

def aggregate():
 p=s.protocol();matrix=s.read(s.STUDY/'matrix_complete.json');assert matrix['matrix_complete'] and matrix['complete_cells']==matrix['required_cells']==6
 assert matrix['protocol_SHA256']==s.sha(s.PROTOCOL)
 assert not s.read(s.PLAN)['enabled'];assert not s.read(b.PROJECT/'project_gpu_budget.json')['open_reservations']
 rows={};pins={str(s.STUDY/'matrix_complete.json'):s.sha(s.STUDY/'matrix_complete.json')}
 for seed in (5,6,7):
  for arm in ('fixed','adaptive'):
   cell=s.cell(arm,seed);key=b.attempt_key(cell);path=s.STUDY/'results'/(key+'.json');q=s.read(path)
   assert s.sha(path)==matrix['result_SHA256'][key] and q['cell']==cell and q['complete'] and q['all_new_training_from_scratch'] and q['initial_completed_updates']==0
   assert q['independent_CPU_saved_score_and_controller_audit'] and q['native_saved_state_matches_GPU_restore'] and q['benchmark_protocol_SHA256']==s.sha(s.PROTOCOL)
   for saved,pin in q['files_SHA256'].items():assert b.sha(saved)==pin
   rows[seed,arm]=q;pins[str(path)]=s.sha(path)
 for seed in (5,6,7):assert rows[seed,'fixed']['baseline_metrics']==rows[seed,'adaptive']['baseline_metrics']
 metrics=('cell_energy_distance','energy_score','spread_ratio','within_condition_spread_ratio','coverage80');arms={}
 for arm in ('fixed','adaptive'):
  arms[arm]={k:dict(mean=statistics.mean(rows[seed,arm]['metrics'][k] for seed in (5,6,7)),initialization_SD=statistics.stdev(rows[seed,arm]['metrics'][k] for seed in (5,6,7))) for k in metrics}
 chosen=min(('fixed','adaptive'),key=lambda arm:arms[arm]['cell_energy_distance']['mean'])
 report=dict(status='COMPLETE_SIX_FRESH20K_MATCHED_SCALAR_LR_CALIBRATION_COMPARISON',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),protocol_SHA256=s.sha(s.PROTOCOL),required_fresh_fits=6,complete_fresh_fits=6,training_seeds=[5,6,7],updates_per_fit=20000,arms=arms,calibration_preferred_arm=chosen,selection_rule='Mean fixed20000update calibration cell energy distance across all three matched fresh seeds; an exact tie retains fixed control.',paired_ED_deltas_adaptive_minus_fixed={str(seed):rows[seed,'adaptive']['metrics']['cell_energy_distance']-rows[seed,'fixed']['metrics']['cell_energy_distance'] for seed in (5,6,7)},matched_resampling={k:statistics.mean(rows[seed,'fixed']['baseline_metrics'][k] for seed in (5,6,7)) for k in metrics},all_six_outcomes_retained=True,from_scratch=True,source_checkpoint=None,controller_uses_only_calibration_cell_ED=True,secondary_metrics_control_LR=False,old45_candidate_unchanged=True,external_outcomes_used_for_selection=False,actual_GPU_seconds=sum(q['charged_GPU_seconds'] for q in rows.values()),source_SHA256={str(Path(__file__)):s.sha(__file__)},input_SHA256=pins,limitations='Four adaptively reused calibration donors; three initialization seeds describe optimizer variability, not biological replication or an independent population interval. No SOTA conclusion or automatic promotion of this candidate into the separate frozen45-model comparison.',SOTA_claim=False,full_goal_complete=False)
 path=s.STUDY/'comparison.json';assert not path.exists();path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(status=report['status'],calibration_preferred_arm=chosen,arms=arms)))
if __name__=='__main__':aggregate()
