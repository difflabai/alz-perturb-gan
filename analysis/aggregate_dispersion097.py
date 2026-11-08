"""Report every fresh theta pair at the declared20k endpoint; calibration only."""
import datetime,hashlib,json,math,statistics,sys
from pathlib import Path
ROOT=Path('/ml2/alz');sys.path.insert(0,str(ROOT/'dispersion097_runtime'))
import binding as b,preset_scope as scope

METRICS=('cell_energy_distance','energy_score','spread_ratio','within_condition_spread_ratio','coverage80')
def summarize(rows):
 assert len(rows)==6 and {(r['arm'],r['seed']) for r in rows}=={(a,s) for a in ('theta1','theta5') for s in (8,9,10)}
 by={a:sorted([r for r in rows if r['arm']==a],key=lambda r:r['seed']) for a in ('theta1','theta5')}
 assert all(r['complete'] and r['all_new_training_from_scratch'] and r['initial_completed_updates']==0 for r in rows)
 assert all(math.isfinite(r['metrics'][k]) for r in rows for k in METRICS)
 means={a:{k:statistics.mean(r['metrics'][k] for r in by[a]) for k in METRICS} for a in by}
 deltas=[dict(seed=s,**{k:by['theta5'][i]['metrics'][k]-by['theta1'][i]['metrics'][k] for k in METRICS}) for i,s in enumerate((8,9,10))]
 winner='theta5' if means['theta5']['cell_energy_distance']<means['theta1']['cell_energy_distance'] else 'theta1'
 return dict(mean_metrics_by_arm=means,paired_theta5_minus_theta1=deltas,paired_mean_differences={k:statistics.mean(d[k] for d in deltas) for k in METRICS},calibration_preferred_arm=winner,theta5_ED_paired_wins=sum(d['cell_energy_distance']<0 for d in deltas))
def main():
 scope.protocol();scope.prior_study_ready()
 assert not scope.read(scope.PLAN)['enabled']
 matrix=scope.STUDY/'matrix_complete.json';m=scope.read(matrix)
 assert m['matrix_complete'] and m['complete_cells']==m['required_cells']==6 and len(m['result_SHA256'])==6
 rows=[]
 for key,pin in m['result_SHA256'].items():
  path=scope.STUDY/'results'/(key+'.json');assert b.sha(path)==pin;row=scope.read(path)
  assert row['attempt']==key and row['complete'] and row['independent_CPU_saved_score_audit'] and row['native_saved_state_matches_GPU_restore']
  assert row['benchmark_protocol_SHA256']==b.sha(scope.PROTOCOL)
  rows.append(row)
 summary=summarize(rows)
 for seed in (8,9,10):
  pair=[r for r in rows if r['seed']==seed]
  assert pair[0]['baseline_metrics']==pair[1]['baseline_metrics'],'Matched-arm resampling streams differ'
 q=dict(status='ALL_SIX_FRESH_THETA_INITIALIZATION_FITS_CALIBRATION_COMPARISON_COMPLETE',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),complete_fresh_fits=6,training_seeds=[8,9,10],updates_per_fit=20000,**summary,all_seed_outcomes=[dict(arm=r['arm'],seed=r['seed'],metrics=r['metrics'],baseline_metrics=r['baseline_metrics'],charged_GPU_seconds=r['charged_GPU_seconds']) for r in sorted(rows,key=lambda r:(r['seed'],r['arm']))],actual_GPU_seconds=sum(r['charged_GPU_seconds'] for r in rows),protocol_SHA256=b.sha(scope.PROTOCOL),matrix_SHA256=b.sha(matrix),result_SHA256=m['result_SHA256'],fresh_model_initializations=True,four_reused_calibration_donors=True,calibration_only=True,external45_candidate_unchanged=True,negative_results_retained=True,source_SHA256=b.sha(__file__),limitations='Fresh training seeds are optimization replicates, not new biological donors. This is one preregistered fitting-diagnostic-motivated initialization hypothesis. Four reused calibration donors provide no independent generalization or SOTA test. New external evaluation would require a separately frozen comparison.',full_goal_complete=False,SOTA_claim=False)
 path=scope.STUDY/'comparison.json';assert not path.exists();path.write_text(json.dumps(q,indent=2,allow_nan=False)+'\n');print(json.dumps({k:q[k] for k in ('status','complete_fresh_fits','calibration_preferred_arm','paired_mean_differences')}))
if __name__=='__main__':main()
