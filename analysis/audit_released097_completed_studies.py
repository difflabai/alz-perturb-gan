"""Final evidence audit of the prespecified studies; never an automatic SOTA claim.

No quality report is opened before both scientific reports and all six fresh
calibration outcomes exist. This script does not start, alter or resume training.
"""
import argparse,datetime,hashlib,json,math,os,statistics,subprocess,sys
from pathlib import Path
ROOT=Path('/ml2/alz');PROJECT=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark')
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def required_paths(root=ROOT):
 root=Path(root)
 return [root/p for p in ('analysis/benchmark097/matrix_complete.json','analysis/benchmark097/comparison.json','analysis/benchmark097_calibration_excluded53_report.json','analysis/benchmark097_infra1/scientific_reports_completed.json','artifacts/benchmark097-infra1-pipeline/completed.json','analysis/scalarcal097/matrix_complete.json','analysis/scalarcal097/comparison.json','artifacts/scalarcal097-after-recovered-science-v5/completed.json')]
def readiness(root=ROOT):
 paths=required_paths(root);missing=[str(p) for p in paths if not p.is_file()]
 return dict(status='INCOMPLETE_PRESPECIFIED_STUDIES' if missing else 'INPUT_PATHS_PRESENT_NOT_A_COMPLETION_CLAIM',missing_paths=missing,quality_values_read=False,full_goal_complete=False,SOTA_claim=False)
def close(a,b):assert math.isclose(float(a),float(b),rel_tol=0,abs_tol=1e-10),(a,b)
def independent_cell_ED(a,b):
 import numpy as np
 from scipy.spatial.distance import cdist,pdist
 a=np.asarray(a);b=np.asarray(b)
 assert a.ndim==b.ndim==2 and a.shape[1]==b.shape[1] and min(len(a),len(b))>0 and np.isfinite(a).all() and np.isfinite(b).all()
 # Condensed within-sample sums independently reproduce the declared V statistic.
 return float(2*cdist(a,b).sum()/(len(a)*len(b))-2*pdist(a).sum()/len(a)**2-2*pdist(b).sum()/len(b)**2)
def validate_summary(value,donors_by_fold):
 assert len(value['fold_seed_rows'])==15 and {(x['fold'],x['seed']) for x in value['fold_seed_rows']}=={(f,s) for f in range(3) for s in range(5)}
 close(statistics.mean(x['cell_energy_distance'] for x in value['fold_seed_rows']),value['cell_energy_distance'])
 close(statistics.mean(x['spread_ratio'] for x in value['fold_seed_rows']),value['mean_spread_ratio'])
 close(statistics.mean(x['coverage80'] for x in value['fold_seed_rows']),value['mean_coverage80'])
 assert [x['fold'] for x in value['folds']]==list(range(3));pooled=[]
 for x in value['folds']:
  assert x['donors']==donors_by_fold[x['fold']] and len(x['per_donor_energy_score'])==len(x['donors'])
  close(statistics.mean(x['per_donor_energy_score']),x['energy_score']);pooled.extend(x['per_donor_energy_score'])
 close(statistics.mean(pooled),value['pooled_donor_energy_score'])
 close(statistics.mean(x['energy_score'] for x in value['folds']),value['equal_fold_energy_score'])
 return value
def tested_ED_win(summaries,contrasts):
 assert set(summaries)=={'pg','scvi','cfgen','hierarchical_resampling'} and set(contrasts)=={'scvi','cfgen','hierarchical_resampling'}
 supported=[]
 for method,c in contrasts.items():
  delta=summaries['pg']['cell_energy_distance']-summaries[method]['cell_energy_distance']
  close(c.get('PG_minus_other_primary_distance',c.get('PG_minus_other_distance')),delta)
  interval=c.get('conditional_real_donor_cluster_interval',c.get('conditional_donor_cluster_interval'))
  if interval is None:supported.append(False);continue
  close(interval['mean_delta'],delta);assert interval['lower']<=interval['upper'];close(interval['interval_percent'],98.33333333333333)
  assert interval['draws']==10000 and interval['model_and_generation_reservoirs_fixed'] and not interval['independent_population_confirmation']
  assert interval['unit']=='real donor cluster, stratified by fold and stage'
  supported.append(interval['upper']<0)
 return all(supported)
def preferred_arm(arms):return min(('fixed','adaptive'),key=lambda a:arms[a]['cell_energy_distance']['mean'])
def verify_reverted_plan(path):
 path=Path(path);relative=str(path.relative_to(ROOT));assert not read(path)['enabled']
 message=subprocess.check_output(['git','log','-1','--format=%B','--',relative],cwd=ROOT,text=True)
 import re
 matched=re.search(r'reverts commit ([0-9a-f]{40})',message);assert matched,'Expected explicit activation reversion: '+relative
 activation=matched.group(1);names=subprocess.check_output(['git','diff','--name-only',activation+'^',activation],cwd=ROOT,text=True).splitlines();assert names==[relative]
 assert json.loads(subprocess.check_output(['git','show',activation+':'+relative],cwd=ROOT))['enabled']
 assert subprocess.check_output(['git','show',activation+'^:'+relative],cwd=ROOT)==path.read_bytes()
 return activation
def audit(output):
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 ready=readiness();assert not ready['missing_paths'],'All45/57/53 and six fresh outcomes are mandatory before numerical reads'
 sys.path[:0]=[str(ROOT/'analysis'),str(ROOT/'scalarcal097_runtime'),str(PROJECT/'current_base_20261029/runtime/lib/python3.12/site-packages'),str(PROJECT/'.venv312/lib/python3.12/site-packages'),str(ROOT)]
 import numpy as np,preset_scope as scope,binding as b
 from cell_metadata import completed_metadata
 from benchmark097_recovered_lineage import load_complete_audits
 from calibration import targets,score,average
 sys.path.insert(0,str(ROOT/'analysis/adaptive_lr097'))
 from calibration_distance_lr import CalibrationDistanceLR,Config
 prior=scope.prior_study_ready();original,matrix=load_complete_audits();assert len(original)==45 and matrix['physical_attempts']==48
 plans={name:verify_reverted_plan(ROOT/name/'plan.json') for name in ('benchmark097_dual_runtime','benchmark097_balanced_runtime','benchmark097_infra1_runtime','scalarcal097_runtime')}
 assert not read(PROJECT/'project_gpu_budget.json')['open_reservations']
 primary=read(ROOT/'analysis/benchmark097/comparison.json');sensitivity=read(ROOT/'analysis/benchmark097_calibration_excluded53_report.json');plan=read(ROOT/'analysis/benchmark097_calibration_excluded53_plan.json')
 original_donors={f['fold']:f['original_donors'] for f in plan['folds']};kept={f['fold']:f['retained_donors'] for f in plan['folds']}
 assert len(set(d for ds in original_donors.values() for d in ds))==57 and len(set(d for ds in kept.values() for d in ds))==53
 for values,donors in ((primary['methods'],original_donors),(sensitivity['models'],kept)):
  for value in values.values():validate_summary(value,donors)
 for q,values in ((primary,primary['methods']),(sensitivity,sensitivity['models'])):
  key='ranking_by_primary_distance' if q is primary else 'ranking_by_distance';assert q[key]==sorted(values,key=lambda method:values[method]['cell_energy_distance']) and not q['SOTA_claim']
 primary_win=tested_ED_win(primary['methods'],primary['PG_pairwise_primary_contrasts']);sensitivity_win=tested_ED_win(sensitivity['models'],sensitivity['contrasts'])
 close(primary['actual_GPU_seconds'],matrix['actual_GPU_seconds'])
 sm=read(scope.STUDY/'matrix_complete.json');comparison=read(scope.STUDY/'comparison.json');assert sm['matrix_complete'] and sm['complete_cells']==sm['required_cells']==6 and comparison['complete_fresh_fits']==6
 scalar_parent=read(required_paths()[-1]);assert scalar_parent['status']=='SCALARCAL097_COMPLETE_CALIBRATION_REPORT_FINAL_RESEARCH_AUDIT_REMAINS' and scalar_parent['report_path']==str(scope.STUDY/'comparison.json')
 assert comparison['training_seeds']==[5,6,7] and comparison['updates_per_fit']==20000 and comparison['all_six_outcomes_retained'] and comparison['controller_uses_only_calibration_cell_ED'] and not comparison['external_outcomes_used_for_selection'] and comparison['old45_candidate_unchanged']
 assert sm['protocol_SHA256']==comparison['protocol_SHA256']==sha(scope.PROTOCOL)
 manifest,ys,cells=targets();real=cells['equal_stage'].copy();ys.close();cells.close();calibrators={};rows={};charges=[];source_pins={**prior,str(scope.STUDY/'matrix_complete.json'):sha(scope.STUDY/'matrix_complete.json'),str(scope.STUDY/'comparison.json'):sha(scope.STUDY/'comparison.json')}
 for seed in (5,6,7):
  traces=set()
  for arm in ('fixed','adaptive'):
   value=scope.cell(arm,seed);meta=completed_metadata(value);attempt=meta['attempt'];path=Path(meta['source_result_path']);assert sm['result_SHA256'][attempt]==sha(path)
   q=read(path);assert q['complete'] and q['cell']==value and q['arm']==arm and q['all_new_training_from_scratch'] and q['initial_completed_updates']==0 and q['source_checkpoint'] is None
   assert q['independent_CPU_saved_score_and_controller_audit'] and q['native_saved_state_matches_GPU_restore'];close(q['charged_GPU_seconds'],meta['charged_GPU_seconds']);charges.append(meta['charged_GPU_seconds'])
   a=read(meta['authorization_path']);assert a['source_model_path'] is a['source_model_SHA256'] is None and not a['is_continuation'] and not a['reuse_completed_model'] and a['released_hyperparameters']==scope.hyperparameters() and a['variant_config']==scope.CFG
   for name,pin in a['files_SHA256'].items():
    if Path(name).name=='plan.json':assert not read(name)['enabled']
    else:assert sha(name)==pin,name
   for name,pin in q['files_SHA256'].items():assert sha(name)==pin,name
   source_pins.update(q['files_SHA256']);source_pins[meta['authorization_path']]=sha(meta['authorization_path'])
   source_pins[a['ledger_path']]=sha(a['ledger_path'])
   settlement=Path(a['ledger_path']).with_name(attempt+'-settlement.json');source_pins[str(settlement)]=sha(settlement)
   curve=q['calibration_observations'];assert len(curve)==20;control=CalibrationDistanceLR(Config()) if arm=='adaptive' else None
   for step,row in zip(range(1000,20001,1000),curve):
    ob=row['observation'];assert ob['completed_updates']==step and ob['training_seed']==seed and ob['split_role']=='calibration' and ob['metric']=='calibration_cell_energy_distance' and ob['cells_per_replicate']==320000 and ob['calibration_manifest_SHA256']==sha(ROOT/'data/calibration_fold0/manifest.json')
    assert sha(ob['arrays_path'])==ob['arrays_SHA256'];traces.add(ob['trace_SHA256'])
    with np.load(ob['arrays_path'],allow_pickle=False) as z:
     pools=z['generated_cells_by_stage'];assert pools.shape==(2,4,375,50);np.testing.assert_array_equal(z['real_equal_stage_cells'],real);distances=[independent_cell_ED(pool.reshape(-1,50),real) for pool in pools]
    np.testing.assert_allclose(distances,ob['replicate_values'],rtol=0,atol=1e-10);close(statistics.mean(distances),ob['value'])
    decision=None if control is None else control.observe(step,ob['value'],split_role='calibration');assert decision==row['decision']
   records=[];out=Path(a['output_dir'])
   for rep in (0,1):
    with np.load(out/f'r{rep}/arrays.npz',allow_pickle=False) as arrays:records.append(score(arrays['stage_synthetic_summaries'],arrays['generated_cells_by_stage']))
    for stage in range(4):
     with np.load(out/f'r{rep}/stage{stage}_exemplar.npz',allow_pickle=False) as ex:
      c=ex['counts'];assert c.shape==(1,400,2000) and np.isfinite(c).all() and (c>=0).all() and np.equal(c,np.floor(c)).all()
   metrics=average(records)
   for key in ('cell_energy_distance','energy_score','spread_ratio','within_condition_spread_ratio','coverage80'):close(metrics[key],q['metrics'][key])
   np.testing.assert_allclose([x['cell_energy_distance'] for x in records],curve[-1]['observation']['replicate_values'],rtol=0,atol=1e-10)
   rows[seed,arm]=q;source_pins[str(path)]=sha(path)
  assert len(traces)==1,'Declared CRN fitting traces must agree across both arms and all updates'
  assert rows[seed,'fixed']['baseline_metrics']==rows[seed,'adaptive']['baseline_metrics']
 for arm in ('fixed','adaptive'):
  for key in ('cell_energy_distance','energy_score','spread_ratio','within_condition_spread_ratio','coverage80'):
   values=[rows[seed,arm]['metrics'][key] for seed in (5,6,7)];close(statistics.mean(values),comparison['arms'][arm][key]['mean']);close(statistics.stdev(values),comparison['arms'][arm][key]['initialization_SD'])
 assert preferred_arm(comparison['arms'])==comparison['calibration_preferred_arm'];close(sum(charges),comparison['actual_GPU_seconds']);close(sum(charges),sm['actual_GPU_seconds'])
 assert 'torch' not in sys.modules,'The final saved-array audit is CPU numerical analysis, not native model execution'
 record=dict(status='PRESPECIFIED45_LOGICAL48_PHYSICAL_AND_SIX_FRESH_CALIBRATION_STUDIES_INDEPENDENTLY_AUDITED',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),complete_logical_benchmark_models=45,retained_physical_benchmark_attempts=48,retained_infrastructure_failures=matrix['infrastructure_failed_attempts'],original57_primary_preserved=True,prespecified53_complete=True,original57_ranking=primary['ranking_by_primary_distance'],prespecified53_ranking=sensitivity['ranking_by_distance'],original57_PG_ED_win_supported_against_tested_recipes=primary_win,prespecified53_PG_ED_win_supported=sensitivity_win,scalar_fresh20k_fits=6,scalar_observations_independently_recomputed=120,scalar_CRN_generation_replicate_distances_recomputed=240,scalar_preferred_arm=comparison['calibration_preferred_arm'],scalar_calibration_arms=comparison['arms'],scalar_candidate_not_inserted_into_frozen_benchmark=True,benchmark_successful_model_GPU_seconds=matrix['successful_model_GPU_seconds'],benchmark_failed_infrastructure_GPU_seconds=matrix['failed_infrastructure_GPU_seconds'],benchmark_total_GPU_seconds=matrix['actual_GPU_seconds'],scalar_GPU_seconds=sum(charges),actual_plan_reversions_verified=plans,source_SHA256={str(Path(__file__).resolve()):sha(__file__)},input_SHA256=source_pins,no_training_or_GPU_allocation=True,full_goal_complete=False,SOTA_claim=False,remaining_goal_limits=['A preferred scalar-calibration arm requires a new frozen validation study before any claim of external superiority; it cannot replace the original45 candidate retrospectively.','Current count-FM/scLDM and other applicable modern/donor-aware methods remain untested on ALZ; native CPU fixtures are not comparative ALZ scores.','Historical57 cohort exposure, crossfold calibration overlap and unequal search budgets limit any claim to the tested recipes/protocol.','Further PG optimization and independent confirmation remain required if the requested SOTA objective is unsupported by actual results.'])
 output=Path(output);assert output.parent==ROOT/'analysis' and not output.exists();output.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n');return record
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--check-readiness',action='store_true');parser.add_argument('--output');args=parser.parse_args()
 if args.check_readiness:print(json.dumps(readiness(),indent=2));return
 assert args.output,'Explicit new audit output path required';result=audit(args.output);print(json.dumps({k:result[k] for k in ('status','original57_ranking','prespecified53_ranking','scalar_preferred_arm','full_goal_complete','SOTA_claim')}))
if __name__=='__main__':main()
