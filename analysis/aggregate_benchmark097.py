"""Distribution-primary comparison of the complete frozen45-fit benchmark.

Real evaluation values are read only after all required cells are complete.
Donor uncertainty is conditional on fixed fitted models and reservoirs; the
historical/crossfold selection limitations remain explicit in every report.
"""
import datetime,hashlib,json,os,re,subprocess,sys
from pathlib import Path
ROOT=Path('/ml2/alz');PROJECT=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark')
SCIENCE=Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/core-comparison-20261103')
sys.path[:0]=[str(ROOT/'benchmark097_runtime'),str(ROOT/'analysis'),str(SCIENCE),str(PROJECT/'current_base_20261029/runtime/lib/python3.12/site-packages'),str(PROJECT/'.venv312/lib/python3.12/site-packages')]
import numpy as np
import preset_scope as scope
from distribution_primary_statistics import paired_components,average_fixed_models,donor_cluster_interval

METHODS=('pg','scvi','cfgen','hierarchical_resampling')
KEYS={(f,s,r) for f in range(3) for s in range(5) for r in (0,1)}
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def summarize_points(records):
 if set(records)!=KEYS:raise ValueError('Complete threefold/fiveseed/tworeplicate point matrix required')
 folds=[];energies=[];seed_rows=[];outside=0
 for fold in range(3):
  values=[records[fold,seed,rep] for seed in range(5) for rep in (0,1)]
  donors=values[0]['donors']
  if any(m['donors']!=donors for m in values):raise ValueError('Target donor order changed across fitted models')
  e=np.asarray([m['per_donor_energy_score'] for m in values],dtype=float)
  if e.shape!=(10,len(donors)) or not np.isfinite(e).all():raise ValueError('Finite complete donor scores required')
  if any(not np.isclose(m['energy_score'],np.mean(m['per_donor_energy_score']),rtol=0,atol=1e-10) for m in values):raise ValueError('Donor-energy aggregate mismatch')
  energy=e.mean(0);energies.extend(energy.tolist())
  metrics={k:float(np.mean([m[k] for m in values])) for k in ('cell_energy_distance','spread_ratio','within_condition_spread_ratio','coverage80','within_stage_spread_ratio_fit_reference')}
  folds.append(dict(fold=fold,donors=donors,energy_score=float(energy.mean()),per_donor_energy_score=energy.tolist(),**metrics))
  for seed in range(5):
   pair=[records[fold,seed,rep] for rep in (0,1)]
   row=dict(fold=fold,seed=seed,**{k:float(np.mean([m[k] for m in pair])) for k in ('cell_energy_distance','energy_score','spread_ratio','coverage80')})
   outside+=int(not .7<=row['spread_ratio']<=1.3);seed_rows.append(row)
 return dict(cell_energy_distance=float(np.mean([f['cell_energy_distance'] for f in folds])),
             pooled_donor_energy_score=float(np.mean(energies)),equal_fold_energy_score=float(np.mean([f['energy_score'] for f in folds])),
             mean_spread_ratio=float(np.mean([f['spread_ratio'] for f in folds])),
             mean_within_condition_spread_ratio=float(np.mean([f['within_condition_spread_ratio'] for f in folds])),
             mean_within_stage_spread_ratio_fit_reference=float(np.mean([f['within_stage_spread_ratio_fit_reference'] for f in folds])),
             mean_coverage80=float(np.mean([f['coverage80'] for f in folds])),models_outside_spread_band=outside,
             folds=folds,fold_seed_rows=seed_rows)

def contrast_interval(pg,other,targets,*,draws=10000):
 components=[];stages=[]
 for fold in range(3):
  target=targets[fold];items=[]
  for seed in range(5):
   for rep in (0,1):
    items.append(paired_components(pg[fold,seed,rep],other[fold,seed,rep],target['cells'],target['labels'],target['donors']))
  components.append(average_fixed_models(items));stages.append(target['stages'])
 return donor_cluster_interval(components,stages,draws=draws,seed=20261106,interval_percent=98.33333333333333)

def load_complete_audits():
 scope.protocol()
 study=ROOT/'analysis/benchmark097';matrix=read(study/'matrix_complete.json')
 assert matrix['matrix_complete'] and matrix['complete_cells']==matrix['required_cells']==45
 assert matrix['protocol_SHA256']==sha(scope.PROTOCOL) and matrix['all_new_training_from_scratch']
 assert read(ROOT/'artifacts/benchmark097-fullmatrix/progress.json')['activation_reverted']
 assert not read(scope.PLAN)['enabled']
 identity=read(ROOT/'artifacts/benchmark097-fullmatrix/identity.json');activation=identity['activation']
 assert re.fullmatch('[0-9a-f]{7,40}',activation)
 full_activation=subprocess.check_output(['git','rev-parse',activation],cwd=ROOT,text=True).strip()
 last_plan_message=subprocess.check_output(['git','log','-1','--format=%B','--','benchmark097_runtime/plan.json'],cwd=ROOT,text=True)
 assert 'reverts commit '+full_activation in last_plan_message
 assert subprocess.check_output(['git','diff','--name-only',activation+'^',activation],cwd=ROOT,text=True).splitlines()==['benchmark097_runtime/plan.json']
 assert json.loads(subprocess.check_output(['git','show',activation+':benchmark097_runtime/plan.json'],cwd=ROOT,text=True))==scope.ENABLED
 budget=read(PROJECT/'project_gpu_budget.json');assert not budget['open_reservations']
 assert not read(ROOT/'artifacts/gpu1-benchmark097-fullmatrix-20261106/live_registry/registry.json')['jobs']
 rows={}
 for fold in range(3):
  for seed in range(5):
   for method in ('pg','scvi','cfgen'):
    key=f'CAL_GPU0_V1_external097_f{fold}_{method}_s{seed}';path=study/'results'/(key+'.json');q=read(path)
    assert matrix['result_SHA256'][key]==sha(path)
    assert q['complete'] and q['status']=='EXTERNAL097_NATIVE_CHECKPOINT_COUNTS_SCORES_CLEAN_SETTLEMENT_AUDIT_PASS'
    assert q['cell']==scope.cell(fold,method,seed) and q['benchmark_protocol_SHA256']==sha(scope.PROTOCOL)
    assert q['all_new_training_from_scratch'] and q['initial_completed_updates']==0 and not q['reuse_completed_model']
    folder=ROOT/'artifacts/allocations'/key;deployment=read(folder/'deployment.json');auth=read(deployment['authorization_path'])
    assert sha(deployment['authorization_path'])==deployment['authorization_SHA256']
    assert auth['cell']==q['cell'] and auth['fixed_endpoint']==scope.endpoint(method) and auth['new_model_updates']==scope.update_count(auth['cell'])
    assert not auth['reuse_completed_model'] and not auth['is_continuation'] and auth['initial_completed_updates']==0
    assert auth['source_model_path'] is auth['source_model_SHA256'] is None
    for file,pin in auth['files_SHA256'].items():
     if Path(file)==scope.PLAN:assert read(file)=={'enabled':False}
     else:assert sha(file)==pin
    ledger=read(auth['ledger_path']);assert len(ledger['jobs'])==1;job=ledger['jobs'][0]
    assert job['status']=='completed' and job['exit_code']==0 and job['reservation_released']
    settled=read(Path(auth['ledger_path']).with_name(key+'-settlement.json'))
    assert settled['uncertain'] is None and settled['settlement']['status']=='settled'
    assert np.isclose(q['charged_GPU_seconds'],settled['settlement']['charged_GPU_seconds'],rtol=0,atol=1e-8)
    for file,pin in q['files_SHA256'].items():assert sha(file)==pin
    rows[fold,method,seed]=q
 assert len(rows)==45 and set(matrix['result_SHA256'])=={q['attempt'] for q in rows.values()}
 assert np.isclose(sum(q['charged_GPU_seconds'] for q in rows.values()),matrix['actual_GPU_seconds'],rtol=0,atol=1e-7)
 return rows,matrix

def main():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 rows,matrix=load_complete_audits()  # Full-matrix guard precedes target reads.
 targets={};all_donors=[]
 from contracts import checked_cache
 for fold in range(3):
  cache=SCIENCE/'caches'/f'fold{fold}';_,declared=checked_cache(cache,'evaluation')
  with np.load(cache/'evaluation/metadata.npz',allow_pickle=False) as m:
   full=np.load(cache/'evaluation/real_pcs.npy',allow_pickle=False)
   assert len(m['donor'])==len(full)
   idx=np.random.default_rng(20261026).choice(len(full),min(1500,len(full)),replace=False)
   raw=dict(zip(m['donors'],m['raw_c']));donors=declared['evaluation_donors']
   targets[fold]=dict(cells=full[idx],labels=m['donor'][idx],donors=donors,stages=np.asarray([int(round(raw[d][3]*3)) for d in donors]))
   assert set(targets[fold]['labels'].tolist())<=set(donors)
   all_donors.extend(donors)
 assert len(all_donors)==len(set(all_donors))==57
 records={m:{} for m in METHODS};reservoirs={m:{} for m in METHODS};baseline_arrays={};swaps={}
 for (fold,method,seed),audit in rows.items():
  out=ROOT/'runs'/audit['attempt']
  for rep in (0,1):
   for name,actual_method in ((f'evaluation_r{rep}',method),(f'resampling_r{rep}','hierarchical_resampling')):
    dest=out/name;m=read(dest/'metrics.json');key=(fold,seed,rep)
    assert m['donors']==targets[fold]['donors'] and m['generated_cells']==320000 and m['arrays_SHA256']==sha(dest/'arrays.npz')
    with np.load(dest/'arrays.npz',allow_pickle=False) as a:
     arrays={k:a[k].copy() for k in a.files}
    np.testing.assert_array_equal(arrays['real_cell_reservoir'],targets[fold]['cells'])
    if actual_method=='hierarchical_resampling' and key in baseline_arrays:
     assert set(arrays)==set(baseline_arrays[key])
     for k,value in arrays.items():np.testing.assert_array_equal(value,baseline_arrays[key][k])
     assert m['energy_score']==records[actual_method][key]['energy_score'] and m['cell_energy_distance']==records[actual_method][key]['cell_energy_distance']
     continue
    if actual_method=='hierarchical_resampling':baseline_arrays[key]=arrays
    records[actual_method][key]=m;reservoirs[actual_method][key]=arrays['generated_cell_reservoir']
  m=read(out/'stage_swap_r0/metrics.json')
  swaps[fold,method,seed]=dict(energy_score_delta=m['energy_score']-records[method][fold,seed,0]['energy_score'],
                              cell_energy_distance_delta=m['cell_energy_distance']-records[method][fold,seed,0]['cell_energy_distance'])
 summaries={method:summarize_points(records[method]) for method in METHODS}
 comparisons={}
 for other in METHODS[1:]:
  delta=summaries['pg']['cell_energy_distance']-summaries[other]['cell_energy_distance']
  missing={str(f):sorted(set(t['donors'])-set(t['labels'].tolist())) for f,t in targets.items()}
  if any(missing.values()):
   interval=None;unavailable='Declared target donors missing from original reservoir: '+json.dumps(missing)
  else:
   interval=contrast_interval(reservoirs['pg'],reservoirs[other],targets)
   np.testing.assert_allclose(interval['mean_delta'],delta,rtol=0,atol=1e-10)
   unavailable=None
  comparisons[other]=dict(PG_minus_other_primary_distance=delta,conditional_real_donor_cluster_interval=interval,interval_unavailable_reason=unavailable,
      paired_fold_seed_deltas=[dict(fold=f,seed=s,delta=float(np.mean([records['pg'][f,s,r]['cell_energy_distance']-records[other][f,s,r]['cell_energy_distance'] for r in (0,1)]))) for f in range(3) for s in range(5)])
 supported=all(c['conditional_real_donor_cluster_interval'] is not None and c['conditional_real_donor_cluster_interval']['upper']<0 for c in comparisons.values())
 result=dict(status='COMPLETE_FROZEN45_MODEL_DISTRIBUTION_PRIMARY_COMPARISON_REQUIRES_FINAL_REVIEW',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
             required_cells=45,complete_cells=45,methods=summaries,ranking_by_primary_distance=sorted(METHODS,key=lambda m:summaries[m]['cell_energy_distance']),
             PG_pairwise_primary_contrasts=comparisons,conditional_distribution_win_supported_against_all_three=supported,
             stage_swap_diagnostics={f'f{f}_{m}_s{s}':v for (f,m,s),v in swaps.items()},
             real_donor_reservoir_counts={str(f):{d:int(np.sum(t['labels']==d)) for d in t['donors']} for f,t in targets.items()},
             actual_GPU_seconds=matrix['actual_GPU_seconds'],protocol_SHA256=sha(scope.PROTOCOL),
             source_SHA256={str(p):sha(p) for p in (Path(__file__),ROOT/'analysis/distribution_primary_statistics.py',ROOT/'analysis/distribution097_external_evaluation_plan_20261106.json')},
             all_new_models_from_scratch=True,no_heldout_driven_parameter_reselection=True,
             independent_population_confirmation=False,automatic_SOTA_claim=False,SOTA_claim=False,full_goal_complete=False,
             limitations='Historically exposed57-donor cohort, crossfold calibration/evaluation overlap, overlapping fitting folds and unequal method-tuning budgets. Paired98.333% intervals use10000 within-stage real-donor cluster draws with fixed models/reservoirs; cells, generation replicates and training seeds are not biological replicates. An unavailable interval remains unavailable; no silent donor removal. Report all negative outcomes and energy/spread tradeoffs. Final requirement-by-requirement review and a properly scoped claim remain necessary.')
 path=ROOT/'analysis/benchmark097/comparison.json';assert not path.exists();path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 print(json.dumps(dict(path=str(path),ranking=result['ranking_by_primary_distance'],conditional_distribution_win_supported=supported,SOTA_claim=False)))

if __name__=='__main__':main()
