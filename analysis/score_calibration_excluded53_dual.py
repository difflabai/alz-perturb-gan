"""Prespecified53-donor sensitivity, preserving the original57-donor primary.

Requires the completed45fresh-model matrix and original primary comparison.
No fitting, new generated RNA counts, target-driven reselection, or model reload.
"""
import argparse,datetime,hashlib,json,os,sys
from pathlib import Path
ROOT=Path('/ml2/alz');SCIENCE=Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/core-comparison-20261103');PROJECT=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark')
sys.path[:0]=list(map(str,[ROOT/'analysis',ROOT/'benchmark097_dual_runtime',SCIENCE,PROJECT,PROJECT/'current_base_20261029/runtime/lib/python3.12/site-packages',PROJECT/'.venv312/lib/python3.12/site-packages']))
import numpy as np
import aggregate_benchmark097_dual as core
from scoring import score_summary_arrays
from evaluation import Reservoir
from bench_core import cell_energy
PLAN=ROOT/'analysis/benchmark097_calibration_excluded53_plan.json';OUT=ROOT/'artifacts/benchmark097-calibration-excluded53'
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def validate_plan(p,calibration):
 if p['status']!='PRESPECIFIED_CALIBRATION_EXCLUDED53_SENSITIVITY_BEFORE_NEW_BENCHMARK_VALUES' or not p['preserve_original57_donor_primary_comparison'] or not p['preserve_all45_fresh_model_fits']:raise ValueError('Original full comparison must be preserved')
 if set(p['current_global_calibration_donors'])!=set(calibration) or len(set(calibration))!=4:raise ValueError('Exact current four tuning donors required')
 if [f['fold'] for f in p['folds']]!=[0,1,2]:raise ValueError('All three original folds required')
 all_original=[];all_kept=[]
 for f in p['folds']:
  expected=[d for d in f['original_donors'] if d not in set(calibration)]
  if f['retained_donors']!=expected or f['excluded_calibration_donors']!=[d for d in f['original_donors'] if d in calibration]:raise ValueError('No extra donor removal or calibration leakage')
  all_original+=f['original_donors'];all_kept+=f['retained_donors']
 if len(all_original)!=57 or len(set(all_original))!=57 or len(all_kept)!=53 or len(set(all_kept))!=53 or set(all_kept)&set(calibration):raise ValueError('Exact57original/53retained distinct donors required')
 if not p['no_tuning_on_supplementary_scores'] or not p['no_new_training'] or not p['no_new_generated_counts']:raise ValueError('Read-only fixed-model sensitivity required')
 return p

def repool(stages,donor_stages):
 if len(stages)!=4 or set(np.asarray(donor_stages).tolist())!=set(range(4)):raise ValueError('All four fixed stage distributions required')
 donor_stages=np.asarray(donor_stages);reservoir=Reservoir()
 for s in range(4):
  if stages[s].shape!=(200,400,50) or stages[s].dtype!=np.float32 or not np.isfinite(stages[s]).all():raise ValueError('Complete original float32 stage PCs required')
  for _ in range(int(np.sum(donor_stages==s))):reservoir.add(stages[s])
 return reservoir.cells

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--check-only',action='store_true');args=parser.parse_args();assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 p=validate_plan(read(PLAN),read(ROOT/'data/calibration_fold0/manifest.json')['donors'])
 for f in p['folds']:
  assert sha(f['source_manifest_path'])==f['source_manifest_SHA256'];m=read(f['source_manifest_path']);assert m['evaluation_donors']==f['original_donors'] and not set(f['retained_donors'])&set(m['train_donors'])
 if args.check_only:print(json.dumps(dict(status='METADATA_ONLY53_PLAN_VALID_SOURCE_TARGET_IDS_CHECKED',donors=53,all45_required=True,new_target_values_read=False,GPU_allocation_created=False)));return
 rows,matrix=core.load_complete_audits()  # Mandatory full-matrix gate before np.load.
 primary=ROOT/'analysis/benchmark097/comparison.json';assert read(primary)['complete_cells']==45 and not read(primary)['SOTA_claim']
 assert not OUT.exists();OUT.mkdir()
 records={m:{} for m in core.METHODS};reservoirs={m:{} for m in core.METHODS};targets={};pins={str(PLAN):sha(PLAN),str(primary):sha(primary)};derived={};shape_map={}
 for f in p['folds']:
  fold=f['fold'];cache=SCIENCE/'caches'/f'fold{fold}';meta=np.load(cache/'evaluation/metadata.npz',allow_pickle=False);pc=np.load(cache/'evaluation/real_pcs.npy',mmap_mode='r',allow_pickle=False)
  kept=f['retained_donors'];mask=np.isin(meta['donor'],kept);candidate_indices=np.flatnonzero(mask);chosen=np.random.default_rng(20261026).choice(len(candidate_indices),min(1500,len(candidate_indices)),replace=False);indices=candidate_indices[chosen]
  true_cells=np.asarray(pc[indices]);labels=meta['donor'][indices];assert len(meta['donor'])==len(pc) and set(meta['donor'][mask].tolist())==set(kept) and not set(labels.tolist())&set(p['current_global_calibration_donors']);original_real_indices=np.random.default_rng(20261026).choice(len(pc),min(1500,len(pc)),replace=False);original_real_cells=np.asarray(pc[original_real_indices]);train_pc=np.load(cache/'train/real_pcs.npy',mmap_mode='r',allow_pickle=False)
  for seed in range(5):
   for rep in (0,1):
    baseline_done=False
    for method in ('pg','scvi','cfgen'):
     original=ROOT/'runs'/rows[fold,method,seed]['attempt']
     for source_name,actual_method in [(f'evaluation_r{rep}',method),(f'resampling_r{rep}','hierarchical_resampling')]:
      if actual_method=='hierarchical_resampling' and baseline_done:continue
      folder=original/source_name;metric=read(folder/'metrics.json');source=folder/'arrays.npz';pins[str(source)]=sha(source)
      with np.load(source,allow_pickle=False) as a:saved={k:a[k].copy() for k in a.files}
      assert saved['donors'].tolist()==f['original_donors'];np.testing.assert_array_equal(original_real_cells,saved['real_cell_reservoir']);keep=np.asarray([d in set(kept) for d in saved['donors']]);stage=saved['stage'][keep]
      if fold not in targets:targets[fold]=dict(cells=true_cells,labels=labels,donors=kept,stages=stage)
      else:np.testing.assert_array_equal(targets[fold]['stages'],stage)
      full=[]
      for s in range(4):
       if actual_method=='hierarchical_resampling':
        path=folder/f'stage{s}_trace.npz';trace=np.load(path,allow_pickle=False);full.append(np.asarray(train_pc[trace['row']]))
       else:path=folder/f'stage{s}_all_generated_PCs.npy';full.append(np.load(path,mmap_mode='r',allow_pickle=False))
       pins[str(path)]=sha(path)
      np.testing.assert_array_equal(repool(full,saved['stage']),saved['generated_cell_reservoir'])
      synthetic_cells=repool(full,stage);value=score_summary_arrays(saved['stage_synthetic_summaries'],saved['real_summaries'][keep],stage)
      value.update(status='completed',fold=fold,method=actual_method,training_seed=seed,generation_replicate=rep,donors=kept,cell_energy_distance=cell_energy(synthetic_cells,true_cells),within_stage_spread_ratio_fit_reference=metric['within_stage_spread_ratio_fit_reference'],source_original_metrics_path=str(folder/'metrics.json'),no_new_model_or_counts=True)
      key=(fold,seed,rep);records[actual_method][key]=value;reservoirs[actual_method][key]=synthetic_cells
      dest=OUT/f'f{fold}_{actual_method}_s{seed}_r{rep}.npz';np.savez_compressed(dest,generated_cells=synthetic_cells,real_cells=true_cells,real_labels=labels,real_source_row=indices,donors=np.asarray(kept),stage=stage,real_summaries=saved['real_summaries'][keep],stage_synthetic_summaries=saved['stage_synthetic_summaries']);derived[str(dest)]=sha(dest)
      if actual_method=='hierarchical_resampling':baseline_done=True
  shape_map[str(fold)]=dict(donors=len(kept),real_reservoir_cells=len(true_cells),missing_donors=sorted(set(kept)-set(labels.tolist())))
 summary={m:core.summarize_points(records[m]) for m in core.METHODS};contrasts={}
 for other in core.METHODS[1:]:
  delta=summary['pg']['cell_energy_distance']-summary[other]['cell_energy_distance'];missing={str(f):sorted(set(t['donors'])-set(t['labels'].tolist())) for f,t in targets.items()}
  interval=None if any(missing.values()) else core.contrast_interval(reservoirs['pg'],reservoirs[other],targets)
  if interval is not None:np.testing.assert_allclose(interval['mean_delta'],delta,rtol=0,atol=1e-10)
  contrasts[other]=dict(PG_minus_other_distance=delta,conditional_donor_cluster_interval=interval,missing_donors=missing,paired_fold_seed_deltas=[dict(fold=f,seed=s,delta=float(np.mean([records['pg'][f,s,t]['cell_energy_distance']-records[other][f,s,t]['cell_energy_distance'] for t in (0,1)]))) for f in range(3) for s in range(5)])
 report=dict(status='COMPLETE_PRESPECIFIED_CALIBRATION_EXCLUDED53_SENSITIVITY',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),original57_primary_preserved=True,complete_fresh_generative_models=45,new_model_fits=0,new_generated_RNA_counts=0,models=summary,ranking_by_distance=sorted(summary,key=lambda m:summary[m]['cell_energy_distance']),contrasts=contrasts,target_donor_counts=shape_map,per_model_seed_replicate_metrics={m:{'_'.join(map(str,k)):v for k,v in records[m].items()} for m in core.METHODS},source_SHA256={str(Path(__file__).resolve()):sha(__file__),str(ROOT/'analysis/aggregate_benchmark097_dual.py'):sha(ROOT/'analysis/aggregate_benchmark097_dual.py')},input_SHA256=pins,derived_SHA256=derived,current_tuning_donors_excluded=True,independent_population_confirmation=False,primary_replaced=False,SOTA_claim=False,full_goal_complete=False,limitations=p['claim_limits'])
 dest=ROOT/'analysis/benchmark097_calibration_excluded53_report.json';assert not dest.exists();dest.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(status=report['status'],ranking=report['ranking_by_distance'],new_model_fits=0,new_generated_counts=0)))
if __name__=='__main__':main()
