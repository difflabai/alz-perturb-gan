"""Independently recompute completed pure-only scores, counts and control parity."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path('/ml2/alz');sys.path.insert(0,str(ROOT))
from pure_suite import read,sha,load_entry
from calibration import score,average,dump

def verify(label,attempt):
 entry=load_entry(label);dep=read(ROOT/'artifacts/allocations'/attempt/'deployment.json');out=Path(dep['output_dir']);ledger=Path(dep['ledger_path']);settled=read(ledger.with_name(attempt+'-settlement.json'));job=read(ledger)['jobs'][0]
 assert settled['uncertain'] is None and settled['settlement']['status']=='settled'
 assert job['reservation_released'] and job['status']=='completed' and job['exit_code']==0
 q=read(out/'completed.json');assert q['pure_only'] is True and q['generation']=='direct_iter_pg' and q['real_bag_replacements']==0
 assert q['hybrid_config'] is None and q['new_model_updates']==0 and q['completed_updates']==q['fixed_endpoint']==entry['completed_updates']
 assert q['method']==('pg_40k' if entry['completed_updates']==40000 else 'pg')
 assert q['checkpoint_SHA256']==entry['checkpoint_SHA256']==sha(entry['checkpoint_path'])
 assert q['variant_config']==entry['variant_config'] and q['training_seed']==entry['training_seed'] and q['pure_formulation']==label
 records=[];control=entry['prior_generation_is_pure']
 for rep in [0,1]:
  folder=out/f'r{rep}';m=read(folder/'metrics.json');assert m['arrays_SHA256']==sha(folder/'arrays.npz') and m['generated_cells']==320000
  arrays=np.load(folder/'arrays.npz',allow_pickle=False);recomputed=score(arrays['stage_synthetic_summaries'],arrays['generated_cells_by_stage'])
  for key in ['energy_score','cell_energy_distance','spread_ratio','per_donor_energy_score','per_stage_cell_energy_distance','coverage80']:
   np.testing.assert_allclose(m[key],recomputed[key],rtol=0,atol=1e-10)
  assert m['donors']==q['donors'] and m['method']==q['method']
  if control:
   old=np.load(Path(entry['prior_score_output'])/f'r{rep}'/'arrays.npz',allow_pickle=False)
   assert set(old.files)==set(arrays.files)
   for k in old.files:np.testing.assert_array_equal(arrays[k],old[k])
  for stage in range(4):
   a=np.load(folder/f'stage{stage}_exemplar.npz',allow_pickle=False)['counts']
   assert a.shape==(1,400,2000) and np.isfinite(a).all() and (a>=0).all() and np.equal(a,np.floor(a)).all()
   if control:np.testing.assert_array_equal(a,np.load(Path(entry['prior_score_output'])/f'r{rep}'/f'stage{stage}_exemplar.npz',allow_pickle=False)['counts'])
  records.append(m)
 avg=average(records)
 for key in ['energy_score','cell_energy_distance','spread_ratio','coverage80']:np.testing.assert_allclose(q[key],avg[key],rtol=0,atol=1e-10)
 result={'status':'VERIFIED_PURE_PG_ONLY','formulation':label,'attempt':attempt,'completed_updates':entry['completed_updates'],'checkpoint_SHA256':q['checkpoint_SHA256'],'variant_config':q['variant_config'],'energy_score':q['energy_score'],'cell_energy_distance':q['cell_energy_distance'],'spread_ratio':q['spread_ratio'],'coverage80':q['coverage80'],'real_bag_replacements':0,'new_model_updates':0,'complete_clean_settlement_verified':True,'independent_saved_array_score_and_integer_count_verification':True,'control_arrays_and_count_exemplars_bitwise_equal':True if control else None,'charged_GPU_seconds':settled['settlement']['charged_GPU_seconds'],'calibration_manifest_SHA256':q['calibration_manifest_SHA256'],'trace_manifest_SHA256':q['trace_manifest_SHA256'],'donors':q['donors'],'sampling_contract':q['sampling_contract'],'training_seed':q['training_seed'],'generation_replicates':2,'hybrid_incumbent_unchanged':True}
 dest=ROOT/'analysis'/f'pure_{label}_verification.json';assert not dest.exists();dump(dest,result);print(json.dumps(result,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--formulation',required=True);p.add_argument('--attempt',required=True);a=p.parse_args();verify(a.formulation,a.attempt)
