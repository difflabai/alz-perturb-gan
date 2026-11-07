"""Independently rescore existing seed0/1 controls; no new count generation."""
import datetime,hashlib,json,os,sys
from pathlib import Path
ROOT=Path('/ml2/alz');PROJECT=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark')
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
sys.path[:0]=[str(ROOT),str(PROJECT/'current_base_20261029/runtime/lib/python3.12/site-packages'),str(PROJECT/'.venv312/lib/python3.12/site-packages')]
import numpy as np
from calibration import score,average
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 output=ROOT/'analysis/refinement097/resampling_controls_CPU.json';assert not output.exists()
 metrics={};pins={}
 for seed in (0,1):
  records=[]
  for rep in (0,1):
   folder=ROOT/('artifacts/calibration_resampling_seed0_validation_replay' if seed==0 else 'artifacts/calibration_resampling_seed1')/f'r{rep}';m=read(folder/'metrics.json')
   assert m['training_seed']==seed and m['generation_replicate']==rep and m['method']=='hierarchical_resampling' and m['generated_cells']==320000
   assert m['arrays_SHA256']==sha(folder/'arrays.npz')
   with np.load(folder/'arrays.npz',allow_pickle=False) as a:s=score(a['stage_synthetic_summaries'],a['generated_cells_by_stage'])
   if seed==0:
    original=ROOT/'artifacts/calibration_resampling_seed0'/f'r{rep}/arrays.npz'
    with np.load(original,allow_pickle=False) as old,np.load(folder/'arrays.npz',allow_pickle=False) as replay:
     assert set(old.files)==set(replay.files)
     for key in old.files:np.testing.assert_array_equal(old[key],replay[key])
    pins[str(original)]=sha(original)
   for k,v in s.items():
    if isinstance(v,(int,float,list)) and k!='donors':np.testing.assert_allclose(v,m[k],rtol=0,atol=1e-10)
   for p in (folder/'arrays.npz',folder/'metrics.json'):pins[str(p)]=sha(p)
   records.append(m)
  metrics[str(seed)]=average(records)
 old=read(ROOT/'analysis/confirmation097_resampling_controls_CPU_20261106.json');assert metrics['1']==old['metrics_by_seed']['1']
 old0=read(ROOT/'analysis/baseline_calibration_seed0.json')
 for k,v in metrics['0'].items():assert v==old0[k]
 assert 'torch' not in sys.modules
 files=[Path(__file__),ROOT/'calibration.py',ROOT/'data/calibration_fold0/manifest.json',ROOT/'analysis/confirmation097_resampling_controls_CPU_20261106.json',ROOT/'analysis/baseline_calibration_seed0.json']
 proof=dict(status='TWO_MATCHED_SELECTION_SEED_RESAMPLING_CONTROLS_CPU_VERIFIED',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),metrics_by_seed=metrics,
   selection_seeds=[0,1],independently_rescored320k_cell_replicates=4,source_SHA256={str(p):sha(p) for p in files},files_SHA256=pins,
   seed0_existing_validation_replay_bitwise_equal_to_original_arrays=True,
   new_count_generation=False,new_model_fitted=False,optimizer_updates=0,CUDA_initialized=False,GPU_allocation_created=False,heldout_values_read=False)
 output.write_text(json.dumps(proof,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(path=str(output),replicates=4,GPU_allocation_created=False)))
if __name__=='__main__':main()
