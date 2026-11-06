"""Read-only CPU resampling controls for the four fresh initialization seeds."""
import datetime,json,os,sys
from pathlib import Path
ROOT=Path('/ml2/alz');HERE=ROOT/'distribution097b_runtime'
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
sys.path.insert(0,str(HERE))
import binding as b

def main():
 proof=ROOT/'analysis/confirmation097_resampling_controls_CPU_20261106.json'
 out=ROOT/'artifacts/confirmation097-resampling-controls'
 assert not proof.exists() and not out.exists()
 auth=b.read(ROOT/'artifacts/allocations/CAL_GPU0_V1_dist097_d003_f0_pg_s0/authorization.json')
 b.source_closure(auth,b.NATIVE_LAYOUT)  # Source/data check, not a GPU admission.
 sys.path.extend(map(str,b.native_site_roots(auth['cell'],auth['files_SHA256'])))
 sys.path[:0]=[str(ROOT),str(b.SCIENCE)]
 import torch,numpy as np
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 from sampling_context import Context
 from generation import iter_resampling
 from calibration import evaluate,score,average
 out.mkdir();ctx=Context(b.SCIENCE/'caches/fold0','cpu');controls={};pins={};reused=[];new=[]
 for seed in range(1,5):
  records=[]
  for rep in (0,1):
   if seed in (1,2):
    folder=ROOT/f'artifacts/calibration_resampling_seed{seed}'/f'r{rep}'
    record=b.read(folder/'metrics.json')
   else:
    folder=out/f's{seed}'/f'r{rep}'
    record=evaluate(ctx,'hierarchical_resampling',seed,rep,lambda schedule,c,s:iter_resampling(ctx,schedule,c,s),folder)
   assert record['training_seed']==seed and record['generation_replicate']==rep and record['generated_cells']==320000
   assert record['arrays_SHA256']==b.sha(folder/'arrays.npz')
   with np.load(folder/'arrays.npz',allow_pickle=False) as a:measured=score(a['stage_synthetic_summaries'],a['generated_cells_by_stage'])
   for key,value in measured.items():
    if isinstance(value,(int,float,list)) and key!='donors':np.testing.assert_allclose(record[key],value,rtol=0,atol=1e-10)
   for p in (folder/'metrics.json',folder/'arrays.npz'):pins[str(p)]=b.sha(p)
   records.append(record)
  controls[str(seed)]=average(records)
  (reused if seed in (1,2) else new).append(seed)
  print(json.dumps(dict(seed=seed,energy_score=controls[str(seed)]['energy_score'],cell_energy_distance=controls[str(seed)]['cell_energy_distance'],CUDA_initialized=torch.cuda.is_initialized())),flush=True)
 assert not torch.cuda.is_initialized()
 sources=[Path(__file__),ROOT/'calibration.py',b.SCIENCE/'sampling_context.py',b.SCIENCE/'generation.py',b.SCIENCE/'contracts.py',b.PROJECT/'evaluation.py',b.PROJECT/'bench_core.py',b.SCIENCE/'caches/fold0/manifest.json',ROOT/'data/calibration_fold0/manifest.json']
 q=dict(status='ALL_FOUR_MATCHED_FRESH_SEED_RESAMPLING_CONTROLS_CPU_VERIFIED',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        metrics_by_seed=controls,reused_control_seeds_independently_rescored=reused,new_CPU_resampling_control_seeds=new,
        generation_replicates_per_seed=2,complete320k_cell_replicates_verified=8,source_SHA256={str(p):b.sha(p) for p in sources},files_SHA256=pins,
        new_generative_model_fitted=False,optimizer_updates=0,CUDA_initialized=False,GPU_allocation_created=False,real_evaluation_target_values_read=False,
        limitation='Resampling of frozen optimization-cell PCs with common training-derived donor/depth traces, scored only on the same four calibration donors. This is a matched Monte Carlo control, not an independent biological confirmation or a new PG training run.')
 proof.write_text(json.dumps(q,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
