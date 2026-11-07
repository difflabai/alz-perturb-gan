"""Decompose scored closed calibration endpoints; no fitting or new samples."""
import datetime,hashlib,json,os,runpy,sys
from pathlib import Path
ROOT=Path('/ml2/alz');PROJECT=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark')
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
sys.path[:0]=[str(ROOT),str(PROJECT/'current_base_20261029/runtime/lib/python3.12/site-packages'),str(PROJECT/'.venv312/lib/python3.12/site-packages')]
import numpy as np,torch
from scipy.spatial.distance import cdist
from calibration import targets
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def decompose(result_path):
 r=read(result_path);assert r['finite'] and r['endpoint']==20000 and r['clean_settlement'] and r['owned_processes_and_lease_absent']
 auth=read(ROOT/'artifacts/allocations'/r['attempt']/'authorization.json');out=Path(auth['output_dir'])
 assert auth['initial_completed_updates']==0 and auth['source_model_path'] is None and not auth['reuse_completed_model']
 _,target,real_cells=targets();real=real_cells['equal_stage'];stage=target['stage'];truth=target['summary'];reps=[]
 for rep in (0,1):
  dest=out/f'r{rep}';metrics=read(dest/'metrics.json');assert metrics==r['replicate_metrics'][rep]
  with np.load(dest/'arrays.npz',allow_pickle=False) as a:
   y=a['stage_synthetic_summaries'].astype(np.float64);cells=a['generated_cells_by_stage'].reshape(-1,50).astype(np.float64)
  cross=float(2*cdist(cells,real).mean());within=float(cdist(cells,cells).mean());realwithin=float(cdist(real,real).mean())
  np.testing.assert_allclose(cross-within-realwithin,metrics['cell_energy_distance'],rtol=0,atol=1e-10)
  accuracy=[];diversity=[]
  for s,t in zip(stage,truth):
   accuracy.append(float(np.linalg.norm(y[s]-t,axis=1).mean()))
   diversity.append(float(.5*cdist(y[s],y[s]).sum()/(200*199)))
  np.testing.assert_allclose(np.array(accuracy)-diversity,metrics['per_donor_energy_score'],rtol=0,atol=1e-10)
  reps.append(dict(replicate=rep,cross_distance_twice=cross,generated_within_distance=within,target_within_distance=realwithin,
    cell_energy_distance=cross-within-realwithin,donor_accuracy_distance=accuracy,donor_diversity_reward=diversity,
    energy_score=float(np.mean(np.array(accuracy)-diversity)),
    pooled_generated_PC_variance_trace=float(cells.var(0,ddof=0).sum()),pooled_real_PC_variance_trace=float(real.var(0,ddof=0).sum()),
    generated_stage_composition_means=np.square(y[:,:,:6]).mean(1).tolist(),
    real_compositions_by_donor=np.square(truth[:,:6]).tolist(),metrics_SHA256=sha(dest/'metrics.json'),arrays_SHA256=sha(dest/'arrays.npz')))
 state=runpy.run_path(str(ROOT/'analysis/diagnose_fresh20k_training.py'),run_name='read_only_endpoint_diagnostics')['summarize'](out/'fit')
 return dict(result_path=str(result_path),result_SHA256=sha(result_path),candidate_id=r['candidate_id'],seed=r['seed'],
   donor_order=target['donors'].tolist(),donor_stages=stage.tolist(),metrics=r['metrics'],replicate_decompositions=reps,training_state_diagnostics=state)
def main():
 paths=[ROOT/'analysis/distribution097/results/CAL_GPU0_V1_dist097_d000_f0_pg_s0.json']
 paths+=sorted((ROOT/'analysis/confirmation097/results').glob('*.json'))
 rows=[decompose(p) for p in paths if read(p)['finite']]
 assert not torch.cuda.is_initialized()
 q=dict(status='READ_ONLY_NATIVE097_CLOSED_ENDPOINT_SEED_DIAGNOSTICS',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
   rows=rows,source_SHA256={str(Path(__file__).resolve()):sha(__file__),str(ROOT/'analysis/diagnose_fresh20k_training.py'):sha(ROOT/'analysis/diagnose_fresh20k_training.py')},
   no_model_or_optimizer_constructed=True,no_new_samples_generated=True,heldout_values_read=False,CUDA_initialized=False,SOTA_claim=False,
   interpretation='Arithmetic decomposition and finite training/particle diagnostics only. Different initialization seeds change the full training trajectory. Differences in cross distance, generated spread, means or gradient summaries are associations, not causal evidence for any individual hyperparameter. No checkpoint is selected or reused for training.')
 path=ROOT/'analysis/confirmation097_completed_seed_diagnostics_20261107.json';path.write_text(json.dumps(q,indent=2,allow_nan=False)+'\n')
 print(json.dumps(dict(path=str(path),completed_seed_count=len(rows),seeds=[r['seed'] for r in rows])))
if __name__=='__main__':main()
