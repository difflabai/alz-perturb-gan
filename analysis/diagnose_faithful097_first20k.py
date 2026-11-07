"""Read-only calibration mixture decomposition for the first corrected20k run."""
import datetime,hashlib,json,os,sys
from pathlib import Path
ROOT=Path('/ml2/alz');P=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark')
sys.path[:0]=list(map(str,[ROOT,P/'current_base_20261029/runtime/lib/python3.12/site-packages',P/'.venv312/lib/python3.12/site-packages']))
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
import numpy as np
from scipy.spatial.distance import cdist
from calibration import targets,cell_energy

def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def components(g,r):
 gg=np.array([[cdist(a,b).mean() for b in g] for a in g]);gr=np.array([[cdist(a,b).mean() for b in r] for a in g]);rr=np.array([[cdist(a,b).mean() for b in r] for a in r])
 contributions=(gr+gr.T-gg-rr)/16
 pooled=cell_energy(g.reshape(-1,50),r.reshape(-1,50));np.testing.assert_allclose(contributions.sum(),pooled,rtol=0,atol=5e-13)
 stage=np.diag(contributions)*16
 for i in range(4):np.testing.assert_allclose(stage[i],cell_energy(g[i],r[i]),rtol=0,atol=5e-13)
 return dict(pooled_energy_distance=pooled,weighted_diagonal_contribution=float(np.trace(contributions)),weighted_cross_stage_contribution=float(contributions.sum()-np.trace(contributions)),per_stage_distance=stage.tolist(),contribution_matrix=contributions.tolist(),expected_GG_distances=gg.tolist(),expected_GR_distances=gr.tolist(),expected_RR_distances=rr.tolist(),centroid_error_norms=np.linalg.norm(g.mean(1)-r.mean(1),axis=1).tolist())
def main():
 attempt='CAL_GPU0_V1_faith097_f000_f0_pg_s0';result=ROOT/'analysis/faithful097/results'/(attempt+'.json');q=read(result);assert q['finite'] and q['from_scratch'] and q['physical_start_update']==0 and q['new_phase_model_updates']==20000 and q['clean_settlement']
 out=ROOT/'runs'/attempt;_,_,real=targets();r=real['by_stage'];assert r.shape==(4,375,50);np.testing.assert_array_equal(r.reshape(-1,50),real['equal_stage'])
 pins={str(result):sha(result)};rows=[]
 for rep in (0,1):
  row=dict(replicate=rep)
  for name,prefix in [('pg','r'),('resampling','resampling_r')]:
   folder=out/(prefix+str(rep));metric=read(folder/'metrics.json');path=folder/'arrays.npz';assert sha(path)==metric['arrays_SHA256']==q['files_SHA256'][str(path)];pins[str(path)]=sha(path)
   with np.load(path,allow_pickle=False) as a:g=a['generated_cells_by_stage']
   value=components(g,r);np.testing.assert_allclose(value['pooled_energy_distance'],metric['cell_energy_distance'],rtol=0,atol=5e-13);np.testing.assert_allclose(value['per_stage_distance'],metric['per_stage_cell_energy_distance'],rtol=0,atol=5e-13);row[name]=value
  row['PG_minus_resampling']={k:row['pg'][k]-row['resampling'][k] for k in ('pooled_energy_distance','weighted_diagonal_contribution','weighted_cross_stage_contribution')};rows.append(row)
 mean_delta={key:float(np.mean([r['PG_minus_resampling'][key] for r in rows])) for key in rows[0]['PG_minus_resampling']}
 np.testing.assert_allclose(mean_delta['pooled_energy_distance'],q['metrics']['cell_energy_distance']-q['baseline_metrics']['cell_energy_distance'],rtol=0,atol=5e-13)
 result=dict(status='FIRST_FRESH_RELEASED20K_CALIBRATION_MIXTURE_DECOMPOSITION_VERIFIED',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),attempt=attempt,training_seed=0,endpoint=20000,formulation='released097_full512row_SBNP_bound_thermal_nonEMA',metrics=q['metrics'],matched_resampling=q['baseline_metrics'],PG_minus_resampling_mean=mean_delta,replicates=rows,actual_GPU_seconds=q['charged_GPU_seconds'],new_optimizer_updates=0,calibration_only=True,evaluation_target_values_read=False,all_four_baseline_fits_complete=False,actual_new_model_trained=False,interpretation='The pooled energy distance is exactly the sum of16equal-stage weighted within/cross-stage terms. Stage-specific improvement need not improve the pooled mixture; cross-stage terms are diagnostics, not standalone distances or proof of causation.',source_SHA256={str(Path(__file__).resolve()):sha(__file__),str(ROOT/'calibration.py'):sha(ROOT/'calibration.py')},input_SHA256=pins,SOTA_claim=False,full_goal_complete=False)
 dest=ROOT/'analysis/faithful097_first20k_mixture_diagnostic_20261107.json';assert not dest.exists();dest.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(dict(status=result['status'],PG_minus_resampling_mean=mean_delta,PG_energy_score=q['metrics']['energy_score'],resampling_energy_score=q['baseline_metrics']['energy_score'],actual_GPU_seconds=q['charged_GPU_seconds'])))
if __name__=='__main__':main()
