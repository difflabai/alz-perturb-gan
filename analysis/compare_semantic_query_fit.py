"""Diagnose semantic trial from generated fitting-query arrays only; no held-out targets."""
import datetime,hashlib,json
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist

ROOT=Path('/ml2/alz')

def energy(x,y):
 return float(2*cdist(x,y).mean()-cdist(x,x).mean()-cdist(y,y).mean())

def load(folder,rep,pins):
 path=folder/f'r{rep}'/'arrays.npz';metrics=json.loads(path.with_name('metrics.json').read_text())
 pin=hashlib.sha256(path.read_bytes()).hexdigest();assert pin==metrics['arrays_SHA256'];pins[str(path)]=pin
 with np.load(path,allow_pickle=False) as z:
  summaries=z['stage_synthetic_summaries'];cells=z['generated_cells_by_stage']
 assert summaries.shape==(4,200,26) and cells.shape==(4,375,50)
 assert np.isfinite(summaries).all() and np.isfinite(cells).all()
 return summaries,cells,metrics

def main():
 folders={'PG8':ROOT/'runs/CAL_GPU0_V1_latent8_v1_f0_pg_s0',
          'semantic':ROOT/'runs/CAL_GPU0_V1_semantic_v1_f0_pg_s0'}
 baseline=ROOT/'artifacts/calibration_resampling_seed0_validation_replay';pins={};records=[]
 for rep in (0,1):
  original=ROOT/'artifacts/calibration_resampling_seed0'/f'r{rep}'/'arrays.npz'
  replay=baseline/f'r{rep}'/'arrays.npz'
  with np.load(original,allow_pickle=False) as a,np.load(replay,allow_pickle=False) as b:
   for key in ['stage_synthetic_summaries','generated_cells_by_stage']:np.testing.assert_array_equal(a[key],b[key])
  pins[str(original)]=hashlib.sha256(original.read_bytes()).hexdigest()
 for name,folder in folders.items():
  per_rep=[]
  for rep in (0,1):
   a,ac,am=load(folder,rep,pins);b,bc,bm=load(baseline,rep,pins)
   for key in ['donors','training_seed','sampling_contract','calibration_manifest_SHA256','trace_manifest_SHA256']:
    assert am[key]==bm[key]
   stages=[]
   for stage in range(4):
    pa=np.square(a[stage,:,:6]).mean(0);pb=np.square(b[stage,:,:6]).mean(0)
    stages.append({'stage':stage,'generated_composition':pa.tolist(),'resampling_composition':pb.tolist(),
                   'composition_L1':float(np.abs(pa-pb).sum()),
                   'PC_mean_error_L2':float(np.linalg.norm(a[stage,:,6:].mean(0)-b[stage,:,6:].mean(0))),
                   'donor_summary_energy_distance_V':energy(a[stage],b[stage]),
                   'cell_energy_distance_V':energy(ac[stage],bc[stage])})
   per_rep.append({'replicate':rep,'stages':stages,'pooled_cell_energy_distance_V':energy(ac.reshape(-1,50),bc.reshape(-1,50))})
  means=[]
  for stage in range(4):
   values=[q['stages'][stage] for q in per_rep]
   means.append({'stage':stage,**{key:float(np.mean([v[key] for v in values])) for key in ['composition_L1','PC_mean_error_L2','donor_summary_energy_distance_V','cell_energy_distance_V']}})
  records.append({'method':name,'replicates':per_rep,'stage_replicate_means':means,
                  'pooled_cell_energy_distance_V_mean':float(np.mean([q['pooled_cell_energy_distance_V'] for q in per_rep]))})
 changes=[]
 for stage in range(4):
  a,b=[q['stage_replicate_means'][stage] for q in records]
  changes.append({'stage':stage,**{key+'_semantic_minus_PG8':b[key]-a[key] for key in ['composition_L1','PC_mean_error_L2','donor_summary_energy_distance_V','cell_energy_distance_V']}})
 result={'status':'GENERATED_FITTING_QUERY_SEMANTIC_COMPARISON','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'records':records,'semantic_minus_PG8':changes,'source_arrays_SHA256':pins,
         'scope':'Generated summaries/cells versus matched optimization-donor resampling, fixed fitting-derived condition/library query law. No calibration or evaluation target arrays loaded. Existing adaptive calibration selection is not independent.',
         'interpretation_limits':'These V-statistic distances measure fit to resampling, not held-out fidelity or the selection objective.375 cells/stage per replicate and400 generated bags/stage total; no new generation or fitting. A changed fitting-query mismatch alone cannot establish a causal training mechanism or select a winner.',
         'new_model_updates':0,'GPU_used':False}
 (ROOT/'analysis/semantic_query_fit_diagnostic.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 print(json.dumps({'changes':changes,'pooled_fit_cell_ED':{q['method']:q['pooled_cell_energy_distance_V_mean'] for q in records}},indent=2))

if __name__=='__main__':main()
