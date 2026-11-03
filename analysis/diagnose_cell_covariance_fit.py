"""Generated fitting-query covariance diagnostic; no held-out target arrays."""
import datetime,hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path('/ml2/alz')

def load(folder,rep,pins):
 p=folder/f'r{rep}'/'arrays.npz';m=json.loads(p.with_name('metrics.json').read_text());pin=hashlib.sha256(p.read_bytes()).hexdigest();assert pin==m['arrays_SHA256'];pins[str(p)]=pin
 with np.load(p,allow_pickle=False) as z:x=z['generated_cells_by_stage'].astype(np.float64)
 assert x.shape==(4,375,50) and np.isfinite(x).all()
 return x

def metrics(a,b):
 ca,cb=np.cov(a,rowvar=False),np.cov(b,rowvar=False)
 delta=ca-cb;diag=np.diag(delta);off=delta-np.diag(diag)
 eig=np.linalg.eigvalsh(ca)
 return {'PC_mean_error_L2':float(np.linalg.norm(a.mean(0)-b.mean(0))),
         'covariance_relative_Frobenius_error':float(np.linalg.norm(delta)/np.linalg.norm(cb)),
         'covariance_diagonal_squared_error':float(diag@diag),
         'covariance_offdiagonal_squared_error':float(np.square(off).sum()),
         'offdiagonal_fraction_of_covariance_squared_error':float(np.square(off).sum()/np.square(delta).sum()),
         'cell_covariance_trace_ratio':float(np.trace(ca)/np.trace(cb)),
         'cell_covariance_participation_rank':float(eig.sum()**2/np.square(eig).sum())}

def main():
 pins={};baseline=ROOT/'artifacts/calibration_resampling_seed0_validation_replay'
 real=[load(baseline,rep,pins) for rep in (0,1)];rows=[]
 for name,folder in [('PG8',ROOT/'runs/CAL_GPU0_V1_latent8_v1_f0_pg_s0'),('semantic',ROOT/'runs/CAL_GPU0_V1_semantic_v1_f0_pg_s0'),('BCAP',ROOT/'runs/CAL_GPU0_V1_bcap_v1_f0_pg_s0')]:
  all_rep=[]
  for rep in (0,1):
   generated=load(folder,rep,pins)
   all_rep.append({'replicate':rep,'stages':[{'stage':stage,**metrics(generated[stage],real[rep][stage])} for stage in range(4)]})
  means=[]
  for stage in range(4):
   q=[rep['stages'][stage] for rep in all_rep]
   means.append({'stage':stage,**{k:float(np.mean([v[k] for v in q])) for k in q[0] if k!='stage'}})
  rows.append({'method':name,'replicates':all_rep,'stage_replicate_means':means})
 control=[{'stage':stage,**metrics(real[0][stage],real[1][stage])} for stage in range(4)]
 report={'status':'GENERATED_FITTING_QUERY_COVARIANCE_DIAGNOSTIC','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'records':rows,
         'resampling_rep0_vs_rep1_descriptive_control':control,'source_arrays_SHA256':pins,
         'scope':'Only generated 375-cell/stage fitting-derived query reservoirs. Baseline resamples optimization donors. No calibration or evaluation target arrays loaded; no new sampling, model fitting or GPU allocation.',
         'limits':'Descriptive finite-reservoir covariance, not an independent donor test or causal ablation. Offdiagonal error sums2450 entries versus50 diagonal entries; its fraction is not evidence by itself. Different replicate query draws also change donor composition, so the resampling contrast is not a pure measurement-noise estimate.'}
 (ROOT/'analysis/cell_covariance_fit_diagnostic.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 print(json.dumps({'models':[{q['method']:q['stage_replicate_means']} for q in rows],'resampling_rep0_vs_rep1':control},indent=2))

if __name__=='__main__':main()
