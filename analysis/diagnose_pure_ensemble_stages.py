"""Saved-array calibration stage diagnostics; no new training, generation or selection."""
import datetime,json,sys
from pathlib import Path
import numpy as np
ROOT=Path('/ml2/alz');sys.path[:0]=[str(ROOT),str(ROOT/'analysis')]
from calibration import score,average,sha
from diagnose_reference import profile

def main():
 freeze=json.loads((ROOT/'analysis/pure_ensemble_confirmation_freeze.json').read_text());candidate=freeze['seed0_candidate']
 assert sha(ROOT/'pure_incumbent.json')==freeze['pure_incumbent_SHA256']
 decision=json.loads((ROOT/'analysis'/(freeze['seed0_attempt']+'-result.json')).read_text());assert decision['decision']['accept'] and decision['candidate']==candidate
 target=np.load(ROOT/'data/calibration_fold0/real_summaries.npz',allow_pickle=False);real=np.load(ROOT/'data/calibration_fold0/cell_reservoir.npz',allow_pickle=False)['by_stage']
 baseline=json.loads((ROOT/'analysis/baseline_calibration_seed0.json').read_text());assert baseline['both_original_seed0_arrays_bitwise_reproduced']
 paths={'Resampling':ROOT/'artifacts/calibration_resampling_seed0_validation_replay','PG8':ROOT/'runs/CAL_GPU0_V1_latent8_v1_f0_pg_s0','Learned':ROOT/'runs/CAL_GPU0_V1_learned_obs_v1_f0_pg_s0','Pure ensemble':ROOT/'runs'/freeze['seed0_attempt']}
 records=[];pins={str(path):sha(path) for path in [ROOT/'analysis/pure_ensemble_confirmation_freeze.json',ROOT/'analysis'/(freeze['seed0_attempt']+'-result.json'),ROOT/'analysis/baseline_calibration_seed0.json',ROOT/'data/calibration_fold0/manifest.json',ROOT/'data/calibration_fold0/real_summaries.npz',ROOT/'data/calibration_fold0/cell_reservoir.npz']};means={}
 for label,folder in paths.items():
  metrics=[]
  for rep in (0,1):
   data=folder/f'r{rep}/arrays.npz';mp=folder/f'r{rep}/metrics.json';m=json.loads(mp.read_text());assert sha(data)==m['arrays_SHA256'];pins.update({str(data):sha(data),str(mp):sha(mp)})
   if label=='Resampling':
    original=ROOT/f'artifacts/calibration_resampling_seed0/r{rep}/arrays.npz';pins[str(original)]=sha(original)
    with np.load(original,allow_pickle=False) as old,np.load(data,allow_pickle=False) as replay:
     assert set(old.files)==set(replay.files)
     for key in old.files:assert old[key].dtype==replay[key].dtype and old[key].tobytes()==replay[key].tobytes()
   with np.load(data,allow_pickle=False) as a:
    y,c=a['stage_synthetic_summaries'],a['generated_cells_by_stage'];recomputed=score(y,c)
    for key in ['energy_score','cell_energy_distance','per_donor_energy_score','per_stage_cell_energy_distance','spread_ratio']:
     np.testing.assert_allclose(m[key],recomputed[key],rtol=0,atol=1e-10)
    assert m['training_seed']==0 and m['generation_replicate']==rep and m['donors']==target['donors'].tolist()
    for stage in range(4):
     index=int(np.flatnonzero(target['stage']==stage)[0]);prop=np.square(y[stage,:,:6]).mean(0);truth=np.square(target['summary'][index,:6])
     records.append({'method':label,'replicate':rep,'stage':stage,'donor':str(target['donors'][index]),
        'donor_energy_score':recomputed['per_donor_energy_score'][index],'cell_energy_distance':recomputed['per_stage_cell_energy_distance'][stage],
        'composition_L1':float(np.abs(prop-truth).sum()),**profile(c[stage],real[stage])})
   metrics.append(m)
  means[label]=average(metrics)
 for label,reference in [('Resampling',baseline),('Pure ensemble',candidate)]:
  for key in ['energy_score','cell_energy_distance','per_donor_energy_score','per_stage_cell_energy_distance','spread_ratio']:np.testing.assert_allclose(means[label][key],reference[key],rtol=0,atol=1e-10)
 keys=['donor_energy_score','cell_energy_distance','composition_L1','PC_mean_error_L2','covariance_relative_Frobenius_error','cell_covariance_trace_ratio','cell_covariance_participation_rank']
 stages=[]
 for stage in range(4):
  methods={}
  for method in paths:
   rows=[q for q in records if q['stage']==stage and q['method']==method]
   methods[method]={key:float(np.mean([q[key] for q in rows])) for key in keys}
  b,e=methods['Resampling'],methods['Pure ensemble']
  stages.append({'stage':stage,'donor':next(q['donor'] for q in records if q['stage']==stage),'methods':methods,
      'ensemble_donor_ES_change':e['donor_energy_score']-b['donor_energy_score'],'ensemble_donor_ES_relative_improvement':1-e['donor_energy_score']/b['donor_energy_score'],
      'ensemble_stage_cell_ED_change':e['cell_energy_distance']-b['cell_energy_distance'],'both_stage_scores_improve':e['donor_energy_score']<b['donor_energy_score'] and e['cell_energy_distance']<=b['cell_energy_distance']})
 assert [q['stage'] for q in stages if not q['both_stage_scores_improve']]==[2]
 result={'status':'VERIFIED_SAVED_ARRAY_PURE_ENSEMBLE_STAGE_DIAGNOSTICS','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'training_seed':0,'generation_replicates':2,
    'donor_order_stage_codes':target['stage'].tolist(),'stages':stages,'replicate_records':records,'pooled_metrics':means,
    'source_array_metrics_SHA256':pins,'source_SHA256':{str(Path(__file__).resolve()):sha(Path(__file__).resolve()),str(ROOT/'analysis/diagnose_reference.py'):sha(ROOT/'analysis/diagnose_reference.py'),str(ROOT/'calibration.py'):sha(ROOT/'calibration.py')},
    'scope':'Secondary descriptive calibration breakdown of the already accepted fixed ensemble. Same4 adaptive donors, one per stage; replicate differences are sampling variability, not independent donors. No new model generation/fitting or candidate selection; frozen3seed target and protocol unchanged.',
    'limits':'Pooled equal-stage cell energy distance is not the mean of conditional stage energy distances. No significance or uniformly better stage-wise claim. Each cell reservoir has375 rows; stage3 real donor has208 cells and was sampled with replacement.'}
 (ROOT/'analysis/pure_ensemble_stage_diagnostics.json').write_text(json.dumps(result,indent=2)+'\n')
 lines=['Pure ensemble seed0 calibration stage breakdown','','Stage | Donor | ES resampling | ES ensemble | ES improvement | Cell ED resampling | Cell ED ensemble','---|---|---:|---:|---:|---:|---:']
 for q in stages:
  b,e=q['methods']['Resampling'],q['methods']['Pure ensemble'];lines.append(f"{q['stage']} | {q['donor']} | {b['donor_energy_score']:.8f} | {e['donor_energy_score']:.8f} | {100*q['ensemble_donor_ES_relative_improvement']:.3f}% | {b['cell_energy_distance']:.8f} | {e['cell_energy_distance']:.8f}")
 lines+=['','Stage2 is slightly worse on both conditional scores; the other3 improve. Pooled seed0 joint target passed. The registered pooled3seed confirmation is unchanged. Four adaptive donors, one per stage; no independent biological or uniform stage improvement claim.','',result['limits']]
 (ROOT/'analysis/pure_ensemble_stage_diagnostics.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
if __name__=='__main__':main()
