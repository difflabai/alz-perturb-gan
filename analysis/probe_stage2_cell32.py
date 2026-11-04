"""One preregistered empirical whole-stage substitution; not a native result."""
import datetime,json,os,sys
from pathlib import Path
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
ROOT=Path('/ml2/alz');sys.path.insert(0,str(ROOT))
import numpy as np
from calibration import score,sha

def main():
 hp=ROOT/'analysis/stage2_cell32_hypothesis.json';h=json.loads(hp.read_text());assert h['status']=='POST_HOC_CALIBRATION_HYPOTHESIS_BEFORE_SINGLE_SAVED_DISTRIBUTION_STITCH_SCORE'
 assert h['parent_rule_by_stage']=={str(s):'retained_pure_cell32' if s==2 else 'selected_pg8_learned_ensemble75' for s in range(4)}
 for key in ['selected_reference','cell32_reference','diagnostic_reference']:assert sha(h[key]['path'])==h[key]['SHA256']
 inc=json.loads(Path(h['selected_reference']['path']).read_text());cp=json.loads(Path(h['cell32_reference']['path']).read_text());assert cp['independent_saved_array_score_and_integer_count_verification']
 dirs=[Path(inc['checkpoint_path']).parent.parent,ROOT/'runs'/cp['attempt']]
 protected=[ROOT/'pure_incumbent.json',ROOT/'incumbent.json',ROOT/'analysis/pure_ensemble_confirmation_freeze.json',ROOT/'analysis/exact_seed1_recovery_protocol.json']
 before={str(p):sha(p) for p in protected};pins={str(hp):sha(hp),**before};records=[]
 for rep in (0,1):
  arrays=[];metrics=[]
  for folder in dirs:
   path=folder/f'r{rep}/arrays.npz';mp=path.with_name('metrics.json');m=json.loads(mp.read_text());assert m['arrays_SHA256']==sha(path) and m['status']=='completed' and m['generation_replicate']==rep and m['training_seed']==0
   pins.update({str(path):sha(path),str(mp):sha(mp)})
   with np.load(path,allow_pickle=False) as z:y=z['stage_synthetic_summaries'].copy();c=z['generated_cells_by_stage'].copy()
   recomputed=score(y,c)
   for key in ['energy_score','cell_energy_distance','spread_ratio','per_donor_energy_score','per_stage_cell_energy_distance']:np.testing.assert_allclose(recomputed[key],m[key],rtol=0,atol=1e-10)
   arrays.append((y,c));metrics.append(m)
  for key in ['donors','stage','training_seed','generation_replicate','calibration_manifest_SHA256','trace_manifest_SHA256','sampling_contract']:assert metrics[0][key]==metrics[1][key]
  y,c=[a.copy() for a in arrays[0]];y[2]=arrays[1][0][2];c[2]=arrays[1][1][2]
  for s in (0,1,3):assert y[s].tobytes()==arrays[0][0][s].tobytes() and c[s].tobytes()==arrays[0][1][s].tobytes()
  assert y[2].tobytes()==arrays[1][0][2].tobytes() and c[2].tobytes()==arrays[1][1][2].tobytes()
  q=score(y,c);q['generation_replicate']=rep;q['computed_from_saved_distribution_only']=True;records.append(q)
 keys=['energy_score','cell_energy_distance','spread_ratio','within_condition_spread_ratio','coverage80','per_donor_energy_score','per_stage_cell_energy_distance']
 means={k:np.mean([q[k] for q in records],axis=0).tolist() for k in keys}
 expected_ES=inc['energy_score']+(cp['per_donor_energy_score'][3]-inc['per_donor_energy_score'][3])/4 if 'per_donor_energy_score' in cp else None
 # Independent donor additivity using the separately recomputed all-method diagnostics.
 d=json.loads(Path(h['diagnostic_reference']['path']).read_text());row=next(q for q in d['rows'] if q['formulation']=='cell32')
 delta=row['stages'][2]['donor_energy_score']-inc['per_donor_energy_score'][inc['donors'].index(row['stages'][2]['donor'])]
 np.testing.assert_allclose(means['energy_score'],inc['energy_score']+delta/4,rtol=0,atol=1e-12)
 baseline=json.loads((ROOT/'analysis/baseline_calibration_seed0.json').read_text())
 assert before=={str(p):sha(p) for p in protected} and 'torch' not in sys.modules
 result={'status':'SAVED_EMPIRICAL_STAGE_SUBSTITUTION_DIAGNOSTIC_NOT_NATIVE_MODEL_RESULT','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'declared_rule':h['parent_rule_by_stage'],'records':records,'mean_saved_empirical_scores':means,
 'change_from_selected_ensemble':{k:means[k]-inc[k] for k in ['energy_score','cell_energy_distance','spread_ratio']},'comparison_to_matched_resampling':{k:means[k]-baseline[k] for k in ['energy_score','cell_energy_distance']},'donor_energy_additivity_verified':True,
 'scope':'Exactly one preregistered saved-array stage2 replacement. Exact recomputed score of the constructed saved empirical forecasts, not a native-regenerated conditional generator. This is post hoc on4adaptivecalibrationdonors, not independent evidence. Conditional native count/stream/exposure/replay verification and separately reviewed operation not performed; no model acceptance or ATH claim.',
 'parent_training_updates_total':60000,'new_training_updates':0,'new_model_generation':False,'GPU_allocation_created':False,'native_result_verified':False,'incumbents_changed':False,'frozen_confirmation_changed':False,
 'source_array_metrics_hypothesis_SHA256':pins,'immutable_before_and_after_SHA256':before,'source_SHA256':{str(Path(__file__).resolve()):sha(Path(__file__).resolve()),str(ROOT/'calibration.py'):sha(ROOT/'calibration.py')}}
 path=ROOT/'analysis/stage2_cell32_saved_distribution_probe.json';assert not path.exists();path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 print(json.dumps({k:result[k] for k in ['status','mean_saved_empirical_scores','change_from_selected_ensemble','native_result_verified','incumbents_changed']},indent=2))
if __name__=='__main__':main()
