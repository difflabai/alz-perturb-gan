"""One prospective saved-forecast complementarity estimate; no native model selection."""
import datetime,json,os,sys
from pathlib import Path
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
ROOT=Path('/ml2/alz');sys.path.insert(0,str(ROOT/'analysis'))
import numpy as np
from diagnose_pure_ensemble import donor_complementarity,expected_fixed_mask_score,cell_complementarity,sha

def main():
 hp=ROOT/'analysis/cell_energy_learned_fixed_ensemble_hypothesis.json';h=json.loads(hp.read_text());assert [m['weight'] for m in h['models']]==[.75,.25] and h['fixed_learned_bags_per200']==50 and h['replicates']==[0,1]
 incpath=ROOT/'pure_incumbent.json';freeze=ROOT/'analysis/pure_ensemble_confirmation_freeze.json';assert sha(incpath)==h['current_pure_incumbent_SHA256'] and sha(freeze)==h['frozen_confirmation_SHA256'];inc=json.loads(incpath.read_text());pins={str(p):sha(p) for p in [hp,incpath,freeze]};refs=[]
 for model in h['models']:
  proof_path=Path(model['result_proof']);assert sha(proof_path)==model['result_proof_SHA256'];q=json.loads(proof_path.read_text());ref=q['candidate'];assert q['independent_saved_scores_checkpoint_and_count_exemplars_verified']
  assert ref['status']=='completed' and ref['training_seed']==0 and ref['completed_updates']==20000 and ref['real_bag_replacements']==0 and ref['hybrid_config'] is None
  cp=Path(model['checkpoint_path']);assert sha(cp)==ref['checkpoint_SHA256']==model['checkpoint_SHA256'];pins.update({str(proof_path):sha(proof_path),str(cp):sha(cp)});refs.append(ref)
 records=[];component_metrics=[[],[]]
 for rep in (0,1):
  forecasts=[];metrics=[]
  for index,model in enumerate(h['models']):
   folder=ROOT/'runs'/model['attempt']/f'r{rep}';path=folder/'arrays.npz';mp=folder/'metrics.json';m=json.loads(mp.read_text());assert m['status']=='completed' and m['training_seed']==0 and m['generation_replicate']==rep and sha(path)==m['arrays_SHA256'];pins.update({str(path):sha(path),str(mp):sha(mp)})
   with np.load(path,allow_pickle=False) as z:y=z['stage_synthetic_summaries'].copy();cells=z['generated_cells_by_stage'].copy()
   assert y.shape==(4,200,26) and cells.shape==(4,375,50) and np.isfinite(y).all() and np.isfinite(cells).all();forecasts.append((y,cells));metrics.append(m);component_metrics[index].append(m)
  for key in ('donors','stage','training_seed','generation_replicate','sampling_contract','calibration_manifest_SHA256','trace_manifest_SHA256'):assert metrics[0][key]==metrics[1][key]
  between=[donor_complementarity(forecasts[0][0][s],forecasts[1][0][s]) for s in range(4)];es=expected_fixed_mask_score(metrics[0]['energy_score'],metrics[1]['energy_score'],float(np.mean(between)),200,50)
  cross=cell_complementarity(forecasts[0][1].reshape(-1,50),forecasts[1][1].reshape(-1,50));ed=.75*metrics[0]['cell_energy_distance']+.25*metrics[1]['cell_energy_distance']-.75*.25*cross
  records.append({'replicate':rep,'donor_complementarity_U_by_stage':between,'expected_energy_score_over_uniform_fixed50of200_masks':es,'weighted_empirical_cell_reservoir_energy_distance':ed,'cell_complementarity_V':cross})
 for i,ref in enumerate(refs):
  for key in ('energy_score','cell_energy_distance'):np.testing.assert_allclose(np.mean([m[key] for m in component_metrics[i]]),ref[key],rtol=0,atol=1e-12)
 es=float(np.mean([q['expected_energy_score_over_uniform_fixed50of200_masks'] for q in records]));ed=float(np.mean([q['weighted_empirical_cell_reservoir_energy_distance'] for q in records]));joint=es<inc['energy_score']-1e-9 and ed<=inc['cell_energy_distance']+1e-9
 assert sha(incpath)==h['current_pure_incumbent_SHA256'] and sha(freeze)==h['frozen_confirmation_SHA256'] and 'torch' not in sys.modules
 result={'status':'ANALYTIC_FIXED_TASK_EXTENDED_LEARNED_ESTIMATE_NOT_NATIVE_MODEL_RESULT','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'registered_weights':[.75,.25],'replicates':records,'mean_expected_energy_score':es,'mean_weighted_empirical_cell_ED':ed,'selected_pure_incumbent_energy_score':inc['energy_score'],'selected_pure_incumbent_cell_ED':inc['cell_energy_distance'],'expected_ES_change':es-inc['energy_score'],'empirical_cell_ED_change':ed-inc['cell_energy_distance'],'joint_analytic_screen_passed':bool(joint),'native_ensemble_result_verified':False,'model_selected':False,'incumbents_changed':False,'current_three_seed_confirmation_changed':False,'scope':'One predeclared fixed-pair analytical check on adaptive seed0 calibration only. Components include an explicit ALZ task-loss extension; not a new published formulation. Totalparenttraining40k updates. No new target/evaluation arrays, fitting, generation, native allocation or modelsource changes. No weight/mask search.','limits':'Donor estimate averages over hypothetical uniformly selected fixed-count masks, not the registered native mask. Cell estimate uses weighted empirical reservoirs, not a newly sampled mixture. Joint native RNG changes, spread and integer-count mixture validity remain unverified. A screen cannot establish a new ATH, independent confirmation or biological generalization.','next_step':'Retain the original failed three-seed confirmation and current incumbent. '+('This pair merits separately qualified native generation after full native source/resource qualification; no automatic successor.' if joint else 'This fixed pair does not meet the joint analytic screen; retain the current incumbent. No native candidate activated.'),'source_checkpoint_array_metric_SHA256':pins,'source_SHA256':{str(Path(__file__).resolve()):sha(Path(__file__).resolve()),str(ROOT/'analysis/diagnose_pure_ensemble.py'):sha(ROOT/'analysis/diagnose_pure_ensemble.py'),str(ROOT/'analysis/test_ensemble_estimate.py'):sha(ROOT/'analysis/test_ensemble_estimate.py')}}
 out=ROOT/'analysis/cell_energy_learned_fixed_ensemble_estimate.json';assert not out.exists();out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({k:result[k] for k in ['status','mean_expected_energy_score','mean_weighted_empirical_cell_ED','expected_ES_change','empirical_cell_ED_change','joint_analytic_screen_passed','native_ensemble_result_verified','next_step']},indent=2))
if __name__=='__main__':main()
