"""Descriptive conditional diagnostics of completed saved forecasts; no selection.

Recompute original scores against calibration only. Keep the frozen confirmation
and incumbents unchanged. Conditional frontiers are post hoc on four adaptive
calibration donors, not independent confirmation or a stage-specific model.
"""
import datetime,json,os,sys
from pathlib import Path
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
ROOT=Path('/ml2/alz');sys.path.insert(0,str(ROOT))
import numpy as np
from calibration import score,average,sha

def main():
 leader=ROOT/'analysis/pure_formulations_leaderboard.json';freeze=ROOT/'analysis/pure_ensemble_confirmation_freeze.json'
 table=json.loads(leader.read_text());f=json.loads(freeze.read_text())
 immutable=[ROOT/'pure_incumbent.json',ROOT/'incumbent.json',freeze,ROOT/'analysis/exact_seed1_recovery_protocol.json']
 before={str(p):sha(p) for p in immutable};assert before[str(ROOT/'pure_incumbent.json')]==f['pure_incumbent_SHA256']
 data=ROOT/'data/calibration_fold0';manifest=json.loads((data/'manifest.json').read_text())
 for name,pin in manifest['files_SHA256'].items():assert sha(data/name)==pin
 with np.load(data/'real_summaries.npz',allow_pickle=False) as z:donors=z['donors'].tolist();stages=z['stage'].tolist()
 assert sorted(stages)==list(range(4)) and len(set(donors))==4
 pins={str(p):sha(p) for p in [leader,*immutable,data/'manifest.json',data/'real_summaries.npz',data/'cell_reservoir.npz']}
 entries=[('Resampling',ROOT/'artifacts/calibration_resampling_seed0_validation_replay',json.loads((ROOT/'analysis/baseline_calibration_seed0.json').read_text()))]
 for row in table['rows']:
  folder=ROOT/'runs'/row['attempt'] if row.get('attempt') else Path(row['checkpoint_path']).parent.parent
  entries.append((row['formulation'],folder,row))
 rows=[]
 for label,folder,reference in entries:
  reps=[]
  for rep in (0,1):
   path=folder/f'r{rep}/arrays.npz';mp=path.with_name('metrics.json');m=json.loads(mp.read_text())
   assert m['status']=='completed' and m['training_seed']==0 and m['generation_replicate']==rep and m['donors']==donors and m['stage']==stages
   assert sha(path)==m['arrays_SHA256'];pins.update({str(path):sha(path),str(mp):sha(mp)})
   with np.load(path,allow_pickle=False) as z:computed=score(z['stage_synthetic_summaries'],z['generated_cells_by_stage'])
   for key in ['energy_score','cell_energy_distance','spread_ratio','per_donor_energy_score','per_stage_cell_energy_distance']:np.testing.assert_allclose(computed[key],m[key],rtol=0,atol=1e-10)
   reps.append({**m,**computed})
  mean=average(reps)
  for key in ['energy_score','cell_energy_distance','spread_ratio']:np.testing.assert_allclose(mean[key],reference[key],rtol=0,atol=1e-10)
  rows.append({'formulation':label,'pooled_energy_score':mean['energy_score'],'pooled_cell_energy_distance':mean['cell_energy_distance'],'spread_ratio':mean['spread_ratio'],
   'stages':[{'stage':s,'donor':donors[stages.index(s)],'donor_energy_score':mean['per_donor_energy_score'][stages.index(s)],'cell_energy_distance':mean['per_stage_cell_energy_distance'][s]} for s in range(4)]})
 base=rows[0];pure=rows[1:];conditional=[]
 for stage in range(4):
  bs=base['stages'][stage];joint=[];frontier=[]
  for row in pure:
   x=row['stages'][stage]
   if x['donor_energy_score']<bs['donor_energy_score'] and x['cell_energy_distance']<=bs['cell_energy_distance']:joint.append(row['formulation'])
   dominated=any(other['stages'][stage]['donor_energy_score']<=x['donor_energy_score'] and other['stages'][stage]['cell_energy_distance']<=x['cell_energy_distance'] and (other['stages'][stage]['donor_energy_score']<x['donor_energy_score'] or other['stages'][stage]['cell_energy_distance']<x['cell_energy_distance']) for other in pure)
   if not dominated:frontier.append(row['formulation'])
  conditional.append({'stage':stage,'donor':bs['donor'],'pure_formulations_better_on_both_than_stage_resampling':joint,'descriptive_pure_ES_cellED_Pareto_frontier':frontier})
 assert before=={str(p):sha(p) for p in immutable};assert 'torch' not in sys.modules
 result={'status':'VERIFIED_ALL_COMPLETED_PURE_SEED0_CONDITIONAL_DIAGNOSTICS_NO_SELECTION','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pure_formulations':len(pure),'generation_replicates':2,'donor_order_stage_codes':stages,'rows':rows,'conditional_diagnostics':conditional,
 'scope':'Post hoc descriptive breakdown of already completed pure seed0 forecasts. Same four adaptive calibration donors, one per stage, not independent biological replicates. Recomputes all original saved scores, preserves original pooled ranking/selection and frozen seeds0/1/2. No new candidate, mask/weight search, model generation, training, native admission, or incumbent change. No evaluation donors read.',
 'limits':'Conditional stage ED is not pooled cell ED. An apparent per-stage frontier is not evidence that switching parent by stage would improve the pooled metrics or generalize; that would need a separately predeclared native generation comparison and new confirmation, retaining the current experiment.',
 'immutable_before_and_after_SHA256':before,'source_array_metrics_SHA256':pins,'source_SHA256':{str(Path(__file__).resolve()):sha(Path(__file__).resolve()),str(ROOT/'calibration.py'):sha(ROOT/'calibration.py')}}
 out=ROOT/'analysis/all_pure_stage_diagnostics.json';assert not out.exists();out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 lines=['Completed pure seed0 conditional diagnostics','','All13 pure formulations, same4 adaptive calibration donors, two generation replicates. Descriptive only; no model selection or frozen confirmation changes.','']
 for q in conditional:lines.append(f"Stage{q['stage']}: joint-better-than-resampling = {', '.join(q['pure_formulations_better_on_both_than_stage_resampling']) or 'NONE'}; observed pure frontier = {', '.join(q['descriptive_pure_ES_cellED_Pareto_frontier'])}")
 lines.extend(['',result['limits']]);(ROOT/'analysis/all_pure_stage_diagnostics.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
if __name__=='__main__':main()
