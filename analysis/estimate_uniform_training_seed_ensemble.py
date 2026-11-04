"""One preregistered three-model saved-forecast screen, after frozen confirmation.

No target arrays, model fitting, native generation, selection or GPU admission.
The estimate averages over fixed-count outer masks; a native result would need
fresh common queries, exact fixed masks, count/spread audits and new confirmation.
"""
import argparse,datetime,json,os,sys
from pathlib import Path
import numpy as np
ROOT=Path('/ml2/alz');sys.path.insert(0,str(ROOT/'analysis'))
from diagnose_pure_ensemble import donor_complementarity,cell_complementarity,sha

def expected_multimask_score(scores,forecasts,counts):
    scores=np.asarray(scores,dtype=float);counts=np.asarray(counts)
    m=len(forecasts);n=len(forecasts[0])
    if scores.shape!=(m,) or counts.shape!=(m,) or not np.issubdtype(counts.dtype,np.integer) or (counts<0).any() or counts.sum()!=n or n<=1:raise ValueError('Fixed nonnegative counts must sum to forecast length')
    if not np.isfinite(scores).all() or any(x.shape!=forecasts[0].shape or not np.isfinite(x).all() for x in forecasts):raise ValueError('Finite equal-shaped forecasts and scores required')
    value=float(counts@scores/n);between=np.zeros((m,m))
    for i in range(m):
        for j in range(i+1,m):
            between[i,j]=between[j,i]=donor_complementarity(forecasts[i],forecasts[j])
            value-=.5*counts[i]*counts[j]/(n*(n-1))*between[i,j]
    return float(value),between

def empirical_cell_mixture(scores,forecasts,weights):
    scores=np.asarray(scores,dtype=float);weights=np.asarray(weights,dtype=float);m=len(forecasts)
    if scores.shape!=(m,) or weights.shape!=(m,) or not np.isfinite(scores).all() or not np.isfinite(weights).all() or (weights<0).any() or not np.isclose(weights.sum(),1,rtol=0,atol=1e-12):raise ValueError('Finite scores and normalized nonnegative weights required')
    if any(x.ndim!=2 or x.shape[1]!=forecasts[0].shape[1] or not len(x) or not np.isfinite(x).all() for x in forecasts):raise ValueError('Finite empirical point clouds required')
    value=float(weights@scores);between=np.zeros((m,m))
    for i in range(m):
        for j in range(i+1,m):
            between[i,j]=between[j,i]=cell_complementarity(forecasts[i],forecasts[j])
            value-=weights[i]*weights[j]*between[i,j]
    return float(value),between

def fixed_counts(rep,stage):
    if type(rep) is not int or rep not in (0,1) or type(stage) is not int or stage not in range(4):raise ValueError('Fixed replica/stage required')
    counts=np.full(3,67,dtype=int);counts[(4*rep+stage)%3]=66
    return counts

def read(path):return json.loads(Path(path).read_text())

def require_complete_set(path,hypothesis):
    result=read(path)
    if [r.get('seed') for r in result.get('records',[])]!=[0,1,2] or [r.get('attempt') for r in result['records']]!=hypothesis['attempts']:raise ValueError('Require complete original frozen seed0/1/2 set in fixed order')
    from aggregate_pure_confirmation import stable_target
    expected=stable_target(result['records'])
    for key,value in expected.items():
        if result.get(key)!=value:raise ValueError('Frozen aggregate target was not computed from all seeds')
    if not result.get('training_seeds_were_not_ranked') or not result.get('incumbent_unchanged'):raise ValueError('Original confirmation/selection must be preserved')
    if not result.get('evidence_SHA256'):raise ValueError('Audited aggregate evidence required')
    for source,pin in result['evidence_SHA256'].items():
        if sha(source)!=pin:raise ValueError('Aggregate evidence changed: '+source)
    return result

def estimate(aggregate_path,output):
    if output.exists():raise ValueError('Preserve earlier screen evidence')
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    hp=ROOT/'analysis/uniform_training_seed_ensemble_hypothesis.json';h=read(hp)
    assert h['model_training_seeds']==[0,1,2] and h['nominal_weights']==[1/3]*3
    assert h['inner_whole_bag_mixture']=={'PG8':.75,'learned':.25} and h['outer_mask']['bags_per_stage_replica']==200
    aggregate=require_complete_set(aggregate_path,h)
    freeze_path=ROOT/'analysis/pure_ensemble_confirmation_freeze.json';f=read(freeze_path)
    assert sha(freeze_path)==aggregate['original_confirmation_freeze_SHA256']
    incpath=ROOT/'pure_incumbent.json';inc=read(incpath)
    assert sha(incpath)==aggregate['pure_incumbent_SHA256']==f['pure_incumbent_SHA256']
    pins={str(p):sha(p) for p in [hp,aggregate_path,freeze_path,incpath]}
    candidates=[]
    for seed,attempt in enumerate(h['attempts']):
        if seed==0:
            proof_path=ROOT/'analysis'/(attempt+'-result.json');proof=read(proof_path)
            assert proof['independent_saved_scores_checkpoint_and_count_exemplars_verified']
        else:
            proof_path=ROOT/'analysis'/(attempt+'-confirmation.json');proof=read(proof_path)
            assert proof['status']=='EXACT_LINEAGE_ENSEMBLE_SEED_CONFIRMATION_RECORDED_NO_MODEL_SELECTION'
            assert proof['independently_verified_saved_scores_counts_and_checkpoint'] and proof['full_checkpoint_and_native_recipe_verified'] and proof['owned_release_and_clean_settlement_verified']
            assert aggregate['evidence_SHA256'][str(proof_path)]==sha(proof_path)
        candidate=proof['candidate'];assert candidate['training_seed']==seed and candidate['completed_updates']==40000
        assert candidate['pure_only'] and candidate['real_bag_replacements']==0 and candidate['hybrid_config'] is None
        assert candidate['ensemble_config']==f['seed0_candidate']['ensemble_config']
        cp=Path(candidate['checkpoint_path']);assert sha(cp)==candidate['checkpoint_SHA256']
        pins.update({str(proof_path):sha(proof_path),str(cp):sha(cp)});candidates.append(candidate)
    rows=[];component_metrics=[[],[],[]]
    for rep in (0,1):
        forecasts=[];metrics=[]
        for seed,attempt in enumerate(h['attempts']):
            path=ROOT/'runs'/attempt/f'r{rep}/arrays.npz';mp=path.with_name('metrics.json');m=read(mp)
            assert m['training_seed']==seed and m['generation_replicate']==rep and m['status']=='completed' and m['arrays_SHA256']==sha(path)
            with np.load(path,allow_pickle=False) as arrays:y=arrays['stage_synthetic_summaries'].copy();cells=arrays['generated_cells_by_stage'].copy()
            assert y.shape==(4,200,26) and cells.shape==(4,375,50) and np.isfinite(y).all() and np.isfinite(cells).all()
            for key in ('donors','calibration_manifest_SHA256','trace_manifest_SHA256','sampling_contract'):assert m[key]==candidates[seed][key]
            metrics.append(m);forecasts.append((y,cells));component_metrics[seed].append(m);pins.update({str(path):sha(path),str(mp):sha(mp)})
        for key in ('donors','stage','calibration_manifest_SHA256','trace_manifest_SHA256','sampling_contract'):assert all(m[key]==metrics[0][key] for m in metrics)
        stages=[]
        for stage in range(4):
            donor=metrics[0]['stage'].index(stage);scores=[m['per_donor_energy_score'][donor] for m in metrics]
            counts=fixed_counts(rep,stage);es,between=expected_multimask_score(scores,[x[0][stage] for x in forecasts],counts)
            stages.append({'stage':stage,'outer_model_counts':counts.tolist(),'expected_energy_score_over_uniform_fixed_count_masks':es,'donor_complementarity_U':between.tolist()})
        cell_ed,between=empirical_cell_mixture([m['cell_energy_distance'] for m in metrics],[x[1].reshape(-1,50) for x in forecasts],[1/3]*3)
        rows.append({'replicate':rep,'stages':stages,'expected_energy_score':float(np.mean([s['expected_energy_score_over_uniform_fixed_count_masks'] for s in stages])),
                     'ideal_uniform_weighted_empirical_cell_ED':cell_ed,'cell_complementarity_V':between.tolist()})
    for seed,candidate in enumerate(candidates):
        for key in ('energy_score','cell_energy_distance','per_donor_energy_score','per_stage_cell_energy_distance'):
            np.testing.assert_allclose(np.mean([m[key] for m in component_metrics[seed]],axis=0),candidate[key],rtol=0,atol=1e-12)
    es=float(np.mean([r['expected_energy_score'] for r in rows]));ed=float(np.mean([r['ideal_uniform_weighted_empirical_cell_ED'] for r in rows]))
    passed=es<inc['energy_score']-1e-9 and ed<=inc['cell_energy_distance']+1e-9
    result={'status':'ANALYTIC_UNIFORM_TRAINING_SEED_ENSEMBLE_SCREEN_NOT_NATIVE_RESULT','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'model_training_seeds':[0,1,2],'nominal_weights':[1/3]*3,'replicates':rows,'mean_expected_energy_score':es,'mean_ideal_uniform_empirical_cell_ED':ed,
            'selected_pure_incumbent_energy_score':inc['energy_score'],'selected_pure_incumbent_cell_ED':inc['cell_energy_distance'],
            'expected_ES_change':es-inc['energy_score'],'empirical_cell_ED_change':ed-inc['cell_energy_distance'],'joint_analytic_screen_passed':bool(passed),
            'native_result_verified':False,'model_selected':False,'incumbents_changed':False,'original_frozen_confirmation_changed':False,
            'scope':'One preregistered uniform inclusion of ALL three original training-seed models. Saved forecasts only, no target arrays or evaluation donors loaded. Original three-seed outcomes retained; this is new adaptive candidate screening, not independent confirmation.',
            'limits':'Donor expectation uses hypothetical uniformly assigned outer masks applied to existing sample indices. Source model generation traces differ by training seed; a native ensemble would require common fresh traces and exact fixed masks. Cell estimate is ideal1/3 mixture of saved empirical reservoirs, not the rounded66/67 counts or native reservoir. Spread and integer-count validity unverified. All new candidate confirmation must use fresh independently trained groups.',
            'next_step':('Qualify one separately bound pure native six-parent generation experiment; no automatic successor.' if passed else 'Reject this fixed uniform ensemble screen and retain incumbent. Do not activate a native candidate or scan weights.'),
            'input_SHA256':pins,'source_SHA256':{str(Path(__file__).resolve()):sha(__file__),str(ROOT/'analysis/diagnose_pure_ensemble.py'):sha(ROOT/'analysis/diagnose_pure_ensemble.py')}}
    assert sha(incpath)==aggregate['pure_incumbent_SHA256'] and 'torch' not in sys.modules
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('replicates','input_SHA256','source_SHA256')},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--confirmation-set',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args();estimate(args.confirmation_set,args.output)
