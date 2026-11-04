"""CPU audit of the actual native trial; record selection without auto-promotion."""
import argparse,datetime,json,os,sys
from pathlib import Path
ROOT=Path('/ml2/alz');HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(ROOT),str(ROOT/'analysis')]
import binding as b
import preset_scope as scope

def finish(attempt):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    sys.path.extend(map(str,b.native_site_roots(scope.CELL,b.read(ROOT/'artifacts/allocations'/attempt/'authorization.json')['files_SHA256'])))
    sys.path[:0]=[str(ROOT/'ema_readout_model'),str(ROOT/'model'),str(b.SCIENCE),str(b.PROJECT)]
    import numpy as np
    import torch
    import ka2_ema
    from calibration import score,average
    from selection import decide
    assert Path(ka2_ema.__file__).resolve()==ROOT/'ema_readout_model/ka2_ema.py'
    dep=b.read(ROOT/'artifacts/allocations'/attempt/'deployment.json');auth=b.read(dep['authorization_path']);scope.validate_authorization(auth)
    assert b.sha(dep['authorization_path'])==dep['authorization_SHA256'] and attempt==b.attempt_key(scope.CELL)
    ledger=Path(dep['ledger_path']);job=b.read(ledger)['jobs'][0];settlement=b.read(ledger.with_name(attempt+'-settlement.json'))
    assert job['status']=='completed' and job['exit_code']==0 and job['reservation_released']
    assert settlement['uncertain'] is None and settlement['settlement']['status']=='settled'
    out=Path(dep['output_dir']);candidate=b.read(out/'completed.json');cp=Path(candidate['checkpoint_path']);ref=b.read(cp)
    assert candidate['operation']=='evaluate_ka2_ema' and candidate['native_recipe']=='ka2' and candidate['pure_formulation']=='ka2_saved_native_ema_readout'
    assert candidate['method']=='pg_ka2_native_ema' and candidate['generation']=='direct_iter_pg' and candidate['pure_only'] and candidate['real_bag_replacements']==0
    assert candidate['completed_updates']==candidate['fixed_endpoint']==candidate['initial_completed_updates']==candidate['parent_training_updates_total']==20000 and candidate['new_model_updates']==0
    assert candidate['variant_config']==scope.CFG and candidate['hybrid_config'] is None and candidate['qualification_only'] is False
    assert candidate['readout_config']==auth['readout_config']==scope.READOUT and candidate['parent_checkpoint_SHA256']==auth['parent_checkpoint_SHA256']
    assert candidate['preset_protocol_SHA256']==b.sha(scope.PROTOCOL) and candidate['checkpoint_SHA256']==b.sha(cp)
    assert candidate['checkpoint_kind']==ref['kind']=='explicit_native_KA2_EMA_readout_reference'
    for key in ('parent_checkpoint_path','parent_checkpoint_SHA256','parent_result_path','parent_result_SHA256'):assert ref[key]==auth[key]
    assert b.sha(auth['parent_checkpoint_path'])==auth['parent_checkpoint_SHA256'] and b.sha(auth['parent_result_path'])==auth['parent_result_SHA256']
    q=torch.load(auth['parent_checkpoint_path'],map_location='cpu',weights_only=False)
    G,P,info=ka2_ema.load(torch.ones(2000),torch.ones(2000),'cpu',q,cache_sha=b.sha(Path(auth['cache_dir'])/'manifest.json'),fold=0,seed=0)
    assert ref['fold']==ref['seed']==0 and ref['cache_manifest_SHA256']==q['cache_manifest_SHA256']
    for key,value in info.items():assert ref[key]==value
    assert info['training_updates_executed']==0 and info['native_default_served_source']=='fast' and info['original_parent_contract_preserved']
    assert ka2_ema.fingerprint(G.state_dict())==info['native_saved_EMA_state_fingerprints']['generator']
    assert ka2_ema.fingerprint(P.state_dict())==info['native_saved_EMA_state_fingerprints']['prior']
    finite_values=info['EMA_model_floating_values'];assert finite_values==316832
    rows=[];array_pins={}
    for rep in (0,1):
        folder=out/f'r{rep}';m=b.read(folder/'metrics.json');assert m['generated_cells']==320000
        assert m['arrays_SHA256']==b.sha(folder/'arrays.npz')
        with np.load(folder/'arrays.npz',allow_pickle=False) as a:recomputed=score(a['stage_synthetic_summaries'],a['generated_cells_by_stage'])
        assert m['donors']==recomputed['donors'] and m['stage']==recomputed['stage']
        for key,value in recomputed.items():
            if isinstance(value,(int,float,list)) and key not in ('donors','stage'):np.testing.assert_allclose(m[key],value,rtol=0,atol=1e-10)
        for stage in range(4):
            path=folder/f'stage{stage}_exemplar.npz'
            with np.load(path,allow_pickle=False) as a:
                x=a['counts'];assert x.shape==(1,400,2000) and np.isfinite(x).all() and (x>=0).all() and np.equal(x,np.floor(x)).all()
            array_pins[str(path)]=b.sha(path)
        rows.append(m);array_pins[str(folder/'arrays.npz')]=b.sha(folder/'arrays.npz')
    avg=average(rows)
    for key in ('donors','training_seed','calibration_manifest_SHA256','trace_manifest_SHA256','sampling_contract'):assert avg[key]==candidate[key]
    for key,value in avg.items():
        if isinstance(value,(int,float,list)) and key not in ('donors',):np.testing.assert_allclose(candidate[key],value,rtol=0,atol=1e-10)
    assert not torch.cuda.is_initialized()
    incumbent=b.read(ROOT/'pure_incumbent.json');assert b.sha(ROOT/'pure_incumbent.json')==auth['frozen_pure_incumbent_SHA256']
    decision=decide(candidate,incumbent)
    result={'status':'VERIFIED_NATIVE_KA2_SAVED_EMA_READOUT_RESULT','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'attempt':attempt,
            'candidate':candidate,'incumbent':incumbent,'decision':decision,'candidate_commit':auth['candidate_commit'],
            'independent_saved_scores_checkpoint_and_count_exemplars_verified':True,'full_native_recipe_model_and_seed_streams_verified':True,'finite_model_floating_values':finite_values,'audited_complete_parent_lineage_verified':True,'exact_coupled_native_EMA_readout_weights_verified':True,'new_training_updates':0,
            'owned_release_and_clean_settlement_verified':True,'charged_GPU_seconds':settlement['settlement']['charged_GPU_seconds'],
            'array_and_exemplar_SHA256':array_pins,'authorization_SHA256':b.sha(dep['authorization_path']),'preset_protocol_SHA256':b.sha(scope.PROTOCOL),
            'source_SHA256':{str(Path(__file__).resolve()):b.sha(__file__)},'incumbent_changed':False,
            'next_step':('Freeze this accepted seed0 candidate for further fixed-seed confirmation before a stable result claim.' if decision['accept'] else 'Commit negative evidence and revert trial activation; keep current incumbent. Try the next scientifically defined candidate.'),
            'scope':'Four adaptive calibration donors only; explicit savednativeKA2 G+priorEMA readout, unchanged parent recipe/defaultfastcontract and no new training.20k completedparent vs40k selectedensemble. No evaluation donor access or independent biological confirmation.'}
    dest=ROOT/'analysis'/(attempt+'-result.json');assert not dest.exists();dest.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('candidate','incumbent','array_and_exemplar_SHA256')},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);finish(p.parse_args().attempt)
