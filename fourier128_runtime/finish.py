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
    sys.path[:0]=[str(ROOT/'model'),str(b.SCIENCE),str(b.PROJECT)]
    import numpy as np
    import torch
    from particlegan import get_recipe
    from pg_model import CountGenerator
    from fourier128_observation import VectorCritic
    from fourier128_adapter import training_streams
    from calibration import score,average
    from selection import decide
    dep=b.read(ROOT/'artifacts/allocations'/attempt/'deployment.json');auth=b.read(dep['authorization_path']);scope.validate_authorization(auth)
    assert b.sha(dep['authorization_path'])==dep['authorization_SHA256'] and attempt==b.attempt_key(scope.CELL)
    ledger=Path(dep['ledger_path']);job=b.read(ledger)['jobs'][0];settlement=b.read(ledger.with_name(attempt+'-settlement.json'))
    assert job['status']=='completed' and job['exit_code']==0 and job['reservation_released']
    assert settlement['uncertain'] is None and settlement['settlement']['status']=='settled'
    out=Path(dep['output_dir']);candidate=b.read(out/'completed.json');cp=Path(candidate['checkpoint_path'])
    assert candidate['operation']=='train_fourier128' and candidate['native_recipe']=='bispf_optimized' and candidate['pure_formulation']=='fourier128'
    assert candidate['method']=='pg_fourier128' and candidate['generation']=='direct_iter_pg' and candidate['pure_only'] and candidate['real_bag_replacements']==0
    assert candidate['completed_updates']==candidate['fixed_endpoint']==candidate['new_model_updates']==20000 and candidate['initial_completed_updates']==0
    assert candidate['variant_config']==scope.CFG and candidate['hybrid_config'] is None and candidate['qualification_only'] is False
    assert candidate['preset_protocol_SHA256']==b.sha(scope.PROTOCOL) and candidate['checkpoint_SHA256']==b.sha(cp)
    q=torch.load(cp,map_location='cpu',weights_only=False);rec=get_recipe('bispf_optimized',total_steps=20000,batch_size=32,num_particles=512,z_dim=8)
    assert q['schema']==1 and q['native_recipe']=='bispf_optimized' and q['method']=='pg' and q['fold']==q['seed']==0
    assert q['status']=='completed' and q['completed_updates']==q['policy']['completed_steps']==q['fixed_endpoint']==20000
    assert q['cache_manifest_SHA256']==b.sha(Path(auth['cache_dir'])/'manifest.json') and q['training_streams']==training_streams(0,0)
    assert q['recipe']==q['policy']['recipe']==rec.to_dict() and q['variant_config']==scope.CFG
    models={'generator':CountGenerator(torch.ones(2000),torch.ones(2000),zd=8,zc=8),'critic':VectorCritic(356,h=128),'prior':rec.make_prior(generator=torch.Generator().manual_seed(0))}
    assert q['fourier_observation_config']==candidate['fourier_observation_config']==auth['fourier_observation_config']==scope.OBS
    rec.apply_prior_policy(models['prior'])
    assert set(q['policy']['models'])==set(models);finite_values=0
    for name,module in models.items():
        expected=module.state_dict();saved=q['policy']['models'][name];assert saved.keys()==expected.keys()
        for key,v in saved.items():
            assert v.shape==expected[key].shape and v.dtype==expected[key].dtype
            if v.is_floating_point():assert torch.isfinite(v).all();finite_values+=v.numel()
        module.load_state_dict(saved,strict=True)
    assert finite_values==397545
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
    result={'status':'VERIFIED_FIXED_FOURIER128_NATIVE_RESULT','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'attempt':attempt,
            'candidate':candidate,'incumbent':incumbent,'decision':decision,'candidate_commit':auth['candidate_commit'],
            'independent_saved_scores_checkpoint_and_count_exemplars_verified':True,'full_native_recipe_model_and_seed_streams_verified':True,'finite_model_floating_values':finite_values,
            'owned_release_and_clean_settlement_verified':True,'charged_GPU_seconds':settlement['settlement']['charged_GPU_seconds'],
            'array_and_exemplar_SHA256':array_pins,'authorization_SHA256':b.sha(dep['authorization_path']),'preset_protocol_SHA256':b.sha(scope.PROTOCOL),
            'source_SHA256':{str(Path(__file__).resolve()):b.sha(__file__)},'incumbent_changed':False,
            'next_step':('Freeze this accepted seed0 candidate for further fixed-seed confirmation before a stable result claim.' if decision['accept'] else 'Commit negative evidence and revert trial activation; keep current incumbent. Try the next scientifically defined candidate.'),
            'scope':'Four adaptive calibration donors only; fixed observation/critic-input-capacity comparison under unchanged native optimizedBISPF.20k singleparent vs40k selectedensemble. No evaluation donor access or independent biological confirmation.'}
    dest=ROOT/'analysis'/(attempt+'-result.json');assert not dest.exists();dest.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('candidate','incumbent','array_and_exemplar_SHA256')},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);finish(p.parse_args().attempt)
