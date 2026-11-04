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
    sys.path[:0]=[str(ROOT/'halloween_model'),str(ROOT/'model'),str(b.SCIENCE),str(b.PROJECT)]
    import numpy as np
    import torch
    from particlegan import get_recipe
    from pg_model import CountGenerator
    from critic import VectorCritic
    from halloween_pg_adapter import training_streams,build
    assert Path(sys.modules[build.__module__].__file__).resolve()==ROOT/'halloween_model/halloween_pg_adapter.py'
    from particlegan.policy import _validate_optimizer_state
    from calibration import score,average
    from selection import decide
    dep=b.read(ROOT/'artifacts/allocations'/attempt/'deployment.json');auth=b.read(dep['authorization_path']);scope.validate_authorization(auth)
    assert b.sha(dep['authorization_path'])==dep['authorization_SHA256'] and attempt==b.attempt_key(scope.CELL)
    ledger=Path(dep['ledger_path']);job=b.read(ledger)['jobs'][0];settlement=b.read(ledger.with_name(attempt+'-settlement.json'))
    assert job['status']=='completed' and job['exit_code']==0 and job['reservation_released']
    assert settlement['uncertain'] is None and settlement['settlement']['status']=='settled'
    out=Path(dep['output_dir']);candidate=b.read(out/'completed.json');cp=Path(candidate['checkpoint_path'])
    assert candidate['operation']=='train_halloween_native' and candidate['native_recipe']=='halloween' and candidate['pure_formulation']=='halloween'
    assert candidate['method']=='pg_halloween' and candidate['generation']=='direct_iter_pg' and candidate['pure_only'] and candidate['real_bag_replacements']==0
    assert candidate['completed_updates']==candidate['fixed_endpoint']==candidate['new_model_updates']==20000 and candidate['initial_completed_updates']==0
    assert candidate['variant_config']==scope.CFG and candidate['hybrid_config'] is None and candidate['qualification_only'] is False
    assert candidate['preset_protocol_SHA256']==b.sha(scope.PROTOCOL) and candidate['checkpoint_SHA256']==b.sha(cp)
    q=torch.load(cp,map_location='cpu',weights_only=False);rec=get_recipe('halloween',total_steps=20000,batch_size=32,num_particles=512,z_dim=8)
    assert set(q['generators'])=={'cell','count'}
    assert q['schema']==1 and q['native_recipe']=='halloween' and q['method']=='pg' and q['fold']==q['seed']==0
    assert q['status']=='completed' and q['completed_updates']==q['policy']['completed_steps']==q['fixed_endpoint']==20000
    assert q['cache_manifest_SHA256']==b.sha(Path(auth['cache_dir'])/'manifest.json') and q['training_streams']==training_streams(0,0)
    assert q['recipe']==q['policy']['recipe']==rec.to_dict() and q['variant_config']==scope.CFG
    bundle,*_=build(torch.ones(2000),torch.ones(2000),'cpu',0,0)
    assert bundle[3].z.requires_grad and list(bundle[3].parameters())
    assert len(bundle[4].param_groups)==2
    assert not torch.equal(q['policy']['models']['prior']['z'],bundle[3].z.detach())
    assert q['usage']['steps']==20000 and int(q['usage']['draws'].sum())==640000
    assert int(q['usage']['moved_steps'].sum())>0 and int(q['usage']['unsampled_moved_steps'].sum())==0
    assert bool((q['usage']['draws']>0).all())
    assert q['prior_regularizer']=={}
    models={'generator':bundle[1],'critic':bundle[2],'prior':bundle[3]}
    assert set(q['policy']['models'])==set(models);finite_values=0
    for name,module in models.items():
        expected=module.state_dict();saved=q['policy']['models'][name];assert saved.keys()==expected.keys()
        for key,v in saved.items():
            if isinstance(v,torch.Tensor):
                assert v.shape==expected[key].shape and v.dtype==expected[key].dtype
                if v.is_floating_point():assert torch.isfinite(v).all();finite_values+=v.numel()
            else:raise AssertionError('Unexpected non-tensor Halloween model state')
        module.load_state_dict(saved,strict=True)
    assert finite_values==371105
    # Validate every native optimizer/EMA control on CPU without relabeling the
    # saved CUDA policy or pretending CPU and CUDA RNG states are interchangeable.
    expected_averages=bundle[-1].state_dict()['averages']
    assert q['policy']['averages'].keys()==expected_averages.keys()
    for name,expected in expected_averages.items():
        saved=q['policy']['averages'][name];assert saved.keys()==expected.keys()
        for key,v in saved.items():
            if isinstance(v,torch.Tensor):assert v.shape==expected[key].shape and v.dtype==expected[key].dtype
            else:raise AssertionError('Unexpected non-tensor Halloween model state')
    def finite_tree(value):
        if isinstance(value,torch.Tensor) and value.is_floating_point():assert torch.isfinite(value).all()
        elif isinstance(value,dict):
            for child in value.values():finite_tree(child)
        elif isinstance(value,(list,tuple)):
            for child in value:finite_tree(child)
    finite_tree(q['policy'])
    assert len(q['policy']['optimizers'])==2
    for saved in q['policy']['optimizers']:
        assert saved['tensorflow_v1']=={'schema_version':1,'gradient_law':'dense_real'}
        for group in saved['param_groups']:assert group['_tf_step']==20000
    for optimizer,saved in zip(bundle[4:6],q['policy']['optimizers']):
        _validate_optimizer_state(optimizer,saved);optimizer.load_state_dict(saved)
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
    result={'status':'VERIFIED_NATIVE_HALLOWEEN_COUNT_PRESET_RESULT','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'attempt':attempt,
            'candidate':candidate,'incumbent':incumbent,'decision':decision,'candidate_commit':auth['candidate_commit'],
            'independent_saved_scores_checkpoint_and_count_exemplars_verified':True,'full_native_recipe_model_and_seed_streams_verified':True,'finite_model_floating_values':finite_values,'native_optimizer_and_EMA_checkpoint_state_verified':True,'native_TensorFlowV1Adam_all_role_clocks_and_marker_verified':True,'sampled_raw_row_ownership_and_all512rows_used_verified':True,
            'owned_release_and_clean_settlement_verified':True,'charged_GPU_seconds':settlement['settlement']['charged_GPU_seconds'],
            'array_and_exemplar_SHA256':array_pins,'authorization_SHA256':b.sha(dep['authorization_path']),'preset_protocol_SHA256':b.sha(scope.PROTOCOL),
            'source_SHA256':{str(Path(__file__).resolve()):b.sha(__file__)},'incumbent_changed':False,
            'next_step':('Freeze this accepted seed0 candidate for further fixed-seed confirmation before a stable result claim.' if decision['accept'] else 'Commit negative evidence and revert trial activation; keep current incumbent. Try the next scientifically defined candidate.'),
            'scope':'Four adaptive calibration donors only; published native Halloween count-preset, exact native TensorFlow-v1 Adam roles/clocks/moments, least-squares labels, zero penalty/priorreg/noise/EMA and constant rates. SamePG8 data/networks/512x8learnedparticles/11streams/directcounts; fullpreset mechanisms confounded.20k singleparent vs40k selectedensemble. No evaluation donor access or independent biological confirmation.'}
    dest=ROOT/'analysis'/(attempt+'-result.json');assert not dest.exists();dest.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('candidate','incumbent','array_and_exemplar_SHA256')},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);finish(p.parse_args().attempt)
