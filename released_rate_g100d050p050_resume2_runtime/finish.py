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
    sys.path[:0]=[str(ROOT/'analysis/released_tuning'),str(ROOT/'model'),str(b.SCIENCE),str(b.PROJECT)]
    import numpy as np
    import torch
    from particlegan import get_recipe
    from pg_model import CountGenerator
    from critic import VectorCritic
    from released_adapter import training_streams,build
    import released_update as update
    from state_fingerprint import fingerprint
    assert Path(sys.modules[build.__module__].__file__).resolve()==ROOT/'analysis/released_tuning/released_adapter.py'
    from particlegan.policy import _validate_optimizer_state
    from calibration import score,average
    from selection import decide
    dep=b.read(ROOT/'artifacts/allocations'/attempt/'deployment.json');auth=b.read(dep['authorization_path']);scope.validate_authorization(auth)
    assert b.sha(dep['authorization_path'])==dep['authorization_SHA256'] and attempt==b.attempt_key(scope.CELL)
    ledger=Path(dep['ledger_path']);job=b.read(ledger)['jobs'][0];settlement=b.read(ledger.with_name(attempt+'-settlement.json'))
    assert job['status']=='completed' and job['exit_code']==0 and job['reservation_released']
    assert settlement['uncertain'] is None and settlement['settlement']['status']=='settled'
    out=Path(dep['output_dir']);candidate=b.read(out/'completed.json');cp=Path(candidate['checkpoint_path'])
    assert candidate['operation']=='resume_released_hyperparameters' and candidate['native_recipe']=='bispf_optimized' and candidate['pure_formulation']=='released_hyperparameters'
    assert candidate['method']=='pg_released_params' and candidate['generation']=='direct_iter_pg' and candidate['pure_only'] and candidate['real_bag_replacements']==0
    assert candidate['completed_updates']==candidate['fixed_endpoint']==20000 and candidate['new_model_updates']==17253 and candidate['initial_completed_updates']==2747
    assert candidate['variant_config']==scope.CFG and candidate['hybrid_config'] is None and candidate['qualification_only'] is False
    assert candidate['preset_protocol_SHA256']==b.sha(scope.PROTOCOL) and candidate['checkpoint_SHA256']==b.sha(cp)
    assert candidate['resumed_same_model'] is True and candidate['continuation']==auth['continuation']
    assert candidate['continuation_protocol_SHA256']==auth['continuation_protocol_SHA256']==b.sha(scope.RESUME_PROTOCOL)
    assert candidate['parent_checkpoint_SHA256']==auth['continuation']['checkpoint_SHA256']
    parent_receipt=out/'fit/native_parent_restore.json'
    assert candidate['native_parent_restore_path']==str(parent_receipt) and candidate['native_parent_restore_SHA256']==b.sha(parent_receipt)
    restored=b.read(parent_receipt)
    assert restored['status']=='EXACT_FULL_NATIVE_AND_ALL_SCIENTIFIC_PARENT_STATE_RESTORED_ON_CUDA'
    assert restored['whole_parent_scientific_fingerprint']==auth['continuation']['whole_parent_scientific_fingerprint']
    assert restored['initial_completed_updates']==2747 and restored['remaining_model_updates']==17253
    assert restored['native_or_usage_or_data_clock_reset'] is False
    assert restored['original_diagnostic_checkpoints_and_log_prefix_byte_exact'] is True
    assert restored['original_absolute_deadline_unix']==auth['continuation']['original_absolute_deadline_unix']
    for diagnostic in auth['continuation']['diagnostic_checkpoints']:
        assert b.sha(out/'fit'/('checkpoint%05d.pt'%diagnostic['completed_updates']))==diagnostic['SHA256']
    update.configure(scope.CFG);update.configure_hyperparameters(scope.HYPER)
    q=torch.load(cp,map_location='cpu',weights_only=False);rec=update.recipe()
    assert candidate['released_hyperparameters']==q['released_hyperparameters']==auth['released_hyperparameters']==scope.HYPER
    assert candidate['tuning_candidate_id']==auth['tuning_candidate_id']==scope.CANDIDATE_ID
    assert q['policy']['device']=='cuda:0'
    assert set(q['generators'])=={'cell','count'}
    assert q['schema']==1 and q['native_recipe']=='bispf_optimized' and q['method']=='pg' and q['fold']==q['seed']==0
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
            else:raise AssertionError('Unexpected non-tensor released parameter model state')
        module.load_state_dict(saved,strict=True)
    assert finite_values==371177
    # Validate every native optimizer/EMA control on CPU without relabeling the
    # saved CUDA policy or pretending CPU and CUDA RNG states are interchangeable.
    expected_averages=bundle[-1].state_dict()['averages']
    assert q['policy']['averages'].keys()==expected_averages.keys()
    for name,expected in expected_averages.items():
        saved=q['policy']['averages'][name];assert saved.keys()==expected.keys()
        for key,v in saved.items():
            if isinstance(v,torch.Tensor):assert v.shape==expected[key].shape and v.dtype==expected[key].dtype
            else:raise AssertionError('Unexpected non-tensor released parameter model state')
    def finite_tree(value):
        if isinstance(value,torch.Tensor) and value.is_floating_point():assert torch.isfinite(value).all()
        elif isinstance(value,dict):
            for child in value.values():finite_tree(child)
        elif isinstance(value,(list,tuple)):
            for child in value:finite_tree(child)
    finite_tree(q['policy'])
    assert len(q['policy']['optimizers'])==2
    for optimizer,saved in zip(bundle[4:6],q['policy']['optimizers']):
        _validate_optimizer_state(optimizer,saved);optimizer.load_state_dict(saved)
        assert fingerprint(optimizer.state_dict())==fingerprint(saved)
    receipt=out/'native_policy_reload_receipts.json'
    assert candidate['native_reload_path']==str(receipt) and candidate['native_reload_SHA256']==b.sha(receipt)
    reloads=b.read(receipt);assert len(reloads)==3 and [x['updates'] for x in reloads]==[5000,10000,20000]
    assert reloads[-1]['policy_fingerprint']==fingerprint(q['policy'])
    for r in reloads:
        assert r['native_device_metadata_preserved']=='cuda:0' and r['public_CUDA_reload_byte_exact'] is True
        assert b.sha(r['checkpoint_path'])==r['checkpoint_SHA256']
    curve_path=out/'convergence_curve.json'
    assert candidate['convergence_curve_path']==str(curve_path) and candidate['convergence_curve_SHA256']==b.sha(curve_path)
    curve=b.read(curve_path);assert curve['training_finished_before_calibration'] is True and curve['best_checkpoint_selection'] is False
    assert [x['completed_updates'] for x in curve['entries']]==[5000,10000,20000]
    log=out/'fit/learning_curve.jsonl'
    assert candidate['learning_curve_path']==str(log) and candidate['learning_curve_SHA256']==b.sha(log)
    parentlog=Path(auth['continuation']['training_log']['immutable_copy_path']);assert b.sha(parentlog)==auth['continuation']['training_log']['SHA256']
    assert log.read_bytes().startswith(parentlog.read_bytes())
    logs=[__import__('json').loads(line) for line in log.read_text().splitlines()]
    assert logs[0]['step']==1 and logs[-1]['step']==20000
    assert all(a['step']<brow['step'] for a,brow in zip(logs,logs[1:]))
    assert any(row['G_gradient_norm'] is not None for row in logs)
    for row in logs:
        assert row['particle_usage']['total_draws']==32*row['step']
        assert row['particle_usage']['total_unsampled_row_moves']==0
    array_pins={}
    for entry,reload in zip(curve['entries'],reloads):
        updates=entry['completed_updates'];rows=[]
        assert entry['diagnostic_only'] is (updates!=20000) and entry['eligible_for_selection'] is (updates==20000)
        assert entry['checkpoint_path']==reload['checkpoint_path'] and entry['checkpoint_SHA256']==reload['checkpoint_SHA256']
        cq=torch.load(entry['checkpoint_path'],map_location='cpu',weights_only=False)
        assert cq['completed_updates']==cq['policy']['completed_steps']==updates and cq['fixed_endpoint']==20000
        assert cq['status']==('completed' if updates==20000 else 'diagnostic')
        assert cq['recipe']==rec.to_dict() and cq['released_hyperparameters']==scope.HYPER
        assert fingerprint(cq['policy'])==reload['policy_fingerprint']
        destination=out if updates==20000 else out/'diagnostics'/('u%05d'%updates)
        for rep in (0,1):
            folder=destination/f'r{rep}';m=b.read(folder/'metrics.json');assert m['generated_cells']==320000
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
        for key in ('donors','training_seed','calibration_manifest_SHA256','trace_manifest_SHA256','sampling_contract'):assert avg[key]==entry['metrics'][key]
        for key,value in avg.items():
            if isinstance(value,(int,float,list)) and key not in ('donors',):np.testing.assert_allclose(entry['metrics'][key],value,rtol=0,atol=1e-10)
    for key,value in avg.items():
        if isinstance(value,(int,float,list)) and key not in ('donors',):np.testing.assert_allclose(candidate[key],value,rtol=0,atol=1e-10)
    assert not torch.cuda.is_initialized()
    incumbent=b.read(ROOT/'pure_incumbent.json');assert b.sha(ROOT/'pure_incumbent.json')==auth['frozen_pure_incumbent_SHA256']
    decision=decide(candidate,incumbent)
    result={'status':'VERIFIED_RELEASED_PARAMETER_FIXED_ENDPOINT_AND_CONVERGENCE_CURVE_RESULT','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'attempt':attempt,
            'candidate':candidate,'incumbent':incumbent,'decision':decision,'candidate_commit':auth['candidate_commit'],
            'independent_saved_scores_checkpoint_and_count_exemplars_verified':True,'full_native_recipe_model_and_seed_streams_verified':True,'finite_model_floating_values':finite_values,'native_optimizer_and_EMA_checkpoint_state_verified':True,'native_public_CUDA_reload_and_optimizer_state_byte_exact_verified':True,'predeclared_all_three_checkpoint_saved_scores_and_counts_verified':True,'sampled_raw_row_ownership_and_all512rows_used_verified':True,
            'owned_release_and_clean_settlement_verified':True,'charged_GPU_seconds':settlement['settlement']['charged_GPU_seconds'],
            'parent_charged_GPU_seconds':auth['continuation']['parent_charged_GPU_seconds'],
            'charged_GPU_seconds_total_scientific_trial':settlement['settlement']['charged_GPU_seconds']+auth['continuation']['parent_charged_GPU_seconds'],
            'native_parent_all_scientific_streams_and_diagnostics_lineage_verified':True,
            'array_and_exemplar_SHA256':array_pins,'authorization_SHA256':b.sha(dep['authorization_path']),'preset_protocol_SHA256':b.sha(scope.PROTOCOL),
            'source_SHA256':{str(Path(__file__).resolve()):b.sha(__file__)},'incumbent_changed':False,
            'next_step':('Freeze this accepted seed0 candidate for further fixed-seed confirmation before a stable result claim.' if decision['accept'] else 'Commit negative evidence and revert trial activation; keep current incumbent. Try the next scientifically defined candidate.'),
            'scope':'Four adaptive calibration donors only, no evaluation selection. Fixed released bispf_optimized count family and all data/architecture/seed streams preserved; same registered G/D/prior learning rates applied via public recipe fields, exact2747parent continuation preserving all scientific streams and original journal plus pending5k/10k snapshots. Original20k endpoint alone eligible;5k/10k diagnostics evaluated after training, with same traces. No stable or biological ATH claim.'}
    dest=ROOT/'analysis'/(attempt+'-result.json');assert not dest.exists();dest.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('candidate','incumbent','array_and_exemplar_SHA256')},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);finish(p.parse_args().attempt)
