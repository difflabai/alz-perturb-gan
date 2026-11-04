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
    sys.path[:0]=[str(ROOT/'routed_model'),str(ROOT/'model'),str(b.SCIENCE),str(b.PROJECT)]
    import numpy as np
    import torch
    import routed_pg_adapter as model
    from checkpoint_audit import audit_checkpoint
    assert Path(model.__file__).resolve()==ROOT/'routed_model/routed_pg_adapter.py'
    from calibration import score,average
    from selection import decide
    dep=b.read(ROOT/'artifacts/allocations'/attempt/'deployment.json');auth=b.read(dep['authorization_path']);scope.validate_authorization(auth)
    assert b.sha(dep['authorization_path'])==dep['authorization_SHA256'] and attempt==b.attempt_key(scope.CELL)
    ledger=Path(dep['ledger_path']);job=b.read(ledger)['jobs'][0];settlement=b.read(ledger.with_name(attempt+'-settlement.json'))
    assert job['status']=='completed' and job['exit_code']==0 and job['reservation_released']
    assert settlement['uncertain'] is None and settlement['settlement']['status']=='settled'
    out=Path(dep['output_dir']);candidate=b.read(out/'completed.json');cp=Path(candidate['checkpoint_path'])
    assert candidate['operation']=='resume_routed_e22' and candidate['native_recipe']=='e22_routed' and candidate['pure_formulation']=='e22_routed_count'
    assert candidate['method']=='pg_e22_routed_counts' and candidate['generation']=='native_served_routed_count_forward'
    assert candidate['pure_only'] and candidate['real_bag_replacements']==0 and candidate['integer_count_contract_passed']
    assert candidate['completed_updates']==candidate['fixed_endpoint']==20000 and candidate['new_model_updates']==12987 and candidate['initial_completed_updates']==7013
    assert candidate['native_recipe_schedule_horizon'] is None and candidate['parent_checkpoint_SHA256']==auth['continuation']['checkpoint_SHA256'] and candidate['automatic_successor'] is False
    assert candidate['variant_config']==scope.CFG and candidate['routed_config']==scope.ROUTED and candidate['hybrid_config'] is None and candidate['qualification_only'] is False
    assert candidate['preset_protocol_SHA256']==b.sha(scope.PROTOCOL) and candidate['checkpoint_SHA256']==b.sha(cp)
    assert cp==out/'fit/final.pt'
    assert candidate['continuation']==auth['continuation'] and candidate['resumed_same_model'] is True
    parent_receipt=out/'fit/native_parent_restore.json'
    assert candidate['native_parent_restore_path']==str(parent_receipt) and candidate['native_parent_restore_SHA256']==b.sha(parent_receipt)
    parent_restored=b.read(parent_receipt)
    assert parent_restored['status']=='EXACT_NATIVE_PARENT_CUDA_POLICY_AND_BOTH_DATA_SAMPLERS_RESTORED'
    assert parent_restored['parent_checkpoint_SHA256']==auth['continuation']['checkpoint_SHA256']
    assert parent_restored['initial_completed_updates']==7013 and parent_restored['remaining_model_updates']==12987
    assert parent_restored['original_absolute_deadline_unix']==auth['continuation']['original_absolute_deadline_unix']
    assert parent_restored['native_or_data_clock_reset'] is False
    torch.set_num_threads(1)
    q=torch.load(cp,map_location='cpu',weights_only=False)
    assert q['status']=='completed'
    ctx=model.Context(Path(auth['cache_dir']),'cpu')
    audit=audit_checkpoint(q,ctx)
    assert audit['model_floating_values']==371617 and candidate['partition']==ctx.partition.summary()
    receipt=out/'native_checkpoint_reload.json';reference=out/'native_served_reference.json'
    assert candidate['native_reload_receipt_path']==str(receipt) and candidate['native_reload_receipt_SHA256']==b.sha(receipt)
    assert candidate['served_reference_path']==str(reference) and candidate['served_reference_SHA256']==b.sha(reference)
    reload=b.read(receipt);served=b.read(reference)
    assert reload['status']=='FULL_NATIVE_PUBLIC_CUDA_POLICY_RELOAD_AND_BYTE_FINGERPRINT_PASSED'
    assert reload['policy_fingerprint']==audit['policy_fingerprint']
    assert reload['native_device_metadata_preserved']==q['policy']['device']=='cuda:0'
    assert reload['both_independent_data_sampler_states_match'] and reload['CUDA_rng_states_used_only_on_admitted_CUDA_policy']
    assert served['status']=='NATIVE_COUPLED_SERVED_COUNT_REFERENCE'
    for record in (reload,served):
        assert record['checkpoint_path']==str(cp) and record['checkpoint_SHA256']==b.sha(cp) and record['completed_updates']==20000
    assert served['models_fingerprint']==audit['served_models_fingerprint'] and served['table_fingerprint']==audit['served_table_fingerprint']
    assert candidate['serving_contract']==served['serving_contract']=={'weights':'native_coupled_served_snapshot','source':audit['served_source'],
        'output_noise':False,'perturb':False,'raw_count_half_primary':True}
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
    result={'status':'VERIFIED_NATIVE_ROUTED_E22_COUNT_ADAPTER_RESULT','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'attempt':attempt,
            'candidate':candidate,'incumbent':incumbent,'decision':decision,'candidate_commit':auth['candidate_commit'],
            'independent_saved_scores_checkpoint_and_count_exemplars_verified':True,'full_native_recipe_model_and_seed_streams_verified':True,'model_floating_values':audit['model_floating_values'],
            'native_optimizer_EMA_and_CPU_compatible_controller_audit_verified':True,
            'full_original_device_native_public_policy_reload_verified':True,
            'native_coupled_served_weights_table_mass_and_count_reference_verified':True,
            'fitting_protection_sampler_and_reservoir_count_contract_verified':True,'checkpoint_audit':audit,
            'owned_release_and_clean_settlement_verified':True,'charged_GPU_seconds':settlement['settlement']['charged_GPU_seconds'],
            'parent_charged_GPU_seconds':auth['continuation']['parent_charged_GPU_seconds'],
            'charged_GPU_seconds_total_scientific_trial':settlement['settlement']['charged_GPU_seconds']+auth['continuation']['parent_charged_GPU_seconds'],
            'native_7013_parent_and_remaining12987_update_lineage_verified':True,
            'array_and_exemplar_SHA256':array_pins,'authorization_SHA256':b.sha(dep['authorization_path']),'preset_protocol_SHA256':b.sha(scope.PROTOCOL),
            'source_SHA256':{str(Path(__file__).resolve()):b.sha(__file__)},'incumbent_changed':False,
            'next_step':('Freeze this accepted seed0 candidate for further fixed-seed confirmation before a stable result claim.' if decision['accept'] else 'Commit negative evidence and revert trial activation; keep current incumbent. Try the next scientifically defined candidate.'),
            'scope':scope.read(scope.PROTOCOL)['scope']}
    dest=ROOT/'analysis'/(attempt+'-result.json');assert not dest.exists();dest.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('candidate','incumbent','array_and_exemplar_SHA256')},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);finish(p.parse_args().attempt)
