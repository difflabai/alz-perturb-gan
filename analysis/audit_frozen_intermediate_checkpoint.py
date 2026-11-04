"""Read-only CPU portability audit of an actual native recovery checkpoint.

No GPU context, next update, calibration forecasts, model ranking or native
source mutation. The copied checkpoint is a milestone, not a completed model.
"""
import datetime,hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path('/ml2/alz');HERE=ROOT/'frozen_runtime';sys.path.insert(0,str(HERE))
import binding as b
import preset_scope as scope

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    attempt='CAL_GPU0_V1_frozen_prior_v1_f0_pg_s0'
    allocation=ROOT/'artifacts/allocations'/attempt
    auth=b.read(allocation/'authorization.json');scope.validate_authorization(auth)
    fit=ROOT/'runs'/attempt/'fit'
    for _ in range(5):
        receipt=b.read(fit/'checkpoint_receipt.json')
        payload=(fit/receipt['path']).read_bytes()
        pin=hashlib.sha256(payload).hexdigest()
        if receipt==b.read(fit/'checkpoint_receipt.json') and pin==receipt['SHA256']:break
        time.sleep(.25)
    else:raise RuntimeError('Checkpoint publication changed; reobserve rather than assume a stable snapshot')
    step=receipt['completed_updates']
    assert 0<step<20000 and step%1000==0 and receipt['status']=='recovery'
    folder=ROOT/'artifacts/checkpoint_milestones'/attempt/(str(step)+'-'+pin)
    folder.mkdir(parents=True,exist_ok=False)
    preserved=folder/'recovery.pt';preserved.write_bytes(payload);assert b.sha(preserved)==pin
    sys.path.extend(map(str,b.native_site_roots(scope.CELL,auth['files_SHA256'])))
    sys.path[:0]=[str(ROOT/'frozen_model'),str(ROOT/'model'),str(b.SCIENCE),str(b.PROJECT)]
    import torch
    torch.set_num_threads(1)
    from particlegan.policy import _validate_optimizer_state
    from frozen_pg_adapter import build,training_streams
    q=torch.load(preserved,map_location='cpu',weights_only=False)
    assert q['status']=='recovery' and q['completed_updates']==q['policy']['completed_steps']==step
    assert q['fixed_endpoint']==20000 and q['native_recipe']=='bispf_optimized'
    assert q['prior_update']==auth['prior_update']=='frozen'
    assert auth['frozen_prior_config']==scope.FROZEN
    assert set(q['generators'])=={'cell','count'}
    assert q['cache_manifest_SHA256']==b.sha(Path(auth['cache_dir'])/'manifest.json')
    assert q['training_streams']==training_streams(0,0) and q['fold']==q['seed']==0
    bundle,*_=build(torch.ones(2000),torch.ones(2000),'cpu',0,0)
    assert q['recipe']==q['policy']['recipe']==bundle[0].to_dict()
    assert not bundle[3].z.requires_grad and not list(bundle[3].parameters())
    assert len(bundle[4].param_groups)==1 and len(q['policy']['optimizers'][0]['param_groups'])==1
    assert torch.equal(q['policy']['models']['prior']['z'],bundle[3].z.detach())
    assert q['prior_regularizer']=={}
    assert q['usage']['steps']==step and int(q['usage']['draws'].sum())==32*step
    assert int(q['usage']['moved_steps'].sum())==int(q['usage']['unsampled_moved_steps'].sum())==0
    models={'generator':bundle[1],'critic':bundle[2],'prior':bundle[3]};floats=0
    assert q['policy']['models'].keys()==models.keys()
    for name,module in models.items():
        expected=module.state_dict();saved=q['policy']['models'][name];assert expected.keys()==saved.keys()
        for key,value in saved.items():
            assert value.shape==expected[key].shape and value.dtype==expected[key].dtype
            if value.is_floating_point():assert torch.isfinite(value).all();floats+=value.numel()
        module.load_state_dict(saved,strict=True)
    assert floats==371105
    def finite_tree(value):
        if isinstance(value,torch.Tensor) and value.is_floating_point():assert torch.isfinite(value).all()
        elif isinstance(value,dict):
            for child in value.values():finite_tree(child)
        elif isinstance(value,(tuple,list)):
            for child in value:finite_tree(child)
    finite_tree(q)
    expected_averages=bundle[-1].state_dict()['averages']
    assert q['policy']['averages'].keys()==expected_averages.keys()
    for name,expected in expected_averages.items():
        saved=q['policy']['averages'][name];assert saved.keys()==expected.keys()
        for key,value in saved.items():assert value.shape==expected[key].shape and value.dtype==expected[key].dtype
    for optimizer,saved in zip(bundle[4:6],q['policy']['optimizers']):
        _validate_optimizer_state(optimizer,saved);optimizer.load_state_dict(saved)
    assert not torch.cuda.is_initialized()
    result={'status':'ACTUAL_INTERMEDIATE_GPU_CHECKPOINT_CPU_PORTABILITY_AUDIT_PASSED',
            'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'attempt':attempt,
            'completed_updates':step,'fixed_endpoint':20000,'native_recovery_checkpoint_SHA256':pin,
            'preserved_checkpoint_path':str(preserved),'full_model_floating_values':floats,
            'native_optimizer_and_average_states_verified':True,
            'native_fixed_initial_particle_bank_exactly_preserved':True,
            'no_prior_optimizer_group_or_prior_regularization_verified':True,
            'raw_particle_movement_counters_zero':True,
            'total_generator_particle_draws':int(q['usage']['draws'].sum()),
            'all_saved_floating_tensors_finite_including_regularizer_and_usage':True,
            'GPU_policy_device_metadata_preserved':q['policy']['device'],
            'source_SHA256':{str(Path(__file__).resolve()):b.sha(__file__)},
            'authorization_SHA256':b.sha(allocation/'authorization.json'),
            'CUDA_initialized':False,'training_update_executed':False,
            'calibration_forecasts_generated':False,'quality_measured':False,
            'scope':'Actual stable intermediate checkpoint copied by matching native receipt/hash before and after read. Full CPU model and optimizer validation succeeds without rewriting saved CUDA policy metadata or using CUDA RNG states as CPU streams. This is not GPU next-update replay, completed endpoint, quality selection, confirmation or lease/admission.'}
    output=ROOT/'analysis'/(attempt+f'-checkpoint-{step}-portability.json')
    assert not output.exists();output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
