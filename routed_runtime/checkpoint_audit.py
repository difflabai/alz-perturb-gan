"""CPU tensor/control audit; original CUDA engines are validated by GPU reload.

No device metadata or saved random stream is rewritten. Selected serving state
follows the pinned native UpdatePolicy.served_snapshot implementation.
"""
import copy


def selected_states(policy, runtime):
    from state_fingerprint import fingerprint
    models=copy.deepcopy(policy['models'])
    source=policy['served_source']
    settled=any(value is not None and role=='table' and value.get('last_decisive')==-1
                for row,roles in zip(policy['lr_settle'],policy['roles'])
                for value,role in zip(row,roles))
    assert source==('averaged' if runtime.recipe.serve_average>0 and settled else 'fast')
    if source=='averaged':
        for name,module in runtime.policy._average_modules().items():
            for key,_ in module.named_parameters(remove_duplicate=False):
                models[name][key]=policy['averages'][name][key].clone()
        spec=runtime.policy.routed_control.spec
        for key in (spec.log_mass_key,*spec.row_buffers):
            if key in policy['averages']['router']:
                models['router'][key]=policy['averages']['router'][key].clone()
        owner,key=policy['table_location']
        models[owner][key]=policy['averaged_table'].clone()
    table=policy['averaged_table'] if source=='averaged' else policy['table']
    owner,key=policy['table_location']
    assert fingerprint(models[owner][key])==fingerprint(table)
    return models,table


def audit_checkpoint(q,ctx,*,steps=20000,expected_device='cuda:0'):
    import numpy as np
    import torch
    from particlegan.policy import _validate_optimizer_state
    import routed_pg_adapter as a
    from routed_counts import validate_context
    from state_fingerprint import fingerprint
    assert ctx.device.type=='cpu'
    runtime=a.build(ctx,0);policy=q['policy'];expected=runtime.policy.state_dict()
    assert q['schema']==1 and q['method']=='pg' and q['fold']==q['seed']==0
    assert q['native_recipe']=='e22_routed' and q['variant_config']==a.CONFIG and q['routed_config']==a.ROUTED_CONFIG
    assert q['fixed_endpoint']==20000 and q['completed_updates']==policy['completed_steps']==steps
    assert q['recipe']==policy['recipe']==runtime.recipe.to_dict() and q['training_streams']==a.training_streams(0,0)
    assert q['partition']==ctx.partition.summary()
    assert q['cache_manifest_SHA256']==a.sha(ctx.cache/'manifest.json') and q['prior_regularizer']=={}
    assert q['fitting_contexts_seen']==q['protected_contexts_seen']==32*steps
    for name in ('fitting_rng','protected_rng'):
        rng=np.random.default_rng();rng.bit_generator.state=q[name]
        assert fingerprint(rng.bit_generator.state)==fingerprint(q[name])
    assert set(policy)==set(expected) and policy['schema']==1
    assert policy['device']==expected_device and policy['dtype']==expected['dtype']
    for key in ('roles','row_semantics','table_requires_grad','table_location','requires_grad','initial_lrs'):
        assert policy[key]==expected[key]
    values=0
    for family,modules in (('models',runtime.policy._training_modules()),('averages',runtime.policy._average_modules())):
        assert set(policy[family])==set(modules)
        for name,module in modules.items():
            saved=policy[family][name];reference=module.state_dict();assert saved.keys()==reference.keys()
            for key,value in saved.items():
                assert isinstance(value,torch.Tensor) and value.shape==reference[key].shape and value.dtype==reference[key].dtype
                if name=='router' and key=='log_mass':
                    assert not torch.isnan(value).any() and not torch.isposinf(value).any() and torch.isfinite(value).any()
                else:assert torch.isfinite(value).all()
                if family=='models' and value.is_floating_point():values+=value.numel()
            module.load_state_dict(saved,strict=True)
    assert values==sum(v.numel() for m in runtime.policy._training_modules().values() for v in m.state_dict().values() if v.is_floating_point())
    for family,key in (('models','table'),('averages','averaged_table')):
        assert fingerprint(policy[key])==fingerprint(policy[family]['prior']['z'])
    assert not torch.equal(policy['models']['prior']['z'],expected['models']['prior']['z']) if steps else True
    assert torch.isfinite(policy['output_noise']).all() and policy['output_noise'].shape==expected['output_noise'].shape
    assert len(policy['optimizers'])==len(runtime.policy.optimizers)==2
    def finite_tree(value):
        if isinstance(value,torch.Tensor) and value.is_floating_point():assert torch.isfinite(value).all()
        elif isinstance(value,dict):
            for child in value.values():finite_tree(child)
        elif isinstance(value,(list,tuple)):
            for child in value:finite_tree(child)
    for optimizer,saved in zip(runtime.policy.optimizers,policy['optimizers']):
        finite_tree(saved);_validate_optimizer_state(optimizer,saved);optimizer.load_state_dict(saved)
    # These controllers do not consume the CUDA stream and support CPU tensors.
    assert policy['controller'] is not None and policy['lr_settle'] is not None
    copy.deepcopy(runtime.policy.controller).load_state_dict(policy['controller'])
    copy.deepcopy(runtime.policy.lr_settle).load_state_dict(policy['lr_settle'],runtime.policy.optimizers)
    assert policy['birth_death'] is None and policy['row_evidence'] is None
    route=policy['routing'];control=runtime.policy.routed_control;reference=control.state_dict()
    assert route.keys()==reference.keys() and route['schema']==1
    for key in ('config','table_shape','row_ownership'):assert route[key]==reference[key]
    clock=route['probe_clock'];assert clock['observed_updates']==steps
    assert 0<=clock['last_probe_update']<=steps and steps-clock['last_probe_update']<20
    assert set(route['pools'])==set(reference['pools'])
    for prefix in ('fit','guard'):
        context=route['pools'][prefix+'_context'];targets=route['pools'][prefix+'_targets']
        assert context is not None and targets is not None
        assert context.shape==(control.spec.reservoir_size,8+7+128+12)
        assert targets.shape==(control.spec.reservoir_size,128,2*len(ctx.mean))
        assert context.dtype==targets.dtype==torch.float32
        assert torch.isfinite(targets).all();validate_context(context)
        raw=targets[...,:len(ctx.mean)]
        assert (raw>=0).all() and torch.equal(raw,raw.floor())
        assert torch.equal(context[:,15:143],raw.sum(-1))
        assert 0<route[prefix+'_fill']<=control.spec.reservoir_size
        assert 0<=route[prefix+'_cursor']<control.spec.reservoir_size
    control._check_disjoint(route['pools']['fit_context'][:route['fit_fill']],route['pools']['guard_context'][:route['guard_fill']])
    control.evidence.check_state(route['evidence'])
    assert type(route['rows_since_eval']) is int and route['rows_since_eval']>=0
    assert type(route['probe_cursor']) is int and 0<=route['probe_cursor']<512
    assert type(route['restart_router']) is bool
    assert route['counters'].keys()==reference['counters'].keys()
    assert all(type(v) is int and v>=0 for v in route['counters'].values())
    for name in ('latest_gradient','moved_rows'):
        value=route[name]
        if value is not None:
            if name=='latest_gradient':assert value.shape==(512,8) and torch.isfinite(value).all()
            else:assert value.ndim==1 and value.dtype==torch.long and (value>=0).all() and (value<512).all()
    # Only metadata is checked here: these exact engines are reloaded on CUDA
    # by the worker. Native full-CUDA receipt is independently bound below.
    assert policy['streams'].keys()==expected['streams'].keys()
    for value in [*policy['streams'].values(),policy['cpu_rng'],route['stream']]:
        assert value.ndim==1 and value.dtype==torch.uint8 and value.numel()>0
    if expected_device.startswith('cuda'):
        assert policy['cuda_rng'] is not None and policy['cuda_rng'].dtype==torch.uint8
    models,table=selected_states(policy,runtime)
    assert not torch.cuda.is_initialized()
    return {'model_floating_values':values,'policy_fingerprint':fingerprint(policy),
            'served_source':policy['served_source'],'served_models_fingerprint':fingerprint(models),
            'served_table_fingerprint':fingerprint(table),'routing_probe_clock':clock,
            'routing_counters':route['counters'],'native_output_sigma':float(policy['output_noise'].exp())}
