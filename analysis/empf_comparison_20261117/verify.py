"""Actual TRAIN-only GPU semantics and full fresh-process EMPF continuation."""
import os
os.environ.update(OMP_NUM_THREADS='3', OPENBLAS_NUM_THREADS='3', MKL_NUM_THREADS='3',
                  CUDA_VISIBLE_DEVICES='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
import argparse
import json
import time
import host
import common as C

V = host.load('frozen_release_verify_helpers', host.BASE/'verify.py')
setup, scope, snapshot = V.setup, V.scope, V.snapshot


def observer(T, b):
    rec,G,D,P,og,od,pen,loss,reg,policy = b
    def binding():
        assert rec.critic_formulation == 'empf' and rec.rhbi_weight == 0
        assert rec.reg_every == pen.regularizer.lazy_k == 1
        assert pen.regularizer.coeff == rec.reg_coeff == 1.
        assert pen.regularizer.input_domain == 'continuous'
        assert type(loss).__name__ == 'GANLoss' and loss.loss == 'non_saturating'
        assert reg.weight == reg.effective_weight(P.z) == 1.
        assert len(reg.references)==1 and reg.references[0] is P
        assert int(P.moment_tether_count)==512
        assert reg.empf_record is od.record is D._empf_record is P._empf_record is og._empf_record is pen.regularizer.record
        assert od.record.calls == od.record.observed_steps == policy.completed_steps
        packet=od.record.state_dict()['empf']
        assert packet['schema']==2 and not packet['proto_initialized']
        assert packet['proto_sum'] is packet['proto_count'] is None
        assert packet['thermal_scale']==packet['step_scale']==1.
        assert packet['chord_phase']==packet['revival_phase']==packet['revivals']==0
    binding()
    for field,invalid in (('lazy_k',2),('input_domain','auto')):
        saved=getattr(pen.regularizer,field);setattr(pen.regularizer,field,invalid)
        try:
            try:binding()
            except AssertionError:pass
            else:raise RuntimeError('semantic control failed to reject '+field)
        finally:setattr(pen.regularizer,field,saved)
    stats=dict(lazy_k=1, regularizer_calls=0, differentiable_penalty_calls=0,
               nonzero_penalty_calls=0, prototype_bank_inactive_native_highdim_route=True,
               feature_width=2150, condition_width=7, shared_record=True,
               rejected_wrong_lazy_k_and_automatic_domain=True,
               initial_prior_moment_reference_bound=True)
    original=pen.regularizer.penalty
    def observed(live,x,y,step,*args,**kwargs):
        assert x.shape==y.shape==(32,2150) and kwargs['is_conditioned'] is True
        assert len(kwargs['cond_tensors'])==1 and kwargs['cond_tensors'][0].shape==(32,7)
        assert pen.regularizer.lazy_k==1 and pen.regularizer.input_domain=='continuous'
        value,detail=original(live,x,y,step,*args,**kwargs)
        assert T.torch.isfinite(value)
        stats['regularizer_calls']+=1
        stats['differentiable_penalty_calls']+=int(value.requires_grad)
        stats['nonzero_penalty_calls']+=int(float(value.detach())!=0)
        return value,detail
    pen.regularizer.penalty=observed
    return stats,binding


def advance(T,ctx,b,g,r,u,n):
    return [T.advance(ctx,b,g,r,u,execution_limit=20000) for _ in range(n)]


def verify(lane, replay=False):
    started=time.monotonic();C.check_sources();control=C.control_identity()
    T,G,provenance=setup('0.10.0',lane)
    cache,opened,permitted=scope();anchor=C.cache_anchor()
    root=host.RUNS/'verification';root.mkdir(parents=True,exist_ok=True)
    ctx=T.Context(cache,'cuda',T.parse_arm('pbulk'))
    middle_path=root/f'replay200_{lane}.pt';expected_path=root/f'expected300_{lane}.pt'
    rejected=[];initial_equal=None
    if replay:
        q=T.torch.load(middle_path,map_location='cpu',weights_only=False)
        b,g,r,u,streams=T.restore(ctx,q,0)
        expected=T.torch.load(expected_path,map_location='cpu',weights_only=False)
        stats,binding=observer(T,b)
        metrics=advance(T,ctx,b,g,r,u,100);binding()
        assert T._same(metrics,expected['metrics'])
        assert T._same(snapshot(T,ctx,b,g,r,u,streams),expected['snapshot'])
    else:
        b,g,r,u,streams=T.build(ctx,0)
        stats,binding=observer(T,b)
        baseline=T.torch.load(control['checkpoint'],map_location='cpu',weights_only=False)['initial']['policy']
        actual=snapshot(T,ctx,b,g,r,u,streams)['policy']
        for field, module in (('generator',b[1]),('critic',b[2]),('prior',b[3])):
            assert field in actual['models'] and field in baseline['models']
            for key,param in module.named_parameters():
                assert T.torch.equal(actual['models'][field][key].cpu(),baseline['models'][field][key].cpu()),(field,key)
        initial_equal=True
        advance(T,ctx,b,g,r,u,5);first=snapshot(T,ctx,b,g,r,u,streams)
        for field in ('particlegan_version','native_recipe'):
            try:T.restore(ctx,first|{field:'incompatible'},0)
            except ValueError:rejected.append(field)
            else:raise AssertionError('accepted incompatible '+field)
        advance(T,ctx,b,g,r,u,195);binding()
        T.torch.save(snapshot(T,ctx,b,g,r,u,streams),middle_path)
        fresh=advance(T,ctx,b,g,r,u,100);binding()
        endpoint=snapshot(T,ctx,b,g,r,u,streams)
        T.torch.save(dict(snapshot=endpoint,metrics=fresh),expected_path)
        del b,g,r,u
        T.torch.cuda.empty_cache()
        ctx2=T.Context(cache,'cuda',T.parse_arm('pbulk'))
        b,g,r,u,streams=T.restore(ctx2,T.torch.load(middle_path,map_location='cpu',weights_only=False),0)
        replay_stats,replay_binding=observer(T,b)
        metrics=advance(T,ctx2,b,g,r,u,100);replay_binding()
        assert T._same(fresh,metrics)
        assert T._same(endpoint,snapshot(T,ctx2,b,g,r,u,streams))
        assert replay_stats['regularizer_calls']==100
    assert stats['regularizer_calls']==stats['differentiable_penalty_calls']==(100 if replay else 300)
    assert stats['nonzero_penalty_calls']>0
    assert opened==permitted and C.cache_anchor()==anchor
    C.check_sources()
    total=T.torch.cuda.get_device_properties(0).total_memory
    allocated=T.torch.cuda.max_memory_allocated();reserved=T.torch.cuda.max_memory_reserved()
    assert max(allocated,reserved)<=total*.04 and len(os.sched_getaffinity(0))<=3
    value=dict(status='passed_fresh_process_exact_200_to_300' if replay else 'passed_actual_EMPF_semantics_and_full_replay',
        lane=lane,full_checkpoint_and_metrics_exact=True,verification_code_sha256=host.sha(__file__),
        provenance=provenance,semantics=stats,TRAIN_files_opened=sorted(opened),cache_anchor=anchor,
        initial_parameter_control=control,
        checkpoint_sha256={p.name:host.sha(p) for p in (middle_path,expected_path)},
        affinity=sorted(os.sched_getaffinity(0)),seconds=time.monotonic()-started,
        gpu=dict(selector='1',memory_fraction=.04,max_allocated_bytes=allocated,max_reserved_bytes=reserved,total_bytes=total))
    if not replay:value.update(rejected_checkpoint_mismatches=rejected,initial_G_D_prior_parameters_equal_release_control=initial_equal)
    C.write(root/f'{"fresh_process" if replay else "verification"}_{lane}.json',value)
    print(json.dumps(value),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--lane',choices=('native','matched'),required=True);p.add_argument('--replay',action='store_true')
    a=p.parse_args();verify(a.lane,a.replay)
