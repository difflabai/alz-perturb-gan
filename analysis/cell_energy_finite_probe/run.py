"""Four predeclared CPU counterfactuals; never create/use fitted candidate weights."""
import copy,datetime,json,os,sys,time
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parent;ROOT=Path('/ml2/alz');B=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark');S=Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/core-comparison-20261103')
sys.path[:0]=list(map(str,[ROOT/'model',ROOT,S,B,B/'current_base_20261029/runtime/lib/python3.12/site-packages',B/'.venv312/lib/python3.12/site-packages']))
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
import numpy as np
import torch
torch.set_num_threads(1);torch.set_num_interop_threads(1)
import cell_ed_adapter as a
import cell_ed_update as u
from pg_model import critic_features
from particlegan.policy import _validate_optimizer_state
from calibration import sha

def read(p):return json.loads(Path(p).read_text())

def main():
    output=HERE/'result.json';assert not output.exists()
    protocol=read(HERE/'protocol.json');proof_path=Path(protocol['frozen_result_path'])
    assert sha(proof_path)==protocol['frozen_result_SHA256'];proof=read(proof_path)
    assert not proof['decision']['accept'] and proof['owned_release_and_clean_settlement_verified']
    cp=Path(proof['candidate']['checkpoint_path']);assert sha(cp)==proof['candidate']['checkpoint_SHA256']
    q=torch.load(cp,map_location='cpu',weights_only=False)
    assert q['completed_updates']==q['fixed_endpoint']==20000 and q['recipe']['constraint_geometry_mode']=='direction_blend'
    auth=read(ROOT/'artifacts/allocations'/proof['attempt']/'authorization.json')
    for path in (ROOT/'model/cell_ed_adapter.py',ROOT/'model/cell_ed_update.py',ROOT/'model/task_cell_energy.py'):
        assert sha(path)==auth['files_SHA256'][str(path)]
    print('Loading fitting-only CPU context; no calibration arrays or GPU.',flush=True)
    ctx=a.Context(Path(auth['cache_dir']),'cpu');assert len(ctx.ids)==33
    records=[]
    for case,seed in enumerate(protocol['case_seeds']):
        started=time.monotonic();print('Counterfactual case',case,'seed',seed,flush=True)
        bundle,g,r,usage,streams=a.build(ctx.mu,ctx.theta,'cpu',0,0)
        rec,G,D,P,og,od,pen,loss,reg,policy=bundle
        assert rec.to_dict()==q['recipe']
        for name,module in [('generator',G),('critic',D),('prior',P)]:module.load_state_dict(q['policy']['models'][name],strict=True)
        for optimizer,state in zip((og,od),q['policy']['optimizers']):
            _validate_optimizer_state(optimizer,state);optimizer.load_state_dict(copy.deepcopy(state))
        rec.restore_prior_regularizer(reg,q['prior_regularizer'].get('prior_regularizer'))
        # This is explicitly fresh CPU diagnostic entropy, NOT a conversion of
        # saved CUDA RNG state or an exact continuation of the native trajectory.
        for offset,name in enumerate(policy._STREAMS):getattr(policy,name).manual_seed(seed+10+offset)
        g['cell'].manual_seed(seed+1);g['count'].manual_seed(seed+2);torch.manual_seed(seed+3)
        r=np.random.default_rng(seed);policy.completed_steps=20000
        saved={};batches=[];draws=0;original_batch=ctx.batch;original_sample=P.sample
        def batch(rng):
            result=original_batch(rng);batches.append(result);return result
        def sample(*args,**kwargs):
            nonlocal draws
            result=original_sample(*args,**kwargs);draws+=1
            if draws==2:
                saved['idx']=result[1].detach().clone()
                saved['real'],saved['condition'],saved['library']=batches[-1]
                saved['cpu']=torch.get_rng_state().clone()
                saved['streams']={name:generator.get_state().clone() for name,generator in g.items()}
            return result
        ctx.batch=batch;P.sample=sample
        def evaluate():
            cpu=torch.get_rng_state().clone();states={name:gen.get_state().clone() for name,gen in g.items()}
            try:
                torch.set_rng_state(saved['cpu'])
                for name,state in saved['streams'].items():g[name].set_state(state)
                with torch.no_grad():
                    z=P.z[saved['idx']];c=saved['condition'];lib=saved['library']
                    rate=G.rates(z,c,128,lib,g['cell'])
                    counts=torch.poisson(rate.detach(),generator=g['count'])
                    pcs=critic_features(counts,rate,ctx.mean,ctx.sd)@ctx.W
                    adversary=loss.g_loss(D(u.bag_features(pcs),c))
                    cells=u.pooled_cell_energy(saved['real'][:,:4],pcs[:,:4],ctx.cell_energy_weights(c))
                    return torch.stack((adversary,cells)).double()
            finally:
                torch.set_rng_state(cpu)
                for name,state in states.items():g[name].set_state(state)
        original_backward=u.constraint_geometry_backward;original_step=og.step
        def backward(total,optimizer,protected_losses,**kwargs):
            saved['actual_protected_values']=torch.stack([v.detach() for v in protected_losses]).double()
            return original_backward(total,optimizer,protected_losses,**kwargs)
        def step(*args,**kwargs):
            parameters=og._parameters();old=[p.detach().clone() for p in parameters]
            normals=og._protected.detach().clone();stats=copy.deepcopy(og.direction_blend_stats)
            saved['before']=evaluate()
            torch.testing.assert_close(saved['before'],saved['actual_protected_values'],rtol=1e-6,atol=1e-6)
            # Repeated replay itself must be deterministic and consume none of
            # the actual diagnostic training stream states.
            before_cpu=torch.get_rng_state().clone();before_g={name:gen.get_state().clone() for name,gen in g.items()}
            assert torch.equal(saved['before'],evaluate())
            assert torch.equal(before_cpu,torch.get_rng_state())
            for name,state in before_g.items():assert torch.equal(state,g[name].get_state())
            result=original_step(*args,**kwargs)
            displacement=torch.cat([(p.detach()-previous).flatten() for p,previous in zip(parameters,old)])
            saved['first_order']=(normals.double()@displacement.double()).tolist()
            saved['after']=evaluate();saved['conflict']=og.direction_blend_stats['conflict_steps']-stats['conflict_steps']
            saved['blend']=og.direction_blend_stats['blended_steps']-stats['blended_steps']
            return result
        og.step=step
        try:
            with patch.object(u,'constraint_geometry_backward',side_effect=backward):
                metrics=a.advance(ctx,bundle,g,r,usage,execution_limit=20001)
        finally:ctx.batch=original_batch;P.sample=original_sample
        assert draws==2 and len(batches)==2 and metrics['step']==20001
        before=saved['before'];after=saved['after'];delta=after-before
        assert torch.isfinite(before).all() and torch.isfinite(after).all()
        record={'case':case,'fresh_CPU_diagnostic_seed':seed,'protected_order':['bag_adversary','weighted_pooled_cell_energy'],
                'before':before.tolist(),'after':after.tolist(),'finite_change':delta.tolist(),
                'first_order_derivatives_of_applied_displacement':saved['first_order'],
                'native_conflict_step':bool(saved['conflict']),'native_blended_step':bool(saved['blend']),
                'any_finite_loss_increased':bool((delta>1e-7).any()),
                'cell_finite_loss_increased':bool(delta[1]>1e-7),
                'replay_exact_and_training_stream_states_unchanged':True,'wall_seconds':time.monotonic()-started}
        records.append(record);print(json.dumps(record),flush=True)
        # The CPU shadow state is discarded, never checkpointed or used as a
        # candidate or as the starting point for the next diagnostic case.
    assert not torch.cuda.is_initialized()
    result={'status':'FOUR_FIXED_FITTING_ONLY_CPU_COUNTERFACTUAL_FINITE_STEP_DIAGNOSTICS',
            'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'records':records,
            'cases_with_any_finite_loss_increase':sum(r['any_finite_loss_increased'] for r in records),
            'cases_with_cell_finite_loss_increase':sum(r['cell_finite_loss_increased'] for r in records),
            'input_SHA256':{str(HERE/'protocol.json'):sha(HERE/'protocol.json'),str(proof_path):sha(proof_path),str(cp):sha(cp)},
            'source_SHA256':{str(Path(__file__).resolve()):sha(__file__),str(ROOT/'model/cell_ed_adapter.py'):sha(ROOT/'model/cell_ed_adapter.py'),str(ROOT/'model/cell_ed_update.py'):sha(ROOT/'model/cell_ed_update.py')},
            'CUDA_initialized':False,'native_training_continuation_claimed':False,
            'candidate_or_checkpoint_created_from_shadow_updates':False,'calibration_forecasts_generated':False,
            'scope':protocol['limits']}
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('status','cases_with_any_finite_loss_increase','cases_with_cell_finite_loss_increase')},indent=2))

if __name__=='__main__':main()
