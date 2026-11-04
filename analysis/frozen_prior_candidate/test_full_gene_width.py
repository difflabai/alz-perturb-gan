"""Full2000-gene synthetic first-update parity, without GPU/data/quality claims."""
import copy,datetime,json,os,sys
from pathlib import Path
import numpy as np
import torch
import frozen_pg_adapter as a
import frozen_pg_update as u
import variant_pg_adapter as base
import variant_pg_update as base_u
from test_variant import same
from contracts import sha
ROOT=Path('/ml2/alz')
CFG={'z_dim':8,'stage_balance':False,'cell_z_dim':8,'theta_init_scale':1.,'fourier_feature_scale':1.}

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    u.configure(CFG);base_u.configure(CFG)
    x=torch.randint(1,8,(24,2000),generator=torch.Generator().manual_seed(2026110421)).float()
    ctx=object.__new__(a.Context);ctx.device=torch.device('cpu');ctx.fold=0
    ctx.ids=np.arange(6);ctx.by=[np.arange(4*i,4*i+4) for i in ctx.ids];ctx.X=x
    ctx.conditions=torch.zeros(6,7);ctx.conditions[:,3]=torch.tensor([0.,1/3,2/3,1.,1.,1.]);ctx.fitc=ctx.conditions.numpy().copy()
    ctx.mean=torch.zeros(2000);ctx.sd=torch.ones(2000);ctx.W=torch.randn(2000,50,generator=torch.Generator().manual_seed(2026110422))/np.sqrt(2000)
    ctx.trainpcs=ctx.project(x);ctx.mu=x.mean(0);ctx.theta=(ctx.mu.square()/(x.var(0,unbiased=False)-ctx.mu).clamp_min(1e-4)).clamp(.1,10000)
    b,g,r,usage,seeds=a.build(ctx.mu,ctx.theta,'cpu',0,0);initial_rng=torch.get_rng_state().clone();initial_z=b[3].z.clone()
    initial_models=copy.deepcopy([b[i].state_dict() for i in (1,2)])
    rb,rg,rr,ru,rs=base.build(ctx.mu,ctx.theta,'cpu',0,0);same(initial_rng,torch.get_rng_state());same(seeds,rs);same(initial_z,rb[3].z)
    torch.set_rng_state(initial_rng);expected=base.advance(ctx,rb,rg,rr,ru,execution_limit=20000)
    torch.set_rng_state(initial_rng);actual=a.advance(ctx,b,g,r,usage,execution_limit=20000)
    for key in ('D_adversarial','D_penalty','D_total','G_adversarial','D_gradient_norm','G_gradient_norm'):same(actual[key],expected[key])
    for i in (1,2):same(b[i].state_dict(),rb[i].state_dict())
    for name in g:same(g[name].get_state(),rg[name].get_state())
    same(r.bit_generator.state,rr.bit_generator.state);same(initial_z,b[3].z);assert not torch.equal(initial_z,rb[3].z)
    floats=0
    for module in (b[1],b[2],b[3]):
        for v in module.state_dict().values():
            if isinstance(v,torch.Tensor) and v.is_floating_point():assert torch.isfinite(v).all();floats+=v.numel()
    assert floats==371105
    assert b[1].out[-1].weight.shape[0]==2000 and b[1].log_theta.numel()==2000
    assert b[1].log_theta.grad is not None and torch.isfinite(b[1].log_theta.grad).all() and b[1].log_theta.grad.abs().sum()>0
    for module,initial in zip((b[1],b[2]),initial_models):assert any(not torch.equal(v,module.state_dict()[k]) for k,v in initial.items())
    assert not torch.cuda.is_initialized()
    result={'status':'PASSED_FULL_2000_GENE_SYNTHETIC_NATIVE_FROZEN_CONTROL_FIRST_UPDATE_PARITY','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'full_model_floating_values':floats,'first_G_D_models_and_objectives_bitwise_match_learned_PG8':True,'initial_models_prior_and_all_streams_match':True,'fixed_prior_unchanged_and_learned_prior_moves':True,'G_D_dispersion_train_and_all_model_floats_finite':True,'CUDA_initialized':False,'real_data_loaded':False,'quality_measured':False,'source_SHA256':{str(p):sha(p) for p in [Path(__file__).resolve(),ROOT/'analysis/frozen_prior_candidate/frozen_pg_adapter.py',ROOT/'analysis/frozen_prior_candidate/frozen_pg_update.py',ROOT/'model/variant_pg_adapter.py',ROOT/'model/variant_pg_update.py',ROOT/'test_variant.py']},'scope':'One synthetic full-width first native optimizer pair, actual2000-gene CountGenerator/150featurecritic/512x8 fixedbank. Not data-dependent gradients, repeated fullwidth replay, nativeGPU training/serving/quality, final model relocation or runtime qualification.'}
    dest=ROOT/'analysis/frozen_prior_control_full_gene_width_CPU_qualification.json';assert not dest.exists();dest.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
