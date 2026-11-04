"""Native BCAP/Adam preset and unchanged ALZ count update; no owner/data loader."""
import sys
from pathlib import Path
import numpy as np,torch
from contracts import B
sys.path.insert(0,str(B))
from particlegan import get_recipe,serial_autograd
from particlegan.optim.constraint_geometry import constraint_geometry_backward
from pg_model import critic_features
from critic import bag_features as original_bag_features,N_FEATURES

CONFIG={'z_dim':8,'stage_balance':False,'cell_z_dim':8,'theta_init_scale':1.0,'fourier_feature_scale':1.0}

def normalized_config(value):
    if not isinstance(value,dict) or 'z_dim' not in value or set(value)-{'z_dim','stage_balance','cell_z_dim','theta_init_scale','fourier_feature_scale'} or type(value['z_dim']) is not int or value['z_dim'] not in (2,8) or type(value.get('stage_balance',False)) is not bool or type(value.get('cell_z_dim',8)) is not int or value.get('cell_z_dim',8) not in (8,32) or type(value.get('theta_init_scale',1.0)) not in (int,float) or value.get('theta_init_scale',1.0) not in (1.,10.) or type(value.get('fourier_feature_scale',1.0)) not in (int,float) or value.get('fourier_feature_scale',1.0) not in (1.,5.):
        raise ValueError('registered architecture, sampling and feature-scale configuration required')
    return {'z_dim':value['z_dim'],'stage_balance':value.get('stage_balance',False),'cell_z_dim':value.get('cell_z_dim',8),'theta_init_scale':float(value.get('theta_init_scale',1.0)),'fourier_feature_scale':float(value.get('fourier_feature_scale',1.0))}

def configure(value):
    selected=normalized_config(value)
    if selected!={'z_dim':8,'stage_balance':False,'cell_z_dim':8,'theta_init_scale':1.,'fourier_feature_scale':1.}:
        raise ValueError('native BCAP/Adam trial requires matched PG8 architecture')
    CONFIG.clear();CONFIG.update(selected)

def bag_features(pcs):
    features=original_bag_features(pcs)
    scale=CONFIG['fourier_feature_scale']
    if scale==1.:return features
    return torch.cat([features[:,:100],features[:,100:]*scale],dim=1)

def recipe():
    # Preserve native defaults apart from registered donor dimension and fixed task limits.
    return get_recipe('bcap_adam',total_steps=20000,batch_size=32,num_particles=512,z_dim=CONFIG['z_dim'])

def finite(module,label,grads=False):
    bad=[n for n,p in module.named_parameters() if (p.grad is not None if grads else True)
         and not bool(torch.isfinite(p.grad if grads else p).all())]
    if bad:raise FloatingPointError(label+': '+repr(bad))

def gradnorm(module):
    return float(sum(p.grad.detach().square().sum() for p in module.parameters() if p.grad is not None).sqrt())

def advance(ctx,bundle,generators,rng,usage,*,execution_limit):
    """One unchanged canonical donor-bag update; shared with CPU parity fixture."""
    rec,G,D,P,og,od,pen,loss,reg,policy=bundle
    step=policy.completed_steps
    def fake(z,c,lib):
        rate=G.rates(z,c,128,lib,generators['cell'])
        counts=torch.poisson(rate.detach(),generator=generators['count'])
        return bag_features(critic_features(counts,rate,ctx.mean,ctx.sd)@ctx.W)
    with serial_autograd():
        real,c,lib=ctx.batch(rng)
        assert real.shape==(32,N_FEATURES)
        noise=policy.begin_step(real,execution_limit=execution_limit)
        assert noise.input_sigma==0 and float(noise.output_sigma)==0
        assert [x['lr'] for x in og.param_groups]==[rec.lr,rec.lr*rec.prior_lr_mult]
        assert [x['lr'] for x in od.param_groups]==[rec.lr*rec.d_lr_mult]
        D.train();G.eval()
        with torch.no_grad():z,_=P.sample(32,generator=policy.latent_generator);f=fake(z,c,lib)
        assert f.shape==real.shape
        policy.observe_critic_pair(real,f)
        dr,df=D(real,c),D(f,c);adv=loss.d_loss(dr,df);pen.collect_stats=step%250==0
        penalty=pen(D,real,f,c);ld=adv+penalty
        od.zero_grad();ld.backward();finite(D,'D gradients',True)
        dn=gradnorm(D) if step%250==0 else None
        od.step();finite(D,'D parameters');policy.after_critic_step()
        D.eval();G.train();D.requires_grad_(False)
        real2,c2,lib2=ctx.batch(rng);z,idx=P.sample(32,generator=policy.latent_generator)
        f=fake(z,c2,lib2);lg=loss.g_loss(D(f,c2),D(real2,c2).detach())
        raw=reg(P.z[torch.unique(idx)]);total=lg+rec.prior_reg*raw
        og.zero_grad();policy.observe_sampled_rows(idx)
        constraint_geometry_backward(total,og,(lg,))
        finite(G,'G gradients',True);finite(P,'prior gradients',True)
        if step==0:assert all(p.grad is not None and bool(p.grad.abs().sum()>0) for p in [G.out[-1].weight,G.log_theta,P.z])
        gn=gradnorm(G) if step%250==0 else None;pn=gradnorm(P) if step%250==0 else None
        policy.after_generator_backward(loss_gan=lg.detach(),loss_critic=adv.detach())
        before_prior=P.z.detach().clone()
        og.step();finite(G,'G parameters');finite(P,'prior parameters')
        policy.after_generator_step();D.requires_grad_(True);policy.finish_step()
        usage.observe(P.z,before_prior,idx)
    assert policy.completed_steps==step+1
    return {'step':policy.completed_steps,'D_adversarial':float(adv.detach()),'D_penalty':float(penalty.detach()),'D_total':float(ld.detach()),
            'G_adversarial':float(lg.detach()),'prior_regularization_unweighted':float(raw.detach()),'G_total':float(total.detach()),
            'G_prior_LRs':[g['lr'] for g in og.param_groups],'D_LRs':[g['lr'] for g in od.param_groups],
            'critic_gap':float((dr-df).mean().detach()),'D_gradient_norm':dn,'G_gradient_norm':gn,'prior_gradient_norm':pn,
            'penalty_stats':pen.last_stats,'penalty_diagnostics':pen.diagnostics()}
