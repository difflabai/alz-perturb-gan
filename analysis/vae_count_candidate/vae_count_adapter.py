"""CPU feasibility of the public VAE-GAN family on invariant count statistics.

This module supplies no admitted real-data fitting path. The reconstruction
likelihood is Gaussian on the existing150bag features, not a cell-count NLL.
"""
import copy
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from particlegan import UpdatePolicy,get_recipe,serial_autograd
from particlegan.optim.constraint_geometry import constraint_geometry_backward
from contracts import B,PG_UPDATES,SEEDS,sha
sys.path.insert(0,str(B))
from pg_model import CountGenerator,critic_features
from critic import VectorCritic,bag_features,N_FEATURES

CONFIG={'native_family':'vae_gan','trainer':'bispf_optimized','z_dim':8,'cell_z_dim':8,
        'posterior_draws':2,'reconstruction_space':'original_invariant150bagfeatures',
        'reconstruction_likelihood':'native_Gaussian_negative_ELBO','observation_sigma':.03,
        'encoder_hidden':[128,128],'encoder_activation':'SiLU',
        'protected_objectives':['GAN','weighted_reconstruction'],
        'private_reconstruction_Gamma':True}
STREAM_NAMES=['generator_init','critic_init','prior_init','policy_latent','policy_penalty',
              'policy_eval','policy_noise','cell','count','gamma_global','donor_sampler','prior_noise',
              'encoder_init','posterior','reconstruction_cell','reconstruction_count','reconstruction_gamma']
GENERATOR_NAMES=('cell','count','prior_noise','posterior','reconstruction_cell','reconstruction_count','reconstruction_gamma')


def training_streams(fold,seed):
    if type(fold) is not int or fold!=0 or type(seed) is not int or seed not in SEEDS:
        raise ValueError('Registered first-fold training seed required')
    children=np.random.SeedSequence([20261103,fold,seed]).spawn(len(STREAM_NAMES))
    return {name:int(c.generate_state(1,dtype=np.uint64)[0])%(2**63-16) for name,c in zip(STREAM_NAMES,children)}


class BagEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(157,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,8))

    def forward(self,features,condition):
        if features.ndim!=2 or features.shape[1]!=150 or condition.shape!=(len(features),7):
            raise ValueError('Encoder requires150invariant bag features and7condition slots')
        return self.net(torch.cat([features,condition],dim=1))


class Usage:
    def __init__(self,table):
        self.steps=0
        for name in ('draws','selected_steps','moved_steps','unsampled_moved_steps'):
            setattr(self,name,torch.zeros(len(table),dtype=torch.long,device=table.device))
        self.unique_batch_histogram=[0]*(len(table)+1)

    @torch.no_grad()
    def observe(self,table,before,indices):
        self.draws.index_add_(0,indices,torch.ones_like(indices,dtype=torch.long))
        rows=torch.unique(indices);selected=torch.zeros_like(self.draws,dtype=torch.bool);selected[rows]=True
        moved=(table.detach()!=before).any(-1)
        self.selected_steps+=selected.long();self.moved_steps+=moved.long()
        self.unsampled_moved_steps+=(moved&~selected).long();self.steps+=1
        self.unique_batch_histogram[len(rows)]+=1

    def state_dict(self):
        return {**{name:getattr(self,name).cpu().clone() for name in ('draws','selected_steps','moved_steps','unsampled_moved_steps')},
                'steps':self.steps,'unique_batch_histogram':list(self.unique_batch_histogram)}


@dataclass
class Runtime:
    recipe: object
    G: object
    D: object
    P: object
    E: object
    og: object
    od: object
    penalty: object
    loss: object
    regularizer: object
    policy: object
    generators: dict
    rng: object
    usage: object
    streams: dict
    seed: int
    reconstruction_draws_seen: int=0


def build(ctx,seed):
    streams=training_streams(ctx.fold,seed)
    rec=get_recipe('vae_gan',trainer='bispf_optimized',total_steps=PG_UPDATES,batch_size=32,num_particles=512,z_dim=8)
    if not (rec.encoder_mode=='hard' and rec.prior_kind=='mog' and rec.sigma_rel==.025 and rec.standardize
            and rec.observation_sigma==.03 and rec.reconstruction_weight==1.
            and rec.input_noise_std==rec.output_noise_std==rec.ema_decay==0
            and not rec.row_evidence_gate and not rec.particle_birth_death):
        raise ValueError('Native VAE-GAN/BISPF recipe contract changed')
    torch.manual_seed(streams['generator_init']);G=CountGenerator(ctx.mu.cpu(),ctx.theta.cpu(),zd=8,zc=8).to(ctx.device)
    torch.manual_seed(streams['critic_init']);D=VectorCritic().to(ctx.device)
    P=rec.make_prior(generator=torch.Generator().manual_seed(streams['prior_init'])).to(ctx.device)
    torch.manual_seed(streams['encoder_init']);E=BagEncoder().to(ctx.device)
    og,od=rec.make_optimizers(G,D,P,encoder=E);penalty=rec.make_critic_penalty(od)
    policy=UpdatePolicy(rec,G,D,prior=P,encoder=E,generator_optimizer=og,critic_optimizer=od,penalty=penalty,
                        row_semantics='conditional',seed=seed,
                        streams={name:torch.Generator(device=ctx.device).manual_seed(streams['policy_'+key])
                                 for name,key in [('latent_generator','latent'),('penalty_generator','penalty'),
                                                  ('eval_generator','eval'),('noise_generator','noise')]})
    generators={name:torch.Generator(device=ctx.device).manual_seed(streams[name]) for name in GENERATOR_NAMES}
    torch.manual_seed(streams['gamma_global'])
    return Runtime(rec,G,D,P,E,og,od,penalty,rec.make_loss(),rec.make_prior_regularizer(prior=P,weight=1.),policy,
                   generators,np.random.default_rng(streams['donor_sampler']),Usage(P.z),streams,seed)


def count_features(ctx,runtime,z,condition,libraries,*,reconstruction=False):
    if z.shape!=(len(condition),8) or condition.shape!=(len(z),7) or libraries.shape!=(len(z),128):
        raise ValueError('Qualified8D/7condition/128cell count decoder shapes required')
    g=runtime.generators
    if reconstruction:
        # Isolate implicit Gamma.rsample RNG without changing its native law.
        private=g['reconstruction_gamma']
        with torch.random.fork_rng(devices=[ctx.device.index] if ctx.device.type=='cuda' else []):
            if ctx.device.type=='cuda':torch.cuda.set_rng_state(private.get_state(),ctx.device)
            else:torch.set_rng_state(private.get_state())
            rate=runtime.G.rates(z,condition,128,libraries,g['reconstruction_cell'])
            private.set_state(torch.cuda.get_rng_state(ctx.device) if ctx.device.type=='cuda' else torch.get_rng_state())
        counts=torch.poisson(rate.detach(),generator=g['reconstruction_count'])
    else:
        rate=runtime.G.rates(z,condition,128,libraries,g['cell'])
        counts=torch.poisson(rate.detach(),generator=g['count'])
    if not bool(torch.isfinite(counts).all() and (counts>=0).all() and (counts==counts.floor()).all()):
        raise FloatingPointError('Direct Gamma-Poisson integer count validity failed')
    return bag_features(critic_features(counts,rate,ctx.mean,ctx.sd)@ctx.W)


def finite(module,label,*,gradients=False):
    for name,p in module.named_parameters():
        value=p.grad if gradients else p
        if value is not None and not bool(torch.isfinite(value).all()):raise FloatingPointError(label+': '+name)


def advance(ctx,runtime):
    rec,G,D,P,E,og,od=runtime.recipe,runtime.G,runtime.D,runtime.P,runtime.E,runtime.og,runtime.od
    policy=runtime.policy;step=policy.completed_steps
    with serial_autograd():
        real,c,lib=ctx.batch(runtime.rng)
        if real.shape!=(32,N_FEATURES):raise ValueError('Original32bag150feature batch required')
        noise=policy.begin_step(real,execution_limit=PG_UPDATES)
        if noise.input_sigma!=0 or float(noise.output_sigma)!=0:raise ValueError('Native zero training noise changed')
        D.train();G.eval();E.eval()
        with torch.no_grad():
            z,_=P.sample(32,generator=policy.latent_generator,noise_generator=runtime.generators['prior_noise'])
            fake=count_features(ctx,runtime,z,c,lib)
        policy.observe_critic_pair(real,fake)
        adv=runtime.loss.d_loss(D(real,c),D(fake,c));penalty=runtime.penalty(D,real,fake,c)
        ld=adv+penalty;od.zero_grad();ld.backward();finite(D,'D gradients',gradients=True)
        od.step();finite(D,'D parameters');policy.after_critic_step()
        D.eval();D.requires_grad_(False);G.train();E.train()
        try:
            real2,c2,lib2=ctx.batch(runtime.rng)
            z,indices=P.sample(32,generator=policy.latent_generator,noise_generator=runtime.generators['prior_noise'])
            fake=count_features(ctx,runtime,z,c2,lib2);lg=runtime.loss.g_loss(D(fake,c2))
            encoding=rec.encode(E(real2,c2),P,draws=2,generator=runtime.generators['posterior'])
            prediction=count_features(ctx,runtime,encoding.codes.reshape(64,8),c2.repeat_interleave(2,dim=0),
                                      lib2.repeat_interleave(2,dim=0),reconstruction=True).reshape(32,2,150)
            reconstruction=encoding.negative_elbo(prediction,real2,observation_sigma=rec.observation_sigma)
            observed=torch.cat([indices,encoding.indices.reshape(-1)])
            raw=runtime.regularizer(P.z[torch.unique(observed)])
            total=lg+rec.reconstruction_weight*reconstruction+rec.prior_reg*raw
            og.zero_grad();policy.observe_sampled_rows(observed)
            constraint_geometry_backward(total,og,(lg,rec.reconstruction_weight*reconstruction))
            for label,module in [('G',G),('E',E),('prior',P)]:finite(module,label+' gradients',gradients=True)
            if step==0:
                for p in (G.out[-1].weight,G.log_theta,E.net[-1].weight,P.z):
                    if p.grad is None or not bool(p.grad.abs().sum()>0):raise FloatingPointError('Initial G/dispersion/E/P gradient missing')
            gradient_sums={name:float(sum(p.grad.abs().sum() for p in module.parameters() if p.grad is not None))
                           for name,module in [('G',G),('encoder',E),('prior',P)]}
            policy.after_generator_backward(loss_gan=lg.detach(),loss_critic=adv.detach())
            before=P.z.detach().clone();og.step()
            for label,module in [('G',G),('E',E),('prior',P)]:finite(module,label+' parameters')
            policy.after_generator_step()
        finally:D.requires_grad_(True)
        policy.finish_step();runtime.usage.observe(P.z,before,observed);runtime.reconstruction_draws_seen+=64
    if policy.completed_steps!=step+1:raise ValueError('Native optimizer-pair clock differs')
    return {'step':policy.completed_steps,'D_adversarial':float(adv.detach()),'D_penalty':float(penalty.detach()),
            'D_total':float(ld.detach()),'G_adversarial':float(lg.detach()),
            'reconstruction_Gaussian_negative_ELBO':float(reconstruction.detach()),
            'posterior_joint_KL':float(encoding.kl.mean()),'prior_regularization_unweighted':float(raw.detach()),
            'G_total':float(total.detach()),'gradient_absolute_sums':gradient_sums,
            'G_encoder_prior_LRs':[g['lr'] for g in og.param_groups],'D_LRs':[g['lr'] for g in od.param_groups],
            'native_integer_count_draws_validated':True,'prior_draws_observed':len(observed)}


def snapshot(ctx,runtime,status):
    if runtime.policy._phase!='ready':raise ValueError('Only complete optimizer pairs may be checkpointed')
    return copy.deepcopy({'schema':1,'native_family':'vae_gan','method':'pg','fold':ctx.fold,'seed':runtime.seed,
                          'adapter_config':CONFIG,'recipe':runtime.recipe.to_dict(),'policy':runtime.policy.state_dict(),
                          'cache_manifest_SHA256':sha(Path(ctx.cache)/'manifest.json'),
                          'prior_regularizer':runtime.recipe.prior_regularizer_checkpoint(runtime.regularizer),
                          'generators':{name:g.get_state() for name,g in runtime.generators.items()},
                          'numpy_rng':runtime.rng.bit_generator.state,'usage':runtime.usage.state_dict(),
                          'training_streams':runtime.streams,'reconstruction_draws_seen':runtime.reconstruction_draws_seen,
                          'completed_updates':runtime.policy.completed_steps,'fixed_endpoint':PG_UPDATES,'status':status,
                          'initial_kernel':{'sigma':float(runtime.P.sigma),'d0':float(runtime.P.d0)}})


def restore(ctx,q):
    if (q.get('schema')!=1 or q.get('native_family')!='vae_gan' or q.get('adapter_config')!=CONFIG
            or q.get('method')!='pg' or q.get('fold')!=ctx.fold or q.get('fixed_endpoint')!=PG_UPDATES
            or q.get('cache_manifest_SHA256')!=sha(Path(ctx.cache)/'manifest.json')):
        raise ValueError('VAE family/configuration/cache/endpoint differs')
    if set(q.get('generators',{}))!=set(GENERATOR_NAMES):raise ValueError('Complete seven private generator states required')
    runtime=build(ctx,q['seed'])
    if q['recipe']!=runtime.recipe.to_dict() or q['training_streams']!=runtime.streams:
        raise ValueError('Resolved native VAE recipe or registered streams differ')
    if q['initial_kernel']!={'sigma':float(runtime.P.sigma),'d0':float(runtime.P.d0)}:
        raise ValueError('Original native fixed MoG kernel differs')
    runtime.policy.load_state_dict(q['policy'])
    runtime.recipe.restore_prior_regularizer(runtime.regularizer,q['prior_regularizer'].get('prior_regularizer'))
    for name,value in q['generators'].items():runtime.generators[name].set_state(value.cpu())
    runtime.rng.bit_generator.state=q['numpy_rng']
    for name in ('draws','selected_steps','moved_steps','unsampled_moved_steps'):
        value=q['usage'][name]
        if value.shape!=(512,) or value.dtype!=torch.long or not bool((value>=0).all()):raise ValueError('Invalid passive row counter')
        setattr(runtime.usage,name,value.to(ctx.device).clone())
    steps=runtime.policy.completed_steps;usage=q['usage']
    if (type(q['completed_updates']) is not int or q['completed_updates']!=steps or not 0<=steps<=PG_UPDATES
            or usage['steps']!=steps or int(usage['draws'].sum())!=96*steps
            or len(usage['unique_batch_histogram'])!=513 or sum(usage['unique_batch_histogram'])!=steps
            or q['reconstruction_draws_seen']!=64*steps):raise ValueError('VAE reconstruction/row/optimizer clocks differ')
    runtime.usage.steps=steps;runtime.usage.unique_batch_histogram=list(usage['unique_batch_histogram'])
    runtime.reconstruction_draws_seen=q['reconstruction_draws_seen']
    return runtime


def fit(*args,**kwargs):
    raise RuntimeError('CPU feasibility only; no separately qualified native count-VAE fitting runtime exists')
