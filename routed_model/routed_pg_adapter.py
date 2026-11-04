"""Fitting-only loader and complete native routed-E22 lifecycle/checkpoints."""
import copy
from dataclasses import dataclass
from pathlib import Path
import signal
import sys
import time

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from particlegan import E22Policy, RoutedRows, RoutedBatch, get_recipe, serial_autograd
from particlegan.optim.constraint_geometry import constraint_geometry_backward
from contracts import B, PG_UPDATES, SEEDS, checked_cache, sha, dump, validate_train_donors
sys.path.insert(0,str(B))
from pg_model import CountGenerator, standardize_counts
from bench_core import load_space
from critic import VectorCritic
from stage_law import StageLaw, mask_adnc
from partition import partition
from routed_counts import CountForward, CONTEXT_CONFIG, make_context, make_guard_context

CONFIG={'z_dim':8,'stage_balance':False,'cell_z_dim':8,'theta_init_scale':1.,'fourier_feature_scale':1.}
ROUTED_CONFIG={'native_recipe':'e22_routed','row_semantics':'dense_soft','probe_interval':20,
               'protected_fraction':'fixed_per_donor_hash_rank_floor_one_fifth',
               'context':CONTEXT_CONFIG,'output_noise_coordinates':'fake_150D_critic_observation_only',
               'paired_fit_contexts':'same_fit_context_and_targets_for_D_and_G_substeps'}
STREAM_NAMES=['generator_init','critic_init','prior_init','policy_latent','policy_penalty',
              'policy_eval','policy_noise','cell','count','gamma_global','donor_sampler','guard_sampler']


def training_streams(fold,seed):
    if type(fold) is not int or fold!=0 or type(seed) is not int or seed not in SEEDS:
        raise ValueError('First-fold registered training seed required')
    children=np.random.SeedSequence([20261103,fold,seed]).spawn(len(STREAM_NAMES))
    return {k:int(c.generate_state(1,dtype=np.uint64)[0])%(2**63-16) for k,c in zip(STREAM_NAMES,children)}


def fitting_statistics(X,split):
    ids=torch.as_tensor(np.sort(split.fitting),device=X.device)
    fitting=X[ids]
    mu=fitting.mean(0);variance=fitting.var(0,unbiased=False)
    theta=(mu.square()/(variance-mu).clamp_min(1e-4)).clamp(.1,10000)
    return mu,theta


class Context:
    def __init__(self,cache,device):
        manifest,f=checked_cache(cache,'train');cache=Path(cache)
        if manifest['fold']!=0:raise ValueError('First-fold routed count adapter only')
        self.cache,self.manifest,self.fold,self.device=cache,manifest,0,torch.device(device)
        with np.load(cache/'train/metadata.npz',allow_pickle=False) as m:
            self.fit_labels=m['donor'].copy();self.cell_names=m['cell'].copy()
            self.ids=m['donors'].copy();self.fitc=m['raw_c'].astype(np.float32)
        validate_train_donors(self.fit_labels,f['optimization_donors'])
        if self.ids.tolist()!=f['optimization_donors']:raise ValueError('Ordered fitting donors changed')
        self.partition=partition(self.fit_labels,self.cell_names,self.ids)
        counts=sp.load_npz(cache/'train/counts_hvg.npz').astype(np.float32)
        if (counts.shape!=(len(self.fit_labels),2000) or not np.isfinite(counts.data).all()
                or (counts.data<0).any() or not np.equal(counts.data,np.floor(counts.data)).all()):
            raise ValueError('Fitting integer count shape/range invalid')
        self.X=torch.as_tensor(counts.toarray(),device=self.device)
        self.all_by=[np.flatnonzero(self.fit_labels==d) for d in self.ids]
        self.by=self.partition.fitting_by_donor;self.guard_by=self.partition.protected_by_donor
        self.conditions=torch.as_tensor(mask_adnc(self.fitc),device=self.device)
        self.space=load_space(cache)
        self.mean=torch.as_tensor(self.space.mean,dtype=torch.float32,device=self.device)
        self.sd=torch.as_tensor(self.space.sd,dtype=torch.float32,device=self.device)
        self.W=torch.as_tensor(self.space.W/self.space.pc_sd,dtype=torch.float32,device=self.device)
        self.fitlibs=self.X.sum(-1)
        if not bool((self.fitlibs>0).all()):raise ValueError('Common zero-HVG exclusion changed')
        self.mu,self.theta=fitting_statistics(self.X,self.partition)
        # Preserve all original fitting-source identities/exposures for the
        # already registered calibration trace. Only optimizer batches and
        # initial count moments use the smaller fitting-cell pool.
        self.law=StageLaw(self.fitc,self.all_by,self.fitlibs.cpu().numpy())

    def project(self,x):return standardize_counts(x,self.mean,self.sd)@self.W

    def draw(self,role,rng):
        if role not in ('fitting','protected'):raise ValueError('Explicit fitting or protected pool required')
        pools=self.by if role=='fitting' else self.guard_by
        donors=rng.integers(len(self.ids),size=32)
        rows=np.stack([rng.choice(pools[i],128,replace=True) for i in donors])
        index=torch.as_tensor(rows,device=self.device)
        counts=self.X[index]
        condition=self.conditions[torch.as_tensor(donors,device=self.device)]
        return counts,condition,counts.sum(-1),rows


class Router(nn.Module):
    def __init__(self,n,device):
        super().__init__();self.register_buffer('log_mass',torch.zeros(n,device=device))


@dataclass
class Runtime:
    recipe: object
    policy: object
    forward: object
    loss: object
    regularizer: object
    fitting_rng: object
    protected_rng: object
    seed: int
    streams: dict
    fitting_contexts_seen: int=0
    protected_contexts_seen: int=0


def build(ctx,seed):
    streams=training_streams(ctx.fold,seed)
    rec=get_recipe('e22_routed',total_steps=None,batch_size=32,num_particles=512,z_dim=8)
    if (rec.name!='e22_routed' or rec.row_policy!='routed_paired' or rec.continuous_policy!='dv12'
            or not rec.row_evidence_gate or not rec.particle_birth_death
            or rec.input_noise_std!=0 or rec.prior_reg!=0 or rec.prior_l2!=0
            or rec.kinetic_transport_weight!=0 or rec.kinetic_transport_local_weight!=0):
        raise ValueError('Qualified native routed E22 recipe mechanisms changed')
    torch.manual_seed(streams['generator_init'])
    G=CountGenerator(ctx.mu.cpu(),ctx.theta.cpu(),zd=8,zc=8).to(ctx.device)
    torch.manual_seed(streams['critic_init']);D=VectorCritic().to(ctx.device)
    P=rec.make_prior(generator=torch.Generator().manual_seed(streams['prior_init'])).to(ctx.device)
    router=Router(512,ctx.device)
    forward=CountForward(ctx.mean,ctx.sd,ctx.W,temperature=rec.routing_temperature)
    rows=RoutedRows(route=forward.route,generate=forward.generate,features=forward.features,probe_interval=20)
    og,od=rec.make_optimizers(G,D,P,ema_critic=copy.deepcopy(D));pen=rec.make_critic_penalty(od)
    policy=E22Policy(rec,G,D,prior=P,router=router,routed_rows=rows,generator_optimizer=og,
                     critic_optimizer=od,penalty=pen,row_semantics='dense_soft',seed=seed,
                     streams={name:torch.Generator(device=ctx.device).manual_seed(streams['policy_'+key])
                              for name,key in [('latent_generator','latent'),('penalty_generator','penalty'),
                                               ('eval_generator','eval'),('noise_generator','noise')]})
    return Runtime(rec,policy,forward,rec.make_loss(),rec.make_prior_regularizer(prior=P,weight=1.),
                   np.random.default_rng(streams['donor_sampler']),np.random.default_rng(streams['guard_sampler']),seed,streams)


def training_batch(ctx,runtime):
    counts,c,lib,_=ctx.draw('fitting',runtime.fitting_rng)
    protected,gc,gl,_=ctx.draw('protected',runtime.protected_rng)
    fc=make_context(c,lib,generator=runtime.policy.latent_generator)
    guard=make_guard_context(gc,gl,rng=runtime.protected_rng)
    runtime.fitting_contexts_seen+=len(c);runtime.protected_contexts_seen+=len(gc)
    return RoutedBatch(fc,runtime.forward.targets(counts),guard,runtime.forward.targets(protected))


def finite(module,label,*,gradients=False):
    for name,p in module.named_parameters():
        value=p.grad if gradients else p
        if value is not None and not bool(torch.isfinite(value).all()):
            raise FloatingPointError(label+': '+name)


def advance(ctx,runtime):
    rec,policy,forward=runtime.recipe,runtime.policy,runtime.forward
    G,D,P,og,od=policy.G,policy.D,policy.prior,policy.opt_g,policy.opt_d
    with serial_autograd():
        batch=training_batch(ctx,runtime);c=batch.context[:,8:15]
        real=forward.observation(batch.targets)
        noise=policy.begin_step(real,routed=batch,execution_limit=PG_UPDATES)
        if noise.input_sigma!=0:raise ValueError('Native zero input noise changed')
        D.train();G.eval()
        with torch.no_grad():
            fake=forward.observation(policy.routed_generate(batch.context,sigma=0))
            fake=fake+noise.output_sigma*torch.randn(fake.shape,device=ctx.device,generator=policy.noise_generator)
        policy.observe_critic_pair(real,fake)
        adv=runtime.loss.d_loss(D(real,c),D(fake,c))
        penalty=policy.penalty(D,real,fake,c);ld=adv+penalty
        od.zero_grad();ld.backward();finite(D,'critic gradients',gradients=True)
        od.step();finite(D,'critic parameters');policy.after_critic_step()
        D.eval();G.train();D.requires_grad_(False)
        try:
            fake=forward.observation(policy.routed_generate(batch.context,sigma=0))
            fake=fake+noise.output_sigma*torch.randn(fake.shape,device=ctx.device,generator=policy.noise_generator)
            lg=runtime.loss.g_loss(D(fake,c),D(real,c))
            raw=runtime.regularizer(P.z);total=lg+rec.prior_reg*raw
            og.zero_grad();constraint_geometry_backward(total,og,(lg,))
            finite(G,'generator gradients',gradients=True);finite(P,'table gradients',gradients=True)
            if policy.log_output_sigma.grad is None or not bool(torch.isfinite(policy.log_output_sigma.grad)):
                raise FloatingPointError('Native learned noise gradient missing or nonfinite')
            if policy.completed_steps==0:
                for p in (G.out[-1].weight,G.log_theta,P.z,policy.log_output_sigma):
                    if p.grad is None or not bool(p.grad.abs().sum()>0):raise FloatingPointError('Initial native gradient missing')
            policy.after_generator_backward(loss_gan=lg.detach(),loss_critic=adv.detach())
            og.step();finite(G,'generator parameters');finite(P,'table parameters')
            if not bool(torch.isfinite(policy.log_output_sigma)):raise FloatingPointError('Native noise parameter nonfinite')
            policy.after_generator_step()
        finally:D.requires_grad_(True)
        policy.finish_step()
        finite(G,'generator after native row control');finite(P,'table after native row control')
    return {'step':policy.completed_steps,'D_adversarial':float(adv.detach()),'D_penalty':float(penalty.detach()),
            'D_total':float(ld.detach()),'G_adversarial':float(lg.detach()),'G_total':float(total.detach()),
            'prior_regularization_unweighted':float(raw.detach()),'native_output_sigma':float(policy.output_sigma()),
            'G_prior_noise_LRs':[g['lr'] for g in og.param_groups],'D_LRs':[g['lr'] for g in od.param_groups],
            'routing_probe_clock':dict(policy.routed_control.probe_clock),
            'routing_counters':dict(policy.routed_control.counters)}


def snapshot(ctx,runtime,status):
    if runtime.policy._phase!='ready':raise ValueError('No checkpoint at partial native optimizer-pair boundary')
    return copy.deepcopy({'schema':1,'method':'pg','fold':ctx.fold,'seed':runtime.seed,
                          'native_recipe':'e22_routed','recipe':runtime.recipe.to_dict(),
                          'policy':runtime.policy.state_dict(),'prior_regularizer':runtime.recipe.prior_regularizer_checkpoint(runtime.regularizer),
                          'variant_config':CONFIG,'routed_config':ROUTED_CONFIG,
                          'cache_manifest_SHA256':sha(ctx.cache/'manifest.json'),
                          'partition':ctx.partition.summary(),'training_streams':runtime.streams,
                          'fitting_rng':runtime.fitting_rng.bit_generator.state,'protected_rng':runtime.protected_rng.bit_generator.state,
                          'fitting_contexts_seen':runtime.fitting_contexts_seen,'protected_contexts_seen':runtime.protected_contexts_seen,
                          'completed_updates':runtime.policy.completed_steps,'fixed_endpoint':PG_UPDATES,'status':status})


def restore(ctx,q):
    if (q['schema']!=1 or q['method']!='pg' or q['fold']!=ctx.fold or q['fixed_endpoint']!=PG_UPDATES
            or q['native_recipe']!='e22_routed' or q['variant_config']!=CONFIG or q['routed_config']!=ROUTED_CONFIG
            or q['cache_manifest_SHA256']!=sha(ctx.cache/'manifest.json') or q['partition']!=ctx.partition.summary()):
        raise ValueError('Checkpoint cache/partition/configuration/fixed endpoint differs')
    runtime=build(ctx,q['seed'])
    if q['recipe']!=runtime.recipe.to_dict() or q['training_streams']!=runtime.streams:
        raise ValueError('Native recipe or seed streams differ')
    runtime.policy.load_state_dict(q['policy'])
    runtime.recipe.restore_prior_regularizer(runtime.regularizer,q['prior_regularizer'].get('prior_regularizer'))
    runtime.fitting_rng.bit_generator.state=q['fitting_rng'];runtime.protected_rng.bit_generator.state=q['protected_rng']
    runtime.fitting_contexts_seen=q['fitting_contexts_seen'];runtime.protected_contexts_seen=q['protected_contexts_seen']
    if q['completed_updates']!=runtime.policy.completed_steps:raise ValueError('Completed optimizer-pair clock differs')
    if (not 0<=q['completed_updates']<=PG_UPDATES or q['fitting_contexts_seen']!=32*q['completed_updates']
            or q['protected_contexts_seen']!=32*q['completed_updates']):raise ValueError('Fitting/protection clocks differ')
    return runtime


def fit(ctx,out,seed,*,end_monotonic):
    if ctx.device.type!='cuda':raise RuntimeError('Real fitting requires separately guarded GPU admission; CPU fixtures use synthetic advance')
    out=Path(out);out.mkdir(parents=True,exist_ok=False);runtime=build(ctx,seed)
    stops=[];handlers={s:signal.signal(s,lambda sig,frame:stops.append(sig)) for s in (signal.SIGINT,signal.SIGTERM)}
    def save(status):
        path=out/('final.pt' if status=='completed' else 'recovery.pt');temp=path.with_suffix('.partial.pt')
        torch.save(snapshot(ctx,runtime,status),temp);temp.replace(path)
        dump(out/'checkpoint_receipt.json',{'path':path.name,'SHA256':sha(path),'status':status,'completed_updates':runtime.policy.completed_steps})
        return path
    try:
        while runtime.policy.completed_steps<PG_UPDATES:
            if stops or time.monotonic()>=end_monotonic:raise InterruptedError('Finite allocation ending; preserve completed optimizer-pair checkpoint')
            metrics=advance(ctx,runtime)
            if metrics['step']==1 or metrics['step']%250==0:dump(out/'progress.json',metrics)
            if metrics['step']%1000==0:save('recovery')
        return runtime,save('completed')
    except BaseException as exc:
        if runtime.policy._phase=='ready':save('incomplete')
        dump(out/'failure.json',{'error':repr(exc),'completed_updates':runtime.policy.completed_steps,'phase':runtime.policy._phase,'primary_endpoint_complete':False})
        raise
    finally:
        for s,handler in handlers.items():signal.signal(s,handler)
