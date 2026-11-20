"""Conditional raw-count donor GAN, refactored from hash-verified CPU reference.

The adversarial forward uses exact Gamma-Poisson NB counts. The backward path
uses a predeclared biased straight-through derivative through the transformed,
reparameterized Gamma rate. Mean library sizes are sampled from training cells.
This avoids a real-count versus continuous-rate support mismatch.
No unannounced jitter or objective fallback is used.
"""
from __future__ import annotations
import copy
import json
from pathlib import Path
import signal
import time
import numpy as np
import torch
from torch import nn
from particlegan import get_recipe,scale_learning_rates
from particlegan.training import InputNoise,input_noise_std,output_noise_std


def inv_softplus(x):
    return x+torch.log(-torch.expm1(-x))


class CountGenerator(nn.Module):
    def __init__(self,mean,theta,zd=4,zc=8,h=128):
        super().__init__()
        self.zc=zc
        self.donor=nn.Sequential(nn.Linear(zd+7,h),nn.SiLU(),nn.Linear(h,2*h))
        self.cell=nn.Sequential(nn.Linear(zc,h),nn.SiLU())
        self.out=nn.Sequential(nn.SiLU(),nn.Linear(h,h),nn.SiLU(),nn.Linear(h,len(mean)))
        nn.init.normal_(self.out[-1].weight,std=.005)
        with torch.no_grad():
            mass=torch.as_tensor(mean).clamp_min(1e-5)
            self.out[-1].bias.copy_(torch.log(mass/mass.sum()))
        self.log_theta=nn.Parameter(inv_softplus(torch.as_tensor(theta).clamp_min(.1)-.05))
    def parameters_nb(self,z,c,k,library,gen=None):
        scale,shift=self.donor(torch.cat([z,c],1)).chunk(2,1)
        noise=torch.randn(len(z),k,self.zc,device=z.device,generator=gen)
        h=self.cell(noise)*(1+scale[:,None])+shift[:,None]
        proportions=torch.softmax(self.out(h),dim=-1).clamp_min(1e-8)
        proportions=proportions/proportions.sum(-1,keepdim=True)
        mu=library.clamp_min(1)[...,None]*proportions
        theta=nn.functional.softplus(self.log_theta)+.05
        return mu,theta
    def rates(self,z,c,k,library,gen=None):
        mu,theta=self.parameters_nb(z,c,k,library,gen)
        return torch.distributions.Gamma(theta,theta/mu).rsample()*(library>0)[...,None]
    def forward(self,z,c,k,library,gen=None):
        rate=self.rates(z,c,k,library,gen)
        counts=torch.poisson(rate.detach(),generator=gen)
        return counts,rate
    def sample_counts(self,z,c,k,library,gen=None):
        return torch.poisson(self.rates(z,c,k,library,gen),generator=gen)


class LogGenerator(nn.Module):
    """Predeclared NB-instability fallback in standardized log-HVG space."""
    def __init__(self,n_genes,zd=4,zc=8,h=128):
        super().__init__();self.zc=zc
        self.donor=nn.Sequential(nn.Linear(zd+7,h),nn.SiLU(),nn.Linear(h,2*h))
        self.cell=nn.Sequential(nn.Linear(zc,h),nn.SiLU())
        self.out=nn.Sequential(nn.SiLU(),nn.Linear(h,h),nn.SiLU(),nn.Linear(h,n_genes))
    def forward(self,z,c,k,gen=None):
        scale,shift=self.donor(torch.cat([z,c],1)).chunk(2,1)
        noise=torch.randn(len(z),k,self.zc,device=z.device,generator=gen)
        return self.out(self.cell(noise)*(1+scale[:,None])+shift[:,None])


class SetCritic(nn.Module):
    def __init__(self,n_genes,h=128,std_epsilon=1e-6):
        super().__init__()
        self.std_epsilon=std_epsilon
        self.phi=nn.Sequential(nn.Linear(n_genes+7,h),nn.SiLU(),nn.Linear(h,h),nn.SiLU())
        self.rho=nn.Sequential(nn.Linear(2*h+7,h),nn.SiLU(),nn.Linear(h,1))
    def forward(self,x,c):
        b,k,_=x.shape
        f=self.phi(torch.cat([x,c[:,None].expand(b,k,7)],2))
        sd=(f.var(1,unbiased=False)+self.std_epsilon).sqrt() if self.std_epsilon else f.std(1,unbiased=False)
        pooled=torch.cat([f.mean(1),sd],1)
        return self.rho(torch.cat([pooled,c],1)).squeeze(1)


def standardize_counts(x,mean,sd):
    return (torch.log1p(x*(1e4/x.sum(-1,keepdim=True).clamp_min(1)))-mean)/sd


def critic_features(counts,rate,mean,sd):
    discrete=standardize_counts(counts,mean,sd)
    smooth=standardize_counts(rate,mean,sd)
    return discrete+(smooth-smooth.detach())


def train_pg(counts,cell_donor,donor_c,space,folder,seed,*,steps=20000,bag=128,batch=32,head="nb",diagnostic=False,stop_after=None,pooling_epsilon=1e-6):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    device=torch.device("cuda")
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError("A single guarded CUDA device is required; no CPU fallback")
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    recipe=get_recipe("gan",total_steps=steps,batch_size=batch,z_dim=4,num_particles=512,
                      network_lr_horizon_cap=steps)
    X=torch.as_tensor(counts,dtype=torch.float32,device=device)
    if head=="nb":
        mu=X.mean(0);var=X.var(0,unbiased=False)
        theta=(mu.square()/(var-mu).clamp_min(1e-4)).clamp(.1,10000)
        G=CountGenerator(mu.cpu(),theta.cpu()).to(device)
        semantics="Exact Gamma-Poisson counts forward; biased transformed-rate straight-through backward; training-only sampled library"
    elif head=="log_cp10k":
        G=LogGenerator(X.shape[1]).to(device)
        semantics="Predeclared continuous standardized logCP10k fallback after S1 NB failure at completed step841; no NB/count claims"
    else:
        raise ValueError(head)
    D=SetCritic(X.shape[1],std_epsilon=pooling_epsilon).to(device);prior=recipe.make_prior().to(device)
    optg,optd=recipe.make_optimizers(G,D,prior,ema_critic=copy.deepcopy(D))
    penalty=recipe.make_critic_penalty(optd);loss=recipe.make_loss()
    base=[[g["lr"] for g in o.param_groups] for o in (optg,optd)]
    noisy=InputNoise(D,0.,torch.Generator(device=device).manual_seed(seed+5))
    emaG,emaP=copy.deepcopy(G).eval(),copy.deepcopy(prior).eval()
    lat=torch.Generator(device=device).manual_seed(seed+2)
    cg=torch.Generator(device=device).manual_seed(seed+3)
    mean=torch.as_tensor(space.mean,device=device);sd=torch.as_tensor(space.sd,device=device)
    ids=sorted(donor_c)
    C=torch.as_tensor(np.stack([donor_c[d] for d in ids]),dtype=torch.float32,device=device)
    by=[np.where(cell_donor==d)[0] for d in ids]
    checkpoint=folder/"checkpoint.pt";start_step=0;history=[];previous_seconds=0.
    if checkpoint.exists():
        q=torch.load(checkpoint,map_location=device,weights_only=False)
        if q["recipe"]!=recipe.to_dict() or q["seed"]!=seed:
            raise RuntimeError("Checkpoint recipe/seed differs; inspect before resume")
        if q.get("head","nb")!=head or q.get("nb_training")!=semantics:
            raise RuntimeError("Checkpoint NB training estimator differs; do not mix objectives")
        if q.get("critic_std_epsilon",0.)!=pooling_epsilon:
            raise RuntimeError("Checkpoint critic pooling epsilon differs; do not mix numerical formulations")
        for target,key in [(G,"G"),(D,"D"),(prior,"prior"),(emaG,"emaG"),(emaP,"emaP"),(optg,"optg"),(optd,"optd")]:
            target.load_state_dict(q[key])
        start_step=q["completed_steps"];history=q["history"];previous_seconds=q["training_seconds"]
        torch.set_rng_state(q["cpu_rng"].cpu());torch.cuda.set_rng_state(q["cuda_rng"].cpu())
        lat.set_state(q["latent_rng"].cpu());cg.set_state(q["cell_rng"].cpu());rng.bit_generator.state=q["numpy_rng"]
    stopped=[]
    def stop(signum,frame):
        stopped.append(signum)
    handlers={s:signal.signal(s,stop) for s in [signal.SIGTERM,signal.SIGINT]}
    started=time.monotonic();completed=start_step
    def save(status):
        target=folder/"failed_checkpoint.pt" if status=="failed" else checkpoint
        temp=target.with_suffix(".partial.pt")
        torch.save({"G":G.state_dict(),"D":D.state_dict(),"prior":prior.state_dict(),
                    "emaG":emaG.state_dict(),"emaP":emaP.state_dict(),"optg":optg.state_dict(),"optd":optd.state_dict(),
                    "recipe":recipe.to_dict(),"seed":seed,"completed_steps":completed,"history":history,
                    "training_seconds":previous_seconds+time.monotonic()-started,"status":status,
                    "cpu_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state(),
                    "latent_rng":lat.get_state(),"cell_rng":cg.get_state(),"numpy_rng":rng.bit_generator.state,
                    "head":head,"critic_std_epsilon":pooling_epsilon,"nb_training":semantics,
                    "nb_evaluation":"Gamma-Poisson counts" if head=="nb" else "continuous standardized log-HVG output; direct same PCA projection; recipe final output noise retained",
                    "output_noise":"recipe noise in critic feature space during training; retained for continuous fallback sampling, excluded from NB count evaluation"},temp)
        temp.replace(target)
    def require_finite(named,label,*,grad=False):
        tensors=[(n,p.grad if grad else p) for n,p in named if not grad or p.grad is not None]
        flags=torch.stack([torch.isfinite(t).all() for _,t in tensors])
        if not bool(flags.all()):
            bad=[n for (n,_),ok in zip(tensors,flags.cpu().tolist()) if not ok]
            raise FloatingPointError("First nonfinite stage: "+label+": "+repr(bad))
    def real_batch():
        donor_idx=rng.integers(len(ids),size=batch)
        indices=np.stack([rng.choice(by[i],bag,replace=True) for i in donor_idx])
        raw=X[torch.as_tensor(indices,device=device)]
        return standardize_counts(raw,mean,sd),C[torch.as_tensor(donor_idx,device=device)],raw.sum(-1)
    def fake_batch(z,c,library):
        if head=="log_cp10k":
            return G(z,c,bag,cg)
        fake_counts,rate=G(z,c,bag,library,cg)
        return critic_features(fake_counts,rate,mean,sd)
    try:
        limit=min(steps,stop_after) if stop_after is not None else steps
        for step in range(start_step,limit):
            if stopped:
                raise InterruptedError("Stop signal received; checkpoint before exit")
            scale_learning_rates(step,recipe,(optg,optd),base,prior)
            noisy.std=input_noise_std(recipe,step);so=output_noise_std(recipe,step)
            real,c,library=real_batch()
            with torch.no_grad():
                z,_=prior.sample(batch,generator=lat)
                fake=fake_batch(z,c,library)
                fake=fake+so*torch.randn(fake.shape,device=device,generator=cg)
            dr,df=noisy(real,c),noisy(fake,c)
            ld=loss.d_loss(dr,df)+penalty(noisy,real,fake,c)
            optd.zero_grad();ld.backward()
            require_finite(list(D.named_parameters()),"critic gradients before optimizer",grad=True)
            optd.step()
            require_finite(list(D.named_parameters()),"critic parameters after optimizer")
            D.requires_grad_(False)
            real,c,library=real_batch();z,_=prior.sample(batch,generator=lat)
            fake=fake_batch(z,c,library)
            fake=fake+so*torch.randn(fake.shape,device=device,generator=cg)
            lg=loss.g_loss(noisy(fake,c),noisy(real,c))
            optg.zero_grad();lg.backward()
            named_g=list(G.named_parameters())+[("prior."+n,p) for n,p in prior.named_parameters()]
            require_finite(named_g,"generator/prior gradients before optimizer",grad=True)
            if step==start_step:
                grads=[G.out[-1].weight.grad,prior.z.grad]+([G.log_theta.grad] if head=="nb" else [])
                if any(g is None or not bool(torch.isfinite(g).all()) or not bool(g.abs().sum()>0) for g in grads):
                    raise RuntimeError("Missing/nonfinite/zero generator, applicable dispersion or particle gradients")
            optg.step();D.requires_grad_(True)
            require_finite(named_g,"generator/prior parameters after optimizer")
            with torch.no_grad():
                for target,source in [(emaG,G),(emaP,prior)]:
                    for a,b in zip(target.parameters(),source.parameters()):
                        a.mul_(recipe.ema_decay).add_(b,alpha=1-recipe.ema_decay)
            completed=step+1
            if step%250==0 or completed==steps:
                rec={"step":completed,"d_loss":float(ld.detach()),"g_loss":float(lg.detach()),
                     "critic_gap":float((dr-df).mean().detach()),"elapsed_seconds":previous_seconds+time.monotonic()-started,
                     "network_lr":optg.param_groups[0]["lr"]}
                if not np.isfinite([rec["d_loss"],rec["g_loss"],rec["critic_gap"]]).all():
                    raise FloatingPointError("Nonfinite GAN training diagnostics")
                history.append(rec);print(json.dumps(rec),flush=True)
            if completed%1000==0:
                save("active")
        save("completed" if completed==steps else "diagnostic_stopped_before_full_horizon")
    except InterruptedError:
        save("interrupted_at_complete_update_boundary")
        raise
    except BaseException:
        save("failed")
        raise
    finally:
        for s,h in handlers.items():
            signal.signal(s,h)
    if head=="log_cp10k":
        emaG.final_output_noise=float(output_noise_std(recipe,steps))
    return emaG,emaP,history,previous_seconds+time.monotonic()-started


@torch.no_grad()
def sample_pg(G,prior,condition,space,seed,*,library_pool=None,m=200,k=400):
    device=next(G.parameters()).device
    torch.manual_seed(seed)
    gen=torch.Generator(device=device).manual_seed(seed)
    C=torch.as_tensor(condition,device=device,dtype=torch.float32)[None]
    mean=torch.as_tensor(space.mean,device=device);sd=torch.as_tensor(space.sd,device=device)
    W=torch.as_tensor(space.W/space.pc_sd,device=device)
    libraries=None if isinstance(G,LogGenerator) else torch.as_tensor(library_pool,device=device,dtype=torch.float32)
    out=[]
    for start in range(0,m,5):
        b=min(5,m-start);z,_=prior.sample(b,generator=gen)
        if isinstance(G,LogGenerator):
            features=G(z,C.expand(b,-1),k,gen)
            features=features+G.final_output_noise*torch.randn(features.shape,device=device,generator=gen)
            pc=features@W
        else:
            lib=libraries[torch.randint(len(libraries),(b,k),device=device,generator=gen)]
            counts=G.sample_counts(z,C.expand(b,-1),k,lib,gen)
            pc=standardize_counts(counts,mean,sd)@W
        if not bool(torch.isfinite(pc).all()):
            raise FloatingPointError("Nonfinite generated count projection")
        out.append(pc.cpu().numpy())
    return np.concatenate(out)
