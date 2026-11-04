"""Fold-only input adapter around the exact preserved PG update."""
from pathlib import Path
import copy,json,signal,sys,time
import numpy as np,scipy.sparse as sp,torch
from contracts import B,PG_UPDATES,SEEDS,checked_cache,sha,dump,validate_train_donors
sys.path.insert(0,str(B))
from particlegan import UpdatePolicy
from particlegan.optim.dualnorm import NormalizedOptimizer
from pg_model import CountGenerator,standardize_counts
from bench_core import load_space
from critic import VectorCritic,N_FEATURES
from stage_law import StageLaw,mask_adnc
from usage import Usage
from stage_prior import CONFIG as STAGE_PRIOR_CONFIG
from stage_pg_update import recipe,advance,CONFIG,normalized_config,bag_features
STREAM_NAMES=['generator_init','critic_init','prior_init','policy_latent','policy_penalty','policy_eval','policy_noise','cell','count','gamma_global','donor_sampler']
def training_streams(fold,seed):
 if fold not in range(3) or seed not in SEEDS:raise ValueError('unregistered training cell')
 children=np.random.SeedSequence([20261103,fold,seed]).spawn(len(STREAM_NAMES))
 return {k:int(c.generate_state(1,dtype=np.uint64)[0])%(2**63-16) for k,c in zip(STREAM_NAMES,children)}
class Context:
 def __init__(self,cache,device):
  manifest,f=checked_cache(cache,'train');cache=Path(cache);self.cache=cache;self.manifest=manifest;self.fold=manifest['fold'];self.device=torch.device(device)
  m=np.load(cache/'train/metadata.npz',allow_pickle=False);self.fit_labels=m['donor'].copy();self.cell_names=m['cell'].copy();self.ids=m['donors'].copy();validate_train_donors(self.fit_labels,f['optimization_donors'])
  if list(self.ids)!=f['optimization_donors']:raise ValueError('ordered optimizer donor set differs')
  self.fitc=m['raw_c'].astype(np.float32);self.space=load_space(cache);counts=sp.load_npz(cache/'train/counts_hvg.npz').astype(np.float32)
  if counts.shape!=(len(self.fit_labels),2000) or not np.isfinite(counts.data).all() or (counts.data<0).any():raise ValueError('training count shape/range invalid')
  self.X=torch.as_tensor(counts.toarray(),device=self.device);self.by=[np.where(self.fit_labels==d)[0] for d in self.ids]
  self.conditions=torch.as_tensor(mask_adnc(self.fitc),device=self.device);self.mean=torch.as_tensor(self.space.mean,device=self.device);self.sd=torch.as_tensor(self.space.sd,device=self.device);self.W=torch.as_tensor(self.space.W/self.space.pc_sd,device=self.device)
  self.trainpcs=self.project(self.X);self.fitlibs=self.X.sum(-1)
  if not bool((self.fitlibs>0).all()):raise ValueError('common training zero-HVG rule not applied')
  self.mu=self.X.mean(0);v=self.X.var(0,unbiased=False);self.theta=(self.mu.square()/(v-self.mu).clamp_min(1e-4)).clamp(.1,10000)
  self.law=StageLaw(self.fitc,self.by,self.fitlibs.cpu().numpy())
 def project(self,x):return standardize_counts(x,self.mean,self.sd)@self.W
 def batch(self,rng):
  if CONFIG['stage_balance']:
   stage=np.rint(self.fitc[:,3]*3).astype(int);groups=[np.flatnonzero(stage==s) for s in range(4)]
   if any(len(group)==0 for group in groups):raise ValueError('balanced fitting requires optimization donors at every stage')
   order=np.repeat(np.arange(4),8);rng.shuffle(order);ds=np.asarray([rng.choice(groups[s]) for s in order])
  else:ds=rng.integers(len(self.ids),size=32)
  idx=torch.as_tensor(np.stack([rng.choice(self.by[i],128,replace=True) for i in ds]),device=self.device)
  return bag_features(self.trainpcs[idx]),self.conditions[torch.as_tensor(ds,device=self.device)],self.X[idx].sum(-1)
def build(mu,theta,device,fold,seed):
 streams=training_streams(fold,seed);rec=recipe();torch.manual_seed(streams['generator_init']);G=CountGenerator(mu.cpu(),(theta*CONFIG['theta_init_scale']).cpu(),zd=rec.z_dim,zc=CONFIG['cell_z_dim']).to(device)
 torch.manual_seed(streams['critic_init']);D=VectorCritic(N_FEATURES,h=128).to(device)
 P=rec.make_prior(generator=torch.Generator().manual_seed(streams['prior_init'])).to(device);og,od=rec.make_optimizers(G,D,P);pen=rec.make_critic_penalty(od);loss=rec.make_loss();reg=rec.make_prior_regularizer(prior=P,weight=1.)
 policy=UpdatePolicy(rec,G,D,prior=P,generator_optimizer=og,critic_optimizer=od,penalty=pen,seed=seed,row_semantics='conditional',streams={name:torch.Generator(device=device).manual_seed(streams['policy_'+key]) for name,key in [('latent_generator','latent'),('penalty_generator','penalty'),('eval_generator','eval'),('noise_generator','noise')]})
 if not(rec.name=='bispf_optimized' and rec.loss=='non_saturating' and rec.num_particles==512 and rec.z_dim==CONFIG['z_dim'] and rec.ema_decay==rec.input_noise_std==rec.output_noise_std==0 and rec.network_lr_horizon_cap is None and rec.lr_floor==rec.network_lr_floor==1 and isinstance(og,NormalizedOptimizer) and isinstance(od,NormalizedOptimizer)):raise ValueError('canonical PG recipe changed')
 generators={k:torch.Generator(device=device).manual_seed(streams[k]) for k in ['cell','count']};rng=np.random.default_rng(streams['donor_sampler']);torch.manual_seed(streams['gamma_global'])
 return (rec,G,D,P,og,od,pen,loss,reg,policy),generators,rng,Usage(P.z),streams
def snapshot(ctx,bundle,generators,rng,usage,seed,streams,status):
 rec,G,D,P,og,od,pen,loss,reg,policy=bundle
 if policy._phase!='ready':raise ValueError('no checkpoint at a partial optimizer-pair boundary')
 q={'schema':1,'method':'pg','fold':ctx.fold,'seed':seed,'cache_manifest_SHA256':sha(ctx.cache/'manifest.json'),'recipe':rec.to_dict(),'policy':policy.state_dict(),'prior_regularizer':rec.prior_regularizer_checkpoint(reg),'generators':{k:g.get_state() for k,g in generators.items()},'numpy_rng':rng.bit_generator.state,'usage':usage.state_dict(),'training_streams':streams,'torch_cpu_rng':torch.get_rng_state(),'torch_device_rng':torch.cuda.get_rng_state(ctx.device) if ctx.device.type=='cuda' else None,'completed_updates':policy.completed_steps,'fixed_endpoint':PG_UPDATES,'status':status}
 q['variant_config']=dict(CONFIG);q['native_recipe']='bispf_optimized';q['stage_prior_config']=dict(STAGE_PRIOR_CONFIG)
 return copy.deepcopy(q)
def restore(ctx,q,*,expected_endpoint=PG_UPDATES):
 if normalized_config(q.get('variant_config',{'z_dim':q['recipe']['z_dim']}))!=CONFIG:raise ValueError('variant sampling configuration differs')
 if expected_endpoint not in (20000,40000) or q['cache_manifest_SHA256']!=sha(ctx.cache/'manifest.json') or q['fold']!=ctx.fold or q['fixed_endpoint']!=expected_endpoint:raise ValueError('checkpoint fold/cache/endpoint mismatch')
 if q.get('native_recipe')!='bispf_optimized' or q.get('stage_prior_config')!=STAGE_PRIOR_CONFIG:raise ValueError('Conditional stage prior checkpoint configuration differs')
 b,g,r,u,seeds=build(ctx.mu,ctx.theta,ctx.device,ctx.fold,q['seed']);rec=b[0];policy=b[-1]
 if q['recipe']!=rec.to_dict() or q['training_streams']!=seeds:raise ValueError('checkpoint recipe/seeds differ')
 policy.load_state_dict(q['policy']);rec.restore_prior_regularizer(b[-2],q['prior_regularizer'].get('prior_regularizer'))
 for k,v in q['generators'].items():g[k].set_state(v.cpu())
 r.bit_generator.state=q['numpy_rng']
 for k in ['draws','selected_steps','moved_steps','unsampled_moved_steps']:setattr(u,k,q['usage'][k].to(ctx.device).clone())
 u.steps=q['usage']['steps'];u.unique_batch_histogram=list(q['usage']['unique_batch_histogram']);torch.set_rng_state(q['torch_cpu_rng'].cpu())
 if ctx.device.type=='cuda':torch.cuda.set_rng_state(q['torch_device_rng'].cpu(),ctx.device)
 return b,g,r,u,seeds
def fit(ctx,out,seed,*,end_monotonic):
 if ctx.device.type!='cuda':raise RuntimeError('real PG fitting requires a separately reviewed guarded GPU admission; CPU tests use synthetic advance only')
 out=Path(out);out.mkdir(exist_ok=False,parents=True);b,g,r,u,seeds=build(ctx.mu,ctx.theta,ctx.device,ctx.fold,seed);stops=[];handlers={s:signal.signal(s,lambda sig,frame:stops.append(sig)) for s in [signal.SIGINT,signal.SIGTERM]}
 def save(status):
  p=out/'final.pt' if status=='completed' else out/'recovery.pt';tmp=p.with_suffix('.partial.pt');torch.save(snapshot(ctx,b,g,r,u,seed,seeds,status),tmp);tmp.replace(p);dump(out/'checkpoint_receipt.json',{'path':p.name,'SHA256':sha(p),'status':status,'completed_updates':b[-1].completed_steps});return p
 try:
  while b[-1].completed_steps<PG_UPDATES:
   if stops or time.monotonic()>=end_monotonic:raise InterruptedError('finite scientific allocation ending; preserve completed-update checkpoint')
   metrics=advance(ctx,b,g,r,u,execution_limit=PG_UPDATES)
   if metrics['step']%250==0 or metrics['step']==1:dump(out/'progress.json',metrics)
   if metrics['step']%1000==0:save('recovery')
  p=save('completed');return b,p
 except BaseException as exc:
  if b[-1]._phase=='ready':save('incomplete')
  dump(out/'failure.json',{'error':repr(exc),'completed_updates':b[-1].completed_steps,'phase':b[-1]._phase,'primary_endpoint_complete':False});raise
 finally:
  for s,h in handlers.items():signal.signal(s,h)
