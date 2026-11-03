"""Optimization-only learned-observation gradient probe, with zero fitting updates."""
import datetime,hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path('/ml2/alz');B=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark');S=Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/core-comparison-20261103')
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
sys.path[:0]=list(map(str,[ROOT,ROOT/'model',S,B,B/'current_base_20261029/runtime/lib/python3.12/site-packages',B/'.venv312/lib/python3.12/site-packages']))
import numpy as np,torch
import learned_pg_adapter as adapter
import learned_pg_update as update
from learned_cell_observation import OPTIONS
from critic import moment_features
from pg_model import critic_features
from contracts import fold

torch.set_num_threads(1);torch.set_num_interop_threads(1)
update.configure(dict(z_dim=8,stage_balance=False,cell_z_dim=8,theta_init_scale=1.,fourier_feature_scale=1.));update.configure_observation({**OPTIONS,'enabled':True})
started=time.monotonic();ctx=adapter.Context(S/'caches/fold0','cpu');assert set(ctx.ids)==set(fold(0)['optimization_donors']) and len(ctx.ids)==33
b,g,r,u,streams=adapter.build(ctx.mu,ctx.theta,'cpu',0,0);rec,G,D,P,*_=b
stage=np.rint(ctx.fitc[:,3]*3).astype(int);ds=[int(np.flatnonzero(stage==s)[0]) for s in range(4)]
rows=np.stack([np.resize(ctx.by[d],128) for d in ds]);condition=ctx.conditions[torch.as_tensor(ds)];library=ctx.X[torch.as_tensor(rows)].sum(-1);real=ctx.trainpcs[torch.as_tensor(rows)]
def fake():
 z,_=P.sample(4,generator=b[-1].latent_generator);rate=G.rates(z,condition,128,library,g['cell']);counts=torch.poisson(rate.detach(),generator=g['count']);return critic_features(counts,rate,ctx.mean,ctx.sd)@ctx.W
with torch.no_grad():fake_D=fake()
encoder_grads=[]
for label,pcs in [('real',real),('fake',fake_D)]:
 features=D.features(pcs);assert torch.equal(features[:,:100],moment_features(pcs))
 grads=torch.autograd.grad(D(features,condition).mean(),tuple(D.cell_encoder.parameters()))
 assert all(torch.isfinite(x).all() for x in grads) and sum(float(x.abs().sum()) for x in grads)>0
 encoder_grads.append({'side':label,'norm':float(sum(x.square().sum() for x in grads).sqrt()),'all_finite':True})
D.requires_grad_(False);pcs=fake();features=D.features(pcs);loss=b[7].g_loss(D(features,condition))
grads=torch.autograd.grad(loss,(G.out[-1].weight,G.log_theta,P.z),retain_graph=True)
generator_grads=[]
for name,x in zip(['G_output_weight','G_dispersion','particle_prior'],grads):
 assert torch.isfinite(x).all() and x.abs().sum()>0;generator_grads.append({'name':name,'norm':float(x.norm()),'nonzero_fraction':float((x!=0).float().mean())})
isolated=(features[:,100:]*torch.arange(1,51,dtype=features.dtype)).sum()
isolated_grads=torch.autograd.grad(isolated,(G.out[-1].weight,G.log_theta,P.z));isolated_summary=[]
for name,x in zip(['G_output_weight','G_dispersion','particle_prior'],isolated_grads):
 assert torch.isfinite(x).all() and x.abs().sum()>0;isolated_summary.append({'name':name,'norm':float(x.norm()),'nonzero_fraction':float((x!=0).float().mean())})
assert all(p.grad is None and not p.requires_grad for p in D.parameters())
assert torch.equal(features[:,:100].detach(),moment_features(pcs).detach()) and float(features[:,100:].detach().square().sum(1).max())<=1.0001
assert b[-1].completed_steps==0 and u.steps==0 and not torch.cuda.is_initialized()
paths=[Path(__file__),ROOT/'model/learned_pg_adapter.py',ROOT/'model/learned_pg_update.py',ROOT/'model/learned_cell_observation.py']
q={'status':'PASSED_FITTING_ONLY_LEARNED_OBSERVATION_PROBE_NO_TRAINING','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_CPU_seconds':time.monotonic()-started,
   'optimization_donors':33,'probe_donors':ctx.ids[ds].tolist(),'calibration_or_evaluation_targets_read':False,'real_probe_bags':4,'generated_probe_bags':8,'cells_per_bag':128,
   'optimizer_updates':0,'completed_policy_steps':0,'native_recipe':rec.name,'learned_observation_config':dict(update.OBSERVATION_CONFIG),'observation_dimensions':150,
   'first100_moment_coordinates_exact':True,'new_feature_block_norm_bound_verified':True,'encoder_gradients_real_and_fake':encoder_grads,'generator_gradients_through_frozen_encoder':generator_grads,
   'isolated_learned50_generator_gradients':isolated_summary,
   'cache_manifest_SHA256':adapter.sha(ctx.cache/'manifest.json'),'CUDA_initialized':False,'source_SHA256':{str(p):adapter.sha(p) for p in paths}}
(ROOT/'analysis/learned_observation_fitting_probe.json').write_text(json.dumps(q,indent=2,allow_nan=False)+'\n');print(json.dumps(q,indent=2))
