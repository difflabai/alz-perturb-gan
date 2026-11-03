"""Actual optimization-only classifier/count-model gradient probe; no fitting."""
import datetime,hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path('/ml2/alz');B=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark');S=Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/core-comparison-20261103')
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
sys.path[:0]=list(map(str,[ROOT,ROOT/'model',S,B,B/'current_base_20261029/runtime/lib/python3.12/site-packages',B/'.venv312/lib/python3.12/site-packages']))
import numpy as np,torch
import semantic_pg_adapter as adapter
import semantic_pg_update as update
from pg_model import critic_features
from contracts import fold

torch.set_num_threads(1);torch.set_num_interop_threads(1)
config=dict(z_dim=8,stage_balance=False,cell_z_dim=8,theta_init_scale=1.,fourier_feature_scale=1.)
update.configure(config);update.configure_semantic(dict(enabled=True,temperature=1.,epsilon=1e-4))
started=time.monotonic();ctx=adapter.Context(S/'caches/fold0','cpu');assert set(ctx.ids)==set(fold(0)['optimization_donors']) and len(ctx.ids)==33
b,g,r,u,streams=adapter.build(ctx.mu,ctx.theta,'cpu',0,0);rec,G,D,P,*_=b
stage=np.rint(ctx.fitc[:,3]*3).astype(int);ds=[int(np.flatnonzero(stage==s)[0]) for s in range(4)]
rows=np.stack([np.resize(ctx.by[d],128) for d in ds]);conditions=ctx.conditions[torch.as_tensor(ds)];libraries=ctx.X[torch.as_tensor(rows)].sum(-1)
z,idx=P.sample(4,generator=torch.Generator().manual_seed(20261103));rate=G.rates(z,conditions,128,libraries,torch.Generator().manual_seed(20261104));counts=torch.poisson(rate.detach(),generator=torch.Generator().manual_seed(20261105));pcs=critic_features(counts,rate,ctx.mean,ctx.sd)@ctx.W
semantic=ctx.observation.subtype_features(pcs);objective=(semantic*torch.arange(1,7,dtype=semantic.dtype)).sum()
gradients=torch.autograd.grad(objective,(pcs,G.out[-1].weight,G.log_theta,P.z))
summary=[]
for name,grad in zip(['PC_inputs','G_output_weight','G_dispersion','particle_prior'],gradients):
 assert torch.isfinite(grad).all() and grad.abs().sum()>0,name
 summary.append({'name':name,'norm':float(grad.norm()),'max_abs':float(grad.abs().max()),'nonzero_fraction':float((grad!=0).float().mean())})
with torch.no_grad():
 logits=ctx.observation.classifier(pcs.detach().reshape(-1,50));prob=logits.softmax(-1);labels=logits.argmax(-1).reshape(4,128)
 independent=torch.stack([torch.bincount(row,minlength=6).float().div(128).sqrt() for row in labels]);assert torch.equal(semantic.detach(),independent)
assert ctx.observation.state_digest()==ctx.observation.digest and all(p.grad is None for p in ctx.observation.classifier.parameters())
assert b[-1].completed_steps==0 and not torch.cuda.is_initialized()
result={'status':'PASSED_FITTING_ONLY_SEMANTIC_GRADIENT_PROBE_NO_TRAINING','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_CPU_seconds':time.monotonic()-started,'optimization_donors':33,'probe_donors':ctx.ids[ds].tolist(),'calibration_or_evaluation_targets_read':False,'bags':4,'cells_per_bag':128,'optimizer_updates':0,'completed_policy_steps':0,'classifier_state_SHA256':ctx.observation.digest,'classifier_file_SHA256':ctx.classifier_file_SHA256,'cache_manifest_SHA256':adapter.sha(ctx.cache/'manifest.json'),'semantic_forward_equals_independent_hard_composition':True,'semantic_features':semantic.detach().tolist(),'gradient_diagnostics':summary,'max_probability_percentiles':np.percentile(prob.max(-1).values.numpy(),[0,10,50,90,100]).tolist(),'saturated_gt_09999_fraction':float((prob.max(-1).values>.9999).float().mean()),'native_recipe':rec.name,'source_SHA256':{str(p):adapter.sha(p) for p in [Path(__file__),ROOT/'model/semantic_pg_adapter.py',ROOT/'model/semantic_pg_update.py',ROOT/'model/semantic_observation.py']}}
(ROOT/'analysis/semantic_fitting_gradient_probe.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
