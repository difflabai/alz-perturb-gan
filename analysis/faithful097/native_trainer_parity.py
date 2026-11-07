"""Compare the conditional caller loop to the actual released GANTrainer.

GANTrainer only accepts scalar unconditional models. Its exact update is exercised
with a critic condition facade and public policy.generation callback on the same
fixed synthetic conditional batches; application sampling stays outside native PG.
No native package edits, real donor data, CUDA, or old weights are used.
"""
import datetime,hashlib,json,os,sys,tempfile
from pathlib import Path
ROOT=Path('/ml2/alz');P=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark');S=Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/core-comparison-20261103')
sys.path[:0]=list(map(str,[ROOT/'analysis/faithful097/model',ROOT/'analysis/duration096',ROOT/'refinement097_runtime',ROOT/'model',ROOT,S,P,P/'current_base_20261029/runtime/lib/python3.12/site-packages',P/'.venv312/lib/python3.12/site-packages']))
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
import torch,numpy as np,released_adapter as app,released_update as up
from particlegan.training import GANTrainer
from native_fixture import context
from pg_model import critic_features
from state_fingerprint import fingerprint
from release_env import PENALTY_OPTIONS
class ConditionedCritic(torch.nn.Module):
 def __init__(self,base):super().__init__();self.base=base;self.condition=None
 def forward(self,x):return self.base(x,self.condition)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(hyper,seed,pairs=3):
 up.configure_hyperparameters(hyper)
 with tempfile.TemporaryDirectory() as folder:
  cache=Path(folder);(cache/'manifest.json').write_text('{}');ctx=context(cache,fullwidth=True)
  states=[];losses=[]
  for native in (False,True):
   b,g,r,u,streams=app.build(ctx.mu,ctx.theta,'cpu',0,seed);rec,G,D,prior,og,od,pen,loss,reg,policy=b
   assert policy.controller is None and rec.ema_decay==0 and not rec.kinetic_transport_weight and not rec.kinetic_transport_local_weight and not rec.prior_l2
   assert policy.completed_steps==0 and not og.state and not od.state
   if native:
    facade=ConditionedCritic(D)
    trainer=GANTrainer(rec,G,facade,prior=prior,seed=seed,max_steps=20000,serve_after_step=False,
     penalty_options=PENALTY_OPTIONS,**{name:torch.Generator().manual_seed(streams['policy_'+key]) for name,key in [('latent_generator','latent'),('penalty_generator','penalty'),('eval_generator','eval'),('noise_generator','noise')]})
    assert trainer.policy.controller is None
    state={};calls=[0]
    def generation(model,z):
     if calls[0]%2:
      real,c,lib=ctx.batch(r);state.update(real=real,c=c,lib=lib);facade.condition=c
     calls[0]+=1
     rate=model.rates(z,state['c'],128,state['lib'],g['cell'])
     counts=torch.poisson(rate.detach(),generator=g['count'])
     return up.bag_features(critic_features(counts,rate,ctx.mean,ctx.sd)@ctx.W)
    trainer.policy.generation=generation
    for i in range(pairs):
     real,c,lib=ctx.batch(r);state.update(real=real,c=c,lib=lib);facade.condition=c
     values=trainer.step(real,generator_real=lambda:state['real'],collect_stats=(i%250==0))
     losses.append(('native',i,{k:float(values[k]) for k in ('loss_d','loss_g','loss_gan','prior_regularization','penalty')}))
    og,od,policy=trainer.opt_g,trainer.opt_d,trainer.policy
   else:
    for i in range(pairs):
     values=up.advance(ctx,b,g,r,u,execution_limit=20000)
     losses.append(('app',i,{k:values[v] for k,v in dict(loss_d='D_total',loss_g='G_total',loss_gan='G_adversarial',prior_regularization='prior_regularization_unweighted',penalty='D_penalty').items()}))
   states.append(dict(models=[x.state_dict() for x in (G,D,prior)],optimizers=[og.state_dict(),od.state_dict()],rng={k:v.get_state() for k,v in g.items()},numpy_rng=r.bit_generator.state,torch_rng=torch.get_rng_state(),latent_rng=policy.latent_generator.get_state(),penalty_rng=policy.penalty_generator.get_state(),eval_rng=policy.eval_generator.get_state(),noise_rng=policy.noise_generator.get_state(),completed_steps=policy.completed_steps))
  for i in range(pairs):
   assert losses[i][2]==losses[i+pairs][2],(hyper,seed,i,losses[i],losses[i+pairs])
  assert fingerprint(states[0])==fingerprint(states[1]),(hyper,seed,'actual native trainer state differs')
  return dict(seed=seed,hyperparameters=hyper,optimizer_pairs_per_loop=pairs,exact_models_optimizers_rng_and_update_clock=True,exact_all_losses_and_penalty=True,from_scratch=True)
def main():
 defaults=dict(generator_lr=.012,critic_lr=.018,particle_lr=.03,reg_coeff=1.,reg_kappa=1.,prior_reg=.01)
 half=dict(defaults,generator_lr=.006,critic_lr=.009,particle_lr=.015,reg_kappa=.5)
 rows=[run(h,seed) for h in (defaults,half) for seed in (0,1)]
 assert not torch.cuda.is_initialized()
 files=[Path(__file__).resolve(),*list((ROOT/'analysis/faithful097/model').glob('*.py')),ROOT/'analysis/duration096/native_fixture.py',Path(sys.modules['particlegan.training'].__file__)]
 report=dict(status='EXACT_RELEASED_GANTRAINER_UPDATE_PARITY_ON_FULLWIDTH_CONDITIONAL_FIXTURE_PASS',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),native_version='0.9.7',rows=rows,actual_cases=len(rows),actual_optimizer_pairs_per_implementation=sum(q['optimizer_pairs_per_loop'] for q in rows),fullwidth_genes=2000,prior_regularizer_domain='all512rows',explicit_bound_critic_thermal=True,native_package_modified=False,conditional_facade_for_fixture_only=True,real_ALZ_data_read=False,CUDA_initialized=False,GPU_allocation_created=False,source_SHA256={str(p):sha(p) for p in files})
 path=ROOT/'analysis/faithful097/native_trainer_parity_CPU.json';assert not path.exists();path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ('source_SHA256','rows')}))
if __name__=='__main__':main()
