"""CPU-only synthetic full-2000gene native parity and evaluator isolation."""
import ast,copy,datetime,hashlib,json,os,random,sys,tempfile,unittest
from pathlib import Path
ROOT=Path('/ml2/alz');HERE=Path(__file__).resolve().parent
P=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark');S=Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/core-comparison-20261103')
sys.path[:0]=list(map(str,[HERE/'model',ROOT/'analysis/adaptive_lr097',ROOT/'analysis/faithful097/model',ROOT/'analysis/duration096',ROOT/'refinement097_runtime',ROOT/'model',ROOT,S,P,P/'current_base_20261029/runtime/lib/python3.12/site-packages',P/'.venv312/lib/python3.12/site-packages']))
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
import torch,numpy as np,scalar_adapter as app,scalar_update as up
from calibration_distance_lr import CalibrationDistanceLR,Config
from calibration_scalar import CalibrationScalar
from native_fixture import context
from stage_law import StageLaw
from state_fingerprint import fingerprint
from particlegan.training import GANTrainer
from pg_model import critic_features
from release_env import PENALTY_OPTIONS
HYPER=dict(generator_lr=.012,critic_lr=.018,particle_lr=.06,reg_coeff=1.,reg_kappa=1.,prior_reg=.01)
up.configure_hyperparameters(HYPER)
CFG=Config(interval=1,warmup=0,window=1,patience=1,cooldown=0)
class ConditionedCritic(torch.nn.Module):
 def __init__(self,base):super().__init__();self.base=base;self.condition=None
 def forward(self,x):return self.base(x,self.condition)
def fixture(cache):
 ctx=context(cache,fullwidth=True);ctx.__class__=app.Context;ctx.synthetic=True
 ctx.fitlibs=ctx.X.sum(-1);ctx.law=StageLaw(ctx.fitc,ctx.by,ctx.fitlibs.numpy())
 return ctx
def checkpoint(ctx,b,g,r,u,seed,streams):return app.snapshot(ctx,b,g,r,u,seed,streams,'synthetic')
def module_modes(b):return [m.training for model in b[1:4] for m in model.modules()]
def extra_rng():return dict(numpy_global=np.random.get_state(),python=random.getstate())
QUALIFIED_FILES=[Path(__file__),*sorted((HERE/'model').glob('*.py')),ROOT/'analysis/adaptive_lr097/calibration_distance_lr.py',ROOT/'analysis/faithful097/model/released_update.py',ROOT/'analysis/faithful097/model/released_adapter.py',ROOT/'analysis/faithful097/model/release_env.py',ROOT/'analysis/duration096/native_fixture.py',ROOT/'test_variant.py',Path(sys.modules['particlegan.training'].__file__),Path(sys.modules['particlegan.policy'].__file__),Path(sys.modules['pg_model'].__file__),Path(sys.modules['critic'].__file__),Path(sys.modules['stage_law'].__file__),Path(sys.modules['evaluation'].__file__),Path(sys.modules['calibration'].__file__),Path(sys.modules['state_fingerprint'].__file__)]
START_SOURCE_SHA256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in QUALIFIED_FILES}
ROWS=[]
class Integration(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.cache=Path(self.tmp.name);(self.cache/'manifest.json').write_text('{}');self.ctx=fixture(self.cache)
 def tearDown(self):
  self.tmp.cleanup();self.assertFalse(torch.cuda.is_initialized())
 def test_update_math_source_unchanged(self):
  source=(HERE/'model/scalar_update.py').read_text()
  source=source.replace("assert [x['lr'] for x in og.param_groups]==[rec.lr*policy.schedule(step,rec)[0],(rec.lr*rec.prior_lr_mult)*policy.schedule(step,rec)[1]]", "assert [x['lr'] for x in og.param_groups]==[rec.lr,rec.lr*rec.prior_lr_mult]")
  source=source.replace("assert [x['lr'] for x in od.param_groups]==[(rec.lr*rec.d_lr_mult)*policy.schedule(step,rec)[0]]", "assert [x['lr'] for x in od.param_groups]==[rec.lr*rec.d_lr_mult]")
  self.assertEqual(source,(ROOT/'analysis/faithful097/model/released_update.py').read_text())
  old=ast.parse((ROOT/'analysis/faithful097/model/released_adapter.py').read_text());new=ast.parse((HERE/'model/scalar_adapter.py').read_text())
  pick=lambda tree,name:ast.dump(next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name),include_attributes=False)
  self.assertEqual(pick(old,'Context'),pick(new,'Context'))
 def test_all_fresh_registered_seeds_fullwidth_native_parity(self):
  for seed in (5,6,7):
   states=[];all_losses=[]
   for native in (False,True):
    c=CalibrationDistanceLR(CFG);b,g,r,u,streams=app.build(self.ctx.mu,self.ctx.theta,'cpu',0,seed,controller=c)
    rec,G,D,prior,og,od,pen,loss,reg,policy=b
    self.assertEqual(policy.completed_steps,0);self.assertFalse(og.state);self.assertFalse(od.state)
    self.assertIsNone(policy.controller);self.assertIsNone(policy.lr_settle)
    losses=[];rates=[]
    if native:
     facade=ConditionedCritic(D)
     trainer=GANTrainer(rec,G,facade,prior=prior,seed=seed,max_steps=20000,serve_after_step=False,penalty_options=PENALTY_OPTIONS,**{name:torch.Generator().manual_seed(streams['policy_'+key]) for name,key in [('latent_generator','latent'),('penalty_generator','penalty'),('eval_generator','eval'),('noise_generator','noise')]})
     trainer.policy.schedule=c;state={};calls=[0]
     def generation(model,z):
      if calls[0]%2:
       real,cond,lib=self.ctx.batch(r);state.update(real=real,c=cond,lib=lib);facade.condition=cond
      calls[0]+=1
      rate=model.rates(z,state['c'],128,state['lib'],g['cell']);counts=torch.poisson(rate.detach(),generator=g['count'])
      return up.bag_features(critic_features(counts,rate,self.ctx.mean,self.ctx.sd)@self.ctx.W)
     trainer.policy.generation=generation
    for i in range(5):
     if native:
      real,cond,lib=self.ctx.batch(r);state.update(real=real,c=cond,lib=lib);facade.condition=cond
      values=trainer.step(real,generator_real=lambda:state['real'],collect_stats=i%250==0)
      losses.append({k:float(values[k]) for k in ('loss_d','loss_g','loss_gan','prior_regularization','penalty')})
      og,od,policy=trainer.opt_g,trainer.opt_d,trainer.policy
     else:
      values=up.advance(self.ctx,b,g,r,u,execution_limit=20000)
      losses.append({k:values[v] for k,v in dict(loss_d='D_total',loss_g='G_total',loss_gan='G_adversarial',prior_regularization='prior_regularization_unweighted',penalty='D_penalty').items()})
     rates.append([og.param_groups[0]['lr'],og.param_groups[1]['lr'],od.param_groups[0]['lr']])
     c.observe(i+1,1.,split_role='calibration')
    states.append(dict(models=[x.state_dict() for x in (G,D,prior)],optimizers=[og.state_dict(),od.state_dict()],rng={k:v.get_state() for k,v in g.items()},numpy_rng=r.bit_generator.state,torch_rng=torch.get_rng_state(),latent_rng=policy.latent_generator.get_state(),penalty_rng=policy.penalty_generator.get_state(),eval_rng=policy.eval_generator.get_state(),noise_rng=policy.noise_generator.get_state(),completed_steps=policy.completed_steps,controller=c.state_dict()))
    all_losses.append(losses)
    self.assertEqual([row[0] for row in rates],[.012,.012,.006,.003,.0015])
   self.assertEqual(all_losses[0],all_losses[1]);self.assertEqual(fingerprint(states[0]),fingerprint(states[1]))
   ROWS.append(dict(seed=seed,optimizer_pairs_per_implementation=5,exact_native_losses_models_optimizer_moments_and_all_RNG=True,all_three_LR_groups_scaled=True,all512_SBNP_rows=True,fresh_initialization=True))
 def test_fullwidth_checkpoint_exact_future_after_LR_changes(self):
  seed=5;c=CalibrationDistanceLR(CFG);b,g,r,u,streams=app.build(self.ctx.mu,self.ctx.theta,'cpu',0,seed,controller=c)
  def step(b,g,r,u):
   up.advance(self.ctx,b,g,r,u,execution_limit=20000);b[-1].calibration_lr_controller.observe(b[-1].completed_steps,1.,split_role='calibration')
  for _ in range(3):step(b,g,r,u)
  saved=checkpoint(self.ctx,b,g,r,u,seed,streams);saved['status']='recovery'
  for _ in range(2):step(b,g,r,u)
  expected=fingerprint(checkpoint(self.ctx,b,g,r,u,seed,streams))
  b,g,r,u,streams=app.restore(self.ctx,saved,controller_config=CFG)
  for _ in range(2):step(b,g,r,u)
  self.assertEqual(expected,fingerprint(checkpoint(self.ctx,b,g,r,u,seed,streams)))
  with self.assertRaises(ValueError):app.restore(self.ctx,saved)
 def test_synthetic_count_evaluation_repeatable_preserves_all_training_state(self):
  seed=6;b,g,r,u,streams=app.build(self.ctx.mu,self.ctx.theta,'cpu',0,seed)
  b[1].train();b[1].out.eval();b[3].train()
  metric=CalibrationScalar(synthetic_target=np.random.default_rng(44).normal(size=(1500,50)),bags=1)
  before=fingerprint(dict(snapshot=checkpoint(self.ctx,b,g,r,u,seed,streams),modes=module_modes(b),extra_rng=extra_rng()))
  first=metric(self.ctx,b,seed,completed_updates=0);second=metric(self.ctx,b,seed,completed_updates=0)
  self.assertEqual(first,second);self.assertEqual(first['cells_per_replicate'],1600)
  after=fingerprint(dict(snapshot=checkpoint(self.ctx,b,g,r,u,seed,streams),modes=module_modes(b),extra_rng=extra_rng()))
  self.assertEqual(before,after)
  project=self.ctx.project
  def fail(x):raise RuntimeError('fixture interrupted during generation')
  self.ctx.project=fail
  with self.assertRaisesRegex(RuntimeError,'fixture interrupted'):metric(self.ctx,b,seed,completed_updates=0)
  self.ctx.project=project
  with self.assertRaises(TimeoutError):metric(self.ctx,b,seed,completed_updates=0,end_monotonic=0.)
  self.assertEqual(before,fingerprint(dict(snapshot=checkpoint(self.ctx,b,g,r,u,seed,streams),modes=module_modes(b),extra_rng=extra_rng())))
 def test_full200bag_count_horizon_on_synthetic_CPU(self):
  seed=7;b,g,r,u,streams=app.build(self.ctx.mu,self.ctx.theta,'cpu',0,seed)
  metric=CalibrationScalar(synthetic_target=np.random.default_rng(45).normal(size=(1500,50)),bags=200)
  before=fingerprint(dict(snapshot=checkpoint(self.ctx,b,g,r,u,seed,streams),modes=module_modes(b),extra_rng=extra_rng()))
  actual=metric(self.ctx,b,seed,completed_updates=0)
  self.assertEqual(actual['cells_per_replicate'],320000);self.assertEqual(actual['generation_replicates'],2)
  self.assertEqual(before,fingerprint(dict(snapshot=checkpoint(self.ctx,b,g,r,u,seed,streams),modes=module_modes(b),extra_rng=extra_rng())))
 def test_unregistered_identity_and_production_shortcuts_rejected(self):
  for fold,seed in ((0,4),(1,5),(0,True),(0,8)):
   with self.assertRaises(ValueError):app.training_streams(fold,seed)
  with self.assertRaises(ValueError):CalibrationScalar(bags=1)
  b,g,r,u,streams=app.build(self.ctx.mu,self.ctx.theta,'cpu',0,5)
  self.ctx.synthetic=False
  metric=CalibrationScalar(synthetic_target=np.zeros((1500,50)),bags=1)
  with self.assertRaises(ValueError):metric(self.ctx,b,5,completed_updates=0)

def main():
 started=datetime.datetime.now(datetime.timezone.utc)
 result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Integration))
 if not result.wasSuccessful():raise SystemExit(1)
 report=dict(status='FULL2000GENE_SYNTHETIC_CPU_NATIVE_SCHEDULE_AND_EVALUATOR_QUALIFICATION_PASS',started_UTC=started.isoformat(),UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),tests_run=result.testsRun,native_parity_cases=ROWS,fullwidth_genes=2000,parity_actual_optimizer_pairs_total=30,checkpoint_actual_optimizer_pairs_total=7,calibration_generation='one full200bag two-replicate640000cell measurement, plus two repeated two-replicate scalar measurements with1bag400cells/stage; an exception after another generated chunk; synthetic targets only',full200bag320000cell_per_replicate_generation_on_synthetic_CPU_qualified=True,production_GPU_calibration_not_yet_executed=True,real_ALZ_calibration_values_read=False,real_ALZ_adaptive_fits=0,CUDA_initialized=torch.cuda.is_initialized(),new_GPU_allocations=0,live45_source_or_settings_modified=False,full_goal_complete=False,SOTA_claim=False)
 end_pins={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in QUALIFIED_FILES}
 assert START_SOURCE_SHA256==end_pins,'source changed during qualification'
 report['source_SHA256']=end_pins
 report['qualified_sources_unchanged_during_execution']=True
 (HERE/'ALZ_integration_CPU_qualification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
