"""Fresh fixed-LR theta trial, exact native restore and calibration endpoints."""
import argparse,json,os,socket,sys,time
from contextlib import contextmanager
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import binding as b,preset_scope as scope

def validate_pg(q,auth):
 v=auth['cell']
 b.require(q['schema']==1 and q['method']=='pg' and q['native_recipe']=='bispf_optimized','Exact native PG checkpoint required')
 b.require(q['fold']==0 and q['seed']==v['seed'],'Fresh fold/seed differs')
 b.require(q['status']=='completed' and q['completed_updates']==q['fixed_endpoint']==q['policy']['completed_steps']==20000,'Full20000 optimizer pairs required')
 b.require(q['particlegan_version']=='0.9.7' and q['penalty_options']=={'input_domain':'continuous'},'Native release/input domain differs')
 b.require(q['released_hyperparameters']==auth['released_hyperparameters'] and q['variant_config']==scope.configuration(auth['arm']),'Fixed scalars or theta initialization differ')
 b.require(q['recipe']==q['policy']['recipe'] and q['recipe']['total_steps']==20000 and q['recipe']['ema_decay']==0,'Native horizon or live weights differ')
 b.require(q['cache_manifest_SHA256']==b.sha(Path(auth['cache_dir'])/'manifest.json'),'Optimization cache differs')
 b.require('calibration_lr_controller' not in q,'This trial uses fixed t005 scalar rates')

@contextmanager
def inference_mode(ctx,G,P):
 import torch
 modules=list(G.modules())+list(P.modules());modes=[m.training for m in modules]
 devices=[] if ctx.device.type=='cpu' else [ctx.device.index if ctx.device.index is not None else torch.cuda.current_device()]
 try:
  with torch.random.fork_rng(devices=devices),torch.no_grad():
   G.eval();P.eval();yield
 finally:
  for module,mode in zip(modules,modes):module.training=mode

def scientific(auth,admission,layout):
 seed=auth['cell']['seed'];arm=auth['arm']
 sys.path.extend(map(str,b.native_site_roots(auth['cell'],auth['files_SHA256'])))
 sys.path[:0]=list(map(str,[scope.MODEL_DIR,scope.ROOT/'model',layout.science,scope.ROOT]))
 def deny(self,address):raise RuntimeError('Scientific worker network disabled')
 socket.socket.connect=deny;socket.socket.connect_ex=deny
 os.environ.update(HF_HUB_OFFLINE='1',WANDB_MODE='disabled',WANDB_DISABLED='true',OTEL_SDK_DISABLED='true',DO_NOT_TRACK='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
 import torch
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest')
 allocator=3*2**30;torch.cuda.set_per_process_memory_fraction(allocator/torch.cuda.get_device_properties(0).total_memory,0)
 import fresh_seed_adapter as model,released_update as update
 from calibration import average,dump,sha
 from endpoint_evaluation import evaluate
 from generation import iter_pg,iter_resampling
 from state_fingerprint import fingerprint
 assert Path(model.__file__).resolve()==scope.MODEL_DIR/'fresh_seed_adapter.py'
 update.configure(scope.configuration(arm));update.configure_hyperparameters(auth['released_hyperparameters'])
 out=Path(auth['output_dir']);out.mkdir(parents=True,exist_ok=False)
 end=time.monotonic()+max(0,min(admission['authorization_expiry'],admission['ledger_started_unix']+auth['cap_seconds'])-time.time()-120)
 try:
  ctx=model.Context(Path(auth['cache_dir']),'cuda:0')
  bundle,checkpoint=model.fit(ctx,out/'fit',seed,end_monotonic=end)
  q=torch.load(checkpoint,map_location='cpu',weights_only=False);validate_pg(q,auth)
  saved=fingerprint(q['policy']);bundle,*_=model.restore(ctx,q)
  assert fingerprint(bundle[-1].state_dict())==saved
  dump(out/'native_restore.json',dict(status='EXACT_NATIVE_DISPERSION_PG_CUDA_RESTORE_PASS',checkpoint=str(checkpoint),checkpoint_SHA256=sha(checkpoint),policy_fingerprint=saved,completed_updates=20000,arm=arm))
  def iterator(schedule,c,s):
   with inference_mode(ctx,bundle[1],bundle[3]):
    for item in iter_pg(ctx,bundle[1],bundle[3],schedule,c,s):
     if time.monotonic()>=end:raise TimeoutError('Finite theta calibration allocation ending')
     yield item
  records=[evaluate(ctx,'pg_dispersion097',seed,rep,iterator,out/('r%d'%rep)) for rep in (0,1)]
  baselines=[evaluate(ctx,'hierarchical_resampling',seed,rep,lambda schedule,c,s:iter_resampling(ctx,schedule,c,s),out/('resampling_r%d'%rep)) for rep in (0,1)]
  dump(out/'completed.json',dict(status='completed',metric_role='calibration_selection',endpoint=20000,new_model_updates=20000,initial_completed_updates=0,source_checkpoint=None,arm=arm,seed=seed,cell=auth['cell'],protocol_SHA256=sha(scope.PROTOCOL),hyperparameters=auth['released_hyperparameters'],variant_config=auth['variant_config'],metrics=average(records),replicate_metrics=records,baseline_metrics=average(baselines),baseline_replicate_metrics=baselines,checkpoint_path=str(checkpoint),checkpoint_SHA256=sha(checkpoint),policy_fingerprint=saved,from_scratch=True,training_finished_before_calibration=True,heldout_values_used=False,ParticleGAN_version='0.9.7',GPU_allocator_limit_bytes=allocator,GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(0),GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved(0)))
 except BaseException as exc:
  dump(out/'failure.json',dict(status='incomplete',error=repr(exc),automatic_retry=False));raise

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--authorization',type=Path,required=True);parser.add_argument('--cpu-fixture-root',type=Path);args=parser.parse_args()
 layout=b.Layout.fixture(args.cpu_fixture_root) if args.cpu_fixture_root else b.NATIVE_LAYOUT
 auth,admission=b.require_live(args.authorization,layout)
 import project_runner_gpu0
 project_runner_gpu0.controller_gate(b.read(layout.project/'project_gpu_budget.json'),auth)
 if layout.fixture_mode:
  b.require('torch' not in sys.modules and os.environ.get('CUDA_VISIBLE_DEVICES')=='','Fixture exposed CUDA/model')
  if auth.get('fixture_fail_exit'):return 7
  if auth.get('fixture_dual_barrier'):
   barrier=layout.fixture_root/'dual_barrier';barrier.mkdir(exist_ok=True)
   (barrier/(auth['allocation_id']+'.ready')).write_text('CPU_ONLY\n');deadline=time.monotonic()+5
   while len(list(barrier.glob('*.ready')))<2:
    if time.monotonic()>deadline:raise TimeoutError('Both confined CPU guards must overlap')
    time.sleep(.02)
  out=Path(auth['output_dir']);out.mkdir(parents=True,exist_ok=False)
  (out/'CPU_FIXTURE_COMPLETE.json').write_text(json.dumps(dict(status='CPU_BINDING_FIXTURE_ONLY',allocation_id=auth['allocation_id'],attempt_id=auth['attempt_id'],model_imported=False,CUDA_VISIBLE_DEVICES=''))+'\n');return 0
 return scientific(auth,admission,layout)
if __name__=='__main__':raise SystemExit(main())
