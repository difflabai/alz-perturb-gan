"""Fresh fixed/adaptive PG fit, exact native restore, calibration-only diagnostics."""
import argparse,json,os,socket,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import binding as b,preset_scope as scope

def validate_pg(q,auth):
 v=auth['cell']
 b.require(q['schema']==1 and q['method']=='pg' and q['native_recipe']=='bispf_optimized','Native PG checkpoint required')
 b.require(q['fold']==0 and q['seed']==v['seed'],'Fresh fold/seed differs')
 b.require(q['status']=='completed' and q['completed_updates']==q['fixed_endpoint']==q['policy']['completed_steps']==20000,'Full20000pairs required')
 b.require(q['particlegan_version']=='0.9.7' and q['penalty_options']=={'input_domain':'continuous'},'Release/input domain differs')
 b.require(q['released_hyperparameters']==auth['released_hyperparameters'] and q['variant_config']==scope.CFG,'Scalars/architecture differ')
 b.require(q['recipe']==q['policy']['recipe'] and q['recipe']['total_steps']==20000 and q['recipe']['ema_decay']==0,'Native horizon or nonEMA formulation differs')
 b.require(q['cache_manifest_SHA256']==b.sha(Path(auth['cache_dir'])/'manifest.json'),'Optimization cache differs')
 state=q.get('calibration_lr_controller')
 if auth['arm']=='fixed':b.require(state is None,'Fixed control must have no adaptive schedule')
 else:
  b.require(isinstance(state,dict) and state['metric']=='calibration_cell_energy_distance' and state['config']==scope.protocol()['controller'],'Wrong external controller')
  b.require([v['completed_updates'] for v in state['decisions']]==list(range(1000,20001,1000)),'Incomplete calibration observation history')

def scientific(auth,admission,layout):
 seed=auth['cell']['seed'];arm=auth['arm'];sys.path.extend(map(str,b.native_site_roots(auth['cell'],auth['files_SHA256'])))
 sys.path[:0]=list(map(str,[scope.MODEL_DIR,scope.CONTROLLER_DIR,scope.ROOT/'analysis/faithful097/model',scope.ROOT/'model',layout.science,scope.ROOT]))
 def deny(self,address):raise RuntimeError('Scientific worker network disabled')
 socket.socket.connect=deny;socket.socket.connect_ex=deny
 os.environ.update(HF_HUB_OFFLINE='1',WANDB_MODE='disabled',WANDB_DISABLED='true',OTEL_SDK_DISABLED='true',DO_NOT_TRACK='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
 import numpy as np,torch
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest')
 allocator=3*2**30;torch.cuda.set_per_process_memory_fraction(allocator/torch.cuda.get_device_properties(0).total_memory,0)
 import scalar_update as update,scalar_adapter as model
 from calibration_distance_lr import CalibrationDistanceLR,Config
 from calibration_scalar import CalibrationScalar,isolated_generation
 from calibration import average,dump,sha
 from generation import iter_pg,iter_resampling
 from endpoint_evaluation import evaluate
 from state_fingerprint import fingerprint
 assert Path(model.__file__).resolve()==scope.MODEL_DIR/'scalar_adapter.py'
 update.configure(scope.CFG);update.configure_hyperparameters(auth['released_hyperparameters'])
 out=Path(auth['output_dir']);out.mkdir(parents=True,exist_ok=False)
 end=time.monotonic()+max(0,min(admission['authorization_expiry'],admission['ledger_started_unix']+auth['cap_seconds'])-time.time()-120)
 try:
  ctx=model.Context(Path(auth['cache_dir']),'cuda:0');controller=CalibrationDistanceLR(Config()) if arm=='adaptive' else None
  bundle,checkpoint=model.fit(ctx,out/'fit',seed,end_monotonic=end,evaluator=CalibrationScalar(),controller=controller)
  q=torch.load(checkpoint,map_location='cpu',weights_only=False);validate_pg(q,auth)
  saved=fingerprint(q['policy']);bundle,*_=model.restore(ctx,q,controller_config=Config() if arm=='adaptive' else None)
  assert fingerprint(bundle[-1].state_dict())==saved
  dump(out/'native_restore.json',dict(status='EXACT_NATIVE_SCALAR_PG_CUDA_RESTORE_PASS',checkpoint=str(checkpoint),checkpoint_SHA256=sha(checkpoint),policy_fingerprint=saved,completed_updates=20000,arm=arm))
  def iterator(schedule,c,s):
   with isolated_generation(ctx,bundle[1],bundle[3]):
    for item in iter_pg(ctx,bundle[1],bundle[3],schedule,c,s):
     if time.monotonic()>=end:raise TimeoutError('Finite scalar calibration allocation ending')
     yield item
  records=[evaluate(ctx,'pg_scalarcal097',seed,rep,iterator,out/('r%d'%rep)) for rep in (0,1)]
  baselines=[evaluate(ctx,'hierarchical_resampling',seed,rep,lambda schedule,c,s:iter_resampling(ctx,schedule,c,s),out/('resampling_r%d'%rep)) for rep in (0,1)]
  curve=[json.loads(line) for line in (out/'fit/calibration_curve.jsonl').read_text().splitlines()]
  assert len(curve)==20 and curve[-1]['observation']['completed_updates']==20000
  assert np.allclose([row['cell_energy_distance'] for row in records],curve[-1]['observation']['replicate_values'],rtol=0,atol=1e-10),'Training observation/final fixed-stream distance differ'
  dump(out/'completed.json',dict(status='completed',metric_role='calibration_selection',endpoint=20000,new_model_updates=20000,initial_completed_updates=0,source_checkpoint=None,arm=arm,seed=seed,cell=auth['cell'],protocol_SHA256=sha(scope.PROTOCOL),hyperparameters=auth['released_hyperparameters'],metrics=average(records),replicate_metrics=records,baseline_metrics=average(baselines),baseline_replicate_metrics=baselines,checkpoint_path=str(checkpoint),checkpoint_SHA256=sha(checkpoint),policy_fingerprint=saved,from_scratch=True,EMA_decay=0,ParticleGAN_version='0.9.7',heldout_values_used=False,adaptive_decisions=controller.decisions if controller is not None else None,GPU_allocator_limit_bytes=allocator,GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(0),GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved(0)))
 except BaseException as exc:
  dump(out/'failure.json',dict(status='incomplete',error=repr(exc),automatic_retry=False,source_checkpoint=None,from_scratch=True));raise

def main():
 p=argparse.ArgumentParser();p.add_argument('--authorization',type=Path,required=True);p.add_argument('--cpu-fixture-root',type=Path);args=p.parse_args()
 layout=b.Layout.fixture(args.cpu_fixture_root) if args.cpu_fixture_root else b.NATIVE_LAYOUT
 auth,admission=b.require_live(args.authorization,layout)
 import project_runner_gpu0
 project_runner_gpu0.controller_gate(b.read(layout.project/'project_gpu_budget.json'),auth)
 if layout.fixture_mode:
  b.require('torch' not in sys.modules and os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU fixture exposed model/CUDA')
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
