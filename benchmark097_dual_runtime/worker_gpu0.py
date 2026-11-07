"""One frozen fresh PG/scVI/CFGen fit, native restore and original scoring."""
import argparse,gc,os,socket,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import binding as b,preset_scope as scope

def validate_pg(q,auth):
 v=auth['cell']
 b.require(q['schema']==1 and q['method']=='pg' and q['native_recipe']=='bispf_optimized','Native PG checkpoint required')
 b.require(q['fold']==v['fold'] and q['seed']==v['seed'],'Checkpoint fold/seed differs')
 b.require(q['status']=='completed' and q['completed_updates']==q['fixed_endpoint']==q['policy']['completed_steps']==20000,'Full20k final policy required')
 b.require(q['cache_manifest_SHA256']==b.sha(Path(auth['cache_dir'])/'manifest.json'),'Checkpoint optimization cache differs')
 b.require(q['particlegan_version']=='0.9.7' and q['penalty_options']=={'input_domain':'continuous'},'Native release/domain differs')
 b.require(q['released_hyperparameters']==auth['released_hyperparameters'] and q['variant_config']==scope.CFG,'Frozen PG configuration differs')
 b.require(q['recipe']==q['policy']['recipe'] and q['recipe']['total_steps']==20000 and q['recipe']['ema_decay']==0 and q['recipe'].get('generator_gradient_response','none')=='none','PG formulation or schedule changed')
 b.require(q['usage']['steps']==20000 and int(q['usage']['draws'].sum())==640000 and int(q['usage']['unsampled_moved_steps'].sum())==0,'Native particle ownership/clock differs')

def scientific(auth,admission,layout):
 v=b.cell(auth);method=v['method'];seed=v['seed']
 b.require(Path(sys.executable).absolute()==b.runtime_path(layout,v).absolute(),'Scientific interpreter differs')
 scope.validate_authorization(auth)
 sys.path.extend(map(str,b.native_site_roots(v,auth['files_SHA256'])))
 sys.path[:0]=[str(scope.MODEL_DIR),str(b.HERE.parent/'model'),str(layout.science),str(b.HERE.parent)]
 def deny(self,address):raise RuntimeError('Scientific worker network disabled')
 socket.socket.connect=deny;socket.socket.connect_ex=deny
 os.environ.update(HF_HUB_OFFLINE='1',WANDB_MODE='disabled',WANDB_DISABLED='true',OTEL_SDK_DISABLED='true',DO_NOT_TRACK='1',JAX_PLATFORMS='cpu',CUBLAS_WORKSPACE_CONFIG=':4096:8')
 if method=='cfgen':
  # The unchanged native target module sets thread/inter-op determinism before
  # tensor work. Do not call set_num_interop_threads twice in this process.
  sys.path.insert(0,str(scope.CFGEN_DIR))
  import fit_adapter as cfgen
  from native_reload import load_complete
  assert Path(cfgen.__file__).resolve()==scope.CFGEN_DIR/'fit_adapter.py'
  torch=cfgen.torch
 else:
  import torch
  torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 torch.set_float32_matmul_precision('highest')
 allocator=b.read(auth['policy_path'])['roles'][auth['role']]['gpu_bytes']
 assert allocator==3*2**30
 torch.cuda.set_per_process_memory_fraction(allocator/torch.cuda.get_device_properties(0).total_memory,0)
 from contracts import dump,sha
 from state_fingerprint import fingerprint
 from generation import iter_pg,iter_scvi,iter_resampling
 from scoring import evaluate
 out=Path(auth['output_dir']);out.mkdir(parents=True,exist_ok=False)
 end=time.monotonic()+max(0,min(admission['authorization_expiry'],admission['ledger_started_unix']+auth['cap_seconds'])-time.time()-120)
 cache=Path(auth['cache_dir'])
 try:
  if method=='pg':
   import released_update as update,released_adapter as adapter
   from release_env import VERSION
   assert VERSION=='0.9.7' and Path(adapter.__file__).resolve()==scope.MODEL_DIR/'released_adapter.py'
   update.configure(scope.CFG);update.configure_hyperparameters(auth['released_hyperparameters'])
   ctx=adapter.Context(cache,'cuda:0');bundle,checkpoint=adapter.fit(ctx,out/'fit',seed,end_monotonic=end)
   q=torch.load(checkpoint,map_location='cpu',weights_only=False);validate_pg(q,auth)
   assert q['training_streams']==adapter.training_streams(v['fold'],seed) and q['recipe']==bundle[0].to_dict()
   saved=fingerprint(q['policy']);bundle[-1].load_state_dict(q['policy']);assert fingerprint(bundle[-1].state_dict())==saved
   native=dict(status='EXACT_NATIVE097_FINAL_POLICY_CUDA_RESTORE_PASS',checkpoint=str(checkpoint),checkpoint_SHA256=sha(checkpoint),policy_fingerprint=saved,completed_updates=20000)
   iterator=lambda schedule,c,s:iter_pg(ctx,bundle[1],bundle[3],schedule,c,s)
  elif method=='scvi':
   from scvi_adapter import fit,ARCHITECTURE,PLAN
   from native_scvi_reload import load_native
   from sampling_context import Context
   fit_dir=out/'fit';model,receipt=fit(cache,fit_dir,seed,end_monotonic=end);del model;gc.collect()
   model,receipt,saved=load_native(cache,fit_dir,'cuda:0')
   assert receipt['seed']==seed and receipt['fold']==v['fold'] and receipt['completed_epochs']==400
   assert receipt['architecture']==ARCHITECTURE and receipt['training_plan']==PLAN and receipt['validation_cells']==0 and receipt['early_stopping'] is False
   ctx=Context(cache,'cuda:0');checkpoint=fit_dir/'model/model.pt'
   native=dict(status='EXACT_NATIVE_SCVI_CUDA_RESTORE_PASS',checkpoint=str(checkpoint),checkpoint_SHA256=sha(checkpoint),model_fingerprint=saved,completed_epochs=400)
   iterator=lambda schedule,c,s:iter_scvi(ctx,model,schedule,c,s)
  else:
   from sampling_context import Context
   fit_dir=out/'fit';model,receipt=cfgen.fit(cache,fit_dir,seed,'cuda:0',end);del model;gc.collect()
   model,receipt,saved=load_complete(cache,fit_dir,seed,'cuda:0')
   assert receipt['fold']==v['fold'] and receipt['total_optimizer_updates']==auth['new_model_updates']
   ctx=Context(cache,'cuda:0');checkpoint=fit_dir/'FM/final.ckpt'
   native=dict(status='EXACT_NATIVE_CFGEN_CUDA_RESTORE_PASS',checkpoint=str(checkpoint),**saved)
   iterator=lambda schedule,c,s:cfgen.iter_counts(ctx,model,schedule,c,s)
  dump(out/'native_restore.json',native)
  def finite(factory):
   def run(schedule,c,s):
    for item in factory(schedule,c,s):
     if time.monotonic()>=end:raise TimeoutError('Finite benchmark allocation ending; preserve incomplete outcome')
     yield item
   return run
  for rep in (0,1):
   from retained_pcs import retain
   evaluate(ctx,method,seed,rep,finite(retain(iterator,out/f'evaluation_r{rep}')),out/f'evaluation_r{rep}')
   evaluate(ctx,'hierarchical_resampling',seed,rep,finite(lambda schedule,c,s:iter_resampling(ctx,schedule,c,s)),out/f'resampling_r{rep}')
  evaluate(ctx,method,seed,0,finite(iterator),out/'stage_swap_r0',stage_swap=True)
  dump(out/'completed.json',dict(status='completed',metric_role='final_evaluation',method=method,fold=v['fold'],seed=seed,
       endpoint=auth['fixed_endpoint'],generation_replicates=2,cache_manifest_SHA256=sha(cache/'manifest.json'),
       benchmark_protocol_SHA256=sha(scope.PROTOCOL),reuse_completed_model=False,from_scratch=True,initial_completed_updates=0,
       new_model_updates=auth['new_model_updates'],new_model_update_unit=auth['new_model_update_unit'],
       checkpoint_path=str(checkpoint),checkpoint_SHA256=sha(checkpoint),native_restore_SHA256=sha(out/'native_restore.json'),
       PG_configuration=auth['released_hyperparameters'] if method=='pg' else None,real_bag_replacements=0,automatic_successor=False,
       GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(0),GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved(0),GPU_allocator_limit_bytes=allocator))
 except BaseException as exc:
  dump(out/'failure.json',dict(status='incomplete',error=repr(exc),automatic_retry=False,from_scratch=True,initial_completed_updates=0));raise

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--authorization',type=Path,required=True);parser.add_argument('--cpu-fixture-root',type=Path)
 args=parser.parse_args();layout=b.Layout.fixture(args.cpu_fixture_root) if args.cpu_fixture_root else b.NATIVE_LAYOUT
 auth,admission=b.require_live(args.authorization,layout)
 import project_runner_gpu0
 project_runner_gpu0.controller_gate(b.read(layout.project/'project_gpu_budget.json'),auth)
 if layout.fixture_mode:
  b.require('torch' not in sys.modules and os.environ.get('CUDA_VISIBLE_DEVICES')=='','Fixture exposed CUDA/model')
  if auth.get('fixture_fail_exit'):return 7
  out=Path(auth['output_dir']);out.mkdir(parents=True,exist_ok=False)
  (out/'CPU_FIXTURE_COMPLETE.json').write_text(__import__('json').dumps(dict(status='CPU_BINDING_FIXTURE_ONLY',allocation_id=auth['allocation_id'],attempt_id=auth['attempt_id'],model_imported=False,CUDA_VISIBLE_DEVICES=''))+'\n')
  return 0
 return scientific(auth,admission,layout)
if __name__=='__main__':raise SystemExit(main())
