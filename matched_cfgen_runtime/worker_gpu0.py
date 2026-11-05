"""Native matched CFGen fixed fit then fresh owned inference process."""
import argparse,json,os,sys,time,subprocess,gc
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import binding as b
def scientific(auth,admission,layout,scoring):
 b.require(auth['operation']=='train_matched_cfgen','One matched CFGen fit only')
 b.require(Path(sys.executable).absolute()==b.runtime_path(layout,b.cell(auth)).absolute() and sys.version_info[:2]==(3,11),'Exact native CFGen interpreter required')
 sys.path.extend(map(str,b.native_site_roots(b.cell(auth),auth['files_SHA256'])))
 sys.path[:0]=[str(b.HERE.parent/'analysis/matched_cfgen'),str(layout.science),str(b.HERE.parent)]
 os.environ.update(WANDB_MODE='disabled',WANDB_DISABLED='true',JAX_PLATFORMS='cpu',HF_HUB_OFFLINE='1',DO_NOT_TRACK='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
 from fit_adapter import fit,restore,read,dump,validate_complete,iter_counts,torch,NETWORK
 from state_fingerprint import fingerprint
 torch.set_float32_matmul_precision('highest');torch.cuda.set_per_process_memory_fraction((3*2**30)/torch.cuda.get_device_properties(0).total_memory,0)
 out=Path(auth['output_dir']);end=time.monotonic()+max(0,min(admission['authorization_expiry'],admission['ledger_started_unix']+auth['cap_seconds'])-time.time()-120)
 if not scoring:
  out.mkdir(parents=True,exist_ok=False);model,receipt=fit(Path(auth['cache_dir']),out/'fit',0,'cuda:0',end)
  validate_complete(receipt,auth['cache_dir'],Path(receipt['FM_checkpoint_path']))
  dump(out/'fit/progress.json',{'status':'fit_complete','phase':'FM','total_optimizer_updates':102600,'completed_AE_epochs':300,'completed_FM_epochs':1500})
  del model;gc.collect();torch.cuda.empty_cache();remaining=end-time.monotonic();assert remaining>1
  subprocess.run([sys.executable,'-I','-S','-B',str(Path(__file__).resolve()),'--authorization',auth['_authorization_path'],'--score'],check=True,timeout=remaining)
  assert read(out/'completed.json')['status']=='completed';return
 receipt=read(out/'fit/fit_receipt.json');checkpoint=Path(receipt['FM_checkpoint_path']);validate_complete(receipt,auth['cache_dir'],checkpoint)
 model=restore(checkpoint,'cuda:0');assert all(str(v.device)=='cuda:0' for v in model.state_dict().values())
 assert all(str(p.device)=='cuda:0' for f in model.feature_embeddings.values() for p in f.parameters())
 assert all(str(f.device)=='cuda:0' for f in model.feature_embeddings.values())
 assert fingerprint(model.encoder_model.state_dict())==receipt['encoder_fingerprint']
 native_state=fingerprint(model.state_dict());external_state=fingerprint({n:f.state_dict() for n,f in model.feature_embeddings.items()})
 dump(out/'native_reload_receipt.json',{'status':'NATIVE_CFGEN_FRESH_GPU_CHECKPOINT_RELOAD_VERIFIED','checkpoint_SHA256':b.sha(checkpoint),'registered_state_fingerprint':native_state,'external_state_fingerprint':external_state,'encoder_fingerprint':receipt['encoder_fingerprint'],'registered_tensor_devices':['cuda:0'],'external_tensor_devices':['cuda:0'],'external_feature_devices':[str(f.device) for f in model.feature_embeddings.values()]})
 from sampling_context import Context
 from calibration import evaluate,average
 ctx=Context(Path(auth['cache_dir']),'cuda:0')
 def iterator(schedule,c,s):
  for item in iter_counts(ctx,model,schedule,c,s):
   if time.monotonic()>=end:raise TimeoutError('Finite nativeCFGen generation ending; no partial score')
   yield item
 rows=[evaluate(ctx,'cfgen_matched_stage_only',0,rep,iterator,out/f'r{rep}') for rep in (0,1)];result=average(rows)
 assert not NETWORK and b.sha(checkpoint)==receipt['FM_checkpoint_SHA256']
 result.update(status='completed',method='cfgen_matched_stage_only',operation=auth['operation'],fold=0,fixed_endpoint=1800,AE_epochs=300,FM_epochs=1500,total_optimizer_updates=102600,new_model_updates=102600,CFGen_source_commit=auth['CFGen_source_commit'],integer_count_contract_passed=True,checkpoint_path=str(checkpoint),checkpoint_SHA256=b.sha(checkpoint),fit_receipt_path=str(out/'fit/fit_receipt.json'),fit_receipt_SHA256=b.sha(out/'fit/fit_receipt.json'),native_reload_path=str(out/'native_reload_receipt.json'),native_reload_SHA256=b.sha(out/'native_reload_receipt.json'),comparator_protocol_SHA256=auth['comparator_protocol_SHA256'],allocation_id=admission['allocation_id'],qualification_only=False,evaluation_used=False,query_information=['stage'],real_bag_replacements=0,automatic_successor=False,SOTA_claim=False)
 dump(out/'completed.json',result)
def main():
 p=argparse.ArgumentParser();p.add_argument('--authorization',type=Path,required=True);p.add_argument('--cpu-fixture-root',type=Path);p.add_argument('--score',action='store_true');args=p.parse_args();layout=b.Layout.fixture(args.cpu_fixture_root) if args.cpu_fixture_root else b.NATIVE_LAYOUT
 auth,admission=b.require_live(args.authorization,layout,allow_existing_output=args.score)
 import project_runner_gpu0
 project_runner_gpu0.controller_gate(b.read(layout.project/'project_gpu_budget.json'),auth)
 if layout.fixture_mode:
  assert not args.score and 'torch' not in sys.modules and os.environ.get('CUDA_VISIBLE_DEVICES')==''
  if auth.get('fixture_fail_exit'):return 7
  out=Path(auth['output_dir']);out.mkdir(parents=True,exist_ok=False);(out/'CPU_FIXTURE_COMPLETE.json').write_text(json.dumps({'status':'CPU_BINDING_FIXTURE_ONLY','allocation_id':auth['allocation_id'],'attempt_id':auth['attempt_id'],'model_imported':False,'CUDA_VISIBLE_DEVICES':os.environ['CUDA_VISIBLE_DEVICES']})+'\n');return 0
 auth['_authorization_path']=str(args.authorization.resolve())
 try:return scientific(auth,admission,layout,args.score)
 except BaseException as exc:
  out=Path(auth['output_dir'])
  if out.exists() and not args.score:(out/'failure.json').write_text(json.dumps({'status':'incomplete','error':repr(exc),'automatic_retry':False})+'\n')
  raise
if __name__=='__main__':raise SystemExit(main())
