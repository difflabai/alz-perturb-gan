"""Native CFGen writer/fresh-reader qualification under one owned GPU lease."""
import argparse,json,os,socket,sys,time,subprocess,gc
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import binding as b
TARGET=Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/execution-control-20261103')
def scientific(auth,admission,layout,read_only):
 b.require(auth['operation']=='qualify_cfgen_gpu_reload' and auth['new_model_updates']==0,'qualification only')
 b.require(Path(sys.executable).absolute()==b.runtime_path(layout,b.cell(auth)).absolute() and sys.version_info[:2]==(3,11),'exact CFGen runtime required')
 sys.path.extend(map(str,b.native_site_roots(b.cell(auth),auth['files_SHA256'])))
 sys.path.insert(0,str(TARGET));os.environ.update(WANDB_MODE='disabled',WANDB_DISABLED='true',JAX_PLATFORMS='cpu',HF_HUB_OFFLINE='1',DO_NOT_TRACK='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
 from target_model import restore,sample,seed,torch,np,random,NETWORK,SOURCE
 from state_fingerprint import fingerprint
 torch.set_float32_matmul_precision('highest');torch.cuda.set_per_process_memory_fraction((3*2**30)/torch.cuda.get_device_properties(0).total_memory,0)
 assert Path(sys.modules['cfgen'].__file__).resolve().is_relative_to(SOURCE)
 checkpoint=Path(auth['checkpoint_path']);assert b.sha(checkpoint)==auth['checkpoint_SHA256']
 out=Path(auth['output_dir']);end=time.monotonic()+max(0,min(admission['authorization_expiry'],admission['ledger_started_unix']+900)-time.time()-120)
 def placed(model):
  assert all(str(v.device)=='cuda:0' for v in model.state_dict().values())
  assert all(str(p.device)=='cuda:0' for f in model.feature_embeddings.values() for p in f.parameters())
  assert all(str(f.device)=='cuda:0' for f in model.feature_embeddings.values())
  assert not model.training
  return {'registered_tensor_devices':['cuda:0'],'external_parameter_devices':['cuda:0'],'explicit_feature_devices':[str(f.device) for f in model.feature_embeddings.values()]}
 model=restore(checkpoint,'cuda:0');placement=placed(model)
 registered={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
 features={n:{k:v.detach().cpu().clone() for k,v in f.state_dict().items()} for n,f in model.feature_embeddings.items()}
 assert len(registered)==59
 expected=b.read(b.HERE.parent/'analysis/cfgen_device_explicit_CPU_replay_20261105.json')
 assert fingerprint(registered)==expected['registered_state_fingerprint'] and fingerprint(features)==expected['external_embedding_state_fingerprint']
 if read_only:
  saved=torch.load(out/'writer.pt',map_location='cpu',weights_only=False)
  assert fingerprint(registered)==saved['registered_fingerprint'] and fingerprint(features)==saved['external_fingerprint']
  rng=saved['RNG_before_sampling'];random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.set_rng_state(rng['torch_cpu']);torch.cuda.set_rng_state_all(rng['torch_cuda'])
  labels=saved['labels'].to('cuda:0');exposure=saved['log_exposure'].to('cuda:0')
 else:
  out.mkdir(parents=True,exist_ok=False);labels=torch.arange(4,device='cuda:0').repeat_interleave(64);exposure=torch.full((256,1),float(np.log(6000)),device='cuda:0');seed(77412)
  rng={'python':random.getstate(),'numpy':np.random.get_state(),'torch_cpu':torch.get_rng_state(),'torch_cuda':torch.cuda.get_rng_state_all()}
 start=time.monotonic();counts=sample(model,labels,exposure);seconds=time.monotonic()-start
 assert tuple(counts.shape)==(256,2000) and bool(torch.isfinite(counts).all() and (counts>=0).all() and (counts==counts.floor()).all())
 assert all(torch.equal(v,model.state_dict()[k].detach().cpu()) for k,v in registered.items())
 assert not NETWORK and b.sha(checkpoint)==auth['checkpoint_SHA256']
 record={'status':'NATIVE_CFGEN_FRESH_GPU_READER_REPLAY_PASS' if read_only else 'NATIVE_CFGEN_GPU_WRITER_PASS','registered_fingerprint':fingerprint(registered),'external_fingerprint':fingerprint(features),'count_fingerprint':fingerprint(counts),'placement':placement,'shape':list(counts.shape),'native_sampling_seconds':seconds,'native_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'native_peak_reserved_bytes':torch.cuda.max_memory_reserved(),'checkpoint_SHA256':b.sha(checkpoint),'new_model_updates':0,'quality_score_computed':False,'ODE_requested_output_timepoints':2,'ODE_internal_function_evaluations_measured':False}
 if read_only:
  assert torch.equal(counts.cpu(),saved['expected_counts']);(out/'reader.json').write_text(json.dumps(record,indent=2)+'\n');return
 torch.save({'registered_state':registered,'external_state':features,'registered_fingerprint':record['registered_fingerprint'],'external_fingerprint':record['external_fingerprint'],'expected_counts':counts.cpu(),'RNG_before_sampling':rng,'labels':labels.cpu(),'log_exposure':exposure.cpu()},out/'writer.pt')
 (out/'writer.json').write_text(json.dumps(record,indent=2)+'\n')
 del model,counts,registered,features,labels,exposure;gc.collect();torch.cuda.empty_cache()
 remaining=end-time.monotonic();assert remaining>1
 subprocess.run([sys.executable,'-I','-S','-B',str(Path(__file__).resolve()),'--authorization',str(auth['_authorization_path']),'--read'],check=True,timeout=remaining)
 reader=b.read(out/'reader.json');assert reader['count_fingerprint']==record['count_fingerprint'] and b.sha(checkpoint)==auth['checkpoint_SHA256']
 result={'status':'completed','operation':auth['operation'],'qualification_only':True,'quality_score_computed':False,'new_model_updates':0,'fresh_GPU_processes':2,'registered_external_state_and_counts_replayed':True,'checkpoint_unchanged':True,'writer_SHA256':b.sha(out/'writer.json'),'reader_SHA256':b.sha(out/'reader.json'),'payload_SHA256':b.sha(out/'writer.pt'),'source_checkpoint_SHA256':auth['checkpoint_SHA256'],'allocation_id':admission['allocation_id'],'comparator_protocol_SHA256':auth['comparator_protocol_SHA256'],'automatic_successor':False,'SOTA_claim':False}
 (out/'completed.json').write_text(json.dumps(result,indent=2)+'\n')
def main():
 p=argparse.ArgumentParser();p.add_argument('--authorization',type=Path,required=True);p.add_argument('--cpu-fixture-root',type=Path);p.add_argument('--read',action='store_true');args=p.parse_args();layout=b.Layout.fixture(args.cpu_fixture_root) if args.cpu_fixture_root else b.NATIVE_LAYOUT
 auth,admission=b.require_live(args.authorization,layout,allow_existing_output=args.read)
 import project_runner_gpu0
 project_runner_gpu0.controller_gate(b.read(layout.project/'project_gpu_budget.json'),auth)
 if layout.fixture_mode:
  assert not args.read and 'torch' not in sys.modules and os.environ.get('CUDA_VISIBLE_DEVICES')==''
  if auth.get('fixture_fail_exit'):return 7
  out=Path(auth['output_dir']);out.mkdir(parents=True,exist_ok=False);(out/'CPU_FIXTURE_COMPLETE.json').write_text(json.dumps({'status':'CPU_BINDING_FIXTURE_ONLY','allocation_id':auth['allocation_id'],'attempt_id':auth['attempt_id'],'model_imported':False,'CUDA_VISIBLE_DEVICES':os.environ['CUDA_VISIBLE_DEVICES']})+'\n');return 0
 auth['_authorization_path']=str(args.authorization.resolve())
 try:return scientific(auth,admission,layout,args.read)
 except BaseException as exc:
  out=Path(auth['output_dir'])
  if out.exists() and not args.read:(out/'failure.json').write_text(json.dumps({'status':'incomplete','error':repr(exc),'automatic_retry':False})+'\n')
  raise
if __name__=='__main__':raise SystemExit(main())
