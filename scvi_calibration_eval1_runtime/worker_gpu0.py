"""Matched native scVI fit and calibration; exact live gate before imports."""
import argparse,json,os,socket,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import binding as b
def validate_fit(receipt,auth):
 b.require(receipt['status']=='completed' and receipt['method']=='scvi','Native scVI fit required')
 b.require(receipt['completed_epochs']==receipt['fixed_endpoint']==auth['fixed_endpoint']==400,'Complete400epoch endpoint required')
 b.require(receipt['fold']==0 and receipt['seed']==0 and receipt['scvi_tools']=='1.3.3','Fixed comparator identity differs')
 b.require(receipt['training_cells']==14762 and len(receipt['training_donors'])==33,'Optimization population differs')
 b.require(receipt['validation_cells']==0 and receipt['early_stopping'] is False and receipt['train_indices_exactly_all_optimization_cells'] is True,'Selection-free all-optimization fitting required')
 b.require(receipt['architecture']==auth['scvi_architecture'] and receipt['training_plan']==auth['scvi_training_plan'],'Native architecture or plan differs')
 b.require(receipt['query_keys']==['stage'] and receipt['cache_manifest_SHA256']==b.sha(Path(auth['cache_dir'])/'manifest.json'),'Information/cache mismatch')
def run_scientific(auth,admission,layout):
 b.require(auth['operation']=='evaluate_matched_scvi_calibration','One comparator operation only')
 b.require(Path(sys.executable).absolute()==b.runtime_path(layout,b.cell(auth)).absolute() and sys.version_info[:2]==(3,11),'Exact scVI scientific interpreter required')
 sys.path.extend(map(str,b.native_site_roots(b.cell(auth),auth['files_SHA256'])))
 sys.path[:0]=[str(layout.science),str(b.HERE.parent)]
 def deny(self,address):raise RuntimeError('Scientific worker network disabled')
 socket.socket.connect=deny;socket.socket.connect_ex=deny
 os.environ.update(WANDB_MODE='disabled',WANDB_DISABLED='true',OTEL_SDK_DISABLED='true',DO_NOT_TRACK='1',HF_HUB_OFFLINE='1',JAX_PLATFORMS='cpu',CUBLAS_WORKSPACE_CONFIG=':4096:8')
 import torch,scvi
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest')
 torch.cuda.set_per_process_memory_fraction((3*2**30)/torch.cuda.get_device_properties(0).total_memory,0)
 from reload_model import load_native
 from sampling_context import Context
 from generation import iter_scvi
 from calibration import evaluate,average,dump
 from state_fingerprint import fingerprint
 out=Path(auth['output_dir']);out.mkdir(parents=True,exist_ok=False)
 end=time.monotonic()+max(0,min(admission['authorization_expiry'],admission['ledger_started_unix']+auth['cap_seconds'])-time.time()-120)
 try:
  model,receipt,original=load_native(auth['cache_dir'],auth['parent_fit_path'],'cuda:0');validate_fit(receipt,auth)
  reload_receipt={'status':'NATIVE_SCVI_PUBLIC_CUDA_RELOAD_EXACT','model_state_fingerprint':original,'scvi_version':scvi.__version__,'torch_version':torch.__version__,'package_origin':str(Path(scvi.__file__).resolve()),'completed_epochs':400,'checkpoint_selected_with_calibration':False}
  dump(out/'native_reload_receipt.json',reload_receipt)
  ctx=Context(Path(auth['cache_dir']),'cuda:0')
  def iterator(schedule,c,s):
   for item in iter_scvi(ctx,model,schedule,c,s):
    if time.monotonic()>=end:raise TimeoutError('Finite allocation ending; no partial score')
    yield item
  rows=[evaluate(ctx,'scvi_matched_stage_only',0,rep,iterator,out/f'r{rep}') for rep in (0,1)]
  result=average(rows);result.update(status='completed',method='scvi_matched_stage_only',operation=auth['operation'],fold=0,completed_epochs=400,fixed_endpoint=400,scvi_version=scvi.__version__,scvi_architecture=auth['scvi_architecture'],scvi_training_plan=auth['scvi_training_plan'],query_information=['stage'],evaluation_used=False,integer_count_contract_passed=True,allocation_id=admission['allocation_id'],fit_receipt_path=str(Path(auth['parent_fit_path'])/'fit_receipt.json'),fit_receipt_SHA256=b.sha(Path(auth['parent_fit_path'])/'fit_receipt.json'),native_reload_path=str(out/'native_reload_receipt.json'),native_reload_SHA256=b.sha(out/'native_reload_receipt.json'),comparator_protocol_SHA256=auth['comparator_protocol_SHA256'],parent_fit_path=auth['parent_fit_path'],parent_authorization_SHA256=auth['parent_authorization_SHA256'],source_model_training_attempt='CAL_GPU0_V1_matched_scvi_calibration_f0_scvi_s0',new_model_updates=0,real_bag_replacements=0,automatic_successor=False,SOTA_claim=False)
  dump(out/'completed.json',result)
 except BaseException as exc:
  dump(out/'failure.json',{'status':'incomplete','error':repr(exc),'automatic_retry':False});raise
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--authorization',type=Path,required=True);parser.add_argument('--cpu-fixture-root',type=Path);args=parser.parse_args();layout=b.Layout.fixture(args.cpu_fixture_root) if args.cpu_fixture_root else b.NATIVE_LAYOUT
 auth,admission=b.require_live(args.authorization,layout)
 import project_runner_gpu0
 project_runner_gpu0.controller_gate(b.read(layout.project/'project_gpu_budget.json'),auth)
 if layout.fixture_mode:
  b.require('torch' not in sys.modules and os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU fixture imported models or exposed CUDA')
  if auth.get('fixture_fail_exit'):return 7
  out=Path(auth['output_dir']);out.mkdir(parents=True,exist_ok=False)
  (out/'CPU_FIXTURE_COMPLETE.json').write_text(json.dumps({'status':'CPU_BINDING_FIXTURE_ONLY','allocation_id':auth['allocation_id'],'attempt_id':auth['attempt_id'],'model_imported':False,'CUDA_VISIBLE_DEVICES':os.environ['CUDA_VISIBLE_DEVICES']})+'\n');return 0
 return run_scientific(auth,admission,layout)
if __name__=='__main__':raise SystemExit(main())
