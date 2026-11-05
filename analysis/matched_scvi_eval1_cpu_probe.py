"""Inspect the preserved400epoch model and qualify the one-argument API fix."""
import datetime,inspect,json,hashlib,os,sys
from pathlib import Path
from unittest.mock import patch
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
ROOT=Path('/ml2/alz');SCIENCE=Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/core-comparison-20261103')
sys.path[:0]=[str(ROOT/'scvi_calibration_eval1_runtime'),str(SCIENCE),str(ROOT)]
import torch,scvi
torch.set_num_threads(1);torch.set_num_interop_threads(1)
from reload_model import load_native
from scvi_adapter import load_final
from state_fingerprint import fingerprint
from scvi.model import _utils
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
parent=ROOT/'runs/CAL_GPU0_V1_matched_scvi_calibration_f0_scvi_s0/fit';cache=SCIENCE/'caches/fold0'
model,receipt,expected=load_native(cache,parent,'cpu')
original=load_final(cache,parent,device='cpu')
assert fingerprint(original.module.state_dict())==fingerprint(model.module.state_dict())==expected
assert inspect.signature(scvi.model.SCVI.load).parameters['device'].default=='auto'
class CounterfactualSingleGPU:
 _accelerator_flag='cuda';_devices_flag=[0]
with patch.object(_utils,'_AcceleratorConnector',return_value=CounterfactualSingleGPU()) as wire:
 _,_,device=_utils.parse_device_args(accelerator='gpu',devices='auto',return_device='torch',validate_single_device=True)
 wire.assert_called_once_with(accelerator='gpu',devices='auto')
 assert str(device)=='cuda:0'
assert not torch.cuda.is_initialized()
paths=[Path(__file__),ROOT/'scvi_calibration_eval1_runtime/reload_model.py',SCIENCE/'scvi_adapter.py',Path(inspect.getfile(scvi.model.SCVI.load)),Path(_utils.__file__),parent/'fit_receipt.json',parent/'model/model.pt',parent/'model/adata.h5ad']
q={'status':'PRESERVED_SCVI400_NATIVE_CPU_RELOAD_AND_AUTO_DEVICE_API_VERIFIED','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed_epochs':400,'model_state_fingerprint':expected,'native_CPU_reload_identical_to_preserved_model_and_original_CPU_loader':True,'public_load_default_device':'auto','private_single_visible_GPU_argument_wire':'gpu/auto resolves cuda:0; connector substituted only in isolated CPU fixture','actual_GPU_reload_proven':False,'CUDA_initialized':torch.cuda.is_initialized(),'new_model_updates':0,'parent_model_files_unchanged':True,'quality_score_computed':False,'source_SHA256':{str(p):sha(p) for p in paths},'scope':'Native real400epoch saved-modelCPUreload and exact weights/registry inspection; private auto-device argument wire only forGPU. Actual inference requires separate finite900s oneuse allocation within originalparentexpiry. No refit or new seed/checkpoint selection.'}
destination=ROOT/'analysis/matched_scvi/eval1_preserved_model_reload_CPU.json';assert not destination.exists();destination.write_text(json.dumps(q,indent=2)+'\n');print(json.dumps({k:v for k,v in q.items() if k not in ('source_SHA256','scope')},indent=2))
