"""Actual historical CUDA checkpoint, native CPU model/optimizer audit only."""
import datetime,hashlib,json,os,sys
from pathlib import Path
ROOT=Path('/ml2/alz');HERE=Path(__file__).resolve().parent
PROJECT=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark')
SCIENCE=Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/core-comparison-20261103')
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
sys.path[:0]=list(map(str,[HERE,ROOT/'model',ROOT/'released_rate_runtime',SCIENCE,PROJECT,
    PROJECT/'current_base_20261029/runtime/lib/python3.12/site-packages',PROJECT/'.venv312/lib/python3.12/site-packages']))
import torch
torch.set_num_threads(1)
from particlegan.policy import _validate_optimizer_state
from state_fingerprint import fingerprint
import released_update as update
import released_adapter as adapter
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
cp=ROOT/'runs/CAL_GPU0_V1_network_rate_half_v1_f0_pg_s0/fit/final.pt'
assert sha(cp)=='7b312554f6b4f8c26da7bc1eeb22573b9d4045e628b5698212cc0bd2cf014be3'
q=torch.load(cp,map_location='cpu',weights_only=False)
assert q['status']=='completed' and q['policy']['device']=='cuda:0' and q['completed_updates']==20000
h={'generator_lr':.006,'critic_lr':.009,'particle_lr':.03,'reg_coeff':1.,'reg_kappa':1.,'prior_reg':.01}
update.configure_hyperparameters(h)
b,*_=adapter.build(torch.ones(2000),torch.ones(2000),'cpu',0,0)
assert q['recipe']==q['policy']['recipe']==b[0].to_dict()
floating=0
for name,module in zip(('generator','critic','prior'),b[1:4]):
    module.load_state_dict(q['policy']['models'][name],strict=True)
    assert fingerprint(module.state_dict())==fingerprint(q['policy']['models'][name])
    for v in module.state_dict().values():
        if isinstance(v,torch.Tensor) and v.is_floating_point():
            assert torch.isfinite(v).all();floating+=v.numel()
assert floating==371177
for optimizer,saved in zip(b[4:6],q['policy']['optimizers']):
    _validate_optimizer_state(optimizer,saved);optimizer.load_state_dict(saved)
    assert fingerprint(optimizer.state_dict())==fingerprint(saved)
assert not torch.cuda.is_initialized()
result={'status':'PASSED_ACTUAL_HISTORICAL_NATIVE_CUDA_CHECKPOINT_CPU_MODEL_OPTIMIZER_COMPATIBILITY_AUDIT',
    'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'historical_checkpoint_path':str(cp),'historical_checkpoint_SHA256':sha(cp),
    'historical_hyperparameters':h,'new_tuning_training':False,'fullwidth_model_floating_values':floating,
    'all_models_strict_loading_and_byte_fingerprint_verified':True,'both_native_optimizer_validation_loading_and_byte_fingerprint_verified':True,
    'original_CUDA_device_metadata_unchanged':'cuda:0','CUDA_initialized':False,
    'source_SHA256':{str(p):sha(p) for p in (Path(__file__),HERE/'released_adapter.py',HERE/'released_update.py',ROOT/'released_rate_runtime/state_fingerprint.py')},
    'scope':'Read-only actual old20k network-rate negative checkpoint. Public native model/optimizer component loading on CPU; no optimizer steps, no CUDA policy relabel, no new tuning quality claim.'}
(HERE/'historical_native_checkpoint_CPU_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
