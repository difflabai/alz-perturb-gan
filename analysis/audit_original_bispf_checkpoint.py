"""CPU read-only audit of the completed native-recipe checkpoint and API declaration."""
import json,os,sys,hashlib,datetime
from pathlib import Path
ROOT=Path('/ml2/alz');B=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark')
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
sys.path[:0]=list(map(str,[B/'current_base_20261029/runtime/lib/python3.12/site-packages',B/'.venv312/lib/python3.12/site-packages']))
import torch
from particlegan import get_recipe
p=ROOT/'runs/CAL_GPU0_V1_original_bispf_v1_f0_pg_s0/fit/final.pt';receipt=json.loads(p.with_name('checkpoint_receipt.json').read_text());pin=hashlib.sha256(p.read_bytes()).hexdigest();assert receipt['status']=='completed' and receipt['completed_updates']==20000 and pin==receipt['SHA256']
q=torch.load(p,map_location='cpu',weights_only=False)
assert q['native_recipe']=='bispf' and q['method']=='pg' and q['status']=='completed' and q['fold']==q['seed']==0
assert q['completed_updates']==q['fixed_endpoint']==20000
expected=get_recipe('bispf',total_steps=20000,batch_size=32,num_particles=512,z_dim=8).to_dict();assert q['recipe']==expected
assert q['variant_config']==dict(z_dim=8,stage_balance=False,cell_z_dim=8,theta_init_scale=1.,fourier_feature_scale=1.)
count=0
def finite(value):
 global count
 if isinstance(value,torch.Tensor):
  assert value.device.type=='cpu'
  if value.is_floating_point():assert torch.isfinite(value).all();count+=value.numel()
 elif isinstance(value,dict):
  for val in value.values():finite(val)
 elif isinstance(value,(tuple,list)):
  for val in value:finite(val)
finite(q['policy']['models']);assert count>100000
assert not torch.cuda.is_initialized()
result={'status':'VERIFIED_COMPLETE_NATIVE_ORIGINAL_BISPF_CHECKPOINT','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checkpoint_SHA256':pin,'completed_updates':20000,'native_recipe_exactly_matches_official_API_factory':True,'resolved_recipe':expected,'variant_config':q['variant_config'],'finite_model_floating_point_values':count,'CUDA_initialized':False,'scope':'CPU read-only real final checkpoint audit; no fitting or GPU allocation.'}
(ROOT/'analysis/original_bispf_checkpoint_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='resolved_recipe'},indent=2))
