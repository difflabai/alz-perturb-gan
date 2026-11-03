"""Read one atomic owned recovery snapshot; record mechanism checks, no quality score."""
import datetime,hashlib,io,json,os,sys
from pathlib import Path
ROOT=Path('/ml2/alz');B=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark')
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
sys.path[:0]=list(map(str,[B/'current_base_20261029/runtime/lib/python3.12/site-packages',B/'.venv312/lib/python3.12/site-packages']))
import torch
from particlegan import get_recipe
p=ROOT/'runs/CAL_GPU0_V1_semantic_v1_f0_pg_s0/fit/recovery.pt';raw=p.read_bytes();q=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False);probe=json.loads((ROOT/'analysis/semantic_fitting_gradient_probe.json').read_text())
assert q['completed_updates']>=1000 and q['status']=='recovery' and q['recipe']==get_recipe(total_steps=20000,batch_size=32,num_particles=512,z_dim=8).to_dict()
assert q['semantic_observation']=={'enabled':True,'temperature':1.,'epsilon':1e-4,'classifier_state_SHA256':probe['classifier_state_SHA256']} and q['classifier_file_SHA256']==probe['classifier_file_SHA256']
matrices=[]
def visit(value):
 if isinstance(value,torch.Tensor):
  if value.is_floating_point():assert torch.isfinite(value).all()
  if tuple(value.shape)==(128,163):matrices.append(value)
 elif isinstance(value,dict):
  for v in value.values():visit(v)
 elif isinstance(value,(tuple,list)):
  for v in value:visit(v)
visit(q['policy']['models']);assert len(matrices)==1;semantic=matrices[0][:,150:156];assert semantic.abs().sum()>0 and not torch.cuda.is_initialized()
result={'status':'VERIFIED_NATIVE_SEMANTIC_RECOVERY_MECHANISM_ONLY','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed_updates':q['completed_updates'],'snapshot_SHA256':hashlib.sha256(raw).hexdigest(),'semantic_observation':q['semantic_observation'],'classifier_file_SHA256':q['classifier_file_SHA256'],'semantic_critic_columns_L2':float(semantic.norm()),'native_recipe_matches_official_factory':True,'all_model_tensors_finite':True,'scope':'CPU read-only snapshot, no optimizer update, no generation, no calibration/evaluation score. Incomplete model cannot be ranked.'}
(ROOT/'analysis/semantic_native_recovery_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
