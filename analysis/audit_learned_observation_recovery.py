"""Preserve one native recovery and check encoder learning; no quality selection."""
import datetime,hashlib,io,json,os,sys
from pathlib import Path
ROOT=Path('/ml2/alz');B=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark');S=Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/core-comparison-20261103')
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
sys.path[:0]=list(map(str,[ROOT/'model',S,B,B/'current_base_20261029/runtime/lib/python3.12/site-packages',B/'.venv312/lib/python3.12/site-packages']))
import torch
from particlegan import get_recipe
from learned_cell_observation import OPTIONS,make_critic
from learned_pg_adapter import training_streams

options={**OPTIONS,'enabled':True};attempt='CAL_GPU0_V1_learned_obs_v1_f0_pg_s0'
p=ROOT/'runs'/attempt/'fit/recovery.pt';raw=p.read_bytes();q=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)
assert q['status']=='recovery' and q['completed_updates']>=1000 and q['fold']==q['seed']==0
assert q['learned_observation_config']==options and q['recipe']==get_recipe(total_steps=20000,batch_size=32,num_particles=512,z_dim=8).to_dict()
torch.manual_seed(training_streams(0,0)['critic_init']);initial=make_critic(options);trained=make_critic(options)
trained.load_state_dict(q['policy']['models']['critic'],strict=True)
delta=sum((trained.cell_encoder.state_dict()[key]-value).square().sum() for key,value in initial.cell_encoder.state_dict().items()).sqrt()
assert torch.isfinite(delta) and delta>0
x=torch.randn(3,11,50,generator=torch.Generator().manual_seed(20261103))
with torch.no_grad():
 a,b=initial.features(x),trained.features(x);assert torch.equal(a[:,:100],b[:,:100]);feature_delta=(a[:,100:]-b[:,100:]).norm();assert torch.isfinite(feature_delta) and feature_delta>1e-10
def finite(value):
 if isinstance(value,torch.Tensor):
  if value.is_floating_point():assert torch.isfinite(value).all()
 elif isinstance(value,dict):
  for v in value.values():finite(v)
 elif isinstance(value,(tuple,list)):
  for v in value:finite(v)
finite(q['policy']['models']);assert not torch.cuda.is_initialized()
archive=ROOT/'artifacts/native_recovery_audits'/attempt/f"step{q['completed_updates']}.pt";archive.parent.mkdir(parents=True,exist_ok=True)
if archive.exists():assert archive.read_bytes()==raw
else:archive.write_bytes(raw)
result={'status':'VERIFIED_NATIVE_LEARNED_OBSERVATION_RECOVERY_MECHANISM_ONLY','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed_updates':q['completed_updates'],
        'archived_snapshot':str(archive),'snapshot_SHA256':hashlib.sha256(raw).hexdigest(),'encoder_parameter_delta_from_initial_L2':float(delta),'synthetic_probe_learned50_feature_delta_L2':float(feature_delta),
        'original100_moment_coordinates_bitwise_unchanged':True,'all_model_tensors_finite':True,'native_recipe_exact_API_factory':True,'learned_observation_config':options,'CUDA_initialized':False,
        'source_SHA256':{str(Path(__file__).resolve()):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        'scope':'CPU read-only snapshot and synthetic3bag/11cell probe only. Confirms weights and learned observations changed; does not establish quality, causal benefit or an endpoint result. No optimizer update, held-out target, GPU allocation or restart.'}
(ROOT/'analysis/learned_observation_native_recovery_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
