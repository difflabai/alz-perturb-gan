"""CPU read-only audit of the completed native-recipe checkpoint and API declaration."""
import json,os,sys,hashlib,datetime
from pathlib import Path
ROOT=Path('/ml2/alz');B=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark');S=Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/core-comparison-20261103')
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
sys.path[:0]=list(map(str,[S,B,B/'current_base_20261029/runtime/lib/python3.12/site-packages',B/'.venv312/lib/python3.12/site-packages']))
import torch
from particlegan import get_recipe
from pg_model import CountGenerator
from critic import VectorCritic
p=ROOT/'runs/CAL_GPU0_V1_learned_obs_v1_f0_pg_s0/fit/final.pt';receipt=json.loads(p.with_name('checkpoint_receipt.json').read_text());pin=hashlib.sha256(p.read_bytes()).hexdigest();assert receipt['status']=='completed' and receipt['completed_updates']==20000 and pin==receipt['SHA256']
q=torch.load(p,map_location='cpu',weights_only=False)
assert q['learned_observation_config']=={'enabled':True,'cell_hidden':64,'cell_features':25,'feature_scale':.2,'pool_epsilon':1e-6,'layernorm_epsilon':1e-5} and q['method']=='pg' and q['status']=='completed' and q['fold']==q['seed']==0
assert q['completed_updates']==q['fixed_endpoint']==20000
expected=get_recipe(total_steps=20000,batch_size=32,num_particles=512,z_dim=8).to_dict();assert q['recipe']==expected
assert q['variant_config']==dict(z_dim=8,stage_balance=False,cell_z_dim=8,theta_init_scale=1.,fourier_feature_scale=1.)
models={'generator':CountGenerator(torch.ones(2000),torch.ones(2000),zd=8,zc=8),
        'critic':torch.nn.ModuleDict({'head':VectorCritic(150,h=128),'cell_encoder':torch.nn.Sequential(torch.nn.Linear(50,64),torch.nn.SiLU(),torch.nn.Linear(64,25),torch.nn.LayerNorm(25,eps=1e-5,elementwise_affine=False))}),
        'prior':get_recipe(total_steps=20000,batch_size=32,num_particles=512,z_dim=8).make_prior(generator=torch.Generator().manual_seed(0))}
get_recipe(total_steps=20000,batch_size=32,num_particles=512,z_dim=8).apply_prior_policy(models['prior'])
assert set(q['policy']['models'])==set(models)
expected_count=0
for name,module in models.items():
 expected_state=module.state_dict();saved=q['policy']['models'][name];assert saved.keys()==expected_state.keys()
 for key,value in saved.items():
  assert value.shape==expected_state[key].shape and value.dtype==expected_state[key].dtype
  if value.is_floating_point():expected_count+=value.numel()
 module.load_state_dict(saved,strict=True)
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
finite(q['policy']['models']);assert count==expected_count==376066
assert not torch.cuda.is_initialized()
result={'status':'VERIFIED_COMPLETE_PURE_LEARNED_OBSERVATION_CHECKPOINT','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checkpoint_SHA256':pin,'completed_updates':20000,'native_recipe_exactly_matches_official_API_factory':True,'all_model_keys_shapes_and_dtypes_match_declared_architecture':True,'resolved_recipe':expected,'variant_config':q['variant_config'],'learned_observation_config':q['learned_observation_config'],'finite_model_floating_point_values':count,'CUDA_initialized':False,'scope':'CPU read-only real final checkpoint audit. Strict full generator/critic/prior state loading; no optimizer update or CUDA RNG restore, fitting or GPU allocation.'}
(ROOT/'analysis/learned_observation_checkpoint_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='resolved_recipe'},indent=2))
