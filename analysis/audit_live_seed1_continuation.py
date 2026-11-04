"""Read-only CPU inventory and strict complete model/optimizer finiteness audit."""
import datetime,hashlib,json,os,sys
from pathlib import Path
ROOT=Path('/ml2/alz');B=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark');S=Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/core-comparison-20261103')
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
sys.path[:0]=list(map(str,[ROOT,S,B,B/'current_base_20261029/runtime/lib/python3.12/site-packages',B/'.venv312/lib/python3.12/site-packages']))
import numpy as np
import torch
from particlegan import get_recipe
from pg_model import CountGenerator
from critic import VectorCritic
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
a='CAL_GPU0_V1_pure_recover_bf27534c3a4250b2b4d2_f0_pg_s1';receipt_path=Path('/ml2/alz/artifacts/confirmation_recovery_lineage/CAL_GPU0_V1_pure_recover_bf27534c3a4250b2b4d2_f0_pg_s1/live-checkpoint-audit-input.json');interrupted=json.loads(receipt_path.read_text());p=Path(interrupted['archived_checkpoint']);assert sha(p)==interrupted['checkpoint_SHA256'];expected_step=interrupted['completed_updates']
q=torch.load(p,map_location='cpu',weights_only=False)
assert q['schema']==1 and q['status']=='recovery' and q['method']=='pg' and q['fold']==0 and q['seed']==1
assert q['completed_updates']==q['policy']['completed_steps']==expected_step and q['fixed_endpoint']==20000
assert q['cache_manifest_SHA256']==sha(S/'caches/fold0/manifest.json') and q['variant_config']==dict(z_dim=8,stage_balance=False,cell_z_dim=8,theta_init_scale=1.,fourier_feature_scale=1.)
rec=get_recipe(total_steps=20000,batch_size=32,num_particles=512,z_dim=8);assert q['recipe']==q['policy']['recipe']==rec.to_dict()
names=['generator_init','critic_init','prior_init','policy_latent','policy_penalty','policy_eval','policy_noise','cell','count','gamma_global','donor_sampler'];streams={k:int(c.generate_state(1,dtype=np.uint64)[0])%(2**63-16) for k,c in zip(names,np.random.SeedSequence([20261103,0,1]).spawn(len(names)))};assert q['training_streams']==streams
models={'generator':CountGenerator(torch.ones(2000),torch.ones(2000),zd=8,zc=8),'critic':VectorCritic(150,h=128),'prior':rec.make_prior(generator=torch.Generator().manual_seed(0))};rec.apply_prior_policy(models['prior']);count=0
assert set(q['policy']['models'])==set(models)
for name,module in models.items():
 saved=q['policy']['models'][name];expected=module.state_dict();assert saved.keys()==expected.keys()
 for key,v in saved.items():
  assert v.shape==expected[key].shape and v.dtype==expected[key].dtype
  if v.is_floating_point():assert torch.isfinite(v).all();count+=v.numel()
 module.load_state_dict(saved,strict=True)
assert count==371177
state_inventory={'tensor_count':0,'finite_floating_tensor_values':0,'numpy_array_count':0,'nonfinite_tensor_paths':[]}
def inventory(x,path):
 if isinstance(x,torch.Tensor):
  assert x.device.type=='cpu';state_inventory['tensor_count']+=1
  if x.is_floating_point():
   if not bool(torch.isfinite(x).all()):state_inventory['nonfinite_tensor_paths'].append(path)
   else:state_inventory['finite_floating_tensor_values']+=x.numel()
 elif isinstance(x,np.ndarray):state_inventory['numpy_array_count']+=1;assert np.isfinite(x).all()
 elif isinstance(x,dict):
  for k,v in x.items():inventory(v,path+'.'+str(k))
 elif isinstance(x,(tuple,list)):
  for i,v in enumerate(x):inventory(v,path+'.'+str(i))
inventory(q,'checkpoint');assert not state_inventory['nonfinite_tensor_paths']
assert len(q['policy']['optimizers'])==2
rates=[[v['lr'] for v in op['param_groups']] for op in q['policy']['optimizers']];assert rates==q['policy']['initial_lrs']==[[.012,.03],[.018000000000000002]]
for opt in q['policy']['optimizers']:assert all(k in opt for k in ['state','param_groups','dualnorm'])
assert set(q['generators'])=={'cell','count'} and all(v.dtype==torch.uint8 and v.numel()==16 for v in q['generators'].values())
assert q['torch_cpu_rng'].numel()==5056 and q['torch_device_rng'].numel()==16 and q['usage']['steps']==expected_step
assert torch.equal(q['policy']['cpu_rng'],q['torch_cpu_rng'])
assert not torch.cuda.is_initialized()
freeze=json.loads((ROOT/'analysis/pure_ensemble_confirmation_freeze.json').read_text());assert all(sha(path)==pin for path,pin in freeze['scientific_source_SHA256'].items())
result={'status':'VERIFIED_LIVE_NATIVE_PG8_CONTINUATION_FULL_STATE_CPU_ONLY','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'attempt':a,'training_seed':1,'completed_updates':expected_step,'fixed_endpoint':20000,'checkpoint_path':str(p),'checkpoint_SHA256':sha(p),
        'strict_complete_G_D_prior_load_verified':True,'finite_model_floating_values':count,'full_saved_state_inventory':state_inventory,'official_native_recipe_and_independent_seed_streams_verified':True,'optimizer_rate_and_state_blocks_present':True,'actual_optimizer_rates':rates,
        'explicit_CUDA_generator_states_retained_bytes':{k:v.numel() for k,v in q['generators'].items()},'global_CPU_and_CUDA_rng_states_present':True,'policy_and_snapshot_CPU_rng_match':True,
        'CUDA_initialized':False,'actual_CUDA_restore_verified':False,'scope':'Read-only full native saved-state inventory and strict model load on CPU. Explicit CUDA RNG bytes are preserved, not restored on CPU. This audit generated no updates/quality/nativeallocation and does not pause the active worker. Actual8942 CUDA restoration was separately qualified; this later saved checkpoint has not been used for any new allocation, and a future resume would require its own exact-state qualification.',
        'source_SHA256':{str(Path(__file__).resolve()):sha(Path(__file__).resolve()),str(receipt_path):sha(receipt_path)}}
dest=ROOT/'analysis/live_seed1_continuation_checkpoint_audit.json';assert not dest.exists();dest.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
