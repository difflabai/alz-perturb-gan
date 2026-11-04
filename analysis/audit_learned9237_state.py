"""CPU-only full-state audit of both terminal parents after desktop reboot."""
import datetime,hashlib,json,os,sys
from pathlib import Path
R=Path('/ml2/alz');B=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark');S=Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/core-comparison-20261103')
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
sys.path[:0]=list(map(str,[R,R/'model',B,S,B/'current_base_20261029/runtime/lib/python3.12/site-packages',B/'.venv312/lib/python3.12/site-packages']))
import numpy as np,torch
from particlegan import get_recipe
from pg_model import CountGenerator
from critic import VectorCritic
from learned_cell_observation import make_critic
from ensemble_scope import CFG,OBS
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
evidence=R/'analysis/learned9237_interruption.json';history=json.loads(evidence.read_text());rows=[]
streams={k:int(c.generate_state(1,dtype=np.uint64)[0])%(2**63-16) for k,c in zip(['generator_init','critic_init','prior_init','policy_latent','policy_penalty','policy_eval','policy_noise','cell','count','gamma_global','donor_sampler'],np.random.SeedSequence([20261103,0,1]).spawn(11))}
for index,(name,ref) in enumerate(history['checkpoints'].items()):
 p=Path(ref['path']);assert sha(p)==ref['SHA256'];q=torch.load(p,map_location='cpu',weights_only=False);n=ref['completed_updates']
 assert q['schema']==1 and q['method']=='pg' and q['fold']==0 and q['seed']==1
 assert q['status']==('completed' if index==0 else 'incomplete') and q['completed_updates']==q['policy']['completed_steps']==n
 assert q['fixed_endpoint']==20000 and q['cache_manifest_SHA256']==sha(S/'caches/fold0/manifest.json') and q['variant_config']==CFG
 rec=get_recipe(total_steps=20000,batch_size=32,num_particles=512,z_dim=8);assert q['recipe']==q['policy']['recipe']==rec.to_dict() and q['training_streams']==streams
 if index:assert q['learned_observation_config']==OBS
 else:assert 'learned_observation_config' not in q
 models={'generator':CountGenerator(torch.ones(2000),torch.ones(2000),zd=8,zc=8),'critic':make_critic(OBS) if index else VectorCritic(150,h=128),'prior':rec.make_prior(generator=torch.Generator().manual_seed(0))};rec.apply_prior_policy(models['prior']);count=0
 assert set(q['policy']['models'])==set(models)
 for key,m in models.items():
  saved=q['policy']['models'][key];expected=m.state_dict();assert saved.keys()==expected.keys()
  for k,v in saved.items():
   assert v.shape==expected[k].shape and v.dtype==expected[k].dtype
   if v.is_floating_point():assert torch.isfinite(v).all();count+=v.numel()
  m.load_state_dict(saved,strict=True)
 assert count==[371177,376066][index]
 inventory={'tensors':0,'finite_floating_values':0}
 def visit(x):
  if isinstance(x,torch.Tensor):
   assert x.device.type=='cpu';inventory['tensors']+=1
   if x.is_floating_point():assert torch.isfinite(x).all();inventory['finite_floating_values']+=x.numel()
  elif isinstance(x,np.ndarray):assert np.isfinite(x).all()
  elif isinstance(x,dict):
   for v in x.values():visit(v)
  elif isinstance(x,(tuple,list)):
   for v in x:visit(v)
 visit(q)
 assert len(q['policy']['optimizers'])==2 and [[v['lr'] for v in o['param_groups']] for o in q['policy']['optimizers']]==q['policy']['initial_lrs']==[[.012,.03],[.018000000000000002]]
 assert all(all(k in o for k in ['state','param_groups','dualnorm']) for o in q['policy']['optimizers'])
 assert set(q['generators'])=={'cell','count'} and all(v.dtype==torch.uint8 and v.numel()==16 for v in q['generators'].values())
 assert q['torch_cpu_rng'].numel()==5056 and q['torch_device_rng'].numel()==16 and q['usage']['steps']==n and torch.equal(q['policy']['cpu_rng'],q['torch_cpu_rng'])
 rows.append({**ref,'name':name,'strict_G_D_prior_load_verified':True,'finite_model_values':count,'full_state_inventory':inventory,'native_recipe_and_seed_streams_verified':True,'optimizer_states_and_rates_verified':True,'CPU_and_CUDA_rng_bytes_present':True})
assert not torch.cuda.is_initialized()
result={'status':'CPU_FULL_STATE_VERIFIED_PG8_COMPLETED_AND_LEARNED9237','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'parents':rows,'total_completed_updates':29237,'remaining_learned_updates':10763,'CUDA_initialized':False,'native_CUDA_restoration_verified':False,'source_SHA256':{str(Path(__file__).resolve()):sha(__file__),str(evidence):sha(evidence)},'scope':'CPU full-state and exact native model/recipe/seed/rate/RNG audit only. No updates, generation or quality. Completed PG8 retained; learned9237 requires new same-GPU initial-state and two-next-update replay verification before remaining10763 training.'}
p=R/'analysis/learned9237_full_state_audit.json';assert not p.exists();p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
