"""Independent CPU strict two-parent nativeAPI/state/seed audit; no CUDA restore."""
import argparse,datetime,hashlib,json,os,sys
from pathlib import Path
ROOT=Path('/ml2/alz');B=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark');S=Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/core-comparison-20261103')
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
sys.path[:0]=list(map(str,[Path(__file__).parent,S,B,ROOT,ROOT/'analysis',B/'current_base_20261029/runtime/lib/python3.12/site-packages',B/'.venv312/lib/python3.12/site-packages']))
import recovery_scope as scope
from aggregate_validation import verified_deployment
import numpy as np
import torch
from particlegan import get_recipe
from pg_model import CountGenerator
from critic import VectorCritic

def audit(attempt):
 dep,auth,candidate=verified_deployment(attempt);scope.validate_authorization(auth);reference=scope.verify_reference(candidate)
 seed=auth['cell']['seed'];rows=[];f=scope.frozen();manifest=scope.read(ROOT/'ensemble_parents.json')
 names=['generator_init','critic_init','prior_init','policy_latent','policy_penalty','policy_eval','policy_noise','cell','count','gamma_global','donor_sampler']
 streams={k:int(c.generate_state(1,dtype=np.uint64)[0])%(2**63-16) for k,c in zip(names,np.random.SeedSequence([20261103,0,seed]).spawn(len(names)))}
 for index,parent in enumerate(reference['parents']):
  p=Path(parent['checkpoint_path']);assert scope.sha(p)==parent['checkpoint_SHA256']
  q=torch.load(p,map_location='cpu',weights_only=False)
  assert q['schema']==1 and q['status']=='completed' and q['method']=='pg' and q['fold']==0 and q['seed']==seed
  assert q['completed_updates']==q['fixed_endpoint']==q['policy']['completed_steps']==20000
  assert q['cache_manifest_SHA256']==scope.sha(Path(auth['cache_dir'])/'manifest.json') and q['training_streams']==streams
  rec=get_recipe(total_steps=20000,batch_size=32,num_particles=512,z_dim=8)
  assert q['recipe']==q['policy']['recipe']==rec.to_dict()
  original=manifest['parents'][index];assert scope.sha(original['checkpoint_path'])==original['checkpoint_SHA256']
  seed0=torch.load(original['checkpoint_path'],map_location='cpu',weights_only=False)
  assert q['recipe']==seed0['recipe'] and q['variant_config']==original['variant_config']
  critic=VectorCritic(150,h=128)
  if index==1:
   assert q['learned_observation_config']==original['learned_observation_config']
   critic=torch.nn.ModuleDict({'head':critic,'cell_encoder':torch.nn.Sequential(torch.nn.Linear(50,64),torch.nn.SiLU(),torch.nn.Linear(64,25),torch.nn.LayerNorm(25,eps=1e-5,elementwise_affine=False))})
  else:assert 'learned_observation_config' not in q
  models={'generator':CountGenerator(torch.ones(2000),torch.ones(2000),zd=8,zc=8),'critic':critic,'prior':rec.make_prior(generator=torch.Generator().manual_seed(0))};rec.apply_prior_policy(models['prior'])
  assert set(q['policy']['models'])==set(models);count=0
  for name,module in models.items():
   expected=module.state_dict();saved=q['policy']['models'][name];assert saved.keys()==expected.keys()
   for key,v in saved.items():
    assert v.shape==expected[key].shape and v.dtype==expected[key].dtype
    if v.is_floating_point():assert torch.isfinite(v).all();count+=v.numel()
   module.load_state_dict(saved,strict=True)
  assert count==[371177,376066][index]
  assert any(not torch.equal(q['policy']['models']['generator'][key],seed0['policy']['models']['generator'][key]) for key in q['policy']['models']['generator']),'Not a seed0 checkpoint copy'
  rows.append({'name':parent['name'],'checkpoint_path':str(p),'checkpoint_SHA256':scope.sha(p),'finite_model_floating_values':count,'seed_streams':streams,'seed0_checkpoint_SHA256':original['checkpoint_SHA256']})
 assert not torch.cuda.is_initialized()
 result={'status':'VERIFIED_EXACT_RECOVERED_OR_FROZEN_SEED2_CHECKPOINTS','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'attempt':attempt,'training_seed':seed,'checkpoint_SHA256':candidate['checkpoint_SHA256'],'original_confirmation_freeze_SHA256':scope.lineage()[0]['original_confirmation_freeze_SHA256'],'recovery_protocol_SHA256':scope.sha(scope.PROTOCOL),
         'seed0_checkpoint_SHA256':f['seed0_candidate']['checkpoint_SHA256'],'parents':rows,'recipe_exactly_matches_frozen_seed0':True,
         'complete_model_keys_shapes_dtypes_finite_verified':True,'policy_update_count_and_seed_streams_verified':True,'CUDA_initialized':False,
         'scope':'CPU strict full G/D/prior state, officialrecipe, fixed20k EACH and independent documented training streams. No fitting or optimizer/CUDA RNG restoration claim.',
         'source_SHA256':{str(Path(__file__).resolve()):scope.sha(Path(__file__).resolve())}}
 dest=ROOT/'analysis'/(attempt+'-checkpoint-audit.json');assert not dest.exists();dest.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--attempt',required=True);audit(parser.parse_args().attempt)
