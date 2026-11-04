"""Archive one frozen-parent recovery; verify original API, rates and independent seed streams."""
import argparse, datetime, hashlib, io, json, os, sys
from pathlib import Path
ROOT=Path('/ml2/alz');B=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark');S=Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/core-comparison-20261103')
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
sys.path[:0]=list(map(str,[S,B,B/'current_base_20261029/runtime/lib/python3.12/site-packages',B/'.venv312/lib/python3.12/site-packages']))
import torch
import numpy as np
from particlegan import get_recipe
from pg_model import CountGenerator
from critic import VectorCritic

parser=argparse.ArgumentParser();parser.add_argument('--attempt',required=True);parser.add_argument('--parent',choices=['pg8','learned'],required=True);args=parser.parse_args()
attempt=args.attempt
auth=json.loads((ROOT/'artifacts/allocations'/attempt/'authorization.json').read_text());seed=auth['cell']['seed'];assert auth['operation']=='confirm_pure_ensemble' and seed in (1,2)
sys.path[:0]=[str(ROOT/'confirmation_runtime')];import ensemble_confirmation_scope as scope
scope.validate_authorization(auth)
p=ROOT/'runs'/attempt/'fit'/args.parent/'recovery.pt';raw=p.read_bytes()
q=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)
assert q['status']=='recovery' and 1000<=q['completed_updates']<20000 and q['fold']==0 and q['seed']==seed
assert q['variant_config']==scope.frozen()['seed0_candidate']['variant_config']
names=['generator_init','critic_init','prior_init','policy_latent','policy_penalty','policy_eval','policy_noise','cell','count','gamma_global','donor_sampler']
streams={k:int(c.generate_state(1,dtype=np.uint64)[0])%(2**63-16) for k,c in zip(names,np.random.SeedSequence([20261103,0,seed]).spawn(len(names)))}
assert q['training_streams']==streams
rec=get_recipe(total_steps=20000,batch_size=32,num_particles=512,z_dim=8)
assert q['recipe']==q['policy']['recipe']==rec.to_dict()
assert q['policy']['completed_steps']==q['completed_updates']
expected_lrs=[[rec.lr,rec.lr*rec.prior_lr_mult],[rec.lr*rec.d_lr_mult]]
assert q['policy']['initial_lrs']==expected_lrs
actual_lrs=[[g['lr'] for g in opt['param_groups']] for opt in q['policy']['optimizers']]
assert actual_lrs==expected_lrs
models={'generator':CountGenerator(torch.ones(2000),torch.ones(2000),zd=8,zc=8),
        'critic':VectorCritic(150,h=128),
        'prior':rec.make_prior(generator=torch.Generator().manual_seed(0))}
rec.apply_prior_policy(models['prior'])
if args.parent=='learned':
    assert q['learned_observation_config']=={'enabled':True,'cell_hidden':64,'cell_features':25,'feature_scale':.2,'pool_epsilon':1e-6,'layernorm_epsilon':1e-5}
    models['critic']=torch.nn.ModuleDict({'head':models['critic'],'cell_encoder':torch.nn.Sequential(torch.nn.Linear(50,64),torch.nn.SiLU(),torch.nn.Linear(64,25),torch.nn.LayerNorm(25,eps=1e-5,elementwise_affine=False))})
assert set(q['policy']['models'])==set(models)
count=0
for name,module in models.items():
    saved=q['policy']['models'][name];expected=module.state_dict();assert saved.keys()==expected.keys()
    for key,value in saved.items():
        assert value.device.type=='cpu' and value.shape==expected[key].shape and value.dtype==expected[key].dtype
        if value.is_floating_point():assert torch.isfinite(value).all();count+=value.numel()
    module.load_state_dict(saved,strict=True)
assert count=={'pg8':371177,'learned':376066}[args.parent] and not torch.cuda.is_initialized()
archive=ROOT/'artifacts/native_recovery_audits'/attempt/args.parent/f"step{q['completed_updates']}.pt"
archive.parent.mkdir(parents=True,exist_ok=True)
if archive.exists():assert archive.read_bytes()==raw
else:archive.write_bytes(raw)
result={'status':'VERIFIED_FROZEN_ENSEMBLE_PARENT_RECOVERY_STATE_ONLY','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'completed_updates':q['completed_updates'],'archived_snapshot':str(archive),'snapshot_SHA256':hashlib.sha256(raw).hexdigest(),
        'attempt':attempt,'parent':args.parent,'training_seed':seed,'independent_seed_streams_verified':True,'training_streams':streams,'native_recipe_exact_API_factory':True,
        'saved_initial_and_actual_optimizer_rates_match':True,'actual_optimizer_rates':actual_lrs,
        'strict_complete_G_D_prior_model_load_verified':True,'finite_model_floating_values':count,'CUDA_initialized':False,
        'scope':'CPU read-only atomic recovery inspection. No optimizer update or restore, CUDA RNG restore, new fitting, sampling, held-out targets, quality score or GPU allocation. Does not establish improvement or native continuation qualification.',
        'source_SHA256':{str(Path(__file__).resolve()):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
destination=ROOT/'analysis'/(attempt+'-'+args.parent+'-recovery-audit.json');assert not destination.exists()
destination.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
