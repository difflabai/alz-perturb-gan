"""Archive one atomic native recovery and inspect rates; no quality selection."""
import datetime, hashlib, io, json, os, sys
from pathlib import Path
ROOT=Path('/ml2/alz');B=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark');S=Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/core-comparison-20261103')
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
sys.path[:0]=list(map(str,[S,B,B/'current_base_20261029/runtime/lib/python3.12/site-packages',B/'.venv312/lib/python3.12/site-packages']))
import torch
from particlegan import get_recipe
from pg_model import CountGenerator
from critic import VectorCritic

attempt='CAL_GPU0_V1_network_rate_half_v1_f0_pg_s0'
p=ROOT/'runs'/attempt/'fit/recovery.pt';raw=p.read_bytes()
q=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)
assert q['status']=='recovery' and 1000<=q['completed_updates']<20000 and q['fold']==q['seed']==0
assert q['native_recipe']=='bispf_optimized' and q['network_rate_config']=={'lr':.006,'prior_lr_mult':5.}
rec=get_recipe(total_steps=20000,batch_size=32,num_particles=512,z_dim=8,lr=.006,prior_lr_mult=5.)
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
assert set(q['policy']['models'])==set(models)
count=0
for name,module in models.items():
    saved=q['policy']['models'][name];expected=module.state_dict();assert saved.keys()==expected.keys()
    for key,value in saved.items():
        assert value.device.type=='cpu' and value.shape==expected[key].shape and value.dtype==expected[key].dtype
        if value.is_floating_point():assert torch.isfinite(value).all();count+=value.numel()
    module.load_state_dict(saved,strict=True)
assert count==371177 and not torch.cuda.is_initialized()
archive=ROOT/'artifacts/native_recovery_audits'/attempt/f"step{q['completed_updates']}.pt"
archive.parent.mkdir(parents=True,exist_ok=True)
if archive.exists():assert archive.read_bytes()==raw
else:archive.write_bytes(raw)
result={'status':'VERIFIED_NATIVE_NETWORK_RATE_RECOVERY_STATE_ONLY','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'completed_updates':q['completed_updates'],'archived_snapshot':str(archive),'snapshot_SHA256':hashlib.sha256(raw).hexdigest(),
        'native_recipe_exact_API_factory':True,'network_rate_config':q['network_rate_config'],
        'saved_initial_and_actual_optimizer_rates_match':True,'actual_optimizer_rates':actual_lrs,
        'strict_complete_G_D_prior_model_load_verified':True,'finite_model_floating_values':count,'CUDA_initialized':False,
        'scope':'CPU read-only atomic recovery inspection. No optimizer update or restore, CUDA RNG restore, new fitting, sampling, held-out targets, quality score or GPU allocation. Does not establish improvement or native continuation qualification.',
        'source_SHA256':{str(Path(__file__).resolve()):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
(ROOT/'analysis/network_rate_native_recovery_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
