"""Independently load both full native parent architectures; no fitting or CUDA."""
import datetime, hashlib, json, os, sys
from pathlib import Path
ROOT=Path('/ml2/alz');B=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark');S=Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/core-comparison-20261103')
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
sys.path[:0]=list(map(str,[S,B,B/'current_base_20261029/runtime/lib/python3.12/site-packages',B/'.venv312/lib/python3.12/site-packages']))
import torch
from particlegan import get_recipe
from pg_model import CountGenerator
from critic import VectorCritic
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
manifest_path=ROOT/'ensemble_parents.json';manifest=json.loads(manifest_path.read_text());rows=[]
for index,parent in enumerate(manifest['parents']):
    p=Path(parent['checkpoint_path']);assert sha(p)==parent['checkpoint_SHA256']
    receipt=json.loads(p.with_name('checkpoint_receipt.json').read_text())
    assert receipt['status']=='completed' and receipt['completed_updates']==20000 and receipt['SHA256']==sha(p)
    q=torch.load(p,map_location='cpu',weights_only=False)
    assert q['status']=='completed' and q['method']=='pg' and q['fold']==q['seed']==0 and q['completed_updates']==q['fixed_endpoint']==20000
    assert q['policy']['completed_steps']==20000 and q['cache_manifest_SHA256']==sha(S/'caches/fold0/manifest.json')
    rec=get_recipe(total_steps=20000,batch_size=32,num_particles=512,z_dim=8)
    assert q['recipe']==q['policy']['recipe']==rec.to_dict()
    variant=q.get('variant_config')
    if variant is None:
        assert index==0 and parent['checkpoint_SHA256']=='444c22a5829b79d8ed0cd300be255d9410916d6c2248f930cf4cc0f886744da2'
        variant={'z_dim':q['recipe']['z_dim'],'stage_balance':False,'cell_z_dim':8,'theta_init_scale':1.,'fourier_feature_scale':1.}
    assert variant==parent['variant_config']
    critic=VectorCritic(150,h=128)
    if index==1:
        assert q['learned_observation_config']==parent['learned_observation_config']
        critic=torch.nn.ModuleDict({'head':critic,'cell_encoder':torch.nn.Sequential(torch.nn.Linear(50,64),torch.nn.SiLU(),torch.nn.Linear(64,25),torch.nn.LayerNorm(25,eps=1e-5,elementwise_affine=False))})
    models={'generator':CountGenerator(torch.ones(2000),torch.ones(2000),zd=8,zc=8),'critic':critic,'prior':rec.make_prior(generator=torch.Generator().manual_seed(0))};rec.apply_prior_policy(models['prior'])
    assert set(q['policy']['models'])==set(models);count=0
    for name,module in models.items():
        expected=module.state_dict();saved=q['policy']['models'][name];assert saved.keys()==expected.keys()
        for key,v in saved.items():
            assert v.shape==expected[key].shape and v.dtype==expected[key].dtype
            if v.is_floating_point():assert torch.isfinite(v).all();count+=v.numel()
        module.load_state_dict(saved,strict=True)
    assert count==[371177,376066][index]
    rows.append({'name':parent['name'],'checkpoint_SHA256':sha(p),'finite_model_floating_values':count,'exact_official_recipe_and_complete_G_D_prior_load':True,'configuration_binding':'saved variant_config' if 'variant_config' in q else 'registered exact legacy PG8 checkpoint and existing default restore semantics; native replay still required'})
assert len(rows)==2 and not torch.cuda.is_initialized()
result={'status':'VERIFIED_COMPLETE_PURE_ENSEMBLE_PARENT_CHECKPOINTS','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'parent_manifest_SHA256':sha(manifest_path),'parents':rows,'CUDA_initialized':False,'new_model_updates':0,'scope':'Independent CPU strict generator/critic/prior state audits only. No optimizer/CUDA RNG restore claim or native control replay, fitting or GPU allocation.','source_SHA256':{str(Path(__file__).resolve()):sha(Path(__file__).resolve())}}
(ROOT/'analysis/ensemble_parent_checkpoint_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
