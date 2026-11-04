"""Serving-only native KA2 saved EMA modules; parent training state stays intact."""
import hashlib,json
import torch
from particlegan import get_recipe
import ka2_adapter as base
import ka2_update as update
CFG={'z_dim':8,'stage_balance':False,'cell_z_dim':8,'theta_init_scale':1.,'fourier_feature_scale':1.}
READOUT={'weights':'native_saved_generator_and_prior_EMA','ema_decay':.995,'training_serve_average':0.,'output_noise':False}

def fingerprint(state):
    h=hashlib.sha256()
    for key in sorted(state):
        value=state[key]
        if not isinstance(value,torch.Tensor):raise ValueError('Tensor-only KA2 module state required')
        header=json.dumps([key,str(value.dtype),list(value.shape)],separators=(',',':')).encode()
        h.update(len(header).to_bytes(8,'big'));h.update(header);h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def validate(q,cache_sha,fold,seed):
    rec=get_recipe('ka2',total_steps=20000,batch_size=32,num_particles=512,z_dim=8)
    if q.get('schema')!=1 or q.get('method')!='pg' or q.get('native_recipe')!='ka2':raise ValueError('Native KA2 parent required')
    if q.get('status')!='completed' or q.get('completed_updates')!=q.get('fixed_endpoint') or q.get('completed_updates')!=20000:raise ValueError('Complete20k parent required')
    if q.get('fold')!=fold or q.get('seed')!=seed or q.get('cache_manifest_SHA256')!=cache_sha:raise ValueError('Parent fold/seed/fitting cache differs')
    if q.get('variant_config')!=CFG or q.get('training_streams')!=base.training_streams(fold,seed):raise ValueError('Parent architecture or streams differ')
    if q.get('serving_contract')!={'weights':'final_raw','output_noise':False}:raise ValueError('Keep original raw parent contract; EMA is a separate readout')
    if q.get('recipe')!=rec.to_dict() or q['policy']['recipe']!=rec.to_dict() or q['policy']['completed_steps']!=20000:raise ValueError('Native parent recipe/policy differs')
    if rec.ema_decay!=.995 or rec.serve_average!=0.:raise ValueError('Prescribed accumulated EMA/default fast-serving settings changed')
    if set(q['policy']['models'])!={'generator','critic','prior'} or set(q['policy']['averages'])!={'generator','prior'}:raise ValueError('Full raw and coupled EMA module state required')
    return rec

def load(mu,theta,device,q,*,cache_sha,fold=0,seed=0):
    """Load original raw modules for validation and independent saved native EMA for sampling.

    Does not restore/relabel a CUDA policy as CPU, load CUDA RNG as CPU RNG,
    advance an optimizer, alter recipe flags, or recompute/tune any EMA.
    """
    validate(q,cache_sha,fold,seed)
    if mu.shape!=(2000,) or theta.shape!=(2000,):raise ValueError('Full frozen2000gene count architecture required')
    update.configure(CFG);bundle,*_=base.build(mu,theta,device,fold,seed);policy=bundle[-1]
    raw={'generator':bundle[1],'critic':bundle[2],'prior':bundle[3]}
    averages={'generator':policy.ema_G,'prior':policy.ema_prior}
    raw_count=0;average_count=0
    for family,modules in (('models',raw),('averages',averages)):
        for name,module in modules.items():
            expected=module.state_dict();saved=q['policy'][family][name]
            if saved.keys()!=expected.keys():raise ValueError('Module state keys differ')
            for key,value in saved.items():
                if not isinstance(value,torch.Tensor) or value.shape!=expected[key].shape or value.dtype!=expected[key].dtype:raise ValueError('Module state shape/dtype differs')
                if value.is_floating_point():
                    if not bool(torch.isfinite(value).all()):raise ValueError('Nonfinite saved module')
                    if family=='models':raw_count+=value.numel()
                    else:average_count+=value.numel()
            module.load_state_dict(saved,strict=True)
    if raw_count!=371105 or average_count!=316832:raise ValueError('Full native raw/EMA state inventory differs')
    if policy.served_snapshot()['source']!='fast':raise ValueError('Original default native serving source changed')
    for name,module in averages.items():
        for key,_ in module.named_buffers():
            if not torch.equal(q['policy']['averages'][name][key],q['policy']['models'][name][key]):raise ValueError('Native current-buffer EMA contract differs')
        module.eval().requires_grad_(False)
    info={'readout':dict(READOUT),'native_default_served_source':'fast','parent_completed_updates':20000,'raw_model_floating_values':raw_count,'EMA_model_floating_values':average_count,'native_saved_EMA_state_fingerprints':{name:fingerprint(q['policy']['averages'][name]) for name in averages},'training_updates_executed':0,'training_policy_restored':False,'original_parent_contract_preserved':True}
    return policy.ema_G,policy.ema_prior,info
