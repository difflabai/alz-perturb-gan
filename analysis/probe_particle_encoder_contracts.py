"""Exercise official encoder-family contracts on synthetic CPU vectors only."""
import copy
import datetime
import hashlib
import json
import math
import os
from pathlib import Path

import torch
from torch import nn
import particlegan
from particlegan import get_recipe,UpdatePolicy,GANTrainer

ROOT=Path('/ml2/alz')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def refuses(callback,fragment):
    try:callback()
    except ValueError as error:
        assert fragment in str(error);return str(error)
    raise AssertionError('Required native contract refusal did not occur')


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    torch.set_num_threads(1)
    pins=json.loads((ROOT/'analysis/routed_e22_model_CPU_qualification.json').read_bytes())['official_wheel_source_SHA256']
    assert len(pins)==51
    for path,pin in pins.items():assert sha(path)==pin
    rows=[]
    for name in ('ae_gan','vae_gan'):
        recipe=get_recipe(name,trainer='bispf_optimized',total_steps=20000,batch_size=32,num_particles=512,z_dim=8)
        assert recipe.prior_kind=='mog' and recipe.sigma_rel==.025
        assert recipe.encoder_mode==('ae' if name=='ae_gan' else 'hard')
        torch.manual_seed(20261104401)
        G=nn.Linear(8,8);D=nn.Linear(8,1);E=nn.Linear(8,16 if name=='ae_gan' else 8)
        prior=recipe.make_prior(generator=torch.Generator().manual_seed(20261104402))
        og,od=recipe.make_optimizers(G,D,prior,encoder=E)
        policy=UpdatePolicy(recipe,G,D,prior=prior,encoder=E,generator_optimizer=og,critic_optimizer=od,row_semantics='conditional')
        state=policy.state_dict();assert set(state['models'])=={'generator','critic','prior','encoder'}
        owned=[p for group in og.param_groups for p in group['params']]
        assert len({id(p) for p in owned})==len(owned)
        assert {id(p) for p in owned}=={id(p) for m in (G,E,prior) for p in m.parameters() if p.requires_grad}
        trainer_refusal=refuses(lambda:GANTrainer(recipe,G,D,prior=prior),'no encoder')
        x=torch.randn(6,8,generator=torch.Generator().manual_seed(20261104403))
        encoded=E(x)
        if name=='ae_gan':
            query,offset=encoded.chunk(2,dim=-1)
            missing=refuses(lambda:recipe.encode(query,prior),'requires offset')
            encoding=recipe.encode(query,prior,offset=offset)
            assert encoding.codes.shape==(6,1,8) and torch.equal(encoding.kl,torch.zeros(6))
            prediction=G(encoding.codes)
            loss=encoding.reconstruction_loss(prediction,x)
            # A sequential float32 reduction and a flattened reduction can
            # differ by one ULP; check the mathematical MSE in float64.
            reference=(prediction.double()-x[:,None].double()).square().mean()
            torch.testing.assert_close(loss,reference,rtol=4*torch.finfo(loss.dtype).eps,atol=0,check_dtype=False)
            elbo_refusal=refuses(lambda:encoding.negative_elbo(prediction,x),'no variational ELBO')
            details={'required_offset_refusal':missing,'ELBO_refusal':elbo_refusal,'objective':'native reconstruction MSE; deterministic AE has no posterior ELBO'}
        else:
            offset_refusal=refuses(lambda:recipe.encode(encoded,prior,offset=torch.zeros_like(encoded)),'does not accept an offset')
            encoding=recipe.encode(encoded,prior,draws=2,generator=torch.Generator().manual_seed(20261104404))
            assert encoding.mode=='hard' and encoding.codes.shape==(6,2,8)
            torch.testing.assert_close(encoding.kl,torch.full((6,),math.log(512)),rtol=0,atol=0)
            prediction=G(encoding.codes)
            sigma=recipe.observation_sigma
            loss=encoding.negative_elbo(prediction,x,observation_sigma=sigma)
            expected=8*(prediction-x[:,None]).square().mean()/(2*sigma*sigma)+math.log(512)+4*math.log(2*math.pi*sigma*sigma)
            torch.testing.assert_close(loss,expected)
            details={'offset_refusal':offset_refusal,'joint_KL_per_bag':float(encoding.kl[0]),'observation_sigma':sigma,
                     'objective':'native Gaussian negative ELBO with normalization constant and joint categorical KL',
                     'posterior':'hard selected table center plus prior-matching local Gaussian; no learned local posterior variance/offset',
                     'encoder_gradient':'native biased hard straight-through routing'}
        loss.backward()
        grads={label:float(sum(p.grad.abs().sum() for p in module.parameters() if p.grad is not None))
               for label,module in [('decoder',G),('encoder',E),('learned_prior',prior)]}
        assert all(math.isfinite(v) and v>0 for v in grads.values())
        assert all(p.grad is None for p in D.parameters()) and policy.completed_steps==0
        rows.append({'model_family':name,'trainer_preset':'bispf_optimized','resolved_recipe':recipe.to_dict(),
                     'public_encoder_and_joint_optimizer_ownership_verified':True,'native_encoder_policy_checkpoint_includes_encoder':True,
                     'GANTrainer_refusal':trainer_refusal,'encoding_shape':list(encoding.codes.shape),'objective_value':float(loss.detach()),
                     'gradient_absolute_sums':grads,'native_optimizer_updates_executed':0,**details})
    assert not torch.cuda.is_initialized()
    package=Path(particlegan.__file__).resolve().parent
    source=[Path(__file__).resolve(),ROOT/'analysis/routed_e22_model_CPU_qualification.json',*(package/n for n in ('recipes.py','autoencoder.py','policy.py','training.py'))]
    result={'status':'ACTUAL_NATIVE_ENCODER_FAMILY_CPU_CONTRACT_AND_GRADIENT_PROBE_PASSED',
            'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'rows':rows,
            'native_release':'0.9.4','official_python_source_files_unchanged':51,
            'source_SHA256':{str(p):sha(p) for p in source},
            'real_data_used':False,'count_adapter_implemented':False,'ALZ_quality_comparison_completed':False,
            'GPU_context_created':False,'optimizer_updates_executed':0,'current_routed_trial_changed':False,
            'initial_probe_assertion_note':'An initial exact comparison of two float32 MSE reduction orders differed by1ULP; failedsource/output retained under artifacts/diagnostics/particle_encoder_contract_probe_float_reference. Final mathematical MSE check uses independent float64 reference with4float32epsilon relative tolerance. No native source/objective/settings or byte-exact checkpoint check changed.',
            'next_adapter_requirements':['Use the actual ae_gan or vae_gan family and an explicit trainer; they remain GAN families with caller-owned reconstruction terms.',
                                         'Bind a permutation-invariant bag encoder and declare the reconstruction observation space/likelihood. Native negative_elbo is Gaussian, not a negative-binomial count likelihood.',
                                         'Preserve native hard-VAE joint KL and prior-matching local Gaussian; do not substitute classical learned-variance VAE assumptions or call reconstruction MSE an ELBO.',
                                         'Generic GANTrainer explicitly refuses encoder families. A faithful external loop needs public UpdatePolicy encoder ownership, the GAN/reconstruction/prior objectives, complete sampler/checkpoint replay and direct integer-count serving.',
                                         'Do not change the live routed E22 source, infer quality from this synthetic helper probe, or add it to completed native-quality counts.'],
            'scope':'Synthetic six8D CPU vectors and toy linear networks only, not an ALZ count adapter or update/quality trial. Native factory/encoder/objective/ownership/refusal and nonzero decoder/encoder/prior backward paths exercised on unmodified official wheel. AE/VAE/DDGAN remain untrained for ALZ; original fixed cohort/seed selection and active routed20k trial unchanged.'}
    output=ROOT/'analysis/particle_encoder_native_contract_probe.json';assert not output.exists()
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','source_SHA256','next_adapter_requirements')},indent=2))


if __name__=='__main__':main()
