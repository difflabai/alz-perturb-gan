"""Synthetic count-VAE optimizer/replay probe; no real fitting or quality score."""
import copy
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import tempfile
import time

import numpy as np
import torch
import vae_count_adapter as a
import mog_pg_adapter as baseline
from critic import bag_features
from test_variant import fixture
from state_fingerprint import fingerprint

ROOT=Path('/ml2/alz')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    started=time.monotonic();torch.set_num_threads(1)
    proofpath=ROOT/'analysis/particle_encoder_native_contract_probe.json'
    contract=json.loads(proofpath.read_bytes())
    assert contract['status']=='ACTUAL_NATIVE_ENCODER_FAMILY_CPU_CONTRACT_AND_GRADIENT_PROBE_PASSED'
    for path,pin in json.loads((ROOT/'analysis/routed_e22_model_CPU_qualification.json').read_bytes())['official_wheel_source_SHA256'].items():assert sha(path)==pin
    with tempfile.TemporaryDirectory() as temp:
        cache=Path(temp);(cache/'manifest.json').write_text('{}');ctx=fixture(cache);ctx.__class__=baseline.Context
        runtime=a.build(ctx,0)
        original,_,_,_,streams=baseline.build(ctx.mu,ctx.theta,ctx.device,0,0)
        assert {name:runtime.streams[name] for name in streams}==streams
        for current,prior in ((runtime.G,original[1]),(runtime.D,original[2]),(runtime.P,original[3])):
            assert fingerprint(current.state_dict())==fingerprint(prior.state_dict())
        before_recipe,after_recipe=original[0].to_dict(),runtime.recipe.to_dict()
        assert {k for k in before_recipe if before_recipe[k]!=after_recipe[k]}=={'name','encoder_mode'}
        initial={name:copy.deepcopy(module.state_dict()) for name,module in [('G',runtime.G),('E',runtime.E),('P',runtime.P)]}
        # Fixed first-bag observations are exactly invariant to cell order.
        pcs=torch.stack([ctx.trainpcs[:12],ctx.trainpcs[12:24]])
        perm=torch.randperm(12,generator=torch.Generator().manual_seed(20261104411))
        first,second=bag_features(pcs),bag_features(pcs[:,perm])
        assert torch.equal(first,second)
        c=torch.zeros(2,7);assert torch.equal(runtime.E(first,c),runtime.E(second,c))
        # Reconstruction consumes private streams and preserves global Gamma.
        state=torch.get_rng_state().clone()
        z=torch.zeros(2,8);condition=torch.zeros(2,7);lib=torch.full((2,128),40.)
        private_before=runtime.generators['reconstruction_gamma'].get_state().clone()
        result=a.count_features(ctx,runtime,z,condition,lib,reconstruction=True)
        assert result.shape==(2,150) and torch.equal(state,torch.get_rng_state())
        assert not torch.equal(private_before,runtime.generators['reconstruction_gamma'].get_state())
        # Reset before the actual optimizer proof: the sampling probe itself
        # is not silently counted as a completed update.
        runtime=a.build(ctx,0)
        one=a.advance(ctx,runtime)
        assert one['step']==1 and one['prior_draws_observed']==96 and one['native_integer_count_draws_validated']
        assert all(math.isfinite(v) and v>0 for v in one['gradient_absolute_sums'].values())
        for name,module in [('G',runtime.G),('E',runtime.E),('P',runtime.P)]:
            assert fingerprint(initial[name])!=fingerprint(module.state_dict())
        np.testing.assert_allclose(one['posterior_joint_KL'],math.log(512),rtol=1e-6,atol=0)
        q=a.snapshot(ctx,runtime,'synthetic')
        assert q['usage']['steps']==1 and int(q['usage']['draws'].sum())==96 and q['reconstruction_draws_seen']==64
        assert len(q['usage']['unique_batch_histogram'])==513
        folder=ROOT/'artifacts/diagnostics/native_vae_count_synthetic_v1';folder.mkdir(parents=True,exist_ok=False)
        checkpoint=folder/'after1.pt';torch.save(q,checkpoint);q=torch.load(checkpoint,map_location='cpu',weights_only=False)
        two=a.advance(ctx,runtime);expected=a.snapshot(ctx,runtime,'synthetic')
        restored=a.restore(ctx,q);replayed=a.advance(ctx,restored);actual=a.snapshot(ctx,restored,'synthetic')
        assert fingerprint(two)==fingerprint(replayed) and fingerprint(expected)==fingerprint(actual)
        for name,value in [('native_family','mog'),('adapter_config',{}),('cache_manifest_SHA256','0'*64),('reconstruction_draws_seen',63)]:
            bad=copy.deepcopy(q);bad[name]=value
            try:a.restore(ctx,bad)
            except ValueError:pass
            else:raise AssertionError('Invalid VAE checkpoint accepted: '+name)
        bad=copy.deepcopy(q);bad['generators'].pop('reconstruction_gamma')
        try:a.restore(ctx,bad)
        except ValueError:pass
        else:raise AssertionError('Missing private Gamma state accepted')
        try:a.fit(ctx,cache/'must_not_exist',0,end_monotonic=0)
        except RuntimeError:pass
        else:raise AssertionError('Unadmitted fitting path exists')
        assert not (cache/'must_not_exist').exists() and not torch.cuda.is_initialized()
        result={'status':'NATIVE_VAE_COUNT_SYNTHETIC_PAIR_AND_FULL_REPLAY_PASSED','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'native_family':'vae_gan','trainer_preset':'bispf_optimized','original_optimizer_pairs':2,'replayed_optimizer_pairs':1,
                'first_update':one,'second_update':two,'same_initial_G_D_native_MoG_and_original12stream_seeds':True,
                'recipe_difference_from_native_MoG':['name','encoder_mode'],'native_E_optimizer_ownership_and_checkpoint':True,
                'permutation_invariant_encoder_observation_verified':True,'private_reconstruction_Gamma_preserves_global_stream':True,
                'complete_native_and_all7private_RNGs_sampler_usage_serialized_next_update_byte_exact':True,
                'wrong_family_config_cache_missingGamma_and_clock_refusals_verified':True,'real_fitting_path_refused':True,
                'checkpoint_path':str(checkpoint),'checkpoint_SHA256':sha(checkpoint),
                'CPU_wall_seconds':time.monotonic()-started,'peak_process_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                'source_SHA256':{str(p):sha(p) for p in [Path(__file__).resolve(),Path(a.__file__).resolve(),proofpath,
                    ROOT/'analysis/vae_count_feasibility_hypothesis.json',ROOT/'model/mog_pg_adapter.py',ROOT/'model/mog_pg_update.py',ROOT/'test_variant.py']},
                'real_data_used':False,'full2000gene_qualification_completed':False,'clean400cell_generation_qualified':False,
                'GPU_context_created':False,'native_admission_ready':False,'ALZ_quality_measured':False,'current_routed_trial_changed':False,
                'scope':'Synthetic8genes,32bagsx128cells, six donor labels, exact original CountGenerator/VectorCritic and150invariantfeatures. Native public hardVAE Gaussian summary objective/learnedMoG/G+E+P optimizer/constraintgeometry with both GAN and reconstruction protected. Two original optimizer pairs plus one exact serialized replay. No real donor optimizer input or calibration score; no cell-count likelihood claim. This remains a CPU feasibility adapter needing fullwidth/serving/data/native-runtime qualification.'}
        out=ROOT/'analysis/vae_count_synthetic_CPU_feasibility.json';assert not out.exists()
        out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k not in ('source_SHA256','scope')},indent=2))


if __name__=='__main__':main()
