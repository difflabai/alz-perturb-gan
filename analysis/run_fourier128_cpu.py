"""Run actual synthetic qualification and pin the exact proposed model sources."""
import datetime,json,os,sys,unittest
from pathlib import Path
ROOT=Path('/ml2/alz');B=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark');S=Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/core-comparison-20261103')
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
sys.path[:0]=list(map(str,[ROOT/'analysis',ROOT/'model',ROOT,S,B,B/'current_base_20261029/runtime/lib/python3.12/site-packages',B/'.venv312/lib/python3.12/site-packages']))
import test_fourier128 as tests
import torch
from calibration import sha
from particlegan import get_recipe

def main():
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests))
    if not result.wasSuccessful():raise SystemExit(1)
    assert not torch.cuda.is_initialized()
    release=ROOT/'analysis/particlegan_release_verification_20261104.json';official=json.loads(release.read_text())
    for path,pin in official['installed_source_SHA256'].items():assert sha(path)==pin
    paths=[Path(__file__).resolve(),ROOT/'analysis/test_fourier128.py',ROOT/'analysis/fourier128_hypothesis.json',ROOT/'model/fourier128_observation.py',ROOT/'model/fourier128_update.py',ROOT/'model/fourier128_adapter.py',ROOT/'test_variant.py',ROOT/'model/variant_pg_update.py',ROOT/'model/variant_pg_adapter.py',S/'critic.py',release]
    proof={'status':'PASSED_SYNTHETIC_CPU_FOURIER128_MODEL_QUALIFICATION','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'tests_passed':result.testsRun,
           'observation_config':tests.obs.OPTIONS,'resolved_recipe':get_recipe(total_steps=20000,batch_size=32,num_particles=512,z_dim=8).to_dict(),
           'first100_moment_coordinates_byteidentical':True,'permutation_invariance_and_unit_fourier_block_bound_verified':True,
           'official_recipe_components_initial_G_prior_and_streams_verified':True,'critic_input_capacity_change_declared':True,
           'native_update_context_batch_and_fit_AST_unchanged':True,'isolated_new_Fourier_block_G_dispersion_prior_gradients_verified':True,
           'finite_nonzero_native_update_gradients_verified':True,'full_serialized_next_update_bitwise_equal':True,'wrong_observation_recipe_cache_architecture_refusals_verified':True,
           'expected_full_model_floating_values':397545,'CUDA_initialized':False,'actual_GPU_qualification_performed':False,'native_admission_ready':False,'quality_result_available':False,
           'source_SHA256':{str(p):sha(p) for p in paths},'official_wheel_source_SHA256':official['installed_source_SHA256'],
           'scope':'Synthetic CPU only. No fitting/calibration/evaluation donors read, no GPU context, no native training or quality measurement. Native optimizedBISPF unchanged; richer fixed observation/critic input dimensions are explicit model-choice confounds. Native source/resource/lease/accounting gates and actual fixed20k plus saved-score/count/fullcheckpoint audit still required.'}
    out=ROOT/'analysis/fourier128_model_CPU_qualification.json';assert not out.exists();out.write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps({k:proof[k] for k in ('status','tests_passed','expected_full_model_floating_values','CUDA_initialized')},indent=2))

if __name__=='__main__':main()
