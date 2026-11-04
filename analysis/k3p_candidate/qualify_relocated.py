"""Actual qualification of the exact K3P prototype at final model import paths."""
import datetime,hashlib,json,os,sys,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=Path('/ml2/alz')
B=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark')
S=Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/core-comparison-20261103')
sys.path[:0]=list(map(str,[ROOT/'model',HERE,ROOT/'analysis',ROOT,S,B,B/'current_base_20261029/runtime/lib/python3.12/site-packages',B/'.venv312/lib/python3.12/site-packages']))
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
import test_k3p as tests
import torch
from particlegan import get_recipe

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    out=ROOT/'analysis/k3p_model_CPU_qualification.json';assert not out.exists()
    old=json.loads((HERE/'CPU_qualification.json').read_text())
    for path,pin in old['source_SHA256'].items():assert sha(path)==pin
    for name in ('k3p_update.py','k3p_adapter.py'):assert sha(ROOT/'model'/name)==sha(HERE/name)
    assert Path(tests.a.__file__).resolve()==ROOT/'model/k3p_adapter.py'
    assert Path(tests.u.__file__).resolve()==ROOT/'model/k3p_update.py'
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests))
    if not result.wasSuccessful():raise SystemExit(1)
    assert not torch.cuda.is_initialized()
    for path,pin in old['official_wheel_source_SHA256'].items():assert sha(path)==pin
    sources=[ROOT/'model/k3p_adapter.py',ROOT/'model/k3p_update.py',ROOT/'model/ka2_update.py',HERE/'test_k3p.py',HERE/'hypothesis.json',HERE/'CPU_qualification.json',Path(__file__).resolve(),ROOT/'test_variant.py',ROOT/'model/variant_pg_adapter.py',ROOT/'model/variant_pg_update.py']
    proof={'status':'PASSED_SYNTHETIC_CPU_K3P_MODEL_QUALIFICATION','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'tests_passed':result.testsRun,'actual_native_updates_before_late_replay':1400,
           'source_relocation_byteidentical_and_final_import_paths_verified':True,
           'exact_official_recipe_components_and_initial_G_D_prior_rows':True,
           'native_noise_and_G_prior_averages_verified':True,
           'actual_LR_dependent_anchor_transition_and_conditional_EMA_evaluation_verified':True,
           'unchanged_task_sampling_features_and_fit_verified':True,
           'finite_nonzero_G_dispersion_prior_gradients_verified':True,
           'early_and_post_transition_full_serialized_next_update_bitwise_equal':True,
           'wrong_recipe_cache_EMA_serving_architecture_refusals_verified':True,
           'resolved_recipe':get_recipe('k3p',total_steps=20000,batch_size=32,num_particles=512,z_dim=8).to_dict(),
           'source_SHA256':{str(p):sha(p) for p in sources},'official_wheel_source_SHA256':old['official_wheel_source_SHA256'],
           'CUDA_initialized':False,'GPU_allocation_created':False,'native_admission_ready':False,'quality_result_available':False,
           'scope':'Actual six synthetic CPU tests rerun at final model import paths, including1400 nativeupdates and conditional EMA/late replay. Unmodified archivedK3P recipe and current task update reused; frozen finalrawclean serving. No donor data, GPU or quality result. Separate native source/resource/admission binding required.'}
    out.write_text(json.dumps(proof,indent=2,allow_nan=False)+'\n')
    print(json.dumps({key:proof[key] for key in ('status','tests_passed','actual_native_updates_before_late_replay','source_relocation_byteidentical_and_final_import_paths_verified')},indent=2))

if __name__=='__main__':main()
