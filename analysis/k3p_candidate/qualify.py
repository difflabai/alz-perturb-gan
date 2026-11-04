"""Qualify an inactive native K3P prototype; never touch GPU admission."""
import datetime,hashlib,json,os,sys,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=Path('/ml2/alz')
B=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark')
S=Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/core-comparison-20261103')
sys.path[:0]=list(map(str,[HERE,ROOT/'analysis',ROOT/'model',ROOT,S,B,B/'current_base_20261029/runtime/lib/python3.12/site-packages',B/'.venv312/lib/python3.12/site-packages']))
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
import test_k3p as tests
import torch
from particlegan import get_recipe

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    output=HERE/'CPU_qualification.json';assert not output.exists()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests))
    if not result.wasSuccessful():raise SystemExit(1)
    assert not torch.cuda.is_initialized()
    official=json.loads((ROOT/'analysis/particlegan_release_verification_20261104.json').read_text())
    for path,pin in official['installed_source_SHA256'].items():assert sha(path)==pin
    sources=[*HERE.glob('*.py'),HERE/'hypothesis.json',ROOT/'model/ka2_update.py',ROOT/'model/ka2_adapter.py',ROOT/'model/variant_pg_adapter.py',ROOT/'model/variant_pg_update.py',ROOT/'test_variant.py',ROOT/'analysis/particlegan_release_verification_20261104.json']
    proof={'status':'PASSED_INACTIVE_SYNTHETIC_CPU_K3P_PROTOTYPE_QUALIFICATION',
           'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'tests_passed':result.testsRun,
           'actual_native_updates_before_late_replay':1400,
           'native_K3P_conditional_EMA_penalty_evaluation_verified':True,
           'early_and_late_complete_state_replay_bitwise_equal':True,
           'same_initial_G_D_prior_rows_and_exact_native_factories_verified':True,
           'native_input_output_noise_G_prior_EMA_and_invalid_state_refusals_verified':True,
           'unchanged_task_features_fitting_sampler_and_fit_verified':True,
           'resolved_recipe':get_recipe('k3p',total_steps=20000,batch_size=32,num_particles=512,z_dim=8).to_dict(),
           'source_SHA256':{str(p):sha(p) for p in sources},
           'official_wheel_source_SHA256':official['installed_source_SHA256'],
           'CUDA_initialized':False,'GPU_allocation_created':False,'native_admission_ready':False,
           'quality_result_available':False,'incumbent_changed':False,
           'scope':'Inactive analysis-only prototype. Actual synthetic native updates cross the LR-dependent K3P anchor transition; KA2 fixed800-call warmup is not assumed for this archived critic. All native controls intact. No donor reads, GPU fitting, calibration/evaluation ranking, native binding or successor. A future source relocation requires explicit hash/AST or replay requalification before any admission.'}
    output.write_text(json.dumps(proof,indent=2,allow_nan=False)+'\n')
    print(json.dumps({key:proof[key] for key in ('status','tests_passed','actual_native_updates_before_late_replay','CUDA_initialized','native_admission_ready')},indent=2))

if __name__=='__main__':main()
