"""Qualify final task-extension source paths; never deploy or initialize CUDA."""
import ast,datetime,hashlib,inspect,json,os,sys,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=Path('/ml2/alz')
B=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark')
S=Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/core-comparison-20261103')
sys.path[:0]=list(map(str,[ROOT/'model',HERE,ROOT/'analysis',ROOT,S,B,B/'current_base_20261029/runtime/lib/python3.12/site-packages',B/'.venv312/lib/python3.12/site-packages']))
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
import task_cell_energy
sys.modules['estimator']=task_cell_energy  # Same tested helper under its final unambiguous name.
import test_estimator,test_task_loop
import torch

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    output=ROOT/'analysis/cell_energy_model_CPU_qualification.json';assert not output.exists()
    old=json.loads((HERE/'CPU_task_loop_qualification.json').read_text())
    for path,pin in old['source_SHA256'].items():assert sha(path)==pin
    assert (ROOT/'model/task_cell_energy.py').read_bytes()==(HERE/'estimator.py').read_bytes()
    for name in ('cell_ed_update.py','cell_ed_adapter.py'):
        expected=(HERE/name).read_text().replace('from estimator import','from task_cell_energy import')
        assert (ROOT/'model'/name).read_text()==expected
    assert Path(test_task_loop.a.__file__).resolve()==ROOT/'model/cell_ed_adapter.py'
    assert Path(test_task_loop.u.__file__).resolve()==ROOT/'model/cell_ed_update.py'
    assert Path(inspect.getfile(test_task_loop.u.pooled_cell_energy)).resolve()==ROOT/'model/task_cell_energy.py'
    torch.use_deterministic_algorithms(True)
    suites=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(module) for module in (test_estimator,test_task_loop)])
    result=unittest.TextTestRunner(verbosity=2).run(suites)
    if not result.wasSuccessful():raise SystemExit(1)
    assert not torch.cuda.is_initialized()
    for path,pin in old['official_wheel_source_SHA256'].items():assert sha(path)==pin
    sources=[ROOT/'model/cell_ed_adapter.py',ROOT/'model/cell_ed_update.py',ROOT/'model/task_cell_energy.py',HERE/'test_estimator.py',HERE/'test_task_loop.py',HERE/'hypothesis.json',HERE/'SCOPE.txt',HERE/'CPU_math_qualification.json',HERE/'CPU_task_loop_qualification.json',HERE/'fitting_stage_weights.json',Path(__file__).resolve(),ROOT/'test_variant.py',ROOT/'model/variant_pg_adapter.py',ROOT/'model/variant_pg_update.py']
    proof={'status':'PASSED_CPU_FINAL_PATH_CELL_ENERGY_TASK_EXTENSION_QUALIFICATION','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'tests_passed':result.testsRun,'math_tests_passed':5,'task_loop_tests_passed':6,
           'final_import_paths_and_only_helper_import_rename_verified':True,
           'source_SHA256':{str(p):sha(p) for p in sources},'official_wheel_source_SHA256':old['official_wheel_source_SHA256'],
           'cell_energy_config':test_task_loop.u.CELL_ENERGY_CONFIG,
           'unchanged_native_recipe_factories_and_initial_G_D_prior_tensors_verified':True,
           'unchanged_fitting_draws_and_native_first_critic_update_verified':True,
           'no_additional_random_draws_all_first_update_stream_states_equal':True,
           'auxiliary_alone_finite_nonzero_G_dispersion_prior_gradients_verified':True,
           'auxiliary_changes_G_update_verified':True,'complete_serialized_next_update_bitwise_equal':True,
           'task_objective_fitting_stage_counts_recipe_config_cache_refusals_verified':True,
           'exact_population_energy_identity_and_cluster_bias_counterexample_verified':True,
           'CUDA_initialized':False,'actual_GPU_qualification_performed':False,'native_admission_ready':False,
           'quality_result_available':False,'scope':'Explicit ALZ task-loss extension to native optimizedBISPF, not a published formulation. Eleven actual CPU tests at final import paths with deterministic algorithms enabled. Only helper import name differs from tested prototype. No donor/model training, calibration/evaluation array access, GPU or quality claim; native resource/source binding and full fixed20k quality audit still required.'}
    output.write_text(json.dumps(proof,indent=2,allow_nan=False)+'\n')
    print(json.dumps({key:proof[key] for key in ('status','tests_passed','math_tests_passed','task_loop_tests_passed','final_import_paths_and_only_helper_import_rename_verified')},indent=2))

if __name__=='__main__':main()
