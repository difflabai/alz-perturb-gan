"""Finite synthetic CPU qualification; no actual ALZ fitting or GPU allocation."""
import datetime, hashlib, json, os, sys, unittest
from pathlib import Path
ROOT=Path('/ml2/alz');B=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark');S=Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/core-comparison-20261103')
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
sys.path[:0]=list(map(str,[ROOT,ROOT/'model',S,B,B/'current_base_20261029/runtime/lib/python3.12/site-packages',B/'.venv312/lib/python3.12/site-packages']))
import test_network_rate
result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_network_rate))
if not result.wasSuccessful():raise SystemExit(1)
paths=[ROOT/'model/network_rate_adapter.py',ROOT/'model/network_rate_update.py',ROOT/'test_network_rate.py',ROOT/'network_rate_protocol.json',Path(__file__)]
q={'status':'PASSED_SYNTHETIC_CPU_NETWORK_RATE_QUALIFICATION','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'tests_passed':result.testsRun,'native_recipe':'bispf_optimized','network_rate_config':{'lr':.006,'prior_lr_mult':5.},'only_two_official_recipe_fields_change':True,'native_api_component_types_verified':True,'same_initial_complete_G_D_prior_models':True,'optimizer_state_only_intended_network_rates_differ':True,'actual_UpdatePolicy_rates_and_native_schedule_verified':True,'full_serialized_restore_next_update_bitwise_equal':True,'update_observation_context_and_fit_AST_unchanged':True,'finite_nonzero_native_gradients_verified':True,'CUDA_initialized':False,'scope':'Synthetic CPU only, no actual ALZ fitting or GPU allocation','source_SHA256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
(ROOT/'analysis/network_rate_cpu_qualification.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(q,indent=2))
