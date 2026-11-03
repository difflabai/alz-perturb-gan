"""Finite synthetic CPU qualification, no real ALZ fit or GPU allocation."""
import datetime,hashlib,json,os,sys,unittest
from pathlib import Path
ROOT=Path('/ml2/alz');B=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark');S=Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/core-comparison-20261103')
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
sys.path[:0]=list(map(str,[ROOT,ROOT/'model',S,B,B/'current_base_20261029/runtime/lib/python3.12/site-packages',B/'.venv312/lib/python3.12/site-packages']))
import test_semantic_observation
result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_semantic_observation))
if not result.wasSuccessful():raise SystemExit(1)
paths=[ROOT/'model/semantic_observation.py',ROOT/'model/semantic_pg_update.py',ROOT/'model/semantic_pg_adapter.py',ROOT/'test_semantic_observation.py',Path(__file__)]
q={'status':'PASSED_SYNTHETIC_CPU_SEMANTIC_QUALIFICATION','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'tests_passed':result.testsRun,'disabled_complete_state_and_next_update_bitwise_equal':True,'hard_forward_and_permutation_verified':True,'finite_nonzero_semantic_input_gradients':True,'frozen_classifier_state_and_gradients_verified':True,'enabled_complete_saved_next_update_bitwise_equal':True,'native_update_AST_and_recipe_unchanged':True,'matched_existing_critic_parameters_and_initial_function':True,'scope':'Synthetic CPU only, no real ALZ fit or GPU allocation','source_SHA256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
(ROOT/'analysis/semantic_cpu_qualification.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(q,indent=2))
