"""Finite CPU composition qualification; no ALZ fitting or model imports."""
import datetime, hashlib, json, os, sys, unittest
from pathlib import Path
ROOT=Path('/ml2/alz');B=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark')
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
sys.path[:0]=list(map(str,[ROOT,ROOT/'model',B/'current_base_20261029/runtime/lib/python3.12/site-packages',B/'.venv312/lib/python3.12/site-packages']))
import test_pure_ensemble
result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_pure_ensemble))
if not result.wasSuccessful():raise SystemExit(1)
assert 'torch' not in sys.modules
paths=[ROOT/'model/pure_ensemble_generation.py',ROOT/'test_pure_ensemble.py',ROOT/'ensemble_scope.py',ROOT/'ensemble_execution.py',ROOT/'ensemble_protocol.json',ROOT/'ensemble_parents.json',Path(__file__)]
q={'status':'PASSED_SYNTHETIC_CPU_PURE_ENSEMBLE_COMPOSITION','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'tests_passed':result.testsRun,'exact_whole_bag_mask_and_count_verified':True,'zero_one_control_data_and_dtype_bitwise_replay':True,'sequential_global_RNG_branch_isolation_verified':True,'malformed_incomplete_and_deadline_refusal_before_yield':True,'torch_imported':False,'GPU_allocation_created':False,'native_control_replay_verified':False,'scope':'Synthetic CPU composition only; actual native zero/one controls remain required before candidate evaluation.','source_SHA256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
(ROOT/'analysis/ensemble_cpu_qualification.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(q,indent=2))
