"""CPU-only synthetic interrupted-state parity; does not admit actual CUDA recovery."""
import datetime,hashlib,json,os,sys,unittest
from pathlib import Path
ROOT=Path('/ml2/alz');B=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark');S=Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/core-comparison-20261103')
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
sys.path[:0]=list(map(str,[ROOT,ROOT/'model',ROOT/'analysis',ROOT/'recovery_runtime',S,B,B/'current_base_20261029/runtime/lib/python3.12/site-packages',B/'.venv312/lib/python3.12/site-packages']))
import test_exact_interrupted_resume
result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_exact_interrupted_resume))
if not result.wasSuccessful():raise SystemExit(1)
import torch
assert not torch.cuda.is_initialized()
paths=[ROOT/'recovery_runtime/exact_resume.py',ROOT/'analysis/test_exact_interrupted_resume.py',Path(__file__).resolve(),ROOT/'model/variant_pg_adapter.py',ROOT/'model/variant_pg_update.py',ROOT/'continuation.py',ROOT/'test_variant.py']
proof={'status':'PASSED_SYNTHETIC_CPU_EXACT_INTERRUPTED_STATE_PARITY','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'tests_passed':result.testsRun,
       'complete_initial_serialized_state_and_next_update_bitwise_verified':True,'native_CUDA_recovery_qualified':False,'native_allocation_created':False,'native_resume_started':False,
       'scope':'Synthetic8gene CPU fixture only, same original optimized nativePG update. Covers complete G/D/prior/optimizer/regularizer/policy/Usage/NumPy/TorchCPU/explicitgenerator state roundtrip and exact next-update replay, source/cache/seed/endpoint refusals, and CPU actual-fitting refusal. Actual8942 CUDA checkpoint full restore/two-branch next-update qualification still required; source/peer/allocation gates remain held.',
       'source_SHA256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
(ROOT/'analysis/exact_interrupted_resume_cpu_qualification.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof,indent=2))
