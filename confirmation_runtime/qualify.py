"""CPU-only finite qualification; never creates native leases or quality measurements."""
import datetime,hashlib,json,os,sys,unittest
from pathlib import Path
ROOT=Path('/ml2/alz');B=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark')
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
sys.path[:0]=list(map(str,[Path(__file__).parent,ROOT,ROOT/'model',ROOT/'runtime',ROOT/'analysis',B/'current_base_20261029/runtime/lib/python3.12/site-packages',B/'.venv312/lib/python3.12/site-packages']))
import ensemble_confirmation_scope as scope
import test_confirmation,test_pure_ensemble
suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(module) for module in (test_confirmation,test_pure_ensemble)])
result=unittest.TextTestRunner(verbosity=2).run(suite)
if not result.wasSuccessful():raise SystemExit(1)
assert 'torch' not in sys.modules
paths=[*scope.HERE.glob('*.py'),ROOT/'runtime/binding.py',ROOT/'runtime/worker_gpu0.py']
proof={'status':'PASSED_CPU_FROZEN_TWO_PARENT_CONFIRMATION','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'freeze_SHA256':scope.sha(scope.FREEZE),
       'tests_passed':result.testsRun,'source_SHA256':{str(p):scope.sha(p) for p in paths},'GPU_allocation_created':False,'native_confirmation_completed':False,
       'torch_imported':False,'frozen_scientific_source_closure_unchanged':True,
       'scope':'CPU fit-orchestration adapter-specific Context ownership and release, two seed identities, incomplete/deadline/seed/config/warm-start/source drift refusals, original wholebag composition and actual immutable native source closure using a private proof-wire fixture. No native model fit, quality result, allocation or actual confirmation claim. Original adapters/scorer/countgenerator unchanged. Native final two-model and saved-array/count audit mandatory.'}
scope.PROOF.write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof,indent=2))
