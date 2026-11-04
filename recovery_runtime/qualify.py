"""Standard-library recovery/seed scope checks; no fitting, CUDA or allocation."""
import datetime,json,os,sys,unittest
from pathlib import Path
ROOT=Path('/ml2/alz');sys.path[:0]=[str(Path(__file__).parent),str(ROOT/'runtime'),str(ROOT),str(ROOT/'analysis')]
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
import recovery_scope as scope
import test_scope
result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_scope))
if not result.wasSuccessful():raise SystemExit(1)
assert 'torch' not in sys.modules
synthetic=scope.read(ROOT/'analysis/exact_learned_resume_cpu_qualification.json');assert synthetic['status']=='PASSED_SYNTHETIC_CPU_BOTH_ADAPTERS_EXACT_REPLAY'
for path,pin in synthetic['source_SHA256'].items():assert scope.sha(path)==pin
paths=[*scope.HERE.glob('*.py'),ROOT/'runtime/binding.py',ROOT/'runtime/worker_gpu0.py',ROOT/'runtime/project_runner_gpu0.py',scope.HERE/'parallel_policy.turnover.prepared.json']
proof={'status':'PASSED_CPU_FROZEN_EXACT_RECOVERY_RUNTIME','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope_checks_passed':result.testsRun,'synthetic_serialized_restore_checks_passed':synthetic['tests_passed'],
       'recovery_protocol_SHA256':scope.sha(scope.PROTOCOL),'native_CUDA_restore_verified':False,'GPU_allocation_created':False,'native_recovery_started':False,'resource_profile_status':'PRE_REBOOT_PROFILE_NOT_DEPLOYABLE_AFTER_REBOOT',
       'source_SHA256':{str(path):scope.sha(path) for path in paths},'scope':'CPU exact lineage/deadline/source/seed/checkpoint refusals and native source closure through private proof-wire fixtures only. Original frozen scientific sources remain identical. Actual learned9237 CUDA fullstate/nextstate plus retained completedPG8 qualification and clean settlement are mandatory before recovery; Separate independently reviewed post-reboot ALZ profile required separately; original task11 nativepolicy/registry unchanged. No incumbent/quality-result mutation.'}
scope.CPU_PROOF.write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof,indent=2))
