"""Run original and learned adapter CPU replay tests; never allocate CUDA."""
import datetime,hashlib,json,os,sys,unittest
from pathlib import Path
R=Path('/ml2/alz');B=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark');S=Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/core-comparison-20261103')
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
sys.path[:0]=list(map(str,[R/'analysis',R/'recovery_runtime',R/'model',R,B,S,B/'current_base_20261029/runtime/lib/python3.12/site-packages',B/'.venv312/lib/python3.12/site-packages']))
import torch
torch.set_num_threads(1)
import test_exact_interrupted_resume,test_exact_learned_resume
suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(m) for m in [test_exact_interrupted_resume,test_exact_learned_resume]])
result=unittest.TextTestRunner(verbosity=2).run(suite);assert result.wasSuccessful() and not torch.cuda.is_initialized()
paths=[Path(__file__).resolve(),R/'analysis/test_exact_interrupted_resume.py',R/'analysis/test_exact_learned_resume.py',R/'recovery_runtime/exact_resume.py',R/'model/variant_pg_adapter.py',R/'model/learned_pg_adapter.py',R/'model/learned_pg_update.py']
proof={'status':'PASSED_SYNTHETIC_CPU_BOTH_ADAPTERS_EXACT_REPLAY','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'tests_passed':result.testsRun,'CUDA_initialized':False,'source_SHA256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'scope':'Synthetic CPU adapter state restoration and uninterrupted-next-update parity, both original PG8 and learned encoder. No actual ALZ state used for fitting, no CUDA allocation or permission claim. Real learned9237 CUDA proof remains mandatory.'}
p=R/'analysis/exact_learned_resume_cpu_qualification.json';p.write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof,indent=2))
