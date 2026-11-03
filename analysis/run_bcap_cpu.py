"""Finite native BCAP synthetic CPU qualification; never fit ALZ on CPU."""
import datetime,hashlib,json,os,sys,unittest
from pathlib import Path
ROOT=Path('/ml2/alz');B=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark');S=Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/core-comparison-20261103')
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
sys.path[:0]=list(map(str,[ROOT,ROOT/'model',S,B,B/'current_base_20261029/runtime/lib/python3.12/site-packages',B/'.venv312/lib/python3.12/site-packages']))
import test_bcap
result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_bcap))
if not result.wasSuccessful():raise SystemExit(1)
paths=[ROOT/'model/bcap_adapter.py',ROOT/'model/bcap_update.py',ROOT/'test_bcap.py',ROOT/'bcap_protocol.json',Path(__file__)]
q={'status':'PASSED_SYNTHETIC_CPU_BCAP_QUALIFICATION','scope':'No actual ALZ fitting; no GPU allocation','tests_passed':result.testsRun,'native_recipe':'bcap','declared_task_overrides':{'total_steps':20000,'batch_size':32,'num_particles':512,'z_dim':8},'full_serialized_restore_next_update_bitwise_equal':True,'native_api_component_types_verified':True,'same_initial_generator_critic_and_prior_rows_verified':True,'native_prior_buffer_difference':'BCAP has only z; optimized BISPF adds three native SBNP reference buffers. Initial learned parameter rows match exactly; complete prior state dictionaries intentionally differ.','optimized_update_AST_unchanged':True,'CUDA_initialized':False,'source_SHA256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(ROOT/'analysis/bcap_cpu_qualification.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(q,indent=2))
