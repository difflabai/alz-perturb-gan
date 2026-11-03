"""Run CPU-only synthetic checks using exact guarded runtime import roots."""
import datetime,hashlib,json,os,sys,unittest
from pathlib import Path
ROOT=Path('/ml2/alz');B=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark');S=Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/core-comparison-20261103')
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
sys.path[:0]=list(map(str,[ROOT,ROOT/'model',S,B,B/'current_base_20261029/runtime/lib/python3.12/site-packages',B/'.venv312/lib/python3.12/site-packages']))
import test_original_bispf
result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_original_bispf))
if not result.wasSuccessful():raise SystemExit(1)
paths=[ROOT/'model/original_bispf_adapter.py',ROOT/'model/original_bispf_update.py',ROOT/'test_original_bispf.py',Path(__file__)]
q={'status':'PASSED_SYNTHETIC_CPU_ORIGINAL_BISPF_QUALIFICATION','scope':'No actual ALZ fitting; no GPU allocation','tests_passed':result.testsRun,'native_recipe':'bispf','declared_task_overrides':{'total_steps':20000,'batch_size':32,'num_particles':512,'z_dim':8},'full_serialized_restore_next_update_bitwise_equal':True,'native_api_component_types_verified':True,'same_initial_generator_critic_prior_as_PG8_verified':True,'only_SRPGAN_real_logits_change_in_update_AST':True,'CUDA_initialized':False,'source_SHA256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(ROOT/'analysis/original_bispf_cpu_qualification.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(q,indent=2))
