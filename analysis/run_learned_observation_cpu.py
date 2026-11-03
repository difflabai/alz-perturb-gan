"""CPU-only synthetic qualification, no ALZ training or native GPU allocation."""
import datetime,hashlib,json,os,sys,unittest
from pathlib import Path
ROOT=Path('/ml2/alz');B=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark');S=Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/core-comparison-20261103')
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
sys.path[:0]=list(map(str,[ROOT,ROOT/'model',S,B,B/'current_base_20261029/runtime/lib/python3.12/site-packages',B/'.venv312/lib/python3.12/site-packages']))
import test_learned_observation
result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_learned_observation))
if not result.wasSuccessful():raise SystemExit(1)
paths=[ROOT/'model/learned_pg_adapter.py',ROOT/'model/learned_pg_update.py',ROOT/'model/learned_cell_observation.py',ROOT/'test_learned_observation.py',ROOT/'learned_observation_protocol.json',Path(__file__)]
q={'status':'PASSED_SYNTHETIC_CPU_LEARNED_OBSERVATION_QUALIFICATION','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'tests_passed':result.testsRun,'disabled_complete_state_and_next_update_bitwise_equal':True,'native_recipe_exact_API_factory':True,'same_initial_G_prior_original_head':True,'fixed100_moments_and_permutations_verified':True,'both_real_and_fake_adversarial_encoder_gradients_verified':True,'head_only_penalty150_vector_inputs_verified':True,'finite_G_dispersion_prior_gradients_through_frozen_encoder':True,'full_serialized_restore_next_update_bitwise_equal':True,'only_declared_observation_update_AST_delta':True,'scope':'Synthetic CPU only; no actual ALZ fitting or GPU allocation','source_SHA256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
(ROOT/'analysis/learned_observation_cpu_qualification.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(q,indent=2))
