"""Run actual synthetic full-width continuation/replay checks on native096."""
import datetime,importlib.util,json,os,sys,unittest
from pathlib import Path
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent
PROJECT=Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark')
SCIENCE=Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/core-comparison-20261103')
sys.path[:0]=list(map(str,[HERE,ROOT/'duration096_runtime',ROOT/'analysis/released_tuning_v096',ROOT/'model',ROOT,ROOT/'routed_resume_runtime',SCIENCE,PROJECT,PROJECT/'current_base_20261029/runtime/lib/python3.12/site-packages',PROJECT/'.venv312/lib/python3.12/site-packages']))
from release_env import activate
activate()
import preset_scope as scope,torch
torch.set_num_threads(1)
if '--parent' in sys.argv:
    import audit_parent
    audit_parent.main()
else:
    import test_extension
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_extension))
    assert result.wasSuccessful() and result.testsRun==4
    assert not torch.cuda.is_initialized()
    paths=[*HERE.glob('*.py'),*scope.MODEL_DIR.glob('*.py'),scope.MODEL_PROOF,scope.NATIVE_PROOF]
    proof=dict(status='ACTUAL_NATIVE096_EXTENSION_CPU_REPLAY_AND_REFUSALS_PASS',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),tests_passed=4,full2000gene_native_over_horizon_complete_state_replay=True,CUDA_initialized=False,GPU_allocation_created=False,source_SHA256={str(p):scope.sha(p) for p in paths})
    assert not scope.EXT_PROOF.exists();scope.EXT_PROOF.write_text(json.dumps(proof,indent=2)+'\n')
