"""CPU-only comparison of actual resumed update20001 against the native replay."""
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path('/ml2/alz')
B = Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark')
SCIENCE = Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/core-comparison-20261103')


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    sys.path[:0] = list(map(str, [ROOT, ROOT / 'model', SCIENCE, B,
                               B / 'current_base_20261029/runtime/lib/python3.12/site-packages',
                               B / '.venv312/lib/python3.12/site-packages']))
    import torch
    torch.set_num_threads(1)
    from continuation import same
    qualified = ROOT / 'runs/CAL_GPU0_V1_continue40k_qualification_v1_f0_pg_s0/qualification_step20001.pt'
    actual = ROOT / 'runs/CAL_GPU0_V1_continue40k_h75_v1_f0_pg_s0/fit/initial_continuation.pt'
    q = torch.load(qualified, map_location='cpu', weights_only=False)
    a = torch.load(actual, map_location='cpu', weights_only=False)
    assert a['status'] == 'recovery' and a['fixed_endpoint'] == 40000 and a['initial_completed_updates'] == 20000
    keys = ['policy', 'prior_regularizer', 'generators', 'numpy_rng', 'usage', 'torch_cpu_rng', 'torch_device_rng',
            'completed_updates', 'recipe', 'variant_config', 'training_streams', 'cache_manifest_SHA256', 'fold', 'seed']
    for key in keys:
        same(q[key], a[key], key)
    assert a['completed_updates'] == 20001 and not torch.cuda.is_initialized()
    result = {'status': 'ACTUAL_CONTINUED_UPDATE20001_BITWISE_MATCHES_NATIVE_QUALIFICATION',
              'compared_fields': keys, 'qualification_checkpoint_SHA256': hashlib.sha256(qualified.read_bytes()).hexdigest(),
              'actual_initial_checkpoint_SHA256': hashlib.sha256(actual.read_bytes()).hexdigest(),
              'new_training_updates_in_compared_state': 1, 'declared_total_endpoint': 40000,
              'GPU_used_by_verifier': False, 'new_model_updates_by_verifier': 0,
              'scope': 'Exact trajectory continuation only; no calibration/evaluation quality result.'}
    (ROOT / 'analysis/continued40k_initial_state_verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
