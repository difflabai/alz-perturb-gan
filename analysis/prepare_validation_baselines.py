"""CPU-only matched calibration baselines for prespecified seed confirmations."""
import json
import os
from pathlib import Path
import sys

ROOT = Path('/ml2/alz')
ORIGINAL = Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-11/gpu0-coexistence/integration')


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    sys.path.insert(0, str(ORIGINAL))
    import binding as b
    original_deployment = b.read(ORIGINAL.parent.parent / 'alz-gpu0-one-use-deployment.json')
    original_auth = b.read(original_deployment['authorization_path'])
    # Read-only CPU source/input check, independent of an expired GPU grant.
    b.source_closure(original_auth, b.NATIVE_LAYOUT)
    sys.path.extend(map(str, b.native_site_roots(original_auth['cell'], original_auth['files_SHA256'])))
    sys.path[:0] = [str(ROOT), str(b.SCIENCE)]
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from pg_adapter import Context
    from generation import iter_resampling
    from calibration import evaluate, average, dump
    ctx = Context(b.SCIENCE / 'caches/fold0', 'cpu')
    for seed in [1, 2]:
        out = ROOT / 'artifacts' / f'calibration_resampling_seed{seed}'
        records = [evaluate(ctx, 'hierarchical_resampling', seed, rep,
                            lambda schedule, c, s: iter_resampling(ctx, schedule, c, s),
                            out / f'r{rep}') for rep in [0, 1]]
        result = average(records)
        result.update(status='CPU_BASELINE_COMPLETED', model='hierarchical_resampling',
                      new_generative_model_fitted=False, CUDA_context_initialized=torch.cuda.is_initialized())
        assert not torch.cuda.is_initialized()
        dump(ROOT / 'analysis' / f'baseline_calibration_seed{seed}.json', result)
        print(json.dumps({'seed': seed, 'ES': result['energy_score'],
                          'cellED': result['cell_energy_distance'], 'spread': result['spread_ratio']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
