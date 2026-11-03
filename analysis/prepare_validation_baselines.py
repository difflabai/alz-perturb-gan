"""CPU-only matched calibration baselines for prespecified seed confirmations."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path('/ml2/alz')
ORIGINAL = Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-11/gpu0-coexistence/integration')


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    parser = argparse.ArgumentParser()
    parser.add_argument('--replay-seed0', action='store_true')
    args = parser.parse_args()
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
    for seed in ([0] if args.replay_seed0 else [1, 2]):
        out = ROOT / 'artifacts' / (f'calibration_resampling_seed{seed}' + ('_validation_replay' if args.replay_seed0 else ''))
        records = [evaluate(ctx, 'hierarchical_resampling', seed, rep,
                            lambda schedule, c, s: iter_resampling(ctx, schedule, c, s),
                            out / f'r{rep}') for rep in [0, 1]]
        result = average(records)
        result.update(status='CPU_BASELINE_COMPLETED', model='hierarchical_resampling',
                      new_generative_model_fitted=False, CUDA_context_initialized=torch.cuda.is_initialized())
        assert not torch.cuda.is_initialized()
        if args.replay_seed0:
            import numpy as np
            for rep in [0, 1]:
                original = np.load(ROOT / 'artifacts/calibration_resampling_seed0' / f'r{rep}/arrays.npz', allow_pickle=False)
                replay = np.load(out / f'r{rep}/arrays.npz', allow_pickle=False)
                assert set(original.files) == set(replay.files)
                for name in original.files:
                    np.testing.assert_array_equal(original[name], replay[name])
            previous = json.loads((ROOT / 'baseline_calibration.json').read_text())
            for key in ['energy_score', 'cell_energy_distance', 'spread_ratio']:
                np.testing.assert_allclose(result[key], previous[key], rtol=0, atol=1e-12)
            result['both_original_seed0_arrays_bitwise_reproduced'] = True
        dump(ROOT / 'analysis' / f'baseline_calibration_seed{seed}.json', result)
        print(json.dumps({'seed': seed, 'ES': result['energy_score'],
                          'cellED': result['cell_energy_distance'], 'spread': result['spread_ratio']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
