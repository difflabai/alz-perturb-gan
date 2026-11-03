"""Recover fixed reservoir positions and predict a recorded mixture grid exactly."""
import json
import os
from pathlib import Path
import sys

ROOT = Path('/ml2/alz')
B = Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark')
SCIENCE = Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/core-comparison-20261103')


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    sys.path.extend([str(B / 'current_base_20261029/runtime/lib/python3.12/site-packages'), str(B / '.venv312/lib/python3.12/site-packages')])
    sys.path[:0] = [str(ROOT), str(SCIENCE), str(B)]
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from evaluation import Reservoir
    from calibration import score
    from hybrid import bag_mask
    fractions = [.50, .55, .60, .65, .70, .75, .80, .85, .90, .95, 1.0]
    predictions = {fraction: [] for fraction in fractions}
    verified = []
    for rep in [0, 1]:
        pg = np.load(ROOT / f'runs/CAL_GPU0_V1_latent8_v1_f0_pg_s0/r{rep}/arrays.npz', allow_pickle=False)
        baseline = np.load(ROOT / f'artifacts/calibration_resampling_seed0/r{rep}/arrays.npz', allow_pickle=False)
        actual = np.load(ROOT / f'runs/CAL_GPU0_V1_hybrid65_v1_f0_pg_s0/r{rep}/arrays.npz', allow_pickle=False)
        indices = []
        for stage in range(4):
            reservoir = Reservoir(seed=20261106 + 10 * rep + stage, size=375)
            for start in range(0, 200, 5):
                reservoir.add(np.arange(start * 400, (start + 5) * 400, dtype=np.float32).reshape(5, 400, 1))
            indices.append(reservoir.cells[:, 0].astype(int) // 400)
        for fraction in fractions:
            masks = np.stack([bag_mask(fraction, rep, stage) for stage in range(4)])
            y = np.where(masks[:, :, None], pg['stage_synthetic_summaries'], baseline['stage_synthetic_summaries'])
            cell_mask = np.stack([masks[s, indices[s]] for s in range(4)])
            cells = np.where(cell_mask[:, :, None], pg['generated_cells_by_stage'], baseline['generated_cells_by_stage'])
            if fraction == .65:
                assert np.array_equal(y, actual['stage_synthetic_summaries'])
                assert np.array_equal(cells, actual['generated_cells_by_stage'])
                verified.append(rep)
            predictions[fraction].append(score(y, cells))
    incumbent = json.loads((ROOT / 'incumbent.json').read_text())
    baseline = json.loads((ROOT / 'baseline_calibration.json').read_text())
    rows = []
    for fraction in fractions:
        rows.append({'pg_bag_fraction': fraction,
                     **{key: float(np.mean([q[key] for q in predictions[fraction]]))
                        for key in ['energy_score', 'cell_energy_distance', 'spread_ratio', 'coverage80']}})
        row = rows[-1]
        row['joint_improvement_over_incumbent'] = row['energy_score'] < incumbent['energy_score'] - 1e-9 and row['cell_energy_distance'] <= incumbent['cell_energy_distance'] + 1e-9 and .7 <= row['spread_ratio'] <= 1.3
        row['seed0_engineering_target'] = row['energy_score'] <= .95 * baseline['energy_score'] and row['cell_energy_distance'] <= baseline['cell_energy_distance'] and .7 <= row['spread_ratio'] <= 1.3
    assert verified == [0, 1] and not torch.cuda.is_initialized()
    result = {'status': 'EXACT_SAVED_ARRAY_MIXTURE_PREDICTIONS_PENDING_NATIVE_COUNT_VERIFICATION',
              'native65percent_both_replicates_bitwise_reproduced': True,
              'grid': rows,
              'method': 'Reconstruct original priority-reservoir bag positions using the exact runtime/seed/chunk geometry; mix corresponding saved summaries/cells. Model and target draws unchanged.',
              'limitation': 'Predictions are calibration-selected. A chosen setting still needs guarded native counts, checkpoint and cleanup verification before acceptance; no evaluation data or independent confirmation.'}
    (ROOT / 'analysis/hybrid_exact_grid_predictions.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
