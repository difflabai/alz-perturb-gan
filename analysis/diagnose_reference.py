"""Calibration-only moment diagnostics from retained fixed sampling arrays."""
import json
from pathlib import Path

import numpy as np

ROOT = Path('/ml2/alz')


def profile(generated, real):
    gc, rc = np.cov(generated.T), np.cov(real.T)
    eigen = np.maximum(np.linalg.eigvalsh(gc), 0)
    return {
        'PC_mean_error_L2': float(np.linalg.norm(generated.mean(0) - real.mean(0))),
        'covariance_relative_Frobenius_error': float(np.linalg.norm(gc - rc) / np.linalg.norm(rc)),
        'cell_covariance_trace_ratio': float(np.trace(gc) / np.trace(rc)),
        'cell_covariance_participation_rank': float(eigen.sum() ** 2 / np.square(eigen).sum()),
    }


def main():
    target = np.load(ROOT / 'data/calibration_fold0/real_summaries.npz', allow_pickle=False)
    real = np.load(ROOT / 'data/calibration_fold0/cell_reservoir.npz', allow_pickle=False)['by_stage']
    records = []
    for rep in [0, 1]:
        paths = {
            'PG_reference': ROOT / f'runs/CAL_GPU0_V1_reference_eval_000_f0_pg_s0/r{rep}/arrays.npz',
            'hierarchical_resampling': ROOT / f'artifacts/calibration_resampling_seed0/r{rep}/arrays.npz',
        }
        for method, path in paths.items():
            q = np.load(path, allow_pickle=False)
            cells, summaries = q['generated_cells_by_stage'], q['stage_synthetic_summaries']
            for stage in range(4):
                index = int(np.flatnonzero(target['stage'] == stage)[0])
                composition = np.square(summaries[stage, :, :6]).mean(0)
                true_composition = np.square(target['summary'][index, :6])
                records.append({
                    'method': method, 'replicate': rep, 'stage': stage,
                    'donor': str(target['donors'][index]),
                    **profile(cells[stage], real[stage]),
                    'composition_L1': float(np.abs(composition - true_composition).sum()),
                    'generated_subtype_proportions': composition.tolist(),
                    'calibration_subtype_proportions': true_composition.tolist(),
                    'donor_summary_variance_trace': float(summaries[stage].var(0, ddof=1).sum()),
                })
    means = []
    keys = ['PC_mean_error_L2', 'covariance_relative_Frobenius_error',
            'cell_covariance_trace_ratio', 'cell_covariance_participation_rank',
            'composition_L1', 'donor_summary_variance_trace']
    for method in ['PG_reference', 'hierarchical_resampling']:
        for stage in range(4):
            rows = [r for r in records if r['method'] == method and r['stage'] == stage]
            means.append({'method': method, 'stage': stage,
                          **{key: float(np.mean([r[key] for r in rows])) for key in keys}})
    result = {'status': 'CALIBRATION_REFERENCE_DIAGNOSTICS', 'records': records,
              'replicate_means': means,
              'scope': 'Four calibration donors, frozen optimization-only PCA/classifier; empirical diagnostics, no causal attribution or independent confirmation.',
              'sampling_limit': 'Each cell reservoir has375rows; stage3real donor has208cells and was sampled with replacement. No claim of375independent cells.'}
    (ROOT / 'analysis/reference_distribution_diagnostics.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(means, indent=2))


if __name__ == '__main__':
    main()
