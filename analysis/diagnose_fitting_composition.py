"""Compare retained generated subtype mixtures under the fitting-derived query law."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path('/ml2/alz')


def main():
    rows, sources = [], {}
    for stage in range(4):
        pg, resampling = [], []
        for rep in [0, 1]:
            paths = [ROOT / f'runs/CAL_GPU0_V1_latent8_v1_f0_pg_s0/r{rep}/arrays.npz',
                     ROOT / f'artifacts/calibration_resampling_seed0/r{rep}/arrays.npz']
            for path, group in zip(paths, [pg, resampling]):
                sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
                group.append(np.load(path, allow_pickle=False)['stage_synthetic_summaries'][stage])
        a, b = np.concatenate(pg), np.concatenate(resampling)
        pa, pb = np.square(a[:, :6]).mean(0), np.square(b[:, :6]).mean(0)
        rows.append({'stage': stage, 'PG8_generated_composition': pa.tolist(),
                     'matched_optimization_resampling_composition': pb.tolist(),
                     'composition_L1': float(np.abs(pa - pb).sum()),
                     'PC_mean_error_L2': float(np.linalg.norm(a[:, 6:].mean(0) - b[:, 6:].mean(0)))})
    result = {'status': 'FITTING_QUERY_COMPOSITION_DIAGNOSTIC', 'records': rows,
              'source_arrays_SHA256': sources,
              'scope': 'Generated arrays only,400bags perstage across2fixedreplicates. Query donor/condition/library law and resampling derive from optimization donors. No calibration or evaluation target arrays parsed. KeptPG8 was adaptively selected previously; no independence/causal claim.',
              'hypothesis_if_further_iteration_needed': 'Fixed low-frequency Fourier and moment observations may leave subtype composition underconstrained. A prospective critic observation extension using the frozen optimization-only subtype classifier could be tested, with model-choice confounds explicit. No implementation or allocation yet.'}
    (ROOT / 'analysis/fitting_composition_mismatch.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
