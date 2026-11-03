"""Compare a settled pure candidate with PG8 using fitting-derived queries only."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

from compare_semantic_query_fit import energy, load
from diagnose_cell_covariance_fit import metrics as covariance_metrics

ROOT = Path('/ml2/alz')
IDENTITY = ['donors', 'training_seed', 'sampling_contract',
            'calibration_manifest_SHA256', 'trace_manifest_SHA256']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt', required=True)
    args = parser.parse_args()
    evidence_path = ROOT / 'analysis' / (args.attempt + '-result.json')
    evidence = json.loads(evidence_path.read_text())
    assert evidence['attempt'] == args.attempt
    assert evidence['independent_saved_scores_checkpoint_and_count_exemplars_verified']
    candidate = evidence['candidate']
    assert candidate['pure_only'] and candidate['real_bag_replacements'] == 0
    assert candidate['status'] == 'completed' and candidate['completed_updates'] == 20000
    assert sha(Path(candidate['checkpoint_path'])) == candidate['checkpoint_SHA256']
    baseline = ROOT / 'artifacts/calibration_resampling_seed0_validation_replay'
    pins = {str(evidence_path): sha(evidence_path)}
    for rep in (0, 1):
        original = ROOT / 'artifacts/calibration_resampling_seed0' / f'r{rep}/arrays.npz'
        replay = baseline / f'r{rep}/arrays.npz'
        with np.load(original, allow_pickle=False) as a, np.load(replay, allow_pickle=False) as b:
            for key in ['stage_synthetic_summaries', 'generated_cells_by_stage']:
                np.testing.assert_array_equal(a[key], b[key])
        pins[str(original)] = sha(original)
    records = []
    for name, folder in [('PG8', ROOT / 'runs/CAL_GPU0_V1_latent8_v1_f0_pg_s0'),
                         ('candidate', ROOT / 'runs' / args.attempt)]:
        replicates = []
        for rep in (0, 1):
            a, ac, am = load(folder, rep, pins)
            b, bc, bm = load(baseline, rep, pins)
            for key in IDENTITY:
                assert am[key] == bm[key] == candidate[key]
            stages = []
            for stage in range(4):
                pa, pb = np.square(a[stage, :, :6]).mean(0), np.square(b[stage, :, :6]).mean(0)
                stages.append({'stage': stage,
                               'composition_L1': float(np.abs(pa - pb).sum()),
                               'donor_PC_mean_error_L2': float(np.linalg.norm(a[stage, :, 6:].mean(0) - b[stage, :, 6:].mean(0))),
                               'donor_summary_energy_distance_V': energy(a[stage], b[stage]),
                               'cell_energy_distance_V': energy(ac[stage], bc[stage]),
                               **covariance_metrics(ac[stage].astype(np.float64), bc[stage].astype(np.float64))})
            replicates.append({'replicate': rep, 'stages': stages,
                               'pooled_cell_energy_distance_V': energy(ac.reshape(-1, 50), bc.reshape(-1, 50))})
        means = []
        for stage in range(4):
            values = [rep['stages'][stage] for rep in replicates]
            means.append({'stage': stage, **{key: float(np.mean([q[key] for q in values]))
                                             for key in values[0] if key != 'stage'}})
        records.append({'method': name, 'replicates': replicates, 'stage_replicate_means': means,
                        'pooled_cell_energy_distance_V_mean': float(np.mean([q['pooled_cell_energy_distance_V'] for q in replicates]))})
    changes = []
    for stage in range(4):
        old, new = [q['stage_replicate_means'][stage] for q in records]
        changes.append({'stage': stage, **{key: new[key] - old[key] for key in old if key != 'stage'}})
    result = {'status': 'SETTLED_PURE_CANDIDATE_FITTING_QUERY_DIAGNOSTIC',
              'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'attempt': args.attempt,
              'candidate_checkpoint_SHA256': candidate['checkpoint_SHA256'],
              'records': records, 'candidate_minus_PG8': changes, 'source_arrays_and_evidence_SHA256': pins,
              'scope': 'Generated summaries/cells versus matched optimization-donor resampling only. No calibration or evaluation target arrays read; no generation, fitting or GPU allocation. The independently verified final calibration decision is unchanged.',
              'limits': 'Descriptive V-statistic distances and finite 375-cell/stage covariance, not independent validation or causal evidence. Offdiagonal sums have 2450 entries versus 50 diagonal entries. Replicate queries differ; their contrast is not a pure sampling-noise floor.',
              'source_SHA256': {str(p): sha(p) for p in [Path(__file__).resolve(), ROOT / 'analysis/compare_semantic_query_fit.py', ROOT / 'analysis/diagnose_cell_covariance_fit.py']}}
    output = ROOT / 'analysis' / (args.attempt + '-query-fit.json')
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'attempt': args.attempt, 'candidate_minus_PG8': changes,
                      'pooled_fit_cell_ED': {q['method']: q['pooled_cell_energy_distance_V_mean'] for q in records}}, indent=2))


if __name__ == '__main__':
    main()
