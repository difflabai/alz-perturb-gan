"""Independently check saved first-fold scores; never loads a model or generates."""
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

ROOT = Path('/ml2/alz')
SCIENCE = Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/core-comparison-20261103')
OUT = SCIENCE / 'runs_gpu0_v1/GPU0_V1_f0_pg_s0'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    complete = read(OUT / 'completed.json')
    assert complete['status'] == 'completed'
    assert (complete['fold'], complete['seed'], complete['method'], complete['endpoint']) == (0, 0, 'pg', 20000)
    checkpoint = read(OUT / 'fit/checkpoint_receipt.json')
    assert checkpoint['status'] == 'completed' and checkpoint['completed_updates'] == 20000
    assert sha(OUT / 'fit' / checkpoint['path']) == checkpoint['SHA256']
    cache_sha = sha(SCIENCE / 'caches/fold0/manifest.json')
    folders = ['evaluation_r0', 'resampling_r0', 'evaluation_r1',
               'resampling_r1', 'stage_swap_r0']
    metrics = {}
    checks = {}
    for name in folders:
        root = OUT / name
        q = read(root / 'metrics.json')
        assert q['status'] == 'completed' and q['fold'] == q['training_seed'] == 0
        assert q['cache_manifest_SHA256'] == cache_sha
        assert q['arrays_SHA256'] == sha(root / 'arrays.npz')
        assert q['diagnostic_only'] == (name == 'stage_swap_r0')
        assert q['generation_query_keys'] == ['stage']
        with np.load(root / 'arrays.npz', allow_pickle=False) as arrays:
            y = arrays['stage_synthetic_summaries']
            real = arrays['real_summaries']
            stage = arrays['stage']
            assert y.shape == (4, 200, 26) and real.shape == (20, 26)
            assert len(set(arrays['donors'].tolist())) == 20
            assert arrays['donors'].tolist() == q['donors']
            assert np.isfinite(y).all() and np.isfinite(real).all()
            scores = np.array([
                np.linalg.norm(y[s] - target, axis=1).mean()
                - 0.5 * cdist(y[s], y[s]).sum() / (200 * 199)
                for s, target in zip(stage, real)
            ])
            np.testing.assert_allclose(scores, q['per_donor_energy_score'], rtol=0, atol=1e-10)
            np.testing.assert_allclose(scores.mean(), q['energy_score'], rtol=0, atol=1e-10)
            balanced = np.mean([scores[stage == s].mean() for s in range(4)])
            np.testing.assert_allclose(balanced, q['stage_balanced_energy_score'], rtol=0, atol=1e-10)
            a, b = arrays['generated_cell_reservoir'], arrays['real_cell_reservoir']
            distance = 2 * cdist(a, b).mean() - cdist(a, a).mean() - cdist(b, b).mean()
            np.testing.assert_allclose(distance, q['cell_energy_distance'], rtol=0, atol=1e-10)
        for stage in range(4):
            with np.load(root / f'stage{stage}_trace.npz', allow_pickle=False) as t:
                assert int(t['target_stage']) == stage
                assert int(t['model_query_stage']) == ((stage + 2) % 4 if name == 'stage_swap_r0' else stage)
                source = SCIENCE / f'traces/f0_s0_r{q["generation_replicate"]}_stage{stage}.npz'
                with np.load(source, allow_pickle=False) as frozen:
                    for actual, expected in [('donor_index', 'donors'), ('row', 'rows'), ('exposure', 'libraries')]:
                        assert np.array_equal(t[actual], frozen[expected])
            exemplar = root / f'stage{stage}_count_exemplar.npz'
            if exemplar.exists():
                with np.load(exemplar, allow_pickle=False) as e:
                    counts = e['counts']
                    assert counts.shape == (1, 400, 2000)
                    assert np.isfinite(counts).all() and (counts >= 0).all()
                    assert np.equal(counts, np.floor(counts)).all()
        metrics[name] = q
        checks[name] = {'saved_array_hash_verified': True,
                        'per_donor_and_mean_energy_scores_recomputed': True,
                        'stage_balanced_energy_score_recomputed': True,
                        'cell_energy_distance_recomputed': True,
                        'shared_trace_matches_frozen_inputs': True,
                        'saved_count_exemplars_valid': True}
    pg = np.mean([metrics[f'evaluation_r{rep}']['per_donor_energy_score'] for rep in range(2)], axis=0)
    baseline = np.mean([metrics[f'resampling_r{rep}']['per_donor_energy_score'] for rep in range(2)], axis=0)
    descriptive = {
        'PG_mean_energy_score': float(pg.mean()),
        'resampling_mean_energy_score': float(baseline.mean()),
        'PG_minus_resampling_energy_score': float((pg - baseline).mean()),
        'relative_improvement_vs_resampling': float(1 - pg.mean() / baseline.mean()),
        'donors_with_lower_PG_energy_score': int((pg < baseline).sum()),
        'evaluated_donors': len(pg),
        'PG_mean_cell_energy_distance': float(np.mean([metrics[f'evaluation_r{r}']['cell_energy_distance'] for r in range(2)])),
        'resampling_mean_cell_energy_distance': float(np.mean([metrics[f'resampling_r{r}']['cell_energy_distance'] for r in range(2)])),
        'PG_mean_spread_ratio': float(np.mean([metrics[f'evaluation_r{r}']['spread_ratio'] for r in range(2)])),
        'resampling_mean_spread_ratio': float(np.mean([metrics[f'resampling_r{r}']['spread_ratio'] for r in range(2)])),
        'stage_swap_energy_score': metrics['stage_swap_r0']['energy_score'],
        'stage_swap_minus_original_replicate0_ES': metrics['stage_swap_r0']['energy_score'] - metrics['evaluation_r0']['energy_score'],
    }
    result = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'status': 'ONE_FOLD_ONE_SEED_COMPLETED_SAVED_SCORES_VERIFIED',
              'scope': 'First fold (index0), seed0; two generation replicates averaged within each of20real donors. Lower energy score/distance is better.',
              'checkpoint': checkpoint, 'verification': checks,
              'descriptive_comparison': descriptive, 'metrics': metrics,
              'limitations': ['One of30planned core matrix cells; no complete-matrix advancement or superiority conclusion.',
                              'No matched scVI result exists for this new fold/seed.',
                              'Historically exposed cohort; no independent biological confirmation.',
                              'Stage swap is a prespecified diagnostic only.']}
    (ROOT / 'first_fold_results.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'descriptive_comparison': descriptive}, indent=2))


if __name__ == '__main__':
    main()
