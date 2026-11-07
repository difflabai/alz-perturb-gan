"""Read-only replication of the calibration mixture decomposition across four fits."""
import datetime
import hashlib
import json
import os
from pathlib import Path

assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
from diagnose_faithful097_first20k import ROOT, components, np, targets


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    destination = ROOT / 'analysis/faithful097_all4_mixture_diagnostic_20261107.json'
    assert not destination.exists()
    audit_path = ROOT / 'analysis/faithful097_final_independent_audit_20261107.json'
    audit = read(audit_path)
    assert audit['status'] == 'INDEPENDENT_RELEASED097_FRESH_BASELINE4_AUDIT_PASS'
    assert audit['required_attempts'] == 4 and audit['all_new_models_from_update_zero']
    assert audit['all_planned_attempts_resolved'] and not audit['heldout_values_read']
    pins = {str(audit_path): sha(audit_path)}
    _, _, real = targets()
    reference = real['by_stage']
    assert reference.shape == (4, 375, 50)
    np.testing.assert_array_equal(reference.reshape(-1, 50), real['equal_stage'])
    rows = []
    keys = ('pooled_energy_distance', 'weighted_diagonal_contribution',
            'weighted_cross_stage_contribution')
    for attempt, result_hash in sorted(audit['result_SHA256'].items()):
        result_path = ROOT / 'analysis/faithful097/results' / (attempt + '.json')
        assert sha(result_path) == result_hash
        pins[str(result_path)] = result_hash
        result = read(result_path)
        assert result['finite'] and result['clean_settlement'] and result['from_scratch']
        assert result['physical_start_update'] == 0 and result['new_phase_model_updates'] == 20000
        replicates = []
        for replicate in (0, 1):
            row = {'replicate': replicate}
            for name, prefix in [('pg', 'r'), ('resampling', 'resampling_r')]:
                folder = ROOT / 'runs' / attempt / (prefix + str(replicate))
                metric_path, array_path = folder / 'metrics.json', folder / 'arrays.npz'
                metric = read(metric_path)
                assert sha(metric_path) == result['files_SHA256'][str(metric_path)]
                assert sha(array_path) == metric['arrays_SHA256'] == result['files_SHA256'][str(array_path)]
                pins[str(metric_path)] = sha(metric_path)
                pins[str(array_path)] = sha(array_path)
                with np.load(array_path, allow_pickle=False) as saved:
                    generated = saved['generated_cells_by_stage']
                assert generated.shape == (4, 375, 50)
                row[name] = components(generated, reference)
                np.testing.assert_allclose(row[name]['pooled_energy_distance'],
                                           metric['cell_energy_distance'], rtol=0, atol=5e-13)
                np.testing.assert_allclose(row[name]['per_stage_distance'],
                                           metric['per_stage_cell_energy_distance'], rtol=0, atol=5e-13)
            row['PG_minus_resampling'] = {k: row['pg'][k] - row['resampling'][k] for k in keys}
            row['per_stage_distance_delta'] = (np.array(row['pg']['per_stage_distance']) -
                                               np.array(row['resampling']['per_stage_distance'])).tolist()
            replicates.append(row)
        delta = {k: float(np.mean([r['PG_minus_resampling'][k] for r in replicates])) for k in keys}
        stage_delta = np.mean([r['per_stage_distance_delta'] for r in replicates], axis=0)
        np.testing.assert_allclose(delta[keys[0]], result['metrics']['cell_energy_distance'] -
                                   result['baseline_metrics']['cell_energy_distance'], rtol=0, atol=5e-13)
        rows.append(dict(attempt=attempt, candidate_id=result['candidate_id'], seed=result['seed'],
                         delta=delta, mean_per_stage_distance_delta=stage_delta.tolist(),
                         stages_with_lower_distance=int((stage_delta < 0).sum()), replicates=replicates))
    assert {(r['candidate_id'], r['seed']) for r in rows} == {
        (candidate, seed) for candidate in ('f000', 'f001') for seed in (0, 1)}
    summary = []
    for candidate in ('f000', 'f001'):
        subset = [r for r in rows if r['candidate_id'] == candidate]
        summary.append(dict(candidate_id=candidate, seeds=[0, 1],
                            mean_delta={k: float(np.mean([r['delta'][k] for r in subset])) for k in keys},
                            mean_per_stage_distance_delta=np.mean(
                                [r['mean_per_stage_distance_delta'] for r in subset], axis=0).tolist()))
    sources = [Path(__file__).resolve(), ROOT / 'analysis/diagnose_faithful097_first20k.py',
               ROOT / 'calibration.py']
    for path, digest in pins.items():
        assert sha(path) == digest
    output = dict(status='FOUR_FRESH_RELEASED20K_CALIBRATION_MIXTURE_DECOMPOSITIONS_VERIFIED',
                  UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  summary=summary, rows=rows, original_primary_metric_unchanged=True,
                  interpretation='The pooled distance is exactly the sum of16 equally weighted within/cross-stage terms. '
                  'These cross-stage contributions may be negative and are not standalone distances or causal effects. '
                  'This diagnostic does not alter the frozen hyperparameter search or choose a model.',
                  calibration_only=True, evaluation_target_values_read=False,
                  new_model_fits=0, new_optimizer_updates=0, SOTA_claim=False, full_goal_complete=False,
                  source_SHA256={str(p): sha(p) for p in sources}, input_SHA256=pins)
    destination.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': output['status'], 'summary': summary}))


if __name__ == '__main__':
    main()
