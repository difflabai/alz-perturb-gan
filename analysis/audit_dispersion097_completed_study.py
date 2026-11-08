"""Independent saved-array audit after all six fresh theta fits have finished.

Preparation is not an ALZ result. Readiness never decodes partial quality files.
This audit allocates no GPU and imports no model or training package.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path('/ml2/alz')
PROJECT = Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark')
sys.path.insert(0, str(ROOT / 'analysis'))
METRICS = ('cell_energy_distance', 'energy_score', 'spread_ratio',
           'within_condition_spread_ratio', 'coverage80')
SEEDS = (8, 9, 10)
ARMS = ('theta1', 'theta5')


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(a, b):
    assert math.isfinite(float(a)) and math.isfinite(float(b))
    assert math.isclose(float(a), float(b), rel_tol=0, abs_tol=1e-10), (a, b)


def required_paths(root=ROOT):
    from audit_released097_completed_studies import required_paths as earlier_paths
    root = Path(root)
    return [*earlier_paths(root), *[root / p for p in (
        'analysis/dispersion097/matrix_complete.json',
        'analysis/dispersion097/comparison.json',
        'artifacts/dispersion097-after-scalarcal097/completed.json')]]


def readiness(root=ROOT):
    missing = [str(p) for p in required_paths(root) if not p.is_file()]
    return dict(status='INCOMPLETE_PRIOR_AND_SIX_FRESH_THETA_STUDIES' if missing
                else 'INPUT_PATHS_PRESENT_NOT_A_COMPLETION_CLAIM',
                missing_paths=missing, quality_values_read=False,
                real_theta_fits_audited=0, full_goal_complete=False, SOTA_claim=False)


def independent_metrics(y, cells, real, stages, equal_stage, by_stage):
    """Use condensed pair sums: ED is V-statistic; donor ES is U-statistic."""
    import numpy as np
    from scipy.spatial.distance import pdist
    from audit_released097_completed_studies import independent_cell_ED
    y, cells, real, stages = map(np.asarray, (y, cells, real, stages))
    assert y.ndim == cells.ndim == 3 and y.shape[0] == cells.shape[0] == 4
    assert y.shape[1] > 1 and real.shape == (4, y.shape[2])
    assert stages.shape == (4,) and set(stages.tolist()) == set(range(4))
    assert all(np.isfinite(a).all() for a in (y, cells, real, equal_stage, by_stage))
    n = y.shape[1]
    es = [float(np.linalg.norm(y[s] - target, axis=1).mean()
                - pdist(y[s]).sum() / (n * (n - 1)))
          for s, target in zip(stages, real)]
    reference_variance = real.var(0, ddof=1).sum()
    assert reference_variance > 0
    low, high = np.percentile(y[stages], [10, 90], axis=1)
    return dict(
        cell_energy_distance=independent_cell_ED(cells.reshape(-1, cells.shape[-1]), equal_stage),
        per_stage_cell_energy_distance=[independent_cell_ED(cells[s], by_stage[s]) for s in range(4)],
        per_donor_energy_score=es, energy_score=float(np.mean(es)),
        spread_ratio=float(np.sqrt(y.reshape(-1, y.shape[-1]).var(0, ddof=1).sum() / reference_variance)),
        within_condition_spread_ratio=float(np.sqrt(y.var(1, ddof=1).mean(0).sum() / reference_variance)),
        coverage80=float(((real >= low) & (real <= high)).mean()),
        per_stage_composition_L1=[float(np.abs(np.square(y[s, :, :6]).mean(0)
            - np.square(real[np.flatnonzero(stages == s)[0], :6])).sum()) for s in range(4)])


def compare_metrics(actual, expected):
    import numpy as np
    for name, value in actual.items():
        np.testing.assert_allclose(value, expected[name], rtol=0, atol=1e-10, err_msg=name)


def count_contract(counts):
    import numpy as np
    assert counts.shape == (1, 400, 2000)
    assert np.isfinite(counts).all() and (counts >= 0).all()
    assert np.equal(counts, np.floor(counts)).all()


def array_fingerprint(arrays):
    import numpy as np
    h = hashlib.sha256()
    for name in sorted(arrays):
        a = np.ascontiguousarray(arrays[name])
        h.update(json.dumps([name, a.dtype.str, a.shape]).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def validate_summary(rows, report):
    """Recompute paired comparisons; never substitute ES for the declared ED."""
    assert len(rows) == 6 and {(r['arm'], r['seed']) for r in rows} == {
        (a, s) for a in ARMS for s in SEEDS}
    by = {(r['arm'], r['seed']): r for r in rows}
    means = {a: {k: statistics.mean(by[a, s]['metrics'][k] for s in SEEDS)
                 for k in METRICS} for a in ARMS}
    expected = []
    for seed in SEEDS:
        assert by['theta1', seed]['baseline_metrics'] == by['theta5', seed]['baseline_metrics']
        expected.append(dict(seed=seed, **{k: by['theta5', seed]['metrics'][k]
                         - by['theta1', seed]['metrics'][k] for k in METRICS}))
    assert len(report['paired_theta5_minus_theta1']) == 3
    for a in ARMS:
        for k in METRICS:
            close(means[a][k], report['mean_metrics_by_arm'][a][k])
    for got, wanted in zip(report['paired_theta5_minus_theta1'], expected):
        assert got['seed'] == wanted['seed']
        for k in METRICS:
            close(got[k], wanted[k])
    for k in METRICS:
        close(statistics.mean(r[k] for r in expected), report['paired_mean_differences'][k])
    winner = 'theta5' if means['theta5']['cell_energy_distance'] < means['theta1']['cell_energy_distance'] else 'theta1'
    assert report['calibration_preferred_arm'] == winner
    assert report['theta5_ED_paired_wins'] == sum(r['cell_energy_distance'] < 0 for r in expected)
    return dict(calibration_preferred_arm=winner, mean_metrics_by_arm=means,
                paired_theta5_minus_theta1=expected,
                paired_training_seed_SD={k: statistics.stdev(r[k] for r in expected) for k in METRICS},
                PG_minus_matched_resampling_by_seed=[dict(seed=s, arm=a, **{
                    k: by[a, s]['metrics'][k] - by[a, s]['baseline_metrics'][k] for k in METRICS})
                    for s in SEEDS for a in ARMS])


def audit(output):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    ready = readiness()
    assert not ready['missing_paths'], 'All prior studies and all six fresh theta outcomes must finish before numerical reads'
    sys.path[:0] = [str(ROOT / 'dispersion097_runtime'), str(ROOT / 'analysis'),
        str(PROJECT / 'current_base_20261029/runtime/lib/python3.12/site-packages'),
        str(PROJECT / '.venv312/lib/python3.12/site-packages'), str(ROOT)]
    import numpy as np
    import preset_scope as scope
    from cell_metadata import completed_metadata
    from calibration import targets
    from audit_released097_completed_studies import verify_reverted_plan
    assert Path(scope.__file__).resolve() == ROOT / 'dispersion097_runtime/preset_scope.py'
    scope.protocol()
    inputs = scope.prior_study_ready()
    inputs.update({str(p): sha(p) for p in required_paths()})
    plans = {name: verify_reverted_plan(ROOT / name / 'plan.json') for name in (
        'benchmark097_dual_runtime', 'benchmark097_balanced_runtime',
        'benchmark097_infra1_runtime', 'scalarcal097_runtime', 'dispersion097_runtime')}
    assert not read(PROJECT / 'project_gpu_budget.json')['open_reservations']
    matrix = read(scope.STUDY / 'matrix_complete.json')
    report = read(scope.STUDY / 'comparison.json')
    parent = read(required_paths()[-1])
    assert parent['status'] == 'DISPERSION097_COMPLETE_CALIBRATION_REPORT_FINAL_RESEARCH_AUDIT_REMAINS'
    assert parent['report_path'] == str(scope.STUDY / 'comparison.json')
    expected_keys = {f'CAL_GPU0_V1_dispersion097_{a}_f0_pg_s{s}' for s in SEEDS for a in ARMS}
    assert matrix['matrix_complete'] and matrix['complete_cells'] == matrix['required_cells'] == 6
    assert matrix['status'] == 'ALL_SIX_FRESH_THETA_INIT_NATIVE_MODELS_AND_CALIBRATION_AUDITS_COMPLETE'
    assert matrix['all_new_training_from_scratch'] and matrix['initial_completed_updates'] == 0 and matrix['calibration_only']
    assert set(matrix['result_SHA256']) == expected_keys
    assert report['status'] == 'ALL_SIX_FRESH_THETA_INITIALIZATION_FITS_CALIBRATION_COMPARISON_COMPLETE'
    assert report['complete_fresh_fits'] == 6 and report['training_seeds'] == list(SEEDS) and report['updates_per_fit'] == 20000
    assert report['result_SHA256'] == matrix['result_SHA256'] and report['matrix_SHA256'] == sha(scope.STUDY / 'matrix_complete.json')
    assert report['protocol_SHA256'] == matrix['protocol_SHA256'] == sha(scope.PROTOCOL)
    assert report['source_SHA256'] == sha(ROOT / 'analysis/aggregate_dispersion097.py')
    assert all(report[k] for k in ('fresh_model_initializations', 'four_reused_calibration_donors',
                                  'calibration_only', 'external45_candidate_unchanged', 'negative_results_retained'))
    assert not report['full_goal_complete'] and not report['SOTA_claim']
    manifest, ys, real_cells = targets()
    try:
        target = {k: ys[k].copy() for k in ys.files}
        references = {k: real_cells[k].copy() for k in real_cells.files}
    finally:
        ys.close()
        real_cells.close()
    assert len(set(target['donors'].tolist())) == 4
    assert sha(ROOT / 'data/calibration_fold0/manifest.json') == scope.protocol()['calibration_manifest_SHA256']
    inputs[str(ROOT / 'data/calibration_fold0/manifest.json')] = sha(ROOT / 'data/calibration_fold0/manifest.json')
    for name, pin in manifest['files_SHA256'].items():
        inputs[str(ROOT / 'data/calibration_fold0' / name)] = pin
    rows, charges, control_arrays, endpoint_arrays, exemplars = [], [], {}, 0, 0
    for seed in SEEDS:
        for arm in ARMS:
            value = scope.cell(arm, seed)
            meta = completed_metadata(value)
            path = Path(meta['source_result_path'])
            assert matrix['result_SHA256'][meta['attempt']] == sha(path)
            q = read(path)
            assert q['complete'] and q['cell'] == value and q['arm'] == arm and q['seed'] == seed
            assert q['all_new_training_from_scratch'] and q['initial_completed_updates'] == 0 and q['source_checkpoint'] is None
            assert q['independent_CPU_saved_score_audit'] and q['native_saved_state_matches_GPU_restore']
            assert q['benchmark_protocol_SHA256'] == sha(scope.PROTOCOL) and q['calibration_only']
            close(q['charged_GPU_seconds'], meta['charged_GPU_seconds'])
            assert math.isfinite(meta['charged_GPU_seconds']) and meta['charged_GPU_seconds'] > 0
            charges.append(meta['charged_GPU_seconds'])
            a = read(meta['authorization_path'])
            assert a['source_model_path'] is a['source_model_SHA256'] is None and not a['is_continuation'] and not a['reuse_completed_model']
            assert a['cell'] == value and a['arm'] == arm and a['particlegan_version'] == '0.9.7'
            assert a['fixed_endpoint'] == a['new_model_updates'] == 20000 and a['initial_completed_updates'] == 0
            assert a['released_hyperparameters'] == scope.hyperparameters() and a['variant_config'] == scope.configuration(arm)
            assert a['metric_role'] == 'calibration_selection'
            for name, pin in a['files_SHA256'].items():
                if Path(name) == scope.PLAN:
                    active_bytes = subprocess.check_output(['git', 'show', plans['dispersion097_runtime'] + ':dispersion097_runtime/plan.json'], cwd=ROOT)
                    assert hashlib.sha256(active_bytes).hexdigest() == pin
                else:
                    assert sha(name) == pin, name
                inputs[name] = sha(name)
            for name, pin in q['files_SHA256'].items():
                assert sha(name) == pin, name
            inputs.update(q['files_SHA256'])
            inputs.update({str(p): sha(p) for p in (path, Path(meta['authorization_path']), Path(a['ledger_path']))})
            ledger = read(a['ledger_path'])['jobs'][0]
            assert ledger['lease_token'] not in read(Path(a['registry_path']) / 'registry.json')['jobs']
            settlement = Path(a['ledger_path']).with_name(meta['attempt'] + '-settlement.json')
            inputs[str(settlement)] = sha(settlement)
            out = Path(a['output_dir'])
            receipt, restored = read(out / 'completed.json'), read(out / 'native_restore.json')
            assert receipt['status'] == 'completed' and receipt['ParticleGAN_version'] == '0.9.7'
            assert receipt['endpoint'] == receipt['new_model_updates'] == 20000
            assert receipt['arm'] == arm and receipt['seed'] == seed and receipt['cell'] == value
            assert receipt['from_scratch'] and receipt['training_finished_before_calibration'] and not receipt['heldout_values_used']
            assert receipt['initial_completed_updates'] == 0 and receipt['source_checkpoint'] is None
            assert receipt['protocol_SHA256'] == sha(scope.PROTOCOL)
            assert receipt['hyperparameters'] == a['released_hyperparameters'] and receipt['variant_config'] == a['variant_config']
            assert sha(receipt['checkpoint_path']) == receipt['checkpoint_SHA256'] == restored['checkpoint_SHA256']
            assert receipt['policy_fingerprint'] == restored['policy_fingerprint']
            assert restored['status'] == 'EXACT_NATIVE_DISPERSION_PG_CUDA_RESTORE_PASS' and restored['completed_updates'] == 20000
            assert restored['arm'] == arm
            for prefix, metric_key, mean_key in [('r', 'replicate_metrics', 'metrics'),
                                                  ('resampling_r', 'baseline_replicate_metrics', 'baseline_metrics')]:
                computed = []
                for rep in (0, 1):
                    dest = out / (prefix + str(rep))
                    saved = read(dest / 'metrics.json')
                    assert saved == q[metric_key][rep] == receipt[metric_key][rep]
                    assert saved['status'] == 'completed' and saved['training_seed'] == seed and saved['generation_replicate'] == rep
                    assert saved['metric_role'] == 'calibration_selection' and saved['generated_cells'] == 320000
                    assert saved['donors'] == target['donors'].tolist() and saved['stage'] == target['stage'].tolist()
                    assert saved['trace_manifest_SHA256'] == sha(scope.PROTOCOL)
                    assert saved['calibration_manifest_SHA256'] == sha(ROOT / 'data/calibration_fold0/manifest.json')
                    assert saved['not_independent_confirmation'] and saved['sampling_contract'] == 'calibration_equal_stage375_priority_v1'
                    assert sha(dest / 'arrays.npz') == saved['arrays_SHA256']
                    with np.load(dest / 'arrays.npz', allow_pickle=False) as arrays:
                        y, c = arrays['stage_synthetic_summaries'], arrays['generated_cells_by_stage']
                        assert y.shape == (4, 200, 26) and c.shape == (4, 375, 50)
                        metrics = independent_metrics(y, c, target['summary'], target['stage'], references['equal_stage'], references['by_stage'])
                        if prefix == 'resampling_r':
                            pin = array_fingerprint(arrays)
                            if (seed, rep) in control_arrays:
                                assert control_arrays[seed, rep] == pin, 'Matched arms must have identical resampling arrays'
                            control_arrays[seed, rep] = pin
                    compare_metrics(metrics, saved)
                    computed.append(metrics)
                    endpoint_arrays += 1
                    if prefix == 'r':
                        for stage in range(4):
                            with np.load(dest / f'stage{stage}_exemplar.npz', allow_pickle=False) as ex:
                                count_contract(ex['counts'])
                            exemplars += 1
                for key in (*METRICS, 'per_donor_energy_score', 'per_stage_cell_energy_distance'):
                    np.testing.assert_allclose(np.mean([m[key] for m in computed], axis=0), q[mean_key][key], rtol=0, atol=1e-10)
                assert receipt[mean_key] == q[mean_key]
            rows.append(q)
    summary = validate_summary(rows, report)
    expected_outcomes = [dict(arm=q['arm'], seed=q['seed'], metrics=q['metrics'],
        baseline_metrics=q['baseline_metrics'], charged_GPU_seconds=q['charged_GPU_seconds'])
        for q in sorted(rows, key=lambda x: (x['seed'], x['arm']))]
    assert report['all_seed_outcomes'] == expected_outcomes
    close(sum(charges), report['actual_GPU_seconds'])
    close(sum(charges), matrix['actual_GPU_seconds'])
    assert endpoint_arrays == 24 and exemplars == 48
    for name, pin in inputs.items():
        assert sha(name) == pin, 'An audited input changed during final verification: ' + name
    assert 'torch' not in sys.modules
    record = dict(status='SIX_FRESH_THETA20K_CALIBRATION_ENDPOINTS_INDEPENDENTLY_AUDITED',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), real_theta_fits_audited=6,
        endpoint_arrays_independently_recomputed=24, integer_count_exemplars_checked=48,
        matched_resampling_array_pairs_checked=6, **summary, actual_GPU_seconds=sum(charges),
        prior_study_hash_barrier_verified=True, prior_studies_not_numerically_reaudited_here=True,
        actual_plan_reversions_verified=plans, source_SHA256={str(Path(__file__).resolve()): sha(__file__),
            str(ROOT / 'analysis/audit_released097_completed_studies.py'): sha(ROOT / 'analysis/audit_released097_completed_studies.py')},
        input_SHA256=inputs, no_new_training_generation_or_GPU_allocation=True,
        native_checkpoint_validation_evidence='Each completed per-cell auditor validated native state/optimizers; this final CPU array audit verifies its pinned evidence and native CUDA-restore receipt without importing Torch.',
        calibration_only=True, fresh_training_seeds_are_not_biological_replication=True,
        old45_candidate_unchanged=True, full_goal_complete=False, SOTA_claim=False,
        limitations=['Only four reused calibration donors; this study provides no independent biological or external SOTA test.',
                    'A preferred theta initialization needs a separately frozen external comparison; do not insert it retrospectively into the existing45.',
                    'Training-seed spread is descriptive optimization variability, not donor-population uncertainty.'])
    output = Path(output)
    assert output.parent == ROOT / 'analysis' and not output.exists()
    output.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-readiness', action='store_true')
    parser.add_argument('--output')
    args = parser.parse_args()
    if args.check_readiness:
        print(json.dumps(readiness(), indent=2))
        return
    assert args.output, 'Explicit new output path required'
    q = audit(args.output)
    print(json.dumps({k: q[k] for k in ('status', 'calibration_preferred_arm', 'actual_GPU_seconds', 'full_goal_complete', 'SOTA_claim')}))


if __name__ == '__main__':
    main()
