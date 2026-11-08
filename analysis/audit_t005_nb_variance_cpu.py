"""Independent saved-array and raw fitting-count audit; no Torch or fitting."""
import datetime
import hashlib
import json
import os
import statistics
import sys
from pathlib import Path

ROOT = Path('/ml2/alz')
PROJECT = Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark')
sys.path[:0] = [str(PROJECT/'current_base_20261029/runtime/lib/python3.12/site-packages'), str(PROJECT/'.venv312/lib/python3.12/site-packages')]
import numpy as np
from scipy import sparse


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    result_path = ROOT/'analysis/t005_nb_variance_CPU_result_20261108.json'
    q = read(result_path)
    plan = read(q['protocol_path'])
    assert sha(q['protocol_path']) == q['protocol_SHA256']
    for path, pin in {**plan['input_SHA256'], **q['source_SHA256']}.items():
        assert sha(path) == pin, path
    assert len(q['records']) == 10 and q['new_training_fits'] == q['new_optimizer_updates'] == q['new_GPU_allocations'] == 0
    assert sha(q['fitting_trace_arrays_path']) == q['fitting_trace_arrays_SHA256']
    traces = np.load(q['fitting_trace_arrays_path'], allow_pickle=False)
    assert traces['rows'].shape == (2, 4, 200, 400)
    cache = Path(plan['fitting_cache'])
    counts = sparse.load_npz(cache/'train/counts_hvg.npz').astype(np.float64).tocsr()
    assert counts.shape[1] == 2000 and np.isfinite(counts.data).all()
    assert (counts.data >= 0).all() and np.equal(counts.data, np.floor(counts.data)).all()
    squares = counts.multiply(counts)
    libraries = np.asarray(counts.sum(axis=1)).reshape(-1)
    mean_error = variance_error = 0.0
    for rep in range(2):
        for stage in range(4):
            indices = traces['rows'][rep, stage]
            np.testing.assert_array_equal(libraries[indices], traces['libraries'][rep, stage])
            weights = np.bincount(indices.reshape(-1), minlength=counts.shape[0]).astype(np.float64)
            n = weights.sum()
            assert n == 80000
            mean = np.asarray(weights @ counts).reshape(-1)/n
            variance = np.asarray(weights @ squares).reshape(-1)/n - mean**2
            np.testing.assert_allclose(mean, traces['real_mean'][rep, stage], rtol=1e-12, atol=1e-8)
            np.testing.assert_allclose(variance, traces['real_variance'][rep, stage], rtol=1e-12, atol=1e-8)
            mean_error = max(mean_error, float(np.max(np.abs(mean-traces['real_mean'][rep, stage]))))
            variance_error = max(variance_error, float(np.max(np.abs(variance-traces['real_variance'][rep, stage]))))
    rows = []
    for row in q['records']:
        assert sha(row['checkpoint_path']) == row['checkpoint_SHA256']
        assert sha(row['retained_per_gene_arrays_path']) == row['retained_per_gene_arrays_SHA256']
        z = np.load(row['retained_per_gene_arrays_path'], allow_pickle=False)
        assert z['theta'].shape == (2000,) and (z['theta'] > 0).all()
        assert all(z[k].shape == (2, 4, 2000) for k in ['mean', 'mean_variation', 'gamma_excess', 'poisson', 'total_variance'])
        gamma = (z['mean_variation'] + z['mean']**2)/z['theta']
        total = z['mean_variation'] + gamma + z['mean']
        np.testing.assert_array_equal(gamma, z['gamma_excess'])
        np.testing.assert_array_equal(z['mean'], z['poisson'])
        np.testing.assert_array_equal(total, z['total_variance'])
        fractions = []
        ratios = []
        for rep, reported in enumerate(row['replicates']):
            component_sums = {'mean_variation': float(z['mean_variation'][rep].sum()), 'gamma_excess': float(gamma[rep].sum()), 'poisson': float(z['mean'][rep].sum()), 'total': float(total[rep].sum())}
            np.testing.assert_allclose(list(component_sums.values()), list(reported['raw_marginal_variance_trace_components'].values()), rtol=1e-14)
            f = {k: component_sums[k]/component_sums['total'] for k in ['mean_variation', 'gamma_excess', 'poisson']}
            np.testing.assert_allclose(list(f.values()), list(reported['raw_marginal_variance_trace_fractions'].values()), rtol=1e-14)
            ratio = component_sums['total']/traces['real_variance'][rep].sum()
            np.testing.assert_allclose(ratio, reported['model_to_matching_fitting_reference_variance_trace_ratio'], rtol=1e-14)
            np.testing.assert_allclose(total[rep].sum(1)/traces['real_variance'][rep].sum(1), reported['per_stage_model_to_reference_variance_trace_ratio'], rtol=1e-14)
            fractions.append(f)
            ratios.append(float(ratio))
        # Posthoc description of mean profiles, not a new selection statistic.
        library_relative_error = np.abs(z['mean'].sum(2)/traces['real_mean'].sum(2)-1)
        assert library_relative_error.max() < 1e-6
        modeled_profile = z['mean']/z['mean'].sum(2, keepdims=True)
        real_profile = traces['real_mean']/traces['real_mean'].sum(2, keepdims=True)
        rows.append(dict(training_seed=row['training_seed'], saved_update=row['saved_update'], fractions={k: statistics.mean(x[k] for x in fractions) for k in fractions[0]}, raw_variance_ratio=statistics.mean(ratios), max_relative_library_mean_error=float(library_relative_error.max()), posthoc_mean_profile_L1_by_rep_stage=np.abs(modeled_profile-real_profile).sum(2).tolist()))
    summary = {}
    for step in [10000, 20000]:
        group = [r for r in rows if r['saved_update'] == step]
        assert sorted(r['training_seed'] for r in group) == list(range(5))
        summary[str(step)] = dict(fractions={k: statistics.mean(r['fractions'][k] for r in group) for k in group[0]['fractions']}, mean_raw_variance_ratio=statistics.mean(r['raw_variance_ratio'] for r in group), raw_variance_ratio_range=[min(r['raw_variance_ratio'] for r in group), max(r['raw_variance_ratio'] for r in group)], posthoc_mean_profile_L1=float(np.mean([r['posthoc_mean_profile_L1_by_rep_stage'] for r in group])))
        np.testing.assert_allclose(list(summary[str(step)]['fractions'].values()), list(q['summary'][str(step)]['mean_variance_trace_fractions'].values()), rtol=1e-14)
        np.testing.assert_allclose(summary[str(step)]['mean_raw_variance_ratio'], q['summary'][str(step)]['mean_model_to_reference_variance_ratio'], rtol=1e-14)
    audit = dict(status='PASSED_INDEPENDENT_RAW_FITTING_MOMENT_AND_TEN_CHECKPOINT_ARRAY_AUDIT', UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), result_path=str(result_path), result_SHA256=sha(result_path), audit_source_SHA256=sha(__file__), original_raw_fitting_reference_recomputed=True, reference_method='Sparse row-multiplicity weighted raw sums and squared sums; independent of original streaming Welford computation.', max_absolute_reference_mean_error=mean_error, max_absolute_reference_variance_error=variance_error, records=rows, summary=summary, model_and_source_checkpoint_hashes_unchanged=True, new_training_fits=0, new_optimizer_updates=0, new_GPU_allocations=0, Torch_imported='torch' in sys.modules, interpretation='Current count-head stage-conditional raw per-gene variance exceeds the matched fitting reference on these traces. This is not an attribution of normalized/PCA energy-distance error; mean-profile mismatch, biological variation and model misspecification can contribute. Gamma variance fraction is not the fraction of distribution error. Larger initial theta is a future hypothesis requiring a separately specified matched fresh training test, not a demonstrated improvement.', existing_frozen45_and_six_fresh_fit_protocols_unchanged=True, new_hyperparameter_experiment_admitted=False, full_goal_complete=False, SOTA_claim=False)
    assert not audit['Torch_imported']
    path = ROOT/'analysis/t005_nb_variance_CPU_root_audit_20261108.json'
    assert not path.exists()
    path.write_text(json.dumps(audit, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status=audit['status'], summary=summary, max_reference_variance_error=variance_error)))


if __name__ == '__main__':
    main()
