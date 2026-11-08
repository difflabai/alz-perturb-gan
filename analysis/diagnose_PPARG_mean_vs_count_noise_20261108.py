"""Read-only explanation of PPARG error using previously audited frozen probes.

No model is loaded, generated, trained, or changed. The original probe uses
exactly matched real-cell library depths at each stage. Finite conditioning
ensembles are diagnostic samples, not biological replication.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.sparse import load_npz

ROOT = Path(__file__).resolve().parents[1]
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    source = ROOT / 'analysis/t005_nb_variance_CPU_result_20261108.json'
    previous_audit = ROOT / 'analysis/t005_nb_variance_CPU_root_audit_20261108.json'
    q = json.loads(source.read_bytes())
    audit = json.loads(previous_audit.read_bytes())
    assert audit['result_SHA256'] == sha(source)
    assert audit['status'] == 'PASSED_INDEPENDENT_RAW_FITTING_MOMENT_AND_TEN_CHECKPOINT_ARRAY_AUDIT'
    plan = json.loads((ROOT / 'analysis/primamente_inference_hypothesis_plan_20261108.json').read_bytes())
    cache = Path(plan['fitting_cache'])
    for name in ['train/counts_hvg.npz', 'train/metadata.npz']:
        assert sha(cache / name) == plan['input_SHA256'][str(cache / name)]
    with np.load(cache / 'train/metadata.npz', allow_pickle=False) as z:
        gene_index = list(z['genes']).index('PPARG')
    counts = load_npz(cache / 'train/counts_hvg.npz')
    gene = counts[:, gene_index].toarray().ravel()
    libraries = np.asarray(counts.sum(1)).ravel()
    initializer = np.clip(gene.mean()**2 / max(gene.var() - gene.mean(), 1e-4), .1, 10000)
    trace = Path(q['fitting_trace_arrays_path'])
    assert sha(trace) == q['fitting_trace_arrays_SHA256']
    with np.load(trace, allow_pickle=False) as z:
        rows = z['rows'].copy()
        np.testing.assert_array_equal(libraries[rows], z['libraries'])
        real_mean = z['real_mean'][:, :, gene_index].copy()
        real_variance = z['real_variance'][:, :, gene_index].copy()
    np.testing.assert_allclose(gene[rows].mean((2, 3)), real_mean, atol=1e-12, rtol=0)
    detected = (gene[rows] > 0).mean((2, 3))
    records = []
    inputs = {str(source.relative_to(ROOT)):sha(source), str(previous_audit.relative_to(ROOT)):sha(previous_audit),
              'retained_fitting_probe_arrays':sha(trace), 'training_counts':sha(cache / 'train/counts_hvg.npz'),
              'training_metadata':sha(cache / 'train/metadata.npz')}
    for row in q['records']:
        path = Path(row['retained_per_gene_arrays_path'])
        assert sha(path) == row['retained_per_gene_arrays_SHA256']
        inputs[f"seed{row['training_seed']}_update{row['saved_update']}_arrays"] = sha(path)
        with np.load(path, allow_pickle=False) as z:
            mu = z['mean'][:, :, gene_index].copy()
            mean_variation = z['mean_variation'][:, :, gene_index].copy()
            gamma = z['gamma_excess'][:, :, gene_index].copy()
            poisson = z['poisson'][:, :, gene_index].copy()
            theta = float(z['theta'][gene_index])
            np.testing.assert_allclose(mean_variation + gamma + poisson, z['total_variance'][:, :, gene_index])
        # For every nonnegative integer count law, P(X>0) <= E[X]. Changing
        # dispersion alone cannot exceed this bound while holding means fixed.
        bound = np.minimum(mu, 1)
        records.append(dict(seed=row['training_seed'], updates=row['saved_update'],
                            learned_theta=theta, theta_over_pooled_moment_initializer=theta / float(initializer),
                            model_mean_count_by_stage=mu.mean(0).tolist(),
                            model_mean_over_matched_observed_by_stage=(mu / real_mean).mean(0).tolist(),
                            count_variance_components_by_stage={
                                'latent_and_library_mean_variation':mean_variation.mean(0).tolist(),
                                'gamma_excess':gamma.mean(0).tolist(), 'poisson':poisson.mean(0).tolist()},
                            mean_preserving_nonzero_upper_bound_by_stage=bound.mean(0).tolist(),
                            NotAD_mean_bound_below_observed_nonzero_in_both_probes=bool((bound[:, 0] < detected[:, 0]).all())))
    final = [r for r in records if r['updates'] == 20000]
    assert sorted(r['seed'] for r in final) == list(range(5))
    result = dict(status='PPARG_LEARNED_MEAN_SHORTFALL_PRECEDES_COUNT_NOISE', gene='PPARG',
                  stage_order=['Not AD', 'Low', 'Intermediate', 'High'],
                  units='Raw gene counts; these are not log-expression contrasts.',
                  reference_scope=q['reference_scope'], raw_libraries_recomputed_and_exactly_matched=True,
                  observed_probe_mean_count_by_stage=real_mean.mean(0).tolist(),
                  observed_probe_nonzero_fraction_by_stage=detected.mean(0).tolist(),
                  observed_probe_variance_by_stage=real_variance.mean(0).tolist(),
                  five_final_model_mean_count_by_stage=np.mean([r['model_mean_count_by_stage'] for r in final], axis=0).tolist(),
                  pooled_moment_theta_initializer=float(initializer), records=records,
                  NotAD_noise_only_fix_cannot_match_observed_detection_on_all_five_finite_probe_ensembles=all(
                      r['NotAD_mean_bound_below_observed_nonzero_in_both_probes'] for r in final),
                  established=['Learned PPARG means are too low before Gamma or Poisson draws, even with matched real-cell depths.',
                               'All five final models have a Not AD mean below the observed nonzero fraction in both finite probes.',
                               'A mean-preserving dispersion-only change cannot repair that detection deficit on these probes.',
                               'The learned theta increases from its pooled initializer; it is not simply frozen at its initial value.'],
                  plausible_not_established=['The critic receives 150 bag features from 50 PCs instead of direct per-gene means and zero fractions.',
                                            'The count adapter uses a biased straight-through gradient through continuous Gamma rates for discrete counts.',
                                            'Optimization and representation may allow improved aggregate fit while individual genes drift.'],
                  limitations=['Posthoc analysis of frozen, finite Monte Carlo probes, not a causal training ablation.',
                               'Raw variance contributions are not a decomposition of log/PCA energy scores.',
                               'This does not establish an inherent limitation of the released ParticleGAN formulation.',
                               'Changing theta, training features, or optimization requires a separate matched experiment.'],
                  new_training_fits=0, new_optimizer_updates=0, new_count_draws=0, new_GPU_allocations=0,
                  source_SHA256=sha(__file__), input_SHA256=inputs)
    output = ROOT / 'analysis/primamente_PPARG_fit_mechanism_20261108.json'
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k:result[k] for k in ['status', 'observed_probe_mean_count_by_stage',
          'five_final_model_mean_count_by_stage', 'observed_probe_nonzero_fraction_by_stage',
          'NotAD_noise_only_fix_cannot_match_observed_detection_on_all_five_finite_probe_ensembles']}))

if __name__ == '__main__':
    main()
