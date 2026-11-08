"""Posthoc analytic theta sensitivity at fixed saved mean distributions.

No training, count generation, checkpoint editing or distribution-distance
evaluation. Multipliers here apply to final theta, not its initialization.
"""
import datetime
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path('/ml2/alz')
PROJECT = Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark')
sys.path[:0] = [str(PROJECT/'current_base_20261029/runtime/lib/python3.12/site-packages'), str(PROJECT/'.venv312/lib/python3.12/site-packages')]
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    source = ROOT/'analysis/t005_nb_variance_CPU_result_20261108.json'
    q = read(source)
    audit_path = ROOT/'analysis/t005_nb_variance_CPU_root_audit_20261108.json'
    audit = read(audit_path)
    assert audit['result_SHA256'] == sha(source)
    assert audit['status'] == 'PASSED_INDEPENDENT_RAW_FITTING_MOMENT_AND_TEN_CHECKPOINT_ARRAY_AUDIT'
    assert sha(q['fitting_trace_arrays_path']) == q['fitting_trace_arrays_SHA256']
    real = np.load(q['fitting_trace_arrays_path'], allow_pickle=False)['real_variance']
    records = []
    input_pins = {str(source):sha(source), str(audit_path):sha(audit_path), q['fitting_trace_arrays_path']:sha(q['fitting_trace_arrays_path'])}
    for row in q['records']:
        if row['saved_update'] != 20000:
            continue
        path = row['retained_per_gene_arrays_path']
        assert sha(path) == row['retained_per_gene_arrays_SHA256']
        input_pins[path] = sha(path)
        z = np.load(path, allow_pickle=False)
        floor = z['mean_variation']+z['poisson']
        gamma = z['gamma_excess']
        denominator = real.sum()-floor.sum()
        multipliers = (1., 3., 5., 10., 100.)
        records.append(dict(training_seed=row['training_seed'], model_theta_values_unchanged=True, theoretical_infinite_theta_raw_trace_ratio=float(floor.sum()/real.sum()), uniform_final_theta_multiplier_matching_pooled_reference_trace=float(gamma.sum()/denominator) if denominator > 0 else None, fixed_mean_counterfactual_raw_trace_ratios={str(m):float((floor+gamma/m).sum()/real.sum()) for m in multipliers}, per_stage_floor_to_reference_ratio=(floor.sum((0,2))/real.sum((0,2))).tolist(), per_stage_uniform_final_theta_multiplier_matching_reference=[float(gamma[:,s].sum()/(real[:,s].sum()-floor[:,s].sum())) if real[:,s].sum()>floor[:,s].sum() else None for s in range(4)], fraction_stage_gene_pairs_floor_already_exceeds_reference=float((floor>real).mean()), floor_excess_sum_as_fraction_of_reference_trace=float(np.maximum(floor-real,0).sum()/real.sum())))
    assert sorted(r['training_seed'] for r in records) == list(range(5))
    scales = [r['uniform_final_theta_multiplier_matching_pooled_reference_trace'] for r in records]
    result = dict(status='COMPLETE_READONLY_FIXED_MEAN_FINAL_THETA_ANALYTIC_SENSITIVITY', UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), records=records, median_uniform_final_theta_multiplier=float(np.median(scales)), scale_range=[min(scales),max(scales)], median_fixed_mean_trace_ratios={str(m):float(np.median([r['fixed_mean_counterfactual_raw_trace_ratios'][str(m)] for r in records])) for m in (1.,3.,5.,10.,100.)}, input_SHA256=input_pins, source_SHA256=sha(__file__), new_training_fits=0, new_optimizer_updates=0, new_count_draws=0, new_GPU_allocations=0, source_models_and_checkpoints_modified=False, mean_distribution_fixed_for_calculation=True, application='May motivate a separately qualified initialization-scale trial using a round factor near5. These final-parameter counterfactuals do not predict the trained final theta, mean profile, normalized energy distance, donor energy score, or convergence after changing initialization.', no_initialization_scale_selected_by_energy_distance=True, no_hyperparameter_study_admitted=True, frozen45_and_six_LR_fits_unchanged=True, full_goal_complete=False, SOTA_claim=False)
    path = ROOT/'analysis/t005_theta_fixed_mean_counterfactual_CPU_20261108.json'
    assert not path.exists() and 'torch' not in sys.modules
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ['status','median_uniform_final_theta_multiplier','scale_range','median_fixed_mean_trace_ratios']}))


if __name__ == '__main__':
    main()
