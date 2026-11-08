"""Exercise the future audit's scorer on40 already-saved calibration arrays.

No new theta result, model, generation, GPU allocation or external target read.
"""
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path('/ml2/alz')
PROJECT = Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark')
PG = PROJECT / 'current_base_20261029/runtime/bin/python'
sys.path[:0] = [str(ROOT / 'analysis'), str(ROOT),
    str(PROJECT / 'current_base_20261029/runtime/lib/python3.12/site-packages'),
    str(PROJECT / '.venv312/lib/python3.12/site-packages')]
import numpy as np
import audit_dispersion097_completed_study as audit
from calibration import targets


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    state = ROOT / 'artifacts/dispersion097-final-audit-CPU-20261108'
    output = ROOT / 'analysis/dispersion097_completed_study_audit_CPU_preparation_20261108.json'
    assert not state.exists() and not output.exists()
    state.mkdir()
    tests = subprocess.run([str(PG), '-I', '-S', '-B', str(ROOT / 'analysis/test_dispersion097_completed_audit.py')],
                           cwd=ROOT, capture_output=True, text=True)
    (state / 'tests.stdout').write_text(tests.stdout)
    (state / 'tests.stderr').write_text(tests.stderr)
    assert tests.returncode == 0 and 'Ran 11 tests' in tests.stderr and '\nOK\n' in tests.stderr
    ready = audit.readiness()
    assert ready['missing_paths'] and not ready['quality_values_read']
    premature = state / 'premature-scientific-audit.json'
    refused = subprocess.run([str(PG), '-I', '-S', '-B', str(ROOT / 'analysis/audit_dispersion097_completed_study.py'),
                              '--output', str(premature)], cwd=ROOT, capture_output=True, text=True)
    (state / 'premature.stdout').write_text(refused.stdout)
    (state / 'premature.stderr').write_text(refused.stderr)
    assert refused.returncode != 0 and 'All prior studies and all six fresh theta outcomes must finish before numerical reads' in refused.stderr
    assert not premature.exists()
    source = ROOT / 'analysis/t005_checkpoint_CPU_multiseed_result_20261108.json'
    old = audit.read(source)
    assert old['training_seeds'] == list(range(5)) and old['saved_updates'] == [5000, 10000, 20000]
    assert old['new_training_fits'] == old['new_optimizer_updates'] == 0
    assert not old['external_evaluation_target_values_read']
    inputs = {str(source): audit.sha(source)}
    manifest, ys, cells = targets()
    try:
        checked = []
        for seed in map(str, range(5)):
            pins = old['all_seed_records'][seed]['output_SHA256']
            paths = sorted(Path(p) for p in pins if p.endswith('/arrays.npz'))
            assert len(paths) == 8
            for path in paths:
                metrics_path = path.with_name('metrics.json')
                assert audit.sha(path) == pins[str(path)] and audit.sha(metrics_path) == pins[str(metrics_path)]
                saved = audit.read(metrics_path)
                assert saved['metric_role'] == 'calibration_selection' and saved['donors'] == ys['donors'].tolist()
                with np.load(path, allow_pickle=False) as z:
                    actual = audit.independent_metrics(z['stage_synthetic_summaries'], z['generated_cells_by_stage'],
                        ys['summary'], ys['stage'], cells['equal_stage'], cells['by_stage'])
                audit.compare_metrics(actual, saved)
                inputs.update({str(p): audit.sha(p) for p in (path, metrics_path)})
                checked.append(str(path))
        assert len(checked) == 40
    finally:
        ys.close()
        cells.close()
    assert 'torch' not in sys.modules
    proof = dict(status='SIX_FRESH_THETA_FINAL_AUDIT_CPU_PREPARATION_CHECK_PASS',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), actual_tests=11,
        actual_readiness=ready, premature_numerical_audit_refused=True,
        existing_calibration_array_pairs_independently_recomputed=40,
        existing_cell_ED_values_checked=40, existing_per_stage_ED_values_checked=160,
        existing_donor_ES_values_checked=160, existing_arrays=checked,
        existing_data_is_not_a_new_theta_outcome=True, real_theta_fits_audited=0,
        no_new_model_generation_or_optimizer_updates=True, no_GPU_allocation=True,
        Torch_imported=False, external_evaluation_target_values_read=False,
        source_SHA256={str(p): audit.sha(p) for p in (Path(__file__).resolve(),
            ROOT / 'analysis/audit_dispersion097_completed_study.py',
            ROOT / 'analysis/test_dispersion097_completed_audit.py',
            ROOT / 'analysis/audit_released097_completed_studies.py', ROOT / 'calibration.py')},
        input_SHA256=inputs, log_SHA256={str(p): audit.sha(p) for p in state.iterdir()},
        eventual_execution='CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 '
            + str(PG) + ' -I -S -B /ml2/alz/analysis/audit_dispersion097_completed_study.py '
            + '--output /ml2/alz/analysis/dispersion097_completed_study_final_audit.json',
        automatic_execution_not_installed=True, active_runtime_sources_unchanged=True,
        remaining='Wait for actual full prior studies and all six fresh theta20k outcomes, then run the final audit; a calibration preference alone cannot establish SOTA.',
        full_goal_complete=False, SOTA_claim=False)
    output.write_text(json.dumps(proof, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: proof[k] for k in ('status', 'actual_tests',
        'existing_calibration_array_pairs_independently_recomputed', 'real_theta_fits_audited', 'full_goal_complete')}))


if __name__ == '__main__':
    main()
