"""Read-only timing proxy from completed PG endpoints and known calibration PCs.

File modification intervals are observational proxies, not instrumented kernel
timings or measured new-study overhead. No partial held-out quality JSON is read.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path('/ml2/alz')
PROJECT = Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark')


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert all(os.environ.get(k) == '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'))
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert args.output.parent == ROOT / 'analysis' and not args.output.exists()
    sys.path[:0] = [str(ROOT / 'benchmark097_balanced_runtime'), str(ROOT),
                   str(PROJECT / 'current_base_20261029/runtime/lib/python3.12/site-packages'),
                   str(PROJECT / '.venv312/lib/python3.12/site-packages')]
    import numpy as np
    from cell_metadata import completed_metadata
    from calibration import cell_energy
    cells = [(0, 0), (2, 0), (1, 1), (1, 0), (0, 1)]
    records, pins, stat_pins = [], {}, {}
    for fold, seed in cells:
        meta = completed_metadata(dict(experiment='external097', fold=fold, seed=seed, method='pg'))
        auth = read(meta['authorization_path'])
        base = Path(auth['output_dir'])
        native = read(base / 'native_restore.json')
        receipt = read(base / 'fit/checkpoint_receipt.json')
        complete = read(base / 'completed.json')
        assert complete['status'] == receipt['status'] == 'completed'
        assert complete['from_scratch'] and complete['initial_completed_updates'] == 0
        assert native['completed_updates'] == receipt['completed_updates'] == complete['new_model_updates'] == 20000
        assert native['checkpoint_SHA256'] == receipt['SHA256'] == complete['checkpoint_SHA256']
        assert complete['native_restore_SHA256'] == sha(base / 'native_restore.json')
        assert complete['PG_configuration'] == auth['released_hyperparameters']
        assert auth['released_hyperparameters'] == dict(generator_lr=.012, critic_lr=.018, particle_lr=.06,
                                                        reg_coeff=1., reg_kappa=1., prior_reg=.01)
        for path, pin in auth['files_SHA256'].items():
            if Path(path).name in ('generation.py', 'scoring.py', 'released_adapter.py', 'released_update.py'):
                assert sha(path) == pin
                pins[path] = pin
        ledger = read(auth['ledger_path'])['jobs'][0]
        order = ['native_restore.json', 'evaluation_r0/metrics.json', 'resampling_r0/metrics.json',
                 'evaluation_r1/metrics.json', 'resampling_r1/metrics.json', 'stage_swap_r0/metrics.json', 'completed.json']
        times = []
        for rel in order:
            path = base / rel
            stat = path.stat()
            times.append(stat.st_mtime_ns / 1e9)
            pins[str(path)] = sha(path)
            stat_pins[str(path)] = [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns]
        assert ledger['started_unix'] < times[0] and all(a <= b for a, b in zip(times, times[1:]))
        durations = [b-a for a, b in zip(times, times[1:])]
        replicate_seconds = [durations[0], durations[2]]
        extra_proxy = 20 * sum(replicate_seconds)
        records.append(dict(attempt=meta['attempt'], gpu_uuid=meta['gpu_uuid'],
            charged_GPU_seconds=meta['charged_GPU_seconds'],
            startup_training_and_native_restore_interval_seconds=times[0]-ledger['started_unix'],
            full_model_generation_and_scoring_replicate_intervals_seconds=replicate_seconds,
            resampling_intervals_seconds=[durations[1], durations[3]],
            stage_swap_model_interval_seconds=durations[4],
            twenty_two_replicate_observation_extra_time_proxy_seconds=extra_proxy,
            whole_allocation_cost_multiplier_proxy=1+extra_proxy/meta['charged_GPU_seconds'],
            timing_evidence='Sequential endpoint file modification intervals; endpoint code generates320000 cells once per replicate, reusing stage pools for repeated donor queries.',
            quality_report_JSON_parsed=False))
        for path in (Path(meta['authorization_path']), Path(auth['ledger_path']), base / 'fit/checkpoint_receipt.json'):
            pins[str(path)] = sha(path)
    manifest = ROOT / 'data/calibration_fold0/manifest.json'
    assert sha(manifest) == '597ff7929037aabd1d7bc2bd8631aa569dff56ca188ce4a48fbf946ef08e908b'
    target_path = ROOT / 'data/calibration_fold0/cell_reservoir.npz'
    with np.load(target_path, allow_pickle=False) as target:
        real = target['equal_stage'].copy()
    pools = []
    diagnostic = read(ROOT / 'analysis/t005_stage_mixture_energy_CPU_result_20261108.json')
    for rep in (0, 1):
        path = ROOT / f'artifacts/t005-checkpoint-calibration-CPU-v3-20261108/update20000_r{rep}/arrays.npz'
        assert diagnostic['input_SHA256'][str(path)] == sha(path)
        with np.load(path, allow_pickle=False) as arrays:
            pool = arrays['generated_cells_by_stage'].copy().reshape(-1, 50)
        assert pool.shape == real.shape == (1500, 50) and np.isfinite(pool).all() and np.isfinite(real).all()
        pools.append(pool)
        pins[str(path)] = sha(path)
    walls, cpu, values = [], [], []
    for repeat in range(20):
        start, start_cpu = time.perf_counter(), time.process_time()
        v = [cell_energy(pool, real) for pool in pools]
        walls.append(time.perf_counter()-start)
        cpu.append(time.process_time()-start_cpu)
        assert np.isfinite(v).all()
        values.append(v)
    np.testing.assert_array_equal(values, np.repeat([values[0]], 20, axis=0))
    for path, previous in stat_pins.items():
        st = Path(path).stat()
        assert [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns] == previous
    for path, pin in pins.items():
        assert sha(path) == pin
    result = dict(status='COMPLETED_PG_TIMING_PROXY_AND_ACTUAL_CALIBRATION_CPU_SCORING_COST_MEASURED',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), completed_native20k_models_examined=5,
        completed_endpoint_timing_records=records, paired_calibration_ED_CPU_repetitions=20,
        actual_CPU_scoring_wall_seconds=walls, actual_CPU_scoring_process_seconds=cpu,
        median_two_replicate_CPU_score_seconds=statistics.median(walls),
        twenty_observation_CPU_score_seconds=sum(walls),
        completed_model_proxy_multiplier_range=[min(r['whole_allocation_cost_multiplier_proxy'] for r in records),
                                                max(r['whole_allocation_cost_multiplier_proxy'] for r in records)],
        actual_scalar_GPU_overhead_measured=False, real_scalar_fits=0, new_GPU_allocations=0,
        model_Torch_or_CUDA_imported=False, heldout_quality_reports_parsed=False,
        scalar_schedule_or_formulation_changed=False, source_SHA256={str(Path(__file__).resolve()):sha(__file__),
            str(ROOT/'calibration.py'):sha(ROOT/'calibration.py')}, input_SHA256=pins,
        timing_file_stat_evidence=stat_pins,
        interpretation='Existing endpoint-generation timings suggest low online-measurement overhead, and repeated actual CPU distance scoring is measured separately. This supports retaining the1.15x scenario as plausible; it does not turn that scenario into a measured scalar-fit cost or remove2x/5x stress cases.',
        limitations=['mtime intervals include endpoint I/O, traces and diagnostics; they are not instrumented generation-only timings or rigorous upper bounds.',
                     'Calibration uses different traces, per-stage reservoirs, RNG isolation and observation-array writes. Early checkpoints, new seeds, adaptive weights and future shared load may change time.',
                     'Completed benchmark stage pools are reused across evaluation donors: generating more counts per repeated donor query is not part of the actual benchmark runtime.',
                     'The CPU scoring repetitions replay previously audited calibration arrays; they add no independent biological or optimization result.'],
        full_goal_complete=False, SOTA_claim=False)
    assert 'torch' not in sys.modules
    pins[str(manifest)] = sha(manifest)
    pins[str(target_path)] = sha(target_path)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({key:result[key] for key in ('status','completed_model_proxy_multiplier_range',
        'median_two_replicate_CPU_score_seconds','twenty_observation_CPU_score_seconds','actual_scalar_GPU_overhead_measured')}))


if __name__ == '__main__':
    main()
