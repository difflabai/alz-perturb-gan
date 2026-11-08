"""Describe two-stream calibration ED variation using existing saved arrays.

This is not an estimate of online controller jitter or a controller replay:
the actual prospective controller uses common random streams at every update.
There are no intermediate 1,000-update observations in these saved arrays.
"""
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys

ROOT = Path('/ml2/alz')
PROJECT = Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert all(os.environ.get(k) == '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'))
    sys.path[:0] = [str(ROOT), str(PROJECT/'current_base_20261029/runtime/lib/python3.12/site-packages'),
                   str(PROJECT/'.venv312/lib/python3.12/site-packages')]
    import numpy as np
    from calibration import cell_energy

    source = ROOT/'analysis/t005_stage_mixture_energy_CPU_result_20261108.json'
    audit_path = ROOT/'analysis/t005_stage_mixture_energy_CPU_root_audit_20261108.json'
    protocol_path = ROOT/'analysis/scalarcal097/protocol.json'
    reservoir = ROOT/'data/calibration_fold0/cell_reservoir.npz'
    audit = json.loads(audit_path.read_bytes())
    for p in (source, reservoir):
        assert audit['input_SHA256'][str(p)] == sha(p)
    old = json.loads(source.read_bytes())
    assert old['retrospective_calibration_only'] and not old['external_evaluation_target_values_read']
    protocol = json.loads(protocol_path.read_bytes())
    threshold = protocol['controller']['relative_improvement']
    assert threshold == .01 and protocol['generation']['common_streams_across_arms_and_updates']
    with np.load(reservoir, allow_pickle=False) as data:
        real = data['equal_stage'].copy()
    groups, pins = {}, {str(p): sha(p) for p in (source, audit_path, protocol_path, reservoir)}
    for record in old['array_records']:
        p = Path(record['arrays_path'])
        assert p.is_relative_to(ROOT/'artifacts') and sha(p) == record['arrays_SHA256']
        with np.load(p, allow_pickle=False) as data:
            generated = data['generated_cells_by_stage'].reshape(-1, 50).copy()
        assert generated.shape == real.shape == (1500, 50)
        value = cell_energy(generated, real)
        assert math.isfinite(value) and abs(value-record['pooled_ED']) < 1e-10
        key = (record['seed'], record['checkpoint_or_control'])
        group = groups.setdefault(key, {})
        assert record['replicate'] not in group
        group[record['replicate']] = value
        pins[str(p)] = sha(p)
    assert len(groups) == 20 and all(set(g) == {0, 1} for g in groups.values())
    assert cell_energy(generated, real) == cell_energy(generated, real)

    pairs = []
    for (seed, checkpoint), values in sorted(groups.items()):
        mean = statistics.mean(values.values())
        gap = abs(values[0]-values[1])
        pairs.append(dict(seed=seed, checkpoint_or_control=checkpoint,
                          replicate_ED=[values[0], values[1]], mean_ED=mean,
                          absolute_stream_gap=gap, relative_stream_gap=gap/mean,
                          gap_in_units_of_one_percent_of_pair_mean=gap/(threshold*mean)))
    transitions = []
    for seed in range(5):
        for start, end in ((5000, 10000), (10000, 20000), (5000, 20000)):
            before, after = groups[(seed, str(start))], groups[(seed, str(end))]
            changes = [after[rep]-before[rep] for rep in (0, 1)]
            mean_before = statistics.mean(before.values())
            transitions.append(dict(seed=seed, start=start, end=end,
                paired_stream_ED_changes=changes, mean_ED_change=statistics.mean(changes),
                both_streams_agree_on_direction=changes[0]*changes[1] > 0,
                stream_improvement_exceeds_registered_fraction=[
                    after[rep] < before[rep]*(1-threshold) for rep in (0, 1)],
                mean_improvement_exceeds_registered_fraction=
                    statistics.mean(after.values()) < mean_before*(1-threshold)))
    pg_pairs = [row for row in pairs if row['checkpoint_or_control'] != 'resampling']
    summary = dict(PG_checkpoint_pairs=len(pg_pairs),
        median_relative_stream_gap=statistics.median(row['relative_stream_gap'] for row in pg_pairs),
        max_relative_stream_gap=max(row['relative_stream_gap'] for row in pg_pairs),
        pairs_with_stream_gap_exceeding_one_percent_of_pair_mean=sum(
            row['relative_stream_gap'] > threshold for row in pg_pairs),
        ten_to_twenty_k_both_streams_agree_on_direction=sum(
            row['both_streams_agree_on_direction'] for row in transitions if row['start'] == 10000))
    result = dict(status='ALL40_CALIBRATION_ARRAY_SCORES_REPLAYED_TWO_STREAM_RESOLUTION_DESCRIBED',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), summary=summary,
        registered_relative_improvement=threshold, pairs=pairs, paired_checkpoint_changes=transitions,
        scalar_protocol_unchanged=True, scalar_protocol_SHA256=sha(protocol_path),
        common_random_streams_across_updates=True, original45_unchanged=True,
        interpretation='A two-stream discrepancy describes sampling-stream sensitivity at a fixed saved model. '
            'It is not the noise of the two-stream mean, a confidence interval, or sequential controller jitter. '
            'The prospective controller averages the same two streams, smooths three observations, and waits '
            'three non-improving observations. Only actual fresh full20k matched fits can assess its benefit.',
        limitations='Only two streams per checkpoint and three sparse checkpoints per existing seed; '
            'cannot validate the 1000-update cadence, replay missing controller decisions, infer false-trigger '
            'rates, or establish whether 1% is optimal. Four adaptively reused calibration donors; no biological '
            'uncertainty estimate. No threshold selection, endpoint reselection or candidate promotion.',
        no_new_training_or_generation=True, new_optimizer_updates=0, new_GPU_allocations=0,
        external_evaluation_target_values_read=False, full_goal_complete=False, SOTA_claim=False,
        input_SHA256=pins, source_SHA256={str(Path(__file__).resolve()):sha(__file__),
                                       str(ROOT/'calibration.py'):sha(ROOT/'calibration.py')})
    assert 'torch' not in sys.modules
    for path, pin in pins.items():
        assert sha(path) == pin
    out = ROOT/'analysis/calibration_ED_two_stream_resolution_CPU_20261108.json'
    assert not out.exists()
    out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
