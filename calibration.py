"""Fixed calibration scores and common-draw distribution diagnostics."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

ROOT = Path('/ml2/alz')
DATA = ROOT / 'data/calibration_fold0'
SCIENCE = Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/core-comparison-20261103')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def targets():
    manifest = json.loads((DATA / 'manifest.json').read_text())
    assert manifest['status'] == 'CPU_CALIBRATION_PROJECTED_WITHOUT_REFITTING'
    for name, pin in manifest['files_SHA256'].items():
        assert sha(DATA / name) == pin
    y = np.load(DATA / 'real_summaries.npz', allow_pickle=False)
    cells = np.load(DATA / 'cell_reservoir.npz', allow_pickle=False)
    assert len(set(y['donors'].tolist())) == 4 and set(y['stage'].tolist()) == set(range(4))
    return manifest, y, cells


def cell_energy(a, b):
    return float(2 * cdist(a, b).mean() - cdist(a, a).mean() - cdist(b, b).mean())


def score(stage_summaries, generated_cells_by_stage):
    manifest, target, real_cells = targets()
    y = np.asarray(stage_summaries)
    cells = np.asarray(generated_cells_by_stage)
    assert y.shape == (4, 200, 26) and cells.shape == (4, 375, 50)
    assert np.isfinite(y).all() and np.isfinite(cells).all()
    stages, real = target['stage'], target['summary']
    values = np.array([np.linalg.norm(y[s] - r, axis=1).mean()
                       - 0.5 * cdist(y[s], y[s]).sum() / (200 * 199)
                       for s, r in zip(stages, real)])
    variance = real.var(0, ddof=1).sum()
    assert variance > 0
    low, high = np.percentile(y[stages], [10, 90], axis=1)
    return {
        'metric_role': 'calibration_selection', 'donors': target['donors'].tolist(),
        'stage': stages.tolist(), 'per_donor_energy_score': values.tolist(),
        'energy_score': float(values.mean()),
        'cell_energy_distance': cell_energy(cells.reshape(-1, 50), real_cells['equal_stage']),
        'per_stage_cell_energy_distance': [cell_energy(cells[s], real_cells['by_stage'][s]) for s in range(4)],
        'spread_ratio': float(np.sqrt(y.reshape(-1, 26).var(0, ddof=1).sum() / variance)),
        'within_condition_spread_ratio': float(np.sqrt(y.var(1, ddof=1).mean(0).sum() / variance)),
        'coverage80': float(((real >= low) & (real <= high)).mean()),
        'per_stage_composition_L1': [float(np.abs(np.square(y[s, :, :6]).mean(0) - np.square(real[np.flatnonzero(stages == s)[0], :6])).sum()) for s in range(4)],
        'calibration_manifest_SHA256': sha(DATA / 'manifest.json'),
        'not_independent_confirmation': True,
    }


def evaluate(ctx, method, seed, rep, iterator, out):
    from evaluation import Reservoir, donor_summaries
    from generation import trace
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    summaries, cell_pools, actual, expected = [], [], [], []
    for stage in range(4):
        s, c, schedule = trace(ctx, seed, rep, stage)
        reservoir = Reservoir(seed=20261106 + 10 * rep + stage, size=375)
        pieces, next_start = [], 0
        for start, pcs, exemplar, totals in iterator(schedule, c, s):
            assert start == next_start and pcs.shape[1:] == (400, 50)
            assert np.isfinite(pcs).all()
            next_start += len(pcs)
            pieces.append(donor_summaries(pcs, ctx.space))
            reservoir.add(pcs)
            actual.append(np.asarray(totals).ravel())
            expected.append(schedule.libraries[start:start + len(pcs)].ravel())
            if start == 0 and exemplar is not None:
                assert exemplar.shape == (1, 400, 2000)
                assert np.isfinite(exemplar).all() and (exemplar >= 0).all()
                assert np.equal(exemplar, np.floor(exemplar)).all()
                np.savez_compressed(out / f'stage{stage}_exemplar.npz', counts=exemplar)
        assert next_start == 200
        summaries.append(np.concatenate(pieces))
        cell_pools.append(reservoir.cells)
    y, cells = np.stack(summaries), np.stack(cell_pools)
    q = score(y, cells)
    actual, expected = np.concatenate(actual), np.concatenate(expected)
    assert len(actual) == len(expected) == 320000
    q.update(status='completed', method=method, training_seed=seed,
             generation_replicate=rep, generated_cells=len(actual),
             generated_zero_count_rate=float((actual == 0).mean()),
             actual_library_quantiles=np.percentile(actual, [0, 10, 50, 90, 100]).tolist(),
             supplied_exposure_quantiles=np.percentile(expected, [0, 10, 50, 90, 100]).tolist())
    np.savez_compressed(out / 'arrays.npz', stage_synthetic_summaries=y,
                        generated_cells_by_stage=cells)
    q['arrays_SHA256'] = sha(out / 'arrays.npz')
    dump(out / 'metrics.json', q)
    return q


def average(records):
    assert len(records) == 2
    assert all(q['status'] == 'completed' and q['metric_role'] == 'calibration_selection' for q in records)
    assert records[0]['donors'] == records[1]['donors']
    assert [q['generation_replicate'] for q in records] == [0, 1]
    keys = ['energy_score', 'cell_energy_distance', 'spread_ratio',
            'within_condition_spread_ratio', 'coverage80', 'generated_zero_count_rate']
    return {'metric_role': 'calibration_selection', 'generation_replicates': 2,
            'donors': records[0]['donors'],
            **{k: float(np.mean([q[k] for q in records])) for k in keys},
            'per_donor_energy_score': np.mean([q['per_donor_energy_score'] for q in records], axis=0).tolist(),
            'per_stage_cell_energy_distance': np.mean([q['per_stage_cell_energy_distance'] for q in records], axis=0).tolist()}
