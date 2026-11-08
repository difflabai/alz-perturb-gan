"""Full calibration diagnostics on the new namespace's fixed fitting-only traces."""
from pathlib import Path
import numpy as np
from calibration import score,sha,dump
from preset_scope import PROTOCOL

def trace(ctx,seed,rep,stage):
    if ctx.fold!=0 or type(seed) is not int or seed not in (8,9,10) or rep not in (0,1) or stage not in range(4):raise ValueError('unregistered scalar calibration trace')
    s=20261104+1000*seed+100*rep+stage
    c=np.zeros(7,dtype=np.float32);c[3]=stage/3
    return s,c,ctx.law.schedule(c,s,m=200,k=400)

def evaluate(ctx, method, seed, rep, iterator, out):
    from evaluation import Reservoir, donor_summaries
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
             trace_manifest_SHA256=sha(PROTOCOL),
             sampling_contract='calibration_equal_stage375_priority_v1',
             generated_zero_count_rate=float((actual == 0).mean()),
             actual_library_quantiles=np.percentile(actual, [0, 10, 50, 90, 100]).tolist(),
             supplied_exposure_quantiles=np.percentile(expected, [0, 10, 50, 90, 100]).tolist())
    np.savez_compressed(out / 'arrays.npz', stage_synthetic_summaries=y,
                        generated_cells_by_stage=cells)
    q['arrays_SHA256'] = sha(out / 'arrays.npz')
    dump(out / 'metrics.json', q)
    return q
