"""Explicit mixture of whole PG donor bags and empirical training donor bags."""
import numpy as np
import torch


def bag_mask(fraction, replicate, stage):
    if type(fraction) not in (int, float) or not np.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError('finite PG bag fraction in[0,1] required')
    if abs(200 * fraction - round(200 * fraction)) > 1e-9:
        raise ValueError('PG fraction must select an exact number of200bags')
    if replicate not in (0, 1) or stage not in range(4):
        raise ValueError('fixed replicate0/1 and stage0..3 required')
    ordering = np.random.default_rng(20261112 + 10 * replicate + stage).permutation(200)
    mask = np.zeros(200, dtype=bool)
    mask[ordering[:round(200 * fraction)]] = True
    return mask


def blend_bags(ctx, schedule, generated, mask):
    from generation import validate_counts
    if np.asarray(mask).shape != (200,) or np.asarray(mask).dtype != np.bool_:
        raise ValueError('exact200bag Boolean model-choice mask required')
    if not hasattr(ctx, 'baseline_pcs'):
        from pathlib import Path
        ctx.baseline_pcs = np.load(Path(ctx.cache) / 'train/real_pcs.npy', allow_pickle=False)
    for start, pcs, exemplar, totals in generated:
        selected = mask[start:start + len(pcs)]
        if selected.all():
            yield start, pcs, exemplar, totals
            continue
        rows = schedule.rows[start:start + len(pcs)][~selected]
        raw = ctx.X[torch.as_tensor(rows, device=ctx.device)]
        validate_counts(raw.reshape(-1, 2000), len(rows) * 400, 2000)
        mixed = pcs.copy()
        mixed[~selected] = ctx.baseline_pcs[rows]
        observed_totals = np.asarray(totals).reshape(len(pcs), 400).copy()
        observed_totals[~selected] = raw.sum(-1).cpu().numpy()
        if start == 0 and not selected[0]:
            exemplar = raw[:1].cpu().numpy()
        yield start, mixed, exemplar, observed_totals.ravel()


def iter_hybrid(ctx, generator, prior, schedule, condition, seed, *, fraction, replicate):
    from generation import iter_pg
    stage = int(round(float(condition[3]) * 3))
    mask = bag_mask(fraction, replicate, stage)
    # Generate the same complete PG stream before replacing selected bags.
    # This preserves the original PG draw identity at every retained row.
    yield from blend_bags(ctx, schedule,
                          iter_pg(ctx, generator, prior, schedule, condition, seed), mask)
