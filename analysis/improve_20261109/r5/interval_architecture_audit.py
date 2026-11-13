"""Frozen, development-only audit of PC-summary intervals and donor geometry.

No optimizer updates or default/metric changes. The existing score averages
coordinatewise central percentile coverage across 50 PC means and 50 PC SDs
of 64 cells; it is neither a patient-level clinical interval nor simultaneous
coverage of all coordinates. With M=32, interpolated sample percentiles have
finite-sample undercoverage even for iid exchangeable perfect-model draws.

Training-fitted shifts/width factors below are DIAGNOSTICS, not conformal
intervals: these calibration donors were used to train the generator. Held-out
donors enter only reported evaluations. The rank-order interval has a perfect
iid-model marginal coverage calculation, not a guarantee for this fitted GAN.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import numpy as np
import evaluate as E

RUNS = E.RUNS
OUT = RUNS / 'results_r5/interval_architecture'
GROUPS = {'PC means': slice(0, 50), 'PC standard deviations': slice(50, 100), 'all coordinates': slice(0, 100)}


def finite_rank_bounds(samples, alpha=.2):
    n = samples.shape[1]
    lower_rank = max(1, int(np.floor(alpha / 2 * (n + 1))))
    upper_rank = min(n, int(np.ceil((1 - alpha / 2) * (n + 1))))
    ordered = np.sort(samples, axis=1)
    return ordered[:, lower_rank - 1], ordered[:, upper_rank - 1], (lower_rank, upper_rank)


def effective_dimension(rows):
    rows = rows - rows.mean(0)
    eigenvalues = np.linalg.eigvalsh(rows.T @ rows / max(1, len(rows) - 1)).clip(0)
    return float(eigenvalues.sum() ** 2 / np.maximum((eigenvalues ** 2).sum(), 1e-30))


def summarize_bounds(low, high, observed, mask, reference_width=None):
    """Observed is R x targets x coordinates; bounds are targets x coordinates."""
    result = {}
    for name, subset in GROUPS.items():
        lo, hi = low[mask, subset], high[mask, subset]
        actual = observed[:, mask, subset]
        result[name] = {'coverage': float(((actual >= lo) & (actual <= hi)).mean()),
                        'below': float((actual < lo).mean()), 'above': float((actual > hi).mean()),
                        'mean_width': float((hi - lo).mean())}
        if reference_width is not None:
            result[name]['width_ratio_to_raw'] = float(np.mean(hi - lo) / np.mean(reference_width[mask, subset]))
    return result


def fit_interval_diagnostics(low, high, observed, train):
    """Fit offsets and two pooled width factors using TRAIN donors only."""
    center, half = (low + high) / 2, (high - low) / 2
    safe_half = np.maximum(half, 1e-8)
    shift = (observed[:, train].mean(0) - center[train]).mean(0)
    scales, shifted_scales = np.ones(100), np.ones(100)
    info = {}
    for name in ('PC means', 'PC standard deviations'):
        subset = GROUPS[name]
        raw_scores = np.abs(observed[:, train, subset] - center[train, subset]) / safe_half[train, subset]
        shifted_scores = np.abs(observed[:, train, subset] - center[train, subset] - shift[subset]) / safe_half[train, subset]
        factor, shifted_factor = np.quantile(raw_scores, .8), np.quantile(shifted_scores, .8)
        scales[subset], shifted_scales[subset] = factor, shifted_factor
        info[name] = {'width_only_factor': float(factor), 'shift_and_width_factor': float(shifted_factor),
                      'mean_abs_center_shift': float(np.mean(np.abs(shift[subset])))}
    return {'center_only': (low + shift, high + shift),
            'train_width_only': (center - scales * half, center + scales * half),
            'train_shift_and_width': (center + shift - shifted_scales * half, center + shift + shifted_scales * half)}, info


def null_experiment(trials=50000):
    result = {}
    rng = np.random.default_rng(20261113)
    for n in (32, 128):
        values = rng.random((trials, n + 1))
        sample, actual = values[:, :n], values[:, n]
        low, high = np.percentile(sample, [10, 90], axis=1)
        a, b, ranks = finite_rank_bounds(sample[:, :, None])
        result[str(n)] = {'interpolated_empirical': float(((actual >= low) & (actual <= high)).mean()),
                          'interpolated_uniform_exact': float(.8 * (n - 1) / (n + 1)),
                          'rank_empirical': float(((actual >= a[:, 0]) & (actual <= b[:, 0])).mean()),
                          'rank_exact': float((ranks[1] - ranks[0]) / (n + 1)), 'ranks': ranks,
                          'scope': 'iid perfect Uniform(0,1) simulation; not fitted-model coverage'}
    return result


def audit_fit(path, real):
    with np.load(path, allow_pickle=False) as f:
        E.check_development_donors(f['targets'].astype(str).tolist(), path)
        order = [f['targets'].tolist().index(d) for d in real.donors]
        assert np.array_equal(f['genes'].astype(str), real.genes)
        Y = E.MX.pc_bag_summary(f['pcs'][order].astype(np.float32))
        k = f['pcs'].shape[2]
    actual = np.stack([np.stack([E.MX.pc_bag_summary(real.pcs[i][E.MX.det_subsample(
        len(real.pcs[i]), k, E.MX.stable_seed('pc-summary-rep', donor, repeat))])
        for i, donor in enumerate(real.donors)]) for repeat in range(E.DEFAULT_PC_REPEATS)])
    low, high = np.percentile(Y, [10, 90], axis=1)
    rank_low, rank_high, ranks = finite_rank_bounds(Y)
    train, held = ~real.is_heldout, real.is_heldout
    diagnostic_bounds, diagnostic_info = fit_interval_diagnostics(low, high, actual, train)
    methods = {'raw': (low, high), 'finite_rank': (rank_low, rank_high), **diagnostic_bounds}
    result = {'M': int(Y.shape[1]), 'PCs': 50, 'cells_per_summary': int(k), 'real_draws': len(actual),
              'rank_endpoints': ranks, 'train_only_diagnostic_fit': diagnostic_info,
              'train': {}, 'heldout': {}, 'geometry': {}, 'scale_levels': {}}
    for split, mask in [('train', train), ('heldout', held)]:
        for name, (lo, hi) in methods.items():
            result[split][name] = summarize_bounds(lo, hi, actual, mask, high - low)
        for stage in range(4):
            stage_mask = mask & (real.stage == stage)
            if stage_mask.any():
                result[split][f'raw_stage{stage}'] = summarize_bounds(low, high, actual, stage_mask)
    ybar = actual.mean(0)
    for split, mask in [('train', train), ('heldout', held)]:
        result['scale_levels'][split] = {
            'real_PC_sd_mean': float(ybar[mask, 50:].mean()),
            'generated_PC_sd_mean': float(Y[mask, :, 50:].mean()),
            'real_PC_sd_by_band': {label: float(ybar[mask, lo:hi].mean())
                                   for label, lo, hi in [('PC1-10', 50, 60), ('PC11-25', 60, 75), ('PC26-50', 75, 100)]},
            'donors_with_fewer_than_64_cells': int(sum(len(real.pcs[i]) < 64 for i in np.flatnonzero(mask)))}
    for name in ('PC means', 'PC standard deviations'):
        subset = GROUPS[name]
        ref = ybar[train, subset].copy()
        for stage in np.unique(real.stage[train]):
            group = real.stage[train] == stage
            ref[group] -= ref[group].mean(0)
        ref_scale = np.std(ybar[train, subset], axis=0).clip(1e-6)
        fake = Y[train, :, subset] - Y[train, :, subset].mean(1, keepdims=True)
        fake = fake.reshape(-1, 50)
        result['geometry'][name] = {'real_within_stage_effective_dimension': effective_dimension(ref / ref_scale),
                                   'generated_within_target_effective_dimension': effective_dimension(fake / ref_scale),
                                   'within_trace_SD_ratio': float(np.sqrt(np.var(fake / ref_scale, axis=0, ddof=1).sum() /
                                                                                  np.var(ref / ref_scale, axis=0, ddof=1).sum()))}
    result['raw_per_donor_coverage'] = [{'donor': str(d), 'heldout': bool(held[i]), 'stage': int(real.stage[i]),
                                        'coverage': float(((actual[:, i] >= low[i]) & (actual[:, i] <= high[i])).mean())}
                                       for i, d in enumerate(real.donors)]
    return result


def projection_control(real):
    """Real-data-only control: train-selected PCs vs fixed random projections.

The random basis is chosen without expression data and normalized on train
cells only. This probes feature-space generalization, not generated counts.
"""
    rng = np.random.default_rng(20261113)
    basis, _ = np.linalg.qr(rng.normal(size=(len(real.genes), 50)))
    projected = [((E.MX.log_cp10k(c.toarray()) - real.mean) / real.sd) @ basis for c in real.cells]
    train_cells = np.concatenate([p for i, p in enumerate(projected) if not real.is_heldout[i]])
    scale = train_cells.std(0).clip(1e-6)
    values = []
    pca_values = []
    for i, donor in enumerate(real.donors):
        rows = [E.MX.det_subsample(len(projected[i]), 64, E.MX.stable_seed('pc-summary-rep', donor, r))
                for r in range(E.DEFAULT_PC_REPEATS)]
        values.append(np.mean([np.std(projected[i][idx] / scale, axis=0) for idx in rows], axis=0))
        pca_values.append(np.mean([np.std(real.pcs[i][idx], axis=0) for idx in rows], axis=0))
    values, pca_values = np.stack(values), np.stack(pca_values)
    report = {}
    for label, array in [('fixed_random', values), ('train_PCA', pca_values)]:
        tr = float(array[~real.is_heldout].mean())
        ho = float(array[real.is_heldout].mean())
        report[label] = {'train_mean_SD': tr, 'heldout_mean_SD': ho, 'heldout_to_train_ratio': ho / tr}
    return report


def aggregate(records):
    output = {}
    for split in ('train', 'heldout'):
        output[split] = {}
        for method in ('raw', 'finite_rank', 'center_only', 'train_width_only', 'train_shift_and_width'):
            output[split][method] = {}
            for group in GROUPS:
                output[split][method][group] = {stat: float(np.mean([r['result'][split][method][group][stat] for r in records]))
                                                for stat in ('coverage', 'below', 'above', 'width_ratio_to_raw')}
    output['geometry'] = {g: {stat: float(np.mean([r['result']['geometry'][g][stat] for r in records]))
                              for stat in records[0]['result']['geometry'][g]} for g in ('PC means', 'PC standard deviations')}
    output['scale_levels'] = {split: {stat: float(np.mean([r['result']['scale_levels'][split][stat] for r in records]))
                                      for stat in ('real_PC_sd_mean', 'generated_PC_sd_mean')}
                              for split in ('train', 'heldout')}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folds', default='0,1,2,3,4,5,6,7,8')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    records, projection_controls = [], {}
    for fold in map(int, args.folds.split(',')):
        real = E.load_real(RUNS / f'caches/hvg2000/fold{fold}')
        projection_controls[str(fold)] = projection_control(real)
        for seed in (0, 1):
            source = RUNS / f'fits_r3/pbulk/hvg2000/fold{fold}/seed{seed}/generated_biascal@stagehalf.npz'
            result = audit_fit(source, real)
            score_root = RUNS / ('results_r3' if fold < 3 else 'results_r5/default_existing_confirm')
            old = json.loads((score_root / f'fits/pbulk~r3#biascal@stagehalf/hvg2000/fold{fold}/seed{seed}.json').read_text())
            assert abs(result['heldout']['raw']['all coordinates']['coverage'] - E.headline(old)['pc_coverage80_rep']) < 1e-12
            record = {'fold': fold, 'seed': seed, 'source': str(source), 'source_sha256': E.sha256(source), 'result': result}
            records.append(record)
            (OUT / f'fold{fold}_seed{seed}.json').write_text(json.dumps(record, indent=2) + '\n')
            print('audited', fold, seed, flush=True)
    report = {'definition': __doc__, 'n_fits': len(records), 'summary': aggregate(records), 'null': null_experiment(),
              'projection_control': projection_controls,
              'per_fit': records, 'limitations': ['37 development donors reused across overlapping folds; no independent final test',
              'Training-fitted affine intervals are diagnostics without conformal guarantees',
              'No network weights, selected defaults or historical scoring rules changed']}
    (OUT / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'summary': report['summary'], 'null': report['null']}, indent=2))


if __name__ == '__main__':
    E.set_torch_threads()
    main()
