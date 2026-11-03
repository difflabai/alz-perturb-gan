"""Calibration-only mixture ES diagnostics; never a qualifying candidate result."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist

ROOT = Path('/ml2/alz')


def energy(y, target):
    return float(np.linalg.norm(y - target, axis=1).mean() - .5 * cdist(y, y).sum() / (200 * 199))


def main():
    target = np.load(ROOT / 'data/calibration_fold0/real_summaries.npz', allow_pickle=False)
    fractions = [0., .25, .5, .75, 1.]
    scores, theoretical = {f: [] for f in fractions}, {f: [] for f in fractions}
    separations = []
    cell_scores = {f: [] for f in fractions}
    real_cells = np.load(ROOT / 'data/calibration_fold0/cell_reservoir.npz', allow_pickle=False)['equal_stage']
    for rep in [0, 1]:
        generated = np.load(ROOT / f'runs/CAL_GPU0_V1_latent8_v1_f0_pg_s0/r{rep}/arrays.npz', allow_pickle=False)
        sampled = np.load(ROOT / f'artifacts/calibration_resampling_seed0/r{rep}/arrays.npz', allow_pickle=False)
        g, r = generated['stage_synthetic_summaries'], sampled['stage_synthetic_summaries']
        gc = generated['generated_cells_by_stage'].reshape(-1, 50)
        rc = sampled['generated_cells_by_stage'].reshape(-1, 50)
        gg, rr, gr = cdist(gc, gc).mean(), cdist(rc, rc).mean(), cdist(gc, rc).mean()
        gt, rt, tt = cdist(gc, real_cells).mean(), cdist(rc, real_cells).mean(), cdist(real_cells, real_cells).mean()
        for fraction in fractions:
            cell_scores[fraction].append(float(2 * (fraction * gt + (1 - fraction) * rt)
                                               - fraction**2 * gg - (1 - fraction)**2 * rr
                                               - 2 * fraction * (1 - fraction) * gr - tt))
        for stage, real in zip(target['stage'], target['summary']):
            a, b = g[stage], r[stage]
            aa = cdist(a, a).sum() / (200 * 199)
            bb = cdist(b, b).sum() / (200 * 199)
            ab = cdist(a, b)
            # Source donor/library templates are paired across methods. Drop
            # the matched diagonal for an independent-template cross term.
            cross = (ab.sum() - np.trace(ab)) / (200 * 199)
            separation = 2 * cross - aa - bb
            separations.append(float(separation))
            ordering = np.random.default_rng(20261112 + 10 * rep + int(stage)).permutation(200)
            for fraction in fractions:
                mask = np.zeros(200, dtype=bool)
                mask[ordering[:round(200 * fraction)]] = True
                y = np.where(mask[:, None], a, b)
                scores[fraction].append(energy(y, real))
                theoretical[fraction].append(fraction * energy(a, real) + (1 - fraction) * energy(b, real)
                                              - .5 * fraction * (1 - fraction) * separation)
    result = {
        'status': 'DIAGNOSTIC_MIXTURE_BOUNDS_NOT_ELIGIBLE_FOR_SELECTION',
        'fixed_grid': fractions,
        'meaning': 'fraction of complete donor bags from PG latent8; remainder hierarchical resampling. One global mixture weight; fixed independent mask ordering.',
        'finite200bag_mixture_ES': {str(f): float(np.mean(scores[f])) for f in fractions},
        'weighted_population_mixture_ES_estimate': {str(f): float(np.mean(theoretical[f])) for f in fractions},
        'weighted_empirical_cell_mixture_distance_estimate': {str(f): float(np.mean(cell_scores[f])) for f in fractions},
        'mean_PG_resampling_donor_distribution_separation': float(np.mean(separations)),
        'limitations': 'Cell reservoirs lack bag identities; weighted empirical mixture is not the exact reservoir of a finite200bag hybrid. Exact hybrid cellED/counts not scored here. This probe cannot satisfy the joint acceptance rule or count as a measured hybrid model. Calibration data only; no evaluation selection.',
    }
    (ROOT / 'analysis/mixture_ES_probe.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
