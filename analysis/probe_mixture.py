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
    for rep in [0, 1]:
        g = np.load(ROOT / f'runs/CAL_GPU0_V1_latent8_v1_f0_pg_s0/r{rep}/arrays.npz', allow_pickle=False)['stage_synthetic_summaries']
        r = np.load(ROOT / f'artifacts/calibration_resampling_seed0/r{rep}/arrays.npz', allow_pickle=False)['stage_synthetic_summaries']
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
        'status': 'DIAGNOSTIC_DONOR_ES_ONLY_NOT_ELIGIBLE_FOR_SELECTION',
        'fixed_grid': fractions,
        'meaning': 'fraction of complete donor bags from PG latent8; remainder hierarchical resampling. One global mixture weight; fixed independent mask ordering.',
        'finite200bag_mixture_ES': {str(f): float(np.mean(scores[f])) for f in fractions},
        'weighted_population_mixture_ES_estimate': {str(f): float(np.mean(theoretical[f])) for f in fractions},
        'mean_PG_resampling_donor_distribution_separation': float(np.mean(separations)),
        'limitations': 'Cell reservoirs lack bag identities; exact hybrid cellED/counts not scored here. This probe cannot satisfy the joint acceptance rule or count as a measured hybrid model. Calibration data only; no evaluation selection.',
    }
    (ROOT / 'analysis/mixture_ES_probe.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
