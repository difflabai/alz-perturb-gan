"""PROTOCOL part A: frozen-metric Monte Carlo variability of the selected native EMPF model.

Scores each replicate archive with the frozen r5 architecture_score (held-out donors) and compares
the replicate spread with the recorded single-archive EMPF, previous-default (0.9.7, current32)
and stage-blind empirical resampling (empirical32) values. Descriptive; one fit only.
"""
from __future__ import annotations

import os
os.environ.update(OMP_NUM_THREADS='3', OPENBLAS_NUM_THREADS='3', MKL_NUM_THREADS='3', CUDA_VISIBLE_DEVICES='')
import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

R5 = C.ROOT / 'analysis/improve_20261109/r5'
RECORDED = C.ROOT / 'analysis/empf_comparison_20261117/completion/comparison.json'
KEYS = ['pb_rmse', 'meanlog_rmse_gene', 'nonzero_rmse_gene', 'logvar_rmse_gene', 'pc_energy_rep', 'cell_energy_rep',
        'pc_coverage80_rep', 'hn_spearman_vs_heldout', 'hn_spearman_vs_train']
INTERVAL = [('sd', 'coverage'), ('sd', 'width'), ('sd', 'interval_score'),
            ('combined', 'coverage'), ('combined', 'width'), ('combined', 'interval_score')]


def better(key, values, reference):
    """Count replicates better than a reference: closer to .8 for coverage, higher for Spearman,
    lower for errors, energies and interval scores; None for widths (no direction)."""
    if reference is None or key.endswith('width'):
        return None
    if 'coverage' in key:
        return int((np.abs(values - .8) < abs(reference - .8)).sum())
    if 'spearman' in key:
        return int((values > reference).sum())
    return int((values < reference).sum())


def flat(score):
    out = {k: score['historical'].get(k) for k in KEYS}
    held = score['intervals']['heldout']['percentile']
    out.update({f'held_{g}_{m}': held[g][m] for g, m in INTERVAL})
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replicates', type=Path, nargs='+', required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(argv)
    sys.path.insert(0, str(R5))
    import architecture_score as A
    rows = {}
    for path in a.replicates:
        receipt = json.loads(path.with_suffix('.receipt.json').read_text())
        if receipt['output_sha256'] != C.sha(path) or receipt['targets'] != 37:
            raise ValueError(f'bad replicate {path}')
        score = A.score(path, C.CACHE, path.with_name(path.stem + '_score.json'))
        rows[int(receipt['replicate'])] = flat(score)
    recorded = json.loads(RECORDED.read_text())['summary']

    def recorded_flat(name):
        s = recorded[name]
        out = {k: s['historical'].get(k) for k in KEYS}
        out.update({f'held_{g}_{m}': s[f'held_{g}_{m}'] for g, m in INTERVAL})
        return out
    ref = {name: recorded_flat(name) for name in ('empf_native', 'current32', 'empirical32')}
    keys = list(next(iter(rows.values())))
    mc = {}
    for k in keys:
        v = np.array([rows[r][k] for r in sorted(rows)], float)
        sd = float(v.std(ddof=1))
        prev, empf, emp = ref['current32'][k], ref['empf_native'][k], ref['empirical32'][k]
        mc[k] = dict(mean=float(v.mean()), sd=sd, min=float(v.min()), max=float(v.max()),
                     recorded_empf_archive=empf, recorded_previous_default=prev, stage_blind_empirical=emp,
                     recorded_empf_minus_previous=(empf - prev) if None not in (empf, prev) else None,
                     difference_exceeds_2_mc_sd=(abs(empf - prev) > 2 * sd) if None not in (empf, prev) else None,
                     replicate_mean_minus_previous=(float(v.mean()) - prev) if prev is not None else None,
                     replicates_better_than_previous=better(k, v, prev),
                     replicates_better_than_empirical=better(k, v, emp))
    result = dict(status='completed_frozen_metric_monte_carlo', protocol='analysis/query_tool_20261120/PROTOCOL.md',
                  scope='one native EMPF fit (fold0/seed0), generation replicates only; fit-to-fit variability unmeasured; '
                        'previous-default and empirical values are single recorded archives without their own MC spread',
                  replicates=sorted(rows), per_replicate=rows, monte_carlo=mc,
                  inputs={str(p): C.sha(p) for p in a.replicates}, recorded_comparison_sha256=C.sha(RECORDED),
                  score_code_sha256=C.sha(R5 / 'architecture_score.py'), source_sha256=C.sha(__file__))
    C.write_json(a.out, result)
    for k in ('pb_rmse', 'held_sd_interval_score', 'held_combined_interval_score', 'held_sd_coverage', 'meanlog_rmse_gene'):
        print(k, {kk: (round(vv, 5) if isinstance(vv, float) else vv) for kk, vv in mc[k].items()})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
