"""Round-5 work item 2 Fix B (pzero): paired, like-for-like comparisons from evaluator per-fit scores.

Added by the round-5 repair (adversarial review F3, F4, F2).  The evaluator's automatic dCtrl does not pair
the pzero arms (``_control_arm`` strips only t./g./d. flags, so 'pbulk,pzero,t.kappa@0.19' looks for a
control 'pbulk,pzero'), so the comparisons are fixed here, in advance:

  block effect (kappa matched)        pbulk,pzero,t.kappa@0.19        - pbulk,t.kappa@0.19
  block + width-matched penalty       pbulk,pzero,t.kappa@0.19        - pbulk,t.kappa@0.25
  vs the round-3 control              pbulk,pzero,t.kappa@0.19        - pbulk
  swap pbulk -> pzero (not an add)    pzero,t.kappa@0.25              - pbulk,t.kappa@0.25
  kappa alone (no block)              pbulk,t.kappa@0.19              - pbulk,t.kappa@0.25
  optional block B (kappa matched)    g.dispz,pbulk,pzero,t.kappa@0.19 - g.dispz,pbulk,t.kappa@0.19
  optional block B (r3 comparator)    g.dispz,pbulk,pzero,t.kappa@0.19 - g.dispz,pbulk,t.kappa@0.25

Each at the same generation mode (snapens, stagehalf, stageblind, biascal@stagehalf), paired by (fold, seed)
on the folds and seeds both units have.  Per metric: fold-balanced mean difference, seed-noise SE
(sqrt(sum_f var_f / n_f) / F over folds with >= 2 seeds), fold-scatter SE and t, folds lower.

Targets (PROTOCOL_R5): meanlog_rmse_gene and nonzero_rmse_gene against resampling (results_r3 baselines,
M = 32) AND against the direct train-mean predictor (route A raw, smoke_r5/shape/real_fold{k}.json; repair
F2: resampling is not a floor for these two metrics), with pb_rmse, pc_energy_rep and cell_energy_rep no
worse than the comparator beyond 2 seed-noise SE.  Repair F4: those three guards also reward collapse toward
the pooled profile, so every comparison also reports pc_spread_rep and pc_coverage80_rep (nominal 0.80); a
guard "improvement" that comes with a large drop of pc_spread_rep / coverage is flagged ``collapse_flag``.

Usage (bash, from /tmp)::

    /mnt/nvme/ai/conda/anaconda3/bin/python3 -B r5/pzero_compare.py --results runs/improve_20261109/results_r5/pzero \\
        [--folds 0,1,2] [--seeds 0,1] [--out runs/improve_20261109/results_r5/pzero/pzero_compare.json]

Only reads evaluator outputs; writes one JSON (and prints a table).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

RUNS = Path('/ml2/alz/runs/improve_20261109')
PANEL = 'hvg2000'
MODES = ['snapens', 'stagehalf', 'stageblind', 'biascal@stagehalf']
PAIRS = [('block effect (kappa matched)', 'pbulk,pzero,t.kappa@0.19', 'pbulk,t.kappa@0.19'),
         ('block + width-matched penalty', 'pbulk,pzero,t.kappa@0.19', 'pbulk,t.kappa@0.25'),
         ('vs round-3 control', 'pbulk,pzero,t.kappa@0.19', 'pbulk'),
         ('swap pbulk->pzero', 'pzero,t.kappa@0.25', 'pbulk,t.kappa@0.25'),
         ('kappa alone', 'pbulk,t.kappa@0.19', 'pbulk,t.kappa@0.25'),
         ('B: block effect (kappa matched)', 'g.dispz,pbulk,pzero,t.kappa@0.19', 'g.dispz,pbulk,t.kappa@0.19'),
         ('B: vs r3 comparator', 'g.dispz,pbulk,pzero,t.kappa@0.19', 'g.dispz,pbulk,t.kappa@0.25')]
GUARDS = ['pb_rmse', 'pc_energy_rep', 'cell_energy_rep']
TARGETS = ['meanlog_rmse_gene', 'nonzero_rmse_gene']
SPREAD = ['pc_spread_rep', 'pc_coverage80_rep']
METRICS = GUARDS + TARGETS + SPREAD + ['logvar_rmse_gene', 'meanlog_rmse_gene_recal', 'nonzero_rmse_gene_recal']


def headline(path):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import evaluate as EV
    return EV.headline(json.loads(Path(path).read_text()))


def scores(results, arm, mode, folds, seeds):
    out = {}
    for k in folds:
        for s in seeds:
            p = results / 'fits' / f'{arm}~r3#{mode}' / PANEL / f'fold{k}' / f'seed{s}.json'
            if p.exists():
                h = headline(p)
                out[(k, s)] = {m: (None if h.get(m) is None else float(h[m])) for m in METRICS}
    return out


def paired(diffs):
    folds = sorted(diffs)
    if not folds:
        return None
    fm = np.array([np.mean(diffs[f]) for f in folds])
    D = float(fm.mean())
    within = [np.var(diffs[f], ddof=1) / len(diffs[f]) for f in folds if len(diffs[f]) > 1]
    se_seed = float(np.sqrt(np.sum(within)) / len(folds)) if within else None
    se_fold = float(fm.std(ddof=1) / np.sqrt(len(folds))) if len(folds) > 1 else None
    return {'mean_diff': D, 'per_fold': dict(zip(map(int, folds), fm.tolist())), 'se_seed_noise': se_seed,
            'se_fold_scatter': se_fold, 't_fold': (D / se_fold) if se_fold else None,
            'z_seed': (D / se_seed) if se_seed else None, 'n_folds_lower': int((fm < 0).sum()),
            'n_pairs': int(sum(len(v) for v in diffs.values()))}


def references(folds):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import evaluate as EV
    ref = {}
    for k in folds:
        r = {}
        for n in ('resample', 'resample_blind'):
            p = RUNS / 'results_r3' / 'baselines' / PANEL / f'fold{k}' / f'{n}.json'
            if p.exists():
                h = EV.headline(json.loads(p.read_text()))
                r[n] = {m: h.get(m) for m in METRICS}
        rj = RUNS / 'smoke_r5' / 'shape' / f'real_fold{k}.json'
        if rj.exists():
            fa = json.loads(rj.read_text())['floor_route_A']
            r['trainmean'] = {'meanlog_rmse_gene': fa['meanlog']['A_raw_train_mean'],
                              'nonzero_rmse_gene': fa['nonzero']['A_raw_train_mean']}
        ref[k] = r
    return ref


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--results', default=str(RUNS / 'results_r5' / 'pzero'))
    ap.add_argument('--folds', default='0,1,2')
    ap.add_argument('--seeds', default='0,1')
    ap.add_argument('--out', default=None)
    a = ap.parse_args(argv)
    results = Path(a.results)
    folds = [int(x) for x in a.folds.split(',')]
    seeds = [int(x) for x in a.seeds.split(',')]
    refs = references(folds)
    rep = {'pairs': {}, 'vs_references': {}, 'note': __doc__.split('Usage')[0]}
    cache = {}

    def S(arm, mode):
        if (arm, mode) not in cache:
            cache[(arm, mode)] = scores(results, arm, mode, folds, seeds)
        return cache[(arm, mode)]

    for label, new, old in PAIRS:
        for mode in MODES:
            A, B = S(new, mode), S(old, mode)
            common = sorted(set(A) & set(B))
            if not common:
                continue
            blk = {}
            for m in METRICS:
                diffs = {}
                for (k, s) in common:
                    if A[(k, s)][m] is not None and B[(k, s)][m] is not None:
                        diffs.setdefault(k, []).append(A[(k, s)][m] - B[(k, s)][m])
                blk[m] = paired(diffs)
            g = {m: blk[m] for m in GUARDS if blk.get(m)}
            blk['guards_worse_beyond_2SE'] = [m for m, p in g.items() if p['se_seed_noise'] and p['mean_diff'] > 2 * p['se_seed_noise']]
            blk['guards_better_beyond_2SE'] = [m for m, p in g.items() if p['se_seed_noise'] and p['mean_diff'] < -2 * p['se_seed_noise']]
            sp, cv = blk.get('pc_spread_rep'), blk.get('pc_coverage80_rep')
            blk['collapse_flag'] = bool(blk['guards_better_beyond_2SE'] and sp and cv and sp['se_seed_noise'] and cv['se_seed_noise']
                                        and sp['mean_diff'] < -2 * sp['se_seed_noise'] and cv['mean_diff'] < -2 * cv['se_seed_noise'])
            blk['pairs_used'] = [list(x) for x in common]
            rep['pairs'][f'{label} | {new} - {old} | {mode}'] = blk
    for arm in sorted({x for _, n, o in PAIRS for x in (n, o)}):
        for mode in MODES:
            A = S(arm, mode)
            if not A:
                continue
            row = {}
            for refn in ('resample', 'resample_blind', 'trainmean'):
                for m in TARGETS:
                    diffs = {}
                    for (k, s), v in A.items():
                        rv = refs.get(k, {}).get(refn, {}).get(m)
                        if rv is not None and v[m] is not None:
                            diffs.setdefault(k, []).append(v[m] - rv)
                    row[f'{m} - {refn}'] = paired(diffs)
            row['means'] = {m: float(np.mean([v[m] for v in A.values() if v[m] is not None])) for m in METRICS
                            if any(v[m] is not None for v in A.values())}
            row['n_fits'] = len(A)
            rep['vs_references'][f'{arm} | {mode}'] = row
    out = Path(a.out) if a.out else results / 'pzero_compare.json'
    out.write_text(json.dumps(rep, indent=1, allow_nan=False))
    show = GUARDS + TARGETS + SPREAD
    print('PAIRED new - old (mean diff [seed SE], folds lower)')
    for name, blk in rep['pairs'].items():
        print(name)
        print('   ' + '; '.join(f"{m} {blk[m]['mean_diff']:+.4f} [{(blk[m]['se_seed_noise'] or float('nan')):.4f}] "
                                f"{blk[m]['n_folds_lower']}/{len(blk[m]['per_fold'])}" for m in show if blk.get(m)))
        print(f"   guards worse >2SE {blk['guards_worse_beyond_2SE']}; better >2SE {blk['guards_better_beyond_2SE']}; "
              f"collapse_flag {blk['collapse_flag']}; n pairs {len(blk['pairs_used'])}")
    print('\nTARGETS vs references (model - reference; meanlog / nonzero)')
    for name, row in rep['vs_references'].items():
        print(f'{name:55s} ' + '  '.join(f"{k.replace('_rmse_gene', '')} {p['mean_diff']:+.4f}" for k, p in row.items()
                                          if isinstance(p, dict) and 'mean_diff' in p))
    print(f'-> {out}')


if __name__ == '__main__':
    main()
