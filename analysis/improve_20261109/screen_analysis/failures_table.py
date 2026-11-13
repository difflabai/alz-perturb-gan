"""failures_table.py - train vs held-out fit quality per arm (generalisation vs optimisation).

Reads only the evaluator's per-fit JSON (results/fits/{arm}/{panel}/fold{k}/seed{s}.json).
For each fit: pb_rmse, mean_log rmse/bias, nonzero bias, pb-distribution and PC-space energy,
coverage80, spread, within-condition spread on TRAIN and HELD-OUT donors.  Then fold-balanced
means (seeds averaged within fold, then mean over folds) and paired differences vs base
(same fold, same seed).  Output: printed tables + runs/.../screen_analysis/failures_table.json.

Run: OMP_NUM_THREADS=6 python3 -B failures_table.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
RES = Path('/ml2/alz/runs/improve_20261109/results/fits')
OUT = Path('/ml2/alz/runs/improve_20261109/screen_analysis/failures_table.json')
ARMS = ['base', 'offset', 'genemean', 'donorpc', 'covars', 'covars,genemean,offset']


def metrics(d, split):
    a = d['subsets']['all'][split]
    pc = d['pc_space'][split]
    return {
        'pb_rmse': a['pseudobulk']['rmse'],
        'ml_rmse': a['mean_log']['rmse_gene'],
        'ml_bias': a['mean_log']['bias'],
        'ml_Q4_bias': a['mean_log']['by_expression_quartile']['Q4']['bias'],
        'nz_bias': a['nonzero']['bias'],
        'pbd_energy': a['distribution_pb']['energy_score'],
        'pbd_cov80': a['distribution_pb']['coverage80'],
        'pbd_wc_spread': a['distribution_pb']['within_condition_spread_ratio'],
        'pc_energy': pc['energy_score'],
        'pc_cov80': pc['coverage80'],
        'pc_spread': pc['spread_ratio'],
        'pc_wc_spread': pc['within_condition_spread_ratio'],
        'cell_energy': pc['cell_energy_distance'],
    }


def load(panel='hvg2000'):
    rows = {}
    for arm in ARMS:
        for k in range(3):
            for s in range(2):
                p = RES / arm / panel / f'fold{k}' / f'seed{s}.json'
                if not p.exists():
                    continue
                d = json.loads(p.read_text())
                rows[(arm, k, s)] = {'train': metrics(d, 'train'), 'heldout': metrics(d, 'heldout'),
                                     'lib_ratio': d['generated_library_vs_target']['median_ratio_generated_lib_to_target_mean_lib']}
    return rows


def fb_mean(rows, arm, split, key):
    folds = []
    for k in range(3):
        v = [rows[(arm, k, s)][split][key] for s in range(2) if (arm, k, s) in rows]
        if v:
            folds.append(np.mean(v))
    return float(np.mean(folds)), float(np.std(folds, ddof=1) / np.sqrt(len(folds))) if len(folds) > 1 else float('nan')


def main():
    rows = load()
    keys = list(next(iter(rows.values()))['train'].keys())
    out = {'fold_balanced': {}, 'paired_vs_base': {}, 'per_fit': {f'{a}|{k}|{s}': v for (a, k, s), v in rows.items()}}
    for split in ['train', 'heldout']:
        print(f'\n== {split} donors (fold-balanced mean, hvg2000) ==')
        print('arm'.ljust(24) + ''.join(k.rjust(13) for k in keys))
        for arm in ARMS:
            vals = [fb_mean(rows, arm, split, k)[0] for k in keys]
            out['fold_balanced'][f'{arm}|{split}'] = dict(zip(keys, vals))
            print(arm.ljust(24) + ''.join(f'{v:13.3f}' for v in vals))
    print('\n== held-out minus train gap (generalisation gap) ==')
    for arm in ARMS:
        g = {k: out['fold_balanced'][f'{arm}|heldout'][k] - out['fold_balanced'][f'{arm}|train'][k] for k in ['pb_rmse', 'ml_rmse', 'pc_energy', 'pc_cov80', 'pbd_energy']}
        out['fold_balanced'][f'{arm}|gap'] = g
        print(arm.ljust(24) + '  '.join(f'{k}={v:+.3f}' for k, v in g.items()))
    print('\n== paired (arm - base, same fold & seed), pb_rmse; train | heldout; count arm worse ==')
    for arm in ARMS[1:]:
        dt = [rows[(arm, k, s)]['train']['pb_rmse'] - rows[('base', k, s)]['train']['pb_rmse'] for k in range(3) for s in range(2)]
        dh = [rows[(arm, k, s)]['heldout']['pb_rmse'] - rows[('base', k, s)]['heldout']['pb_rmse'] for k in range(3) for s in range(2)]
        out['paired_vs_base'][arm] = {'train_pb_rmse_diff': dt, 'heldout_pb_rmse_diff': dh}
        print(f'{arm:24s} train mean {np.mean(dt):+.3f} (worse {sum(x > 0 for x in dt)}/6) | heldout mean {np.mean(dh):+.3f} (worse {sum(x > 0 for x in dh)}/6)')
    print('\n== generated/target library ratio (median over targets) ==')
    for arm in ARMS:
        print(arm.ljust(24), [round(rows[(arm, k, s)]['lib_ratio'], 4) for k in range(3) for s in range(2)])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print('wrote', OUT)


if __name__ == '__main__':
    main()
