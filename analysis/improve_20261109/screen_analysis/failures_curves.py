"""failures_curves.py - learning-curve comparison of the failing arms with base (hvg2000, same fold & seed).

Reads fits/{arm}/hvg2000/fold{k}/seed{s}/learning_curve.jsonl (rows every 250 updates).
Per fit and window of updates: mean critic_gap, D_adversarial, D_penalty, G_adversarial,
D/G gradient norms, prior regulariser, particle covariance trace; plus late-phase drift
(slope of critic gap over 10k-20k) and NaN/spike checks.  Prints window means averaged over
the 6 fits of each arm, and writes runs/.../screen_analysis/failures_curves.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
FITS = Path('/ml2/alz/runs/improve_20261109/fits')
OUT = Path('/ml2/alz/runs/improve_20261109/screen_analysis/failures_curves.json')
ARMS = ['base', 'offset', 'genemean', 'donorpc', 'covars', 'covars,genemean,offset']
WINDOWS = [(1, 2000), (2000, 5000), (5000, 10000), (10000, 15000), (15000, 20001)]
FIELDS = ['critic_gap', 'D_adversarial', 'D_penalty', 'G_adversarial', 'D_gradient_norm', 'G_gradient_norm',
          'prior_regularization_unweighted']


def rows(arm, k, s):
    p = FITS / arm / 'hvg2000' / f'fold{k}' / f'seed{s}' / 'learning_curve.jsonl'
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def get(r, f):
    if f == 'particle_cov_trace':
        return r['particle_usage']['particle_covariance_trace']
    v = r.get(f)
    return np.nan if v is None else float(v)


def main():
    out = {}
    fields = FIELDS + ['particle_cov_trace']
    for arm in ARMS:
        for k in range(3):
            for s in range(2):
                R = rows(arm, k, s)
                steps = np.array([r['step'] for r in R])
                d = {'n_rows': len(R), 'last_step': int(steps.max())}
                for f in fields:
                    v = np.array([get(r, f) for r in R], dtype=float)
                    d[f] = [float(np.nanmean(v[(steps >= a) & (steps < b)])) for a, b in WINDOWS]
                    d[f + '_nonfinite'] = int((~np.isfinite(v[steps > 1])).sum()) if f in ('critic_gap', 'D_adversarial', 'G_adversarial') else None
                g = np.array([get(r, 'critic_gap') for r in R]); m = steps >= 10000
                d['gap_slope_per_1k_10k_20k'] = float(np.polyfit(steps[m] / 1000, g[m], 1)[0])
                d['gap_max'] = float(np.nanmax(g)); d['gap_min_after_1k'] = float(np.nanmin(g[steps > 1000]))
                out[f'{arm}|{k}|{s}'] = d
    print('window means over 6 fits (windows: ' + ', '.join(f'{a}-{b - 1 if b > 20000 else b}' for a, b in WINDOWS) + ')')
    summ = {}
    for f in fields:
        print(f'\n{f}')
        for arm in ARMS:
            m = np.mean([out[f'{arm}|{k}|{s}'][f] for k in range(3) for s in range(2)], 0)
            sd = np.std([out[f'{arm}|{k}|{s}'][f][-1] for k in range(3) for s in range(2)])
            summ[f'{arm}|{f}'] = m.tolist()
            print(f'  {arm:24s} ' + ' '.join(f'{x:9.4g}' for x in m) + f'   (sd last window {sd:.3g})')
    print('\ncritic-gap slope over 10k-20k (per 1k updates), max gap, nonfinite rows')
    for arm in ARMS:
        sl = [out[f'{arm}|{k}|{s}']['gap_slope_per_1k_10k_20k'] for k in range(3) for s in range(2)]
        mx = [out[f'{arm}|{k}|{s}']['gap_max'] for k in range(3) for s in range(2)]
        nf = sum(out[f'{arm}|{k}|{s}']['critic_gap_nonfinite'] for k in range(3) for s in range(2))
        summ[f'{arm}|gap_slope'] = sl
        print(f'  {arm:24s} slope {np.round(sl, 3).tolist()} max {np.round(mx, 2).tolist()} nonfinite {nf}')
    OUT.write_text(json.dumps({'per_fit': out, 'summary': summ, 'windows': WINDOWS}, indent=1))
    print('wrote', OUT)


if __name__ == '__main__':
    main()
