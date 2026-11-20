#!/usr/bin/env python3
"""Post hoc (declared after the primary scores were seen) cell-level diagnostics.

For each generator's cell_pc items: within-donor SD ratio generated/real per
PC, and a quadratic logistic C2ST on [z, z^2] of the 50 PCs (same folds and
batched trainer as the declared linear classifier). Descriptive only.
"""
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(k, '3')
import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import c2st as C  # noqa: E402
import cli  # noqa: E402
import features as F  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', required=True); p.add_argument('--out', required=True)
    p.add_argument('--plan', default=str(HERE / 'plan.json')); p.add_argument('--device', default='cuda')
    a = p.parse_args()
    cli.setup_torch(a.device, .04)
    E = cli.evaluator(HERE.parent / 'improve_20261109/r5')
    plan = cli.load_plan(a.plan)
    real = E.load_real(a.cache)
    out = {}
    rc = {}
    for name in plan['generators']:
        gen, digest = cli._checked_generated(E, plan, name, real)
        X, y, g = F.build(real, gen, 'cell_pc', rc)
        donors = np.unique(g)
        ratio = np.array([[X[(g == d) & (y == 1), j].std() / X[(g == d) & (y == 0), j].std() for j in range(X.shape[1])]
                          for d in donors])
        lr = np.exp(np.log(ratio).mean(0))
        folds, strata = cli.folds_for(g, real)
        mu = np.array([X[(g == d) & (y == 0)].mean(0) for d in donors])  # not used by classifiers; centering below is global
        Z = (X - X.mean(0)) / X.std(0)
        Q = np.concatenate([Z, Z ** 2], 1)
        res = C.run_c2st(Q, y, g, folds, 'mlp', cli.mlp_cfg('cell_pc', a.device, 'linear'))
        summ = res.summary(cli.N_BOOT, 0)
        out[name] = dict(sha256=digest, quadratic_linear_auc=summ['auc'], quadratic_linear_ci=summ['ci'],
                         sd_ratio_geomean_by_pc={f'PC{j+1}': float(lr[j]) for j in range(10)},
                         sd_ratio_geomean_all_pcs=float(np.exp(np.log(ratio).mean())),
                         frac_pcs_ratio_below_1=float((lr < 1).mean()))
        print(name, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out[name].items()
                                if k != 'sd_ratio_geomean_by_pc'}),
              {k: round(v, 3) for k, v in list(out[name]['sd_ratio_geomean_by_pc'].items())[:6]}, flush=True)
    cli.write(a.out, dict(status='post hoc descriptive', plan_sha256=cli.sha(a.plan), results=out))


if __name__ == '__main__':
    main()
