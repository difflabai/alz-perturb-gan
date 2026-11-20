#!/usr/bin/env python3
"""Post hoc fold-disjoint real-donor reference (declared after primary scores).

Empirical resampling reuses the same 25 GAN-training donors as the "generated"
class in every fold, so a C2ST can recognise those source donors across folds.
This reference instead draws class-1 items for a held-out target from real
training donors assigned to that target's CV fold only (training donors are
partitioned into 6 stage-spread groups, one per fold), so no source donor is
ever seen in training as class 1 for another fold. Stage-blind, no depth
thinning: an approximation of an ideal generator of novel donors. Same real
items, folds and classifiers as the declared study; paired difference vs the
primary generator from its saved out-of-fold scores. Descriptive only.
"""
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(k, '3')
import argparse
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import c2st as C  # noqa: E402
import cli  # noqa: E402
import features as F  # noqa: E402


def items(real, level, folds_of_target, groups_of_fold, cap=256, n_bags=32, bag_k=256, pc_k=64):
    X, g = [], []
    for i in F.heldout_index(real):
        src = groups_of_fold[folds_of_target[i]]
        rng = np.random.default_rng(F._seed('c2st-disjoint', level, str(real.donors[i])))
        if level == 'cell_pc':
            n = min(len(real.pcs[i]), cap)
            pool = np.concatenate([real.pcs[s] for s in src]).astype(np.float16).astype(np.float64)
            X.append(pool[rng.choice(len(pool), n, replace=False)]); g += [i] * n
        else:
            bags = []
            for m in range(n_bags):
                s = src[m % len(src)]
                if level == 'bag_pcsum':
                    P = real.pcs[s].astype(np.float16).astype(np.float64)
                    r = rng.choice(len(P), pc_k, replace=len(P) < pc_k)
                    bags.append(np.concatenate([P[r].mean(0), P[r].std(0)]))
                else:
                    Cm = real.cells[s]; r = rng.integers(0, Cm.shape[0], bag_k)
                    bags.append(F.log_cp10k(Cm[r].toarray()).mean(0))
            X.append(np.stack(bags)); g += [i] * n_bags
    X = np.concatenate(X)
    if level in ('bag_inspan', 'bag_offspan'):
        X = F.span_split(X, real)[level == 'bag_offspan']
    return X, np.asarray(g)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', required=True); p.add_argument('--work', required=True); p.add_argument('--out', required=True)
    p.add_argument('--plan', default=str(HERE / 'plan.json')); p.add_argument('--device', default='cuda')
    a = p.parse_args()
    cli.setup_torch(a.device, .04)
    E = cli.evaluator(HERE.parent / 'improve_20261109/r5')
    plan = cli.load_plan(a.plan)
    real = E.load_real(a.cache)
    train = np.flatnonzero(~real.is_heldout)
    rng = np.random.default_rng(cli.FOLD_SEED)
    order = np.concatenate([rng.permutation(train[real.stage[train] == s]) for s in range(4)])
    groups_of_fold = {k: order[k::6].tolist() for k in range(6)}
    out = dict(status='post hoc descriptive', plan_sha256=cli.sha(a.plan), source_groups={
        str(k): [str(real.donors[s]) for s in v] for k, v in groups_of_fold.items()}, levels={})
    scores = Path(a.work) / 'scores'
    for level in plan['levels']:
        Xr, gr = F.real_items(real, level)
        folds_r, _ = cli.folds_for(gr, real)
        fot = {int(i): int(folds_r[np.flatnonzero(gr == i)[0]]) for i in np.unique(gr)}
        Xg, gg = items(real, level, fot, groups_of_fold)
        X = np.concatenate([Xr, Xg]); y = np.r_[np.zeros(len(Xr), int), np.ones(len(Xg), int)]; g = np.r_[gr, gg]
        out['levels'][level] = {}
        for clf in plan['classifiers']:
            res, summ = cli.score_one(X, y, g, real, level, clf, a.device)
            b = np.load(scores / f'{plan["primary"]}_{level}_{clf}.npz')
            assert np.array_equal(b['folds'], res.folds) and np.array_equal(b['y'], res.y)
            summ['real_disjoint_minus_primary'] = C.paired_bootstrap_diff(b['score'], b['y'], b['groups'],
                                                                          res.score, res.y, res.groups, cli.N_BOOT, 0)
            out['levels'][level][clf] = summ
            print(level, clf, round(summ['auc'], 4), [round(c, 3) for c in summ['ci']],
                  'diff vs primary', round(summ['real_disjoint_minus_primary']['diff'], 4),
                  [round(c, 3) for c in summ['real_disjoint_minus_primary']['ci']], flush=True)
        cli.write(a.out, out)


if __name__ == '__main__':
    main()
