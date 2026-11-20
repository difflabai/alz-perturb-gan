#!/usr/bin/env python3
"""Audit repairs declared in PROTOCOL.md Amendment 2 (post hoc, descriptive).

  refit        classifier-refit sensitivity of the paired differences
  secondorder  full second-order cell-level classifiers vs the MLP
  pparg        PPARG permutation importance with repeats and refits
  merge        combine phase outputs into results/repairs.json

Same cache, real items and classifier settings as the declared study.
"""
from __future__ import annotations

import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_k, '3')
import argparse
import dataclasses
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import c2st as C  # noqa: E402
import cli  # noqa: E402
import features as F  # noqa: E402

REPLICATES = (0, 1, 2, 3, 4)
GENS = ('empf05', 'empf025', 'prev097', 'resample_stage', 'resample_blind')
PPARG_GENS = ('empf05', 'empf025', 'prev097', 'resample_blind')


def rep_cfg(level, device, clf, r):
    cfg = cli.mlp_cfg(level, device, clf)
    if r == 0:
        return cfg
    n = len(cfg.seeds)
    return dataclasses.replace(cfg, seeds=tuple(10 * r + k for k in range(n)))


def rep_folds(g, real, r):
    strata = {int(i): int(real.stage[i]) for i in np.unique(g)}
    return C.grouped_folds(g, 6, seed=cli.FOLD_SEED + r, strata=strata)


def load(a):
    E = cli.evaluator(HERE.parent / 'improve_20261109/r5')
    plan = cli.load_plan(a.plan)
    if cli.sha(Path(a.cache) / 'manifest.json') != plan['cache_manifest_sha256']:
        raise SystemExit('cache manifest differs from the declared plan')
    return E, plan, E.load_real(a.cache)


# ------------------------------------------------------------------ refit
def phase_refit(a):
    cli.setup_torch(a.device, a.gpu_mem_fraction)
    E, plan, real = load(a)
    out_dir = Path(a.work) / 'repairs' / 'refit'; out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); rc = {}
    gens = {n: cli._checked_generated(E, plan, n, real)[0] for n in GENS}
    for level in a.levels:
        for name in GENS:
            X, y, g = F.build(real, gens[name], level, rc)
            for clf in plan['classifiers']:
                for r in REPLICATES[1:]:
                    p = out_dir / f'{name}_{level}_{clf}_r{r}.npz'
                    if p.exists():
                        continue
                    folds = rep_folds(g, real, r)
                    res = C.run_c2st(X, y, g, folds, 'mlp', rep_cfg(level, a.device, clf, r))
                    np.savez_compressed(p, score=res.score, y=res.y, groups=res.groups, folds=res.folds)
                    print(f'[{time.time()-t0:6.0f}s] refit {name:15s} {level:9s} {clf:6s} r{r} AUC={res.auc:.4f}', flush=True)


def _rep_scores(work, name, level, clf):
    """Out-of-fold scores of the 5 replicates (r0 = declared run)."""
    out = [np.load(Path(work) / 'scores' / f'{name}_{level}_{clf}.npz')]
    out += [np.load(Path(work) / 'repairs' / 'refit' / f'{name}_{level}_{clf}_r{r}.npz') for r in REPLICATES[1:]]
    return out


def refit_augmented_diff(A, B, n_boot=2000, seed=0):
    """AUC(B) - AUC(A): each draw resamples donors and one replicate (shared by A and B)."""
    g = A[0]['groups']
    for s in A + B:
        if not np.array_equal(s['groups'], g) or not np.array_equal(s['y'], A[0]['y']):
            raise SystemExit('replicates must share items')
    y = A[0]['y']; uniq = np.unique(g); members = {d: np.flatnonzero(g == d) for d in uniq}
    rng = np.random.default_rng(seed); vals = []
    for _ in range(n_boot):
        r = rng.integers(len(A)); pick = rng.choice(uniq, len(uniq), replace=True)
        i = np.concatenate([members[d] for d in pick])
        vals.append(C.auc(B[r]['score'][i], y[i]) - C.auc(A[r]['score'][i], y[i]))
    return [float(np.quantile(vals, .025)), float(np.quantile(vals, .975))]


def merge_refit(work):
    out = {}
    for level in ('cell_pc', 'bag_gene'):
        for clf in ('mlp', 'linear'):
            try:
                base = _rep_scores(work, 'empf05', level, clf)
            except FileNotFoundError:
                continue
            row = dict(auc={'empf05': [C.auc(s['score'], s['y']) for s in base]}, diffs={})
            for name in GENS[1:]:
                try:
                    other = _rep_scores(work, name, level, clf)
                except FileNotFoundError:
                    continue
                for s, b in zip(other, base):
                    if not (np.array_equal(s['folds'], b['folds']) and np.array_equal(s['y'], b['y'])):
                        raise SystemExit('paired replicate must share folds and items')
                row['auc'][name] = [C.auc(s['score'], s['y']) for s in other]
                est = [C.auc(o['score'], o['y']) - C.auc(b['score'], b['y']) for o, b in zip(other, base)]
                per = [C.paired_bootstrap_diff(b['score'], b['y'], b['groups'], o['score'], o['y'], o['groups'],
                                               cli.N_BOOT, 0)['ci'] for o, b in zip(other, base)]
                ci = refit_augmented_diff(base, other, cli.N_BOOT, 0)
                sci = 1 if ci[0] > 0 else (-1 if ci[1] < 0 else 0)
                row['diffs'][f'{name}_minus_empf05'] = dict(
                    replicate_estimates=est, replicate_cis=per, mean=float(np.mean(est)),
                    refit_augmented_ci=ci, robust=bool(sci != 0 and all(np.sign(e) == sci for e in est)))
            out[f'{level}/{clf}'] = row
    return out


# ------------------------------------------------------------ second order
def _quad_features(Z, full):
    if not full:
        return np.concatenate([Z, Z ** 2], 1)
    iu = np.triu_indices(Z.shape[1])
    return np.concatenate([Z, (Z[:, :, None] * Z[:, None, :])[:, iu[0], iu[1]]], 1)


def _logistic_secondorder(X, y, g, folds, device, full):
    """Per fold: standardize with training-fold stats, build features, fit the
    declared linear trainer on training donors only, score the test fold."""
    score = np.full(len(y), np.nan)
    cfg = cli.mlp_cfg('cell_pc', device, 'linear')
    for k in np.unique(folds):
        tr = folds != k
        mu = X[tr].mean(0); sd = X[tr].std(0)
        Q = _quad_features((X - mu) / sd, full)
        two = (folds == k).astype(int)            # fold 1 = test donors, fold 0 = training donors
        models = C.fit_mlp_batched(Q, y.astype(np.float64), two, cfg)
        m = [m for m in models if m.fold == 1][0]  # trained on fold 0 (training donors) only
        score[folds == k] = m.predict(Q[folds == k])
    return score


def _qda(X, y, folds):
    from sklearn.covariance import LedoitWolf
    score = np.full(len(y), np.nan)
    for k in np.unique(folds):
        tr = folds != k; te = folds == k
        ll = []
        for c in (0, 1):
            Xc = X[tr & (y == c)]
            lw = LedoitWolf().fit(Xc)
            P = lw.precision_; _, logdet = np.linalg.slogdet(lw.covariance_)
            D = X[te] - lw.location_
            ll.append(-.5 * np.einsum('ij,jk,ik->i', D, P, D) - .5 * logdet)
        score[te] = ll[1] - ll[0]
    return score


def phase_secondorder(a):
    cli.setup_torch(a.device, a.gpu_mem_fraction)
    E, plan, real = load(a)
    out_dir = Path(a.work) / 'repairs' / 'secondorder'; out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); rc = {}; out = {}
    for name in a.generators or GENS:
        gen, digest = cli._checked_generated(E, plan, name, real)
        X, y, g = F.build(real, gen, 'cell_pc', rc)
        folds, _ = cli.folds_for(g, real)
        base = np.load(Path(a.work) / 'scores' / f'{name}_cell_pc_mlp.npz')
        assert np.array_equal(base['folds'], folds) and np.array_equal(base['y'], y)
        row = dict(sha256=digest, mlp_auc=C.auc(base['score'], y), models={})
        for model, fn in (('quad_diag', lambda: _logistic_secondorder(X, y, g, folds, a.device, False)),
                          ('quad_full', lambda: _logistic_secondorder(X, y, g, folds, a.device, True)),
                          ('qda_lw', lambda: _qda(X, y, folds))):
            s = fn()
            np.savez_compressed(out_dir / f'{name}_{model}.npz', score=s, y=y, groups=g, folds=folds)
            _, ci = C.bootstrap_auc(s, y, g, cli.N_BOOT, 0)
            d = C.paired_bootstrap_diff(s, y, g, base['score'], y, g, cli.N_BOOT, 0)   # MLP minus model
            row['models'][model] = dict(auc=C.auc(s, y), ci=list(ci), mlp_minus_model=d)
            print(f'[{time.time()-t0:6.0f}s] {name:15s} {model:9s} AUC={C.auc(s, y):.4f} CI=[{ci[0]:.3f},{ci[1]:.3f}] '
                  f'MLP-model={d["diff"]:+.4f} [{d["ci"][0]:+.3f},{d["ci"][1]:+.3f}]', flush=True)
        m = row['models']
        row['beyond_second_moments'] = bool(m['quad_full']['mlp_minus_model']['ci'][0] > 0
                                            and m['qda_lw']['mlp_minus_model']['ci'][0] > 0)
        out[name] = row
        cli.write(out_dir / f'secondorder_{"-".join(a.generators or GENS)}.json', out)


# ------------------------------------------------------------------ PPARG
def phase_pparg(a):
    cli.setup_torch(a.device, a.gpu_mem_fraction)
    E, plan, real = load(a)
    out_dir = Path(a.work) / 'repairs' / 'pparg'; out_dir.mkdir(parents=True, exist_ok=True)
    genes = list(real.genes); j = genes.index('PPARG')
    t0 = time.time(); rc = {}; out = {}
    for name in a.generators or PPARG_GENS:
        gen, _ = cli._checked_generated(E, plan, name, real)
        X, y, g = F.build(real, gen, 'bag_gene', rc)
        rows = []
        for r in REPLICATES:
            folds = rep_folds(g, real, r)
            res = C.run_c2st(X, y, g, folds, 'mlp', rep_cfg('bag_gene', a.device, 'mlp', r), keep_models=True)
            allg = C.permutation_importance(res, X, n_repeats=2, seed=100 + r)
            pp = [float(C.permutation_importance(res, X, n_repeats=1, seed=1000 * (r + 1) + k, blocks=[[j]])[0])
                  for k in range(50)]
            rank = int(np.argsort(np.argsort(-allg))[j]) + 1
            rows.append(dict(replicate=r, auc=res.auc, pparg_drop_mean=float(np.mean(pp)), pparg_drops=pp,
                             pparg_rank_2rep=rank, allgene_q95=float(np.quantile(allg, .95)),
                             allgene_median=float(np.median(allg)), allgene_max=float(allg.max())))
            print(f'[{time.time()-t0:6.0f}s] pparg {name:15s} r{r} AUC={res.auc:.4f} drop={np.mean(pp):+.5f} '
                  f'rank={rank} q95={rows[-1]["allgene_q95"]:+.5f}', flush=True)
        pooled = np.concatenate([x['pparg_drops'] for x in rows])
        out[name] = dict(replicates=rows, pooled_drop_mean=float(pooled.mean()),
                         pooled_drop_q025=float(np.quantile(pooled, .025)),
                         pooled_drop_q975=float(np.quantile(pooled, .975)),
                         relies_on_pparg=bool(all(x['pparg_drop_mean'] > x['allgene_q95'] for x in rows)
                                              and np.quantile(pooled, .025) > 0))
        cli.write(out_dir / f'pparg_{"-".join(a.generators or PPARG_GENS)}.json', out)


def phase_merge(a):
    rep = Path(a.work) / 'repairs'
    so, pp = {}, {}
    for p in sorted((rep / 'secondorder').glob('secondorder_*.json')):
        so.update(json.loads(p.read_text()))
    for p in sorted((rep / 'pparg').glob('pparg_*.json')):
        pp.update(json.loads(p.read_text()))
    for v in pp.values():
        for x in v['replicates']:
            x.pop('pparg_drops')
    cli.write(a.out, dict(protocol='analysis/c2st_20261120/PROTOCOL.md#amendment-2', status='post hoc audit repairs',
                          plan_sha256=cli.sha(a.plan), refit=merge_refit(a.work), secondorder=so, pparg=pp))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('phase', choices=('refit', 'secondorder', 'pparg', 'merge'))
    p.add_argument('--cache'); p.add_argument('--work', required=True); p.add_argument('--out')
    p.add_argument('--plan', default=str(HERE / 'plan.json'))
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    p.add_argument('--gpu-mem-fraction', type=float, default=.05)
    p.add_argument('--levels', nargs='+', default=['cell_pc', 'bag_gene'])
    p.add_argument('--generators', nargs='+')
    a = p.parse_args()
    if not 0 < a.gpu_mem_fraction <= .08:
        p.error('GPU memory fraction must be in (0, .08]')
    {'refit': phase_refit, 'secondorder': phase_secondorder, 'pparg': phase_pparg, 'merge': phase_merge}[a.phase](a)


if __name__ == '__main__':
    main()
