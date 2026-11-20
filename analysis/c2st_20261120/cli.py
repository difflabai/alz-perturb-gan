#!/usr/bin/env python3
"""Donor-held-out C2ST for microglia generators (CLI).

  score  one or more generated NPZs (repository generated.npz contract)
         against the held-out development donors of a cache.
  study  the declared comparison in PROTOCOL.md (fixed inputs/settings).

Only held-out development donors are scored; training donors never enter
an item set.  Results are JSON; out-of-fold scores go to --scores-dir.
"""
from __future__ import annotations

import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_k, '3')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import c2st as C  # noqa: E402
import features as F  # noqa: E402

STAGES = ['Not AD', 'Low', 'Intermediate', 'High']
FOLD_SEED = 20261120
CELL_MLP = dict(hidden=(128, 64), epochs=40, batch=256, lr=1e-3, weight_decay=1e-3, dropout=.1, seeds=(0, 1, 2))
BAG_MLP = dict(hidden=(64,), epochs=150, batch=64, lr=1e-3, weight_decay=1e-2, dropout=.2, seeds=(0, 1, 2))
LINEAR = dict(hidden=(), epochs=100, batch=256, lr=1e-2, weight_decay=1e-2, dropout=0., seeds=(0,))
LINEAR_L2 = 10.0   # only used by the NumPy fallback (classifier 'linear' without torch)
N_BOOT = 2000


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=1, allow_nan=False, default=float) + '\n'); tmp.replace(path)


def evaluator(r5):
    sys.path.insert(0, str(Path(r5).resolve()))
    import evaluate as E  # repository evaluator: load_real / load_generated / align_generated
    return E


def setup_torch(device, memory):
    import torch
    torch.set_num_threads(3)
    if device == 'cuda':
        if not os.environ.get('CUDA_VISIBLE_DEVICES'):
            raise SystemExit('set CUDA_VISIBLE_DEVICES explicitly for --device cuda')
        torch.cuda.set_per_process_memory_fraction(memory)
    torch.use_deterministic_algorithms(False)
    return torch


def mlp_cfg(level, device, classifier='mlp'):
    if classifier == 'linear':
        return C.MLPConfig(device=device, **LINEAR)   # logistic regression, same batched trainer
    return C.MLPConfig(device=device, **(CELL_MLP if level == 'cell_pc' else BAG_MLP))


def folds_for(groups, real):
    strata = {int(i): int(real.stage[i]) for i in np.unique(groups)}
    return C.grouped_folds(groups, 6, seed=FOLD_SEED, strata=strata), strata


def stage_named(summary):
    per = summary.pop('per_stratum', {})
    summary['per_stage'] = {STAGES[int(k)]: v for k, v in per.items()}
    return summary


def score_one(X, y, g, real, level, classifier, device, keep=False):
    folds, strata = folds_for(g, real)
    res = C.run_c2st(X, y, g, folds, 'mlp', mlp_cfg(level, device, classifier), keep_models=keep)
    return res, stage_named(res.summary(N_BOOT, 0, strata))


def gene_attribution(real, X, y, g, res, names, top=25, perm=True):
    uni = C.univariate_auc(X, y, g)
    out = dict(univariate_top=[dict(feature=names[j], within_donor_auc=float(uni[j]))
                               for j in np.argsort(-np.abs(uni - .5))[:top]])
    rank = np.argsort(np.argsort(-np.abs(uni - .5)))
    out['univariate_rank'] = rank; out['univariate'] = uni
    if perm:
        imp = C.permutation_importance(res, X, n_repeats=2, seed=0)
        out['permutation_top'] = [dict(feature=names[j], auc_drop=float(imp[j])) for j in np.argsort(-imp)[:top]]
        out['permutation_rank'] = np.argsort(np.argsort(-imp)); out['permutation'] = imp
    return out


def offspan_share(real, Xg_real, Xg_gen, genes, which):
    """Signed real-minus-generated standardized mean gap per gene and the
    fraction of that gap outside the 50-PC column space."""
    Zr = (Xg_real.mean(0) - real.mean) / real.sd; Zg = (Xg_gen.mean(0) - real.mean) / real.sd
    d = Zr - Zg
    coords, resid = F.span_split(d[None] * real.sd + real.mean, real)  # same map on the gap
    resid = resid[0]
    j = {n: k for k, n in enumerate(genes)}
    return {n: dict(z_gap_real_minus_generated=float(d[j[n]]), offspan_component=float(resid[j[n]]),
                    offspan_share=float(resid[j[n]] / d[j[n]]) if abs(d[j[n]]) > 1e-12 else None)
            for n in which if n in j}


def load_generated(E, path, real):
    return E.align_generated(E.load_generated(path), real)


# ------------------------------------------------------------------ score
def cmd_score(a):
    torch = setup_torch(a.device, a.gpu_mem_fraction)
    E = evaluator(a.r5)
    real = E.load_real(a.cache)
    cache, out = {}, {}
    for spec in a.generated:
        name, path = spec.split('=', 1)
        gen = load_generated(E, path, real)
        out[name] = dict(path=str(Path(path).resolve()), sha256=sha(path), levels={})
        for level in a.levels:
            X, y, g = F.build(real, gen, level, cache)
            res, summ = score_one(X, y, g, real, level, a.classifier, a.device)
            out[name]['levels'][level] = summ
            print(json.dumps({name: {level: {k: summ[k] for k in ('auc', 'ci')}}}), flush=True)
    write(a.out, dict(cache_manifest_sha256=sha(Path(a.cache) / 'manifest.json'), classifier=a.classifier,
                      results=out, torch=torch.__version__))


# ------------------------------------------------------------------ study
def _context(a):
    torch = setup_torch(a.device, a.gpu_mem_fraction)
    E = evaluator(a.r5)
    plan = json.loads(Path(a.plan).read_text())
    if sha(Path(a.cache) / 'manifest.json') != plan['cache_manifest_sha256']:
        raise SystemExit('cache manifest differs from the declared plan')
    real = E.load_real(a.cache)
    levels = a.levels or plan['levels']
    if not set(levels) <= set(plan['levels']):
        raise SystemExit('levels outside the declared plan')
    return torch, E, plan, real, levels


def _checked_generated(E, plan, name, real):
    spec = plan['generators'][name]; path = Path(spec['path'])
    digest = sha(path)
    if digest != expected_sha(spec):
        raise SystemExit(f'{name}: generated archive hash differs from the plan')
    return load_generated(E, path, real), digest


def expected_sha(spec):
    """Declared hash, or 'receipt' for an archive generated after the plan was
    declared: then the generation receipt next to it must match."""
    if spec['sha256'] != 'receipt':
        return spec['sha256']
    path = Path(spec['path'])
    return json.loads(path.with_name(path.stem + '.receipt.json').read_text())['generated_SHA256']


def phase_generators(a):
    torch, E, plan, real, levels = _context(a)
    t0 = time.time(); genes = list(real.genes); held = F.heldout_index(real)
    names = a.generators or list(plan['generators'])
    scores_dir = Path(a.work) / 'scores'; scores_dir.mkdir(parents=True, exist_ok=True)
    rcache = {}
    for name in names:
        gen, digest = _checked_generated(E, plan, name, real)
        row = dict(generator=name, path=plan['generators'][name]['path'], sha256=digest,
                   role=plan['generators'][name]['role'], plan_sha256=sha(a.plan), torch=torch.__version__,
                   levels={}, attribution={})
        for level in levels:
            X, y, g = F.build(real, gen, level, rcache)
            row['levels'][level] = {}
            for clf in plan['classifiers']:
                keep = clf == 'mlp' and level in ('cell_pc', 'bag_gene')
                res, summ = score_one(X, y, g, real, level, clf, a.device, keep)
                row['levels'][level][clf] = summ
                np.savez_compressed(scores_dir / f'{name}_{level}_{clf}.npz', score=res.score, y=res.y,
                                    groups=res.groups, folds=res.folds)
                print(f'[{time.time()-t0:7.0f}s] {name:15s} {level:11s} {clf:6s} AUC={summ["auc"]:.4f} '
                      f'CI=[{summ["ci"][0]:.3f},{summ["ci"][1]:.3f}]', flush=True)
                if not keep:
                    continue
                fnames = genes if level == 'bag_gene' else [f'PC{k+1}' for k in range(X.shape[1])]
                att = gene_attribution(real, X, y, g, res, fnames)
                focus = {}
                for key in ('univariate', 'permutation'):
                    for n in plan['focus_features'].get(level, []):
                        if n in fnames:
                            k = fnames.index(n)
                            focus.setdefault(n, {})[key] = float(att[key][k])
                            focus[n][key + '_rank'] = int(att[key + '_rank'][k]) + 1
                rec = dict(univariate_top=att['univariate_top'], permutation_top=att['permutation_top'],
                           focus=focus, n_features=len(fnames))
                np.savez_compressed(scores_dir / f'{name}_{level}_attribution.npz', names=np.array(fnames),
                                    univariate=att['univariate'], permutation=att['permutation'])
                if level == 'bag_gene':
                    Xr = X[y == 0]; Xg = X[y == 1]; gr = g[y == 0]; gg = g[y == 1]
                    rec['offspan_gap'] = offspan_share(real, Xr, Xg, genes, plan['focus_features']['bag_gene'] +
                                                       [d['feature'] for d in att['univariate_top'][:10]])
                    rec['focus_stage_means'] = {
                        n: {STAGES[s]: dict(real=float(Xr[np.isin(gr, held[real.stage[held] == s]), genes.index(n)].mean()),
                                            generated=float(Xg[np.isin(gg, held[real.stage[held] == s]), genes.index(n)].mean()))
                            for s in range(4)} for n in plan['focus_features']['bag_gene'] if n in genes}
                row['attribution'][level] = rec
                del res
        row['seconds'] = time.time() - t0
        write(Path(a.work) / f'generator_{name}_{"-".join(levels)}.json', row)


def phase_null(a):
    torch, E, plan, real, levels = _context(a)
    t0 = time.time()
    gen, digest = _checked_generated(E, plan, plan['primary'], real)
    for level in levels:
        X, y, g = F.build(real, gen, level)
        Xr, gr = X[y == 0], g[y == 0]
        strata = {int(i): int(real.stage[i]) for i in np.unique(gr)}
        folds, _ = folds_for(g, real)
        out = dict(level=level, plan_sha256=sha(a.plan), primary_sha256=digest, classifiers={})
        for clf in plan['classifiers']:
            cfg = mlp_cfg(level, a.device, clf)
            aucs = C.halves_null(Xr, gr, strata, n_splits=plan['null_splits'], seed=FOLD_SEED, classifier='mlp', cfg=cfg)
            perm = C.permutation_null(X, y, g, folds, n_perm=plan['permutations'], seed=FOLD_SEED, classifier='mlp', cfg=cfg)
            out['classifiers'][clf] = dict(
                halves_aucs=aucs, halves_mean=float(np.mean(aucs)), halves_sd=float(np.std(aucs, ddof=1)),
                halves_q95=float(np.quantile(aucs, .95)),
                calibrated=bool(abs(np.mean(aucs) - .5) <= plan['null_tolerance']),
                permutation_aucs=perm, permutation_mean=float(np.mean(perm)), permutation_max=float(np.max(perm)))
            e = out['classifiers'][clf]
            print(f'[{time.time()-t0:7.0f}s] null {level:11s} {clf:6s} halves mean={e["halves_mean"]:.4f} '
                  f'sd={e["halves_sd"]:.4f} q95={e["halves_q95"]:.4f} perm mean={e["permutation_mean"]:.4f} '
                  f'max={e["permutation_max"]:.4f}', flush=True)
        write(Path(a.work) / f'null_{level}.json', out)


def phase_merge(a):
    plan = json.loads(Path(a.plan).read_text())
    work = Path(a.work); scores = work / 'scores'
    gens = {}
    for path in sorted(work.glob('generator_*.json')):
        row = json.loads(path.read_text())
        g = gens.setdefault(row['generator'], dict(path=row['path'], sha256=row['sha256'], role=row['role'],
                                                   levels={}, attribution={}))
        if row['sha256'] != expected_sha(plan['generators'][row['generator']]) or row['plan_sha256'] != sha(a.plan):
            raise SystemExit(f'{path}: produced under a different plan or archive')
        g['levels'].update(row['levels']); g['attribution'].update(row['attribution'])
    null = {}
    for path in sorted(work.glob('null_*.json')):
        n = json.loads(path.read_text())
        if n['plan_sha256'] != sha(a.plan):
            raise SystemExit(f'{path}: different plan')
        null[n['level']] = n['classifiers']
    primary = plan['primary']; comparisons = {}; decisions = {}
    for level in plan['levels']:
        for clf in plan['classifiers']:
            nul = null.get(level, {}).get(clf)
            for name, g in gens.items():
                s = g['levels'].get(level, {}).get(clf)
                if s is None or nul is None:
                    continue
                thr = max(.5, nul['halves_q95'], nul['permutation_max'])
                s['detection_threshold'] = thr
                s['detected'] = bool(s['ci'][0] > .5 and s['auc'] > thr)
            base_path = scores / f'{primary}_{level}_{clf}.npz'
            if not base_path.exists():
                continue
            b = np.load(base_path)
            for name in gens:
                p = scores / f'{name}_{level}_{clf}.npz'
                if name == primary or not p.exists():
                    continue
                o = np.load(p)
                if not (np.array_equal(o['groups'], b['groups']) and np.array_equal(o['folds'], b['folds'])
                        and np.array_equal(o['y'], b['y'])):
                    raise SystemExit('paired comparison requires identical items/folds')
                d = C.paired_bootstrap_diff(b['score'], b['y'], b['groups'], o['score'], o['y'], o['groups'], N_BOOT, 0)
                d['resolved_lower_than_primary'] = bool(d['ci'][1] < 0)
                d['resolved_higher_than_primary'] = bool(d['ci'][0] > 0)
                comparisons.setdefault(level, {}).setdefault(clf, {})[f'{name}_minus_{primary}'] = d
    for level in plan['levels']:
        nul = null.get(level, {})
        decisions[level] = dict(
            null_calibrated={c: v['calibrated'] for c, v in nul.items()},
            detected={n: {c: g['levels'].get(level, {}).get(c, {}).get('detected') for c in plan['classifiers']}
                      for n, g in gens.items()})
    pparg = gens.get(primary, {}).get('attribution', {}).get('bag_gene', {}).get('focus', {}).get('PPARG')
    blind = dict(
        offspan_detected_mlp=decisions.get('bag_offspan', {}).get('detected', {}).get(primary, {}).get('mlp'),
        pparg_univariate_rank=pparg and pparg.get('univariate_rank'),
        pparg_top1pct=bool(pparg and pparg.get('univariate_rank', 10**9) <= 20))
    out = dict(protocol='analysis/c2st_20261120/PROTOCOL.md', plan_sha256=sha(a.plan),
               settings=dict(cell_mlp=CELL_MLP, bag_mlp=BAG_MLP, linear=LINEAR, n_boot=N_BOOT,
                             fold_seed=FOLD_SEED, n_folds=6),
               generators=gens, null=null, comparisons=comparisons, decisions=decisions, blind_spot_link=blind)
    write(a.out, out)
    table = {lv: {n: {c: (round(g['levels'][lv][c]['auc'], 4) if lv in g['levels'] else None)
                      for c in plan['classifiers']} for n, g in gens.items()} for lv in plan['levels']}
    print(json.dumps(dict(auc=table, blind_spot_link=blind), indent=1))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--r5', default=str(HERE.parent / 'improve_20261109/r5'), help='repository evaluator directory')
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    p.add_argument('--gpu-mem-fraction', type=float, default=.05)
    sub = p.add_subparsers(dest='command', required=True)
    s = sub.add_parser('score'); s.add_argument('--cache', required=True)
    s.add_argument('--generated', nargs='+', required=True, help='NAME=path/to/generated.npz')
    s.add_argument('--levels', nargs='+', default=['cell_pc', 'bag_gene'], choices=F.LEVELS)
    s.add_argument('--classifier', choices=('mlp', 'linear'), default='mlp'); s.add_argument('--out', required=True)
    t = sub.add_parser('study', help='declared study: --phase generators | null | merge')
    t.add_argument('--phase', choices=('generators', 'null', 'merge'), required=True)
    t.add_argument('--plan', default=str(HERE / 'plan.json')); t.add_argument('--cache')
    t.add_argument('--work', required=True, help='directory for per-phase JSON and out-of-fold scores')
    t.add_argument('--out', help='merged results JSON (merge phase)')
    t.add_argument('--levels', nargs='+', choices=F.LEVELS, help='subset of the declared levels (parallel runs)')
    t.add_argument('--generators', nargs='+', help='subset of the declared generators')
    a = p.parse_args(argv)
    if not 0 < a.gpu_mem_fraction <= .08:
        p.error('GPU memory fraction must be in (0, .08] on the shared machine')
    if a.command == 'score':
        cmd_score(a)
    else:
        {'generators': phase_generators, 'null': phase_null, 'merge': phase_merge}[a.phase](a)


if __name__ == '__main__':
    main()
