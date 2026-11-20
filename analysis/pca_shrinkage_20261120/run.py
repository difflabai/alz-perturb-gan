"""PCA-shrinkage versus generator under-dispersion study (CPU, fold 0 held-out donors).

See PROTOCOL.md. Reads frozen caches/archives under /ml2/alz/runs read-only,
writes small JSON next to this file and arrays under
/ml2/alz/runs/research_20261120/pca-shrinkage/.
"""
import os
os.environ.update(OMP_NUM_THREADS='3', OPENBLAS_NUM_THREADS='3', MKL_NUM_THREADS='3',
                  CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
R5 = REPO/'analysis/improve_20261109/r5'
DATA = Path('/ml2/alz')
CACHE = DATA/'runs/improve_20261109/caches/hvg2000/fold0'
G0 = DATA/'runs/microglia_empf/verification/final_generated.npz'
G1 = DATA/'runs/coverage_empf_20261117/quarter_generated.npz'
EMP_SCORE = DATA/'runs/improve_20261109/results_r5/empirical_challenge_20261114/empirical32/fold0/seed0.json'
PILOT_CMP = DATA/'runs/coverage_empf_20261117/comparison.json'
OUT = DATA/'runs/research_20261120/pca-shrinkage'
PINS = {'cache_manifest': (CACHE/'manifest.json', '65d5e2f405ceeeb14ac35f8660bc9c42272ac35c64aeda4f7c3bd9ee21dbd09c'),
        'G0': (G0, '04b8d82c25a5b018cc2b19a3b5065a2edd4bf4e908a568fd65c28736015cde0b'),
        'G1': (G1, '241ce3efd5b0e6fac327b0d37baa364ff888134f85a87c47fea2b08e17c48eed'),
        'empirical32_fold0_seed0': (EMP_SCORE, '8915f758cd1ed08bc868e9ed98f31213c5a49ee982d7b67a3eafdafba3f5e436')}
B = 2000
SEED = 20261120

sys.path.insert(0, str(HERE))
sys.path.insert(1, str(R5))
import shrinkage as S  # noqa: E402
import references as RF  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=1, allow_nan=False)+'\n'); tmp.replace(path)


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def pinned():
    out = {}
    for name, (path, digest) in PINS.items():
        got = sha(path)
        if got != digest:
            raise ValueError(f'input changed: {name}')
        out[name] = dict(path=str(path), sha256=got)
    return out


# ------------------------------------------------------------------ data
def empirical_rng(MX, replicate=0):
    return lambda real, t: np.random.default_rng(MX.stable_seed(
        'empirical-challenge', 20261114, real.panel, real.fold, real.donors[t], replicate))


def build_sources(E, real, held):
    """Held-out-target arrays for G0, G1, E_train, E_train11, E_heldpool."""
    import baselines as BL
    src = {}
    for label, path in (('G0', G0), ('G1', G1)):
        g = E.align_generated(E.load_generated(path), real)
        src[label] = dict(full=g, pcs=g['pcs'][held], mean_log=g['mean_log'][held], sq_log=g['sq_log'][held])
    rng_of = empirical_rng(E.MX)
    original = BL._target_rng
    BL._target_rng = rng_of
    try:
        arrays, info = BL.generate('resample_blind', real, M=32, K=256, kpc=64)
    finally:
        BL._target_rng = original
    full = dict(arrays, targets=real.donors.copy(), is_heldout=real.is_heldout.copy(), stage=real.stage.copy(),
                genes=real.genes.copy())
    src['E_train'] = dict(full=full, pcs=arrays['pcs'][held], mean_log=arrays['mean_log'][held],
                          sq_log=arrays['sq_log'][held], info=info)
    train = np.flatnonzero(~real.is_heldout)
    subset_rng = lambda t: np.random.default_rng(E.MX.stable_seed('pca-shrinkage-train11', 20261120, real.donors[t]))
    pools = {
        'E_train11': lambda t: sorted(subset_rng(t).choice(train, 11, replace=False).tolist()),
        'E_heldpool': lambda t: [i for i in held if i != t],
    }
    for label, pool_of in pools.items():
        log('resampling', label)
        r = RF.resample_pool(real, held, pool_of, rng_of, E.MX)
        src[label] = dict(pcs=r['pcs'], mean_log=r['mean_log'], sq_log=r['sq_log'], pools=r['pools'])
    return src


def real_pc_summaries(E, real, rows, k, transform=None):
    """Frozen observed_summaries restricted to ``rows``; optional per-target cell transform."""
    reps = []
    for r in range(E.DEFAULT_PC_REPEATS):
        per = []
        for i in rows:
            cells = real.pcs[i][E.MX.det_subsample(len(real.pcs[i]), k, E.MX.stable_seed('pc-summary-rep', real.donors[i], r))]
            if transform is not None:
                cells = transform(i, cells)
            per.append(E.MX.pc_bag_summary(cells))
        reps.append(np.stack(per))
    return np.stack(reps)


def real_gene_sd(E, real, rows, K=256, repeats=24):
    """[R, n, G] within-donor SD and [R, n, G] mean of log1p CP10k over K-cell subsamples."""
    sd = np.zeros((repeats, len(rows), len(real.genes))); mean = np.zeros_like(sd)
    for j, i in enumerate(rows):
        L = E.MX.log_cp10k(real.cells[i].toarray().astype(np.float64))
        for r in range(repeats):
            idx = E.MX.det_subsample(len(L), K, E.MX.stable_seed('gene-sd-rep', real.donors[i], r))
            m, q = S.gene_moments(L[idx])
            mean[r, j] = m; sd[r, j] = S.gene_sd_from_moments(m, q)
    return sd, mean


def log_hvg32(counts):
    """bench_core.log_hvg (frozen cache recipe): float32 log1p CP10k of panel counts."""
    X = counts.toarray() if hasattr(counts, 'toarray') else np.asarray(counts)
    X = X.astype(np.float32)
    return np.log1p(X * (1e4 / np.maximum(X.sum(-1, keepdims=True), 1)))


def refit_space(L, seed_cells=20000):
    """Frozen build_cache.fit_space recipe on float32 log_hvg cells L [n, G] (kept cells, cache order):
    mean, sd, W, pc_sd. Subsample WITHOUT replacement, as in the frozen recipe (audit repair)."""
    from sklearn.utils.extmath import randomized_svd
    X = L.astype(np.float32)
    m = X.mean(0); sd = X.std(0) + 1e-6; Z = (X - m) / sd
    n = len(Z)
    sub = np.random.default_rng(0).choice(n, min(seed_cells, n), replace=False)
    _, _, Vt = randomized_svd(Z[sub], n_components=50, n_iter=5, random_state=0)
    W = Vt.T.copy(); ps = np.maximum((Z @ W).std(0), 1e-6)
    return m, sd, W, ps


def shrinkage_factors(E, real):
    """Leave-one-TRAIN-donor-out out-of-sample/in-sample within-donor SD per PC rank."""
    train = np.flatnonzero(~real.is_heldout)
    logs = {i: log_hvg32(real.cells[i]) for i in train}
    # recipe check on all TRAIN donors (cache order) versus the frozen space
    Lall = np.concatenate([logs[i] for i in train])
    m0, sd0, Wall, ps0 = refit_space(Lall)
    overlap = real.W.T @ Wall.astype(np.float64)
    principal = np.linalg.svd(overlap, compute_uv=False)
    per_pc = np.linalg.norm(overlap, axis=1)   # |projection of frozen PC k onto the refit 50-PC span|
    same_pc = np.abs(np.sum(real.W * Wall.astype(np.float64), 0))   # |cos| between frozen PC k and refit PC k
    recipe = dict(mean_max_abs_diff=float(np.abs(m0 - real.mean).max()), sd_max_abs_diff=float(np.abs(sd0 - real.sd).max()),
                  pc_sd_max_rel_diff=float(np.abs(ps0 / real.pc_sd - 1).max()),
                  same_rank_abs_cosine=same_pc.tolist())
    log('recipe check', {k: v for k, v in recipe.items() if k != 'same_rank_abs_cosine'},
        'same-rank |cos| min', float(same_pc.min()), 'span min', float(per_pc.min()))
    out_sd, in_sd = [], []
    for d in train:
        log(f'LDO refit without {real.donors[d]}')
        fit = [i for i in train if i != d]
        m, sd, W, ps = refit_space(np.concatenate([logs[i] for i in fit]))
        proj = lambda L: ((L - m) / sd) @ W / ps
        out_sd.append(proj(logs[d]).std(0))
        in_sd.append(np.mean([proj(logs[i]).std(0) for i in fit], 0))
    out_sd, in_sd = np.array(out_sd), np.array(in_sd)
    f = out_sd.mean(0) / in_sd.mean(0)
    return f, dict(out_of_sample_within_sd=out_sd.mean(0).tolist(), in_sample_within_sd=in_sd.mean(0).tolist(),
                   recipe_check_principal_cosines_min=float(principal.min()),
                   recipe_check_best10_principal_cosines_min=float(principal[:10].min()),
                   recipe_check_frozen_pc_in_refit_span=per_pc.tolist(),
                   recipe_check_full_train_refit=recipe,
                   per_donor_factor_sd=(out_sd/in_sd).std(0).tolist())


# ------------------------------------------------------------------ scoring helpers
def ci_table(table, keys=('coverage', 'below', 'above', 'width', 'interval_score')):
    out = {}
    for block, groups in table.items():
        out[block] = {}
        for group in ('mean', 'sd', 'combined'):
            out[block][group] = {k: S.bootstrap(groups[group][k], B, SEED) for k in keys}
        out[block]['log_sd_ratio'] = S.bootstrap(groups['log_sd_ratio'], B, SEED)
    return out


def paired(table_a, table_b):
    out = {}
    for block in table_a:
        out[block] = {'sd_coverage': S.bootstrap(table_a[block]['sd']['coverage'], B, SEED, table_b[block]['sd']['coverage']),
                      'sd_above': S.bootstrap(table_a[block]['sd']['above'], B, SEED, table_b[block]['sd']['above']),
                      'sd_interval_score': S.bootstrap(table_a[block]['sd']['interval_score'], B, SEED, table_b[block]['sd']['interval_score']),
                      'log_sd_ratio': S.bootstrap(table_a[block]['log_sd_ratio'], B, SEED, table_b[block]['log_sd_ratio'])}
    return out


def view(samples_by_source, y, blocks, eps=0.0):
    tables = {k: S.block_table(v, y, blocks, sd_eps=eps) for k, v in samples_by_source.items()}
    return tables, {k: ci_table(t) for k, t in tables.items()}


def gene_view(src, real_sd, real_mean, decile):
    """Gene-space SD/mean intervals per expression decile."""
    out_tables, out = {}, {}
    blocks = {f'decile{d+1:02d}': np.flatnonzero(decile == d) for d in range(10)}
    blocks['all'] = np.arange(len(decile))
    for label, s in src.items():
        g_sd = S.gene_sd_from_moments(s['mean_log'], s['sq_log'])
        g_mean = s['mean_log'].astype(np.float64)
        lo_sd, hi_sd = S.percentile_bounds(g_sd); lo_m, hi_m = S.percentile_bounds(g_mean)
        ratio = S.log_sd_ratio(g_sd, real_sd, eps=.01)
        tab = {}
        for name, idx in blocks.items():
            tab[name] = {'sd': S.per_target(lo_sd[:, idx], hi_sd[:, idx], real_sd[..., idx]),
                         'mean': S.per_target(lo_m[:, idx], hi_m[:, idx], real_mean[..., idx]),
                         'log_sd_ratio': ratio[:, idx].mean(1)}
        out_tables[label] = tab
        out[label] = {name: {'sd': {k: S.bootstrap(v, B, SEED) for k, v in t['sd'].items()},
                             'mean': {k: S.bootstrap(v, B, SEED) for k, v in t['mean'].items()},
                             'log_sd_ratio': S.bootstrap(t['log_sd_ratio'], B, SEED)} for name, t in tab.items()}
    return out_tables, out


def survival(views, label):
    U = lambda c: max(0.0, .8 - c)
    base = views['a_original'][label]['all']['sd']['coverage']['mean']
    res = {'U_original': U(base)}
    for name, v in views.items():
        cov = v[label]['all']['sd']['coverage']['mean']
        res[name] = dict(sd_coverage=cov, U=U(cov), survival=(U(cov)/U(base) if U(base) > 0 else None),
                         log_sd_ratio=v[label]['all']['log_sd_ratio'],
                         under_dispersion_survives=bool(v[label]['all']['log_sd_ratio']['hi'] < 0))
    return res


# ------------------------------------------------------------------ conformal
def conformal(src_samples, y, real, held):
    out = {}
    stages = real.stage[held]
    donors = real.donors[held]
    eligible = real.is_heldout[held]
    names = ['Not AD', 'Low', 'Intermediate', 'High']
    for label, samples in src_samples.items():
        low, high = S.percentile_bounds(samples)
        sd = slice(50, 100)
        res = {}
        for mode, groups in (('global_loo', None), ('mondrian_stage_loo', stages)):
            lo, hi, info = S.loo_conformal(low[:, sd], high[:, sd], y[0][:, sd], donors, groups, eligible, lower_clip=0.0)
            per = S.per_target(lo, hi, y[..., sd])
            raw = S.per_target(low[:, sd], high[:, sd], y[..., sd])
            Lc, Hc = low.copy(), high.copy(); Lc[:, sd], Hc[:, sd] = lo, hi
            comb = S.per_target(Lc, Hc, y)
            by_stage = {}
            for s in sorted(set(stages.tolist())):
                mask = stages == s
                by_stage[names[s]] = dict(n_donors=int(mask.sum()), n_cal=sorted(set(info['n_cal'][mask].tolist())),
                                          rank=sorted(set(info['rank'][mask].tolist())),
                                          calibrated=S.finite_summary(per, mask), raw=S.summarize_targets(raw, mask))
            res[mode] = dict(overall=S.finite_summary(per), combined=S.finite_summary(comb),
                             raw_overall=S.summarize_targets(raw),
                             overall_coverage_ci=S.bootstrap(per['coverage'], B, SEED),
                             fraction_infinite_targets=float((~np.isfinite(per['width'])).mean()),
                             by_stage=by_stage)
        res['_per_target'] = {mode: dict(sd_interval_score=S.per_target(*S.loo_conformal(
            low[:, sd], high[:, sd], y[0][:, sd], donors, None if mode == 'global_loo' else stages, eligible,
            lower_clip=0.0)[:2], y[..., sd])['interval_score']) for mode in ('global_loo',)}
        res['_per_target']['raw'] = dict(sd_interval_score=S.per_target(low[:, sd], high[:, sd], y[..., sd])['interval_score'])
        out[label] = res
    # Exploratory (not declared in PROTOCOL.md): paired donor-bootstrap of SD interval scores.
    pt = {k: v.pop('_per_target') for k, v in out.items()}
    out['exploratory_paired_sd_interval_score'] = {
        f'{a}_{ma}_minus_{b}_{mb}': S.bootstrap(pt[a][ma]['sd_interval_score'], B, SEED, pt[b][mb]['sd_interval_score'])
        for a, ma, b, mb in (('G0', 'global_loo', 'E_train', 'raw'), ('G0', 'global_loo', 'E_train', 'global_loo'),
                             ('G1', 'global_loo', 'E_train', 'raw'), ('G1', 'global_loo', 'G0', 'global_loo'),
                             ('G0', 'global_loo', 'G0', 'raw'))}
    return out


# ------------------------------------------------------------------ main
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--skip-historical', action='store_true')
    args = p.parse_args(argv)
    os.sched_setaffinity(0, set(range(3, 24)) & os.sched_getaffinity(0) or os.sched_getaffinity(0))
    t0 = time.time()
    pins = pinned()
    import evaluate as E
    import architecture_score as A
    import interval_metrics as I
    log('loading real fold')
    real = E.load_real(CACHE)
    held = np.flatnonzero(real.is_heldout)
    assert len(held) == 12
    y_all = A.observed_summaries(real, 64)          # [24, 37, 100]
    y = y_all[:, held]
    src = build_sources(E, real, held)
    samples = {k: S.bag_summary(v['pcs'].astype(np.float32)) for k, v in src.items()}

    checks = {}
    # identity checks against frozen scores
    emp = json.loads(EMP_SCORE.read_text())['intervals']['heldout']['percentile']
    full_emp = E.MX.pc_bag_summary(src['E_train']['full']['pcs'].astype(np.float32))
    got = I.summarize(full_emp, y_all, real.is_heldout, 'percentile')
    checks['E_train_reproduces_empirical32_seed0'] = all(got[g][k] == emp[g][k] for g in emp for k in emp[g])
    pilot = json.loads(PILOT_CMP.read_text())['rows']
    for label, key in (('G0', 'baseline'), ('G1', 'quarter')):
        full = E.MX.pc_bag_summary(src[label]['full']['pcs'].astype(np.float32))
        got = I.summarize(full, y_all, real.is_heldout, 'percentile')
        ref = pilot[key]['raw_heldout']
        checks[f'{label}_reproduces_pilot_raw_heldout'] = all(abs(got[g][k] - ref[g][k]) < 1e-12 for g in ref for k in ref[g])
    tab = S.block_table(samples['E_train'], y, {'all': slice(0, 50)})['all']
    checks['library_matches_frozen_summary_E_train'] = abs(float(tab['sd']['coverage'].mean()) - emp['sd']['coverage']) < 1e-12
    log('checks', checks)
    if not all(checks.values()):
        raise SystemExit(f'identity checks failed: {checks}')

    historical = {}
    if not args.skip_historical:
        for label in ('G0', 'G1', 'E_train'):
            log('historical', label)
            res, _ = E.score_generated(src[label]['full'], real, {'all': (np.arange(len(real.genes)), False)})
            historical[label] = {k: v for k, v in E.headline(res).items()
                                 if k in ('pb_rmse', 'meanlog_rmse_gene', 'nonzero_rmse_gene', 'logvar_rmse_gene',
                                          'pc_energy_rep', 'cell_energy_rep')}
        stored = {'G0': pilot['baseline']['historical'], 'G1': pilot['quarter']['historical'],
                  'E_train': json.loads(EMP_SCORE.read_text())['historical']}
        checks['historical_match_stored'] = {k: all(abs(historical[k][m] - stored[k][m]) <= 1e-9 * max(1, abs(stored[k][m]))
                                                    for m in historical[k]) for k in historical}

    views, tables = {}, {}
    # (a) original basis
    log('view a'); tables['a_original'], views['a_original'] = view(samples, y, S.PC_BLOCKS)
    # (b1) TRAIN-estimated shrinkage correction
    log('shrinkage factors'); f, finfo = shrinkage_factors(E, real)
    corrected = {k: S.shrinkage_scale_sd(v, f) for k, v in samples.items()}
    log('view b1'); tables['b1_train_shrinkage_corrected'], views['b1_train_shrinkage_corrected'] = view(corrected, y, S.PC_BLOCKS)
    # (b2) out-of-sample within-subspace whitening per target
    log('view b2')
    rot = {}
    for i in held:
        others = [j for j in held if j != i]
        rot[i] = S.fit_whitening(np.concatenate([real.pcs[j] for j in others]))
    y_b2 = real_pc_summaries(E, real, held, 64, transform=lambda i, c: S.apply_whitening(c, *rot[i]))
    samples_b2 = {k: np.stack([S.bag_summary(S.apply_whitening(v['pcs'][j].astype(np.float32), *rot[i]))
                               for j, i in enumerate(held)]) for k, v in src.items()}
    tables['b2_heldout_loo_whitened'], views['b2_heldout_loo_whitened'] = view(samples_b2, y_b2, S.PC_BLOCKS)
    # (c) gene space
    log('view c')
    real_sd, real_mean = real_gene_sd(E, real, held)
    decile, edges = S.decile_groups(real.mean)
    tables['c_gene'], views['c_gene'] = gene_view(src, real_sd, real_mean, decile)

    readouts = {
        'projection_effect_model_free': {name: paired(tables[name]['E_train'], tables[name]['E_heldpool'])
                                         for name in ('a_original', 'b1_train_shrinkage_corrected')},
        'pool_size_control': paired(tables['a_original']['E_train'], tables['a_original']['E_train11']),
        'generator_vs_oracle': {name: {g: paired(tables[name][g], tables[name]['E_heldpool']) for g in ('G0', 'G1')}
                                for name in ('a_original', 'b1_train_shrinkage_corrected')},
        'generator_vs_empirical': {name: {g: paired(tables[name][g], tables[name]['E_train']) for g in ('G0', 'G1')}
                                   for name in tables if name != 'c_gene'},
        'survival': {g: survival(views, g) for g in ('G0', 'G1', 'E_train', 'E_heldpool')},
    }
    log('conformal')
    conf = conformal({k: samples[k] for k in ('G0', 'G1', 'E_train')}, y, real, held)

    np.savez_compressed(OUT/'arrays.npz', shrinkage_factors=f, real_gene_sd=real_sd.astype(np.float32),
                        real_gene_mean=real_mean.astype(np.float32), decile=decile, decile_edges=edges,
                        y_b2=y_b2.astype(np.float32), heldout_donors=real.donors[held],
                        **{f'samples_{k}': v.astype(np.float32) for k, v in samples.items()},
                        **{f'samples_b2_{k}': v.astype(np.float32) for k, v in samples_b2.items()},
                        **{f'{k}_mean_log': v['mean_log'] for k, v in src.items() if k.startswith('E_')},
                        **{f'{k}_sq_log': v['sq_log'] for k, v in src.items() if k.startswith('E_')})
    result = dict(
        status='completed', protocol_sha256=sha(HERE/'PROTOCOL.md'),
        code_sha256={n: sha(HERE/n) for n in ('run.py', 'shrinkage.py', 'references.py')},
        inputs=pins, checks=checks, historical=historical,
        heldout=dict(donors=real.donors[held].tolist(), stage=real.stage[held].tolist()),
        pools={k: src[k]['pools'] for k in ('E_train11', 'E_heldpool')},
        shrinkage_factors=dict(f=f.tolist(), **finfo),
        gene_decile_edges=edges.tolist(), views=views, readouts=readouts, conformal=conf,
        bootstrap=dict(B=B, seed=SEED, unit='held-out donor (n=12)'),
        arrays=dict(path=str(OUT/'arrays.npz')), wall_seconds=time.time()-t0)
    write(HERE/'results.json', result)
    log('done', time.time()-t0)


if __name__ == '__main__':
    main()
