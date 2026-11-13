"""Round-5 work item 2 (1): diagnosis of the count-shape gap (meanlog_rmse_gene, nonzero_rmse_gene).

Owner: count-shape task.  Reads round-3 fits (read-only) and the hvg2000 caches of folds 0-2; writes
only under runs/improve_20261109/smoke_r5/shape/.  No calibration here is scored as a model: every
quantity is a DIAGNOSTIC (held-out donors enter only where the question is about held-out error, and
then only as the evaluator uses them).

Subcommands (run from /tmp, python3 -B, 3 threads)::

    shape_diagnose.py reproduce             score one existing results_r3 file with the r5 evaluator
    shape_diagnose.py real   --fold K       real-cell side: depth deciles, per-gene NB maximum likelihood
                                            on REAL train cells, floors, resampling decomposition
    shape_diagnose.py gen    --fold K --arm A --seed S --readout R [--device cuda]
                                            generator side: regenerate the read-out with the fit's common
                                            random numbers (bitwise check against the stored file), the
                                            mean-matched counterfactuals, depth-decile shape, theta
    shape_diagnose.py summary               tables over everything written

Definitions
-----------
Per-gene errors are the evaluator's: e_g = mean over the donor set of (generated value averaged over M
- real value), metric = sqrt(mean_g e_g^2), for x = mean per-cell log1p(panel CP10k) ("meanlog") and
x = fraction of cells with count > 0 ("nonzero").

Mean part / shape part.  The generated cells are re-sampled with the SAME random numbers (latent, cell
noise, library draws, Gamma and Poisson streams: the per-chunk global and count RNG states are saved
and restored for each variant) after multiplying the generator's NB mean mu of gene g by
r_g = (mean over the donor set of the real pseudobulk share of g) / (mean over the set's targets of the
generated pseudobulk share of g).  The generated pseudobulk proportion then equals the real one gene
by gene ("mean matched"; depth is matched by construction: libraries are drawn from the target's own
cells), so the remaining per-gene error e^matched is the SHAPE part (dispersion, zeros, cell-to-cell
heterogeneity, plus donor-to-donor variation of shape), and e - e^matched is the MEAN part.  Done for
the held-out set (r from held-out donors; the held-out question) and for the train set (r from train
donors; in-sample, no donor-sampling noise of the mean).

Matched-depth shape.  Library-size deciles = deciles of the pooled REAL train cells' panel library
sizes.  Generated cells are assigned by their drawn library (a real library of the target's own
cells).  Per decile and gene (train set, mean-matched variant vs real train cells): nonzero fraction,
mean count, phi = (var - mean)/mean^2 (excess dispersion over Poisson), summarised by expression
quartile (quartiles of the train real mean_log, the evaluator's strata).

Best NB on real cells.  Per gene, theta_ML maximises the NB log likelihood of the REAL train cells
with mu_c = L_c q_dg (L_c = cell's panel library, q_dg = donor d's pseudobulk share: within-donor
NB) - "nb_donor"; and with q_g common to all donors - "nb_pop".  The fitted NB then predicts each
train donor's nonzero fraction (exact) and mean-log (4 simulated replicates of every real cell at its
real depth); the in-sample per-gene errors say whether the NB family can reproduce the two
statistics at all.  The same within-group ML is run on 2 x 256 generated cells per train target
(group = generated donor): theta_eff of the generator's counts, compared with theta_ML(real) and
with the generator's own theta parameter (softplus(log_theta)+.05, geometric mean over generated
donors and states).

Floors.  meanlog: ``results_r3/floor.json`` (A_raw = direct train-mean predictor, A_grand = de-biased
oracle).  nonzero: the same route A computed here.  Resampling (M = 32) and the ideal stage-blind
generator (resample_blind) are regenerated with r5/baselines.py (bitwise their results_r3 values,
checked) and their held-out error is split into (prediction - train-donor mean of the real value)
[read-out / thinning part] and (train mean - held-out mean) [donor-to-donor part].

Round-5 repair (adversarial review F2, F5, F7), reporting only (no score or calibration changes):
* F2 (zreport): resampling is NOT a floor for meanlog / nonzero.  The direct train-mean predictor (route A
  raw: predict each held-out donor's per-gene statistic by the train-donor mean; the same train statistic
  zcal calibrates to) beats both resampling and zcal on both metrics.  zreport now carries it as reference
  ``trainmean`` (per fold, and model minus trainmean paired by fold and seed) next to resampling, and a
  ``headline`` block that states the comparison in that order: zcal vs biascal, vs resampling, vs
  trainmean, and the *_recal (train-only additive per-gene recalibration, applied alike to models and
  resampling) gap that zcal does not change.  The train-mean predictor has no recal entry: its per-gene
  recalibration residual is zero on train, so its recal value equals its raw value.
* F5 (summary): ``nb_donor`` fits per-donor means in-sample on the same 25 train donors it is scored on: an
  in-sample UPPER bound on what an NB family can do, not a like-for-like comparator.  The comparator for a
  generator is ``nb_pop`` (one population mean per gene, in-sample too).
* F7 (summary): the shape share of the per-gene MSE is quoted per fit (min-max over the seed-0 fits) for the
  train and held-out sets, and the held-out shape residual is printed next to the train-mean predictor's
  error (= the donor-to-donor part), because the held-out shape residual includes donor-to-donor deviation
  of shape and is about the same size.
"""
from __future__ import annotations

import os
import sys

if __name__ == '__main__':
    for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[_k] = '3'
    _here = os.path.dirname(os.path.abspath(__file__))
    if _here in sys.path:
        sys.path.remove(_here)
    sys.path.insert(0, _here)

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy.special import gammaln  # noqa: E402

RUNS = Path('/ml2/alz/runs/improve_20261109')
OUT = RUNS / 'smoke_r5' / 'shape'
CACHES = RUNS / 'caches'
PANEL = 'hvg2000'
ARMS = {'ctrl': 'pbulk', 'combo': 'g.dispz,pbulk,t.kappa@0.25'}
N_DEC = 10
THETA_MULTS = [0.5, 1.0, 2.0, np.inf]


def jdump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1, allow_nan=True, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)))
    tmp.replace(path)


def rmse(e):
    e = np.asarray(e, dtype=np.float64)
    return float(np.sqrt(np.mean(e ** 2)))


def quartiles(ref):
    q = np.quantile(ref, [.25, .5, .75])
    return np.digitize(ref, q)


def by_quartile(e, qb):
    return {f'Q{b + 1}': {'rmse': rmse(e[qb == b]), 'bias': float(np.mean(e[qb == b]))} for b in range(4)}


# =========================================================================== real side
def real_decile_stats(real, edges, mask):
    """Per decile [10,G]: n, sum nz, sum x, sum x^2, sum log1p cp10k over the donors in mask (all their cells)."""
    import metrics as MX
    G = len(real.genes)
    acc = {k: np.zeros((N_DEC, G)) for k in ('nz', 'x', 'x2', 'lg')}
    n = np.zeros(N_DEC)
    for i in np.where(mask)[0]:
        X = real.cells[i].toarray().astype(np.float64)
        L = X.sum(1)
        dec = np.digitize(L, edges)
        lg = MX.log_cp10k(X)
        for b in range(N_DEC):
            r = dec == b
            if not r.any():
                continue
            n[b] += r.sum()
            acc['nz'][b] += (X[r] > 0).sum(0); acc['x'][b] += X[r].sum(0)
            acc['x2'][b] += (X[r] ** 2).sum(0); acc['lg'][b] += lg[r].sum(0)
    return n, acc


def nb_loglik_theta(X, mu, theta):
    """Per-gene NB log likelihood terms that depend on theta, summed over cells. X, mu [N,G]; theta [G]."""
    t = theta[None, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        ll = gammaln(X + t) - gammaln(t) + t * np.log(t) - (X + t) * np.log(t + mu)
    return ll.sum(0)


def nb_ml_theta(X, L, group, lo=1e-3, hi=1e5, iters=48):
    """theta_ML per gene for counts X [N,G] with mu = L q_group (q = group pseudobulk share). Golden section on log theta."""
    ng = int(group.max()) + 1
    pb = np.zeros((ng, X.shape[1]))
    np.add.at(pb, group, X)
    q = pb / np.maximum(pb.sum(1, keepdims=True), 1)
    mu = L[:, None] * q[group]
    a = np.full(X.shape[1], np.log(lo)); b = np.full(X.shape[1], np.log(hi))
    gr = (np.sqrt(5) - 1) / 2
    c = b - gr * (b - a); d = a + gr * (b - a)
    fc = nb_loglik_theta(X, mu, np.exp(c)); fd = nb_loglik_theta(X, mu, np.exp(d))
    for _ in range(iters):
        left = fc > fd            # maximum in [a, d]
        b = np.where(left, d, b); a = np.where(left, a, c)
        c2 = b - gr * (b - a); d2 = a + gr * (b - a)
        # reuse one evaluation where possible (vectorised: just recompute both)
        c, d = c2, d2
        fc = nb_loglik_theta(X, mu, np.exp(c)); fd = nb_loglik_theta(X, mu, np.exp(d))
    th = np.exp(0.5 * (a + b))
    return th, mu


def nb_predict(mu, theta, group, n_rep=4, seed=0):
    """Per group: exact expected nonzero fraction and simulated mean log1p cp10k (n_rep replicates) under NB(mu, theta)."""
    import metrics as MX
    ng = int(group.max()) + 1
    t = theta[None, :]
    p0 = np.exp(-t * np.log1p(mu / t))
    nz = np.zeros((ng, mu.shape[1])); ml = np.zeros_like(nz)
    cnt = np.bincount(group, minlength=ng).astype(np.float64)
    np.add.at(nz, group, 1 - p0)
    rng = np.random.default_rng(seed)
    for _ in range(n_rep):
        lam = rng.gamma(np.broadcast_to(t, mu.shape), np.maximum(mu, 1e-300) / t)
        x = rng.poisson(lam).astype(np.float64)
        np.add.at(ml, group, MX.log_cp10k(x))
    return nz / cnt[:, None], ml / (cnt[:, None] * n_rep)


def route_a(Xt, Xh):
    """Floor route A for a per-donor statistic: raw train-mean predictor and the de-biased oracle."""
    m = Xt.mean(0); v = Xt.var(0, ddof=1); n = len(Xt)
    e2 = ((Xh.mean(0) - m) ** 2).mean()
    return {'A_raw_train_mean': float(np.sqrt(e2)), 'A_grand_oracle': float(np.sqrt(max(e2 - v.mean() / n, 0.0))),
            'B_analytic_sqrt_var_over_nheld': float(np.sqrt(Xt.var(0, ddof=1).mean() / len(Xh)))}


def cmd_real(a):
    import evaluate as EV
    import baselines as BL
    k = a.fold
    real = EV.load_real(CACHES / PANEL / f'fold{k}', PANEL, k)
    tr, he = ~real.is_heldout, real.is_heldout
    qb = quartiles(real.mean_log[tr].mean(0))
    out = {'fold': k, 'n_train': int(tr.sum()), 'n_heldout': int(he.sum())}
    # ---- depth deciles of pooled real train libraries
    Ltr = np.concatenate([real.libs[i] for i in np.where(tr)[0]])
    edges = np.quantile(Ltr, np.arange(1, N_DEC) / N_DEC)
    out['decile_edges'] = edges
    dec = {}
    for name, mask in (('train', tr), ('heldout', he)):
        n, acc = real_decile_stats(real, edges, mask)
        dec[name] = (n, acc)
    np.savez_compressed(OUT / f'real_fold{k}_deciles.npz', edges=edges, qb=qb,
                        **{f'{s}_{key}': v for s, (n, acc) in dec.items() for key, v in list(acc.items()) + [('n', n)]})
    # ---- floors (route A) for meanlog and nonzero
    out['floor_route_A'] = {'meanlog': route_a(real.mean_log[tr], real.mean_log[he]),
                            'nonzero': route_a(real.nonzero[tr], real.nonzero[he])}
    fl = json.loads((RUNS / 'results_r3' / 'floor.json').read_text())['per_fold'][str(k)]
    out['floor_json_meanlog'] = {'A_raw_train_mean': fl['meanlog']['A_raw_train_mean']['value'],
                                 'A_grand': fl['meanlog']['A_grand']['value'], 'D_ideal_generator': fl['meanlog']['D_ideal_generator']}
    # ---- per-gene NB maximum likelihood on REAL train cells
    t0 = time.monotonic()
    rows = np.where(tr)[0]
    X = np.concatenate([real.cells[i].toarray().astype(np.float64) for i in rows])
    L = X.sum(1)
    grp = np.concatenate([np.full(real.cells[i].shape[0], j) for j, i in enumerate(rows)])
    th_d, mu_d = nb_ml_theta(X, L, grp)
    th_p, mu_p = nb_ml_theta(X, L, np.zeros(len(L), dtype=int))
    nbres = {}
    for name, th, mu in (('nb_donor', th_d, mu_d), ('nb_pop', th_p, mu_p)):
        nz, ml = nb_predict(mu, th, grp, seed=k)
        e_nz = (nz - real.nonzero[tr]).mean(0)
        e_ml = (ml - real.mean_log[tr]).mean(0)
        nbres[name] = {'nonzero_rmse_gene_train_insample': rmse(e_nz), 'nonzero_bias': float(e_nz.mean()),
                       'meanlog_rmse_gene_train_insample': rmse(e_ml), 'meanlog_bias': float(e_ml.mean()),
                       'nonzero_by_quartile': by_quartile(e_nz, qb), 'meanlog_by_quartile': by_quartile(e_ml, qb),
                       'theta_median': float(np.median(th)), 'theta_at_upper_bound': int((th > 0.99e5).sum()),
                       'theta_median_by_quartile': {f'Q{b + 1}': float(np.median(th[qb == b])) for b in range(4)},
                       'n_genes_nb_overpredicts_nonzero_by_0.01': int((e_nz > 0.01).sum()),
                       'n_genes_nb_underpredicts_nonzero_by_0.01': int((e_nz < -0.01).sum())}
        np.save(OUT / f'real_fold{k}_{name}_theta.npy', th)
        np.save(OUT / f'real_fold{k}_{name}_enz.npy', e_nz)
        np.save(OUT / f'real_fold{k}_{name}_eml.npy', e_ml)
    out['nb_ml_real_train'] = nbres
    out['nb_seconds'] = time.monotonic() - t0
    del X, mu_d, mu_p
    # ---- resampling references, regenerated (checked against results_r3)
    refs = {}
    for name in ('resample', 'resample_blind'):
        t0 = time.monotonic()
        gen, _info = BL.generate(name, real)
        gen = EV.align_generated(gen, real)
        gml = gen['mean_log'].astype(np.float64).mean(1); gnz = gen['nonzero'].astype(np.float64).mean(1)
        r = {}
        for stat, gv, rv in (('meanlog', gml, real.mean_log), ('nonzero', gnz, real.nonzero)):
            e = (gv[he] - rv[he]).mean(0)
            readout = gv[he].mean(0) - rv[tr].mean(0)        # prediction minus train-donor mean of the real value
            donor = rv[tr].mean(0) - rv[he].mean(0)          # train mean minus held-out mean (donor-to-donor)
            etr = (gv[tr] - rv[tr]).mean(0)
            r[stat] = {'heldout_rmse_gene': rmse(e), 'readout_part_rmse': rmse(readout), 'donor_part_rmse': rmse(donor),
                       'cross_term_share': float(2 * np.mean(readout * donor) / max(np.mean(e ** 2), 1e-30)),
                       'train_rmse_gene': rmse(etr), 'heldout_bias': float(e.mean()), 'readout_bias': float(readout.mean()),
                       'heldout_by_quartile': by_quartile(e, qb), 'readout_by_quartile': by_quartile(readout, qb)}
        js = json.loads((RUNS / 'results_r3' / 'baselines' / PANEL / f'fold{k}' / f'{name}.json').read_text())
        h = js['subsets']['all']['heldout']
        r['results_r3'] = {'meanlog_rmse_gene': h['mean_log']['rmse_gene'], 'nonzero_rmse_gene': h['nonzero']['rmse_gene'],
                           'meanlog_rmse_gene_recal': h['mean_log_recal']['rmse_gene'], 'nonzero_rmse_gene_recal': h['nonzero_recal']['rmse_gene'],
                           'pb_rmse': h['pseudobulk']['rmse'], 'logvar_rmse_gene': (h.get('second_moment') or {}).get('rmse_gene')}
        r['reproduced'] = bool(abs(r['meanlog']['heldout_rmse_gene'] - r['results_r3']['meanlog_rmse_gene']) < 1e-9
                               and abs(r['nonzero']['heldout_rmse_gene'] - r['results_r3']['nonzero_rmse_gene']) < 1e-9)
        r['seconds'] = time.monotonic() - t0
        refs[name] = r
        np.save(OUT / f'real_fold{k}_{name}_eml_heldout.npy', (gml[he] - real.mean_log[he]).mean(0))
        np.save(OUT / f'real_fold{k}_{name}_enz_heldout.npy', (gnz[he] - real.nonzero[he]).mean(0))
    out['references'] = refs
    np.save(OUT / f'real_fold{k}_trainmean_minus_heldout_ml.npy', real.mean_log[tr].mean(0) - real.mean_log[he].mean(0))
    np.save(OUT / f'real_fold{k}_trainmean_minus_heldout_nz.npy', real.nonzero[tr].mean(0) - real.nonzero[he].mean(0))
    jdump(OUT / f'real_fold{k}.json', out)
    print(json.dumps({'fold': k, 'floor': out['floor_route_A'], 'nb': {n: {kk: v for kk, v in d.items() if not isinstance(v, dict)}
                                                                      for n, d in nbres.items()},
                      'refs': {n: {s: {kk: v for kk, v in d[s].items() if not isinstance(v, dict)} for s in ('meanlog', 'nonzero')}
                               for n, d in refs.items()}}, indent=1))


# =========================================================================== generator side
def load_readout(fit, cache, readout, device):
    """(ctx, models, plans, T, used states, mode record) for a fit read out with ``readout`` (generate_fit's steps)."""
    import torch
    import trainer
    import generate as GEN
    import arms
    import mode_stagedefault  # noqa: F401
    import mode_biascal  # noqa: F401
    import mode_zcal  # noqa: F401
    cfg = json.loads((Path(fit) / 'config.json').read_text())
    ctx = trainer.Context(cache, device, tuple(cfg['flags']), allow_nonprotocol_split=not cfg['protocol_split'])
    T = GEN.targets(ctx)
    states = GEN.available_states(fit, cfg)
    plans, spec, value = GEN.mode_plans(ctx, readout, T, states, updates=cfg.get('updates'))
    used = [s for s in states if any(s in names for _, names in plans)]
    models = GEN.load_models(ctx, fit, cfg, used)
    if spec.prepare is not None:
        with torch.no_grad():
            for name, (G, P) in models.items():
                spec.prepare(ctx, value, name, G, P)
    return ctx, models, plans, T, used


def _rng_state(device):
    import torch
    return torch.cuda.get_rng_state(device) if device.type == 'cuda' else torch.get_rng_state()


def _set_rng_state(device, st):
    import torch
    if device.type == 'cuda':
        torch.cuda.set_rng_state(st, device)
    else:
        torch.set_rng_state(st)


def simulate(ctx, models, plans, T, variants, edges, keep_cells_targets=(), keep_m=2, analytic_variant=None):
    """Replay generation with common random numbers for several variants of the NB mean.

    variants: ordered dict name -> per-gene mean factor (np [G]) or None; 'plain' (None) must be last.
    Returns per variant: mean_log, nonzero, pb [T,G] (averaged / summed over M and K); per-decile sums
    [T,10,G] (n, nz, x, x2, lg); analytic expected nonzero at theta multipliers (analytic_variant);
    the theta log-mean over generated donors per target; kept plain cells of keep_cells_targets."""
    import torch
    with torch.no_grad():
        return _simulate(ctx, models, plans, T, variants, edges, keep_cells_targets, keep_m, analytic_variant)


def _simulate(ctx, models, plans, T, variants, edges, keep_cells_targets, keep_m, analytic_variant):
    import torch
    import mode_zcal as MZ
    from generate import M, K
    dev = ctx.device
    names = list(variants)
    assert names[-1] == 'plain' and variants['plain'] is None
    Tn, Gn = len(T), ctx.n_genes
    res = {v: {'mean_log': np.zeros((Tn, Gn)), 'nonzero': np.zeros((Tn, Gn)), 'pb': np.zeros((Tn, Gn)),
               'dec_n': np.zeros((Tn, N_DEC)), 'dec_nz': np.zeros((Tn, N_DEC, Gn)), 'dec_x': np.zeros((Tn, N_DEC, Gn)),
               'dec_x2': np.zeros((Tn, N_DEC, Gn)), 'dec_lg': np.zeros((Tn, N_DEC, Gn))} for v in names}
    an = np.zeros((Tn, len(THETA_MULTS), Gn))
    logth = np.zeros((Tn, Gn)); nth = np.zeros(Tn)
    kept = []
    e_t = torch.as_tensor(edges, dtype=torch.float32, device=dev)
    fac = {v: (None if f is None else torch.as_tensor(f, dtype=torch.float32, device=dev)) for v, f in variants.items()}
    for ti, start, n, G, z, c, lib, cell, count in MZ.iterate_chunks(ctx, models, plans, T):
        mu, theta = G.parameters_nb(z, c, K, lib, cell)
        th_full = theta.expand_as(mu) if theta.dim() == 3 else theta
        lt = torch.log(theta).reshape(-1, Gn) if theta.dim() == 3 else torch.log(theta)[None].expand(n, Gn)
        logth[ti] += lt.double().sum(0).cpu().numpy()
        nth[ti] += n
        dec = torch.bucketize(lib, e_t, right=True).reshape(-1)          # [n*K] in 0..9 (np.digitize semantics)
        oh = torch.nn.functional.one_hot(dec, N_DEC).float()             # [n*K,10]
        g_state = _rng_state(dev); c_state = count.get_state()
        for v in names:
            _set_rng_state(dev, g_state); count.set_state(c_state)
            mv = mu if fac[v] is None else mu * fac[v]
            rate = torch.distributions.Gamma(theta, theta / mv).rsample() * (lib > 0)[..., None]
            counts = torch.poisson(rate, generator=count)
            tot = counts.sum(-1)
            logcp = torch.log1p(counts * (1e4 / tot.clamp_min(1))[..., None])
            r = res[v]
            r['mean_log'][ti] += logcp.mean(1).sum(0).double().cpu().numpy()
            r['nonzero'][ti] += (counts > 0).float().mean(1).sum(0).double().cpu().numpy()
            r['pb'][ti] += counts.sum((0, 1)).double().cpu().numpy()
            C = counts.reshape(-1, Gn)
            r['dec_n'][ti] += oh.sum(0).double().cpu().numpy()
            r['dec_nz'][ti] += (oh.T @ (C > 0).float()).double().cpu().numpy()
            r['dec_x'][ti] += (oh.T @ C).double().cpu().numpy()
            r['dec_x2'][ti] += (oh.T @ C.square()).double().cpu().numpy()
            r['dec_lg'][ti] += (oh.T @ logcp.reshape(-1, Gn)).double().cpu().numpy()
            if v == analytic_variant:
                for j, mlt in enumerate(THETA_MULTS):
                    if np.isinf(mlt):
                        p = -torch.expm1(-mv)
                    else:
                        p, _ = MZ.expected_nonzero(mv, th_full * mlt)
                    an[ti, j] += p.mean(1).sum(0).double().cpu().numpy()
            if v == 'plain' and ti in keep_cells_targets and start == 0:
                kc = counts[:keep_m].reshape(-1, Gn).cpu().numpy().astype(np.float32)
                kept.append((ti, kc, np.repeat(np.arange(keep_m), K) + ti * keep_m))
    for v in names:
        for key in ('mean_log', 'nonzero'):
            res[v][key] /= M
    an /= M
    return res, an, logth / nth[:, None], kept


def cmd_gen(a):
    import torch
    import evaluate as EV
    import generate as GEN
    torch.set_num_threads(3)
    if a.device == 'cuda' and a.gpu_mem_fraction:
        torch.cuda.set_per_process_memory_fraction(a.gpu_mem_fraction)
    k, arm, seed, ro = a.fold, ARMS[a.arm], a.seed, a.readout
    fit = RUNS / 'fits_r3' / arm / PANEL / f'fold{k}' / f'seed{seed}'
    cache = CACHES / PANEL / f'fold{k}'
    t0 = time.monotonic()
    real = EV.load_real(cache, PANEL, k)
    ctx, models, plans, T, used = load_readout(fit, cache, ro, a.device)
    order = [list(real.donors).index(t[0]) for t in T]            # T order -> real index
    held = real.is_heldout[order]; tr = ~held
    rn = {'mean_log': real.mean_log[order], 'nonzero': real.nonzero[order], 'pb': real.pb[order]}
    qb = quartiles(rn['mean_log'][tr].mean(0))
    rd = np.load(OUT / f'real_fold{k}_deciles.npz')
    edges = rd['edges']
    # pass 1: plain
    res1, _, logth, kept = simulate(ctx, models, plans, T, {'plain': None}, edges,
                                    keep_cells_targets=set(np.where(tr)[0].tolist()))
    p = res1['plain']
    stored = RUNS / 'fits_r3' / arm / PANEL / f'fold{k}' / f'seed{seed}' / f'generated_{ro}.npz'
    chk = {}
    if stored.exists():
        with np.load(stored) as f:
            sm = f['mean_log'].astype(np.float64).mean(1); sn = f['nonzero'].astype(np.float64).mean(1)
            spb = f['pb_counts'].astype(np.float64).sum(1)
        chk = {'stored_file': str(stored), 'max_abs_diff_mean_log': float(np.abs(sm - p['mean_log']).max()),
               'max_abs_diff_nonzero': float(np.abs(sn - p['nonzero']).max()), 'max_abs_diff_pb': float(np.abs(spb - p['pb']).max())}
        print('[check] replay vs stored file', chk, flush=True)

    def share(pb):
        return pb / np.maximum(pb.sum(1, keepdims=True), 1)
    gsh, rsh = share(p['pb']), share(rn['pb'])
    r_held = rsh[held].mean(0) / np.maximum(gsh[held].mean(0), 1e-12)
    r_train = rsh[tr].mean(0) / np.maximum(gsh[tr].mean(0), 1e-12)
    # pass 2: mean-matched variants (+ plain again, last)
    res2, an, _, _ = simulate(ctx, models, plans, T, {'match_heldout': r_held, 'match_train': r_train, 'plain': None}, edges,
                              analytic_variant='match_train')
    out = {'fold': k, 'arm': arm, 'seed': seed, 'readout': ro, 'states': used, 'replay_check': chk,
           'replay_pass2_equal_pass1': bool(np.array_equal(res2['plain']['mean_log'], p['mean_log'])),
           'mean_factor_summary': {'heldout_log_r_rms': rmse(np.log(r_held)), 'train_log_r_rms': rmse(np.log(r_train)),
                                   'train_log_r_by_quartile': {f'Q{b + 1}': float(np.mean(np.log(r_train[qb == b]))) for b in range(4)}}}
    parts = {}
    for sname, mask, var in (('heldout', held, 'match_heldout'), ('train', tr, 'match_train')):
        blk = {}
        for stat in ('mean_log', 'nonzero'):
            e_tot = (p[stat][mask] - rn[stat][mask]).mean(0)
            e_shape = (res2[var][stat][mask] - rn[stat][mask]).mean(0)
            e_mean = e_tot - e_shape
            blk[stat] = {'total_rmse': rmse(e_tot), 'shape_rmse': rmse(e_shape), 'mean_part_rmse': rmse(e_mean),
                         'total_bias': float(e_tot.mean()), 'shape_bias': float(e_shape.mean()), 'mean_part_bias': float(e_mean.mean()),
                         'shape_share_of_mse': float(np.mean(e_shape ** 2) / max(np.mean(e_tot ** 2), 1e-30)),
                         'total_by_quartile': by_quartile(e_tot, qb), 'shape_by_quartile': by_quartile(e_shape, qb)}
            np.save(OUT / f'gen_{a.arm}_f{k}_s{seed}_{ro}_{sname}_{stat}_eshape.npy', e_shape)
            np.save(OUT / f'gen_{a.arm}_f{k}_s{seed}_{ro}_{sname}_{stat}_etot.npy', e_tot)
        parts[sname] = blk
    out['decomposition'] = parts
    # matched-depth shape (train set, mean-matched variant vs real train cells)
    mt = res2['match_train']
    gdn = mt['dec_n'][tr].sum(0); gnz = mt['dec_nz'][tr].sum(0); gx = mt['dec_x'][tr].sum(0); gx2 = mt['dec_x2'][tr].sum(0)
    gl = mt['dec_lg'][tr].sum(0)
    rdn = rd['train_n']; rnz = rd['train_nz']; rx = rd['train_x']; rx2 = rd['train_x2']; rl = rd['train_lg']
    gN, rN = gdn[:, None], rdn[:, None]
    g_nzf, r_nzf = gnz / gN, rnz / rN
    g_m, r_m = gx / gN, rx / rN
    g_v, r_v = gx2 / gN - g_m ** 2, rx2 / rN - r_m ** 2
    with np.errstate(divide='ignore', invalid='ignore'):
        g_phi = (g_v - g_m) / g_m ** 2; r_phi = (r_v - r_m) / r_m ** 2
    tab = {}
    for b in range(4):
        gi = qb == b
        tab[f'Q{b + 1}'] = {
            'nonzero_gen_minus_real_by_decile': [float(np.mean(g_nzf[d, gi] - r_nzf[d, gi])) for d in range(N_DEC)],
            'meanlog_gen_minus_real_by_decile': [float(np.mean(gl[d, gi] / gdn[d] - rl[d, gi] / rdn[d])) for d in range(N_DEC)],
            'mean_count_ratio_gen_over_real_by_decile': [float(np.median(g_m[d, gi] / np.maximum(r_m[d, gi], 1e-12))) for d in range(N_DEC)],
            'phi_gen_median_by_decile': [float(np.nanmedian(g_phi[d, gi])) for d in range(N_DEC)],
            'phi_real_median_by_decile': [float(np.nanmedian(r_phi[d, gi])) for d in range(N_DEC)],
            'real_nonzero_by_decile': [float(np.mean(r_nzf[d, gi])) for d in range(N_DEC)],
            'excess_zero_vs_poisson_real': [float(np.mean((1 - r_nzf[d, gi]) - np.exp(-r_m[d, gi]))) for d in range(N_DEC)],
            'excess_zero_vs_poisson_gen': [float(np.mean((1 - g_nzf[d, gi]) - np.exp(-g_m[d, gi]))) for d in range(N_DEC)]}
    out['matched_depth_train'] = tab
    out['generated_cells_per_decile_share'] = (gdn / gdn.sum()).tolist()
    out['real_cells_per_decile_share'] = (rdn / rdn.sum()).tolist()
    # dispersion reachability on the mean-matched train set (analytic expected nonzero at theta multipliers)
    F = an[tr].mean(0)                     # [mults, G]
    tgt = rn['nonzero'][tr].mean(0)
    out['reachability_train_matched'] = {
        'multipliers': [str(m) for m in THETA_MULTS],
        'expected_nonzero_minus_target_rmse': [rmse(F[j] - tgt) for j in range(len(THETA_MULTS))],
        'expected_nonzero_minus_target_bias': [float(np.mean(F[j] - tgt)) for j in range(len(THETA_MULTS))],
        'frac_genes_target_above_poisson_limit': float(np.mean(tgt > F[-1])),
        'frac_genes_needing_more_dispersion': float(np.mean(tgt < F[1])),
        'frac_genes_target_below_F_at_half_theta': float(np.mean(tgt < F[0])),
        'by_quartile_needing_more_dispersion': {f'Q{b + 1}': float(np.mean(tgt[qb == b] < F[1][qb == b])) for b in range(4)},
        'by_quartile_above_poisson': {f'Q{b + 1}': float(np.mean(tgt[qb == b] > F[-1][qb == b])) for b in range(4)}}
    # theta: generator parameter vs ML effective theta of generated cells vs ML theta of real cells
    gen_theta = np.exp(logth[tr].mean(0))
    kc = np.concatenate([c for _, c, _ in kept]).astype(np.float64)
    kg = np.concatenate([g for _, _, g in kept])
    _, kg = np.unique(kg, return_inverse=True)
    th_eff, _ = nb_ml_theta(kc, kc.sum(1), kg)
    th_real = np.load(OUT / f'real_fold{k}_nb_donor_theta.npy')
    from scipy.stats import spearmanr
    lr_eff = np.log(th_eff / th_real); lr_par = np.log(gen_theta / th_real)
    out['theta'] = {'gen_param_median': float(np.median(gen_theta)), 'gen_effective_ml_median': float(np.median(th_eff)),
                    'real_ml_median': float(np.median(th_real)),
                    'spearman_param_vs_real_ml': float(spearmanr(gen_theta, th_real)[0]),
                    'spearman_effective_vs_real_ml': float(spearmanr(th_eff, th_real)[0]),
                    'median_log_ratio_effective_over_real_by_quartile': {f'Q{b + 1}': float(np.median(lr_eff[qb == b])) for b in range(4)},
                    'median_log_ratio_param_over_real_by_quartile': {f'Q{b + 1}': float(np.median(lr_par[qb == b])) for b in range(4)},
                    'frac_genes_generated_overdispersed_vs_real': float(np.mean(th_eff < th_real)),
                    'n_kept_generated_cells': int(len(kc))}
    np.save(OUT / f'gen_{a.arm}_f{k}_s{seed}_{ro}_theta_param.npy', gen_theta)
    np.save(OUT / f'gen_{a.arm}_f{k}_s{seed}_{ro}_theta_eff.npy', th_eff)
    out['seconds'] = time.monotonic() - t0
    jdump(OUT / f'gen_{a.arm}_f{k}_s{seed}_{ro}.json', out)
    print(json.dumps({kk: v for kk, v in out.items() if kk in ('replay_check', 'replay_pass2_equal_pass1', 'mean_factor_summary', 'theta', 'reachability_train_matched')}, indent=1))
    for s in ('heldout', 'train'):
        for st in ('mean_log', 'nonzero'):
            d = parts[s][st]
            print(s, st, {kk: round(v, 4) for kk, v in d.items() if isinstance(v, float)})


# =========================================================================== reproduce
def cmd_reproduce(a):
    import evaluate as EV
    k, arm, seed, mode = 0, 'pbulk', 0, 'biascal@stagehalf'
    gpath = RUNS / 'fits_r3' / arm / PANEL / f'fold{k}' / f'seed{seed}' / f'generated_{mode}.npz'
    rc = EV.RealCache(CACHES)
    real, subsets, pooled, dam = rc.get(PANEL, k)
    settings = EV.ScoreSettings(EV.DEFAULT_PC_REPEATS, EV.DEFAULT_CELL_REPEATS, dam)
    gen = EV.align_generated(EV.load_generated(gpath), real)
    res, _ = EV.score_generated(gen, real, subsets, pooled, settings=settings)
    old = json.loads((RUNS / 'results_r3' / 'fits' / f'{arm}~r3#{mode}' / PANEL / f'fold{k}' / f'seed{seed}.json').read_text())
    hn, ho = EV.headline(res), EV.headline(old)
    keys = ['pb_rmse', 'meanlog_rmse_gene', 'nonzero_rmse_gene', 'logvar_rmse_gene', 'pc_energy_rep', 'cell_energy_rep',
            'pc_coverage80_rep', 'meanlog_rmse_gene_recal', 'nonzero_rmse_gene_recal']
    rep = {kk: {'r5': hn[kk], 'results_r3': ho[kk], 'abs_diff': abs(hn[kk] - ho[kk])} for kk in keys}
    jdump(OUT / 'reproduce.json', rep)
    print(json.dumps(rep, indent=1))


# =========================================================================== summary
def cmd_summary(a):
    out = {}
    reals = {k: json.loads((OUT / f'real_fold{k}.json').read_text()) for k in (0, 1, 2) if (OUT / f'real_fold{k}.json').exists()}
    out['real'] = reals
    gens = {}
    for p in sorted(OUT.glob('gen_*.json')):
        gens[p.stem] = json.loads(p.read_text())
    out['gen'] = gens
    out['depth'] = {p.stem: json.loads(p.read_text()) for p in sorted(OUT.glob('depth_*.json'))}
    jdump(OUT / 'diagnosis_summary.json', out)
    print(f'{len(reals)} real folds, {len(gens)} generator diagnoses -> {OUT / "diagnosis_summary.json"}')
    print('\nREAL SIDE per fold: floors (route A), NB-ML on real train cells (in-sample), resampling split')
    for k, r in reals.items():
        f = r['floor_route_A']; nb = r['nb_ml_real_train']; rf = r['references']
        print(f'fold {k}: meanlog A_raw {f["meanlog"]["A_raw_train_mean"]:.4f} oracle {f["meanlog"]["A_grand_oracle"]:.4f} | '
              f'nonzero A_raw {f["nonzero"]["A_raw_train_mean"]:.4f} oracle {f["nonzero"]["A_grand_oracle"]:.4f}')
        for n in ('nb_pop', 'nb_donor'):
            lab = 'comparator (one mean per gene, in-sample)' if n == 'nb_pop' else 'in-sample UPPER bound (per-donor means fitted on the scored donors)'
            print(f'   [{lab}]')
            print(f'   {n:9s} nonzero {nb[n]["nonzero_rmse_gene_train_insample"]:.4f} (bias {nb[n]["nonzero_bias"]:+.4f})  '
                  f'meanlog {nb[n]["meanlog_rmse_gene_train_insample"]:.4f} (bias {nb[n]["meanlog_bias"]:+.4f})  theta median {nb[n]["theta_median"]:.3f}')
        for n in ('resample', 'resample_blind'):
            for st in ('meanlog', 'nonzero'):
                q = rf[n][st]
                print(f'   {n:14s} {st:7s} held-out {q["heldout_rmse_gene"]:.4f} = read-out part {q["readout_part_rmse"]:.4f} (bias {q["readout_bias"]:+.4f})'
                      f' + donor part {q["donor_part_rmse"]:.4f}; train {q["train_rmse_gene"]:.4f}; reproduced {rf[n]["reproduced"]}')
    print('\nGENERATOR SIDE (seed 0): total / shape (mean-matched) / mean part, held-out and train')
    for name, g in gens.items():
        d = g['decomposition']
        line = name.removeprefix('gen_') + ': '
        for sname in ('heldout', 'train'):
            for st in ('mean_log', 'nonzero'):
                b = d[sname][st]
                line += f'{sname[:4]} {st[:4]} {b["total_rmse"]:.4f}/{b["shape_rmse"]:.4f}/{b["mean_part_rmse"]:.4f} (shape bias {b["shape_bias"]:+.4f}); '
        print(line)
        rr = g['reachability_train_matched']; th = g['theta']
        print(f'     reach: above Poisson {rr["frac_genes_target_above_poisson_limit"]:.3f}, need more dispersion {rr["frac_genes_needing_more_dispersion"]:.3f}; '
              f'theta median param {th["gen_param_median"]:.3f} effective {th["gen_effective_ml_median"]:.3f} real ML {th["real_ml_median"]:.3f}; '
              f'gen over-dispersed in {th["frac_genes_generated_overdispersed_vs_real"]:.2f} of genes; replay {g["replay_check"].get("max_abs_diff_nonzero")}')
    # repair F7: per-fit ranges of the shape share of MSE, and the held-out shape residual next to the donor part
    share = {}
    for name, g in gens.items():
        d = g['decomposition']
        k = g['fold']
        for sname in ('train', 'heldout'):
            for st in ('mean_log', 'nonzero'):
                b = d[sname][st]
                sh = b.get('shape_share_of_mse', (b['shape_rmse'] / b['total_rmse']) ** 2 if b['total_rmse'] else None)
                share.setdefault(f'{g["readout"]}|{sname}_{st}', []).append(sh)
                if sname == 'heldout' and str(k) in {str(x) for x in reals}:
                    fa = reals[k if k in reals else str(k)]['floor_route_A']['meanlog' if st == 'mean_log' else 'nonzero']
                    share.setdefault(f'{g["readout"]}|heldout_{st}_shape_rmse_minus_trainmean_error', []).append(b['shape_rmse'] - fa['A_raw_train_mean'])
    rng_ = {kk: {'min': float(np.min(v)), 'max': float(np.max(v)), 'mean': float(np.mean(v)), 'n_fits': len(v)} for kk, v in share.items()}
    out['shape_share_per_fit_ranges'] = rng_
    out['shape_share_note'] = ('shape_share = shape MSE / total MSE per fit (seed 0, both arms, snapens and biascal@stagehalf). '
                               'Held-out: the shape residual includes donor-to-donor deviation of shape and is about the size of '
                               'the train-mean predictor error (the donor part), so it is not separable from donor variation. '
                               'nb_donor is an in-sample upper bound; nb_pop is the comparator for the generators (repair F5).')
    jdump(OUT / 'diagnosis_summary.json', out)
    print('\nSHAPE SHARE OF MSE per fit (min / mean / max) and held-out shape rmse minus train-mean predictor error')
    for kk, v in rng_.items():
        print(f'   {kk:70s} {v["min"]:+.3f} / {v["mean"]:+.3f} / {v["max"]:+.3f}  (n {v["n_fits"]})')
    for name, dd in out['depth'].items():
        print(name, 'beta', dd['beta_real_train'],
              {s: {k2: round(v['rmse'], 4) for k2, v in dd[s].items()} for s in ('train', 'heldout')})



# =========================================================================== depth coupling (diagnostic counterfactual)
def depth_elasticity(real, mask, edges, iters=25):
    """Per gene beta_g of x ~ Poisson(n_b exp(a + beta log Lbar_b)) over the 10 depth deciles of the donors in mask
    (aggregated counts; Newton on (a, beta) per gene).  beta = 1: counts proportional to depth (the generator's law)."""
    G = len(real.genes)
    n = np.zeros(N_DEC); Ls = np.zeros(N_DEC); x = np.zeros((N_DEC, G))
    for i in np.where(mask)[0]:
        X = real.cells[i]
        L = np.asarray(X.sum(1)).ravel()
        b = np.digitize(L, edges)
        for d in range(N_DEC):
            r = b == d
            n[d] += r.sum(); Ls[d] += L[r].sum()
            x[d] += np.asarray(X[r].sum(0)).ravel()
    lL = np.log(Ls / n)
    lc = lL - lL.mean()
    a = np.log(np.maximum(x.sum(0), 0.5) / n.sum()); beta = np.ones(G)
    for _ in range(iters):
        eta = a[None] + beta[None] * lc[:, None]
        m = n[:, None] * np.exp(eta)
        ga, gb = (x - m).sum(0), ((x - m) * lc[:, None]).sum(0)
        haa, hab, hbb = m.sum(0), (m * lc[:, None]).sum(0), (m * lc[:, None] ** 2).sum(0)
        det = haa * hbb - hab ** 2
        a = a + (hbb * ga - hab * gb) / det
        beta = beta + (haa * gb - hab * ga) / det
    return beta, float(np.exp(lL.mean())), np.exp(lL), n


def cmd_depth(a):
    """Mean-matched counterfactual with and without the REAL train depth elasticity imposed on the generator's mu."""
    import torch
    import evaluate as EV
    torch.set_num_threads(3)
    torch.cuda.set_per_process_memory_fraction(0.04)
    k, arm, seed, ro = a.fold, ARMS[a.arm], a.seed, a.readout
    fit = RUNS / 'fits_r3' / arm / PANEL / f'fold{k}' / f'seed{seed}'
    cache = CACHES / PANEL / f'fold{k}'
    real = EV.load_real(cache, PANEL, k)
    rd = np.load(OUT / f'real_fold{k}_deciles.npz')
    edges = rd['edges']
    beta, L0, Lbar, _ = depth_elasticity(real, ~real.is_heldout, edges)
    ctx, models, plans, T, used = load_readout(fit, cache, ro, 'cuda')
    order = [list(real.donors).index(t[0]) for t in T]
    held = real.is_heldout[order]; tr = ~held
    rn = {'mean_log': real.mean_log[order], 'nonzero': real.nonzero[order], 'pb': real.pb[order]}
    qb = quartiles(rn['mean_log'][tr].mean(0))
    res1, _, _, _ = simulate(ctx, models, plans, T, {'plain': None}, edges)
    p = res1['plain']
    sh = lambda pb: pb / np.maximum(pb.sum(1, keepdims=True), 1)
    r_train = sh(rn['pb'])[tr].mean(0) / np.maximum(sh(p['pb'])[tr].mean(0), 1e-12)
    r_held = sh(rn['pb'])[held].mean(0) / np.maximum(sh(p['pb'])[held].mean(0), 1e-12)
    # depth factor per cell and gene: (L/L0)^(beta-1), renormalised over the pooled real train libraries
    Ltr = np.concatenate([real.libs[i] for i in np.where(~real.is_heldout)[0]])
    cg = Ltr.mean() / np.array([np.mean(Ltr * (Ltr / L0) ** (b - 1)) for b in beta])
    bt = torch.as_tensor(beta - 1, dtype=torch.float32, device=ctx.device)
    import mode_zcal as MZ
    from generate import K

    def run(variant_r, depth):
        out = {'mean_log': np.zeros((len(T), ctx.n_genes)), 'nonzero': np.zeros((len(T), ctx.n_genes))}
        fac = torch.as_tensor(variant_r * (cg if depth else 1.0), dtype=torch.float32, device=ctx.device)
        with torch.no_grad():
            for ti, start, n, G, z, c, lib, cell, count in MZ.iterate_chunks(ctx, models, plans, T):
                mu, theta = G.parameters_nb(z, c, K, lib, cell)
                mu = mu * fac
                if depth:
                    mu = mu * torch.exp(torch.log(lib / L0)[..., None] * bt)
                rate = torch.distributions.Gamma(theta, theta / mu).rsample() * (lib > 0)[..., None]
                counts = torch.poisson(rate, generator=count)
                tot = counts.sum(-1)
                logcp = torch.log1p(counts * (1e4 / tot.clamp_min(1))[..., None])
                out['mean_log'][ti] += logcp.mean(1).sum(0).double().cpu().numpy() / 32
                out['nonzero'][ti] += (counts > 0).float().mean(1).sum(0).double().cpu().numpy() / 32
        return out
    res = {'fold': k, 'arm': arm, 'seed': seed, 'readout': ro,
           'beta_real_train': {'median': float(np.median(beta)), 'by_quartile_median': {f'Q{b + 1}': float(np.median(beta[qb == b])) for b in range(4)}},
           'note': 'beta = depth elasticity of REAL train counts (1 = proportional, the generator law); depth variant imposes it on mu'}
    for sname, mask, r in (('train', tr, r_train), ('heldout', held, r_held)):
        blk = {}
        for depth in (False, True):
            o = run(r, depth)
            for stat in ('mean_log', 'nonzero'):
                e = (o[stat][mask] - rn[stat][mask]).mean(0)
                blk[f'{stat}_{"matched+depth" if depth else "matched"}'] = {'rmse': rmse(e), 'bias': float(e.mean()), 'by_quartile': by_quartile(e, qb)}
        res[sname] = blk
    jdump(OUT / f'depth_{a.arm}_f{k}_s{seed}_{ro}.json', res)
    print(json.dumps({kk: v for kk, v in res.items()}, indent=1)[:3000])



# =========================================================================== (3) held-out evaluation of zcal
EVAL_SEEDS = {'ctrl': [0, 1], 'combo': [0, 1, 2, 3]}
ZMODES = ['zcal@biascal:stagehalf', 'zcal@stagehalf']
REF_MODES = ['biascal@stagehalf', 'stagehalf']
METRICS = ['pb_rmse', 'meanlog_rmse_gene', 'nonzero_rmse_gene', 'logvar_rmse_gene', 'pc_energy_rep', 'cell_energy_rep',
           'pc_coverage80_rep', 'meanlog_bias', 'nonzero_bias', 'logvar_bias', 'meanlog_rmse_gene_recal', 'nonzero_rmse_gene_recal',
           'pb_rmse_recal']
ZDIR = OUT / 'zeval'


def _score_file(rc, k, path):
    import evaluate as EV
    real, subsets, pooled, dam = rc.get(PANEL, k)
    st = EV.ScoreSettings(EV.DEFAULT_PC_REPEATS, EV.DEFAULT_CELL_REPEATS, dam)
    res, _ = EV.score_generated(EV.align_generated(EV.load_generated(path), real), real, subsets, pooled, settings=st)
    h = EV.headline(res)
    ht = EV.headline(res, 'train')
    blk = res['subsets']['all']['heldout']
    return {'heldout': {m: h.get(m) for m in METRICS}, 'train': {m: ht.get(m) for m in METRICS},
            'meanlog_by_quartile': blk['mean_log']['by_expression_quartile'], 'nonzero_by_quartile': blk['nonzero']['by_expression_quartile'],
            'generated_sha256': EV.sha256(path)}


def cmd_zeval(a):
    """Generate every zcal read-out (GPU, one process at a time), score it with the r5 evaluator, delete the mode file."""
    import subprocess
    import evaluate as EV
    ZDIR.mkdir(parents=True, exist_ok=True)
    rc = EV.RealCache(CACHES)
    py = sys.executable
    here = Path(__file__).resolve().parent
    for k in a.folds:
        for tag, seeds in EVAL_SEEDS.items():
            for seed in seeds:
                fit = RUNS / 'fits_r3' / ARMS[tag] / PANEL / f'fold{k}' / f'seed{seed}'
                for mode in ZMODES:
                    js = ZDIR / f'{tag}_f{k}_s{seed}_{mode}.json'
                    if js.exists():
                        continue
                    out = OUT / 'modes' / f'{tag}_f{k}_s{seed}_{mode}.npz'
                    out.parent.mkdir(parents=True, exist_ok=True)
                    t0 = time.monotonic()
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpu, OMP_NUM_THREADS='3', OPENBLAS_NUM_THREADS='3', MKL_NUM_THREADS='3')
                    cmd = [py, '-B', str(here / 'mode_zcal.py'), '--fit', str(fit), '--cache', str(CACHES / PANEL / f'fold{k}'),
                           '--mode', mode, '--out', str(out), '--device', 'cuda', '--gpu-mem-fraction', '0.04', '--force']
                    log = (RUNS / 'logs_r5' / f'zeval_{tag}_f{k}_s{seed}_{mode}.log')
                    with open(log, 'w') as fh:
                        subprocess.run(cmd, check=True, stdout=fh, stderr=subprocess.STDOUT, cwd='/tmp', env=env)
                    gen_s = time.monotonic() - t0
                    sc = _score_file(rc, k, out)
                    rec = json.loads(out.with_name(out.stem + '.receipt.json').read_text())
                    z = rec['mode_extra']['zcal']
                    sc.update(fit=str(fit), mode=mode, fold=k, seed=seed, arm=ARMS[tag], generation_seconds=gen_s,
                              zcal={kk: v for kk, v in z.items() if kk not in ('s',)},
                              biascal={kk: v for kk, v in (rec['mode_extra'].get('biascal') or {}).items() if kk != 'delta'},
                              receipt=rec)
                    jdump(js, sc)
                    out.unlink()               # rule: delete mode files once scored (the receipt and the score stay)
                    print(f'[zeval] {tag} f{k} s{seed} {mode}: ' + ', '.join(f'{m} {sc["heldout"][m]:.4f}' for m in METRICS[:7]), flush=True)


def _ref_scores(tag, k, seed, mode):
    p = RUNS / 'results_r3' / 'fits' / f'{ARMS[tag]}~r3#{mode}' / PANEL / f'fold{k}' / f'seed{seed}.json'
    import evaluate as EV
    d = json.loads(p.read_text())
    h, ht = EV.headline(d), EV.headline(d, 'train')
    return {'heldout': {m: h.get(m) for m in METRICS}, 'train': {m: ht.get(m) for m in METRICS}}


def paired(diffs):
    """diffs: {fold: [per-seed differences]} -> mean, seed-noise SE, fold-scatter SE and t, folds better (lower)."""
    folds = sorted(diffs)
    fm = np.array([np.mean(diffs[f]) for f in folds])
    D = float(fm.mean())
    within = [np.var(diffs[f], ddof=1) / len(diffs[f]) for f in folds if len(diffs[f]) > 1]
    se_seed = float(np.sqrt(np.sum(within)) / len(folds)) if within else None
    se_fold = float(fm.std(ddof=1) / np.sqrt(len(folds))) if len(folds) > 1 else None
    return {'mean_diff': D, 'per_fold': fm.tolist(), 'se_seed_noise': se_seed, 'se_fold_scatter': se_fold,
            't_fold': (D / se_fold) if se_fold else None, 'z_seed': (D / se_seed) if se_seed else None,
            'n_folds_lower': int((fm < 0).sum()), 'n_fits': int(sum(len(v) for v in diffs.values()))}


def cmd_zreport(a):
    rows = {}
    for p in sorted(ZDIR.glob('*.json')):
        d = json.loads(p.read_text())
        tag = 'ctrl' if d['arm'] == ARMS['ctrl'] else 'combo'
        rows[(tag, d['fold'], d['seed'], d['mode'])] = d
    for (tag, k, seed, _m) in list(rows):
        for rm in REF_MODES:
            if (tag, k, seed, rm) not in rows:
                rows[(tag, k, seed, rm)] = _ref_scores(tag, k, seed, rm)
    refs = {}
    for k in (0, 1, 2):
        rj = json.loads((OUT / f'real_fold{k}.json').read_text())
        refs[k] = {n: rj['references'][n]['results_r3'] for n in ('resample', 'resample_blind')}
        # repair F2: the direct train-mean predictor as a reference (route A raw; no M, no recal: recal value = raw)
        refs[k]['trainmean'] = {'meanlog_rmse_gene': rj['floor_route_A']['meanlog']['A_raw_train_mean'],
                                'nonzero_rmse_gene': rj['floor_route_A']['nonzero']['A_raw_train_mean'],
                                'meanlog_rmse_gene_recal': rj['floor_route_A']['meanlog']['A_raw_train_mean'],
                                'nonzero_rmse_gene_recal': rj['floor_route_A']['nonzero']['A_raw_train_mean']}
        refs[k]['floor'] = {'meanlog_A_raw': rj['floor_route_A']['meanlog']['A_raw_train_mean'],
                            'meanlog_A_oracle': rj['floor_route_A']['meanlog']['A_grand_oracle'],
                            'nonzero_A_raw': rj['floor_route_A']['nonzero']['A_raw_train_mean'],
                            'nonzero_A_oracle': rj['floor_route_A']['nonzero']['A_grand_oracle']}
        for n in ('resample', 'resample_blind'):
            bj = json.loads((RUNS / 'results_r3' / 'baselines' / PANEL / f'fold{k}' / f'{n}.json').read_text())
            import evaluate as EV
            h = EV.headline(bj)
            refs[k][n] = {m: h.get(m) for m in METRICS}
    rep = {'references_per_fold': refs, 'means': {}, 'paired': {}, 'vs_resample': {}}
    for tag in EVAL_SEEDS:
        for mode in ZMODES + REF_MODES:
            keys = [kk for kk in rows if kk[0] == tag and kk[3] == mode]
            if not keys:
                continue
            rep['means'][f'{tag}|{mode}'] = {m: float(np.mean([rows[kk]['heldout'][m] for kk in keys])) for m in METRICS
                                             if all(rows[kk]['heldout'][m] is not None for kk in keys)}
            rep['means'][f'{tag}|{mode}']['n_fits'] = len(keys)
        for new, old in (('zcal@biascal:stagehalf', 'biascal@stagehalf'), ('zcal@stagehalf', 'stagehalf')):
            blk = {}
            for m in METRICS:
                diffs = {}
                for (t2, k, seed, mode) in rows:
                    if t2 == tag and mode == new and (tag, k, seed, old) in rows:
                        diffs.setdefault(k, []).append(rows[(tag, k, seed, new)]['heldout'][m] - rows[(tag, k, seed, old)]['heldout'][m])
                if diffs:
                    blk[m] = paired(diffs)
            rep['paired'][f'{tag}|{new} - {old}'] = blk
        for mode in ZMODES + REF_MODES:
            for refn in ('resample', 'resample_blind', 'trainmean'):
                blk = {}
                for m in METRICS:
                    diffs = {}
                    for (t2, k, seed, md) in rows:
                        if t2 == tag and md == mode and refs[k][refn].get(m) is not None:
                            diffs.setdefault(k, []).append(rows[(t2, k, seed, md)]['heldout'][m] - refs[k][refn][m])
                    if diffs:
                        blk[m] = paired(diffs)
                rep['vs_resample'][f'{tag}|{mode} - {refn}'] = blk
    # zcal calibration records
    rep['zcal_records'] = {f'{kk[0]}_f{kk[1]}_s{kk[2]}_{kk[3]}': {'n_clipped_high': rows[kk]['zcal']['solves'][-1]['n_clipped_high'],
                                                                 'n_clipped_low': rows[kk]['zcal']['solves'][-1]['n_clipped_low'],
                                                                 's_mean': rows[kk]['zcal']['s_mean'], 's_rms': rows[kk]['zcal']['s_rms'],
                                                                 'final': rows[kk]['zcal']['final']}
                           for kk in rows if kk[3] in ZMODES}
    # repair F2: headline in the corrected order (zcal vs biascal; vs resampling; vs the train-mean predictor; recal gap)
    hl = {'note': ('zcal pulls the generator onto train statistics (train nonzero fraction per gene). The direct train-mean '
                   'predictor built from the same train statistics beats resampling and zcal on both metrics (resampling is not a '
                   'floor). After the train-only additive per-gene recalibration (applied alike to models and resampling) zcal '
                   'changes nothing. "gap closed" is relative to resampling only.')}
    for tag in EVAL_SEEDS:
        z, b = rep['means'].get(f'{tag}|zcal@biascal:stagehalf'), rep['means'].get(f'{tag}|biascal@stagehalf')
        if not (z and b):
            continue
        h = {}
        for m in ('meanlog_rmse_gene', 'nonzero_rmse_gene'):
            rs = float(np.mean([refs[k]['resample'][m] for k in (0, 1, 2)]))
            tm = float(np.mean([refs[k]['trainmean'][m] for k in (0, 1, 2)]))
            h[m] = {'biascal@stagehalf': b[m], 'zcal@biascal:stagehalf': z[m], 'resample': rs, 'trainmean': tm,
                    'gap_to_resample_closed': (b[m] - z[m]) / (b[m] - rs) if b[m] != rs else None,
                    'zcal_minus_biascal': rep['paired'][f'{tag}|zcal@biascal:stagehalf - biascal@stagehalf'].get(m),
                    'zcal_minus_resample': rep['vs_resample'][f'{tag}|zcal@biascal:stagehalf - resample'].get(m),
                    'zcal_minus_trainmean': rep['vs_resample'][f'{tag}|zcal@biascal:stagehalf - trainmean'].get(m),
                    'resample_minus_trainmean_per_fold': [refs[k]['resample'][m] - refs[k]['trainmean'][m] for k in (0, 1, 2)]}
            mr = m + '_recal'
            h[mr] = {'biascal@stagehalf': b.get(mr), 'zcal@biascal:stagehalf': z.get(mr),
                     'resample': float(np.mean([refs[k]['resample'][mr] for k in (0, 1, 2)])),
                     'zcal_minus_biascal': rep['paired'][f'{tag}|zcal@biascal:stagehalf - biascal@stagehalf'].get(mr)}
        hl[tag] = h
    rep['headline'] = hl
    jdump(OUT / 'zeval_report.json', rep)
    show = ['pb_rmse', 'meanlog_rmse_gene', 'nonzero_rmse_gene', 'logvar_rmse_gene', 'pc_energy_rep', 'cell_energy_rep', 'pc_coverage80_rep',
            'meanlog_rmse_gene_recal', 'nonzero_rmse_gene_recal']
    print('MEANS (held-out, folds 0-2)')
    print('%-40s' % '' + ''.join('%11s' % m[:10] for m in show))
    for name, v in rep['means'].items():
        print('%-40s' % name + ''.join('%11.4f' % v.get(m, np.nan) for m in show) + f'  n={v["n_fits"]}')
    for n in ('resample', 'resample_blind'):
        print('%-40s' % n + ''.join('%11.4f' % np.mean([refs[k][n][m] for k in (0, 1, 2)]) if refs[0][n].get(m) is not None else '%11s' % '-' for m in show))
    print('\nPAIRED new - old (mean, seed SE, fold SE, folds lower)')
    for name, blk in rep['paired'].items():
        print(name)
        for m in show:
            if m in blk:
                b = blk[m]
                print(f'   {m:26s} {b["mean_diff"]:+.4f}  seedSE {b["se_seed_noise"] if b["se_seed_noise"] is not None else float("nan"):.4f}  '
                      f'foldSE {b["se_fold_scatter"]:.4f}  t_fold {b["t_fold"]:+.2f}  lower {b["n_folds_lower"]}/3  per-fold {np.round(b["per_fold"], 4).tolist()}')
    print('%-40s' % 'trainmean (route A raw; recal = raw)' + ''.join('%11.4f' % np.mean([refs[k]['trainmean'][m] for k in (0, 1, 2)])
                                                                   if m in refs[0]['trainmean'] else '%11s' % '-' for m in show))
    print('\nHEADLINE (repair F2)')
    print(json.dumps({t: {m: {kk: (round(v, 5) if isinstance(v, float) else (round(v['mean_diff'], 5) if isinstance(v, dict) else v))
                              for kk, v in d.items()} for m, d in h.items()} for t, h in hl.items() if t != 'note'}, indent=1))
    print('\nVS RESAMPLING AND THE TRAIN-MEAN PREDICTOR (model - reference; meanlog / nonzero)')
    for name, blk in rep['vs_resample'].items():
        for m in ('meanlog_rmse_gene', 'nonzero_rmse_gene', 'meanlog_rmse_gene_recal', 'nonzero_rmse_gene_recal'):
            if m in blk:
                b = blk[m]
                print(f'{name:55s} {m:26s} {b["mean_diff"]:+.4f}  seedSE {b["se_seed_noise"] if b["se_seed_noise"] is not None else float("nan"):.4f}  '
                      f'foldSE {b["se_fold_scatter"]:.4f}  lower {b["n_folds_lower"]}/3')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('reproduce')
    r = sub.add_parser('real'); r.add_argument('--fold', type=int, required=True)
    g = sub.add_parser('gen')
    g.add_argument('--fold', type=int, required=True); g.add_argument('--arm', choices=sorted(ARMS), required=True)
    g.add_argument('--seed', type=int, required=True); g.add_argument('--readout', required=True)
    g.add_argument('--device', choices=['cpu', 'cuda'], default='cuda'); g.add_argument('--gpu-mem-fraction', type=float, default=0.04)
    sub.add_parser('summary')
    z = sub.add_parser('zeval'); z.add_argument('--folds', type=lambda t: [int(x) for x in t.split(',')], default=[0, 1, 2])
    z.add_argument('--gpu', default='0')
    sub.add_parser('zreport')
    dd = sub.add_parser('depth')
    dd.add_argument('--fold', type=int, required=True); dd.add_argument('--arm', choices=sorted(ARMS), required=True)
    dd.add_argument('--seed', type=int, default=0); dd.add_argument('--readout', default='biascal@stagehalf')
    a = p.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    import trainer  # noqa: F401  (released-module paths)
    {'reproduce': cmd_reproduce, 'real': cmd_real, 'gen': cmd_gen, 'summary': cmd_summary, 'zeval': cmd_zeval,
     'zreport': cmd_zreport, 'depth': cmd_depth}[a.cmd](a)


if __name__ == '__main__':
    sys.modules.setdefault('shape_diagnose', sys.modules['__main__'])
    main()
