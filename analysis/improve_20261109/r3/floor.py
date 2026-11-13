#!/usr/bin/env python3
"""Floor of held-out pb_rmse (and meanlog_rmse_gene) for a stage-only predictor (R7).

Question (PROTOCOL_R3.md, T1): how low can held-out ``pb_rmse`` go for ANY
predictor that knows only the target donor's stage (and its library sizes)?
Even an oracle that knows the true population mean profile of the stage pays
(a) donor-to-donor biological variation and (b) the target donor's own
cell-sampling noise.  Everything here uses only the 37 development donors
(the r3 evaluate.load_real guard refuses any other donor) and is a REFERENCE:
nothing is chosen or fitted for a model with it.

Notation per inner fold k (its own 2,000-gene panel and CP10k denominator):
y_d [G] = observed full-depth pseudobulk log1p(CP10k) of donor d (exactly the
evaluator's ``obs``); per-donor RMSE(p) = sqrt(mean_g (p_g - y_dg)^2);
pb_rmse = mean over the fold's held-out donors of the per-donor RMSE (the
evaluator's definition); folds are averaged within a repeat (folds 0-2 =
repeat 0, 3-5 = 1, 6-8 = 2).  x_d [G] = per-cell mean of log1p(panel CP10k)
over all of d's cells (evaluator ``real.mean_log``); meanlog_rmse_gene =
sqrt(mean_g e_g^2), e_g = mean over held-out donors of (prediction - x_dg).

Routes (each gives the floor for the OPTIMAL constant predictor: the true
population mean mu ("grand", no stage) or the true stage means mu_s ("stage")):

A. Held-out around the train mean, de-biased.  For held-out d,
   MSE_d(ybar_train) = |y_d - mu|^2/G + |ybar_train - mu|^2/G + cross term with
   expectation 0; E|ybar_train - mu|^2/G = mean_g s2_g / n_train (s2 = train
   sample variance, ddof 1).  floor_d = sqrt(max(MSE_d - mean_g s2_g/n_train, 0)),
   floor = mean over held-out d.  Stage: ybar_s from train donors of the same
   stage, correction mean_g(pooled within-stage s2)/n_s.  Held-out donors are
   used only as the evaluation sample, exactly as the evaluator uses them.
   ``raw`` = the uncorrected value (= the direct train-mean predictor).
B. Leave-one-donor-out within all 37 donors (in fold k's gene space):
   floor_d = sqrt(max(|y_d - ybar_{-d}|^2/G - mean_g s2_{-d,g}/36, 0)); stage:
   ybar_{s,-d} over the n_s - 1 other donors of the stage, correction
   pooled within-stage s2 (all donors but d) / (n_s - 1).  Reported averaged
   over the fold's held-out donors (``heldout``) and over all 37 (``all37``).
C. Variance components (ANOVA over the 37 donors): rms_total =
   sqrt(mean_g var_g) (grand floor in root-mean-square form), rms_within =
   sqrt(mean_g pooled within-stage var_g); technical part tau2_d = mean_g of
   the cell-bootstrap variance of y_dg (B = 40 multinomial resamples of d's
   cells); ``floor_without_technical`` = per-donor sqrt(max(floor_d^2 -
   tau2_d, 0)) (what infinitely many cells per donor would give).
D. Ideal stage-blind generator emulation, through the evaluator's READ-OUT:
   baseline ``resample_blind`` (r3/baselines.py: like ``resample`` but the
   source donors are ALL train donors except the target, i.e. a generator that
   reproduces the training-donor population perfectly and ignores stage),
   M = 32 generated donors x K = 256 depth-thinned cells, prediction = mean
   over M of log1p CP10k of the 256-cell pseudobulk (the evaluator's
   prediction).  D - A.raw = cost of the generator read-out (finite M, the
   256-cell log bias) on top of the direct train-mean predictor.  ``D_inf_M``
   splits it: every train donor once, many 256-cell draws (no finite-M
   donor sampling): D_inf - A.raw = K-cell log bias, D - D_inf = finite M.

meanlog floors (same routes): A: corrected e^2 = mean_g (xbar_held - xbar_train)^2
- mean_g s2_train/n_train; B (analytic): sqrt(mean_g var_g / n_held) (grand)
and sqrt(mean_g within-stage var_g / n_held) (stage); ``depth_adjusted``:
variances of the residual of x_dg on log(mean library) (+ stage dummies),
because the generator is given the target's library sizes; D as above.
Note the meanlog floor shrinks like 1/sqrt(n_held): it is a property of
the 12-13 held-out donors, not of the model.

Uncertainty.  Route A, C and D: donor bootstrap (B = 1000 by default) with ONE
multinomial weight vector over the 37 donors per replicate, applied to every
fold of a repeat (folds share donors, so this keeps their dependence); a
replicate is skipped for a fold whose held-out or train weight (or a needed
stage) is empty.  Route B: SE = SD of the per-donor values / sqrt(n)
(approximate; LOO and bootstrap duplicates do not mix).

Usage:
  floor.py [--folds 0,...,8] [--panel hvg2000] [--out runs/improve_20261109/results_r3/floor.json]
           [--boot 1000] [--no-generator] [--baseline-root DIR (read grand_mean / ridge baselines
           from an evaluator output, e.g. results_r2_interim, for the comparison rows)]
"""
from __future__ import annotations

import os
for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    try:
        _n = int(os.environ.get(_v, "3"))
    except ValueError:
        _n = 3
    os.environ[_v] = str(max(1, min(_n, 3)))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
if str(HERE) in sys.path:
    sys.path.remove(str(HERE))
sys.path.insert(0, str(HERE))
import evaluate as EV  # noqa: E402
import metrics as MX  # noqa: E402
import baselines as BL  # noqa: E402

for _m in (EV, MX, BL):
    if Path(_m.__file__).resolve().parent != HERE:
        raise ImportError(f"floor.py imported the wrong module {_m.__file__}")

FLOOR_VERSION = "improve_20261109-floor-r3-v1"
RUNS = EV.RUNS
CELL_BOOT = 40


# --------------------------------------------------------------------------- small helpers
def _neff(w):
    """Effective number of distinct draws behind a weighted mean (Kish): (sum w)^2 / sum w^2.
    Equals n for unit weights; smaller for bootstrap weights with duplicates."""
    w = np.asarray(w, dtype=np.float64)
    return float(w.sum() ** 2 / max((w ** 2).sum(), 1e-300))


def _wmv(Y, w):
    """Weighted mean, frequency-weight variance (ddof 1) and the Kish effective n of rows of Y."""
    n = float(w.sum())
    m = (w[:, None] * Y).sum(0) / n
    v = (w[:, None] * (Y - m) ** 2).sum(0) / max(n - 1.0, 1.0)
    return m, v, _neff(w)


def _within(Y, stage, w):
    """Stage means {s: (mean, n)} and pooled within-stage variance [G] (frequency weights)."""
    means, ss, ntot, ns = {}, np.zeros(Y.shape[1]), 0.0, 0
    for s in np.unique(stage):
        ws = w * (stage == s)
        n = float(ws.sum())
        if n <= 0:
            continue
        m = (ws[:, None] * Y).sum(0) / n
        means[int(s)] = (m, _neff(ws[ws > 0]))
        ss += (ws[:, None] * (Y - m) ** 2).sum(0)
        ntot += n; ns += 1
    return means, ss / max(ntot - ns, 1.0)


def _stats(per, w=None):
    per = np.asarray(per, dtype=np.float64)
    ok = np.isfinite(per)
    if w is None:
        w = np.ones_like(per)
    w = np.where(ok, w, 0.0)
    if w.sum() <= 0:
        return None
    return float((w * np.nan_to_num(per)).sum() / w.sum())


def _summ(point, boots):
    b = np.array([x for x in boots if x is not None and np.isfinite(x)])
    out = {"value": point, "boot_n": int(len(b))}
    if len(b) > 1:
        out.update(se=float(b.std(ddof=1)), ci95=[float(np.quantile(b, .025)), float(np.quantile(b, .975))])
    return out


# --------------------------------------------------------------------------- per-fold quantities
class FoldData:
    def __init__(self, real):
        self.real = real
        self.donors = real.donors.astype(str)
        self.held = real.is_heldout.astype(bool)
        self.stage = real.stage.astype(int)
        self.Y = MX.log_cp10k(real.pb)                       # [37,G] evaluator obs
        self.X = real.mean_log.astype(np.float64)            # [37,G]
        self.loglib = np.log(np.array([l.mean() for l in real.libs]))
        self.G = self.Y.shape[1]
        self.tau2 = None

    def technical(self, n_boot=CELL_BOOT):
        """Cell-bootstrap variance of y_dg (mean over genes) and of x_dg per donor."""
        if self.tau2 is not None:
            return self.tau2, self.tau2_ml
        t2, t2m = [], []
        for i, d in enumerate(self.donors):
            C = self.real.cells[i].toarray().astype(np.float64)
            n = C.shape[0]
            rng = np.random.default_rng(MX.stable_seed("floor-cellboot", self.real.panel, d))
            Wb = rng.multinomial(n, np.full(n, 1.0 / n), size=n_boot).astype(np.float64)
            yb = MX.log_cp10k(Wb @ C)
            L = MX.log_cp10k(C)
            xb = (Wb @ L) / n
            t2.append(float(yb.var(0, ddof=1).mean())); t2m.append(xb.var(0, ddof=1))
        self.tau2 = np.array(t2); self.tau2_ml = np.stack(t2m)
        return self.tau2, self.tau2_ml


def route_a(fd: FoldData, w):
    """Route A (pb and meanlog), donor weights w [37] (1 = point estimate)."""
    tr, he = ~fd.held, fd.held
    wt, wh = w[tr], w[he]
    out = {}
    if wt.sum() < 2 or wh.sum() <= 0:
        return None
    Yt, Yh, st_t, st_h = fd.Y[tr], fd.Y[he], fd.stage[tr], fd.stage[he]
    m, v, n = _wmv(Yt, wt)
    mse = ((Yh - m) ** 2).mean(1)
    corr = v.mean() / n
    out["pb_raw"] = _stats(np.sqrt(mse), wh)
    out["pb_grand"] = _stats(np.sqrt(np.maximum(mse - corr, 0.0)), wh)
    out["pb_grand_correction_mse"] = float(corr)
    means, vw = _within(Yt, st_t, wt)
    per = np.full(len(Yh), np.nan)
    for j in range(len(Yh)):
        s = int(st_h[j])
        if s in means and means[s][1] > 0:
            ms, ns = means[s]
            per[j] = np.sqrt(max(((Yh[j] - ms) ** 2).mean() - vw.mean() / ns, 0.0))
    out["pb_stage"] = _stats(per, wh) if np.isfinite(per[wh > 0]).all() else None
    out["pb_stage_raw_undefined_donors"] = int((~np.isfinite(per)).sum())
    # meanlog
    Xt, Xh = fd.X[tr], fd.X[he]
    mx, vx, nx = _wmv(Xt, wt)
    xh = (wh[:, None] * Xh).sum(0) / wh.sum()
    e2 = ((xh - mx) ** 2).mean()
    out["ml_raw"] = float(np.sqrt(e2))
    out["ml_grand"] = float(np.sqrt(max(e2 - vx.mean() / nx, 0.0)))
    return out


def route_b(fd: FoldData):
    """Leave-one-donor-out among all 37 donors; per-donor floors (grand, stage)."""
    N = len(fd.Y)
    g, s_, raw = np.full(N, np.nan), np.full(N, np.nan), np.full(N, np.nan)
    for d in range(N):
        o = np.arange(N) != d
        m, v, n = _wmv(fd.Y[o], np.ones(int(o.sum())))
        mse = ((fd.Y[d] - m) ** 2).mean()
        raw[d] = np.sqrt(mse)
        g[d] = np.sqrt(max(mse - v.mean() / n, 0.0))
        same = o & (fd.stage == fd.stage[d])
        if same.sum() >= 1:
            _, vw = _within(fd.Y[o], fd.stage[o], np.ones(int(o.sum())))
            ms = fd.Y[same].mean(0)
            s_[d] = np.sqrt(max(((fd.Y[d] - ms) ** 2).mean() - vw.mean() / same.sum(), 0.0))
    return raw, g, s_


def route_c(fd: FoldData, per_grand_b):
    """Variance components and the technical (cell-sampling) part."""
    N = len(fd.Y)
    w1 = np.ones(N)
    _, v, _ = _wmv(fd.Y, w1)
    _, vw = _within(fd.Y, fd.stage, w1)
    tau2, tau2_ml = fd.technical()
    no_tech = np.sqrt(np.maximum(per_grand_b ** 2 - tau2, 0.0))
    return {"rms_total": float(np.sqrt(v.mean())), "rms_within_stage": float(np.sqrt(vw.mean())),
            "technical_rms": float(np.sqrt(tau2.mean())),
            "technical_share_of_total_variance": float(tau2.mean() / v.mean()),
            "per_donor_technical_rms": [float(x) for x in np.sqrt(tau2)],
            "floor_without_technical_all37": float(np.nanmean(no_tech)),
            "floor_without_technical_heldout": float(np.nanmean(no_tech[fd.held]))}


def meanlog_analytic(fd: FoldData):
    N = len(fd.X); nh = int(fd.held.sum())
    w1 = np.ones(N)
    _, v, _ = _wmv(fd.X, w1)
    _, vw = _within(fd.X, fd.stage, w1)
    out = {"n_heldout": nh, "grand": float(np.sqrt(v.mean() / nh)), "stage": float(np.sqrt(vw.mean() / nh))}
    # depth-adjusted: residual variance after regressing on log mean library (+ stage dummies)
    for name, cols in [("grand_depth_adjusted", [np.ones(N), fd.loglib]),
                       ("stage_depth_adjusted", [np.ones(N), fd.loglib] + [(fd.stage == s).astype(float) for s in (1, 2, 3)])]:
        A = np.stack(cols, 1)
        beta, *_ = np.linalg.lstsq(A, fd.X, rcond=None)
        R = fd.X - A @ beta
        rv = (R ** 2).sum(0) / max(N - np.linalg.matrix_rank(A), 1)
        out[name] = float(np.sqrt(rv.mean() / nh))
    tau2, tau2_ml = fd.technical()
    out["technical_part"] = float(np.sqrt(tau2_ml[fd.held].mean() / nh))
    return out


def route_d(fd: FoldData):
    """Ideal stage-blind generator emulation (baseline resample_blind) through the evaluator read-out."""
    gen, info = BL.generate("resample_blind", fd.real)
    gen = EV.align_generated(gen, fd.real)
    pred = MX.predicted_profiles(gen["pb_counts"].astype(np.float64))
    per = np.sqrt(((pred - fd.Y) ** 2).mean(1))
    gml = gen["mean_log"].astype(np.float64).mean(1)
    return per, gml, info


def route_dinf(fd: FoldData, K=256, R=8):
    """Infinite-M version of D: every train donor equally weighted, R draws of K cells (with replacement,
    no thinning) each; prediction = mean over donors and draws of log1p CP10k of the K-cell pseudobulk.
    D_inf - A_raw = cost of the K-cell log read-out alone; D - D_inf = cost of finite M = 32 (donor
    sampling, repeated donors in the 32-cycle) plus thinning."""
    tr = np.where(~fd.held)[0]
    acc = np.zeros(fd.G)
    for i in tr:
        C = fd.real.cells[i]
        rng = np.random.default_rng(MX.stable_seed("floor-dinf", fd.real.panel, fd.real.fold, fd.donors[i]))
        for _ in range(R):
            rows = rng.integers(0, C.shape[0], K)
            acc += MX.log_cp10k(np.asarray(C[rows].sum(0)).ravel().astype(np.float64))
    pred = acc / (len(tr) * R)
    return np.sqrt(((pred - fd.Y) ** 2).mean(1))


def _boot_weights(n_boot, seed, n=37):
    rng = np.random.default_rng(seed)
    return rng.multinomial(n, np.full(n, 1.0 / n), size=n_boot).astype(np.float64)


# --------------------------------------------------------------------------- driver
def run(folds, panel, caches_root, n_boot, with_generator, baseline_root, log=print):
    dev = json.loads(EV.FOLDS_JSON_R1.read_text())["development_donors"]
    idx37 = {d: i for i, d in enumerate(sorted(dev))}
    W = _boot_weights(n_boot, MX.stable_seed("floor-boot", panel))
    res = {"floor_version": FLOOR_VERSION, "panel": panel, "n_boot": n_boot, "cell_boot": CELL_BOOT,
           "definition": __doc__, "per_fold": {}, "pooled": {}}
    boots = {}      # fold -> {key: [B]}
    per_d = {}      # fold -> per-donor arrays for pooling route B / D
    for k in folds:
        t0 = time.monotonic()
        cdir = Path(caches_root) / panel / f"fold{k}"
        real = EV.load_real(cdir, panel, k)
        fd = FoldData(real)
        wmap = np.array([idx37[d] for d in fd.donors])
        point = route_a(fd, np.ones(len(fd.donors)))
        bA = {key: [] for key in ["pb_raw", "pb_grand", "pb_stage", "ml_raw", "ml_grand"]}
        for b in range(n_boot):
            r = route_a(fd, W[b][wmap])
            for key in bA:
                bA[key].append(None if r is None else r.get(key))
        rawb, gb, sb = route_b(fd)
        rc = route_c(fd, gb)
        ml = meanlog_analytic(fd)
        he = fd.held
        f = {"n_train": int((~he).sum()), "n_heldout": int(he.sum()), "n_genes": fd.G,
             "heldout_by_stage": {str(s): int(((fd.stage == s) & he).sum()) for s in range(4)},
             "pb": {"A_raw_train_mean": _summ(point["pb_raw"], bA["pb_raw"]),
                    "A_grand": _summ(point["pb_grand"], bA["pb_grand"]),
                    "A_stage": _summ(point["pb_stage"], bA["pb_stage"]),
                    "A_correction_mse": point["pb_grand_correction_mse"],
                    "B_loo_grand_heldout": {"value": float(np.nanmean(gb[he])), "se": float(np.nanstd(gb[he], ddof=1) / np.sqrt(he.sum()))},
                    "B_loo_stage_heldout": {"value": float(np.nanmean(sb[he])), "se": float(np.nanstd(sb[he], ddof=1) / np.sqrt(np.isfinite(sb[he]).sum()))},
                    "B_loo_raw_heldout": float(np.nanmean(rawb[he])),
                    "B_loo_grand_all37": {"value": float(np.nanmean(gb)), "se": float(np.nanstd(gb, ddof=1) / np.sqrt(len(gb)))},
                    "B_loo_stage_all37": {"value": float(np.nanmean(sb)), "se": float(np.nanstd(sb, ddof=1) / np.sqrt(np.isfinite(sb).sum()))},
                    "C_variance": rc},
             "meanlog": {"A_raw_train_mean": _summ(point["ml_raw"], bA["ml_raw"]),
                         "A_grand": _summ(point["ml_grand"], bA["ml_grand"]),
                         "B_analytic": ml}}
        boots[k] = bA
        per_d[k] = {"donors": fd.donors.tolist(), "held": he.tolist(), "B_grand": gb.tolist(), "B_stage": sb.tolist()}
        if with_generator:
            t1 = time.monotonic()
            perD, gml, info = route_d(fd)
            e = (gml[he] - fd.X[he]).mean(0)
            f["pb"]["D_ideal_generator"] = {"value": float(perD[he].mean()),
                                            "se": float(perD[he].std(ddof=1) / np.sqrt(he.sum())),
                                            "readout_cost_vs_A_raw": float(perD[he].mean() - point["pb_raw"]),
                                            "train_targets_in_sample_value": float(perD[~he].mean()),
                                            "info": info, "seconds": time.monotonic() - t1}
            f["meanlog"]["D_ideal_generator"] = float(np.sqrt((e ** 2).mean()))
            pinf = route_dinf(fd)
            f["pb"]["D_inf_M"] = {"value": float(pinf[he].mean()),
                                  "kcell_readout_cost_vs_A_raw": float(pinf[he].mean() - point["pb_raw"]),
                                  "finite_M_cost_D_minus_Dinf": float(perD[he].mean() - pinf[he].mean())}
            dboot = []
            for b in range(min(n_boot, 1000)):
                wh = W[b][wmap][he]
                if wh.sum() <= 0:
                    dboot.append(None); continue
                dboot.append(float((wh * perD[he]).sum() / wh.sum()))
            f["pb"]["D_ideal_generator"].update({kk: vv for kk, vv in _summ(f["pb"]["D_ideal_generator"]["value"], dboot).items()
                                                 if kk in ("se", "ci95", "boot_n")})
            boots[k]["pb_D"] = dboot
        if baseline_root:
            ref = {}
            for name in ["grand_mean", "ridge", "resample", "resample_blind", "stage_mean_log"]:
                p = Path(baseline_root) / "baselines" / panel / f"fold{k}" / f"{name}.json"
                if p.exists():
                    q = json.loads(p.read_text())
                    ref[name] = q["subsets"]["all"]["heldout"]["pseudobulk"]["rmse"]
                    ref[name + "_meanlog"] = (q["subsets"]["all"]["heldout"].get("mean_log") or {}).get("rmse_gene")
            f["evaluator_baselines_pb_rmse"] = ref
        f["seconds"] = time.monotonic() - t0
        res["per_fold"][str(k)] = f
        log(json.dumps({"fold": k, "A_grand": f["pb"]["A_grand"]["value"], "A_stage": f["pb"]["A_stage"]["value"],
                        "A_raw": f["pb"]["A_raw_train_mean"]["value"], "B_grand": f["pb"]["B_loo_grand_heldout"]["value"],
                        "D": (f["pb"].get("D_ideal_generator") or {}).get("value"), "sec": round(f["seconds"], 1)}))
    # ---- pooled over the folds of each repeat (fold-balanced, like the evaluator) ----
    reps = {}
    for k in folds:
        reps.setdefault(EV.repeat_of(k), []).append(k)
    groups = {f"repeat{r}": ks for r, ks in sorted(reps.items())}
    if len(reps) > 1:
        groups["all_folds"] = list(folds)
    extra = [k for k in folds if EV.repeat_of(k) > 0]
    if extra and len(set(EV.repeat_of(k) for k in extra)) > 1:
        groups["repeats_1_2"] = extra
    for gname, ks in groups.items():
        g = {"folds": ks}
        for key, path in [("pb_A_raw_train_mean", ("pb", "A_raw_train_mean")), ("pb_A_grand", ("pb", "A_grand")),
                          ("pb_A_stage", ("pb", "A_stage")), ("pb_D_ideal_generator", ("pb", "D_ideal_generator")),
                          ("ml_A_raw_train_mean", ("meanlog", "A_raw_train_mean")), ("ml_A_grand", ("meanlog", "A_grand"))]:
            vals = [EV._dig(res["per_fold"][str(k)], path + ("value",)) for k in ks]
            if any(v is None for v in vals):
                continue
            bkey = {"pb_A_raw_train_mean": "pb_raw", "pb_A_grand": "pb_grand", "pb_A_stage": "pb_stage",
                    "pb_D_ideal_generator": "pb_D", "ml_A_raw_train_mean": "ml_raw", "ml_A_grand": "ml_grand"}[key]
            bb = []
            for b in range(n_boot):
                xs = [boots[k].get(bkey, [None] * n_boot)[b] if b < len(boots[k].get(bkey, [])) else None for k in ks]
                bb.append(None if any(x is None for x in xs) else float(np.mean(xs)))
            g[key] = _summ(float(np.mean(vals)), bb)
        for key, path in [("pb_B_loo_grand_heldout", ("pb", "B_loo_grand_heldout", "value")),
                          ("pb_B_loo_stage_heldout", ("pb", "B_loo_stage_heldout", "value")),
                          ("pb_B_loo_grand_all37", ("pb", "B_loo_grand_all37", "value")),
                          ("pb_B_loo_stage_all37", ("pb", "B_loo_stage_all37", "value")),
                          ("pb_C_rms_total", ("pb", "C_variance", "rms_total")),
                          ("pb_C_rms_within_stage", ("pb", "C_variance", "rms_within_stage")),
                          ("pb_C_technical_rms", ("pb", "C_variance", "technical_rms")),
                          ("pb_C_technical_share", ("pb", "C_variance", "technical_share_of_total_variance")),
                          ("pb_floor_without_technical_heldout", ("pb", "C_variance", "floor_without_technical_heldout")),
                          ("ml_B_grand", ("meanlog", "B_analytic", "grand")), ("ml_B_stage", ("meanlog", "B_analytic", "stage")),
                          ("ml_B_grand_depth_adjusted", ("meanlog", "B_analytic", "grand_depth_adjusted")),
                          ("ml_B_stage_depth_adjusted", ("meanlog", "B_analytic", "stage_depth_adjusted")),
                          ("ml_D_ideal_generator", ("meanlog", "D_ideal_generator")),
                          ("pb_D_inf_M", ("pb", "D_inf_M", "value"))]:
            vals = [EV._dig(res["per_fold"][str(k)], path) for k in ks]
            if all(v is not None for v in vals):
                g[key] = {"value": float(np.mean(vals)), "per_fold": vals,
                          "se_folds": float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else None}
        refs = [res["per_fold"][str(k)].get("evaluator_baselines_pb_rmse") or {} for k in ks]
        for name in ["grand_mean", "ridge", "resample", "resample_blind", "stage_mean_log"]:
            v = [r.get(name) for r in refs]
            if all(x is not None for x in v):
                g["evaluator_" + name] = float(np.mean(v))
        if g.get("evaluator_grand_mean") is not None and g.get("pb_A_grand"):
            g["headroom_below_grand_mean_baseline"] = {
                "to_oracle_grand_floor_A": g["evaluator_grand_mean"] - g["pb_A_grand"]["value"],
                "to_oracle_stage_floor_A": (g["evaluator_grand_mean"] - g["pb_A_stage"]["value"]) if g.get("pb_A_stage") else None,
                "to_ideal_generator_D": (g["evaluator_grand_mean"] - g["pb_D_ideal_generator"]["value"]) if g.get("pb_D_ideal_generator") else None}
        res["pooled"][gname] = g
    res["per_donor"] = per_d
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folds", default="0,1,2,3,4,5,6,7,8")
    ap.add_argument("--panel", default="hvg2000")
    ap.add_argument("--caches-root", default=str(RUNS / "caches"))
    ap.add_argument("--out", default=str(RUNS / "results_r3" / "floor.json"))
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--no-generator", action="store_true")
    ap.add_argument("--baseline-root", default=None)
    a = ap.parse_args(argv)
    EV._check_out_dir(a.out)
    t0 = time.monotonic()
    res = run([int(x) for x in a.folds.split(",")], a.panel, a.caches_root, a.boot, not a.no_generator, a.baseline_root)
    res["seconds"] = time.monotonic() - t0
    EV.write_json(a.out, res)
    print(json.dumps({"wrote": a.out, "seconds": res["seconds"]}))
    return res


if __name__ == "__main__":
    main()
