"""Baselines and noise ceilings for improve_20261109 (Interface 3).

Each baseline returns an in-memory dict with exactly the generated.npz layout
of Interface 2 (targets, is_heldout, stage, genes, pb_counts [T,M,G],
mean_log [T,M,G], nonzero [T,M,G], pcs [T,M,64,50] float16, lib [T,M]) for all
T = train + heldout donors of the fold, M = 32 generated donors x K = 256
cells, library sizes drawn with replacement from the TARGET donor's own
observed panel library sizes (depth-matched). They are scored by
evaluate.score_generated, the same function used for fits. Only train-donor
data is used to fit any baseline; held-out data enters only as the target's
library-size distribution (the same information a fit receives) and its
covariates (ridge only, as the `covars` arm receives).

Profile baselines (grand_mean, stage_mean, stage_mean_log, shrunk_stage_mean,
ridge) predict a per-target log1p(CP10k) pseudobulk profile
mu_t. To make distribution metrics defined they generate donors with a
documented simple count model:
  * between-donor variability: generated donor m of target t uses profile
    mu_t + r_j, where r_j = L_j - fitted_j is the residual of train donor j
    under the same baseline (non-parametric residual bootstrap; j cycles over
    a seeded permutation of train donors, excluding t itself when t is a
    train donor). The M drawn residuals are re-centred per target on the mean
    residual of all train donors (v4, integration fix): r_j -> r_j - mean_m(r_jm)
    + mean_j(r_j). Before, the leave-self-out exclusion shifted every train
    target by -r_t/(n-1) (grand_mean train-target contrast Spearman -0.64..-0.80
    vs the real train contrast) and the 7 repeated draws (32 over ~25 donors)
    added a random per-target offset. (v3 centred on 0, which also removed the
    systematic fitted-vs-donor-log level difference and made the baselines
    worse, e.g. grand_mean held-out pb RMSE 0.464 -> 0.500; not kept.)
    Proportions q = expm1(max(mu_t + r_j, 0)), normalised.
  * cells: independent-gene negative binomial (gamma-Poisson) with mean
    lib_i * q_g and per-gene dispersion phi_g estimated by method of moments
    within train donors (phi_g = max(0, sum[(x-mu)^2 - mu]) / sum mu^2,
    mu = lib_i * q_dg with q_dg the donor's pseudobulk proportion; clipped at
    50). No cell-state structure -> PC-space metrics are NOT meaningful for
    these baselines (see MEANINGFUL).
Resampling baseline (5): generated donor m of target t is a real train donor
of the target's stage (cycling over a seeded permutation, excluding t itself;
falls back to all train donors if the stage has no other train donor). For
each of K library sizes drawn from the target donor, a source cell is drawn
uniformly among that source donor's cells whose library is >= the drawn
library and binomially thinned with p = L_target / L_source (depth-qualified
selection). If no source cell is deep enough, the deepest cell is used
unthinned (fraction reported). Caveat: selection favours deeper source cells,
so cell-state composition is biased if depth correlates with state. (A first
version paired cells uniformly and thinned only downward; the synthetic test
showed that systematically lowers depth -> inflated PC spread / nonzero error.)
Generation randomness depends only on (panel, fold, target donor) so all
baselines share library-size draws (common random numbers).
"""
from __future__ import annotations

import os
for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(_v, "6")

import numpy as np

import metrics as MX

BASELINE_VERSION = "improve_20261109-baselines-v5"
BASELINES = ["grand_mean", "stage_mean", "stage_mean_log", "shrunk_stage_mean", "ridge", "resample"]
M_DEFAULT, K_DEFAULT, KPC_DEFAULT = 32, 256, 64
RIDGE_ALPHAS = np.logspace(-2, 4, 13)

_PROFILE = {"pseudobulk": "meaningful", "contrast": "meaningful",
            "mean_log/nonzero": "count-model dependent (independent-gene NB, train dispersion); interpret as a count-model check",
            "distribution_pb": "partially meaningful (between-donor spread = bootstrapped train residuals)",
            "pc_space/cell_energy": "NOT meaningful (cells lack cell-state covariance structure)",
            "seed_stability": "not applicable (deterministic fit)"}
MEANINGFUL = {
    "grand_mean": {**_PROFILE, "contrast": "null reference: no stage information; model contrast is residual-bootstrap noise"},
    "stage_mean": dict(_PROFILE),
    "stage_mean_log": dict(_PROFILE),
    "shrunk_stage_mean": dict(_PROFILE),
    "ridge": dict(_PROFILE),
    "resample": {"pseudobulk": "meaningful", "contrast": "meaningful", "mean_log/nonzero": "meaningful (depth thinning downward only)",
                 "distribution_pb": "meaningful", "pc_space/cell_energy": "meaningful",
                 "seed_stability": "not applicable"},
}


# --------------------------------------------------------------------------
# Train-only quantities
# --------------------------------------------------------------------------
def train_profiles(real):
    tr = ~real.is_heldout
    L = MX.log_cp10k(real.pb[tr])
    P = real.pb[tr] / np.maximum(real.pb[tr].sum(1, keepdims=True), 1)
    return tr, L, P


def estimate_dispersion(real, cap=50.0):
    cached = getattr(real, "_bl_dispersion", None)
    if cached is not None:
        return cached
    phi = _estimate_dispersion(real, cap)
    real._bl_dispersion = phi
    return phi


def _estimate_dispersion(real, cap):
    G = len(real.genes)
    num = np.zeros(G); den = np.zeros(G)
    for i in np.where(~real.is_heldout)[0]:
        X = real.cells[i].toarray().astype(np.float64)
        s = X.sum(1)
        q = real.pb[i] / max(real.pb[i].sum(), 1)
        mu = s[:, None] * q[None, :]
        num += ((X - mu) ** 2 - mu).sum(0)
        den += (mu ** 2).sum(0)
    phi = np.where(den > 0, num / np.maximum(den, 1e-300), 0.0)
    return np.clip(phi, 0.0, cap)


def _stage_means(X, stage, stages=range(4)):
    return {s: X[stage == s].mean(0) for s in stages if (stage == s).any()}


def fit_profile_baseline(name, real):
    """Returns mu [T,G] predicted log1p(CP10k) per target, fitted_train
    [n_train,G], info dict."""
    tr, L, P = train_profiles(real)
    st_tr = real.stage[tr]
    T = len(real.donors)
    info = {}
    if name == "grand_mean":
        pooled = MX.log_cp10k(real.pb[tr].sum(0))
        mu = np.tile(pooled, (T, 1)); fitted = np.tile(pooled, (tr.sum(), 1))
        info["definition"] = "log1p CP10k of pooled train counts; no stage"
    elif name == "stage_mean":
        gp = P.mean(0)
        sm = _stage_means(P, st_tr)
        missing = [s for s in range(4) if s not in sm]
        prof = {s: np.log1p(1e4 * sm.get(s, gp)) for s in range(4)}
        mu = np.stack([prof[s] for s in real.stage]); fitted = np.stack([prof[s] for s in st_tr])
        info.update(definition="log1p(1e4 * mean of train-donor pseudobulk proportions within stage)",
                    stages_missing_in_train_fallback_grand=missing)
    elif name == "stage_mean_log":
        # v5 (fixer): the honest unshrunk stage-mean reference. stage_mean above is a log-of-mean
        # (log1p of the mean proportion), which is biased against donor log profiles and is beaten
        # even by the no-stage grand_mean; this is the mean of the donor log profiles.
        grand = L.mean(0)
        sm = _stage_means(L, st_tr)
        missing = [s for s in range(4) if s not in sm]
        prof = {s: sm.get(s, grand) for s in range(4)}
        mu = np.stack([prof[s] for s in real.stage]); fitted = np.stack([prof[s] for s in st_tr])
        info.update(definition="mean of train-donor log1p CP10k pseudobulk profiles within stage (mean of logs)",
                    stages_missing_in_train_fallback_grand=missing)
    elif name == "shrunk_stage_mean":
        grand = L.mean(0)
        sm = _stage_means(L, st_tr)
        S = list(sm)
        n_s = {s: int((st_tr == s).sum()) for s in S}
        resid_ss = sum(((L[st_tr == s] - sm[s]) ** 2).sum(0) for s in S)
        dof = max(len(L) - len(S), 1)
        sigma2 = resid_ss / dof
        delta = {s: sm[s] - grand for s in S}
        tau2 = np.maximum(0.0, np.mean([delta[s] ** 2 - sigma2 / n_s[s] for s in S], axis=0))
        B = {s: np.where(tau2 + sigma2 / n_s[s] > 0, tau2 / (tau2 + sigma2 / n_s[s]), 0.0) for s in S}
        prof = {s: grand + (B[s] * delta[s] if s in S else 0.0) for s in range(4)}
        mu = np.stack([prof[s] for s in real.stage]); fitted = np.stack([prof[s] for s in st_tr])
        info.update(definition="per-gene empirical-Bayes shrinkage of train stage deviations (log1p CP10k donor profiles) "
                               "toward the donor-weighted grand mean: B_sg = tau2_g/(tau2_g+sigma2_g/n_s), "
                               "tau2_g = max(0, mean_s(delta_sg^2 - sigma2_g/n_s)), sigma2_g pooled within-stage variance",
                    mean_shrinkage_factor={str(s): float(B[s].mean()) for s in S},
                    frac_genes_fully_shrunk=float((tau2 == 0).mean()))
    elif name == "ridge":
        X_all = ridge_design(real)
        Xtr = X_all[tr]
        alpha, loo = choose_alpha(Xtr, L)
        B, b0, xm, xs = ridge_fit(Xtr, L, alpha)
        mu = ((X_all - xm) / xs) @ B + b0
        fitted = ((Xtr - xm) / xs) @ B + b0
        info.update(definition="multi-output ridge of train log1p CP10k profiles on stage one-hot(Low,Int,High)+stage/3+Male+"
                               "APOE4 dose+standardised age (columns standardised within each fit; intercept unpenalised); "
                               "single alpha chosen by leave-one-train-donor-out MSE",
                    alpha=float(alpha), loo_mse={f"{a:.4g}": float(v) for a, v in zip(RIDGE_ALPHAS, loo)})
    else:
        raise ValueError(name)
    return mu, fitted, info


def ridge_design(real):
    st = real.stage
    onehot = np.stack([(st == s).astype(float) for s in (1, 2, 3)], 1)
    age = (real.raw_c[:, 6] - real.age_mean) / real.age_sd
    return np.column_stack([onehot, st / 3.0, real.raw_c[:, 4], real.raw_c[:, 5], age])


def ridge_fit(X, Y, alpha):
    xm = X.mean(0); xs = X.std(0); xs = np.where(xs > 0, xs, 1.0)
    Z = (X - xm) / xs
    ym = Y.mean(0)
    B = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (Y - ym))
    return B, ym, xm, xs


def choose_alpha(X, Y):
    n = len(X); loo = []
    for a in RIDGE_ALPHAS:
        err = 0.0
        for i in range(n):
            keep = np.arange(n) != i
            B, b0, xm, xs = ridge_fit(X[keep], Y[keep], a)
            pred = ((X[i] - xm) / xs) @ B + b0
            err += float(((pred - Y[i]) ** 2).mean())
        loo.append(err / n)
    return RIDGE_ALPHAS[int(np.argmin(loo))], loo


# --------------------------------------------------------------------------
# Cell generation helpers
# --------------------------------------------------------------------------
def _summarise(C, real, kpc):
    """C [K,G] counts -> pb, mean_log, nonzero, pcs(first kpc), mean lib."""
    L = MX.log_cp10k(C)
    pcs = (((L[:kpc] - real.mean) / real.sd) @ real.W / real.pc_sd)
    return C.sum(0), L.mean(0), (C > 0).mean(0), pcs, C.sum(1).mean()


def _alloc(T, M, G, kpc, npc):
    return {"pb_counts": np.zeros((T, M, G), np.float32), "mean_log": np.zeros((T, M, G), np.float32),
            "nonzero": np.zeros((T, M, G), np.float32), "pcs": np.zeros((T, M, kpc, npc), np.float16),
            "lib": np.zeros((T, M), np.float32)}


def _donor_cycle(rng, pool, M):
    pool = np.asarray(pool)
    perm = rng.permutation(pool)
    return np.array([perm[m % len(perm)] for m in range(M)])


def _target_rng(real, t):
    return np.random.default_rng(MX.stable_seed("baseline-gen", real.panel, real.fold, real.donors[t]))


def generate(name, real, M=M_DEFAULT, K=K_DEFAULT, kpc=KPC_DEFAULT):
    T, G = len(real.donors), len(real.genes)
    npc = real.W.shape[1]
    out = _alloc(T, M, G, kpc, npc)
    tr_idx = np.where(~real.is_heldout)[0]
    info = {"M": M, "K": K, "kpc": kpc}
    if name == "resample":
        fallback = []; unmatched = 0; total = 0
        for t in range(T):
            rng = _target_rng(real, t)
            libs = rng.choice(real.libs[t], (M, K), replace=True)
            pool = [i for i in tr_idx if real.stage[i] == real.stage[t] and i != t]
            if not pool:
                pool = [i for i in tr_idx if i != t]; fallback.append(str(real.donors[t]))
            src = _donor_cycle(rng, pool, M)
            for m in range(M):
                cells = real.cells[src[m]]
                slib = real.libs[src[m]]
                order = np.argsort(slib, kind="stable"); sorted_lib = slib[order]
                # depth-qualified selection: for each drawn target library L,
                # a uniform source cell among those with library >= L
                first = np.searchsorted(sorted_lib, libs[m], side="left")
                n_ok = len(slib) - first
                pick = np.where(n_ok > 0, first + (rng.random(K) * np.maximum(n_ok, 1)).astype(int), len(slib) - 1)
                rows = order[np.minimum(pick, len(slib) - 1)]
                unmatched += int((n_ok == 0).sum()); total += K
                X = cells[rows].toarray().astype(np.float64)
                sl = X.sum(1)
                p = np.minimum(1.0, libs[m] / np.maximum(sl, 1))
                nzr, nzc = np.nonzero(X)
                X[nzr, nzc] = rng.binomial(X[nzr, nzc].astype(np.int64), p[nzr])
                pb, ml, nz, pcs, lib = _summarise(X, real, kpc)
                out["pb_counts"][t, m] = pb; out["mean_log"][t, m] = ml; out["nonzero"][t, m] = nz
                out["pcs"][t, m] = pcs; out["lib"][t, m] = lib
        info["stage_fallback_targets"] = fallback
        info["frac_cells_depth_unmatched"] = unmatched / max(total, 1)
        info["depth_rule"] = ("per drawn target library L: uniform source cell among the source donor's cells with library >= L, "
                              "binomially thinned with p = L / L_source; if none qualifies, the highest-library cell unthinned")
    else:
        mu, fitted, finfo = fit_profile_baseline(name, real)
        info.update(finfo)
        resid = MX.log_cp10k(real.pb[tr_idx]) - fitted
        phi = estimate_dispersion(real)
        info["dispersion_median"] = float(np.median(phi)); info["dispersion_frac_zero"] = float((phi == 0).mean())
        theta = np.where(phi > 0, 1.0 / np.maximum(phi, 1e-12), np.inf)
        has_od = np.isfinite(theta)
        for t in range(T):
            rng = _target_rng(real, t)
            libs = rng.choice(real.libs[t], (M, K), replace=True)
            pool = [j for j, i in enumerate(tr_idx) if i != t]
            js = _donor_cycle(rng, pool, M)
            # Re-centre the drawn residuals on the mean residual of ALL train donors, so every target
            # gets the same gene-wise residual mean (which carries the systematic level difference
            # between the fitted profile and the donor log profiles) with no leave-self-out shift
            # (-r_t/(n-1)) and no random-repeat shift (32 draws over ~25 donors).
            rdraw = resid[js] - resid[js].mean(0) + resid.mean(0)
            for m in range(M):
                lp = mu[t] + rdraw[m]
                q = np.expm1(np.maximum(lp, 0.0)); q = q / q.sum()
                lam = libs[m][:, None] * q[None, :]
                if has_od.any():
                    g = np.ones_like(lam)
                    g[:, has_od] = rng.standard_gamma(np.broadcast_to(theta[has_od], (K, has_od.sum()))) / theta[has_od]
                    lam = lam * g
                X = rng.poisson(lam).astype(np.float64)
                pb, ml, nz, pcs, lib = _summarise(X, real, kpc)
                out["pb_counts"][t, m] = pb; out["mean_log"][t, m] = ml; out["nonzero"][t, m] = nz
                out["pcs"][t, m] = pcs; out["lib"][t, m] = lib
    out.update(targets=real.donors.copy(), is_heldout=real.is_heldout.copy(), stage=real.stage.copy(), genes=real.genes.copy())
    return out, info


# --------------------------------------------------------------------------
# Noise ceilings (real data only)
# --------------------------------------------------------------------------
def noise_ceiling(real, subsets):
    """Peer-variability reference (historical name 'noise ceiling'; NOT a ceiling)."""
    held = real.is_heldout; tr = ~held
    out = {"role": ("PEER-VARIABILITY REFERENCE, not a performance ceiling: predictions here are full-depth observed peer "
                    "profiles while model/baseline predictions are means of 256-cell generated pseudobulks, and a no-stage "
                    "baseline can score better (e.g. hvg2000 fold 0: grand_mean 0.446 vs heldout peers 0.532 RMSE)."),
           "definition": {
        "heldout_peers_same_stage": "predict each held-out donor's pseudobulk log profile by the mean log profile of the OTHER "
                                    "held-out donors of the same stage (undefined when it is the only one)",
        "all_dev_peers_same_stage": "same, using all other development donors of this fold (train + held-out) of the same stage",
        "train_stage_mean_log": "mean log profile of train donors of the same stage (count-noise-free stage-mean predictor)",
        "split_half": "split-half reliability of the real High-minus-NotAD contrast (200 random within-stage halvings)",
        "train_vs_heldout_real_contrast": "agreement of the real train contrast with the real held-out contrast"},
        "subsets": {}}
    for sname, (idx, renorm) in subsets.items():
        Lall = MX.log_cp10k(real.pb[:, idx]) if renorm else MX.log_cp10k(real.pb)[:, idx]
        grand = Lall[tr].mean(0)
        block = {}
        for label, pool_mask in [("heldout_peers_same_stage", held), ("all_dev_peers_same_stage", np.ones_like(held)),
                                 ("train_stage_mean_log", tr)]:
            preds, obs, st, dn = [], [], [], []
            for i in np.where(held)[0]:
                peers = np.where(pool_mask & (real.stage == real.stage[i]) & (np.arange(len(held)) != i))[0]
                if len(peers) == 0:
                    continue
                preds.append(Lall[peers].mean(0)); obs.append(Lall[i]); st.append(real.stage[i]); dn.append(real.donors[i])
            if preds:
                r = MX.pseudobulk_metrics(np.stack(preds), np.stack(obs), grand, np.array(st), np.array(dn))
                r["n_heldout_without_peer"] = int(held.sum() - len(preds))
                block[label] = r
            else:
                block[label] = None
        block["split_half"] = {"heldout": MX.split_half_reliability(Lall[held], real.stage[held]),
                               "train": MX.split_half_reliability(Lall[tr], real.stage[tr]),
                               "all_dev": MX.split_half_reliability(Lall, real.stage)}
        ch = MX.stage_contrasts(Lall[held], real.stage[held]); ct = MX.stage_contrasts(Lall[tr], real.stage[tr])
        block["train_vs_heldout_real_contrast"] = {"high_notad": MX.agreement(ct["high_notad"], ch["high_notad"]),
                                                   "slope": MX.agreement(ct["slope"], ch["slope"]),
                                                   "heldout_low_n": ch["low_n_high_notad"], "heldout_n_by_stage": ch["n_by_stage"]}
        out["subsets"][sname] = block
    return out
