"""(c) Donor-level variance components and how contrast reliability scales with donors.

All 37 development donors, genes common to the 3 fold caches of each panel, the
evaluator's profile (log1p CP10k of the donor pseudobulk).

Per gene g:
  v_g    technical variance of a donor profile: cell resampling (bootstrap over the
         donor's cells, B=100, includes Poisson/depth noise because cells are resampled
         with their own counts), averaged over donors;
  s2_g   pooled within-stage variance of donor profiles (ddof per stage);
  tau2_g = max(s2_g - v_g, 0)  between-donor biological variance within stage.
Contrast noise for a design with n0 Not AD and n3 High donors and a cells multiplier m:
  N_g = (tau2_g + v_g/m) * (1/n0 + 1/n3).
Signal variance across genes S = Var_g(delta_g) is estimated two ways on the 37 donors:
  S_mom   = Var_g(c_hat) - mean_g N_g(4,19)       (method of moments)
  S_split = mean over stage-stratified random halves of Cov_g(c_A, c_B) (needs no noise model)
with a permutation null for S_split (stage labels shuffled over the 37 donors) and a
leave-one-NotAD-out jackknife of S_mom.
Projection: delta_g ~ N(0, S) (and a heavy-tailed t3 variant), contrast estimates
c = delta + N(0, N_g); expected Spearman and top-decile sign agreement are simulated.
The projection is validated against the observed per-fold train-vs-held-out agreement.
Output: OUT/c_power.json, OUT/c_components_{panel}.npz (per-gene, small), printed tables.
"""
from stage_common import *

rng = np.random.default_rng(20261111)
B = 100


def components(c):
    P = log_cp10k(c["pb"]); st = c["stage"]
    G = P.shape[1]
    v = np.zeros((len(P), G))
    for i, C in enumerate(c["cells"]):
        X = C.toarray().astype(np.float64)
        n = X.shape[0]
        W = rng.multinomial(n, np.full(n, 1 / n), size=B).astype(np.float64)
        v[i] = log_cp10k(W @ X).var(0, ddof=1)
    ss, df = np.zeros(G), 0
    for s in range(4):
        m = st == s
        if m.sum() > 1:
            ss += ((P[m] - P[m].mean(0)) ** 2).sum(0); df += m.sum() - 1
    s2 = ss / df
    s2_by = [P[st == s].var(0, ddof=1) for s in range(4)]
    return P, st, v, s2, s2_by


def S_split(P, st, nsplit, rng):
    cv = []
    for _ in range(nsplit):
        A, Bh = [], []
        for s in range(4):
            idx = rng.permutation(np.where(st == s)[0]); h = len(idx) // 2
            if len(idx) % 2 and rng.random() < .5:
                h += 1
            A += idx[:h].tolist(); Bh += idx[h:].tolist()
        A, Bh = np.array(A), np.array(Bh)
        ca, cb = contrasts(P[A], st[A])["hn"], contrasts(P[Bh], st[Bh])["hn"]
        cv.append(((ca - ca.mean()) * (cb - cb.mean())).mean())
    return float(np.mean(cv))


def simulate(S, noise_t, noise_h, nsim=200, heavy=False):
    """Expected Spearman, Pearson and top-decile sign agreement (top decile of |c_t|)
    between two independent contrast estimates with per-gene noise variances."""
    G = len(noise_t)
    out = []
    for _ in range(nsim):
        if heavy:
            d = rng.standard_t(3, G) * np.sqrt(S / 3.0)
        else:
            d = rng.normal(0, np.sqrt(S), G)
        ct = d + rng.normal(0, 1, G) * np.sqrt(noise_t)
        ch = d + rng.normal(0, 1, G) * np.sqrt(noise_h)
        k = int(np.ceil(.1 * G)); top = np.argsort(-np.abs(ct))[:k]
        out.append([spearman(ct, ch), pearson(ct, ch), (np.sign(ct[top]) == np.sign(ch[top])).mean(),
                    spearman(ct, d)])
    return np.array(out).mean(0)


res = {}
for panel in ["hvg2000", "wide"]:
    c = all37(panel, keep_cells=True)
    P, st, v, s2, s2_by = components(c)
    vbar = v.mean(0)
    tau2 = np.maximum(s2 - vbar, 0)
    G = P.shape[1]
    chat = contrasts(P, st)["hn"]
    n0, n3 = int((st == 0).sum()), int((st == 3).sum())
    N_now = (tau2 + vbar) * (1 / n0 + 1 / n3)
    S_mom = float(chat.var() - N_now.mean())
    # jackknife over Not AD donors
    jk = []
    for i in np.where(st == 0)[0]:
        keep = np.arange(len(st)) != i
        ci = contrasts(P[keep], st[keep])["hn"]
        jk.append(float(ci.var() - ((tau2 + vbar) * (1 / (n0 - 1) + 1 / n3)).mean()))
    Ssp = S_split(P, st, 400, rng)
    null = np.array([S_split(P, rng.permutation(st), 20, rng) for _ in range(200)])
    R = res[panel] = {
        "G": G, "n_by_stage": np.bincount(st, minlength=4).tolist(),
        "cells_per_donor_by_stage_median": [float(np.median(c["ncells"][st == s])) for s in range(4)],
        "panel_lib_mean_by_stage_median": [float(np.median(c["libmean"][st == s])) for s in range(4)],
        "mean_tech_var": float(vbar.mean()), "mean_within_stage_var": float(s2.mean()),
        "mean_tau2": float(tau2.mean()), "tech_fraction_of_within_stage_var": float(vbar.mean() / s2.mean()),
        "median_gene_tech_fraction": float(np.median(vbar / np.maximum(s2, 1e-12))),
        "frac_genes_tech_gt_within": float((vbar > s2).mean()),
        "within_stage_var_mean_by_stage": [float(x.mean()) if np.isfinite(x).all() else None for x in s2_by],
        "var_chat": float(chat.var()), "mean_noise_now": float(N_now.mean()),
        "S_mom": S_mom, "S_mom_jackknife_notad": jk, "S_split": Ssp,
        "S_split_null_mean": float(null.mean()), "S_split_null_sd": float(null.std()),
        "S_split_null_q95": float(np.quantile(null, .95)),
        "S_split_perm_p": float((1 + (null >= Ssp).sum()) / (1 + len(null))),
    }
    S_c = max(np.mean([S_mom, Ssp]), 1e-12)
    S_hi = max(Ssp + 1.645 * null.std(), S_mom, Ssp)
    S_lo = max(Ssp - 1.645 * null.std(), 0.0)
    R["S_central"], R["S_low"], R["S_high"] = float(S_c), float(S_lo), float(S_hi)
    R["signal_to_noise_now"] = float(S_c / N_now.mean())
    np.savez_compressed(OUT / f"c_components_{panel}.npz", genes=c["genes"], vbar=vbar.astype(np.float32),
                        s2=s2.astype(np.float32), tau2=tau2.astype(np.float32), chat=chat.astype(np.float32),
                        grand=P.mean(0).astype(np.float32))

    def noise(n0_, n3_, m=1.0, t=tau2):
        return (t + vbar / m) * (1 / n0_ + 1 / n3_)

    # validation against observed per-fold train vs held-out agreement (from stage_a output if present)
    val = []
    for f in range(3):
        cf = load_cache(panel, f)
        sh, stf = cf["is_heldout"], cf["stage"]
        nt0, nt3 = int(((stf == 0) & ~sh).sum()), int(((stf == 3) & ~sh).sum())
        nh0, nh3 = int(((stf == 0) & sh).sum()), int(((stf == 3) & sh).sum())
        e = simulate(S_c, noise(nt0, nt3), noise(nh0, nh3), 100)
        el = simulate(max(S_lo, 1e-12), noise(nt0, nt3), noise(nh0, nh3), 100)
        eh = simulate(S_hi, noise(nt0, nt3), noise(nh0, nh3), 100)
        val.append({"fold": f, "train_n0_n3": [nt0, nt3], "heldout_n0_n3": [nh0, nh3],
                    "pred_spearman": float(e[0]), "pred_spearman_low": float(el[0]), "pred_spearman_high": float(eh[0]),
                    "pred_top10_sign": float(e[2])})
    R["validation_folds"] = val
    R["pred_split_half_37"] = float(simulate(S_c, noise(2, 9.5), noise(2, 9.5), 100)[0])

    # projection table
    table = []
    for n0_ in [1, 2, 4, 6, 9, 10, 20, 40, 80]:
        for lab, n3_ in [("High=19", 19), ("High=2xNotAD", max(2 * n0_, 2)), ("High=NotAD", n0_)]:
            row = {"n_notad": n0_, "high_rule": lab, "n_high": n3_}
            for sname, S in [("low", max(S_lo, 1e-12)), ("central", S_c), ("high", S_hi)]:
                e = simulate(S, noise(n0_, n3_), noise(n0_, n3_), 60)
                row[f"rep_spearman_{sname}"] = float(e[0])        # two independent same-size studies
                row[f"acc_spearman_{sname}"] = float(e[3])        # estimate vs truth
                row[f"rep_top10_{sname}"] = float(e[2])
            e = simulate(S_c, noise(n0_, n3_), noise(n0_, n3_), 60, heavy=True)
            row["rep_spearman_central_t3"] = float(e[0]); row["acc_spearman_central_t3"] = float(e[3])
            e = simulate(S_c, noise(n0_, n3_, m=2.0), noise(n0_, n3_, m=2.0), 60)
            row["rep_spearman_central_2xcells"] = float(e[0])
            # train with this design, evaluate against a held-out set like today's (1 NotAD, 6 High)
            e = simulate(S_c, noise(n0_, n3_), noise(1, 6), 60)
            row["heldout_1v6_spearman_central"] = float(e[0])
            # ... against a held-out set of 1/3 of the design size
            h0, h3 = max(1, round(n0_ / 2)), max(1, round(n3_ / 2))
            e = simulate(S_c, noise(n0_, n3_), noise(h0, h3), 60)
            row["heldout_half_spearman_central"] = float(e[0]); row["heldout_half_n"] = [h0, h3]
            table.append(row)
    R["table"] = table

(OUT / "c_power.json").write_text(json.dumps(res, indent=1, default=float))

for panel, R in res.items():
    print(f"\n## {panel}: G={R['G']} n by stage {R['n_by_stage']}")
    for k in ["cells_per_donor_by_stage_median", "panel_lib_mean_by_stage_median", "mean_tech_var", "mean_within_stage_var",
              "mean_tau2", "tech_fraction_of_within_stage_var", "median_gene_tech_fraction", "frac_genes_tech_gt_within",
              "within_stage_var_mean_by_stage", "var_chat", "mean_noise_now", "S_mom", "S_mom_jackknife_notad", "S_split",
              "S_split_null_mean", "S_split_null_sd", "S_split_null_q95", "S_split_perm_p", "S_low", "S_central", "S_high",
              "signal_to_noise_now", "pred_split_half_37"]:
        print(f"  {k}: {R[k]}")
    for v in R["validation_folds"]:
        print(f"  validation {v}")
    print("| Not AD | High | rep Spearman low/central/high | acc vs truth low/central/high | rep top10 sign (central) | rep central t3 | rep central 2x cells | vs held-out 1/6 | vs held-out half (n) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in R["table"]:
        print(f"| {r['n_notad']} | {r['n_high']} ({r['high_rule']}) | {fmt(r['rep_spearman_low'],2)}/{fmt(r['rep_spearman_central'],2)}/{fmt(r['rep_spearman_high'],2)} | "
              f"{fmt(r['acc_spearman_low'],2)}/{fmt(r['acc_spearman_central'],2)}/{fmt(r['acc_spearman_high'],2)} | {fmt(r['rep_top10_central'],2)} | "
              f"{fmt(r['rep_spearman_central_t3'],2)} | {fmt(r['rep_spearman_central_2xcells'],2)} | {fmt(r['heldout_1v6_spearman_central'],2)} | {fmt(r['heldout_half_spearman_central'],2)} {r['heldout_half_n']} |")
