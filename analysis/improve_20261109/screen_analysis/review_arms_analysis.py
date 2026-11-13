"""Review: arm effects, ensemble decomposition, seed projection, panels,
seeds-needed and a covars donor-identity check, from this review's OWN
recomputed metrics (review_arms_recompute.json / review_arms_profiles_*.npz),
not from the analyst's outputs or the evaluator JSONs.

Writes OUT/review_arms_analysis.json and prints markdown to stdout
(OUT/review_arms_analysis.md).
"""
from review_arms_common import *  # noqa

R = json.loads((OUT / "review_arms_recompute.json").read_text())
MET = ["pb_rmse", "meanlog_rmse_gene", "nonzero_rmse_gene", "hn_spearman_vs_heldout", "hn_spearman_vs_train",
       "pc_energy", "pc_coverage80", "cell_energy"]
res = {}
L = []


def good(m, v):
    k = LOWER[m]
    return -abs(v - .8) if k == "cov80" else (-v if k else v)


def arr(src, arm, panel, split, m):
    return np.array([[src[f"{arm}|{panel}|{f}|{s}"][split][m] for s in range(2)] for f in range(3)], float)


def earr(name, arm, panel, split, m):
    return np.array([R[name][f"{arm}|{panel}|{f}"][split][m] for f in range(3)], float)


# ---------------------------------------------------------------- A. effects
L.append("## A. Arm minus base (own recomputation)\n")
L.append("fold diffs = seed-averaged arm-base per fold; crossed = within-fold (arm seed, base seed) wins of 12; "
         "sd = pooled within-fold seed SD base/arm; ens = 2-seed ensemble diff per fold (M=64).\n")
for panel, arms in ARMS.items():
    for split in ["heldout", "train"]:
        L.append(f"\n### {panel} / {split}\n\n| arm | metric | base | arm | diff | fold diffs | crossed | sd base/arm | ens fold diffs |\n|---|---|---|---|---|---|---|---|---|")
        for arm in arms[1:]:
            for m in MET:
                b = arr(R["fits"], "base", panel, split, m); a = arr(R["fits"], arm, panel, split, m)
                if np.isnan(a).any() or np.isnan(b).any():
                    continue
                ga = np.vectorize(lambda v: good(m, v))(a); gb = np.vectorize(lambda v: good(m, v))(b)
                cr = int(sum(ga[f, i] > gb[f, j] for f in range(3) for i in range(2) for j in range(2)))
                sdb = np.sqrt(np.mean((b[:, 0] - b[:, 1]) ** 2) / 2); sda = np.sqrt(np.mean((a[:, 0] - a[:, 1]) ** 2) / 2)
                fd = (a - b).mean(1)
                ed = earr("ens", arm, panel, split, m) - earr("ens", "base", panel, split, m)
                res[f"A|{arm}|{panel}|{split}|{m}"] = dict(base=b.mean(), arm=a.mean(), fold=fd.tolist(), crossed=cr,
                                                           sd_base=sdb, sd_arm=sda, ens_fold=ed.tolist(),
                                                           base_ens=earr("ens", "base", panel, split, m).mean(),
                                                           arm_ens=earr("ens", arm, panel, split, m).mean())
                L.append(f"| {arm} | {m} | {b.mean():.3f} | {a.mean():.3f} | {(a - b).mean():+.3f} | "
                         + "/".join(f"{x:+.3f}" for x in fd) + f" | {cr}/12 | {sdb:.3f}/{sda:.3f} | "
                         + "/".join(f"{x:+.3f}" for x in ed) + " |")

# ------------------------------------------- B. ensemble decomposition (pb)
L.append("\n## B. Seed-ensemble decomposition, held-out pb (per fold)\n")
L.append("D = mean (p0-p1)^2; MCind = V0+V1 (expected D from M=32 MC noise if draws were independent); "
         "crn = correlation of per-generated-donor log-pb deviations at the same m index across the two seeds "
         "(common random numbers); rmse: single (seed mean) / ens16+16 sharing CRN draws / ens16+16 distinct draws / ens M64.\n")
L.append("| arm | panel | fold | D/MCind | crn corr | single | mm_crn | mm_indep | ens64 | ens64 from identity |\n|---|---|---|---|---|---|---|---|---|---|")
proj = {}
for panel, arms in ARMS.items():
    for f in range(3):
        z = np.load(OUT / f"review_arms_profiles_{panel}_fold{f}.npz")
        held = z["held"]; obs = z["obs"][held].astype(float)
        for arm in arms:
            p0 = z[f"{arm}|0|P"][held].astype(float); p1 = z[f"{arm}|1|P"][held].astype(float)
            V = (z[f"{arm}|0|V"][held].astype(float) + z[f"{arm}|1|V"][held].astype(float))
            D = ((p0 - p1) ** 2).mean(1)
            m0 = ((p0 - obs) ** 2).mean(1); m1 = ((p1 - obs) ** 2).mean(1); ms = (m0 + m1) / 2
            ens_id = np.sqrt(ms - D / 4).mean()
            single = (np.sqrt(m0).mean() + np.sqrt(m1).mean()) / 2
            k = f"{arm}|{panel}|{f}"
            rec = dict(D_over_MCind=float(D.mean() / V.mean(1).mean()), crn=R["crn"][k],
                       lib_identical=R["crn"][k + "|lib_identical_across_seeds"], single=single,
                       mm_crn=R["ens_mm_crn"][k]["heldout"]["pb_rmse"], mm_indep=R["ens_mm_indep"][k]["heldout"]["pb_rmse"],
                       ens64=R["ens"][k]["heldout"]["pb_rmse"], ens_identity=float(ens_id))
            # projections: MSE_S = ms - D/2 + D/(2S)
            curve = lambda S, ms=ms, D=D: float(np.sqrt(np.maximum(ms - D / 2 + D / (2 * S), 0)).mean())
            rec["inf"] = curve(np.inf)
            rec["proj"] = {S: curve(S) for S in [1, 2, 4, 8, 16, 64]}
            bl = {b: R["baseline_pb"][f"{b}|{panel}|{f}"] for b in ["grand_mean", "ridge", "shrunk_stage_mean", "stage_mean", "resample"]}
            best = min(bl, key=bl.get)
            rec["best_baseline"] = (best, bl[best])
            rec["S_reach"] = next((S for S in range(1, 513) if curve(S) <= bl[best]), None)
            # how much of the projected floor is bias that is common to both seeds:
            rec["bias_common_rmse"] = float(np.sqrt(np.maximum((((p0 + p1) / 2 - obs) ** 2).mean(1) - D / 4, 0)).mean())
            res["B|" + k] = rec
            L.append(f"| {arm} | {panel} | {f} | {rec['D_over_MCind']:.1f} | {rec['crn']:+.3f} | {single:.3f} | {rec['mm_crn']:.3f} | "
                     f"{rec['mm_indep']:.3f} | {rec['ens64']:.3f} | {ens_id:.3f} |")

L.append("\n## C. Infinite-seed projection of held-out pb_rmse vs best baseline (own rescoring of stored baseline profiles)\n")
L.append("| arm | panel | fold | S=1 | S=2 | S=8 | S=inf | best baseline | S to reach (<=512) |\n|---|---|---|---|---|---|---|---|---|")
for panel in ARMS:
    for arm in ["base", "genemean"]:
        for f in range(3):
            r = res[f"B|{arm}|{panel}|{f}"]
            L.append(f"| {arm} | {panel} | {f} | {r['proj'][1]:.3f} | {r['proj'][2]:.3f} | {r['proj'][8]:.3f} | {r['inf']:.3f} | "
                     f"{r['best_baseline'][0]} {r['best_baseline'][1]:.3f} | {r['S_reach']} |")

# meanlog projection (gene-error vector), base and genemean
L.append("\n### meanlog_rmse_gene projection (held-out)\n\n| arm | panel | fold | S=1 | S=2 | S=inf | best baseline (evaluator JSON) | S to reach |\n|---|---|---|---|---|---|---|---|")
for panel in ARMS:
    for f in range(3):
        real = Real(panel, f)
        for arm in ["base", "genemean"]:
            es = []
            for s in range(2):
                g = np.load(RUNS / "fits" / arm / panel / f"fold{f}" / f"seed{s}" / "generated.npz")
                t = list(g["targets"].astype(str)); o = np.array([t.index(x) for x in real.donors])
                es.append((g["mean_log"][o].astype(float).mean(1) - real.ml)[real.held].mean(0))
            m1 = np.mean([(e ** 2).mean() for e in es]); Dm = ((es[0] - es[1]) ** 2).mean()
            cur = lambda S: float(np.sqrt(max(m1 - Dm / 2 + Dm / (2 * S), 0)))
            bls = {}
            for b in ["grand_mean", "ridge", "shrunk_stage_mean", "stage_mean_log", "resample"]:
                js = json.loads((RUNS / "results" / "baselines" / panel / f"fold{f}" / f"{b}.json").read_text())
                bls[b] = js["subsets"]["all"]["heldout"]["mean_log"]["rmse_gene"]
            prof = {k: v for k, v in bls.items() if k != "resample"}
            best = min(prof, key=prof.get)
            Sr = next((S for S in range(1, 513) if cur(S) <= bls[best]), None)
            Srs = next((S for S in range(1, 513) if cur(S) <= bls["resample"]), None)
            res[f"Cml|{arm}|{panel}|{f}"] = dict(S1=cur(1), S2=cur(2), inf=cur(np.inf), best=(best, bls[best]), S_reach=Sr,
                                                 resample=bls["resample"], S_reach_resample=Srs)
            L.append(f"| {arm} | {panel} | {f} | {cur(1):.3f} | {cur(2):.3f} | {cur(np.inf):.3f} | {best} {bls[best]:.3f} | {Sr} "
                     f"(resample {bls['resample']:.3f}: {Srs}) |")

# ------------------------------------------------------------- D. panels
L.append("\n## D. Panels like-for-like (shared genes, renormalised), held-out pb_rmse\n")
L.append("| arm | hvg2000 | wide | wide-hvg fold diffs | crossed (wide better) | ens hvg / wide |\n|---|---|---|---|---|---|")
for arm in ["base", "genemean"]:
    h = np.array([[R["shared"][f"{arm}|hvg2000|{f}|{s}"] for s in range(2)] for f in range(3)])
    w = np.array([[R["shared"][f"{arm}|wide|{f}|{s}"] for s in range(2)] for f in range(3)])
    cr = int(sum(w[f, i] < h[f, j] for f in range(3) for i in range(2) for j in range(2)))
    eh = np.mean([R["shared"][f"{arm}|hvg2000|{f}|ens"] for f in range(3)]); ew = np.mean([R["shared"][f"{arm}|wide|{f}|ens"] for f in range(3)])
    res[f"D|{arm}"] = dict(hvg=h.mean(), wide=w.mean(), fold=(w - h).mean(1).tolist(), crossed=cr, ens=(eh, ew))
    L.append(f"| {arm} | {h.mean():.3f} | {w.mean():.3f} | " + "/".join(f"{x:+.3f}" for x in (w - h).mean(1)) + f" | {cr}/12 | {eh:.3f} / {ew:.3f} |")
# gene-level agreement of shared predictions across panels
for f in range(3):
    zh = np.load(OUT / f"review_arms_profiles_hvg2000_fold{f}.npz"); zw = np.load(OUT / f"review_arms_profiles_wide_fold{f}.npz")
    assert np.array_equal(zh["shared_genes"], zw["shared_genes"]) and np.array_equal(zh["donors"], zw["donors"])
    res[f"D|n_shared|{f}"] = int(len(zh["shared_genes"]))
    res[f"D|obs_shared_maxdiff|{f}"] = float(np.abs(zh["obs_shared"] - zw["obs_shared"]).max())

# ------------------------------------------------------- E. seeds needed
L.append("\n## E. Seeds needed (|effect| = 2.8 SE), genemean - base, held-out pb_rmse\n")
for panel in ARMS:
    b = arr(R["fits"], "base", panel, "heldout", "pb_rmse"); a = arr(R["fits"], "genemean", panel, "heldout", "pb_rmse")
    eff = (a - b).mean()
    vb = np.mean((b[:, 0] - b[:, 1]) ** 2) / 2; va = np.mean((a[:, 0] - a[:, 1]) ** 2) / 2      # 3 df each
    n_fixed = 2.8 ** 2 * (va + vb) / (3 * eff ** 2)
    fd = (a - b).mean(1); vfm = fd.var(ddof=1); vint = max(vfm - (va + vb) / 2 / 3 * 3 / 3, 0)
    # chi-square 3 df 90% interval on the seed variance
    from scipy.stats import chi2
    lo, hi = 3 / chi2.ppf(.95, 3), 3 / chi2.ppf(.05, 3)
    res[f"E|{panel}"] = dict(effect=eff, sd_base=np.sqrt(vb), sd_arm=np.sqrt(va), n_fixed=n_fixed,
                             n_fixed_90pct_range=(n_fixed * lo, n_fixed * hi), fold_diffs=fd.tolist(),
                             fold_sd=float(np.sqrt(vfm)), t_folds=float(eff / np.sqrt(vfm / 3)))
    L.append(f"- {panel}: effect {eff:+.3f}; seed SD base {np.sqrt(vb):.3f}, genemean {np.sqrt(va):.3f} (3 df each); "
             f"seeds/arm for fixed folds {n_fixed:.1f} (90% range from seed-variance uncertainty {n_fixed*lo:.1f}..{n_fixed*hi:.1f}); "
             f"fold diffs " + "/".join(f"{x:+.3f}" for x in fd) + f", fold SD {np.sqrt(vfm):.3f}, effect/SE_fold {eff/np.sqrt(vfm/3):+.2f}")

# ------------------------------------------- F. Not AD counts per fold
for f in range(3):
    z = np.load(OUT / f"review_arms_profiles_hvg2000_fold{f}.npz")
    st, h = z["stage"], z["held"]
    res[f"F|{f}"] = {"heldout_by_stage": [int(((st == s) & h).sum()) for s in range(4)],
                     "train_by_stage": [int(((st == s) & ~h).sum()) for s in range(4)]}
L.append("\n## F. Stage counts held-out / train per fold: " + "; ".join(
    f"fold{f} held {res[f'F|{f}']['heldout_by_stage']} train {res[f'F|{f}']['train_by_stage']}" for f in range(3)))

# --------------------------------- G. covars: do held-out predictions copy a train donor?
L.append("\n## G. covars donor-identity check (hvg2000, held-out)\n")
L.append("For each held-out donor: which train donor's observed profile deviation (from train grand mean) best correlates "
         "with the predicted deviation; reported: fraction where that train donor is the nearest in covariate space "
         "(same stage, then sex, APOE4, age), and the mean max correlation.\n")
L.append("| arm | fold | seed | frac best-match = nearest-covariate train donor | mean max corr | mean corr with own obs |\n|---|---|---|---|---|---|")
for f in range(3):
    z = np.load(OUT / f"review_arms_profiles_hvg2000_fold{f}.npz")
    real = Real("hvg2000", f)
    held = z["held"]; obs = z["obs"].astype(float); grand = obs[~held].mean(0)
    tr = np.where(~held)[0]; ho = np.where(held)[0]
    raw = real.raw
    # condition vector the covars arm sees: ADNC/3, Male, APOE4 dose, age standardised on train donors (ddof=0)
    age = raw[:, 6]; za = (age - age[tr].mean()) / age[tr].std()
    cvec = np.stack([raw[:, 3], raw[:, 4], raw[:, 5], za], 1)
    near = {i: tr[int(np.argmin(((cvec[tr] - cvec[i]) ** 2).sum(1)))] for i in ho}
    for arm in ["base", "covars", "covars,genemean,offset"]:
        for s in range(2):
            P = z[f"{arm}|{s}|P"].astype(float)
            hit, mx, own = [], [], []
            for i in ho:
                dp = P[i] - grand
                c = np.array([np.corrcoef(dp, obs[j] - grand)[0, 1] for j in tr])
                hit.append(tr[int(np.argmax(c))] == near[i]); mx.append(c.max()); own.append(np.corrcoef(dp, obs[i] - grand)[0, 1])
            res[f"G|{arm}|{f}|{s}"] = dict(frac=float(np.mean(hit)), maxcorr=float(np.mean(mx)), own=float(np.mean(own)))
            L.append(f"| {arm} | {f} | {s} | {np.mean(hit):.2f} | {np.mean(mx):.3f} | {np.mean(own):+.3f} |")

(OUT / "review_arms_analysis.json").write_text(json.dumps(res, indent=1, default=float))
md = "# review_arms_analysis (generated)\n\n" + "\n".join(L)
(OUT / "review_arms_analysis.md").write_text(md)
print(md)
