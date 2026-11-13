"""Adversarial review of ANALYSIS_stage_signal (stage_*.py), real-data part.
Written from scratch (does not import stage_common). Reads only the study caches
(37 development donors; donor lists checked against folds.json).

Checks:
 R1  per-fold train-vs-held-out contrast Spearman (hn, hnl, slope) + held-out label
     permutation null, fold-mean p; held-out Not AD counts.
 R2  split-half reliability of the 37-donor hn contrast on common genes + shuffled null.
 R3  DAM directional test: recompute; competitive null (random same-size gene sets,
     all signed +, expression-matched); covariate confounding (age, sex, APOE4,
     cells, depth) of Not AD vs High; age-adjusted 5-gene score.
 R4  variance model: within-stage var, cell-bootstrap technical var (own code),
     split-half signal covariance; expected held-out Spearman of (i) the train-mean
     predictor and (ii) a PERFECT model (true contrast) against a 1 NotAD/6 High
     held-out set.
 R5  HVG overlap across folds.
Output: /ml2/alz/runs/improve_20261109/screen_analysis/review_stage/real.json + stdout.
"""
import os, sys, json
sys.dont_write_bytecode = True
for v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[v] = "6"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from scipy.stats import rankdata, spearmanr

RUNS = Path("/ml2/alz/runs/improve_20261109")
OUT = RUNS / "screen_analysis" / "review_stage"; OUT.mkdir(parents=True, exist_ok=True)
FOLDS = json.loads(Path("/ml2/alz/analysis/improve_20261109/folds.json").read_text())
DEV = set(FOLDS["development_donors"]); SEALED = set(FOLDS["sealed_evaluation_donors_fold0"])
rng = np.random.default_rng(777)


def check(ds):
    ds = set(map(str, ds))
    assert ds <= DEV and not (ds & SEALED), "non-development donor"


def lcp(X):
    X = np.asarray(X, float)
    return np.log1p(X / np.maximum(X.sum(-1, keepdims=True), 1) * 1e4)


def rho(a, b):
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def load(panel, k, cells=False):
    d = RUNS / "caches" / panel / f"fold{k}"
    genes = np.load(d / "space.npz")["genes"].astype(str)
    D = dict(donor=[], ho=[], stage=[], age=[], male=[], apoe4=[], pb=[], n=[], lib=[], cells=[])
    for split in ["train", "heldout"]:
        m = np.load(d / split / "metadata.npz")
        check(m["donors"]); check(np.unique(m["donor"]))
        assert np.array_equal(m["genes"].astype(str), genes)
        C = sp.load_npz(d / split / "counts.npz").tocsr()
        cd = m["donor"].astype(str)
        for j, dn in enumerate(m["donors"].astype(str)):
            X = C[np.flatnonzero(cd == dn)]
            rc = m["raw_c"][j]
            D["donor"].append(dn); D["ho"].append(split == "heldout")
            D["stage"].append(int(round(rc[3] * 3))); D["male"].append(rc[4]); D["apoe4"].append(rc[5]); D["age"].append(rc[6])
            D["pb"].append(np.asarray(X.sum(0)).ravel()); D["n"].append(X.shape[0])
            D["lib"].append(float(X.sum() / X.shape[0]))
            if cells:
                D["cells"].append(X)
    out = {k_: (np.array(v) if k_ not in ("cells",) else v) for k_, v in D.items()}
    out["genes"] = genes; out["P"] = lcp(out["pb"])
    return out


def hn(P, s):
    return P[s == 3].mean(0) - P[s == 0].mean(0)


def hnl(P, s):
    return P[s == 3].mean(0) - P[s <= 1].mean(0)


def slope(P, s):
    c = s - s.mean()
    return c @ (P - P.mean(0)) / (c @ c)


CF = {"hn": hn, "hnl": hnl, "slope": slope}
res = {}
NP = 1000
for panel in ["hvg2000", "wide"]:
    R = res[panel] = {}
    folds = [load(panel, k) for k in range(3)]
    # ---- R1
    obs = {c: [] for c in CF}; nul = {c: np.zeros((NP, 3)) for c in CF}
    R["heldout_n_by_stage"] = []
    R["train_n_by_stage"] = []
    for k, F in enumerate(folds):
        h, s, P = F["ho"], F["stage"], F["P"]
        R["heldout_n_by_stage"].append(np.bincount(s[h], minlength=4).tolist())
        R["train_n_by_stage"].append(np.bincount(s[~h], minlength=4).tolist())
        for c, fn in CF.items():
            ct = fn(P[~h], s[~h]); obs[c].append(rho(ct, fn(P[h], s[h])))
            sh = s[h]
            for b in range(NP):
                nul[c][b, k] = rho(ct, fn(P[h], rng.permutation(sh)))
    R["R1"] = {}
    for c in CF:
        o = np.array(obs[c]); nm = nul[c].mean(1)
        R["R1"][c] = {"per_fold": o.round(3).tolist(), "foldmean": round(float(o.mean()), 3),
                      "null_q95": round(float(np.quantile(nm, .95)), 3),
                      "p": round(float((1 + (nm >= o.mean()).sum()) / (NP + 1)), 3),
                      "per_fold_null_sd": nul[c].std(0).round(3).tolist()}
    # ---- R5
    gs = [set(F["genes"]) for F in folds]
    R["R5_common_genes"] = len(set.intersection(*gs)); R["R5_panel_sizes"] = [len(g) for g in gs]
    # ---- 37-donor data on common genes (rebuilt from each donor's held-out fold so every donor
    # is read once; CP10k renormalised within the common genes)
    common = sorted(set.intersection(*gs))
    rows = {}
    for F in folds:
        pos = {g: i for i, g in enumerate(F["genes"])}; idx = np.array([pos[g] for g in common])
        for i in np.flatnonzero(F["ho"]):
            rows[F["donor"][i]] = (F["pb"][i][idx], F["stage"][i], F["age"][i], F["male"][i], F["apoe4"][i], F["n"][i], F["lib"][i])
    assert len(rows) == 37
    dn = sorted(rows)
    pb = np.stack([rows[d][0] for d in dn]); st = np.array([rows[d][1] for d in dn])
    cov = {k_: np.array([rows[d][j] for d in dn], float) for j, k_ in enumerate(["_", "_", "age", "male", "apoe4", "ncells", "lib"]) if k_ != "_"}
    P = lcp(pb); G = P.shape[1]

    # ---- R2 split-half (own: halves within each stage, odd extra alternates)
    def halves(s):
        A = []
        for q in range(4):
            ix = rng.permutation(np.flatnonzero(s == q)); hh = len(ix) // 2 + (len(ix) % 2) * rng.integers(2)
            A += ix[:hh].tolist()
        a = np.zeros(len(s), bool); a[A] = True
        return a

    def sh(s, n):
        r, cv = [], []
        for _ in range(n):
            a = halves(s)
            ca, cb = hn(P[a], s[a]), hn(P[~a], s[~a])
            r.append(rho(ca, cb)); cv.append(np.cov(ca, cb)[0, 1])
        return np.mean(r), np.mean(cv)
    r_obs, cov_obs = sh(st, 400)
    nr = np.array([sh(rng.permutation(st), 20) for _ in range(200)])
    R["R2"] = {"G_common": G, "split_half": round(float(r_obs), 3), "SB": round(float(2 * r_obs / (1 + r_obs)), 3),
               "null_q95": round(float(np.quantile(nr[:, 0], .95)), 3), "p": round(float((1 + (nr[:, 0] >= r_obs).sum()) / 201), 3),
               "S_split": float(cov_obs), "S_split_null_sd": float(nr[:, 1].std()),
               "S_split_p": round(float((1 + (nr[:, 1] >= cov_obs).sum()) / 201), 3)}

    # ---- R3 covariates of Not AD vs High; DAM directional test, competitive null
    R["R3_covariates"] = {k_: {"NotAD": cov[k_][st == 0].round(2).tolist(),
                               "High_median": round(float(np.median(cov[k_][st == 3])), 2),
                               "High_range": [round(float(cov[k_][st == 3].min()), 2), round(float(cov[k_][st == 3].max()), 2)]}
                          for k_ in cov}
    UP = ["APOE", "TREM2", "SPP1", "GPNMB", "LPL", "CD74", "HLA-DRA", "TYROBP", "CST7", "ITGAX", "CD9",
          "LGALS3", "CD68", "AXL", "CLEC7A", "ACSL1", "CHI3L1", "CTSD", "CTSB", "FTL"]
    DOWN = ["P2RY12", "CX3CR1", "TMEM119"]
    gl = {g: i for i, g in enumerate(common)}
    up = [g for g in UP if g in gl]; dw = [g for g in DOWN if g in gl]
    idx = np.array([gl[g] for g in up + dw]); sg = np.r_[np.ones(len(up)), -np.ones(len(dw))]

    def wt(Pm, s):
        a, b = Pm[s == 3], Pm[s == 0]
        return (a.mean(0) - b.mean(0)) / np.sqrt(a.var(0, ddof=1) / len(a) + b.var(0, ddof=1) / len(b) + 1e-12)
    tall = wt(P, st)
    o = float((sg * tall[idx]).mean())
    nl = np.array([(sg * wt(P[:, idx], rng.permutation(st))).mean() for _ in range(2000)])
    # competitive null: random gene sets of same size, expression-matched (same expression-decile composition),
    # signs as in the a priori set (so the statistic is comparable)
    grand = P.mean(0); dec = np.digitize(grand, np.quantile(grand, np.linspace(.1, .9, 9)))
    comp = []
    for _ in range(5000):
        pick = np.array([rng.choice(np.flatnonzero(dec == dec[i])) for i in idx])
        comp.append((sg * tall[pick]).mean())
    comp = np.array(comp)
    R["R3_dam"] = {"genes_up": up, "genes_down": dw, "stat": round(o, 3),
                   "label_perm_q95": round(float(np.quantile(nl, .95)), 3),
                   "label_perm_p": round(float((1 + (nl >= o).sum()) / 2001), 4),
                   "competitive_mean": round(float(comp.mean()), 3), "competitive_q95": round(float(np.quantile(comp, .95)), 3),
                   "competitive_p": round(float((1 + (comp >= o).sum()) / 5001), 4),
                   "panel_mean_t": round(float(tall.mean()), 3), "panel_frac_t_pos": round(float((tall > 0).mean()), 3),
                   "per_gene_t": {g: round(float(tall[gl[g]]), 2) for g in up + dw}}
    # 5-gene post-hoc score: age / sex / APOE4 / depth association and age-adjusted stage effect
    five = [g for g in ["GPNMB", "CD9", "LGALS3", "LPL", "TYROBP"] if g in gl]
    Z = P[:, [gl[g] for g in five]]; Z = (Z - Z.mean(0)) / Z.std(0); sc = Z.mean(1)
    R["R3_five"] = {"genes": five, "spearman_stage": round(rho(sc, st), 3),
                    "spearman_age_all37": round(rho(sc, cov["age"]), 3),
                    "spearman_age_within_High": round(rho(sc[st == 3], cov["age"][st == 3]), 3),
                    "spearman_lib": round(rho(sc, cov["lib"]), 3), "spearman_ncells": round(rho(sc, cov["ncells"]), 3),
                    "spearman_apoe4": round(rho(sc, cov["apoe4"]), 3),
                    "notad_scores": sc[st == 0].round(2).tolist(), "high_median": round(float(np.median(sc[st == 3])), 2)}
    m = (st == 0) | (st == 3)
    X = np.c_[np.ones(m.sum()), (st[m] == 3), (cov["age"][m] - cov["age"].mean()) / 10, cov["male"][m], cov["apoe4"][m]]
    beta, *_ = np.linalg.lstsq(X, sc[m], rcond=None)
    bperm = []
    for _ in range(2000):
        Xp = X.copy(); Xp[:, 1] = rng.permutation(X[:, 1]); bperm.append(np.linalg.lstsq(Xp, sc[m], rcond=None)[0][1])
    bperm = np.array(bperm)
    Xn = np.c_[np.ones(m.sum()), (st[m] == 3)]
    R["R3_five"]["high_effect_unadjusted"] = round(float(np.linalg.lstsq(Xn, sc[m], rcond=None)[0][1]), 3)
    R["R3_five"]["high_effect_adj_age_sex_apoe4"] = round(float(beta[1]), 3)
    R["R3_five"]["adj_perm_p_one_sided"] = round(float((1 + (bperm >= beta[1]).sum()) / 2001), 4)
    R["R3_five"]["age_coef_per_decade"] = round(float(beta[2]), 3)

    # ---- R4 variance model (hvg2000 and wide), own bootstrap of cells (B=40)
    F0 = load(panel, 0, cells=True)
    pos = {g: i for i, g in enumerate(F0["genes"])}; cidx = np.array([pos[g] for g in common])
    order = np.array([list(F0["donor"]).index(d) for d in dn])
    tv = np.zeros(G)
    for i in order:
        X = F0["cells"][i][:, cidx].toarray().astype(float); n = X.shape[0]
        W = rng.multinomial(n, np.ones(n) / n, size=40).astype(float)
        tv += lcp(W @ X).var(0, ddof=1)
    tv /= 37
    ss = sum(((P[st == q] - P[st == q].mean(0)) ** 2).sum(0) for q in range(4)); s2 = ss / (37 - 4)
    S = R["R2"]["S_split"]

    def sim(S_, nt, nh, perfect=False, nsim=200):
        out = []
        for _ in range(nsim):
            d = rng.normal(0, np.sqrt(max(S_, 1e-12)), G)
            ch = d + rng.normal(0, 1, G) * np.sqrt(s2 * (1 / nh[0] + 1 / nh[1]))
            pr = d if perfect else d + rng.normal(0, 1, G) * np.sqrt(s2 * (1 / nt[0] + 1 / nt[1]))
            out.append(rho(pr, ch))
        return round(float(np.mean(out)), 3)
    S_hi = S + 1.645 * R["R2"]["S_split_null_sd"]
    R["R4"] = {"tech_frac_mean": round(float(tv.mean() / s2.mean()), 3), "mean_s2": round(float(s2.mean()), 4),
               "mean_tech": round(float(tv.mean()), 4), "S_split": round(S, 4), "S_hi": round(S_hi, 4),
               "SNR_now_4v19": round(S / float((s2 * (1 / 4 + 1 / 19)).mean()), 3),
               "trainmean_vs_ho_1v6_from_3v13": sim(S, (3, 13), (1, 6)),
               "perfect_vs_ho_1v6": sim(S, None, (1, 6), True), "perfect_vs_ho_2v6": sim(S, None, (2, 6), True),
               "perfect_vs_ho_1v6_Shi": sim(S_hi, None, (1, 6), True),
               "trainmean_80v160_vs_ho_1v6": sim(S, (80, 160), (1, 6)),
               "rep_4v19": sim(S, (4, 19), (4, 19)), "rep_20v40": sim(S, (20, 40), (20, 40)),
               "rep_4v19_2xcells": None}
    s2_half = s2 - tv / 2
    R["R4"]["rep_4v19_2xcells"] = round(float(np.mean([rho(*(lambda d: (d + rng.normal(0, 1, G) * np.sqrt(s2_half * (1 / 4 + 1 / 19)),
                                                                           d + rng.normal(0, 1, G) * np.sqrt(s2_half * (1 / 4 + 1 / 19))))(
        rng.normal(0, np.sqrt(S), G))) for _ in range(200)])), 3)
    print(panel, json.dumps(R, indent=1), flush=True)

(OUT / "real.json").write_text(json.dumps(res, indent=1, default=float))
