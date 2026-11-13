"""(b) Does any gene subset carry reproducible High-vs-NotAD signal?

Subsets are defined from TRAIN donors of each fold only (or a priori lists), then
train contrast vs held-out contrast is compared within the subset, with the
held-out label permutation null of stage_a (held-out sets partition the 37 donors).
Profiles use the full-panel CP10k denominator (as the evaluator's 'risk' subset).

Subsets (per fold): top 10% genes by train between-donor variance, top 10% by train
grand-mean expression, top 10% by train |t| of the High-NotAD contrast, bottom 50%
by expression (control), forced risk genes (wide), DAM/homeostatic list present.

A priori directional test (specified in this script before any of these genes'
results were looked at): DAM/activation genes expected UP in High vs Not AD,
homeostatic genes (P2RY12, CX3CR1, TMEM119) expected DOWN. Statistic: mean over
present genes of sign_expected * Welch t of the 37-donor High-NotAD contrast;
null: stage labels permuted over the 37 donors (stage counts kept).
Per-gene table for named genes on the wide panel (all 37 donors).
Output: OUT/b_gene_subsets.json, printed tables.
"""
from stage_common import *

NPERM = 2000
rng = np.random.default_rng(20261113)
UP = ["APOE", "TREM2", "SPP1", "GPNMB", "LPL", "CD74", "HLA-DRA", "TYROBP", "CST7", "ITGAX", "CD9",
      "LGALS3", "CD68", "AXL", "CLEC7A", "ACSL1", "CHI3L1", "CTSD", "CTSB", "FTL"]
DOWN = ["P2RY12", "CX3CR1", "TMEM119"]
KEYS = ["hn", "hnl", "slope"]


def welch_t(P, st):
    a, b = P[st == 3], P[st == 0]
    se = np.sqrt(a.var(0, ddof=1) / len(a) + b.var(0, ddof=1) / len(b))
    return (a.mean(0) - b.mean(0)) / np.maximum(se, 1e-12)


res = {}
for panel in ["hvg2000", "wide"]:
    R = res[panel] = {"subsets": {}}
    obs = {}; null = {}
    for f in range(3):
        c = load_cache(panel, f)
        P = log_cp10k(c["pb"]); st = c["stage"]; h = c["is_heldout"]; t = ~h
        G = P.shape[1]; k10 = int(np.ceil(.1 * G))
        gm = P[t].mean(0)
        wv = np.zeros(G)
        for s in range(4):
            m = t & (st == s)
            if m.sum() > 1:
                wv += ((P[m] - P[m].mean(0)) ** 2).sum(0)
        tv = P[t].var(0)
        ctr = contrasts(P[t], st[t])
        pooled = wv / max(t.sum() - 4, 1)
        tstat = ctr["hn"] / np.sqrt(np.maximum(pooled, 1e-12) * (1 / (st[t] == 0).sum() + 1 / (st[t] == 3).sum()))
        gl = list(c["genes"])
        subs = {"all": np.arange(G),
                "top10_var": np.argsort(-tv)[:k10],
                "top10_expr": np.argsort(-gm)[:k10],
                "top10_train_abs_t": np.argsort(-np.abs(tstat))[:k10],
                "bottom50_expr": np.argsort(gm)[:G // 2],
                "risk_forced": np.array([gl.index(x) for x in RISK_GENES if x in gl], dtype=int),
                "dam_homeo": np.array([gl.index(x) for x in UP + DOWN if x in gl], dtype=int)}
        perms = [rng.permutation(st[h]) for _ in range(NPERM)]
        pc = [contrasts(P[h], p_) for p_ in perms]
        cho = contrasts(P[h], st[h])
        for name, idx in subs.items():
            if len(idx) < 5:
                continue
            for k in KEYS:
                a = agree(ctr[k][idx], cho[k][idx])
                obs.setdefault((name, k), []).append(a)
                null.setdefault((name, k), []).append(np.array([spearman(ctr[k][idx], q[k][idx]) for q in pc]))
            R["subsets"].setdefault(name, {})[f"n_genes_fold{f}"] = int(len(idx))
    ntest = 0
    for (name, k), A in obs.items():
        if len(A) < 3:
            continue
        ntest += 1
        o = float(np.mean([a["spearman"] for a in A]))
        n = np.mean(np.stack(null[(name, k)]), 0)
        R["subsets"][name][k] = {"per_fold_spearman": [a["spearman"] for a in A],
                                 "per_fold_top10_sign_by_train": [a["top10_sign_by_pred"] for a in A],
                                 "per_fold_slope_heldout_on_train": [a["slope_real_on_pred"] for a in A],
                                 "foldmean_spearman": o, "null_q95": float(np.quantile(n, .95)),
                                 "p_one_sided": float((1 + (n >= o).sum()) / (1 + NPERM))}
    R["n_tests"] = ntest
    # a priori directional test and per-gene table on all 37 donors
    c = all37(panel)
    P = log_cp10k(c["pb"]); st = c["stage"]; gl = list(c["genes"])
    up = [x for x in UP if x in gl]; dn = [x for x in DOWN if x in gl]
    idx = np.array([gl.index(x) for x in up + dn]); sg = np.array([1] * len(up) + [-1] * len(dn))
    if len(idx):
        def stat(stv):
            return float((sg * welch_t(P[:, idx], stv)).mean())
        o = stat(st)
        nn = np.array([stat(rng.permutation(st)) for _ in range(NPERM)])
        tt = welch_t(P[:, idx], st)
        cc = contrasts(P[:, idx], st)["hn"]
        # per-gene permutation p (two-sided on |t|) and fold-wise train/held-out signs
        nt = np.stack([welch_t(P[:, idx], rng.permutation(st)) for _ in range(NPERM)])
        pg = ((np.abs(nt) >= np.abs(tt)[None]).sum(0) + 1) / (NPERM + 1)
        fs = []
        for f in range(3):
            cf = load_cache(panel, f); Pf = log_cp10k(cf["pb"]); hf = cf["is_heldout"]; sf = cf["stage"]
            glf = list(cf["genes"])
            ii = np.array([glf.index(x) for x in up + dn])
            fs.append((np.sign(contrasts(Pf[~hf], sf[~hf])["hn"][ii]), np.sign(contrasts(Pf[hf], sf[hf])["hn"][ii])))
        R["apriori"] = {"genes_up": up, "genes_down": dn, "stat_mean_signed_t": o,
                        "null_mean": float(nn.mean()), "null_q95": float(np.quantile(nn, .95)),
                        "p_one_sided": float((1 + (nn >= o).sum()) / (1 + NPERM)),
                        "per_gene": [{"gene": g, "expected": int(s), "hn_log": float(cv), "welch_t": float(tv_),
                                      "perm_p_two_sided": float(p_), "mean_expr": float(P[:, i].mean()),
                                      "fold_train_sign": [int(x[0][j]) for x in fs],
                                      "fold_heldout_sign": [int(x[1][j]) for x in fs]}
                                     for j, (g, s, cv, tv_, p_, i) in enumerate(zip(up + dn, sg, cc, tt, pg, idx))]}

(OUT / "b_gene_subsets.json").write_text(json.dumps(res, indent=1, default=float))

for panel, R in res.items():
    print(f"\n## {panel} ({R['n_tests']} subset x contrast tests)")
    print("| subset | n genes (folds) | contrast | per-fold Spearman | fold-mean | null q95 | p | top10 sign by train (per fold) | slope held~train (per fold) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name, S in R["subsets"].items():
        ng = [S.get(f"n_genes_fold{f}") for f in range(3)]
        for k in KEYS:
            if k not in S:
                continue
            d = S[k]
            print(f"| {name} | {ng} | {k} | {', '.join(fmt(x, 2) for x in d['per_fold_spearman'])} | {fmt(d['foldmean_spearman'])} | {fmt(d['null_q95'])} | {fmt(d['p_one_sided'])} | "
                  f"{', '.join(fmt(x, 2) for x in d['per_fold_top10_sign_by_train'])} | {', '.join(fmt(x, 2) for x in d['per_fold_slope_heldout_on_train'])} |")
    if "apriori" in R:
        a = R["apriori"]
        print(f"a priori directional test: mean signed t {fmt(a['stat_mean_signed_t'])}, null mean {fmt(a['null_mean'])}, q95 {fmt(a['null_q95'])}, p {fmt(a['p_one_sided'])}; up {a['genes_up']} down {a['genes_down']}")
        print("| gene | expected | High-NotAD (log1p CP10k) | Welch t | perm p (2-sided) | mean expr | train sign by fold | held-out sign by fold |")
        print("|---|---|---|---|---|---|---|---|")
        for g in a["per_gene"]:
            print(f"| {g['gene']} | {'+' if g['expected'] > 0 else '-'} | {fmt(g['hn_log'])} | {fmt(g['welch_t'], 2)} | {fmt(g['perm_p_two_sided'])} | {fmt(g['mean_expr'], 2)} | {g['fold_train_sign']} | {g['fold_heldout_sign']} |")
