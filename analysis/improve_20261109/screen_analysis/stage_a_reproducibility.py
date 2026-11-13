"""(a) Is there any reproducible stage signal in the real data?

Per panel and inner fold, from the caches only:
  * train contrast vs held-out contrast (High-NotAD 'hn', High-(NotAD+Low) 'hnl',
    ordinal slope) - Spearman, Pearson, top-decile sign agreement, slopes;
  * label-permutation null: held-out stage labels permuted within each fold's
    held-out set (held-out sets partition the 37 donors, so permutations are
    independent across folds); one-sided p for the per-fold and fold-mean statistic;
  * donor-level cross-fitted score: each held-out donor's deviation from the
    train grand mean, correlated with the train contrast; Spearman(score, stage)
    and AUC High vs NotAD over the 37 pooled held-out donors, same permutation null;
  * split-half reliability on all 37 donors (genes common to the 3 fold caches)
    and on each fold's train donors, with Spearman-Brown, and a null from
    permuting stage over the 37 donors.
Output: OUT/a_reproducibility.json and printed tables.
"""
from stage_common import *

NPERM = 2000
rng = np.random.default_rng(20261110)
KEYS = ["hn", "hnl", "slope"]


def auc(x, pos, neg):
    a, b = x[pos], x[neg]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    return float(((a[:, None] > b[None, :]).sum() + .5 * (a[:, None] == b[None, :]).sum()) / (len(a) * len(b)))


def split_half(P, stage, key, nsplit, rng):
    out = []
    for _ in range(nsplit):
        A, B = [], []
        for s in range(4):
            idx = rng.permutation(np.where(stage == s)[0])
            h = len(idx) // 2
            # odd counts: alternate which half gets the extra donor
            if len(idx) % 2 and rng.random() < .5:
                h += 1
            A += idx[:h].tolist(); B += idx[h:].tolist()
        A, B = np.array(A), np.array(B)
        ca, cb = contrasts(P[A], stage[A])[key], contrasts(P[B], stage[B])[key]
        if ca is None or cb is None:
            continue
        out.append(spearman(ca, cb))
    return float(np.nanmean(out)) if out else np.nan


def sb(r, k=2.0):
    return k * r / (1 + (k - 1) * r) if np.isfinite(r) and r > -1 / (k - 1) else np.nan


res = {}
for panel in ["hvg2000", "wide"]:
    R = res[panel] = {"folds": {}}
    folds = [load_cache(panel, k) for k in range(3)]
    obs_fold_stat = {k: [] for k in KEYS}
    perm_fold_stat = {k: np.zeros((NPERM, 3)) for k in KEYS}
    scores, sstage, sfold = [], [], []
    for f, c in enumerate(folds):
        P = log_cp10k(c["pb"]); st = c["stage"]; h = c["is_heldout"]; t = ~h
        ctr = contrasts(P[t], st[t]); cho = contrasts(P[h], st[h])
        F = R["folds"][f] = {"n_train_by_stage": np.bincount(st[t], minlength=4).tolist(),
                             "n_heldout_by_stage": np.bincount(st[h], minlength=4).tolist(), "G": int(P.shape[1])}
        for k in KEYS:
            F[k] = agree(ctr[k], cho[k])
            obs_fold_stat[k].append(F[k]["spearman"])
        # permutation null on held-out labels
        Ph, sh = P[h], st[h]
        for b in range(NPERM):
            sp_ = rng.permutation(sh)
            cp = contrasts(Ph, sp_)
            for k in KEYS:
                perm_fold_stat[k][b, f] = spearman(ctr[k], cp[k])
        for k in KEYS:
            null = perm_fold_stat[k][:, f]
            F[k]["perm_p_one_sided"] = float((1 + (null >= F[k]["spearman"]).sum()) / (1 + NPERM))
            F[k]["perm_null_mean"] = float(null.mean()); F[k]["perm_null_sd"] = float(null.std())
            F[k]["perm_null_q95"] = float(np.quantile(null, .95))
        # donor-level cross-fitted score
        grand = P[t].mean(0)
        for i in np.where(h)[0]:
            scores.append([pearson(P[i] - grand, ctr[k]) for k in KEYS]); sstage.append(st[i]); sfold.append(f)
        # train-only split half (as the evaluator's real_split_half)
        F["train_split_half"] = {k: split_half(P[t], st[t], k, 200, rng) for k in KEYS}
    for k in KEYS:
        o = float(np.mean(obs_fold_stat[k])); n = perm_fold_stat[k].mean(1)
        R[f"foldmean_{k}"] = {"spearman": o, "perm_p_one_sided": float((1 + (n >= o).sum()) / (1 + NPERM)),
                              "null_mean": float(n.mean()), "null_q95": float(np.quantile(n, .95))}
    # donor score test
    S = np.array(scores); ss = np.array(sstage); sf = np.array(sfold)
    D = R["donor_score"] = {}
    for j, k in enumerate(KEYS):
        x = S[:, j]
        obs_r = spearman(x, ss); obs_auc = auc(x, ss == 3, ss == 0); obs_auc2 = auc(x, ss == 3, ss <= 1)
        nr, na, na2 = [], [], []
        for b in range(NPERM):
            sp_ = ss.copy()
            for f in range(3):
                m = sf == f
                sp_[m] = rng.permutation(ss[m])
            nr.append(spearman(x, sp_)); na.append(auc(x, sp_ == 3, sp_ == 0)); na2.append(auc(x, sp_ == 3, sp_ <= 1))
        nr, na, na2 = map(np.array, (nr, na, na2))
        D[k] = {"spearman_score_stage": obs_r, "p": float((1 + (nr >= obs_r).sum()) / (1 + NPERM)),
                "auc_high_vs_notad": obs_auc, "p_auc": float((1 + (na >= obs_auc).sum()) / (1 + NPERM)),
                "auc_high_vs_notad_low": obs_auc2, "p_auc2": float((1 + (na2 >= obs_auc2).sum()) / (1 + NPERM)),
                "mean_score_by_stage": [float(x[ss == s].mean()) for s in range(4)]}
    # split-half on all 37 donors, with permutation null
    c = all37(panel)
    P = log_cp10k(c["pb"]); st = c["stage"]
    R["all37"] = {"G_common": int(P.shape[1]), "n_by_stage": np.bincount(st, minlength=4).tolist()}
    for k in KEYS:
        r = split_half(P, st, k, 500, rng)
        null = np.array([split_half(P, rng.permutation(st), k, 20, rng) for _ in range(200)])
        R["all37"][k] = {"split_half_spearman": r, "sb_full": sb(r),
                         "null_mean": float(np.nanmean(null)), "null_sd": float(np.nanstd(null)),
                         "null_q95": float(np.nanquantile(null, .95)),
                         "perm_p": float((1 + (null >= r).sum()) / (1 + len(null)))}

(OUT / "a_reproducibility.json").write_text(json.dumps(res, indent=1, default=float))

for panel, R in res.items():
    print(f"\n## {panel}")
    print("| fold | n train (NotAD/Low/Int/High) | n held-out | contrast | Spearman | perm null mean (q95) | p | Pearson | top10 sign (by train) | top10 sign (by held-out) | slope held~train | slope train~held |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for f, F in R["folds"].items():
        for k in KEYS:
            a = F[k]
            print(f"| {f} | {F['n_train_by_stage']} | {F['n_heldout_by_stage']} | {k} | {fmt(a['spearman'])} | {fmt(a['perm_null_mean'])} ({fmt(a['perm_null_q95'])}) | {fmt(a['perm_p_one_sided'])} | {fmt(a['pearson'])} | {fmt(a['top10_sign_by_pred'])} | {fmt(a['top10_sign_by_real'])} | {fmt(a['slope_real_on_pred'])} | {fmt(a['slope_pred_on_real'])} |")
    for k in KEYS:
        m = R[f"foldmean_{k}"]
        print(f"fold-mean {k}: Spearman {fmt(m['spearman'])}, null mean {fmt(m['null_mean'])}, q95 {fmt(m['null_q95'])}, p {fmt(m['perm_p_one_sided'])}")
    for f, F in R["folds"].items():
        print(f"fold {f} train split-half: " + ", ".join(f"{k} {fmt(v)}" for k, v in F["train_split_half"].items()))
    print("donor score (37 pooled held-out donors):")
    for k, d in R["donor_score"].items():
        print(f"  {k}: Spearman(score, stage) {fmt(d['spearman_score_stage'])} p={fmt(d['p'])}; AUC High vs NotAD {fmt(d['auc_high_vs_notad'])} p={fmt(d['p_auc'])}; AUC High vs NotAD+Low {fmt(d['auc_high_vs_notad_low'])} p={fmt(d['p_auc2'])}; mean score by stage {[fmt(x) for x in d['mean_score_by_stage']]}")
    a = R["all37"]
    print(f"all 37 donors, {a['G_common']} common genes, n by stage {a['n_by_stage']}:")
    for k in KEYS:
        d = a[k]
        print(f"  {k}: split-half Spearman {fmt(d['split_half_spearman'])}, SB(full 37) {fmt(d['sb_full'])}, null mean {fmt(d['null_mean'])} sd {fmt(d['null_sd'])} q95 {fmt(d['null_q95'])}, p {fmt(d['perm_p'])}")
