"""(d) Does any arm add or lose stage information relative to the train stage means?

For every fit (arm, panel, fold, seed) and every seed-ensemble, from generated.npz
(read-only) and the fold cache:
  m      model High-minus-NotAD contrast from predicted profiles (mean over the M
         generated donors of log1p CP10k of the generated pseudobulk), on held-out
         targets (as evaluate.py) and on all 37 targets (less CRN noise);
  c_tr   real train contrast, c_ho real held-out contrast (fold gene axis).
Reported: Spearman(m,c_tr), Spearman(m,c_ho), Spearman(c_tr,c_ho); the regression
m = a + b*c_tr + r and whether the residual r (what the model adds beyond a scaled
copy of the train contrast) predicts c_ho: Pearson(r, c_ho) and the partial
correlation of c_ho with m given c_tr, with a held-out label permutation null;
residual agreement across seeds (systematic distortion vs noise) and the residual's
correlation with expression level (train grand mean).
Output: OUT/d_model_contrast.json, printed tables.
"""
from stage_common import *

NPERM = 500
rng = np.random.default_rng(20261112)


def pred_profiles(pb):
    out = np.empty((pb.shape[0], pb.shape[2]))
    for t in range(pb.shape[0]):
        out[t] = log_cp10k(pb[t]).mean(0)
    return out


def resid(y, x):
    X = np.c_[np.ones_like(x), x]
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return y - X @ beta, float(beta[1])


def partial(y, m, x):
    ry, _ = resid(y, x); rm, _ = resid(m, x)
    return pearson(ry, rm)


rows, resid_store = [], {}
caches = {}
for arm_dir in sorted(FITS.iterdir()):
    for panel_dir in sorted(arm_dir.iterdir()):
        panel = panel_dir.name
        for f in range(3):
            key = (panel, f)
            if key not in caches:
                c = load_cache(panel, f)
                P = log_cp10k(c["pb"]); h = c["is_heldout"]; st = c["stage"]
                caches[key] = dict(c=c, P=P, ctr=contrasts(P[~h], st[~h])["hn"], cho=contrasts(P[h], st[h])["hn"],
                                   grand=P[~h].mean(0),
                                   perm=[contrasts(P[h], rng.permutation(st[h]))["hn"] for _ in range(NPERM)])
            K = caches[key]; c = K["c"]
            ens = []
            for sd in sorted((panel_dir / f"fold{f}").glob("seed*")):
                g = sd / "generated.npz"
                if not g.exists():
                    continue
                with np.load(g, allow_pickle=False) as z:
                    tg = z["targets"].astype(str); genes = z["genes"].astype(str)
                    guard(tg.tolist(), g)
                    assert np.array_equal(genes, c["genes"])
                    order = np.array([list(tg).index(d) for d in c["donors"]])
                    pb = z["pb_counts"][order].astype(np.float64)
                ens.append(pb)
            label_sets = [(f"seed{i}", [pb]) for i, pb in enumerate(ens)] + ([("ens", ens)] if len(ens) > 1 else [])
            for lab, pbs in label_sets:
                pred = np.mean([pred_profiles(pb) for pb in pbs], 0) if lab != "ens" else \
                    pred_profiles(np.concatenate(pbs, axis=1))
                h, st = c["is_heldout"], c["stage"]
                m_ho = contrasts(pred[h], st[h])["hn"]
                m_all = contrasts(pred, st)["hn"]
                row = {"arm": arm_dir.name, "panel": panel, "fold": f, "seed": lab}
                for mname, m in [("ho", m_ho), ("all", m_all)]:
                    r, b = resid(m, K["ctr"])
                    obs = pearson(r, K["cho"]); obsp = partial(K["cho"], m, K["ctr"])
                    nullr = np.array([pearson(r, cp) for cp in K["perm"]])
                    row.update({f"{mname}_sp_vs_train": spearman(m, K["ctr"]), f"{mname}_sp_vs_heldout": spearman(m, K["cho"]),
                                f"{mname}_slope_on_train": b, f"{mname}_resid_frac_var": float(r.var() / m.var()),
                                f"{mname}_resid_r_heldout": obs, f"{mname}_partial_r_heldout": obsp,
                                f"{mname}_resid_p": float((1 + (nullr >= obs).sum()) / (1 + NPERM)),
                                f"{mname}_resid_r_grandmean": pearson(r, K["grand"])})
                    resid_store[(arm_dir.name, panel, f, lab, mname)] = r
                row["train_vs_heldout_sp"] = spearman(K["ctr"], K["cho"])
                row["train_vs_heldout_pearson"] = pearson(K["ctr"], K["cho"])
                rows.append(row)
            print(arm_dir.name, panel, f, "done", flush=True)

# residual agreement across seeds
seedagree = []
for (arm, panel, f, lab, mname), r in resid_store.items():
    if lab == "seed0" and (arm, panel, f, "seed1", mname) in resid_store:
        seedagree.append({"arm": arm, "panel": panel, "fold": f, "which": mname,
                          "resid_seed_pearson": pearson(r, resid_store[(arm, panel, f, "seed1", mname)])})

(OUT / "d_model_contrast.json").write_text(json.dumps({"rows": rows, "seed_resid": seedagree}, indent=1, default=float))


def mean(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return np.mean(xs) if xs else np.nan


print("\n### single fits (fold-mean over folds, seed-mean); m on held-out targets [all 37 targets]")
print("| arm | panel | sp(m, train) | sp(m, held-out) | sp(train, held-out) | slope m~train | resid var frac | r(resid, held-out) | partial r | n folds x seeds with resid p<0.05 | r(resid, grand mean) | seed resid r |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for arm in sorted(set(r["arm"] for r in rows)):
    for panel in ["hvg2000", "wide"]:
        for kind in ["single", "ens"]:
            R = [r for r in rows if r["arm"] == arm and r["panel"] == panel and ((r["seed"] == "ens") == (kind == "ens"))]
            if not R:
                continue
            sa = [s["resid_seed_pearson"] for s in seedagree if s["arm"] == arm and s["panel"] == panel and s["which"] == "all"]
            def g(k):
                return f"{fmt(mean([r['ho_' + k] for r in R]))} [{fmt(mean([r['all_' + k] for r in R]))}]"
            nsig = sum(r["ho_resid_p"] < .05 for r in R)
            print(f"| {arm}{' [ens]' if kind == 'ens' else ''} | {panel} | {g('sp_vs_train')} | {g('sp_vs_heldout')} | {fmt(mean([r['train_vs_heldout_sp'] for r in R]))} | "
                  f"{g('slope_on_train')} | {g('resid_frac_var')} | {g('resid_r_heldout')} | {g('partial_r_heldout')} | {nsig}/{len(R)} | {g('resid_r_grandmean')} | "
                  f"{fmt(mean(sa)) if kind == 'single' else '-'} |")
