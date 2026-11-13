"""Adversarial review of ANALYSIS_stage_signal part (d), written from scratch.
Reads generated.npz (development donors only) and the fold caches (read-only).

Per fit and per seed-ensemble: model High-minus-NotAD contrast m on held-out targets
(evaluator convention) and all 37 targets; Spearman vs train / held-out real contrast;
residual of m on the train contrast and its correlation with the held-out contrast
(held-out label permutation null, 500 perms); count of units with p<0.05;
seed-to-seed residual correlation; residual vs train grand mean; and a DEPTH check:
per gene slope of the predicted profile on log target library among same-stage targets
(stage-only arms condition only on stage, so within-stage target differences come from
library size and common random numbers), times the High-minus-NotAD log-library
difference -> depth-induced contrast; its correlation with the residual.
Output: /ml2/alz/runs/improve_20261109/screen_analysis/review_stage/models.json + stdout.
"""
import os, sys, json
sys.dont_write_bytecode = True
for v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[v] = "6"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from scipy.stats import rankdata

RUNS = Path("/ml2/alz/runs/improve_20261109")
OUT = RUNS / "screen_analysis" / "review_stage"; OUT.mkdir(parents=True, exist_ok=True)
FOLDS = json.loads(Path("/ml2/alz/analysis/improve_20261109/folds.json").read_text())
DEV = set(FOLDS["development_donors"])
rng = np.random.default_rng(4242)


def lcp(X):
    X = np.asarray(X, float)
    return np.log1p(X / np.maximum(X.sum(-1, keepdims=True), 1) * 1e4)


def rho(a, b):
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def r(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def hn(P, s):
    return P[s == 3].mean(0) - P[s == 0].mean(0)


def res_on(y, x):
    X = np.c_[np.ones_like(x), x]
    return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]


cache = {}


def real(panel, k):
    if (panel, k) in cache:
        return cache[(panel, k)]
    d = RUNS / "caches" / panel / f"fold{k}"
    genes = np.load(d / "space.npz")["genes"].astype(str)
    dn, ho, st, pb, lib = [], [], [], [], []
    for split in ["train", "heldout"]:
        m = np.load(d / split / "metadata.npz"); assert set(m["donors"].astype(str)) <= DEV
        C = sp.load_npz(d / split / "counts.npz").tocsr(); cd = m["donor"].astype(str)
        for j, x in enumerate(m["donors"].astype(str)):
            X = C[np.flatnonzero(cd == x)]
            dn.append(x); ho.append(split == "heldout"); st.append(int(round(m["raw_c"][j][3] * 3)))
            pb.append(np.asarray(X.sum(0)).ravel()); lib.append(X.sum() / X.shape[0])
    o = dict(genes=genes, donors=np.array(dn), ho=np.array(ho), st=np.array(st), P=lcp(np.array(pb)), lib=np.array(lib))
    h, s, P = o["ho"], o["st"], o["P"]
    o["ctr"] = hn(P[~h], s[~h]); o["cho"] = hn(P[h], s[h]); o["grand"] = P[~h].mean(0)
    o["perm"] = [hn(P[h], rng.permutation(s[h])) for _ in range(500)]
    # real-data depth slope among train High donors (for comparison)
    cache[(panel, k)] = o
    return o


def depth_contrast(pred, s, loglib):
    """Per-gene slope of pred on loglib within stage (pooled within-stage OLS), times
    the High-minus-NotAD mean loglib difference."""
    num = np.zeros(pred.shape[1]); den = 0.
    for q in range(4):
        m = s == q
        if m.sum() > 1:
            x = loglib[m] - loglib[m].mean(); num += x @ (pred[m] - pred[m].mean(0)); den += x @ x
    b = num / den
    return b * (loglib[s == 3].mean() - loglib[s == 0].mean()), b


rows, store = [], {}
for arm_dir in sorted((RUNS / "fits").iterdir()):
    for panel_dir in sorted(arm_dir.iterdir()):
        panel = panel_dir.name
        for k in range(3):
            o = real(panel, k)
            preds, libs = [], []
            for sd in sorted((panel_dir / f"fold{k}").glob("seed*")):
                z = np.load(sd / "generated.npz")
                tg = z["targets"].astype(str); assert set(tg) <= DEV
                assert np.array_equal(z["genes"].astype(str), o["genes"])
                idx = np.array([np.flatnonzero(tg == x)[0] for x in o["donors"]])
                assert np.array_equal(z["stage"][idx], o["st"])
                pbc = z["pb_counts"][idx].astype(float)
                preds.append(np.stack([lcp(pbc[t]).mean(0) for t in range(len(idx))]))
                libs.append(z["lib"][idx].astype(float).mean(1))
            units = [(f"seed{i}", p, l) for i, (p, l) in enumerate(zip(preds, libs))]
            units.append(("ens", np.mean(preds, 0), np.mean(libs, 0)))   # mean of per-seed predictions
            h, s = o["ho"], o["st"]
            for lab, pred, lib in units:
                row = dict(arm=arm_dir.name, panel=panel, fold=k, unit=lab)
                for nm, msk in [("ho", h), ("all", np.ones_like(h))]:
                    m = hn(pred[msk], s[msk])
                    rr = res_on(m, o["ctr"])
                    ob = r(rr, o["cho"]); nl = np.array([r(rr, q) for q in o["perm"]])
                    row.update({f"{nm}_sp_train": rho(m, o["ctr"]), f"{nm}_sp_ho": rho(m, o["cho"]),
                                f"{nm}_resid_frac": float(rr.var() / m.var()), f"{nm}_resid_r_ho": ob,
                                f"{nm}_resid_p": float((1 + (nl >= ob).sum()) / 501), f"{nm}_resid_null_sd": float(nl.std()),
                                f"{nm}_resid_r_grand": r(rr, o["grand"])})
                    store[(arm_dir.name, panel, k, lab, nm)] = rr
                # depth check on all 37 targets
                m = hn(pred, s); rr = res_on(m, o["ctr"])
                dc, b = depth_contrast(pred, s, np.log(lib))
                row["depth_contrast_r_resid"] = r(dc, rr)
                row["depth_contrast_frac_resid_var"] = float(np.var(dc) / np.var(rr))
                row["loglib_diff_High_minus_NotAD"] = float(np.log(lib)[s == 3].mean() - np.log(lib)[s == 0].mean())
                rows.append(row)
            print(arm_dir.name, panel, k, flush=True)

seed_r = {}
for (a, p, k, lab, nm), rr in store.items():
    if lab == "seed0" and nm == "all":
        seed_r.setdefault((a, p), []).append(r(rr, store[(a, p, k, "seed1", nm)]))


def mean(v):
    return round(float(np.mean(v)), 3)


summ = {}
for a, p in sorted(set((x["arm"], x["panel"]) for x in rows)):
    for kind in ["single", "ens"]:
        R = [x for x in rows if x["arm"] == a and x["panel"] == p and ((x["unit"] == "ens") == (kind == "ens"))]
        summ[f"{a}|{p}|{kind}"] = {
            "n": len(R), "sp_train_ho": mean([x["ho_sp_train"] for x in R]), "sp_train_all": mean([x["all_sp_train"] for x in R]),
            "sp_heldout_ho": mean([x["ho_sp_ho"] for x in R]), "resid_frac_ho": mean([x["ho_resid_frac"] for x in R]),
            "resid_r_ho": mean([x["ho_resid_r_ho"] for x in R]), "resid_null_sd": mean([x["ho_resid_null_sd"] for x in R]),
            "n_p05": int(sum(x["ho_resid_p"] < .05 for x in R)), "resid_r_grand_all": mean([x["all_resid_r_grand"] for x in R]),
            "seed_resid_r": mean(seed_r[(a, p)]) if kind == "single" else None,
            "depth_r_resid": mean([x["depth_contrast_r_resid"] for x in R]),
            "depth_frac_resid_var": mean([x["depth_contrast_frac_resid_var"] for x in R]),
            "loglib_diff": mean([x["loglib_diff_High_minus_NotAD"] for x in R])}
tot = sum(x["ho_resid_p"] < .05 for x in rows)
train_sp = {p: [rho(real(p, k)["ctr"], real(p, k)["cho"]) for k in range(3)] for p in ["hvg2000", "wide"]}
out = {"summary": summ, "n_units": len(rows), "n_units_p05_ho": int(tot), "train_vs_heldout_sp": train_sp, "rows": rows}
(OUT / "models.json").write_text(json.dumps(out, indent=1, default=float))
print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
