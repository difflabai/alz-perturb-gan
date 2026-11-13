"""review_gap_core.py - adversarial recomputation of ANALYSIS_gap claims (independent code).

Reads caches (own loader, development-donor guard from folds.json) and generated.npz directly.
Study code is used ONLY to regenerate baseline train-target predictions (baselines.generate,
read-only) because the stored baseline npz files hold held-out predictions only; those are
checked against the stored held-out predictions.

Output: /ml2/alz/runs/improve_20261109/screen_analysis/review_gap_core.json
CPU only, <= 6 threads.
"""
import os
for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[_v] = "6"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys, json, time
from pathlib import Path
import numpy as np
import scipy.sparse as sp

sys.dont_write_bytecode = True
STUDY = Path("/ml2/alz/analysis/improve_20261109")
RUNS = Path("/ml2/alz/runs/improve_20261109")
OUT = RUNS / "screen_analysis"
FOLDS = json.loads((STUDY / "folds.json").read_text())
DEV = set(FOLDS["development_donors"]); SEALED = set(FOLDS["sealed_evaluation_donors_fold0"])
DO_BASELINES = "--no-baselines" not in sys.argv


def lcp(x):
    x = np.asarray(x, dtype=np.float64)
    return np.log1p(x * 1e4 / np.maximum(x.sum(-1, keepdims=True), 1))


def load_cache(panel, fold, n_sub=4, K=256):
    d = RUNS / "caches" / panel / f"fold{fold}"
    out = {"donors": [], "held": [], "stage": [], "pb": [], "ml": [], "nz": [], "pbK": [], "libs": []}
    for split in ["train", "heldout"]:
        with np.load(d / split / "metadata.npz", allow_pickle=False) as m:
            donors = m["donors"].astype(str); cd = m["donor"].astype(str); raw = m["raw_c"]
        bad = [x for x in set(donors.tolist()) | set(cd.tolist()) if x not in DEV or x in SEALED]
        assert not bad, bad
        C = sp.load_npz(d / split / "counts.npz").tocsr()
        for j, dn in enumerate(donors):
            X = C[np.where(cd == dn)[0]].toarray().astype(np.float64)
            out["donors"].append(dn); out["held"].append(split == "heldout")
            out["stage"].append(int(round(raw[j][3] * 3)))
            out["pb"].append(X.sum(0)); out["ml"].append(lcp(X).mean(0)); out["nz"].append((X > 0).mean(0))
            rng = np.random.default_rng([7, fold, j, split == "heldout"])
            subs = []
            for _ in range(n_sub):   # own 256-cell real subsamples (with replacement if n<K)
                idx = rng.choice(len(X), K, replace=len(X) < K)
                subs.append(lcp(X[idx].sum(0)))
            out["pbK"].append(np.stack(subs))
            out["libs"].append(X.sum(1))
    with np.load(d / "space.npz", allow_pickle=False) as f:
        out["W"] = f["W"].astype(np.float64); out["sd"] = f["sd"].astype(np.float64); out["genes"] = f["genes"].astype(str)
    for k in ["held", "stage", "pb", "ml", "nz", "pbK"]:
        out[k] = np.asarray(out[k])
    out["donors"] = np.asarray(out["donors"])
    return out


def load_gen(path, donors):
    with np.load(path, allow_pickle=False) as f:
        tg = f["targets"].astype(str)
        order = np.array([list(tg).index(d) for d in donors])
        pb = f["pb_counts"][order].astype(np.float64)
        ml = f["mean_log"][order].astype(np.float64).mean(1)
        st = f["stage"][order]; hd = f["is_heldout"][order]
    gl = lcp(pb)
    return {"pred": gl.mean(1), "mc": gl.var(1, ddof=1) / gl.shape[1], "pool": lcp(pb.sum(1)), "ml": ml,
            "stage": st, "held": hd}


def prmse(P, O):
    return float(np.mean(np.sqrt(np.mean((P - O) ** 2, 1))))


def span_frac(v, sd, W, rng, nperm=200):
    z = v / sd
    Q, _ = np.linalg.qr(W)
    f = float(np.sum((Q.T @ z) ** 2) / np.sum(z ** 2))
    null = np.array([np.sum((Q.T @ rng.permutation(z)) ** 2) / np.sum(z ** 2) for _ in range(nperm)])
    return {"frac": f, "perm_null_mean": float(null.mean()), "perm_null_q95": float(np.quantile(null, .95)),
            "dim_ratio": W.shape[1] / len(z)}


def main():
    res = {}
    t0 = time.time()
    for panel in ["hvg2000", "wide"]:
        for fold in range(3):
            key = f"{panel}/fold{fold}"
            c = load_cache(panel, fold)
            held = c["held"]; tr = ~held; st = c["stage"]
            obs = lcp(c["pb"]); Lbar = obs[tr].mean(0)
            r = {"n_held": int(held.sum()), "stage_held": np.bincount(st[held], minlength=4).tolist(),
                 "stage_train": np.bincount(st[tr], minlength=4).tolist()}
            # real depth effect: 256-cell real pb vs full depth
            r["real_pbK_minus_obs_mean"] = float((c["pbK"].mean(1) - obs).mean())
            r["real_pbK_minus_obs_mean_heldout"] = float((c["pbK"].mean(1)[held] - obs[held]).mean())
            r["Lbar_score"] = prmse(np.tile(Lbar, (held.sum(), 1)), obs[held])
            r["logpooled_score"] = prmse(np.tile(lcp(c["pb"][tr].sum(0)), (held.sum(), 1)), obs[held])
            P = {}
            for arm in ["base", "genemean"]:
                for s in (0, 1):
                    g = load_gen(RUNS / "fits" / arm / panel / f"fold{fold}" / f"seed{s}" / "generated.npz", c["donors"])
                    assert np.array_equal(g["held"].astype(bool), held) and np.array_equal(g["stage"], st)
                    P[f"{arm}_s{s}"] = g
                    stored = json.loads((RUNS / "results/fits" / arm / panel / f"fold{fold}" / f"seed{s}.json").read_text())
                    r[f"stored_pb_rmse_{arm}_s{s}"] = stored["subsets"]["all"]["heldout"]["pseudobulk"]["rmse"]
                    r[f"stored_meanlog_rmse_{arm}_s{s}"] = stored["subsets"]["all"]["heldout"]["mean_log"]["rmse_gene"]
                g0, g1 = P[f"{arm}_s0"], P[f"{arm}_s1"]
                P[f"{arm}_ens"] = {"pred": (g0["pred"] + g1["pred"]) / 2, "mc": (g0["mc"] + g1["mc"]) / 4,
                                   "pool": None, "ml": (g0["ml"] + g1["ml"]) / 2}
            if DO_BASELINES:
                sys.path.insert(0, str(STUDY))
                import evaluate as E, baselines as BL
                real = E.load_real(RUNS / "caches" / panel / f"fold{fold}", panel, fold)
                assert np.array_equal(real.donors, c["donors"])
                for name in ["grand_mean", "ridge", "shrunk_stage_mean"]:
                    gen, _ = BL.generate(name, real)
                    gl = lcp(gen["pb_counts"])
                    P[f"bl_{name}"] = {"pred": gl.mean(1), "mc": gl.var(1, ddof=1) / gl.shape[1], "pool": lcp(gen["pb_counts"].astype(np.float64).sum(1)),
                                       "ml": gen["mean_log"].astype(np.float64).mean(1)}
                    with np.load(RUNS / "results/baselines" / panel / f"fold{fold}" / f"{name}.npz") as f:
                        r[f"bl_{name}_maxdiff_vs_stored"] = float(np.abs(f["pred_profile_heldout"] - P[f"bl_{name}"]["pred"][held]).max())
                    del gen, gl
                del real
            rng = np.random.default_rng(11)
            preds = {}
            for p, g in P.items():
                pr = g["pred"]; d = {}
                d["pb_rmse"] = prmse(pr[held], obs[held])
                e = pr[held] - obs[held]; eb = e.mean(0)
                d["mse"] = float((e ** 2).mean()); d["shared"] = float((eb ** 2).mean())
                d["donor_specific"] = float(((e - eb) ** 2).mean())
                B = pr[held].mean(0) - Lbar
                d["B_ms"] = float((B ** 2).mean()); d["B_mean"] = float(B.mean())
                d["pred_var_held"] = float(pr[held].var(0).mean())
                d["mc_var_held"] = float(g["mc"][held].mean())
                cg = (pr[tr] - obs[tr]).mean(0)
                d["recal_pb_rmse"] = prmse(pr[held] - cg, obs[held])
                er = pr[held] - cg - obs[held]
                d["recal_shared"] = float((er.mean(0) ** 2).mean())
                d["corr_c_train_vs_ebar_held"] = float(np.corrcoef(cg, eb)[0, 1])
                d["ms_c_train"] = float((cg ** 2).mean())
                # leave-one-train-donor-out check of recal (in-sample vs out-of-sample for the *correction*)
                # model stage deviations only: lambda grid
                Sm = {s: pr[st == s].mean(0) for s in range(4) if (st == s).any()}
                dev = np.stack([Sm[s] for s in st[held]]) - pr[tr].mean(0)
                d["lambda"] = {str(l): prmse(Lbar + l * dev, obs[held]) for l in [0, .25, .5, 1]}
                if g.get("pool") is not None:
                    d["pred_minus_pool_mean"] = float((pr[held] - g["pool"][held]).mean())
                    d["pool_pb_rmse"] = prmse(g["pool"][held], obs[held])
                # per-cell mean-log
                bml_tr = (g["ml"][tr] - c["ml"][tr]).mean(0); bml_h = (g["ml"][held] - c["ml"][held]).mean(0)
                d["meanlog_rmse_gene_held"] = float(np.sqrt((bml_h ** 2).mean()))
                d["meanlog_rmse_gene_held_recal"] = float(np.sqrt(((bml_h - bml_tr) ** 2).mean()))
                d["meanlog_bias_train_ms"] = float((bml_tr ** 2).mean())
                d["span_meanlog_bias_train"] = span_frac(bml_tr, c["sd"], c["W"], rng)
                d["span_B"] = span_frac(B, c["sd"], c["W"], rng)
                d["corr_B_level"] = float(np.corrcoef(B, Lbar)[0, 1])
                q = np.quantile(Lbar, [.25, .5, .75]); bins = np.digitize(Lbar, q)
                d["B_by_Q"] = [float(B[bins == k].mean()) for k in range(4)]
                d["ebar_Q4_share"] = float((eb[bins == 3] ** 2).sum() / (eb ** 2).sum())
                # train-target depth-matched pseudobulk bias vs real 256-cell pb (avg of 4 subsamples) and shape gap
                pbk_b = (pr[tr] - c["pbK"].mean(1)[tr]).mean(0)
                d["pbK_bias_train_ms"] = float((pbk_b ** 2).mean())
                d["shape_gap_ms"] = float(((pbk_b - bml_tr) ** 2).mean())
                d["shape_gap_by_Q"] = [float((pbk_b - bml_tr)[bins == k].mean()) for k in range(4)]
                d["pbK_bias_by_Q"] = [float(pbk_b[bins == k].mean()) for k in range(4)]
                preds[p] = d
                g["_B"], g["_bml"], g["_eb"], g["_shape"], g["_dev"] = B, bml_tr, eb, pbk_b - bml_tr, dev
            # seed reproducibility
            rep = {}
            for arm in ["base", "genemean"]:
                for nm in ["_B", "_bml", "_shape"]:
                    a, b = P[f"{arm}_s0"][nm], P[f"{arm}_s1"][nm]
                    sv = float(((a - b) ** 2).mean() / 2)
                    rep[f"{arm}{nm}"] = {"pearson": float(np.corrcoef(a, b)[0, 1]),
                                         "shared_ms_est": float((((a + b) / 2) ** 2).mean() - sv / 2), "seed_specific_ms": sv}
            r["seed_rep"] = rep
            # stage: real train vs real held-out stage deviations (context for the model's ~0)
            stg = {}
            for s in range(4):
                mt, mh = tr & (st == s), held & (st == s)
                if mt.any() and mh.any():
                    rt = obs[mt].mean(0) - obs[tr].mean(0); rh = obs[mh].mean(0) - obs[held].mean(0)
                    stg[str(s)] = {"n_tr": int(mt.sum()), "n_h": int(mh.sum()),
                                   "pearson_realtrain_vs_realheld": float(np.corrcoef(rt, rh)[0, 1])}
                    for arm in ["base_s0", "base_s1", "genemean_s0", "genemean_s1", "base_ens", "genemean_ens"]:
                        pr = P[arm]["pred"]; dm = pr[st == s].mean(0) - pr.mean(0)
                        stg[str(s)][arm] = {"vs_realheld": float(np.corrcoef(dm, rh)[0, 1]), "vs_realtrain": float(np.corrcoef(dm, rt)[0, 1]),
                                            "var_ratio_vs_train": float(dm.var() / rt.var())}
            r["stage_dev"] = stg
            # shrunk stage-mean-of-logs (noise free) and ridge mu for reference to 'no stage predictor beats grand mean'
            r["preds"] = preds
            res[key] = r
            print(f"[{time.time() - t0:.0f}s] {key}", flush=True)
            (OUT / "review_gap_core.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
