"""gap_decompose.py - why generators lose to trivial baselines on held-out pseudobulk.

Read-only use of the study code (evaluate.load_real / align_generated,
baselines.fit_profile_baseline / generate, metrics.log_cp10k). Works only from
the inner-fold caches (37 development donors; evaluate.load_real enforces the
development-donor guard) and the fits' generated.npz.  CPU only, <=6 threads.

For every panel x fold x arm in {base, genemean} x seed in {0,1} (+ the
seed-ensemble = average of the two seed predictions, identical to the
evaluator's M-concatenation) and for baselines grand_mean / ridge /
shrunk_stage_mean (regenerated with baselines.generate, checked against the
stored pred_profile_heldout), it computes:

  pred_t   = mean_m log1p CP10k(pb_counts[t,m])  (evaluator's pseudobulk prediction)
  pool_t   = log1p CP10k(sum_m pb_counts[t,m])   (log of the 32x256-cell pooled pseudobulk)
  obs_t    = log1p CP10k(all cells of the real donor)
  Lbar     = mean over TRAIN donors of obs (train grand-mean profile, evaluator's `grand`)

  (a) err_ig = pred_ig - obs_ig on held-out donors; exact split
      MSE = mean_g ebar_g^2 (shared across held-out donors) + mean (err-ebar)^2 (donor-specific).
      Level bias vs train: B_g = mean_held pred_g - Lbar_g.
  (b) per-cell mean_log bias on TRAIN targets (depth-matched): b_g = mean_t (gen_ml - real_ml),
      standardised z = b/sd, fraction of ||z||^2 inside span(W) (the 50 PCs whose
      per-cell coordinates the base critic sees).  Same for B_g.
  (c) Monte-Carlo variance of pred (var_m/M), Jensen gap pred - pool, and scores of the
      pooled predictor and of the noise-free baseline profile mu.
  (d) per-gene contributions binned by train level, nonzero fraction, between-donor sd.
  (e) train-only recalibration: c_g = mean_{train t}(pred_t - obs_t), applied to held-out pred.
  seed reproducibility of B and b.

Writes per-gene arrays to /ml2/alz/runs/improve_20261109/screen_analysis/gap_arrays_{panel}_fold{k}.npz
and the summary to .../gap_decompose.json.
"""
import os
for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[_v] = "6"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys, json, time
from pathlib import Path
import numpy as np

STUDY = Path("/ml2/alz/analysis/improve_20261109")
sys.path.insert(0, str(STUDY))
sys.dont_write_bytecode = True
import evaluate as E          # noqa: E402
import baselines as BL        # noqa: E402
import metrics as MX          # noqa: E402

RUNS = Path("/ml2/alz/runs/improve_20261109")
OUT = RUNS / "screen_analysis"
OUT.mkdir(parents=True, exist_ok=True)
PANELS = {"hvg2000": ["base", "genemean"], "wide": ["base", "genemean"]}
BASELINE_NAMES = ["grand_mean", "ridge", "shrunk_stage_mean"]


def per_donor_rmse(pred, obs):
    return float(np.mean(np.sqrt(np.mean((pred - obs) ** 2, axis=1))))


def decompose(pred, obs, Lbar):
    err = pred - obs
    ebar = err.mean(0)
    B = pred.mean(0) - Lbar
    return {"pb_rmse": per_donor_rmse(pred, obs),
            "mse": float(np.mean(err ** 2)),
            "mse_shared": float(np.mean(ebar ** 2)),
            "mse_donor_specific": float(np.mean((err - ebar) ** 2)),
            "level_bias_ms": float(np.mean(B ** 2)),          # mean_g B_g^2 (vs train grand mean)
            "level_bias_mean": float(B.mean()),
            "pred_between_target_var": float(np.mean(pred.var(0))),
            "obs_between_donor_var": float(np.mean(obs.var(0)))}, ebar, B


def span_fraction(v_log, sd, W):
    z = v_log / sd
    Q, _ = np.linalg.qr(W)                     # orthonormal basis of span(W)
    zin = Q @ (Q.T @ z)
    zout = z - zin
    xin, xout = zin * sd, zout * sd
    return {"frac_std_in_span": float((zin ** 2).sum() / (z ** 2).sum()),
            "log_ms_in_part": float(np.mean(xin ** 2)), "log_ms_out_part": float(np.mean(xout ** 2)),
            "log_ms_total": float(np.mean(v_log ** 2))}


def bins_report(contrib_by_pred, keys, edges_src, name):
    q = np.quantile(edges_src, [.25, .5, .75])
    b = np.digitize(edges_src, q)
    rep = {"edges": [float(x) for x in q]}
    for k in range(4):
        m = b == k
        rep[f"Q{k + 1}"] = {"n": int(m.sum())}
        for p in keys:
            arr = contrib_by_pred[p]
            rep[f"Q{k + 1}"][p] = {"ms": float(np.mean(arr[m] ** 2)), "mean": float(np.mean(arr[m])),
                                   "share_of_sum_sq": float((arr[m] ** 2).sum() / (arr ** 2).sum())}
    return rep


def main():
    summary = {}
    t0 = time.time()
    for panel, arms in PANELS.items():
        for fold in range(3):
            key = f"{panel}/fold{fold}"
            real = E.load_real(RUNS / "caches" / panel / f"fold{fold}", panel, fold)
            held, tr = real.is_heldout, ~real.is_heldout
            obs = MX.log_cp10k(real.pb)
            Lbar = obs[tr].mean(0)
            ml_tr = real.mean_log[tr].mean(0)
            nz_tr = real.nonzero[tr].mean(0)
            sd_between = obs[tr].std(0)
            preds, pools, mls, mcvar = {}, {}, {}, {}
            for arm in arms:
                for s in (0, 1):
                    g = E.align_generated(E.load_generated(RUNS / "fits" / arm / panel / f"fold{fold}" / f"seed{s}" / "generated.npz"), real)
                    pb = g["pb_counts"].astype(np.float64)
                    gl = MX.log_cp10k(pb)                       # [T,M,G]
                    preds[f"{arm}_s{s}"] = gl.mean(1)
                    mcvar[f"{arm}_s{s}"] = gl.var(1) / gl.shape[1]
                    pools[f"{arm}_s{s}"] = MX.log_cp10k(pb.sum(1))
                    mls[f"{arm}_s{s}"] = g["mean_log"].astype(np.float64).mean(1)
                    stored = json.loads((RUNS / "results" / "fits" / arm / panel / f"fold{fold}" / f"seed{s}.json").read_text())
                    st = stored["subsets"]["all"]["heldout"]["pseudobulk"]["rmse"]
                    mine = per_donor_rmse(preds[f"{arm}_s{s}"][held], obs[held])
                    assert abs(st - mine) < 1e-6, (key, arm, s, st, mine)
                    del pb, gl, g
                preds[f"{arm}_ens"] = (preds[f"{arm}_s0"] + preds[f"{arm}_s1"]) / 2
                pools[f"{arm}_ens"] = (pools[f"{arm}_s0"] + pools[f"{arm}_s1"]) / 2  # avg of logs of two pools
                mls[f"{arm}_ens"] = (mls[f"{arm}_s0"] + mls[f"{arm}_s1"]) / 2
                mcvar[f"{arm}_ens"] = (mcvar[f"{arm}_s0"] + mcvar[f"{arm}_s1"]) / 4
            for name in BASELINE_NAMES:
                gen, _ = BL.generate(name, real)
                pb = gen["pb_counts"].astype(np.float64)
                gl = MX.log_cp10k(pb)
                preds[f"bl_{name}"] = gl.mean(1)
                mcvar[f"bl_{name}"] = gl.var(1) / gl.shape[1]
                pools[f"bl_{name}"] = MX.log_cp10k(pb.sum(1))
                mls[f"bl_{name}"] = gen["mean_log"].astype(np.float64).mean(1)
                with np.load(RUNS / "results" / "baselines" / panel / f"fold{fold}" / f"{name}.npz") as f:
                    sp_ = f["pred_profile_heldout"].astype(np.float64)
                assert np.max(np.abs(sp_ - preds[f"bl_{name}"][held])) < 1e-4, (key, name)
                mu, _, _ = BL.fit_profile_baseline(name, real)
                preds[f"mu_{name}"] = mu                      # noise-free profile (not the evaluator's path)
                del pb, gl, gen
            res = {"n_heldout": int(held.sum()), "n_train": int(tr.sum()), "G": int(len(real.genes)), "preds": {}}
            ebars, Bs, bmls, cal_c = {}, {}, {}, {}
            for p, P in preds.items():
                d, ebar, B = decompose(P[held], obs[held], Lbar)
                ebars[p], Bs[p] = ebar, B
                # (e) train-only additive recalibration
                c = (P[tr] - obs[tr]).mean(0)
                cal_c[p] = c
                d["pb_rmse_recal_train"] = per_donor_rmse(P[held] - c, obs[held])
                d_cal, _, _ = decompose(P[held] - c, obs[held], Lbar)
                d["recal_mse_shared"] = d_cal["mse_shared"]; d["recal_mse_donor_specific"] = d_cal["mse_donor_specific"]
                # recal using level only (B on train targets) is the same as above up to obs train mean; also
                # a "replace level" variant: keep only the model's between-target deviations around Lbar
                d["pb_rmse_level_replaced_by_Lbar"] = per_donor_rmse(P[held] - P[tr].mean(0) + Lbar, obs[held])
                if p in pools:
                    d["pb_rmse_pooled_predictor"] = per_donor_rmse(pools[p][held], obs[held])
                    cp = (pools[p][tr] - obs[tr]).mean(0)
                    d["pb_rmse_pooled_recal_train"] = per_donor_rmse(pools[p][held] - cp, obs[held])
                    d["jensen_gap_mean"] = float(np.mean(P[held] - pools[p][held]))
                    d["jensen_gap_ms"] = float(np.mean((P[held] - pools[p][held]) ** 2))
                    d["mc_var_of_pred_mean"] = float(np.mean(mcvar[p][held]))
                if p in mls:
                    bml = (mls[p][tr] - real.mean_log[tr]).mean(0)     # depth-matched per-cell mean-log bias, train targets
                    bmls[p] = bml
                    d["meanlog_bias_train_ms"] = float(np.mean(bml ** 2))
                    d["span_meanlog_bias_train"] = span_fraction(bml, real.sd, real.W)
                    bmlh = (mls[p][held] - real.mean_log[held]).mean(0)
                    d["meanlog_rmse_gene_heldout"] = float(np.sqrt(np.mean(bmlh ** 2)))
                    # train-only recal of mean_log
                    d["meanlog_rmse_gene_heldout_recal"] = float(np.sqrt(np.mean((bmlh - bml) ** 2)))
                d["span_pb_level_bias"] = span_fraction(B, real.sd, real.W)
                res["preds"][p] = d
            # (d) bins on shared error ebar and level bias B
            res["bins_level"] = {"ebar": bins_report(ebars, list(ebars), Lbar, "level"),
                                 "B": bins_report(Bs, list(Bs), Lbar, "level")}
            res["bins_nonzero"] = {"ebar": bins_report(ebars, list(ebars), nz_tr, "nonzero")}
            res["bins_between_sd"] = {"ebar": bins_report(ebars, list(ebars), sd_between, "sd")}
            # correlation of model level bias with gene level / sparsity
            res["corr_B_with_level"] = {p: float(np.corrcoef(Bs[p], Lbar)[0, 1]) for p in Bs}
            # seed reproducibility
            rep = {}
            for arm in arms:
                for nm, D in [("B_pb", Bs), ("b_meanlog_train", bmls), ("ebar", ebars)]:
                    a, b = D[f"{arm}_s0"], D[f"{arm}_s1"]
                    seed_var = float(np.mean((a - b) ** 2) / 2)
                    ms_mean = float(np.mean(((a + b) / 2) ** 2))
                    rep[f"{arm}:{nm}"] = {"pearson_s0_s1": float(np.corrcoef(a, b)[0, 1]),
                                          "ms_seed0": float(np.mean(a ** 2)), "ms_seed1": float(np.mean(b ** 2)),
                                          "ms_seed_specific_per_seed": seed_var,
                                          "ms_shared_est": ms_mean - seed_var / 2,
                                          "ms_of_ensemble_mean": ms_mean}
                rep[f"{arm}:corr_B_with_grandmean_B"] = float(np.corrcoef(Bs[f"{arm}_ens"], Bs["bl_grand_mean"])[0, 1])
            res["seed_reproducibility"] = rep
            summary[key] = res
            np.savez_compressed(OUT / f"gap_arrays_{panel}_fold{fold}.npz", genes=real.genes, Lbar=Lbar, nz_tr=nz_tr,
                                sd_between=sd_between, ml_tr=ml_tr, sd=real.sd,
                                **{f"B_{p}": v.astype(np.float32) for p, v in Bs.items()},
                                **{f"ebar_{p}": v.astype(np.float32) for p, v in ebars.items()},
                                **{f"bml_{p}": v.astype(np.float32) for p, v in bmls.items()},
                                **{f"cal_{p}": v.astype(np.float32) for p, v in cal_c.items()})
            print(f"[{time.time() - t0:.0f}s] {key} done", flush=True)
            (OUT / "gap_decompose.json").write_text(json.dumps(summary, indent=1))
    (OUT / "gap_decompose.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
