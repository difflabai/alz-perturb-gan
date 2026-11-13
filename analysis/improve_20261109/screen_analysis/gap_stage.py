"""gap_stage.py - where the generators' donor-specific (between-target) error comes from.

For base / genemean (both panels, 3 folds, seeds 0/1 and the seed-ensemble) it splits the
between-held-out-target variance of the evaluator's prediction
    pred_t = mean_m log1p CP10k(pb_counts[t,m])
into a between-stage part (variance of the model's per-stage mean prediction, weighted by
held-out targets) and a within-stage part (same stage, different target: library-size pool +
Monte-Carlo noise), and compares with the MC variance var_m/M.

It then scores train-only-recalibrated predictions with the model's stage deviations scaled by
lambda in {0, 0.25, 0.5, 1}:
    pred_lambda_t = Lbar + lambda * (S_model[stage_t] - mean_train_targets(pred)) ,
where S_model[s] is the model's mean pred over ALL train+held-out targets of stage s (no
held-out OBSERVED data is used; targets' stage and library pool are inputs the model receives
anyway). lambda=0 equals the train grand mean of donor log profiles (count-noise-free); for
reference the same is done for the evaluator's grand_mean prediction is in gap_decompose.json.

Stage-deviation agreement: Pearson between the model's stage deviation profiles and the
TRAIN real stage-mean deviations (in-sample, what the model was trained on) and the HELD-OUT
real stage-mean deviations; and the variance ratio model/real-train (amplification).

Output: /ml2/alz/runs/improve_20261109/screen_analysis/gap_stage.json
"""
import os
for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[_v] = "6"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys, json
from pathlib import Path
import numpy as np

STUDY = Path("/ml2/alz/analysis/improve_20261109")
sys.path.insert(0, str(STUDY))
sys.dont_write_bytecode = True
import evaluate as E          # noqa: E402
import metrics as MX          # noqa: E402

RUNS = Path("/ml2/alz/runs/improve_20261109")
OUT = RUNS / "screen_analysis"
LAMBDAS = [0.0, 0.25, 0.5, 1.0]


def rmse(pred, obs):
    return float(np.mean(np.sqrt(np.mean((pred - obs) ** 2, axis=1))))


def main():
    out = {}
    for panel in ["hvg2000", "wide"]:
        for fold in range(3):
            real = E.load_real(RUNS / "caches" / panel / f"fold{fold}", panel, fold)
            held, tr = real.is_heldout, ~real.is_heldout
            st = real.stage
            obs = MX.log_cp10k(real.pb)
            Lbar = obs[tr].mean(0)
            res = {}
            for arm in ["base", "genemean"]:
                P, V = {}, {}
                for s in (0, 1):
                    g = E.align_generated(E.load_generated(RUNS / "fits" / arm / panel / f"fold{fold}" / f"seed{s}" / "generated.npz"), real)
                    gl = MX.log_cp10k(g["pb_counts"].astype(np.float64))
                    P[s] = gl.mean(1); V[s] = gl.var(1) / gl.shape[1]
                    del gl, g
                P["ens"] = (P[0] + P[1]) / 2; V["ens"] = (V[0] + V[1]) / 4
                stages = sorted(set(st.tolist()))
                for k in [0, 1, "ens"]:
                    p = P[k]
                    Smod = {s: p[st == s].mean(0) for s in stages}
                    ph = p[held]
                    tot = float(np.mean(ph.var(0)))
                    stage_part = np.stack([Smod[s] for s in st[held]])  # per held-out target, model stage mean (all targets)
                    hs = np.stack([ph[st[held] == s].mean(0) for s in st[held]])
                    between = float(np.mean(((hs - ph.mean(0)) ** 2).mean(0)))
                    within = float(np.mean(((ph - hs) ** 2).mean(0)))
                    mc = float(np.mean(V[k][held]))
                    base_level = p[tr].mean(0)
                    lam = {}
                    for L_ in LAMBDAS:
                        pred = Lbar + L_ * (stage_part - base_level)
                        lam[str(L_)] = rmse(pred, obs[held])
                    # raw recal (keeps within-stage variation) for reference
                    c = (p[tr] - obs[tr]).mean(0)
                    lam["recal_full"] = rmse(p[held] - c, obs[held])
                    # stage deviation agreement
                    dev_mod = {s: Smod[s] - p.mean(0) for s in stages}
                    rt = {s: obs[tr & (st == s)].mean(0) - obs[tr].mean(0) for s in stages if (tr & (st == s)).any()}
                    rh = {s: obs[held & (st == s)].mean(0) - obs[held].mean(0) for s in stages if (held & (st == s)).any()}
                    agree = {}
                    for s in stages:
                        a = {}
                        if s in rt:
                            a["pearson_vs_train_real"] = float(np.corrcoef(dev_mod[s], rt[s])[0, 1])
                            a["var_ratio_model_over_train_real"] = float(dev_mod[s].var() / rt[s].var())
                            a["n_train"] = int((tr & (st == s)).sum())
                        if s in rh:
                            a["pearson_vs_heldout_real"] = float(np.corrcoef(dev_mod[s], rh[s])[0, 1])
                            a["n_heldout"] = int((held & (st == s)).sum())
                        a["model_dev_ms"] = float(np.mean(dev_mod[s] ** 2))
                        agree[str(s)] = a
                    res[f"{arm}_{k if k == 'ens' else 's' + str(k)}"] = {
                        "heldout_pred_var_total": tot, "between_stage": between, "within_stage": within,
                        "mc_var": mc, "rmse_recal_lambda": lam, "stage_dev_agreement": agree}
            # real reference: how much of held-out obs variance is between stage
            oh = obs[held]
            hs = np.stack([oh[st[held] == s].mean(0) for s in st[held]])
            res["real_heldout"] = {"var_total": float(np.mean(oh.var(0))),
                                   "between_stage": float(np.mean(((hs - oh.mean(0)) ** 2).mean(0))),
                                   "stage_counts_heldout": {str(s): int((st[held] == s).sum()) for s in range(4)},
                                   "stage_counts_train": {str(s): int((st[tr] == s).sum()) for s in range(4)}}
            out[f"{panel}/fold{fold}"] = res
            print(panel, fold, "done", flush=True)
    (OUT / "gap_stage.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
