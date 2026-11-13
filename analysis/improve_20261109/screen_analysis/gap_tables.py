"""gap_tables.py - fold-balanced tables from gap_decompose.json and gap_stage.json.

Seeds are averaged within fold for single-seed rows ("s" = mean of s0 and s1), then the three
fold values are averaged (fold-balanced mean, as in results/summary.md). Prints markdown tables.
"""
import json
from pathlib import Path
import numpy as np

OUT = Path("/ml2/alz/runs/improve_20261109/screen_analysis")
D = json.loads((OUT / "gap_decompose.json").read_text())
S = json.loads((OUT / "gap_stage.json").read_text()) if (OUT / "gap_stage.json").exists() else None


def fm(panel, pred, getter):
    vals = []
    for f in range(3):
        d = D[f"{panel}/fold{f}"]
        if pred.endswith("_s"):
            v = [getter(d["preds"][pred + "0"]), getter(d["preds"][pred + "1"])]
            vals.append(np.mean(v))
        else:
            vals.append(getter(d["preds"][pred]))
    return np.mean(vals), np.min(vals), np.max(vals)


def fmt(t):
    return f"{t[0]:.3f} [{t[1]:.3f},{t[2]:.3f}]"


PREDS = ["base_s", "base_ens", "genemean_s", "genemean_ens", "bl_grand_mean", "bl_ridge", "bl_shrunk_stage_mean",
         "mu_grand_mean", "mu_ridge", "mu_shrunk_stage_mean"]

for panel in ["hvg2000", "wide"]:
    print(f"\n### {panel}: (a)+(e) error decomposition, fold mean [min,max over folds]\n")
    cols = [("pb_rmse", lambda p: p["pb_rmse"]), ("MSE", lambda p: p["mse"]),
            ("shared", lambda p: p["mse_shared"]), ("donor-specific", lambda p: p["mse_donor_specific"]),
            ("level bias ms B^2", lambda p: p["level_bias_ms"]), ("mean B", lambda p: p["level_bias_mean"]),
            ("pred var across held-out targets", lambda p: p["pred_between_target_var"]),
            ("pb_rmse train-recal", lambda p: p["pb_rmse_recal_train"]),
            ("shared after recal", lambda p: p["recal_mse_shared"])]
    print("| prediction | " + " | ".join(c for c, _ in cols) + " |")
    print("|---" * (len(cols) + 1) + "|")
    for p in PREDS:
        print(f"| {p} | " + " | ".join(fmt(fm(panel, p, g)) for _, g in cols) + " |")
    print(f"\n### {panel}: (b)(c) span, Jensen, MC, mean_log\n")
    cols2 = [("meanlog bias ms (train targets)", lambda p: p["meanlog_bias_train_ms"]),
             ("frac std meanlog bias in 50-PC span", lambda p: p["span_meanlog_bias_train"]["frac_std_in_span"]),
             ("frac std pb bias B in span", lambda p: p["span_pb_level_bias"]["frac_std_in_span"]),
             ("meanlog_rmse_gene heldout", lambda p: p["meanlog_rmse_gene_heldout"]),
             ("meanlog_rmse_gene heldout train-recal", lambda p: p["meanlog_rmse_gene_heldout_recal"]),
             ("Jensen mean(pred-pool)", lambda p: p["jensen_gap_mean"]),
             ("MC var of pred", lambda p: p["mc_var_of_pred_mean"]),
             ("pb_rmse pooled predictor", lambda p: p["pb_rmse_pooled_predictor"]),
             ("pb_rmse pooled + recal", lambda p: p["pb_rmse_pooled_recal_train"])]
    print("| prediction | " + " | ".join(c for c, _ in cols2) + " |")
    print("|---" * (len(cols2) + 1) + "|")
    for p in PREDS[:7]:
        print(f"| {p} | " + " | ".join(fmt(fm(panel, p, g)) for _, g in cols2) + " |")
    print(f"\n### {panel}: seed reproducibility (per fold: pearson s0-s1, shared ms, seed-specific ms per seed)\n")
    for arm in ["base", "genemean"]:
        for nm in ["B_pb", "b_meanlog_train"]:
            r = [D[f"{panel}/fold{f}"]["seed_reproducibility"][f"{arm}:{nm}"] for f in range(3)]
            print(f"- {arm} {nm}: pearson " + ", ".join(f"{x['pearson_s0_s1']:.2f}" for x in r)
                  + "; shared ms " + ", ".join(f"{x['ms_shared_est']:.4f}" for x in r)
                  + "; seed-specific ms " + ", ".join(f"{x['ms_seed_specific_per_seed']:.4f}" for x in r))
        print(f"- {arm} corr(B_ens, B_grand_mean_baseline): " + ", ".join(
            f"{D[f'{panel}/fold{f}']['seed_reproducibility'][f'{arm}:corr_B_with_grandmean_B']:.2f}" for f in range(3)))
    print(f"\n### {panel}: corr(B, train level Lbar) per fold\n")
    for p in ["base_ens", "genemean_ens", "bl_grand_mean"]:
        print(f"- {p}: " + ", ".join(f"{D[f'{panel}/fold{f}']['corr_B_with_level'][p]:.2f}" for f in range(3)))
    print(f"\n### {panel}: (d) shared held-out error ebar by gene bins (fold mean of ms; share of sum sq)\n")
    for binname, key in [("train level Lbar", "bins_level"), ("train nonzero frac", "bins_nonzero"),
                         ("between-donor sd", "bins_between_sd")]:
        print(f"\n{binname}:\n")
        print("| pred | " + " | ".join(f"Q{q} ms (mean) share" for q in range(1, 5)) + " |")
        print("|---" * 5 + "|")
        for p in ["base_s0", "base_ens", "genemean_s0", "genemean_ens", "bl_grand_mean"]:
            cells = []
            for q in range(1, 5):
                ms = np.mean([D[f"{panel}/fold{f}"][key]["ebar"][f"Q{q}"][p]["ms"] for f in range(3)])
                mn = np.mean([D[f"{panel}/fold{f}"][key]["ebar"][f"Q{q}"][p]["mean"] for f in range(3)])
                sh = np.mean([D[f"{panel}/fold{f}"][key]["ebar"][f"Q{q}"][p]["share_of_sum_sq"] for f in range(3)])
                cells.append(f"{ms:.3f} ({mn:+.3f}) {sh:.2f}")
            print(f"| {p} | " + " | ".join(cells) + " |")
        edges = [D[f"{panel}/fold{f}"][key]["ebar"]["edges"] for f in range(3)]
        print(f"\nquartile edges fold0: {[round(x, 3) for x in edges[0]]}")
    print(f"\n### {panel}: level-bias B by train level bins (fold mean ms (mean))\n")
    for p in ["base_ens", "genemean_ens", "bl_grand_mean"]:
        cells = []
        for q in range(1, 5):
            ms = np.mean([D[f"{panel}/fold{f}"]["bins_level"]["B"][f"Q{q}"][p]["ms"] for f in range(3)])
            mn = np.mean([D[f"{panel}/fold{f}"]["bins_level"]["B"][f"Q{q}"][p]["mean"] for f in range(3)])
            cells.append(f"{ms:.3f} ({mn:+.3f})")
        print(f"- {p}: " + " | ".join(cells))

if S:
    for panel in ["hvg2000", "wide"]:
        print(f"\n### {panel}: (stage) between-target variance and lambda-scaled stage deviations (fold mean)\n")
        keys = ["base_s0", "base_s1", "base_ens", "genemean_s0", "genemean_s1", "genemean_ens"]
        print("| pred | var total | between-stage | within-stage | MC var | rmse lam0 | lam.25 | lam.5 | lam1 | recal_full |")
        print("|---" * 10 + "|")
        for k in keys:
            r = [S[f"{panel}/fold{f}"][k] for f in range(3)]
            v = lambda f_: np.mean([f_(x) for x in r])
            print(f"| {k} | {v(lambda x: x['heldout_pred_var_total']):.4f} | {v(lambda x: x['between_stage']):.4f} | "
                  f"{v(lambda x: x['within_stage']):.4f} | {v(lambda x: x['mc_var']):.4f} | "
                  + " | ".join(f"{v(lambda x, L=L: x['rmse_recal_lambda'][L]):.3f}" for L in ["0.0", "0.25", "0.5", "1.0", "recal_full"]) + " |")
        rh = [S[f"{panel}/fold{f}"]["real_heldout"] for f in range(3)]
        print(f"\nreal held-out obs: var total {np.mean([x['var_total'] for x in rh]):.4f}, between-stage "
              f"{np.mean([x['between_stage'] for x in rh]):.4f}; held-out stage counts "
              + "; ".join(str(x['stage_counts_heldout']) for x in rh))
        print("\nstage deviation agreement (per fold, stage: pearson vs train real / vs heldout real / var ratio model:train-real):")
        for k in ["base_ens", "genemean_ens"]:
            for f in range(3):
                a = S[f"{panel}/fold{f}"][k]["stage_dev_agreement"]
                print(f"- {k} fold{f}: " + "; ".join(
                    f"s{s}: {x.get('pearson_vs_train_real', float('nan')):.2f}/{x.get('pearson_vs_heldout_real', float('nan')):.2f}/"
                    f"{x.get('var_ratio_model_over_train_real', float('nan')):.2f}" for s, x in a.items()))
