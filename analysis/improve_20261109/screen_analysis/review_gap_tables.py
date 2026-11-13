"""review_gap_tables.py - fold-mean summary of review_gap_core.json (prints only)."""
import json, sys
from pathlib import Path
import numpy as np

J = json.loads(Path("/ml2/alz/runs/improve_20261109/screen_analysis/review_gap_core.json").read_text())


def fm(panel, p, path):
    v = []
    for f in range(3):
        k = f"{panel}/fold{f}"
        if k not in J:
            continue
        x = J[k]["preds"][p]
        for q in path:
            x = x[q]
        v.append(x)
    return np.mean(v, 0), v


for panel in ["hvg2000", "wide"]:
    if f"{panel}/fold0" not in J:
        continue
    folds = [k for k in J if k.startswith(panel)]
    print(f"\n## {panel} ({len(folds)} folds)")
    for k in folds:
        r = J[k]
        print(k, "stored vs mine:", {a: round(r[f"stored_pb_rmse_{a}"] - r["preds"][a]["pb_rmse"], 8) for a in ["base_s0", "base_s1", "genemean_s0", "genemean_s1"]},
              "bl maxdiff:", {b: r.get(f"bl_{b}_maxdiff_vs_stored") for b in ["grand_mean", "ridge", "shrunk_stage_mean"]},
              "Lbar", round(r["Lbar_score"], 4), "logpooled", round(r["logpooled_score"], 4),
              "real pbK-obs", round(r["real_pbK_minus_obs_mean"], 4), "stage held", r["stage_held"], "train", r["stage_train"])
    preds = list(J[f"{panel}/fold0"]["preds"])
    gm = fm(panel, "bl_grand_mean", ["pb_rmse"])[0] if "bl_grand_mean" in preds else None
    print("pred | pb_rmse | mse shared donor | B_ms B_mean | predvar mc | recal rmse | closure% | recal_shared | corr(c_tr,ebar_h) | pred-pool | pool rmse")
    for p in preds:
        g = lambda *a: fm(panel, p, list(a))[0]
        rm, rc = g("pb_rmse"), g("recal_pb_rmse")
        clo = 100 * (rm - rc) / (rm - gm) if gm is not None and rm - gm > 0.01 else float("nan")
        pp = g("pred_minus_pool_mean") if "pred_minus_pool_mean" in J[f"{panel}/fold0"]["preds"][p] else float("nan")
        pr = g("pool_pb_rmse") if "pool_pb_rmse" in J[f"{panel}/fold0"]["preds"][p] else float("nan")
        print(f"{p:22s} {rm:.3f} | {g('mse'):.3f} {g('shared'):.3f} {g('donor_specific'):.3f} | {g('B_ms'):.3f} {g('B_mean'):+.3f} | "
              f"{g('pred_var_held'):.3f} {g('mc_var_held'):.4f} | {rc:.3f} | {clo:.0f} | {g('recal_shared'):.3f} | {g('corr_c_train_vs_ebar_held'):.2f} | {pp:+.3f} | {pr:.3f}")
    print("pred | meanlog_rmse held, recal | ml bias ms | span ml: frac perm_null dim | span B: frac perm_null | corr B lvl | B by Q | Q4 share ebar | pbK bias ms | shape ms | shape by Q | lambda")
    for p in preds:
        g = lambda *a: fm(panel, p, list(a))[0]
        print(f"{p:22s} {g('meanlog_rmse_gene_held'):.3f} {g('meanlog_rmse_gene_held_recal'):.3f} | {g('meanlog_bias_train_ms'):.4f} | "
              f"{g('span_meanlog_bias_train','frac'):.3f} {g('span_meanlog_bias_train','perm_null_mean'):.3f} {g('span_meanlog_bias_train','dim_ratio'):.3f} | "
              f"{g('span_B','frac'):.3f} {g('span_B','perm_null_mean'):.3f} | {g('corr_B_level'):+.2f} | {np.round(g('B_by_Q'), 3)} | {g('ebar_Q4_share'):.2f} | "
              f"{g('pbK_bias_train_ms'):.3f} | {g('shape_gap_ms'):.3f} | {np.round(g('shape_gap_by_Q'), 3)} | "
              + " ".join(f"{l}:{g('lambda', l):.3f}" for l in ["0", "0.25", "0.5", "1"]))
    print("seed reproducibility per fold (pearson / shared / seed-specific):")
    for k in folds:
        print(" ", k, {a: (round(v["pearson"], 2), round(v["shared_ms_est"], 4), round(v["seed_specific_ms"], 4)) for a, v in J[k]["seed_rep"].items()})
    print("stage deviations: real-train vs real-heldout pearson, model vs real-heldout, var ratio:")
    for k in folds:
        for s, v in J[k]["stage_dev"].items():
            print(" ", k, "stage", s, f"n_tr {v['n_tr']} n_h {v['n_h']} realtr~realheld {v['pearson_realtrain_vs_realheld']:+.2f}",
                  " ".join(f"{a}:{v[a]['vs_realheld']:+.2f}/{v[a]['vs_realtrain']:.2f}/{v[a]['var_ratio_vs_train']:.1f}" for a in ["base_ens", "genemean_ens"]))
