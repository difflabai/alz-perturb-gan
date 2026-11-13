"""Review: recompute primary metrics for ALL 48 fits, 24 seed-ensembles and two
M-matched ensemble variants from generated.npz + caches with own code, and
compare with the evaluator JSONs. Also saves held-out predicted profiles,
within-M Monte-Carlo variances and the cross-seed CRN correlation needed by
review_arms_analysis.py.

Outputs: OUT/review_arms_recompute.json, OUT/review_arms_profiles_{panel}_fold{k}.npz
"""
from review_arms_common import *  # noqa

RES = RUNS / "results"


def dig(d, *p):
    for k in p:
        if d is None:
            return None
        d = d.get(k)
    return d


def ev_val(js, metric, split):
    if metric in ("pc_energy", "pc_coverage80", "cell_energy"):
        key = {"pc_energy": "energy_score", "pc_coverage80": "coverage80", "cell_energy": "cell_energy_distance"}[metric]
        return dig(js, "pc_space", split, key)
    path = {"pb_rmse": ("pseudobulk", "rmse"), "meanlog_rmse_gene": ("mean_log", "rmse_gene"),
            "nonzero_rmse_gene": ("nonzero", "rmse_gene"),
            "hn_spearman_vs_heldout": ("contrast", "high_notad_vs_heldout", "spearman"),
            "hn_spearman_vs_train": ("contrast", "high_notad_vs_train", "spearman")}[metric]
    return dig(js, "subsets", "all", split, *path)


def main():
    out = {"fits": {}, "ens": {}, "ens_mm_crn": {}, "ens_mm_indep": {}, "shared": {}, "maxdiff": {}, "crn": {}}
    maxdiff = {}
    for panel, arms in ARMS.items():
        for fold in range(3):
            real = Real(panel, fold)
            other = "wide" if panel == "hvg2000" else "hvg2000"
            og = set(np.load(RUNS / "caches" / other / f"fold{fold}" / "space.npz")["genes"].astype(str))
            shared = np.array([i for i, g in enumerate(real.genes) if g in og])
            # shared genes in a common (sorted name) order so panels can be compared gene by gene
            shared = shared[np.argsort(real.genes[shared])]
            arrays = {"donors": real.donors, "held": real.held, "stage": real.stage,
                      "obs": real.obs.astype(np.float32), "genes": real.genes,
                      "obs_shared": lcp(real.pb[:, shared]).astype(np.float32), "shared_genes": real.genes[shared]}
            for arm in arms:
                gs = []
                for seed in range(2):
                    g = load_gen(arm, panel, fold, seed, real); gs.append(g)
                    js = json.loads((RES / "fits" / arm / panel / f"fold{fold}" / f"seed{seed}.json").read_text())
                    key = f"{arm}|{panel}|{fold}|{seed}"
                    out["fits"][key] = {}
                    for split in ["heldout", "train"]:
                        r, P = score(g, real, split, rng_seed=1000 + seed)
                        out["fits"][key][split] = r
                        for m, v in list(r.items()):
                            ev = ev_val(js, m, split)
                            if v is not None and ev is not None:
                                maxdiff.setdefault(m, []).append(abs(v - ev))
                                out["fits"][key][split][m + "_evaluator"] = ev
                    r, _ = score(g, real, "heldout", idx=shared, pc=False)
                    out["shared"][key] = r["pb_rmse"]
                    L = np.stack([lcp(g["pb_counts"][t]) for t in range(len(real.donors))])  # [T,M,G]
                    arrays[f"{arm}|{seed}|P"] = L.mean(1).astype(np.float32)
                    arrays[f"{arm}|{seed}|V"] = (L.var(1, ddof=1) / L.shape[1]).astype(np.float32)  # MC var of the mean
                    arrays[f"{arm}|{seed}|dev"] = (L - L.mean(1, keepdims=True))[real.held].astype(np.float16)
                    arrays[f"{arm}|{seed}|Pshared"] = pred_profiles(g["pb_counts"], shared).astype(np.float32)
                # CRN: correlation of per-generated-donor deviations at the same m across seeds (held-out)
                d0 = arrays.pop(f"{arm}|0|dev").astype(np.float64); d1 = arrays.pop(f"{arm}|1|dev").astype(np.float64)
                out["crn"][f"{arm}|{panel}|{fold}"] = float((d0 * d1).sum() / np.sqrt((d0 ** 2).sum() * (d1 ** 2).sum()))
                lib_same = bool(np.allclose(gs[0]["lib"], gs[1]["lib"]))
                out["crn"][f"{arm}|{panel}|{fold}|lib_identical_across_seeds"] = lib_same
                ejs = json.loads((RES / "ensembles" / arm / panel / f"fold{fold}.json").read_text())
                key = f"{arm}|{panel}|{fold}"
                for name, sel in [("ens", None),
                                  ("ens_mm_crn", [np.arange(16), np.arange(16)]),
                                  ("ens_mm_indep", [np.arange(16), np.arange(16, 32)])]:
                    e = concat(gs, sel)
                    out[name][key] = {}
                    for split in ["heldout", "train"]:
                        r, _ = score(e, real, split, rng_seed=7)
                        out[name][key][split] = r
                        if name == "ens":
                            for m, v in list(r.items()):
                                ev = ev_val(ejs, m, split)
                                if v is not None and ev is not None:
                                    maxdiff.setdefault("ENS_" + m, []).append(abs(v - ev))
                                    out[name][key][split][m + "_evaluator"] = ev
                    if name == "ens":
                        out["shared"][key + "|ens"] = score(e, real, "heldout", idx=shared, pc=False)[0]["pb_rmse"]
                print(panel, fold, arm, "done", flush=True)
            # baselines: stored held-out predicted profiles (evaluator output), rescored here
            for b in ["grand_mean", "ridge", "shrunk_stage_mean", "stage_mean", "resample"]:
                z = np.load(RES / "baselines" / panel / f"fold{fold}" / f"{b}.npz")
                hp = z["pred_profile_heldout"].astype(np.float64)
                assert np.array_equal(z["donors"].astype(str), real.donors)
                arrays[f"BL|{b}"] = hp.astype(np.float32)
                out.setdefault("baseline_pb", {})[f"{b}|{panel}|{fold}"] = float(
                    np.mean(np.sqrt(((hp - real.obs[real.held]) ** 2).mean(1))))
            np.savez_compressed(OUT / f"review_arms_profiles_{panel}_fold{fold}.npz", **arrays)
    out["maxdiff"] = {m: [float(np.max(v)), float(np.median(v)), len(v)] for m, v in maxdiff.items()}
    (OUT / "review_arms_recompute.json").write_text(json.dumps(out, indent=1))
    for m, v in out["maxdiff"].items():
        print(f"{m:32s} max|own-evaluator|={v[0]:.2e} median={v[1]:.2e} n={v[2]}")


if __name__ == "__main__":
    main()
