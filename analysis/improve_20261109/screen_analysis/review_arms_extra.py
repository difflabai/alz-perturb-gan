"""Review extras (own code): absolute seed-difference D per arm, pb_dev_pearson
for covars arms, M-matched PC-space genemean-base differences, sensitivity of
the infinite-seed projection to D, and the depth floor of pb_rmse (a perfect
self-model of each held-out donor scored with the evaluator's definition:
mean over 32 bootstrap 256-cell pseudobulks of log1p CP10k vs the donor's
full-depth pseudobulk).
Writes OUT/review_arms_extra.json and prints markdown.
"""
from review_arms_common import *  # noqa

R = json.loads((OUT / "review_arms_recompute.json").read_text())
A = json.loads((OUT / "review_arms_analysis.json").read_text())
res = {}
print("## D (mean squared seed difference of held-out pb predictions), MC part, pb_dev_pearson\n")
print("| panel | fold | arm | D | V0+V1 (MC) | D - MC | dev_pearson heldout s0/s1 | dev_pearson train s0/s1 |\n|---|---|---|---|---|---|---|---|")
for panel, arms in ARMS.items():
    for f in range(3):
        z = np.load(OUT / f"review_arms_profiles_{panel}_fold{f}.npz")
        held = z["held"]; obs = z["obs"].astype(float); grand = obs[~held].mean(0)
        for arm in arms:
            p0 = z[f"{arm}|0|P"].astype(float); p1 = z[f"{arm}|1|P"].astype(float)
            D = float(((p0 - p1)[held] ** 2).mean())
            V = float((z[f"{arm}|0|V"].astype(float) + z[f"{arm}|1|V"].astype(float))[held].mean())
            dp = {}
            for split, mask in [("heldout", held), ("train", ~held)]:
                dp[split] = [float(np.mean([np.corrcoef(P[i] - grand, obs[i] - grand)[0, 1] for i in np.where(mask)[0]])) for P in (p0, p1)]
            res[f"{arm}|{panel}|{f}"] = dict(D=D, MC=V, dev_pearson=dp)
            print(f"| {panel} | {f} | {arm} | {D:.5f} | {V:.5f} | {D - V:.5f} | {dp['heldout'][0]:+.3f}/{dp['heldout'][1]:+.3f} | {dp['train'][0]:+.3f}/{dp['train'][1]:+.3f} |")

print("\n## M-matched (16+16 distinct draws) ensemble, genemean - base, PC-space (held-out / train)\n")
for panel in ARMS:
    for split in ["heldout", "train"]:
        row = []
        for m in ["pc_energy", "pc_coverage80", "cell_energy"]:
            d = [R["ens_mm_indep"][f"genemean|{panel}|{f}"][split][m] - R["ens_mm_indep"][f"base|{panel}|{f}"][split][m] for f in range(3)]
            res[f"mm|{panel}|{split}|{m}"] = d
            row.append(f"{m} " + "/".join(f"{x:+.3f}" for x in d))
        print(f"- {panel} {split}: " + "; ".join(row))

print("\n## Sensitivity of the infinite-seed pb_rmse projection to D (held-out); D scaled by 0.5 / 1 / 2\n")
for panel, arms in ARMS.items():
    for f in range(3):
        z = np.load(OUT / f"review_arms_profiles_{panel}_fold{f}.npz")
        held = z["held"]; obs = z["obs"][held].astype(float)
        p0 = z["genemean|0|P"][held].astype(float); p1 = z["genemean|1|P"][held].astype(float)
        ms = (((p0 - obs) ** 2).mean(1) + ((p1 - obs) ** 2).mean(1)) / 2; D = ((p0 - p1) ** 2).mean(1)
        vals = [float(np.sqrt(np.maximum(ms - c * D / 2, 0)).mean()) for c in (0.5, 1, 2)]
        bl = A[f"B|genemean|{panel}|{f}"]["best_baseline"]
        res[f"sens|{panel}|{f}"] = dict(inf=vals, baseline=bl)
        print(f"- genemean {panel} fold{f}: inf-seed pb_rmse {vals[0]:.3f} / {vals[1]:.3f} / {vals[2]:.3f} vs {bl[0]} {bl[1]:.3f}")

print("\n## Depth floor of pb_rmse: perfect self-model of each held-out donor (32 x 256-cell bootstrap pseudobulks)\n")
rng = np.random.default_rng(20261110)
for panel in ARMS:
    fl = []
    for f in range(3):
        d = RUNS / "caches" / panel / f"fold{f}" / "heldout"
        m = np.load(d / "metadata.npz"); C = sp.load_npz(d / "counts.npz").tocsr(); cd = m["donor"].astype(str)
        per, ncell = [], []
        for dn in m["donors"].astype(str):
            X = C[np.where(cd == dn)[0]].toarray().astype(np.float64)
            full = lcp(X.sum(0))
            boots = np.stack([X[rng.integers(0, len(X), 256)].sum(0) for _ in range(32)])
            per.append(np.sqrt(((lcp(boots).mean(0) - full) ** 2).mean())); ncell.append(len(X))
        fl.append(float(np.mean(per)))
        res[f"floor|{panel}|{f}"] = dict(pb_rmse_floor=float(np.mean(per)), median_cells=float(np.median(ncell)))
        print(f"- {panel} fold{f}: self-model pb_rmse floor {np.mean(per):.3f} (median cells per held-out donor {np.median(ncell):.0f})")
(OUT / "review_arms_extra.json").write_text(json.dumps(res, indent=1))
