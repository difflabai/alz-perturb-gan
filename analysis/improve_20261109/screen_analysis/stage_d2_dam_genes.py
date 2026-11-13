"""(d, follow-up) Do the models reproduce the High-minus-NotAD effect of the DAM genes that
are sign-consistent in real data (stage_b / stage_b2)? Per arm, panel, fold, seed: the
model contrast (all 37 targets, predicted profiles as evaluate.py) for each gene, next to
the fold's real TRAIN contrast (what the model could have learned) and the fold's real
held-out contrast. Mean over folds (and seeds for models).
Output: printed table; OUT/d2_dam_genes.json.
"""
from stage_common import *

GENES = ["GPNMB", "CD9", "LGALS3", "LPL", "TYROBP", "APOE", "SPP1", "CD74"]
rows = {}
for panel in ["hvg2000", "wide"]:
    for f in range(3):
        c = load_cache(panel, f); P = log_cp10k(c["pb"]); h = c["is_heldout"]; st = c["stage"]
        gl = list(c["genes"]); gi = {g: gl.index(g) for g in GENES if g in gl}
        ctr = contrasts(P[~h], st[~h])["hn"]; cho = contrasts(P[h], st[h])["hn"]
        for g, i in gi.items():
            rows.setdefault((panel, "real train"), {}).setdefault(g, []).append(ctr[i])
            rows.setdefault((panel, "real held-out"), {}).setdefault(g, []).append(cho[i])
        for arm_dir in sorted(FITS.iterdir()):
            pd_ = arm_dir / panel / f"fold{f}"
            if not pd_.exists():
                continue
            for sd in sorted(pd_.glob("seed*")):
                with np.load(sd / "generated.npz", allow_pickle=False) as z:
                    tg = z["targets"].astype(str); guard(tg.tolist(), sd)
                    order = np.array([list(tg).index(d) for d in c["donors"]])
                    pb = z["pb_counts"][order].astype(np.float64)
                pred = np.stack([log_cp10k(pb[t]).mean(0) for t in range(len(pb))])
                m = contrasts(pred, st)["hn"]
                for g, i in gi.items():
                    rows.setdefault((panel, arm_dir.name), {}).setdefault(g, []).append(m[i])
out = {f"{p}|{a}": {g: float(np.mean(v)) for g, v in d.items()} for (p, a), d in rows.items()}
(OUT / "d2_dam_genes.json").write_text(json.dumps(out, indent=1))
for panel in ["hvg2000", "wide"]:
    keys = [k for k in rows if k[0] == panel]
    gs = [g for g in GENES if g in rows[(panel, "real train")]]
    print(f"\n## {panel}: High-minus-NotAD, log1p CP10k, mean over folds (x seeds)")
    print("| source | " + " | ".join(gs) + " |"); print("|---" * (len(gs) + 1) + "|")
    for k in keys:
        print(f"| {k[1]} | " + " | ".join(fmt(np.mean(rows[k][g]), 2) for g in gs) + " |")
