"""(b, follow-up) Is the a priori DAM direction signal (stage_b_gene_subsets.py) carried by
a single Not AD donor? Leave each Not AD donor out (both panels, all 37 donors, common
genes) and recompute the mean signed Welch t and its permutation p. Also a 5-gene score
per donor (mean z over donors of GPNMB, CD9, LGALS3, LPL, TYROBP) by stage.
NB: the 5 genes were picked AFTER seeing stage_b results, so the score table is
descriptive only, not a test. Gene lists copied from stage_b_gene_subsets.py.
Output: printed; OUT/b2_dam_loo.json.
"""
from stage_common import *

UP = ["APOE", "TREM2", "SPP1", "GPNMB", "LPL", "CD74", "HLA-DRA", "TYROBP", "CST7", "ITGAX", "CD9",
      "LGALS3", "CD68", "AXL", "CLEC7A", "ACSL1", "CHI3L1", "CTSD", "CTSB", "FTL"]
DOWN = ["P2RY12", "CX3CR1", "TMEM119"]
FIVE = ["GPNMB", "CD9", "LGALS3", "LPL", "TYROBP"]
NPERM = 2000
rng = np.random.default_rng(20261114)


def welch_t(P, st):
    a, b = P[st == 3], P[st == 0]
    se = np.sqrt(a.var(0, ddof=1) / len(a) + b.var(0, ddof=1) / len(b))
    return (a.mean(0) - b.mean(0)) / np.maximum(se, 1e-12)


res = {}
for panel in ["hvg2000", "wide"]:
    c = all37(panel); P = log_cp10k(c["pb"]); st = c["stage"]; gl = list(c["genes"])
    up = [x for x in UP if x in gl]; dn = [x for x in DOWN if x in gl]
    idx = np.array([gl.index(x) for x in up + dn]); sg = np.array([1] * len(up) + [-1] * len(dn))
    out = res[panel] = {"loo": []}
    for i in np.where(st == 0)[0]:
        keep = np.arange(37) != i
        Pk, sk = P[keep][:, idx], st[keep]
        o = float((sg * welch_t(Pk, sk)).mean())
        nn = np.array([(sg * welch_t(Pk, rng.permutation(sk))).mean() for _ in range(NPERM)])
        out["loo"].append({"dropped": str(c["donors"][i]), "stat": o, "p": float((1 + (nn >= o).sum()) / (1 + NPERM))})
    five = [gl.index(x) for x in FIVE if x in gl]
    Z = (P[:, five] - P[:, five].mean(0)) / P[:, five].std(0)
    sc = Z.mean(1)
    out["five_genes_present"] = [x for x in FIVE if x in gl]
    out["five_score_by_stage"] = {str(s): sorted([round(float(x), 2) for x in sc[st == s]]) for s in range(4)}
    out["five_score_spearman_stage"] = spearman(sc, st)
    print(f"\n## {panel}")
    for r in out["loo"]:
        print(f"  drop Not AD {r['dropped']}: mean signed t {fmt(r['stat'])}, p {fmt(r['p'])}")
    print(f"  5-gene score by stage (sorted donors): {out['five_score_by_stage']}; Spearman(score, stage) {fmt(out['five_score_spearman_stage'])}")
(OUT / "b2_dam_loo.json").write_text(json.dumps(res, indent=1))
