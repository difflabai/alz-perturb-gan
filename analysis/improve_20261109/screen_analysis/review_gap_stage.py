"""review_gap_stage.py - are the generators' stage deviations useless only out of sample, or also in sample?

For base/genemean (seeds 0,1, ensemble) x panel x fold:
  dev_t = S_model[stage_t] - mean_train_targets(pred)   (S_model = model's mean pred over all targets of the stage)
  score Lbar + lambda * dev_t on TRAIN targets (in-sample) and HELD-OUT targets, lambda in 0..1,
  plus the least-squares optimal global lambda on each set, and the same for the REAL
  train stage-mean-of-logs deviations (in sample this is the stage_mean_log fit).
Only caches + generated.npz are read (own loader).  Output: review_gap_stage.json.
"""
import os
for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[_v] = "6"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys, json
from pathlib import Path
import numpy as np
import scipy.sparse as sp

sys.dont_write_bytecode = True
STUDY = Path("/ml2/alz/analysis/improve_20261109")
RUNS = Path("/ml2/alz/runs/improve_20261109")
FOLDS = json.loads((STUDY / "folds.json").read_text())
DEV = set(FOLDS["development_donors"]); SEALED = set(FOLDS["sealed_evaluation_donors_fold0"])


def lcp(x):
    x = np.asarray(x, dtype=np.float64)
    return np.log1p(x * 1e4 / np.maximum(x.sum(-1, keepdims=True), 1))


def prmse(P, O):
    return float(np.mean(np.sqrt(np.mean((P - O) ** 2, 1))))


def load_pb(panel, fold):
    d = RUNS / "caches" / panel / f"fold{fold}"
    donors, held, stage, pb = [], [], [], []
    for split in ["train", "heldout"]:
        with np.load(d / split / "metadata.npz", allow_pickle=False) as m:
            dn = m["donors"].astype(str); cd = m["donor"].astype(str); raw = m["raw_c"]
        assert all(x in DEV and x not in SEALED for x in set(dn) | set(cd))
        C = sp.load_npz(d / split / "counts.npz").tocsr()
        for j, x in enumerate(dn):
            donors.append(x); held.append(split == "heldout"); stage.append(int(round(raw[j][3] * 3)))
            pb.append(np.asarray(C[np.where(cd == x)[0]].sum(0)).ravel())
    return np.array(donors), np.array(held), np.array(stage), np.array(pb, dtype=np.float64)


def main():
    out = {}
    for panel in ["hvg2000", "wide"]:
        for fold in range(3):
            donors, held, st, pb = load_pb(panel, fold)
            tr = ~held; obs = lcp(pb); Lbar = obs[tr].mean(0)
            r = {}
            preds = {}
            for arm in ["base", "genemean"]:
                for s in (0, 1):
                    with np.load(RUNS / "fits" / arm / panel / f"fold{fold}" / f"seed{s}" / "generated.npz") as f:
                        tg = f["targets"].astype(str); order = [list(tg).index(x) for x in donors]
                        preds[f"{arm}_s{s}"] = lcp(f["pb_counts"][order]).mean(1)
                preds[f"{arm}_ens"] = (preds[f"{arm}_s0"] + preds[f"{arm}_s1"]) / 2
            # real train stage-mean-of-logs deviations
            Sreal = {s: obs[tr & (st == s)].mean(0) for s in range(4) if (tr & (st == s)).any()}
            devs = {"real_train_stage_mean_log": np.stack([Sreal.get(s, Lbar) - Lbar for s in st])}
            for p, pr in preds.items():
                Sm = {s: pr[st == s].mean(0) for s in range(4)}
                devs[p] = np.stack([Sm[s] for s in st]) - pr[tr].mean(0)
            for p, dv in devs.items():
                d = {}
                for nm, mask in [("train", tr), ("heldout", held)]:
                    y = obs[mask] - Lbar; x = dv[mask]
                    d[nm] = {str(l): prmse(Lbar + l * x, obs[mask]) for l in [0, .25, .5, 1]}
                    d[nm]["lambda_opt_ls"] = float((x * y).sum() / (x * x).sum())
                r[p] = d
            out[f"{panel}/fold{fold}"] = r
            print(panel, fold, flush=True)
    (RUNS / "screen_analysis" / "review_gap_stage.json").write_text(json.dumps(out, indent=1))
    for panel in ["hvg2000", "wide"]:
        print(panel)
        for p in out[f"{panel}/fold0"]:
            for nm in ["train", "heldout"]:
                v = {k: np.mean([out[f"{panel}/fold{f}"][p][nm][k] for f in range(3)]) for k in ["0", "0.25", "0.5", "1", "lambda_opt_ls"]}
                print(f"  {p:28s} {nm:8s} " + " ".join(f"{k}:{x:.3f}" for k, x in v.items())
                      + "  lam_opt per fold " + ",".join(f"{out[f'{panel}/fold{f}'][p][nm]['lambda_opt_ls']:.2f}" for f in range(3)))


if __name__ == "__main__":
    main()
