"""gap_dispersion.py - is the generators' remaining pseudobulk level bias a per-cell mean-log bias
or a count-distribution-shape (dispersion / tail) bias?

On TRAIN targets (in-sample for the generator, depth matched): per gene
    pbK_bias  = mean_t [ mean_m log1p CP10k(gen 256-cell pb) - log1p CP10k(real 256-cell pb) ]
                (real 256-cell pb = evaluate.real_pb_subsample, the evaluator's depth-matched real summary)
    ml_bias   = mean_t [ gen mean_log - real mean_log ]             (per-cell mean log1p CP10k)
    nz_bias   = mean_t [ gen nonzero - real nonzero ]
    shape_gap = pbK_bias - ml_bias  (pseudobulk = log of MEAN count share; mean_log = mean of per-cell logs;
                with matched mean_log, a negative shape_gap means the generated per-cell counts have a thinner
                upper tail / less between-cell spread than real cells)
Reported as mean and mean-square by train-level quartile, plus the seed-to-seed Pearson of shape_gap.
Output: /ml2/alz/runs/improve_20261109/screen_analysis/gap_dispersion.json
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


def by_q(v, lvl):
    q = np.quantile(lvl, [.25, .5, .75]); b = np.digitize(lvl, q)
    return {f"Q{k + 1}": {"mean": float(v[b == k].mean()), "ms": float((v[b == k] ** 2).mean())} for k in range(4)}


def main():
    out = {}
    for panel in ["hvg2000", "wide"]:
        for fold in range(3):
            real = E.load_real(RUNS / "caches" / panel / f"fold{fold}", panel, fold)
            tr = ~real.is_heldout
            rK = MX.log_cp10k(E.real_pb_subsample(real, 256))
            Lbar = MX.log_cp10k(real.pb)[tr].mean(0)
            res = {"real_pbK_minus_meanlog_train_mean": float((rK[tr] - real.mean_log[tr]).mean())}
            shapes = {}
            for arm in ["base", "genemean"]:
                for s in (0, 1):
                    g = E.align_generated(E.load_generated(RUNS / "fits" / arm / panel / f"fold{fold}" / f"seed{s}" / "generated.npz"), real)
                    pred = MX.log_cp10k(g["pb_counts"].astype(np.float64)).mean(1)
                    ml = g["mean_log"].astype(np.float64).mean(1)
                    nz = g["nonzero"].astype(np.float64).mean(1)
                    pbk = (pred[tr] - rK[tr]).mean(0)
                    mlb = (ml[tr] - real.mean_log[tr]).mean(0)
                    nzb = (nz[tr] - real.nonzero[tr]).mean(0)
                    shape = pbk - mlb
                    shapes[(arm, s)] = shape
                    res[f"{arm}_s{s}"] = {"pbK_bias_ms": float((pbk ** 2).mean()), "ml_bias_ms": float((mlb ** 2).mean()),
                                          "shape_gap_ms": float((shape ** 2).mean()), "nz_bias_ms": float((nzb ** 2).mean()),
                                          "pbK_bias_by_level": by_q(pbk, Lbar), "ml_bias_by_level": by_q(mlb, Lbar),
                                          "shape_gap_by_level": by_q(shape, Lbar), "nz_bias_by_level": by_q(nzb, Lbar),
                                          "corr_shape_gap_with_level": float(np.corrcoef(shape, Lbar)[0, 1])}
                res[f"{arm}_shape_gap_seed_pearson"] = float(np.corrcoef(shapes[(arm, 0)], shapes[(arm, 1)])[0, 1])
            out[f"{panel}/fold{fold}"] = res
            print(panel, fold, flush=True)
    (OUT / "gap_dispersion.json").write_text(json.dumps(out, indent=1))
    # compact print: fold means
    for panel in ["hvg2000", "wide"]:
        print(f"\n{panel}")
        for k in ["base_s0", "base_s1", "genemean_s0", "genemean_s1"]:
            r = [out[f"{panel}/fold{f}"][k] for f in range(3)]
            m = lambda key: np.mean([x[key] for x in r])
            print(f" {k}: pbK_bias_ms {m('pbK_bias_ms'):.4f} ml_bias_ms {m('ml_bias_ms'):.4f} shape_gap_ms {m('shape_gap_ms'):.4f} "
                  f"nz_bias_ms {m('nz_bias_ms'):.5f} corr(shape,level) {m('corr_shape_gap_with_level'):.2f}")
            for nm in ["pbK_bias_by_level", "ml_bias_by_level", "shape_gap_by_level"]:
                print(f"    {nm}: " + " | ".join(
                    f"Q{q} {np.mean([x[nm][f'Q{q}']['mean'] for x in r]):+.3f} (ms {np.mean([x[nm][f'Q{q}']['ms'] for x in r]):.3f})" for q in range(1, 5)))
        for arm in ["base", "genemean"]:
            print(f" {arm} shape_gap seed pearson: " + ", ".join(f"{out[f'{panel}/fold{f}'][f'{arm}_shape_gap_seed_pearson']:.2f}" for f in range(3)))


if __name__ == "__main__":
    main()
