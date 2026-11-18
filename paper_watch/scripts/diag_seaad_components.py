"""Iteration 3 diagnostic: per-type readout components for Sst vs L2/3 IT.

No GAN fits — Gaussian CTRL sampler only. Prints, per target type and per
labeling (true + shuffle seeds), the pieces of the iteration-2 score:
ED(samples, AD_t), ED(samples, CTRL_t), split-half self-ED of CTRL_t,
the normalized score, and the model-free ED(AD_t, CTRL_t) / self-ED.
"""
import sys

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "tests")

import test_seaad_vulnerability_readout as T
from alztest.harness import energy_distance, pca_fit_transform, standardize

X, types, group = T._load_subsample(0)
var = X.var(0)
top = np.argsort(var)[::-1][:2000]
Xp, _ = pca_fit_transform(X[:, top], n_comp=T.N_COMP)
Xp = standardize(Xp)

TARGETS = ["Sst", "L2/3 IT"]


def components(g, seed):
    ctrl = Xp[g == "CTRL"]
    mu, cov = ctrl.mean(0), np.cov(ctrl.T) + 1e-3 * np.eye(ctrl.shape[1])
    rng = np.random.default_rng(seed + 23)
    samples = rng.multivariate_normal(mu, cov, size=1024)
    rng2 = np.random.default_rng(seed + 41)
    out = {}
    for t in TARGETS:
        ad_t = Xp[(types == t) & (g == "AD")]
        ct_t = Xp[(types == t) & (g == "CTRL")]
        half = rng2.permutation(len(ct_t))
        self_ed = energy_distance(ct_t[half[:len(ct_t) // 2]],
                                  ct_t[half[len(ct_t) // 2:]], seed=seed)
        num = (energy_distance(samples, ad_t, seed=seed)
               - energy_distance(samples, ct_t, seed=seed))
        direct = energy_distance(ad_t, ct_t, seed=seed)
        out[t] = dict(n_ad=len(ad_t), n_ctrl=len(ct_t), num=num,
                      self_ed=self_ed, score=num / self_ed,
                      direct=direct, direct_norm=direct / self_ed)
    return out


for label, g in [("true", group)] + [
        (f"shuffle{s}", np.random.default_rng(100 + s).permutation(group))
        for s in range(5)]:
    c = components(g, 0)
    for t in TARGETS:
        d = c[t]
        print(f"{label:9s} {t:8s} n={d['n_ad']}/{d['n_ctrl']} "
              f"num={d['num']:+.4f} selfED={d['self_ed']:.4f} "
              f"score={d['score']:+.3f} direct={d['direct']:.4f} "
              f"direct_norm={d['direct_norm']:+.3f}")
    print(f"{label:9s} MARGIN score={c['Sst']['score'] - c['L2/3 IT']['score']:+.3f} "
          f"direct_norm={c['Sst']['direct_norm'] - c['L2/3 IT']['direct_norm']:+.3f}")
