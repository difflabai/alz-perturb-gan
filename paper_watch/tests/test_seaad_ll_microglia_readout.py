"""Cycle-2 top find — LL (lipid-lysosome) microglial remodeling readout.

Find: Arif et al., Research Square 2026, doi 10.21203/rs.3.rs-10721231/v1.
In SEA-AD, microglia/PVM cells carry a continuous lipid-lysosome (LL)
remodeling axis: LL-remodeled cells (joint high lipid + lysosome module
scores, q0.75/q0.75) are 13.3% of the compartment; donor-level LL burden
tracks CERAD (rho=0.497), ADNC (0.419) and Braak (0.389); the program
localizes to plaque niches (GeoMx plaque-core Cohen's d=1.25; STARmap
plaque-distance rho=-0.388) and is conserved human-mouse (r=0.96).

Route: NEW TARGET + READOUT. Question for a ParticleGAN fit on CONTROL
microglia only: does a donor's distance-from-the-control-model track
that donor's LL burden? If the model has learned healthy microglial
distribution, remodeled donors should sit farther from it.

Salient claim under test (toy scale):
    Spearman rho(model deviation, donor LL burden) > 0.3 across >= 3
    seeds, clear of a shuffled-burden arm, a Gaussian sampler arm, and
    a linear centroid-distance baseline.

Compartment note: the ABC taxonomy places microglia/PVM supertypes
under the "Immune" subclass; that subclass is the toy compartment
(the paper's compartment is the Microglia/PVM-lineage object).
Module genes are the paper's named module constituents:
  lipid:    APOE ABCA1 SOAT1 ACSL1 ACSL3 ACSL4 PLIN2
  lysosome: NPC1 NPC2 LIPA GBA CTSD CTSB CTSS LAMP1 LAMP2

Iteration log: notes/seaad_ll_microglia_readout.md
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import spearmanr

from alztest.data import DataUnavailable, load_seaad_obs, seaad_annotations
from alztest.harness import energy_distance, fit_unconditional, pca_fit_transform, standardize

REGION = "AnG"
LIPID = ["APOE", "ABCA1", "SOAT1", "ACSL1", "ACSL3", "ACSL4", "PLIN2"]
LYSO = ["NPC1", "NPC2", "LIPA", "GBA", "CTSD", "CTSB", "CTSS", "LAMP1", "LAMP2"]
MODULE = LIPID + LYSO
N_COMP = 30
MIN_CELLS_PER_DONOR = 25


def _load_microglia(seed: int = 0):
    adata = load_seaad_obs(REGION)
    ann = seaad_annotations(REGION)
    obs = adata.obs
    key = (obs["cell_barcode"].astype(str) + "-"
           + obs["barcoded_cell_sample_label"].astype(str))
    pos = {label: i for i, label in enumerate(key)}
    common = [label for label in ann.index if label in pos]
    ann = ann.loc[common]
    rows_all = np.array([pos[label] for label in common])
    keep = ann["subclass"].astype(str).to_numpy() == "Immune"
    ann = ann[keep]
    rows = rows_all[keep]
    adnc = ann["adnc"].astype(str)
    group = np.where(adnc.str.contains("High"), "AD",
                     np.where(adnc.str.contains("Not AD|Low"), "CTRL",
                              "OTHER")).astype(object)
    donors = ann["donor_label"].astype(str).to_numpy()
    order = np.argsort(rows)
    sel = adata[rows[order]]
    S = sel.X
    # SEA-AD var index is Ensembl IDs; symbols live in var["gene_symbol"]
    # (loader seam found iteration 1: name lookup against the index
    # silently matched nothing).
    var_names = np.array(sel.var["gene_symbol"].astype(str))
    mod_idx = [int(np.where(var_names == g)[0][0]) for g in MODULE
               if (var_names == g).any()]
    if len(mod_idx) < 10:
        raise DataUnavailable(f"only {len(mod_idx)} module genes found")
    n = S.shape[0]
    s1 = np.zeros(S.shape[1]); s2 = np.zeros(S.shape[1])
    for lo in range(0, n, 512):
        C = np.asarray(S[lo:lo + 512].todense(), dtype=np.float64)
        s1 += C.sum(0); s2 += (C * C).sum(0)
    var = s2 / n - (s1 / n) ** 2
    top = [i for i in np.argsort(var)[::-1][:1500] if i not in mod_idx]
    cols = np.sort(np.array(top + mod_idx))
    X = np.empty((n, len(cols)), dtype=np.float64)
    for lo in range(0, n, 512):
        X[lo:lo + 512] = np.asarray(S[lo:lo + 512][:, cols].todense(),
                                    dtype=np.float64)
    X = X[np.argsort(order)]
    mod_pos = [int(np.where(cols == i)[0][0]) for i in mod_idx]
    lip_pos = mod_pos[:len([g for g in LIPID if g in set(var_names)])]
    lys_pos = mod_pos[len(lip_pos):]
    return X, group, donors, lip_pos, lys_pos


def _ll_burden(X, group, donors, lip_pos, lys_pos):
    ctrl = group == "CTRL"
    def zscore(cols):
        mu, sd = X[ctrl][:, cols].mean(0), X[ctrl][:, cols].std(0) + 1e-9
        return ((X[:, cols] - mu) / sd).mean(1)
    lip, lys = zscore(lip_pos), zscore(lys_pos)
    t_lip = np.quantile(lip[ctrl], 0.75)
    t_lys = np.quantile(lys[ctrl], 0.75)
    high = (lip > t_lip) & (lys > t_lys)
    burden = {}
    for d in np.unique(donors):
        m = donors == d
        if m.sum() >= MIN_CELLS_PER_DONOR:
            burden[d] = float(high[m].mean())
    return burden


def run_once(seed: int = 0, steps: int = 300) -> dict:
    X, group, donors, lip_pos, lys_pos = _load_microglia(seed)
    burden = _ll_burden(X, group, donors, lip_pos, lys_pos)
    Xp, _ = pca_fit_transform(X, n_comp=N_COMP)
    Xp = standardize(Xp)
    ctrl = group == "CTRL"
    donor_ids = sorted(burden)
    dev_cells = {d: Xp[donors == d] for d in donor_ids}
    b = np.array([burden[d] for d in donor_ids])

    def rho(dev):
        return float(spearmanr(dev, b).statistic)

    model = fit_unconditional(Xp[ctrl], steps=steps, seed=seed)
    samples = model.sample(0, 4096, seed=seed + 5)
    dev_model = np.array([energy_distance(samples, dev_cells[d], seed=seed)
                          for d in donor_ids])
    mu, cov = Xp[ctrl].mean(0), np.cov(Xp[ctrl].T) + 1e-3 * np.eye(N_COMP)
    gs = np.random.default_rng(seed + 23).multivariate_normal(mu, cov, 4096)
    dev_gauss = np.array([energy_distance(gs, dev_cells[d], seed=seed)
                          for d in donor_ids])
    centroid = Xp[ctrl].mean(0)
    dev_lin = np.array([float(np.linalg.norm(dev_cells[d].mean(0) - centroid))
                        for d in donor_ids])
    rng = np.random.default_rng(seed + 31)
    rho_shuf = float(spearmanr(dev_model, rng.permutation(b)).statistic)
    return {
        "seed": seed,
        "model_rho": rho(dev_model),
        "gaussian_rho": rho(dev_gauss),
        "linear_rho": rho(dev_lin),
        "shuffled_rho": rho_shuf,
        "n_donors": len(donor_ids),
    }


def test_ll_burden_readout():
    try:
        results = [run_once(seed=s) for s in (0, 1, 2)]
    except DataUnavailable as e:
        pytest.skip(str(e))
    m = np.array([r["model_rho"] for r in results])
    g = np.array([r["gaussian_rho"] for r in results])
    lin = np.array([r["linear_rho"] for r in results])
    sh = np.array([r["shuffled_rho"] for r in results])
    assert (m > 0.3).all(), f"model rhos not all > 0.3: {m}"
    assert m.mean() > sh.mean() + 0.2, f"model {m.mean():.3f} vs shuffled {sh.mean():.3f}"
    assert m.mean() > g.mean(), f"model {m.mean():.3f} vs gaussian {g.mean():.3f}"
    assert m.mean() > lin.mean(), f"model {m.mean():.3f} vs linear {lin.mean():.3f}"
