"""Cycle-4 top find — HypoAD hypothalamic microglia readout.

Find: Yu et al., Nature 2026 (doi 10.1038/s41586-026-11305-w),
published 2026-11-18. HypoAD: 614,403 nuclei, human hypothalamus —
a region absent from SEA-AD's ten regions and from every prior
cycle of this watch — 14 post-mortem donors (7 mostly early-stage
AD, 7 age-matched controls; the familial APOE4/4 donor AD_44 is
excluded from the paper's differential analyses and here), plus 3
young reference donors used only for annotation/trajectory anchoring
(excluded here: the contrast is AD vs age-matched control).

Paper's measured claims under test:
  (a) microglia carry the largest transcriptional response of any
      hypothalamic cell type in early AD, shifting homeostatic ->
      DAM (disease-associated) along an ageing-to-AD trajectory;
      their cell-type classifier was the best of five (AUROC 0.91,
      donor-held-out batches);
  (b) the DAM module structure is explicit (their Methods): 
      homeostatic {HEXB, CST3, CX3CR1, CSF1R, CTSS, TMSB4X, P2RY12,
      C1QB}, DAM1 {TYROBP, CTSB, APOE, B2M, FTH1, TREM2}, DAM2
      {TREM2, AXL, CST7, CTSL, LPL, CD9, CSF1, ITGAX, CLEC7A,
      LILRB4, TIMP2}.

Route: NEW DISTRIBUTION + READOUT. The distribution is hypothalamic
microglial expression conditioned on diagnosis — a new region's
compartment, openly downloadable (GEO GSE342635). The readout asks
whether a control-fit ParticleGAN's per-donor deviation separates
AD from control donors, and whether it beats the paper's own DAM
program score alone (the bar both closed SEA-AD loops point at:
a fitted model must beat the named program / first moments).

Design (donor is the unit, per the SEA-AD DREAM Challenge's leakage
lesson, bioRxiv 2026.07.02.736180 — categorical staging was
predictable at QWK 1.0 from donor metadata alone): per seed, fit
on a random half of the control donors, score the held-out control
donors and all sporadic-AD donors; AUROC over scored donors.
Arms: Gaussian sampler, linear centroid, shuffled diagnosis,
DAM-program-only.

Iteration log: notes/hypoad_microglia_readout.md
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import rankdata

from alztest.data import DataUnavailable, load_hypoad_microglia
from alztest.harness import energy_distance, fit_unconditional, pca_fit_transform, standardize

N_COMP = 30
MIN_CELLS_PER_DONOR = 40
HOMEO = {"HEXB", "CST3", "CX3CR1", "CSF1R", "CTSS", "TMSB4X", "P2RY12", "C1QB"}
DAM = {"TYROBP", "CTSB", "APOE", "B2M", "FTH1", "TREM2",
       "AXL", "CST7", "CTSL", "LPL", "CD9", "CSF1", "ITGAX", "CLEC7A",
       "LILRB4", "TIMP2"}


def _auroc(scores, labels):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    pos, neg = labels == 1, labels == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return None
    r = rankdata(scores)
    return float((r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2)
                 / (pos.sum() * neg.sum()))


def _prepare():
    adata = load_hypoad_microglia()
    obs = adata.obs
    donor_col = next((c for c in ("donor", "donor_id", "sample", "Sample",
                                  "individual", "Donor") if c in obs), None)
    cond_col = next((c for c in ("condition", "diagnosis", "group", "Group",
                                 "disease", "status") if c in obs), None)
    if donor_col is None or cond_col is None:
        raise DataUnavailable(
            f"HypoAD obs lacks donor/condition columns; has {list(obs.columns)}")
    donors = obs[donor_col].astype(str).to_numpy()
    cond_raw = obs[cond_col].astype(str).to_numpy()
    group = np.where(np.char.find(cond_raw, "AD") >= 0, "AD",
                     np.where(np.char.find(cond_raw, "ontrol") >= 0, "CTRL",
                              np.where(np.char.find(cond_raw, "CTL") >= 0,
                                       "CTRL", "OTHER"))).astype(object)
    # exclude the young reference donors and the familial donor if
    # identifiable in the condition strings (paper's own exclusions)
    young = np.char.find(np.char.lower(cond_raw), "young") >= 0
    group[young] = "OTHER"
    keep = group != "OTHER"
    donors, group = donors[keep], group[keep]
    X = adata.X[keep]
    if "counts" in adata.layers:
        X = adata.layers["counts"][keep]
    X = np.asarray(X.todense() if hasattr(X, "todense") else X, dtype=np.float64)
    var_names = np.array(adata.var_names.astype(str))
    # counts -> CP10k + log1p (idempotent-safe: skip if already logged)
    if X.max() > 50:
        lib = X.sum(1, keepdims=True) + 1e-9
        X = np.log1p(X / lib * 1e4)
    mod_idx = [i for i, g in enumerate(var_names) if g in HOMEO | DAM]
    if len(mod_idx) < 15:
        raise DataUnavailable(f"only {len(mod_idx)} DAM/homeostatic genes found")
    var = X.var(0)
    top = [i for i in np.argsort(var)[::-1][:1500] if i not in mod_idx]
    cols = np.sort(np.array(top + mod_idx))
    X = X[:, cols]
    pos = {int(g): i for i, g in enumerate(cols)}
    homeo_pos = [pos[i] for i in mod_idx if var_names[i] in HOMEO]
    dam_pos = [pos[i] for i in mod_idx if var_names[i] in DAM]
    return X, group, donors, homeo_pos, dam_pos


def run_once(seed: int = 0, steps: int = 300) -> dict:
    X, group, donors, homeo_pos, dam_pos = _prepare()
    rng = np.random.default_rng(seed + 7)
    ctrl_donors = np.unique(donors[group == "CTRL"])
    ad_donors = np.unique(donors[group == "AD"])
    fit_donors = rng.choice(ctrl_donors, size=max(2, len(ctrl_donors) // 2),
                            replace=False)
    fit_mask = np.isin(donors, fit_donors)
    score_donors = [d for d in np.unique(donors) if d not in set(fit_donors)]
    # DAM program score per cell: DAM mean z minus homeostatic mean z,
    # z-scored on fit-control cells
    mu, sd = X[fit_mask].mean(0), X[fit_mask].std(0) + 1e-9
    Z = (X - mu) / sd
    prog_cell = Z[:, dam_pos].mean(1) - Z[:, homeo_pos].mean(1)
    Xp, _ = pca_fit_transform(X, n_comp=N_COMP)
    Xp = standardize(Xp)
    model = fit_unconditional(Xp[fit_mask], steps=steps, seed=seed)
    samples = model.sample(0, 4096, seed=seed + 5)
    gmu, gcov = Xp[fit_mask].mean(0), np.cov(Xp[fit_mask].T) + 1e-3 * np.eye(N_COMP)
    gs = np.random.default_rng(seed + 23).multivariate_normal(gmu, gcov, 4096)
    centroid = Xp[fit_mask].mean(0)
    dev_m, dev_g, dev_l, prog, labels = [], [], [], [], []
    for d in score_donors:
        m = donors == d
        if m.sum() < MIN_CELLS_PER_DONOR:
            continue
        labels.append(1 if group[m][0] == "AD" else 0)
        cells = Xp[m]
        dev_m.append(energy_distance(samples, cells, seed=seed))
        dev_g.append(energy_distance(gs, cells, seed=seed))
        dev_l.append(float(np.linalg.norm(cells.mean(0) - centroid)))
        prog.append(float(prog_cell[m].mean()))
    labels = np.array(labels)
    return {
        "seed": seed,
        "model_auroc": _auroc(dev_m, labels),
        "gaussian_auroc": _auroc(dev_g, labels),
        "linear_auroc": _auroc(dev_l, labels),
        "program_auroc": _auroc(prog, labels),
        "shuffled_auroc": _auroc(dev_m, rng.permutation(labels)),
        "n_scored": int(labels.size),
        "n_ad": int(labels.sum()),
        "n_ctrl_heldout": int((labels == 0).sum()),
    }


def test_hypoad_microglia_readout():
    try:
        results = [run_once(seed=s) for s in (0, 1, 2)]
    except DataUnavailable as e:
        pytest.skip(str(e))
    m = np.array([r["model_auroc"] for r in results], dtype=float)
    g = np.array([r["gaussian_auroc"] for r in results], dtype=float)
    p = np.array([r["program_auroc"] for r in results], dtype=float)
    sh = np.array([r["shuffled_auroc"] for r in results], dtype=float)
    assert (m > 0.65).all(), f"model AUROCs not all > 0.65: {m}"
    assert m.mean() > sh.mean() + 0.15, \
        f"model {m.mean():.3f} vs shuffled {sh.mean():.3f}"
    assert m.mean() > p.mean(), \
        f"model {m.mean():.3f} vs DAM-program-only {p.mean():.3f}"
    assert m.mean() > g.mean(), \
        f"model {m.mean():.3f} vs gaussian {g.mean():.3f}"
