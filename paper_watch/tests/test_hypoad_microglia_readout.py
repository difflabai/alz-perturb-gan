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
predictable at QWK 1.0 from donor metadata alone): AD_44 and the
young donors are removed before grouping (alztest.hypoad.EXCLUDE_DONORS).
Per seed, fit on a random half of the control donors and score the
held-out control donors and all sporadic-AD donors; AUROC over
scored donors. Gene selection, module-gene z-scoring, PCA and
component standardization are all fitted on the fit-control cells
and then applied unchanged to every cell. Each AUROC is reported
with its donor counts, the exact label-permutation p over scored
donors and a stratified donor-bootstrap 95% interval.
Arms: Gaussian sampler, linear centroid, shuffled diagnosis,
DAM-program-only.

Iteration log: notes/hypoad_microglia_readout.md
"""
from __future__ import annotations

import numpy as np
import pytest

from alztest.data import DataUnavailable, load_hypoad_microglia
from alztest.harness import energy_distance, fit_unconditional
from alztest.hypoad import (
    COND_COLS, DAM, DONOR_COLS, HOMEO, N_COMP, apply_feature_transform, auroc,
    donor_bootstrap_ci, find_column, fit_feature_transform, log_normalize,
    permutation_p, select_donors,
)

MIN_CELLS_PER_DONOR = 40
ALPHA = 0.05


def _prepare():
    """Load, exclude donors, log-normalize. Nothing here is fitted to cells."""
    adata = load_hypoad_microglia()
    obs = adata.obs
    try:
        donor_col = find_column(obs.columns, DONOR_COLS, "donor")
        cond_col = find_column(obs.columns, COND_COLS, "condition")
    except KeyError as e:
        raise DataUnavailable(f"HypoAD extract: {e}") from e
    donors = obs[donor_col].astype(str).to_numpy()
    cond_raw = obs[cond_col].astype(str).to_numpy()
    keep, group = select_donors(donors, cond_raw)
    rows = np.flatnonzero(keep)
    X = adata.layers["counts"] if "counts" in adata.layers else adata.X
    X = log_normalize(X[rows])
    var_names = np.array(adata.var_names.astype(str))
    mod_idx = [i for i, g in enumerate(var_names) if g in HOMEO | DAM]
    if len(mod_idx) < 15:
        raise DataUnavailable(f"only {len(mod_idx)} DAM/homeostatic genes found")
    homeo_idx = [i for i in mod_idx if var_names[i] in HOMEO]
    dam_idx = [i for i in mod_idx if var_names[i] in DAM]
    return X, group[rows], donors[rows], mod_idx, homeo_idx, dam_idx


def _stats(scores, labels, seed):
    p, n_assign, exact = permutation_p(scores, labels, seed=seed)
    lo, hi = donor_bootstrap_ci(scores, labels, seed=seed)
    return {"auroc": auroc(scores, labels), "p": p, "p_exact": exact,
            "n_assignments": n_assign, "ci95": (lo, hi)}


def run_once(seed: int = 0, steps: int = 300, data=None) -> dict:
    X, group, donors, mod_idx, homeo_idx, dam_idx = data if data is not None else _prepare()
    rng = np.random.default_rng(seed + 7)
    ctrl_donors = np.unique(donors[group == "CTRL"])
    fit_donors = rng.choice(ctrl_donors, size=max(2, len(ctrl_donors) // 2),
                            replace=False)
    fit_mask = np.isin(donors, fit_donors)
    score_donors = [d for d in np.unique(donors) if d not in set(fit_donors)]
    t = fit_feature_transform(X, fit_mask, mod_idx, n_comp=N_COMP)
    Xp, Z = apply_feature_transform(t, X)
    # DAM program per cell: DAM mean z minus homeostatic mean z (fit-control z)
    prog_cell = (np.mean([Z[i] for i in dam_idx], 0)
                 - np.mean([Z[i] for i in homeo_idx], 0))
    model = fit_unconditional(Xp[fit_mask], steps=steps, seed=seed)
    samples = model.sample(0, 4096, seed=seed + 5)
    gmu = Xp[fit_mask].mean(0)
    gcov = np.cov(Xp[fit_mask].T) + 1e-3 * np.eye(Xp.shape[1])
    gs = np.random.default_rng(seed + 23).multivariate_normal(gmu, gcov, 4096)
    dev_m, dev_g, dev_l, prog, labels, scored = [], [], [], [], [], []
    for d in score_donors:
        m = donors == d
        if m.sum() < MIN_CELLS_PER_DONOR:
            continue
        scored.append(str(d))
        labels.append(1 if group[m][0] == "AD" else 0)
        cells = Xp[m]
        dev_m.append(energy_distance(samples, cells, seed=seed))
        dev_g.append(energy_distance(gs, cells, seed=seed))
        dev_l.append(float(np.linalg.norm(cells.mean(0) - gmu)))
        prog.append(float(prog_cell[m].mean()))
    labels = np.array(labels)
    model_stats = _stats(dev_m, labels, seed)
    return {
        "seed": seed,
        "model_auroc": model_stats["auroc"],
        "model_p": model_stats["p"],
        "model_p_exact": model_stats["p_exact"],
        "model_ci95": model_stats["ci95"],
        "min_attainable_p": (1 / model_stats["n_assignments"]
                             if model_stats["n_assignments"] else None),
        "gaussian_auroc": auroc(dev_g, labels),
        "linear_auroc": auroc(dev_l, labels),
        "program_auroc": auroc(prog, labels),
        "program_p": permutation_p(prog, labels, seed=seed)[0],
        "shuffled_auroc": auroc(dev_m, rng.permutation(labels)),
        "n_fit_ctrl_donors": int(len(fit_donors)),
        "n_fit_cells": int(fit_mask.sum()),
        "n_scored": int(labels.size),
        "n_ad": int(labels.sum()),
        "n_ctrl_heldout": int((labels == 0).sum()),
        "scored_donors": scored,
    }


def test_hypoad_microglia_readout():
    try:
        data = _prepare()
    except DataUnavailable as e:
        pytest.skip(str(e))
    results = [run_once(seed=s, data=data) for s in (0, 1, 2)]
    for r in results:
        print({k: r[k] for k in ("seed", "model_auroc", "model_p", "model_ci95",
                                 "n_ad", "n_ctrl_heldout", "program_auroc")})
    m = np.array([r["model_auroc"] for r in results], dtype=float)
    g = np.array([r["gaussian_auroc"] for r in results], dtype=float)
    p = np.array([r["program_auroc"] for r in results], dtype=float)
    sh = np.array([r["shuffled_auroc"] for r in results], dtype=float)
    pv = np.array([r["model_p"] for r in results], dtype=float)
    lo = np.array([r["model_ci95"][0] for r in results], dtype=float)
    assert (pv <= ALPHA).all(), f"model permutation p not all <= {ALPHA}: {pv}"
    assert (lo > 0.5).all(), f"model donor-bootstrap CI lower bounds: {lo}"
    assert m.mean() > sh.mean() + 0.15, \
        f"model {m.mean():.3f} vs shuffled {sh.mean():.3f}"
    assert m.mean() > p.mean(), \
        f"model {m.mean():.3f} vs DAM-program-only {p.mean():.3f}"
    assert m.mean() > g.mean(), \
        f"model {m.mean():.3f} vs gaussian {g.mean():.3f}"
