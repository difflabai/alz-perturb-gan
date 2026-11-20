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

Inference (revision after the iteration-3 close; not yet rerun):
    - Half of the CTRL donors (by a seeded hash of the donor label, so a
      donor is on the same side in both regions) are fit donors. Gene
      selection, PCA, standardization, burden thresholds, the
      ParticleGAN, the Gaussian and the centroid use fit donors only.
      Scored units are held-out CTRL donors plus all non-CTRL donors.
    - Every scored donor is subsampled to MIN_CELLS_PER_DONOR cells
      before energy distance and centroid scoring, and the energy
      distance is the unbiased U-statistic. Each donor's cell count and
      its Spearman with burden are reported as a confound check.
    - The shuffled arm is a donor-grouped permutation p-value (each
      donor's AnG and LEC burdens move together). Arm comparisons use
      donor-bootstrap CIs of the rho difference.

Compartment note: the ABC taxonomy places microglia/PVM supertypes
under the "Immune" subclass; that subclass is the toy compartment
(the paper's compartment is the Microglia/PVM-lineage object).
Module genes are the paper's named module constituents:
  lipid:    APOE ABCA1 SOAT1 ACSL1 ACSL3 ACSL4 PLIN2
  lysosome: NPC1 NPC2 LIPA GBA CTSD CTSB CTSS LAMP1 LAMP2

Iteration log: notes/seaad_ll_microglia_readout.md
"""
from __future__ import annotations

import zlib

import numpy as np
import pytest
from scipy.stats import spearmanr

from alztest.ll_panel import LIPID, LYSO
from alztest.data import DataUnavailable, adnc_group, load_seaad_obs, seaad_annotations
from alztest.harness import (
    donor_bootstrap_ci,
    energy_distance,
    fit_unconditional,
    pca_apply,
    pca_fit,
    permutation_pvalue,
    standardize_apply,
    standardize_fit,
)

REGIONS = ("AnG", "LEC")
MODULE = LIPID + LYSO
N_COMP = 30
MIN_CELLS_PER_DONOR = 25
MODEL_DRAWS = 4096
# Preregistered inference settings for the next run.
ALPHA = 0.05
N_PERM = 2000
N_BOOT = 2000


def _is_fit_donor(donor: str, seed: int) -> bool:
    """Seeded, region-independent split of CTRL donors (about half fit)."""
    return zlib.crc32(f"{seed}:{donor}".encode()) % 2 == 0


def _load_microglia(seed: int = 0, region: str = "AnG"):
    adata = load_seaad_obs(region)
    ann = seaad_annotations(region)
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
    group = adnc_group(ann["adnc"])
    donors = ann["donor_label"].astype(str).to_numpy()
    fit = (group == "CTRL") & np.array([_is_fit_donor(d, seed) for d in donors])
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
    # Gene variance from fit-donor cells only (rows in file order here).
    fit_sorted = fit[order]
    n_fit = int(fit_sorted.sum())
    s1 = np.zeros(S.shape[1]); s2 = np.zeros(S.shape[1])
    for lo in range(0, n, 512):
        C = np.asarray(S[lo:lo + 512].todense(), dtype=np.float64)
        C = C[fit_sorted[lo:lo + 512]]
        s1 += C.sum(0); s2 += (C * C).sum(0)
    var = s2 / n_fit - (s1 / n_fit) ** 2
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
    return X, group, donors, lip_pos, lys_pos, fit


def _ll_burden(X, fit, donors, lip_pos, lys_pos, score_donors):
    """Per-donor LL burden with z-scores and q0.75 thresholds from fit cells."""
    def zscore(cols):
        mu, sd = X[fit][:, cols].mean(0), X[fit][:, cols].std(0) + 1e-9
        return ((X[:, cols] - mu) / sd).mean(1)
    lip, lys = zscore(lip_pos), zscore(lys_pos)
    t_lip = np.quantile(lip[fit], 0.75)
    t_lys = np.quantile(lys[fit], 0.75)
    high = (lip > t_lip) & (lys > t_lys)
    return {d: float(high[donors == d].mean()) for d in score_donors}


def _spearman(x, y):
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(spearmanr(x, y).statistic)


def run_once(seed: int = 0, steps: int = 300, *, n_perm: int = N_PERM,
             n_boot: int = N_BOOT) -> dict:
    # Iteration 3: pool LEC donors alongside AnG. Burden, PCA, and fits
    # stay PER-REGION (representations are not comparable across
    # regions); the pooled Spearman unit is donor x region, and inference
    # treats the donor as the independent unit.
    devs = {"model": [], "gauss": [], "lin": []}
    b_all, unit_donor, unit_region, unit_group, n_cells = [], [], [], [], []
    n_per_region = {}
    for region in REGIONS:
        X, group, donors, lip_pos, lys_pos, fit = _load_microglia(seed, region)
        counts = {d: int((donors == d).sum()) for d in np.unique(donors)}
        fit_donors = set(np.unique(donors[fit]))
        score_ids = sorted(d for d, c in counts.items()
                           if c >= MIN_CELLS_PER_DONOR and d not in fit_donors)
        burden = _ll_burden(X, fit, donors, lip_pos, lys_pos, score_ids)
        pca = pca_fit(X[fit], n_comp=N_COMP)
        Xp = pca_apply(X, pca)
        Xp = standardize_apply(Xp, standardize_fit(Xp[fit]))
        n_per_region[region] = len(score_ids)
        # Depth matching: every scored donor contributes the same number of
        # cells to the energy-distance and centroid arms.
        rng = np.random.default_rng(seed + 7)
        dev_cells = {}
        for d in score_ids:
            idx = np.flatnonzero(donors == d)
            dev_cells[d] = Xp[rng.choice(idx, MIN_CELLS_PER_DONOR, replace=False)]
        b_all.extend(burden[d] for d in score_ids)
        unit_donor.extend(score_ids)
        unit_region.extend([region] * len(score_ids))
        unit_group.extend(group[donors == d][0] for d in score_ids)
        n_cells.extend(counts[d] for d in score_ids)

        ref = Xp[fit]
        model = fit_unconditional(ref, steps=steps, seed=seed)
        samples = model.sample(0, MODEL_DRAWS, seed=seed + 5)
        devs["model"].extend(
            energy_distance(samples, dev_cells[d], max_n=None)
            for d in score_ids)
        mu, cov = ref.mean(0), np.cov(ref.T) + 1e-3 * np.eye(N_COMP)
        gs = np.random.default_rng(seed + 23).multivariate_normal(mu, cov, MODEL_DRAWS)
        devs["gauss"].extend(
            energy_distance(gs, dev_cells[d], max_n=None) for d in score_ids)
        centroid = ref.mean(0)
        devs["lin"].extend(
            float(np.linalg.norm(dev_cells[d].mean(0) - centroid))
            for d in score_ids)
    b = np.array(b_all)
    unit_donor = np.array(unit_donor)
    unit_region = np.array(unit_region)
    non_ctrl = np.array(unit_group) != "CTRL"
    dev = {k: np.array(v) for k, v in devs.items()}

    def rho_on(idx, arm):
        return _spearman(dev[arm][idx], b[idx])

    out = {"seed": seed}
    for arm, key in (("model", "model"), ("gauss", "gaussian"), ("lin", "linear")):
        out[f"{key}_rho"] = _spearman(dev[arm], b)
        out[f"{key}_rho_non_ctrl"] = _spearman(dev[arm][non_ctrl], b[non_ctrl])
        perm = permutation_pvalue(
            lambda labels, arm=arm: _spearman(dev[arm], labels) or 0.0,
            b, unit_donor, within=unit_region, n_perm=n_perm, seed=seed + 31)
        out[f"{key}_p"] = perm["p"]
        if key == "model":
            out["shuffled_rho_mean"] = perm["null_mean"]
            out["shuffled_rho_q95"] = perm["null_q95"]
    out["model_rho_ci"] = donor_bootstrap_ci(
        lambda idx: rho_on(idx, "model"), unit_donor, n_boot=n_boot, seed=seed + 37)
    for arm, key in (("gauss", "gaussian"), ("lin", "linear")):
        def diff(idx, arm=arm):
            m, a = rho_on(idx, "model"), rho_on(idx, arm)
            return None if m is None or a is None else m - a
        out[f"model_minus_{key}_ci"] = donor_bootstrap_ci(
            diff, unit_donor, n_boot=n_boot, seed=seed + 37)
    out.update({
        "n_units": int(len(b)),
        "n_donors": int(len(np.unique(unit_donor))),
        "n_non_ctrl_units": int(non_ctrl.sum()),
        "n_per_region": n_per_region,
        "cells_per_donor_scored": MIN_CELLS_PER_DONOR,
        "n_cells_median": float(np.median(n_cells)),
        # Confound check: burden should not simply track sampling depth.
        "n_cells_burden_rho": _spearman(np.array(n_cells, dtype=float), b),
    })
    return out


def test_ll_burden_readout():
    try:
        results = [run_once(seed=s) for s in (0, 1, 2)]
    except DataUnavailable as e:
        pytest.skip(str(e))
    m = np.array([r["model_rho"] for r in results])
    p = np.array([r["model_p"] for r in results])
    assert (m > 0.3).all(), f"model rhos not all > 0.3: {m}"
    assert (p < ALPHA).all(), f"donor-grouped permutation p not all < {ALPHA}: {p}"
    for arm in ("gaussian", "linear"):
        lo = np.array([r[f"model_minus_{arm}_ci"]["lo"] for r in results])
        assert (lo > 0).all(), f"model - {arm} rho CI lower bounds not all > 0: {lo}"


def _synthetic_region(seed, region):
    """Data-free stand-in for _load_microglia (plumbing check only)."""
    rng = np.random.default_rng(zlib.crc32(region.encode()))
    donors, group = [], []
    for i in range(16):
        g = ("CTRL", "EARLY", "AD", "CTRL")[i % 4]
        n = int(rng.integers(30, 90))
        donors += [f"D{i}"] * n
        group += [g] * n
    donors, group = np.array(donors), np.array(group, dtype=object)
    shift = np.array([{"CTRL": 0.0, "EARLY": 0.5, "AD": 1.0}[g] for g in group])
    X = rng.normal(size=(len(donors), 40)) + shift[:, None] * (np.arange(40) < 16)
    fit = (group == "CTRL") & np.array([_is_fit_donor(d, seed) for d in donors])
    return X, group, donors, list(range(7)), list(range(7, 16)), fit


def test_ll_readout_runs_on_synthetic_donors(monkeypatch):
    import sys
    module = sys.modules[__name__]
    monkeypatch.setattr(module, "_load_microglia", _synthetic_region)
    monkeypatch.setattr(module, "MODEL_DRAWS", 256)
    out = run_once(seed=0, steps=2, n_perm=50, n_boot=50)
    fit_ids = {f"D{i}" for i in range(0, 16, 4) if _is_fit_donor(f"D{i}", 0)} | {
        f"D{i}" for i in range(3, 16, 4) if _is_fit_donor(f"D{i}", 0)}
    assert out["n_donors"] == 16 - len(fit_ids)
    assert out["n_units"] == 2 * out["n_donors"]
    assert 0 < out["model_p"] <= 1
    assert out["model_rho_ci"]["lo"] <= out["model_rho_ci"]["hi"]
    assert out["linear_rho"] > 0.5  # the synthetic burden tracks the mean shift
