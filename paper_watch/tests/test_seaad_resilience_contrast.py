"""Cycle-3 top find — resilience as a donor-level contrast (Lu et al.).

Finds: Tosoni et al., Cell Stem Cell 2026 (resilience = preserved
cellular PROGRAMS at matched pathology, not preserved cell counts),
sharpened by Lu et al., bioRxiv 2026-10-28 (doi
10.64898/2026.10.26.761376): neuron-specific suppression of
nuclear-encoded mitochondrial genes and mitochondrial biogenesis
signaling is the primary driver of the transition to dementia —
pathology metrics alone do not separate demented from resilient.

Route: NEW TARGET. The condition variable our stage-only modeling is
missing is COGNITION AT MATCHED PATHOLOGY. SEA-AD carries both:
42 High-ADNC donors split 31 demented / 11 no-dementia (resilient).

Salient claim under test (toy scale), in cortical neurons,
High-ADNC donors only, pooled AnG + LEC (donor x region units):
  (a) program layer — per-donor neuronal mito-program score is LOWER
      in demented than resilient donors (the paper's direction);
  (b) model layer — a control-fit ParticleGAN's per-donor deviation
      separates demented from resilient donors (AUROC), and beats
      the program score alone, a Gaussian sampler, and shuffled
      cognition labels. Both closed loops so far (SEA-AD ordering,
      LL burden) collapsed to first/second moments — the model layer
      here is only interesting if it clears the program-only bar.

Program genes: nuclear-encoded mitochondrial (NDUF*, SDH*, UQCR*,
COX4-8*, ATP5*) + biogenesis regulators (PPARGC1A/B, NRF1, TFAM,
ESRRA); MT-* excluded (the claim is about nuclear-encoded genes).

Inference (revision after the iteration-3 close; not yet rerun):
each scored donor is subsampled to MIN_CELLS_PER_DONOR IT cells before
the energy-distance and centroid arms, the energy distance is the
unbiased U-statistic averaged over the three fits, the shuffled arm is
a donor-grouped permutation p-value (a donor's AnG and LEC units keep
one cognition label), and arm comparisons use donor-bootstrap CIs of
the AUROC difference. The AUROC of raw cell count is reported as a
depth confound check.

Iteration log: notes/seaad_resilience_contrast.md
"""
from __future__ import annotations

import zlib

import numpy as np
import pytest
from scipy.stats import rankdata

from alztest.data import DataUnavailable, adnc_group, load_seaad_obs, seaad_annotations
from alztest.harness import (
    donor_bootstrap_ci,
    energy_distance,
    fit_unconditional,
    pca_fit_transform,
    permutation_pvalue,
    standardize,
)

REGIONS = ("AnG", "LEC")
NON_NEURONAL = {"Astrocyte", "Endothelial", "Ependymal", "Immune",
                "OPC", "Oligodendrocyte", "VLMC & Perivascular"}
N_COMP = 30
MIN_CELLS_PER_DONOR = 40
N_FITS = 3
MODEL_DRAWS = 4096
# Preregistered inference settings for the next run.
ALPHA = 0.05
N_PERM = 2000
N_BOOT = 2000
BIOGENESIS = {"PPARGC1A", "PPARGC1B", "NRF1", "TFAM", "ESRRA"}
PREFIXES = ("NDUF", "SDHA", "SDHB", "SDHC", "SDHD", "UQCR", "COX4",
            "COX5", "COX6", "COX7", "COX8", "ATP5")


def _auroc(scores, labels):
    """AUROC of scores for label==1, rank-based (ties averaged)."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    pos, neg = labels == 1, labels == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return None
    r = rankdata(scores)
    return float((r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2)
                 / (pos.sum() * neg.sum()))


def _load_region(region):
    adata = load_seaad_obs(region)
    ann = seaad_annotations(region, include_cognition=True)
    obs = adata.obs
    key = (obs["cell_barcode"].astype(str) + "-"
           + obs["barcoded_cell_sample_label"].astype(str))
    pos = {label: i for i, label in enumerate(key)}
    common = [label for label in ann.index if label in pos]
    ann = ann.loc[common]
    rows_all = np.array([pos[label] for label in common])
    # Iteration 3: IT compartment restored (iteration 2's all-neuron
    # widening diluted the signal, mean AUROC 0.580 -> 0.547).
    keep = ann["subclass"].astype(str).str.contains("IT").to_numpy()
    ann = ann[keep]
    rows = rows_all[keep]
    group = adnc_group(ann["adnc"])
    cog = ann["cognition"].astype(str).to_numpy()
    donors = ann["donor_label"].astype(str).to_numpy()
    # Memory seam (iteration 1, attempt 0): IT neurons are ~10x the
    # microglia compartment; densifying all of them OOM-killed the
    # run (exit 137). Donor-level readouts don't need thousands of
    # cells per donor — cap at 600/donor before any dense work.
    rng = np.random.default_rng(12345)
    cap_keep = []
    for d in np.unique(donors):
        idx = np.where(donors == d)[0]
        if idx.size > 600:
            idx = rng.choice(idx, 600, replace=False)
        cap_keep.append(idx)
    cap_keep = np.sort(np.concatenate(cap_keep))
    rows, group, cog, donors = rows[cap_keep], group[cap_keep], \
        cog[cap_keep], donors[cap_keep]
    order = np.argsort(rows)
    sel = adata[rows[order]]
    S = sel.X
    var_names = np.array(sel.var["gene_symbol"].astype(str))
    prog_idx = [i for i, g in enumerate(var_names)
                if (g in BIOGENESIS or g.startswith(PREFIXES))
                and not g.startswith("MT-")]
    if len(prog_idx) < 30:
        raise DataUnavailable(f"only {len(prog_idx)} mito program genes")
    n = S.shape[0]
    s1 = np.zeros(S.shape[1]); s2 = np.zeros(S.shape[1])
    for lo in range(0, n, 512):
        C = np.asarray(S[lo:lo + 512].todense(), dtype=np.float64)
        s1 += C.sum(0); s2 += (C * C).sum(0)
    var = s2 / n - (s1 / n) ** 2
    top = [i for i in np.argsort(var)[::-1][:1500] if i not in prog_idx]
    cols = np.sort(np.array(top + prog_idx))
    X = np.empty((n, len(cols)), dtype=np.float64)
    for lo in range(0, n, 512):
        X[lo:lo + 512] = np.asarray(S[lo:lo + 512][:, cols].todense(),
                                    dtype=np.float64)
    X = X[np.argsort(order)]
    prog_pos = [int(np.where(cols == i)[0][0]) for i in prog_idx]
    return X, group, cog, donors, prog_pos


def run_once(seed: int = 0, steps: int = 300, *, n_perm: int = N_PERM,
             n_boot: int = N_BOOT) -> dict:
    dev_m, dev_g, dev_l, prog, labels = [], [], [], [], []
    unit_donor, n_cells = [], []
    n_arm = {"demented": 0, "resilient": 0}
    for region in REGIONS:
        X, group, cog, donors, prog_pos = _load_region(region)
        ctrl = group == "CTRL"
        # per-cell mito program score, z-scored on CTRL cells
        mu, sd = X[ctrl][:, prog_pos].mean(0), X[ctrl][:, prog_pos].std(0) + 1e-9
        pscore = ((X[:, prog_pos] - mu) / sd).mean(1)
        Xp, _ = pca_fit_transform(X, n_comp=N_COMP)
        Xp = standardize(Xp)
        # Iteration 3: ensemble of 3 independent fits. Revision: each
        # fit's energy distance is computed on all of its draws and the
        # per-fit distances are averaged (iteration 3 pooled the draws
        # and then subsampled 512 rows, so it used about 170 per fit).
        ens_samples = []
        for k in range(N_FITS):
            m_k = fit_unconditional(Xp[ctrl], steps=steps,
                                    seed=seed + 100 * k)
            ens_samples.append(m_k.sample(0, MODEL_DRAWS, seed=seed + 5 + 100 * k))
        gmu, gcov = Xp[ctrl].mean(0), np.cov(Xp[ctrl].T) + 1e-3 * np.eye(N_COMP)
        gs = np.random.default_rng(seed + 23).multivariate_normal(gmu, gcov, MODEL_DRAWS)
        centroid = Xp[ctrl].mean(0)
        depth_rng = np.random.default_rng(seed + 7)
        for d in np.unique(donors):
            m = donors == d
            if group[m][0] != "AD" or m.sum() < MIN_CELLS_PER_DONOR:
                continue
            c = cog[m][0]
            if c not in ("Dementia", "No dementia"):
                continue
            labels.append(1 if c == "Dementia" else 0)
            n_arm["demented" if c == "Dementia" else "resilient"] += 1
            unit_donor.append(d)
            n_cells.append(int(m.sum()))
            prog.append(float(pscore[m].mean()))
            # Depth matching: the same number of cells for every donor.
            cells = Xp[depth_rng.choice(np.flatnonzero(m), MIN_CELLS_PER_DONOR,
                                        replace=False)]
            dev_m.append(float(np.mean([
                energy_distance(s_k, cells, max_n=None) for s_k in ens_samples])))
            dev_g.append(energy_distance(gs, cells, max_n=None))
            dev_l.append(float(np.linalg.norm(cells.mean(0) - centroid)))
    labels = np.array(labels)
    unit_donor = np.array(unit_donor)
    scores = {
        "model": np.array(dev_m),
        "gaussian": np.array(dev_g),
        "linear": np.array(dev_l),
        # paper's direction: demented = SUPPRESSED program, so the
        # program score predicts dementia when negated
        "program": -np.array(prog),
    }
    out = {"seed": seed}
    for key, value in scores.items():
        out[f"{key}_auroc"] = _auroc(value, labels)
    # Shuffled arm: cognition is a donor label, so whole donors are permuted.
    perm = permutation_pvalue(lambda lab: _auroc(scores["model"], lab), labels,
                              unit_donor, n_perm=n_perm, seed=seed + 31)
    out["model_p"] = perm["p"]
    out["shuffled_auroc_mean"] = perm["null_mean"]
    out["shuffled_auroc_q95"] = perm["null_q95"]

    def auroc_on(idx, key):
        return _auroc(scores[key][idx], labels[idx])

    out["model_auroc_ci"] = donor_bootstrap_ci(
        lambda idx: auroc_on(idx, "model"), unit_donor, n_boot=n_boot, seed=seed + 37)
    for key in ("gaussian", "linear", "program"):
        def diff(idx, key=key):
            m, a = auroc_on(idx, "model"), auroc_on(idx, key)
            return None if m is None or a is None else m - a
        out[f"model_minus_{key}_ci"] = donor_bootstrap_ci(
            diff, unit_donor, n_boot=n_boot, seed=seed + 37)
    out.update({
        "n_units": int(labels.size),
        "n_donors": int(len(np.unique(unit_donor))),
        "n_demented": int(labels.sum()),
        "n_resilient": int((labels == 0).sum()),
        "cells_per_donor_scored": MIN_CELLS_PER_DONOR,
        # Confound check: does raw IT-cell count alone separate the groups?
        "n_cells_auroc": _auroc(np.array(n_cells, dtype=float), labels),
    })
    return out


def test_resilience_contrast():
    try:
        results = [run_once(seed=s) for s in (0, 1, 2)]
    except DataUnavailable as e:
        pytest.skip(str(e))
    m = np.array([r["model_auroc"] for r in results])
    p = np.array([r["model_p"] for r in results])
    assert (m > 0.65).all(), f"model AUROCs not all > 0.65: {m}"
    assert (p < ALPHA).all(), f"donor-grouped permutation p not all < {ALPHA}: {p}"
    for arm in ("program", "gaussian"):
        lo = np.array([r[f"model_minus_{arm}_ci"]["lo"] for r in results])
        assert (lo > 0).all(), f"model - {arm} AUROC CI lower bounds not all > 0: {lo}"


def _synthetic_region(region):
    """Data-free stand-in for _load_region (plumbing check only)."""
    rng = np.random.default_rng(zlib.crc32(region.encode()))
    rows = []
    for i in range(20):
        g = "CTRL" if i < 6 else "AD"
        c = "Dementia" if i % 2 and g == "AD" else "No dementia"
        n = int(rng.integers(45, 120))
        rows += [(g, c, f"D{i}")] * n
    group = np.array([r[0] for r in rows], dtype=object)
    cog = np.array([r[1] for r in rows])
    donors = np.array([r[2] for r in rows])
    X = rng.normal(size=(len(rows), 50))
    X[cog == "Dementia", :10] += 0.8
    return X, group, cog, donors, list(range(30, 50))


def test_resilience_runs_on_synthetic_donors(monkeypatch):
    import sys
    module = sys.modules[__name__]
    monkeypatch.setattr(module, "_load_region", _synthetic_region)
    monkeypatch.setattr(module, "MODEL_DRAWS", 256)
    monkeypatch.setattr(module, "N_FITS", 2)
    out = run_once(seed=0, steps=2, n_perm=50, n_boot=50)
    assert out["n_donors"] == 14 and out["n_units"] == 28
    assert out["n_demented"] == 14
    assert 0 < out["model_p"] <= 1
    assert out["linear_auroc"] > 0.8  # the synthetic shift is a mean shift
    assert out["model_minus_program_ci"]["n_valid"] == 50
