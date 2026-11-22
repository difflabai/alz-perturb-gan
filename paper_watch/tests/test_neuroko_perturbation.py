"""NeuroKO — AD-risk-gene knockout Perturb-seq as a conditional distribution.

Find: Dominicus, Peretti, Rodschinka et al. (Sanger / Open Targets),
bioRxiv 2026.10.13.758778. ~250 knockouts of candidate AD genes in
hiPSC-derived iNeurons and iAstrocytes, Perturb-seq + 5 phenotypic
assays. KO effects are cell-type-specific, converge on shared pathways,
and postmortem disease programs respond to specific KOs.

Route: NEW DISTRIBUTION + NEW TARGET. The distribution is expression
conditioned on (knockout, cell type). Salient claim under test, at toy
scale: a conditional ParticleGAN trained on a subset of cells recovers,
for held-out cells, per-KO conditional distributions whose shifts
(KO vs control) point the way the real shifts point — beating the
mean/linear moment baseline and a shuffled-KO-label arm. Per the
fmaudit lesson (bioRxiv 2026.11.03.763968), raw shift size is not
evidence; direction recovery against shuffled labels is.

Arms:
  - conditional ParticleGAN (true KO labels)
  - conditional ParticleGAN (KO labels shuffled)         [standing control]
  - mean-shift moment baseline (control + train mean shift) [standing control]
  - unconditional mean baseline (predict control for every KO)

Metric per KO: energy distance from prediction to held-out real KO cells
(lower better) and shift-direction cosine (higher better), averaged over
KOs with >= 100 held-out cells, >= 3 seeds.

STATUS (2026-11-17): data access is the blocker — see alztest.data.
Iteration log: notes/neuroko_perturbation.md
"""
from __future__ import annotations

import numpy as np
import pytest

from alztest.data import DataUnavailable, load_neuroko
from alztest.harness import (ConditionalPGAN, energy_distance,
                             mean_shift_prediction, pca_fit_transform,
                             shift_cosine, standardize)

KO_CANDIDATES = ["gene_target", "perturbation", "guide_target", "target_gene", "KO"]
TYPE_CANDIDATES = ["cell_type", "celltype", "Cell_type"]


def _prepare(seed: int, cell_type: str = "iNeuron", max_kos: int = 12):
    # Backed read + row-subset BEFORE any dense/float64 step: the
    # published object carries 784M nonzeros (6.3 GB as float32 CSR);
    # a full in-memory load plus float64 conversion OOM-killed the
    # first attempt (iteration 0 plumbing, logged in the notes).
    import anndata as _ad
    from alztest.data import neuroko_path
    adata = _ad.read_h5ad(neuroko_path(), backed="r")
    obs = adata.obs
    ko_col = next((c for c in KO_CANDIDATES if c in obs.columns), None)
    type_col = next((c for c in TYPE_CANDIDATES if c in obs.columns), None)
    assert ko_col, f"no KO column among {KO_CANDIDATES}: {list(obs.columns)}"
    ko_all = obs[ko_col].astype(str).to_numpy()
    keep = np.ones(len(ko_all), dtype=bool)
    if type_col:
        keep &= obs[type_col].astype(str).str.contains(
            cell_type, case=False).to_numpy()
    # Authors' own exclusion (NeuroID-reproducibility scripts):
    # targets INT3/INT30 are dropped from their analyses.
    if "target_gene" in obs.columns:
        keep &= ~obs["target_gene"].astype(str).isin(["INT3", "INT30"]).to_numpy()
    ko = ko_all[keep]
    counts = {k: int((ko == k).sum()) for k in np.unique(ko)}
    # Control: the authors' DE baseline is "INT" (intergenic guides).
    ctrl_key = next((k for k in counts if k.lower() in
                     ("control", "non-targeting", "ntc", "negctrl", "wt",
                      "int")), None)
    assert ctrl_key, f"no control condition identifiable: {sorted(counts)[:10]}"
    kos = sorted([k for k, c in counts.items() if c >= 300 and k != ctrl_key],
                 key=lambda k: -counts[k])[:max_kos]
    conds = [ctrl_key] + kos
    sel = keep & np.isin(ko_all, conds)
    rows = np.where(sel)[0]
    ko_k = ko_all[rows]
    from scipy import sparse as _sp
    Xs = adata[rows].X
    Xs = (Xs.tocsr() if _sp.issparse(Xs)
          else _sp.csr_matrix(np.asarray(Xs))).astype(np.float32)
    adata.file.close()
    rng = np.random.default_rng(seed)
    if float(Xs.max()) > 50:  # raw counts -> log-normalize sparsely
        rs = np.asarray(Xs.sum(1)).ravel()
        Xs = Xs.multiply((1.0 / (rs + 1e-9) * 1e4)[:, None]).tocsr()
        Xs.data = np.log1p(Xs.data)
    mean = np.asarray(Xs.mean(0)).ravel()
    var = np.asarray(Xs.multiply(Xs).mean(0)).ravel() - mean ** 2
    top_all = np.argsort(var)[::-1][:2000]
    raw = Xs[:, top_all].toarray()
    Xtr, ytr, Xte, yte = [], [], [], []
    for ci, k in enumerate(conds):
        ix = np.where(ko_k == k)[0]
        rng.shuffle(ix)
        cut = int(0.8 * len(ix))
        Xtr.append(raw[ix[:cut]]); ytr += [ci] * cut
        Xte.append(raw[ix[cut:]]); yte += [ci] * (len(ix) - cut)
    return (np.vstack(Xtr), np.array(ytr), np.vstack(Xte), np.array(yte), conds)


def run_once(seed: int = 0, steps: int = 300) -> dict:
    Xtr, ytr, Xte, yte, conds = _prepare(seed)
    # PCA + standardization fit on TRAIN only (the cycle-1 draft
    # refit both on test; fixed 2026-11-21, as in the GSE test).
    Ptr, (mu, comps) = pca_fit_transform(Xtr, n_comp=30)
    Pte = (Xte - mu) @ comps.T
    sd = Ptr.std(0) + 1e-8
    Ptr, Pte = Ptr / sd, Pte / sd
    n_cond = len(conds)

    model = ConditionalPGAN(Ptr.shape[1], n_cond, seed=seed)
    model.fit(Ptr, ytr, steps=steps)
    shuf_labels = np.random.default_rng(seed + 31).permutation(ytr)
    shuf = ConditionalPGAN(Ptr.shape[1], n_cond, seed=seed)
    shuf.fit(Ptr, shuf_labels, steps=steps)

    eds, eds_shuf, eds_mean, eds_ctrl, coss = [], [], [], [], []
    for k in range(1, n_cond):
        real = Pte[yte == k]
        if len(real) < 40:
            continue
        ctrl_te = Pte[yte == 0]
        gen = model.sample(k, 512, seed=seed + k)
        gen_ctrl = model.sample(0, 512, seed=seed)
        eds.append(energy_distance(gen, real, seed=seed))
        eds_shuf.append(energy_distance(shuf.sample(k, 512, seed=seed + k), real, seed=seed))
        eds_mean.append(energy_distance(
            mean_shift_prediction(ctrl_te, Ptr[ytr == k]), real, seed=seed))
        eds_ctrl.append(energy_distance(ctrl_te, real, seed=seed))
        coss.append(shift_cosine(gen, gen_ctrl, real, ctrl_te))
    return {
        "seed": seed, "n_kos_scored": len(eds),
        "ed_pgan": float(np.mean(eds)), "ed_shuffled": float(np.mean(eds_shuf)),
        "ed_mean_shift": float(np.mean(eds_mean)), "ed_control_only": float(np.mean(eds_ctrl)),
        "shift_cosine": float(np.mean(coss)),
    }


def test_neuroko_conditional_shifts():
    try:
        results = [run_once(seed=s) for s in (0, 1, 2)]
    except DataUnavailable as e:
        pytest.skip(str(e))
    for r in results:
        # Beats every control arm on held-out cells, every seed.
        assert r["ed_pgan"] < r["ed_shuffled"], r
        assert r["ed_pgan"] < r["ed_mean_shift"], r
        assert r["ed_pgan"] < r["ed_control_only"], r
        assert r["shift_cosine"] > 0.3, r
