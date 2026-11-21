"""GSE289721 — CRISPRi Perturb-seq in human iPSC microglia.

Find: Cardona et al., J Neuroinflammation 2025 (PMID 41340152);
GEO GSE289721, public 2025-12-31. CRISPRi screens of 119 AD GWAS
hits + 52 control genes in human iPSC-derived microglia; the
Perturb-seq arm profiles 6 targets (AGFG2, EED, INPP5D, MS4A6A,
PVR, RABEP1) + non-targeting control with per-cell guide capture
(72 guides; assignment from the GEO feature reference).

Route: NEW DISTRIBUTION — expression conditioned on perturbation
target in real human microglia: the distribution class the NeuroKO
test (PR #10) was built for, on data that is actually downloadable.
Design mirrors that test: conditional ParticleGAN (unmodified EMPF)
vs shuffled-target, mean-shift, and control-only arms; per-target
energy distance to held-out cells + shift-direction cosine. Per the
fmaudit lesson, direction recovery against shuffled labels is the
evidence, not raw shift size. PCA and standardization are fit on
the training split only (the NeuroKO draft refit them on test;
fixed here).

Iteration log: notes/gse289721_perturbseq.md
"""
from __future__ import annotations

import numpy as np
import pytest

from alztest.data import DataUnavailable, load_gse289721
from alztest.harness import (ConditionalPGAN, energy_distance,
                             mean_shift_prediction, pca_fit_transform,
                             shift_cosine)

MIN_CELLS = 300
HELDOUT_MIN = 40
TRAIN_CAP = 800  # iteration 2: fixed subsample, as in the SEA-AD tests


def _prepare(seed: int):
    adata = load_gse289721()
    tgt = adata.obs["target"].astype(str).to_numpy()
    counts = {k: int((tgt == k).sum()) for k in np.unique(tgt)}
    ctrl = "Non-Targeting"
    assert ctrl in counts, f"no NT control in {sorted(counts)}"
    kos = sorted([k for k, c in counts.items()
                  if c >= MIN_CELLS and k not in (ctrl, "Unassigned")],
                 key=lambda k: -counts[k])
    conds = [ctrl] + kos
    rng = np.random.default_rng(seed)
    # Sparse-first preparation: the full dense matrix is 49,568 x
    # 36,601 (~14 GB float64) and OOM-killed the first attempt.
    # Subset to guide-assigned cells, log-normalize sparsely, pick
    # top-variable genes from sparse moments, densify only those.
    from scipy import sparse as _sp
    keep = tgt != "Unassigned"
    Xs = adata.X[keep].tocsr().astype(np.float64)
    tgt_k = tgt[keep]
    rs = np.asarray(Xs.sum(1)).ravel()
    Xn = Xs.multiply((1.0 / (rs + 1e-9) * 1e4)[:, None]).tocsr()
    Xn.data = np.log1p(Xn.data)
    mean = np.asarray(Xn.mean(0)).ravel()
    var = np.asarray(Xn.multiply(Xn).mean(0)).ravel() - mean ** 2
    top_all = np.argsort(var)[::-1][:2000]
    raw = Xn[:, top_all].toarray()
    Xtr, ytr, Xte, yte = [], [], [], []
    for ci, k in enumerate(conds):
        ix = np.where(tgt_k == k)[0]
        rng.shuffle(ix)
        cut = int(0.8 * len(ix))
        tr_ix = ix[:cut][:TRAIN_CAP]
        Xtr.append(raw[tr_ix]); ytr += [ci] * len(tr_ix)
        Xte.append(raw[ix[cut:]]); yte += [ci] * (len(ix) - cut)
    return (np.vstack(Xtr), np.array(ytr), np.vstack(Xte),
            np.array(yte), conds, counts)


def run_once(seed: int = 0, steps: int = 300) -> dict:
    Xtr, ytr, Xte, yte, conds, counts = _prepare(seed)
    Ptr, (mu, comps) = pca_fit_transform(Xtr, n_comp=30)
    Pte = (Xte - mu) @ comps.T
    sd = Ptr.std(0) + 1e-8
    Ptr = Ptr / sd
    Pte = Pte / sd
    n_cond = len(conds)

    model = ConditionalPGAN(Ptr.shape[1], n_cond, seed=seed)
    model.fit(Ptr, ytr, steps=steps)
    shuf_labels = np.random.default_rng(seed + 31).permutation(ytr)
    shuf = ConditionalPGAN(Ptr.shape[1], n_cond, seed=seed)
    shuf.fit(Ptr, shuf_labels, steps=steps)

    # Iteration 3 readout decomposition: model-vs-real ED on the
    # CONTROL condition itself separates marginal fit quality from
    # conditioning (same fits, no retraining choices changed).
    ed_ctrl_cond = energy_distance(model.sample(0, 512, seed=seed),
                                   Pte[yte == 0], seed=seed)
    eds, eds_shuf, eds_mean, eds_ctrl, coss = [], [], [], [], []
    per_target = {}
    for k in range(1, n_cond):
        real = Pte[yte == k]
        if len(real) < HELDOUT_MIN:
            continue
        ctrl_te = Pte[yte == 0]
        gen = model.sample(k, 512, seed=seed + k)
        gen_ctrl = model.sample(0, 512, seed=seed)
        per_target[conds[k]] = {
            "ed_pgan": energy_distance(gen, real, seed=seed),
            "cos": shift_cosine(gen, gen_ctrl, real, ctrl_te),
        }
        eds.append(per_target[conds[k]]["ed_pgan"])
        eds_shuf.append(energy_distance(
            shuf.sample(k, 512, seed=seed + k), real, seed=seed))
        eds_mean.append(energy_distance(
            mean_shift_prediction(ctrl_te, Ptr[ytr == k]), real, seed=seed))
        eds_ctrl.append(energy_distance(ctrl_te, real, seed=seed))
        coss.append(per_target[conds[k]]["cos"])
    return {
        "seed": seed, "n_targets_scored": len(eds),
        "ed_pgan": float(np.mean(eds)), "ed_shuffled": float(np.mean(eds_shuf)),
        "ed_mean_shift": float(np.mean(eds_mean)),
        "ed_control_only": float(np.mean(eds_ctrl)),
        "shift_cosine": float(np.mean(coss)),
        "ed_pgan_control_condition": float(ed_ctrl_cond),
        "per_target": per_target,
    }


def test_gse289721_conditional_shifts():
    try:
        results = [run_once(seed=s) for s in (0, 1, 2)]
    except DataUnavailable as e:
        pytest.skip(str(e))
    for r in results:
        assert r["ed_pgan"] < r["ed_shuffled"], r
        assert r["ed_pgan"] < r["ed_mean_shift"], r
        assert r["ed_pgan"] < r["ed_control_only"], r
    assert float(np.mean([r["shift_cosine"] for r in results])) > 0.3, results
