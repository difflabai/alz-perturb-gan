"""Perez-Alcantara iteration 2 — genotype-conditioned fit.

Condition: donor genotype group at the BIN1 cis-eQTL
2_127135234_C_T (the strongest AD-locus eQTL in the paper's own
untreated nominal table: p=7.05e-22, AF=0.392, slope +0.78).
Untreated arm only; donor is the unit — donors are split 80/20
(stratified by dosage), never cells, so no donor appears on both
sides. Arms: conditional ParticleGAN (unmodified EMPF) vs
shuffled-genotype ParticleGAN vs mean-shift vs dosage-0-only.
Metrics per non-reference dosage group (1, 2): energy distance of
generated vs held-out real cells, and the shift cosine of the
group centroid displacement from the dosage-0 centroid.

Iteration log: notes/perez_alcantara_genotype.md
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from alztest.data import (DataUnavailable, load_perez_untreated_metadata,
                          perez_bin1_genotypes, perez_untreated_h5ad)
from alztest.harness import (ConditionalPGAN, energy_distance,
                             mean_shift_prediction, pca_fit_transform,
                             shift_cosine, standardize)

MIN_CELLS_PER_DONOR = 200
MAX_CELLS_PER_DONOR = 300
N_COMP = 30


def _prepare(seed: int):
    import anndata as ad
    meta = load_perez_untreated_metadata()
    geno = perez_bin1_genotypes()
    counts = meta["donor_id"].astype(str).value_counts()
    donors = [d for d in counts.index[counts >= MIN_CELLS_PER_DONOR]
              if d in geno]
    if len(donors) < 60:
        raise DataUnavailable(f"only {len(donors)} genotyped donors "
                              f"with >={MIN_CELLS_PER_DONOR} cells")
    rng = np.random.default_rng(seed)
    train_d, test_d = [], []
    for g in (0, 1, 2):
        gd = [d for d in donors if geno[d] == g]
        rng.shuffle(gd)
        k = max(1, int(round(0.2 * len(gd))))
        test_d += gd[:k]; train_d += gd[k:]
    adata = ad.read_h5ad(perez_untreated_h5ad(), backed="r")
    obs = adata.obs
    did = obs["donor_id"].astype(str).to_numpy()
    rows_by = {}
    for split, dset in (("train", set(train_d)), ("test", set(test_d))):
        rows = []
        for d in dset:
            idx = np.where(did == d)[0]
            if len(idx) > MAX_CELLS_PER_DONOR:
                idx = rng.choice(idx, MAX_CELLS_PER_DONOR,
                                   replace=False)
            rows.append(np.sort(idx))
        rows_by[split] = np.sort(np.concatenate(rows))
    def load(rows):
        X = adata[rows].X
        X = X.tocsr() if sparse.issparse(X) else sparse.csr_matrix(X)
        X = X.astype(np.float32)
        tot = np.asarray(X.sum(1)).ravel() + 1e-9
        X = X.multiply(1e4 / tot[:, None]).tocsr()
        X.data = np.log1p(X.data)
        return X, did[rows]
    Xtr, dtr = load(rows_by["train"])
    Xte, dte = load(rows_by["test"])
    n = Xtr.shape[0]
    s1 = np.asarray(Xtr.sum(0)).ravel(); s2 = np.asarray(
        Xtr.multiply(Xtr).sum(0)).ravel()
    var = s2 / n - (s1 / n) ** 2
    top = np.argsort(var)[::-1][:2000]
    Ptr, proj = pca_fit_transform(np.asarray(
        Xtr[:, top].todense(), dtype=np.float64), n_comp=N_COMP,
        seed=seed)
    mu, sd = Ptr.mean(0), Ptr.std(0) + 1e-9
    Ptr = (Ptr - mu) / sd
    Pte = ((np.asarray(Xte[:, top].todense(), dtype=np.float64)
            - proj[0]) @ proj[1].T - mu) / sd
    return (Ptr, np.array([geno[d] for d in dtr]),
            Pte, np.array([geno[d] for d in dte]))


def run_once(seed: int = 0, steps: int = 300) -> dict:
    Xtr, ctr, Xte, cte = _prepare(seed)
    groups = [1, 2]
    def evaluate(sample_fn):
        eds, coss = [], []
        real0 = Xte[cte == 0]
        for g in groups:
            real = Xte[cte == g]
            if len(real) < 40:
                continue
            gen = sample_fn(g, len(real))
            eds.append(energy_distance(gen, real, seed=seed))
            coss.append(shift_cosine(gen, sample_fn(0, len(real0)),
                                     real, real0))
        return float(np.mean(eds)), float(np.mean(coss))
    model = ConditionalPGAN(Xtr.shape[1], 3, seed=seed)
    model.fit(Xtr, ctr, steps=steps)
    shuf = ConditionalPGAN(Xtr.shape[1], 3, seed=seed + 1)
    shuf.fit(Xtr, np.random.default_rng(seed).permutation(ctr),
             steps=steps)
    out = {"seed": seed}
    out["ed_model"], out["cos_model"] = evaluate(
        lambda g, n: model.sample(g, n, seed=seed))
    out["ed_shuffled"], out["cos_shuffled"] = evaluate(
        lambda g, n: shuf.sample(g, n, seed=seed))
    ms = mean_shift_prediction(Xtr[ctr == 0], Xtr[ctr == 1])
    eds, coss = [], []
    real0 = Xte[cte == 0]
    rng = np.random.default_rng(seed)
    for g in groups:
        real = Xte[cte == g]
        if len(real) < 40:
            continue
        pred = mean_shift_prediction(Xtr[ctr == 0], Xtr[ctr == g])
        eds.append(energy_distance(
            pred[rng.integers(0, len(pred), len(real))], real,
            seed=seed))
        coss.append(shift_cosine(pred, Xtr[ctr == 0], real, real0))
    out["ed_meanshift"] = float(np.mean(eds))
    out["cos_meanshift"] = float(np.mean(coss))
    eds0 = []
    for g in groups:
        real = Xte[cte == g]
        if len(real) < 40:
            continue
        eds0.append(energy_distance(
            Xtr[ctr == 0][rng.integers(0, int((ctr == 0).sum()),
                                       len(real))], real, seed=seed))
    out["ed_controlonly"] = float(np.mean(eds0))
    return out


def test_perez_genotype_fit():
    try:
        results = [run_once(seed=s) for s in (0, 1, 2)]
    except DataUnavailable as e:
        pytest.skip(str(e))
    for r in results:
        assert r["ed_model"] < r["ed_shuffled"], r
        assert r["ed_model"] < r["ed_meanshift"], r
        assert r["ed_model"] < r["ed_controlonly"], r
        assert r["cos_model"] > 0.3, r
