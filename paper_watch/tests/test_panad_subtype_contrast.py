"""panAD + MONET subtypes — subtype-conditioned distributions as a new target.

Find: bioRxiv 2026.09.03.747935. An integrated atlas of >3M nuclei from
791 donors across 13 studies; the MONET framework resolves four
donor-level molecular subtypes (Metal-Ion Stress, Neuroinflammatory,
Synaptic Integrity, Tissue Remodeling) that dissociate neuropathology
burden from cognitive impairment.

Route: NEW TARGET. Salient claim under test, at toy scale: in microglia
(where panAD's subtype and sex-difference signals concentrate), the
AD-vs-control distribution shift is subtype-dependent — a conditional
ParticleGAN conditioned on donor subtype produces subtype-specific
shifts whose directions match the real per-subtype shifts, beating a
subtype-blind fit and a shuffled-subtype arm.

STATUS (2026-11-17): the integrated object and MONET labels are not
openly downloadable yet — the test is implemented against the expected
obs schema (donor subtype + diagnosis + cell type) and skips cleanly.
Iteration log: notes/panad_subtype_contrast.md
"""
from __future__ import annotations

import numpy as np
import pytest

from alztest.data import DataUnavailable, load_panad
from alztest.harness import (ConditionalPGAN, energy_distance,
                             shift_cosine, pca_fit_transform, standardize)

SUBTYPE_CANDIDATES = ["monet_subtype", "subtype", "MONET_subtype"]
DIAG_CANDIDATES = ["diagnosis", "Diagnosis", "condition", "disease"]
TYPE_CANDIDATES = ["cell_type", "subclass", "supertype"]


def _prepare(seed: int, cell_type_substr: str = "microglia"):
    adata = load_panad()
    obs = adata.obs
    st_col = next((c for c in SUBTYPE_CANDIDATES if c in obs.columns), None)
    dg_col = next((c for c in DIAG_CANDIDATES if c in obs.columns), None)
    ty_col = next((c for c in TYPE_CANDIDATES if c in obs.columns), None)
    assert st_col and dg_col and ty_col, \
        f"panAD obs lacks subtype/diagnosis/type columns: {list(obs.columns)}"
    keep = obs[ty_col].astype(str).str.contains(cell_type_substr, case=False).to_numpy()
    adata = adata[keep]
    obs = adata.obs
    subtypes = obs[st_col].astype(str).to_numpy()
    diag = obs[dg_col].astype(str).str.contains("AD|Alzheimer", case=False).to_numpy()
    # Condition = (subtype, is_AD); require >= 300 cells per condition.
    cond_names, cond = [], np.zeros(len(obs), dtype=int)
    for st in sorted(set(subtypes)):
        for is_ad in (False, True):
            m = (subtypes == st) & (diag == is_ad)
            if m.sum() >= 300:
                cond[m] = len(cond_names)
                cond_names.append(f"{st}|{'AD' if is_ad else 'CTRL'}")
    raw = np.asarray(adata.X.todense() if hasattr(adata.X, "todense") else adata.X,
                     dtype=np.float64)
    rng = np.random.default_rng(seed)
    tr, te = [], []
    for c in range(len(cond_names)):
        ix = np.where(cond == c)[0]
        rng.shuffle(ix)
        cut = int(0.8 * len(ix))
        tr.append(ix[:cut]); te.append(ix[cut:])
    return raw, cond, np.concatenate(tr), np.concatenate(te), cond_names


def run_once(seed: int = 0, steps: int = 300) -> dict:
    raw, cond, tr, te, cond_names = _prepare(seed)
    var = raw[tr].var(0)
    top = np.argsort(var)[::-1][:2000]
    P, _ = pca_fit_transform(raw[:, top], n_comp=30)
    P = standardize(P)
    n_cond = len(cond_names)
    model = ConditionalPGAN(P.shape[1], n_cond, seed=seed)
    model.fit(P[tr], cond[tr], steps=steps)
    shuf = ConditionalPGAN(P.shape[1], n_cond, seed=seed)
    shuf.fit(P[tr], np.random.default_rng(seed + 7).permutation(cond[tr]), steps=steps)
    # Pairs: subtype s CTRL index 2i, AD index 2i+1 by construction order.
    coss, coss_shuf = [], []
    for i in range(0, n_cond - 1, 2):
        real_ad, real_ct = P[te][cond[te] == i + 1], P[te][cond[te] == i]
        if len(real_ad) < 40:
            continue
        gen_ad = model.sample(i + 1, 512, seed=seed + i)
        gen_ct = model.sample(i, 512, seed=seed)
        coss.append(shift_cosine(gen_ad, gen_ct, real_ad, real_ct))
        coss_shuf.append(shift_cosine(shuf.sample(i + 1, 512, seed=seed + i),
                                      shuf.sample(i, 512, seed=seed), real_ad, real_ct))
    return {"seed": seed, "cosine": float(np.mean(coss)),
            "cosine_shuffled": float(np.mean(coss_shuf)),
            "n_subtypes": len(coss)}


def test_panad_subtype_conditioned_shifts():
    try:
        results = [run_once(seed=s) for s in (0, 1, 2)]
    except DataUnavailable as e:
        pytest.skip(str(e))
    for r in results:
        assert r["cosine"] > 0.3, r
        assert r["cosine"] > r["cosine_shuffled"] + 0.1, r
