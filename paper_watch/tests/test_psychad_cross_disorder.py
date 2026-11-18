"""PsychAD — AD-specific vs shared-degeneration shift (cross-disorder target).

Find: Lee et al., PsychAD Consortium, Nature 657:988-1002 (2026).
6.3M DLPFC nuclei, 1,494 donors, eight disorders + controls.
Cross-disorder transcriptomic vulnerability tracks genetic
co-heritability; programs are part shared, part disease-specific.

Route: NEW TARGET. Salient claim under test, at toy scale: the AD
distribution shift is partly AD's own. A conditional ParticleGAN
conditioned on diagnosis (control / AD / DLBD — the disorder PsychAD
found most AD-like) should generate an AD shift that is closer to the
real AD shift than to the real DLBD shift, and vice versa — a
specificity margin that a shuffled-diagnosis arm cannot produce.
This is the guard against modeling generic sick-brain and calling
it Alzheimer's.

STATUS (2026-11-17): controlled access (Synapse syn60084804, DUA).
Test implemented against the published obs schema; skips cleanly
until an approved export is present.
Iteration log: notes/psychad_cross_disorder.md
"""
from __future__ import annotations

import numpy as np
import pytest

from alztest.data import DataUnavailable, load_psychad
from alztest.harness import (ConditionalPGAN, shift_cosine,
                             pca_fit_transform, standardize)

DIAG_CANDIDATES = ["diagnosis", "Diagnosis", "disease", "condition"]
TYPE_CANDIDATES = ["cell_type", "subclass", "supertype"]


def _prepare(seed: int, cell_type_substr: str = "microglia"):
    adata = load_psychad()
    obs = adata.obs
    dg_col = next((c for c in DIAG_CANDIDATES if c in obs.columns), None)
    ty_col = next((c for c in TYPE_CANDIDATES if c in obs.columns), None)
    assert dg_col and ty_col, f"PsychAD obs schema: {list(obs.columns)}"
    keep = obs[ty_col].astype(str).str.contains(cell_type_substr, case=False).to_numpy()
    adata, obs = adata[keep], adata.obs[keep]
    diag = obs[dg_col].astype(str)
    groups = {}
    for name, pat in [("CTRL", "control|neurotypical|none"),
                      ("AD", "Alzheimer"), ("DLBD", "Lewy")]:
        m = diag.str.contains(pat, case=False).to_numpy()
        if m.sum() >= 300:
            groups[name] = m
    assert set(groups) == {"CTRL", "AD", "DLBD"}, f"groups found: {list(groups)}"
    cond = np.zeros(len(obs), dtype=int)
    for i, name in enumerate(["CTRL", "AD", "DLBD"]):
        cond[groups[name]] = i
    raw = np.asarray(adata.X.todense() if hasattr(adata.X, "todense") else adata.X,
                     dtype=np.float64)
    rng = np.random.default_rng(seed)
    tr, te = [], []
    for c in range(3):
        ix = np.where(cond == c)[0]
        rng.shuffle(ix)
        cut = int(0.8 * len(ix))
        tr.append(ix[:cut]); te.append(ix[cut:])
    return raw, cond, np.concatenate(tr), np.concatenate(te)


def run_once(seed: int = 0, steps: int = 300) -> dict:
    raw, cond, tr, te = _prepare(seed)
    var = raw[tr].var(0)
    top = np.argsort(var)[::-1][:2000]
    P, _ = pca_fit_transform(raw[:, top], n_comp=30)
    P = standardize(P)
    model = ConditionalPGAN(P.shape[1], 3, seed=seed)
    model.fit(P[tr], cond[tr], steps=steps)
    gen = {k: model.sample(k, 512, seed=seed + k) for k in range(3)}
    real = {k: P[te][cond[te] == k] for k in range(3)}
    ad_self = shift_cosine(gen[1], gen[0], real[1], real[0])
    ad_cross = shift_cosine(gen[1], gen[0], real[2], real[0])
    dlbd_self = shift_cosine(gen[2], gen[0], real[2], real[0])
    dlbd_cross = shift_cosine(gen[2], gen[0], real[1], real[0])
    return {"seed": seed,
            "specificity": float((ad_self - ad_cross) + (dlbd_self - dlbd_cross))}


def test_psychad_specificity_margin():
    try:
        results = [run_once(seed=s) for s in (0, 1, 2)]
    except DataUnavailable as e:
        pytest.skip(str(e))
    for r in results:
        assert r["specificity"] > 0.2, r
