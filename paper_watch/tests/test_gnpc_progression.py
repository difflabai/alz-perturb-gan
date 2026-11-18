"""GNPC longitudinal plasma proteomics — progression as a modelable condition.

Find: medRxiv 2026.07.01.26356385. Longitudinal SomaScan (7,362
aptamers), 17,269 samples from 13,449 participants. Requiring
concordant baseline + AD-by-time effects yields a 30-protein signature
separating progression-linked change from static diagnostic
difference; the signature aligns more with tau / neuronal-injury
markers than with Abeta42/40.

Route: NEW DISTRIBUTION + NEW TARGET. Everything else we model is a
cross-sectional snapshot; here the unit is the per-person delta
(follow-up minus baseline proteome). Salient claim under test, at toy
scale: the delta distribution conditioned on converter status (CN at
baseline -> AD at follow-up) differs from the stable-CN delta
distribution in a way a ParticleGAN can capture on held-out people —
beating the mean-delta baseline and a shuffled-status arm.

STATUS (2026-11-17): GNPC data is consortium-access only; no public
download. Test implemented against the expected long-format schema
(participant_id, visit, diagnosis, aptamer matrix); skips cleanly.
Iteration log: notes/gnpc_progression.md
"""
from __future__ import annotations

import numpy as np
import pytest

from alztest.data import DataUnavailable, load_gnpc
from alztest.harness import (ConditionalPGAN, energy_distance,
                             mean_shift_prediction, pca_fit_transform,
                             standardize)


def _prepare(seed: int):
    df = load_gnpc()
    # Expected columns: participant_id, diagnosis (per visit), plus one
    # column per aptamer named APT_*. Two visits per participant used:
    # first (baseline) and last (follow-up).
    apt_cols = [c for c in df.columns if c.startswith("APT_")]
    assert apt_cols, "no APT_* aptamer columns in GNPC export"
    deltas, status = [], []
    for pid, g in df.groupby("participant_id"):
        if len(g) < 2:
            continue
        g = g.sort_values("visit")
        base, last = g.iloc[0], g.iloc[-1]
        d0, d1 = str(base["diagnosis"]), str(last["diagnosis"])
        if d0.startswith("CN") and d1.startswith("CN"):
            s = 0  # stable control
        elif d0.startswith("CN") and d1.startswith("AD"):
            s = 1  # converter
        else:
            continue
        deltas.append((last[apt_cols] - base[apt_cols]).to_numpy(dtype=np.float64))
        status.append(s)
    X = np.vstack(deltas)
    y = np.array(status)
    assert (y == 1).sum() >= 60 and (y == 0).sum() >= 60, \
        f"too few converters/stable: {np.bincount(y)}"
    rng = np.random.default_rng(seed)
    tr, te = [], []
    for c in (0, 1):
        ix = np.where(y == c)[0]
        rng.shuffle(ix)
        cut = int(0.8 * len(ix))
        tr.append(ix[:cut]); te.append(ix[cut:])
    return X, y, np.concatenate(tr), np.concatenate(te)


def run_once(seed: int = 0, steps: int = 300) -> dict:
    X, y, tr, te = _prepare(seed)
    P, _ = pca_fit_transform(X, n_comp=min(30, X.shape[1]))
    P = standardize(P)
    model = ConditionalPGAN(P.shape[1], 2, seed=seed)
    model.fit(P[tr], y[tr], steps=steps)
    real_conv = P[te][y[te] == 1]
    gen_conv = model.sample(1, 512, seed=seed + 1)
    ed = energy_distance(gen_conv, real_conv, seed=seed)
    ed_mean = energy_distance(
        mean_shift_prediction(P[te][y[te] == 0], P[tr][y[tr] == 1]),
        real_conv, seed=seed)
    shuf = ConditionalPGAN(P.shape[1], 2, seed=seed)
    shuf.fit(P[tr], np.random.default_rng(seed + 3).permutation(y[tr]), steps=steps)
    ed_shuf = energy_distance(shuf.sample(1, 512, seed=seed + 1), real_conv, seed=seed)
    return {"seed": seed, "ed_pgan": ed, "ed_mean_shift": float(ed_mean),
            "ed_shuffled": float(ed_shuf)}


def test_gnpc_progression_deltas():
    try:
        results = [run_once(seed=s) for s in (0, 1, 2)]
    except DataUnavailable as e:
        pytest.skip(str(e))
    for r in results:
        assert r["ed_pgan"] < r["ed_mean_shift"], r
        assert r["ed_pgan"] < r["ed_shuffled"], r
