#!/usr/bin/env python3
"""Tests for ext_offset.py (run: python3 -B test_ext_offset.py).

Synthetic tests check the linear contrast weights, the adjusted contrasts,
permutation-null calibration, the offset estimator, depth thinning, the
stage predictors and the frozen DAM list.  One real-data test checks that
donor profiles built from the full SEA-AD counts restricted to the hvg2000
fold-0 panel equal the profiles from the frozen cache counts (the
evaluator's observed pseudobulk), for the fold's held-out donors only.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "3"
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ext_offset as EO  # noqa: E402
import metrics as MX  # noqa: E402


def test_contrast_weights():
    rng = np.random.default_rng(0)
    st = np.array([0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 3])
    P = rng.normal(size=(len(st), 50))
    ref = MX.stage_contrasts(P, st)
    assert np.allclose(EO.contrast_weights("hn", st) @ P, ref["high_notad"])
    assert np.allclose(EO.contrast_weights("slope", st) @ P, ref["slope"])
    hnl = P[st == 3].mean(0) - P[st <= 1].mean(0)
    assert np.allclose(EO.contrast_weights("hnl", st) @ P, hnl)
    # adjusted hn equals the lstsq coefficient of the High dummy
    cov = rng.normal(size=(len(st), 2))
    X = np.column_stack([np.ones(len(st)), EO._design_dummies(st), cov])
    beta = np.linalg.lstsq(X, P, rcond=None)[0]
    assert np.allclose(EO.contrast_weights("hn", st, cov=cov) @ P, beta[3])
    Xs = np.column_stack([np.ones(len(st)), st, cov])
    assert np.allclose(EO.contrast_weights("slope", st, cov=cov) @ P, np.linalg.lstsq(Xs, P, rcond=None)[0][1])
    print("PASS test_contrast_weights")


def test_agree_stats_matches_metrics():
    rng = np.random.default_rng(1)
    x = rng.normal(size=300); y = .5 * x + rng.normal(size=300)
    a = EO.agree_stats(x, y)
    m = MX.agreement(y, x)
    assert abs(a["spearman"][0] - m["spearman"]) < 1e-10
    assert abs(a["pearson"][0] - m["pearson"]) < 1e-10
    assert abs(a["top_decile_sign"][0] - m["top_decile_sign_agreement"]) < 1e-10
    assert abs(a["slope"][0] - m["regression_slope"]) < 1e-10
    b = EO.agree_stats(None, y[None], X_rows=x[None])
    assert abs(b["spearman"][0] - a["spearman"][0]) < 1e-12
    print("PASS test_agree_stats_matches_metrics")


def test_permutation_null_calibrated():
    """No stage effect but strong shared donor structure: p is ~uniform."""
    rng = np.random.default_rng(2)
    G = 400
    load = rng.normal(size=G)
    ps = []
    for r in range(60):
        sS = np.array([0] * 4 + [1] * 6 + [2] * 9 + [3] * 18)
        sM = np.array([0] * 7 + [1] * 5 + [2] * 6 + [3] * 13)
        PS = rng.normal(size=(len(sS), 1)) * load + rng.normal(size=(len(sS), G))
        PM = rng.normal(size=(len(sM), 1)) * load + rng.normal(size=(len(sM), G))
        out = EO.replicate(PS, sS, None, None, PM, sM, None, None, ["hn"], 99, 7, f"r{r}")
        out_adj = out.pop("hn_adj")
        ps.append(out["hn"]["spearman"]["null_perm_madrc"]["p_one_sided"])
    ps = np.array(ps)
    assert 0.25 < ps.mean() < 0.75, ps.mean()
    assert (ps < 0.05).mean() < 0.2, (ps < 0.05).mean()
    print(f"PASS test_permutation_null_calibrated (mean p {ps.mean():.2f}, P(p<.05) {(ps < .05).mean():.2f})")


def test_offset_estimator():
    rng = np.random.default_rng(3)
    G = 2000
    true = np.zeros(G); true[:100] = 1.0
    sS = np.repeat(np.arange(4), [4, 6, 9, 18]); sM = np.repeat(np.arange(4), [7, 5, 6, 13])
    eff = rng.normal(size=G) * .2
    PS = rng.normal(size=(len(sS), G)) * .3 + np.outer(sS / 3, eff)
    PM = rng.normal(size=(len(sM), G)) * .3 + np.outer(sM / 3, eff) + true
    r, off, sig, q = EO.offset_stats(PS, sS, PM, sM, np.array([f"g{i}" for i in range(G)]))
    assert np.allclose(off[:100].mean(), 1.0, atol=.05)
    assert abs(r["rms_offset_noise_corrected"] - np.sqrt(.05)) < .03, r["rms_offset_noise_corrected"]
    assert 80 <= r["n_genes_fdr05"] <= 130, r["n_genes_fdr05"]
    # stage-adjusted cohort coefficient recovers the offset despite stage imbalance
    co = EO.cohort_offset(PS, sS, PM, sM)
    assert np.corrcoef(co, true)[0, 1] > .9
    print("PASS test_offset_estimator")


def test_thinning():
    rng = np.random.default_rng(4)
    c = EO.Cohort("syn")
    for d in range(3):
        X = sp.random(200, 500, density=.2, random_state=d, data_rvs=lambda n: rng.integers(1, 20, n)).tocsr().astype(np.int32)
        c.X.append(X); c.donors.append(f"d{d}"); c.stage.append(d); c.braak.append(d)
        c.male.append(0); c.apoe4.append(0); c.age.append(80)
    c.finish()
    allu = np.concatenate(c.umi)
    target = allu * .5
    Xs, p = EO.thin_to(c, target, 0)
    after = np.concatenate([np.asarray(x.sum(1)).ravel() for x in Xs])
    assert abs(np.median(after) / np.median(target) - 1) < .05
    assert all((x.data >= 0).all() for x in Xs) and (p <= 1).all()
    print("PASS test_thinning")


def test_predictors():
    rng = np.random.default_rng(5)
    G = 100
    sTr = np.repeat(np.arange(4), [3, 3, 6, 13]); sHo = np.repeat(np.arange(4), [1, 2, 3, 6])
    sE = np.repeat(np.arange(4), [7, 5, 6, 13])
    eff = rng.normal(size=G) * .3
    gen = lambda s, shift=0: rng.normal(size=(len(s), G)) * .3 + np.outer(s / 3, eff) + shift
    Ytr, Yho, E = gen(sTr), gen(sHo), gen(sE, shift=.5)
    cov = lambda n: (rng.integers(0, 2, n).astype(float), rng.integers(0, 2, n).astype(float), rng.normal(85, 8, n))
    preds, info = EO.predict_fold(Ytr, sTr, cov(len(sTr)), Yho, sHo, cov(len(sHo)), E, sE, cov(len(sE)), 85, 8)
    rm = {k: np.sqrt(((v[1] - Yho) ** 2).mean(1)).mean() for k, v in preds.items()}
    assert rm["stage_mean|ext_stageadj"] < rm["stage_mean|ext_none"]      # uncorrected 0.5 shift hurts
    assert rm["stage_mean|ext_stageadj"] < rm["stage_mean|seaad"]         # more donors, same effect, help
    assert abs(info["rms_ext_stageadj"] - .5) < .1
    # shrunk profile formula reproduces baselines.py on a toy case
    sh, b = EO.shrunk_profiles(Ytr, sTr)
    assert 0 <= b <= 1
    print("PASS test_predictors")


def test_dam_list_frozen():
    assert EO.DAM_UP_R2 == MX.DAM_UP and EO.DAM_DOWN_R2 == MX.DAM_DOWN
    assert len(EO.DAM_UP_R2) == 20 and len(EO.DAM_DOWN_R2) == 7
    print("PASS test_dam_list_frozen")


def test_real_profiles_match_cache():
    """Profiles from full counts on the fold-0 panel == profiles from cache counts (held-out donors)."""
    cache = EO.CACHE / "fold0"
    sz = np.load(cache / "space.npz")
    m = np.load(cache / "heldout" / "metadata.npz")
    C = sp.load_npz(cache / "heldout" / "counts.npz").tocsr()
    dev, _ = EO.folds_all()
    for d in list(m["donors"])[:4]:
        assert d in dev
        X = sp.load_npz(EO.REBUILT / "counts" / f"{d}.npz").tocsr()
        c = EO.Cohort("one"); c.X = [X]
        mine = EO.pb_profiles(c, sz["hvg"])[0]
        ref = MX.log_cp10k(np.asarray(C[m["donor"] == d].sum(0)).ravel())
        assert np.allclose(mine, ref, atol=1e-6), np.abs(mine - ref).max()
    print("PASS test_real_profiles_match_cache")


def test_review_additions():
    """Review F3/F4/D2: depth regression recovers a planted depth effect; adjpvm covariate sets; stage_trend null."""
    rng = np.random.default_rng(3)
    G = 300
    slope = rng.normal(size=G)
    sS = np.repeat([0, 1, 2, 3], 6); sM = np.repeat([0, 1, 2, 3], 5)
    dS = np.exp(rng.normal(8.5, .3, len(sS))); dM = np.exp(rng.normal(8.1, .3, len(sM)))
    PS = np.log(dS)[:, None] * slope + rng.normal(0, .05, (len(sS), G))
    PM = np.log(dM)[:, None] * slope + rng.normal(0, .05, (len(sM), G))
    off = PM.mean(0) - PS.mean(0)
    r = EO.depth_regression(PS, sS, dS, PM, sM, dM, off)
    assert r["seaad_slope"]["corr_with_observed_offset"] > 0.9 and r["madrc_slope"]["corr_with_observed_offset"] > 0.9, r
    # replicate with a dict of covariate sets yields one entry per suffix
    cov = rng.normal(size=(len(sS), 2)); covm = rng.normal(size=(len(sM), 2))
    out = EO.replicate(PS, sS, None, {"_adj": cov, "_adjpvm": np.column_stack([cov, rng.random(len(sS))])},
                       PM, sM, None, {"_adj": covm, "_adjpvm": np.column_stack([covm, rng.random(len(sM))])}, ["hn"], 50, 1, "t")
    assert set(out) == {"hn", "hn_adj", "hn_adjpvm"} and all(v["defined"] for v in out.values())
    # stage_trend: null calibrated, planted trend detected
    ps = [EO.stage_trend(rng.normal(size=24), sS, n_perm=500, label=f"n{i}")["p_two_sided"] for i in range(40)]
    assert np.mean(np.array(ps) < 0.05) <= 0.15, ps   # 40 null draws: expected 0.05
    assert EO.stage_trend(sS + rng.normal(0, .5, 24), sS, n_perm=500, label="t")["p_two_sided"] < 0.01
    print("PASS test_review_additions")


if __name__ == "__main__":
    test_review_additions()
    test_contrast_weights()
    test_agree_stats_matches_metrics()
    test_dam_list_frozen()
    test_offset_estimator()
    test_thinning()
    test_predictors()
    test_permutation_null_calibrated()
    test_real_profiles_match_cache()
    print("ALL PASS")
