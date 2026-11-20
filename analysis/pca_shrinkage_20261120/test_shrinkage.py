"""Tests for the PCA-shrinkage scoring package (run: python -B -m pytest -q this_file)."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '3')
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
R5 = HERE.parents[1]/'analysis/improve_20261109/r5'
sys.path.insert(0, str(HERE)); sys.path.insert(1, str(R5))
import shrinkage as S  # noqa: E402
import interval_metrics as I  # noqa: E402

CACHE = Path('/ml2/alz/runs/improve_20261109/caches/hvg2000/fold0')


def rand(seed=0, n=7, M=32, D=100, R=5):
    rng = np.random.default_rng(seed)
    samples = rng.normal(size=(n, M, D)); samples[..., 50:] = np.abs(samples[..., 50:])
    y = rng.normal(size=(R, n, D)) * 1.3; y[..., 50:] = np.abs(y[..., 50:])
    return samples, y


def test_per_target_matches_frozen_summary():
    samples, y = rand()
    frozen = I.summarize(samples, y, np.ones(len(samples), bool), 'percentile')
    t = S.block_table(samples, y, {'all': slice(0, 50)})['all']
    for g in ('mean', 'sd', 'combined'):
        for k, v in frozen[g].items():
            assert abs(float(t[g][k].mean()) - v) < 1e-12


def test_coverage_invariant_to_two_sided_rescaling():
    samples, y = rand(1)
    scale = np.exp(np.random.default_rng(2).normal(size=100))
    a = S.block_table(samples, y, S.PC_BLOCKS)
    b = S.block_table(samples*scale, y*scale, S.PC_BLOCKS)
    for blk in a:
        for g in ('mean', 'sd', 'combined'):
            for k in ('coverage', 'below', 'above'):
                assert np.allclose(a[blk][g][k], b[blk][g][k])
        assert np.allclose(a[blk]['log_sd_ratio'], b[blk]['log_sd_ratio'])


def test_shrinkage_scale_only_touches_sd_half():
    samples, _ = rand(3)
    f = np.linspace(.5, 1, 50)
    out = S.shrinkage_scale_sd(samples, f)
    assert np.array_equal(out[..., :50], samples[..., :50])
    assert np.allclose(out[..., 50:], samples[..., 50:]*f)
    with pytest.raises(ValueError):
        S.shrinkage_scale_sd(samples, -f)


def test_log_sd_ratio_sign():
    real = np.ones((3, 2, 4)); gen = np.full((2, 8, 4), .5)
    assert np.all(S.log_sd_ratio(gen, real) < 0)
    assert np.allclose(S.log_sd_ratio(gen, real), np.log(.5))


def test_whitening_identity_and_determinism():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(4000, 6)) @ rng.normal(size=(6, 6)) + 3
    mu, A = S.fit_whitening(X)
    Z = S.apply_whitening(X, mu, A)
    assert np.allclose(np.cov(Z, rowvar=False, ddof=0), np.eye(6), atol=1e-8)
    mu2, A2 = S.fit_whitening(X)
    assert np.array_equal(A, A2)
    assert np.all(np.diff(np.var(X @ (A/np.linalg.norm(A, axis=0)), 0)) <= 1e-9)  # variance-ordered


def test_bag_summary_matches_frozen():
    import metrics as MX
    x = np.random.default_rng(5).normal(size=(3, 64, 50))
    assert np.allclose(S.bag_summary(x), MX.pc_bag_summary(x))


def test_gene_moments_sd():
    L = np.abs(np.random.default_rng(6).normal(size=(256, 30))) * 2
    m, q = S.gene_moments(L)
    sd = S.gene_sd_from_moments(m, q)
    assert np.allclose(sd, L.std(0), atol=2e-2)  # float16 second moment
    assert np.all(sd >= 0)


def test_conformal_never_uses_target_and_ranks():
    rng = np.random.default_rng(7)
    n, D = 12, 5
    low = rng.normal(size=(n, D)); high = low + 1; y = rng.normal(size=(n, D))
    donors = np.array([f'd{i}' for i in range(n)])
    lo, hi, info = S.loo_conformal(low, high, y, donors)
    assert set(info['n_cal']) == {11} and set(info['rank']) == {10}
    y2 = y.copy(); y2[3] = 1e6
    lo2, hi2, _ = S.loo_conformal(low, high, y2, donors)
    assert np.array_equal(lo2[3], lo[3]) and np.array_equal(hi2[3], hi[3])
    stages = np.array([0] + [1]*2 + [2]*3 + [3]*6)
    lo3, hi3, info3 = S.loo_conformal(low, high, y, donors, stages)
    assert list(info3['n_cal']) == [0, 1, 1, 2, 2, 2] + [5]*6
    assert np.all(np.isinf(hi3[:6])) and np.all(np.isfinite(hi3[6:]))
    eligible = np.ones(n, bool); eligible[6:9] = False
    _, _, info4 = S.loo_conformal(low, high, y, donors, stages, eligible)
    assert list(info4['n_cal'][6:]) == [3, 3, 3, 2, 2, 2]  # ineligible targets still never self-calibrate
    with pytest.raises(ValueError):
        S.loo_conformal(low, high, y, np.array(['a']*n))


def test_conformal_marginal_coverage_exchangeable():
    rng = np.random.default_rng(8)
    covs = []
    for _ in range(400):
        n, D = 12, 3
        y = rng.normal(size=(n, D)) * 2          # base interval badly too narrow
        low, high = np.full((n, D), -.3), np.full((n, D), .3)
        lo, hi, _ = S.loo_conformal(low, high, y, np.arange(n).astype(str))
        covs.append(((y >= lo) & (y <= hi)).mean())
    assert np.mean(covs) >= 10/12 - .02


def test_lower_clip_and_finite_summary():
    low = np.zeros((4, 2)); high = np.ones((4, 2)); y = np.full((4, 2), 3.0)
    lo, hi, _ = S.loo_conformal(low, high, y, np.arange(4).astype(str), np.array([0, 0, 1, 1]), lower_clip=0.0)
    assert np.all(lo >= 0)
    per = S.per_target(lo, hi, y[None])
    summ = S.finite_summary(per)
    assert summ['width'] is None and summ['width_finite_targets'] == 0 and summ['coverage'] == 1.0


def test_bootstrap():
    v = np.arange(12.0)
    r = S.bootstrap(v)
    assert r['lo'] < r['mean'] < r['hi'] and r['n'] == 12
    p = S.bootstrap(v, paired=v)
    assert p['mean'] == p['lo'] == p['hi'] == 0
    assert S.bootstrap(np.array([1.0]))['lo'] is None


def test_decile_groups():
    g, edges = S.decile_groups(np.arange(100.0))
    assert np.bincount(g).tolist() == [10]*10 and len(edges) == 11


@pytest.mark.skipif(not CACHE.exists(), reason='development cache not available')
def test_resample_pool_reproduces_frozen_blind_baseline_and_real_summaries():
    import evaluate as E
    import baselines as BL
    import architecture_score as A
    import references as RF
    import run
    real = E.load_real(CACHE)
    rng_of = run.empirical_rng(E.MX)
    held = np.flatnonzero(real.is_heldout)[:2]
    train = np.flatnonzero(~real.is_heldout)
    BL_rng = BL._target_rng
    BL._target_rng = rng_of
    try:
        ref, _ = BL.generate('resample_blind', real, M=4, K=256, kpc=64)
    finally:
        BL._target_rng = BL_rng
    got = RF.resample_pool(real, held, lambda t: [i for i in train if i != t], rng_of, E.MX, M=4)
    assert np.array_equal(got['pcs'], ref['pcs'][held])
    assert np.array_equal(got['mean_log'], ref['mean_log'][held])
    assert np.array_equal(got['sq_log'], ref['sq_log'][held])
    with pytest.raises(ValueError):
        RF.resample_pool(real, held[:1], lambda t: [t], rng_of, E.MX, M=2)
    obs = A.observed_summaries(real, 64)
    mine = run.real_pc_summaries(E, real, held, 64)
    assert np.array_equal(mine, obs[:, held])
