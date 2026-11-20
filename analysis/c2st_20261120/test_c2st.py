"""Synthetic tests for the donor-held-out C2ST (CPU, no repository data)."""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c2st as C  # noqa: E402
import features as F  # noqa: E402

FAST = C.MLPConfig(hidden=(32,), epochs=15, batch=128, seeds=(0, 1))


def donor_data(n_donors=12, n=80, d=6, shift=0.0, scale=1.0, shift_dim=0, seed=0, donor_sd=.5):
    """Each donor has its own random effect; class 1 items of a donor share
    that effect (same covariates) plus an optional systematic distortion."""
    rng = np.random.default_rng(seed)
    X, y, g = [], [], []
    for k in range(n_donors):
        eff = rng.normal(0, donor_sd, d)
        a = eff + rng.normal(0, 1, (n, d))
        b = eff + rng.normal(0, 1, (n, d))
        b[:, shift_dim] = b[:, shift_dim] * scale + shift
        X += [a, b]; y += [0] * n + [1] * n; g += [k] * 2 * n
    return np.concatenate(X), np.array(y), np.array(g)


def test_auc_matches_mann_whitney_with_ties():
    rng = np.random.default_rng(1)
    s = rng.integers(0, 5, 300).astype(float); y = rng.integers(0, 2, 300)
    sklearn = pytest.importorskip('sklearn.metrics')
    assert abs(C.auc(s, y) - sklearn.roc_auc_score(y, s)) < 1e-12
    assert C.auc([0, 1], [0, 1]) == 1 and C.auc([1, 0], [0, 1]) == 0
    assert np.isnan(C.auc([1, 2], [1, 1]))


def test_grouped_folds_hold_out_whole_donors():
    g = np.repeat(np.arange(12), 5)
    f = C.grouped_folds(g, 6, seed=3, strata={k: k % 3 for k in range(12)})
    for k in range(12):
        assert len(set(f[g == k])) == 1
    assert sorted(set(f)) == list(range(6))
    p = C.grouped_folds(g, 6, pairs=[(2 * i, 2 * i + 1) for i in range(6)])
    assert (p[g == 0] == p[g == 1][0]).all()
    with pytest.raises(ValueError):
        C.grouped_folds(g, 6, pairs=[(0, 1)])


def test_run_c2st_rejects_leaky_folds():
    X, y, g = donor_data(4, 10)
    with pytest.raises(ValueError):
        C.run_c2st(X, y, g, np.arange(len(y)) % 2, 'linear')


@pytest.mark.parametrize('classifier', ['linear', 'mlp'])
def test_null_same_distribution_is_near_half(classifier):
    X, y, g = donor_data(seed=2)
    f = C.grouped_folds(g, 6, seed=0)
    r = C.run_c2st(X, y, g, f, classifier, FAST)
    assert abs(r.auc - .5) < .07


@pytest.mark.parametrize('classifier', ['linear', 'mlp'])
def test_detects_mean_shift(classifier):
    X, y, g = donor_data(shift=1.0, seed=3)
    f = C.grouped_folds(g, 6, seed=0)
    r = C.run_c2st(X, y, g, f, classifier, FAST)
    s = r.summary(n_boot=200)
    assert r.auc > .7 and s['ci'][0] > .6


def test_mlp_detects_variance_shift_linear_cannot():
    X, y, g = donor_data(scale=2.5, seed=4, donor_sd=0.)
    f = C.grouped_folds(g, 6, seed=0)
    lin = C.run_c2st(X, y, g, f, 'linear').auc
    mlp = C.run_c2st(X, y, g, f, 'mlp', C.MLPConfig(hidden=(32,), epochs=30, seeds=(0,))).auc
    assert abs(lin - .5) < .07 and mlp > .62


def test_halves_null_calibrated_despite_donor_effects():
    X, y, g = donor_data(seed=5, donor_sd=1.0)
    real = y == 0
    aucs = C.halves_null(X[real], g[real], {k: k % 2 for k in range(12)}, n_splits=6, classifier='linear')
    assert abs(np.mean(aucs) - .5) < .1


def test_permutation_importance_finds_shifted_feature():
    X, y, g = donor_data(shift=1.0, shift_dim=3, seed=6)
    f = C.grouped_folds(g, 6, seed=0)
    r = C.run_c2st(X, y, g, f, 'linear', keep_models=True)
    imp = C.permutation_importance(r, X, n_repeats=2)
    assert int(np.argmax(imp)) == 3
    uni = C.univariate_auc(X, y, g)
    assert int(np.argmax(np.abs(uni - .5))) == 3 and uni[3] > .5


def test_paired_bootstrap_diff_sign():
    X, y, g = donor_data(shift=1.0, seed=7)
    f = C.grouped_folds(g, 6, seed=0)
    a = C.run_c2st(X, y, g, f, 'linear')
    X2, y2, g2 = donor_data(shift=0.0, seed=7)
    b = C.run_c2st(X2, y2, g2, f, 'linear')
    d = C.paired_bootstrap_diff(a.score, a.y, a.groups, b.score, b.y, b.groups, n_boot=300)
    assert d['diff'] < 0 and d['ci'][1] < 0


def test_permutation_null_is_centered():
    X, y, g = donor_data(shift=1.0, seed=8)
    f = C.grouped_folds(g, 6, seed=0)
    aucs = C.permutation_null(X, y, g, f, n_perm=5, classifier='linear')
    assert abs(np.mean(aucs) - .5) < .07


def fake_real(n_genes=30, n_pc=5):
    rng = np.random.default_rng(0)
    W, _ = np.linalg.qr(rng.normal(size=(n_genes, n_pc)))
    donors = np.array([f'd{i}' for i in range(6)])
    cells = [sp.csr_matrix(rng.poisson(1., (40 + 10 * i, n_genes)).astype(np.float32)) for i in range(6)]
    real = SimpleNamespace(donors=donors, is_heldout=np.array([0, 0, 1, 1, 1, 1], bool), stage=np.array([0, 1, 0, 1, 2, 3]),
                           cells=cells, mean=np.zeros(n_genes), sd=np.ones(n_genes), W=W)
    real.pcs = [(F.log_cp10k(c.toarray()) @ W).astype(np.float32) for c in cells]
    gen = dict(pcs=rng.normal(size=(6, 8, 16, n_pc)).astype(np.float16),
               mean_log=rng.normal(size=(6, 8, n_genes)).astype(np.float32))
    return real, gen


@pytest.mark.parametrize('level', F.LEVELS)
def test_feature_builders_balance_and_reuse(level):
    real, gen = fake_real()
    cache = {}
    X, y, g = F.build(real, gen, level, cache, cap=20)
    assert set(np.unique(g)) == {2, 3, 4, 5}
    if level == 'cell_pc':
        for k in (2, 3, 4, 5):
            assert (y[g == k] == 0).sum() == (y[g == k] == 1).sum() == 20
    X2, y2, g2 = F.build(real, gen, level, cache, cap=20)
    assert np.array_equal(X[y == 0], X2[y2 == 0])
    width = {'cell_pc': 5, 'bag_gene': 30, 'bag_pcsum': 10, 'bag_inspan': 5, 'bag_offspan': 30}[level]
    assert X.shape[1] == width


def test_span_split_reconstructs_and_is_orthogonal():
    real, _ = fake_real()
    Z = np.random.default_rng(1).normal(size=(7, 30))
    coords, resid = F.span_split(Z, real)
    assert np.allclose(resid @ real.W, 0, atol=1e-10)
    Q, _ = np.linalg.qr(real.W)
    assert np.allclose(coords @ Q.T + resid, Z)


def test_batched_and_unbatched_mlp_agree_on_signal():
    X, y, g = donor_data(shift=1.0, seed=9)
    f = C.grouped_folds(g, 6, seed=0)
    a = C.run_c2st(X, y, g, f, 'mlp', C.MLPConfig(hidden=(32,), epochs=15, seeds=(0,), batched=False)).auc
    b = C.run_c2st(X, y, g, f, 'mlp', C.MLPConfig(hidden=(32,), epochs=15, seeds=(0,), batched=True)).auc
    assert abs(a - b) < .03 and b > .7


def test_batched_models_score_only_their_test_donors():
    X, y, g = donor_data(shift=1.0, seed=10)
    f = C.grouped_folds(g, 6, seed=0)
    r = C.run_c2st(X, y, g, f, 'mlp', FAST, keep_models=True)
    assert len(r.models) == 6 and sorted(m.fold for m in r.models) == list(range(6))
    assert np.isfinite(r.score).all()
    imp = C.permutation_importance(r, X, n_repeats=1)
    assert int(np.argmax(imp)) == 0
