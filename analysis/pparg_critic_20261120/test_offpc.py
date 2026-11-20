"""CPU tests for the offpc arm, gene selection and scoring helpers.

PYTHONPATH not needed; run:  python -B -m pytest -q analysis/pparg_critic_20261120/test_offpc.py
"""
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault('PPARG_CRITIC_RUN', str(Path(os.environ.get('TMPDIR', '/tmp')) / 'pparg_critic_test_run'))

import numpy as np
import pytest
import scipy.sparse as sp

import host_pparg as H
T, G, PROV = H.activate()
torch = T.torch
import arm_offpc as A
import select_genes as SG
import score as SC


def _orth(G_, k, seed=0):
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(G_, k)))
    return q


def fake_ctx(tmp_path, genes_sel, G_=120, cells=600, donors=6, seed=1):
    rng = np.random.default_rng(seed)
    genes = np.array([f'G{i}' for i in range(G_)], dtype='<U10')
    genes[7] = 'PPARG'
    X = rng.poisson(rng.gamma(0.5, 1.0, size=(cells, G_))).astype(np.float32)
    X[:, 0] += 1
    lab = rng.integers(donors, size=cells)
    W = _orth(G_, 50)
    lib = X.sum(1, keepdims=True)
    lz = np.log1p(X * (1e4 / lib))
    space = dict(W=W.astype(np.float32), pc_sd=np.ones(50, np.float32), mean=lz.mean(0).astype(np.float32),
                 sd=(lz.std(0) + 1e-3).astype(np.float32), genes=genes)
    Xt = torch.as_tensor(X)
    mean, sd = torch.as_tensor(space['mean']), torch.as_tensor(space['sd'])
    Z = T.standardize_counts(Xt, mean, sd)
    f = tmp_path / 'sel.json'
    f.write_text(json.dumps({'genes': genes_sel}))
    return SimpleNamespace(X=Xt, Z=Z, by=[np.flatnonzero(lab == d) for d in range(donors)], genes=genes, space=space,
                           device=torch.device('cpu'), mean=mean, sd=sd, n_genes=G_), f


def make_arm(monkeypatch, ctx, f):
    monkeypatch.setattr(A, 'GENES_FILE', f)
    arm = A.OffpcArm(None, 'offpc')
    arm.setup(ctx)
    return arm, arm.feature_blocks(ctx)[0]


def test_registered_and_arm_string():
    assert T.arms.parse_arm('pbulk,offpc') == ('offpc', 'pbulk')
    assert T.arms.arm_class('offpc') is A.OffpcArm


def test_residual_is_off_pc_and_zero_inside_span():
    W = _orth(80, 50, 3)
    rng = np.random.default_rng(4)
    S = np.arange(80)
    z = rng.normal(size=(5, 9, 80))
    r = A.offpc_bag_mean(z, W, S)
    assert np.abs(r @ W).max() < 1e-10            # with all genes, residual is orthogonal to the PC span
    inside = (rng.normal(size=(5, 9, 50)) @ W.T)
    assert np.abs(A.offpc_bag_mean(inside, W, np.array([3, 7, 11]))).max() < 1e-10
    zt = torch.as_tensor(z)
    assert np.allclose(A.offpc_bag_mean(zt, torch.as_tensor(W), torch.as_tensor(S)).numpy(), r)


def test_real_fake_forward_equal_and_gradient(tmp_path, monkeypatch):
    ctx, f = fake_ctx(tmp_path, ['PPARG', 'G3', 'G50'])
    arm, blk = make_arm(monkeypatch, ctx, f)
    assert blk.width == 3 and arm.S.tolist() == [7, 3, 50]
    idx = torch.as_tensor(np.random.default_rng(0).integers(len(ctx.X), size=(4, 16)))
    real = blk.real(ctx, idx, None)
    counts = ctx.X[idx]
    rate = counts.clone().requires_grad_(True)
    cf = T.critic_features(counts, rate, ctx.mean, ctx.sd)
    fake = blk.fake(ctx, cf, rate, counts)
    assert torch.allclose(real, fake, atol=1e-5)
    fake.square().sum().backward()
    assert rate.grad is not None and float(rate.grad[..., 7].abs().sum()) > 0


def test_reference_normalisation(tmp_path, monkeypatch):
    ctx, f = fake_ctx(tmp_path, ['PPARG', 'G3', 'G50', 'G60'])
    arm, blk = make_arm(monkeypatch, ctx, f)
    ys = []
    for ds, rows in A.ref_bag_rows(ctx.by):
        ys.append(blk.real(ctx, torch.as_tensor(rows), ds))
    y = torch.cat(ys).double()
    assert float(y.mean(0).abs().max()) < 1e-3 * arm.s
    n150 = arm.block_info['E_norm2_150_reference']
    assert abs(float(y.square().sum(1).mean()) / n150 - 1) < 1e-3   # same mean squared norm as the released block


def test_setup_is_deterministic(tmp_path, monkeypatch):
    ctx, f = fake_ctx(tmp_path, ['PPARG', 'G9'])
    a1, _ = make_arm(monkeypatch, ctx, f)
    a2, _ = make_arm(monkeypatch, ctx, f)
    assert T._same(a1.state(), a2.state())


def test_missing_gene_refused(tmp_path, monkeypatch):
    ctx, f = fake_ctx(tmp_path, ['PPARG', 'NOPE'])
    with pytest.raises(ValueError):
        make_arm(monkeypatch, ctx, f)


def _selection_inputs(tmp_path, heldout_value):
    tmp_path = tmp_path / f'h{heldout_value:g}'
    rng = np.random.default_rng(5)
    G_, D = 90, 8
    genes = np.array([f'G{i}' for i in range(G_)], dtype='<U10'); genes[7] = 'PPARG'
    cache = tmp_path / 'cache'; (cache / 'train').mkdir(parents=True)
    donors = np.array([f'D{i}' for i in range(D)])
    lab = np.repeat(donors, 30)
    X = rng.poisson(1.0, size=(len(lab), G_))
    sp.save_npz(cache / 'train/counts.npz', sp.csr_matrix(X))
    np.savez(cache / 'train/metadata.npz', donor=lab, donors=donors, raw_c=np.zeros((D, 7)))
    np.savez(cache / 'space.npz', genes=genes, W=_orth(G_, 50, 6), sd=np.ones(G_), mean=np.zeros(G_), pc_sd=np.ones(50))
    T_ = D + 3
    ml = rng.normal(size=(T_, 4, G_)).astype(np.float32)
    ml[D:] = heldout_value
    gen = tmp_path / 'gen.npz'
    np.savez(gen, mean_log=ml, is_heldout=np.r_[np.zeros(D, bool), np.ones(3, bool)],
             targets=np.r_[donors, ['H1', 'H2', 'H3']], genes=genes, stage=np.r_[np.arange(D) % 4, [0, 1, 2]])
    return gen, cache


def test_selection_ignores_heldout_rows(tmp_path):
    g1, c = _selection_inputs(tmp_path, 0.0)
    a = SG.select(g1, c, k=5)
    g2, c = _selection_inputs(tmp_path, 1e6)
    b = SG.select(g2, c, k=5)
    assert a['genes'] == b['genes'] and a['genes'][0] == 'PPARG' and len(a['genes']) == 5


def test_bootstrap_paired_and_reproducible():
    rng = np.random.default_rng(0)
    x = rng.normal(1.0, .1, size=12); y = x - .05
    r1 = SC.paired_bootstrap(x, y, reps=2000, seed=1)
    r2 = SC.paired_bootstrap(x, y, reps=2000, seed=1)
    assert r1 == r2 and r1['lo'] > 0 and r1['estimate'] == pytest.approx(.05)
