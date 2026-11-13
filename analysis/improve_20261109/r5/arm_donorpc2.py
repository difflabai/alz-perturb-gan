"""Round-2 arm ``donorpc2`` / ``donorpc2@share`` (failure F4): donor-PC critic block with a bounded variance share.

Why.  Round-1 ``donorpc`` appended ``((mean_cells F) - mu) V / sd_k``: the
bag-mean standardised profile projected on the top K = 16 principal
directions of the train-donor mean profiles, each component divided by its own
train-donor score SD (per-component whitening).  Whitening gives the 16
components equal weight, so components 10-16, which carry little real
between-donor variance (and largely encode individual train donors), get the
same input scale as the first one.

Measured on the fold-0 hvg2000 cache (test_arm_critic.py ``shares``; 400
real train bags, 16 x 128 cells per train donor; "share" = V_b / (V_150 +
V_b), V = total variance over bags; V_150 = 10.17):

    block                      width  share  between-donor share  top-1 / top-2 / top-16 direction share
    round-1 donorpc (whitened)   16   0.614        0.610           0.040 / 0.080 / 0.614
    round-1 genemean           2000   0.837        0.667           0.134 / 0.244 / 0.610
    donorpc2 (default 0.25)      16   0.251        0.250           0.056 / 0.101 / 0.251
    pbulk                      2000   0.838        0.513           0.100 / 0.194 / 0.466

So by INPUT variance the round-1 donorpc block was not larger than
genemean's (0.61 vs 0.84; genemean's 16 leading directions alone carry
0.61), even though trained donorpc critics put ~75% of their first-layer
pre-activation variance on it.  What differs is how the variance is spread:
whitening spreads it evenly over 16 donor-identity directions (PC1 only
0.04), and it is almost pure between-donor signal (bag noise ~1%).  On fake
bags of the frozen round-1 fits (256 bags each, seeds 0/1) the donorpc
generators matched their own block (standardised mean offset 0.04-0.05,
like base's 0.02-0.03) while their gene-level error grew: genemean-block
offset 1.44-1.87 vs 0.61-0.62 for base, pbulk-block offset 1.57-1.94 vs
0.76-0.79.  I.e. the 16 whitened directions absorbed the critic and the
gene levels outside them drifted.  The variance-share hypothesis of
RESULTS.md therefore is at most part of the story; donorpc2 tests the
bounded, unwhitened version of the idea.

What ``donorpc2@share`` does.  Same directions as round 1 (top K = min(16,
n_train - 1) right singular vectors V of the centred train-donor mean
profiles, same sign convention), but NO per-component whitening: the block is

    x_pc = alpha * ((mean_cells F) - mu) V          [B, K]

with one global scale alpha chosen so that, on R = 256 reference real train
bags (uniform train donor, 128 cells with replacement, exactly the training
batch recipe; numpy stream SeedSequence([20261109, crc32('donorpc2'), 0])),

    share = alpha^2 V_pc / (V_150 + alpha^2 V_pc)

where V_150 and V_pc are the total variances (sum over columns of the
variance across the R bags) of the released 150 bag features and of the
unscaled block.  So ``share`` is the block's fraction of the total variance of
the critic's feature input (released 150 + this block; the 7 condition
columns are not counted), on real train bags.  The natural PC spectrum is kept
inside the block (PC1 dominates, minor PCs stay small).  F is the same tensor
the released features use: real bags standardize_counts(real counts), fake
bags the straight-through critic_features (gradient through the Gamma rate,
as in round 1).

Default share.  ``donorpc2`` (no value) uses share = DEFAULT_SHARE = 0.25:
the variance genemean puts in its two leading directions (0.244, table
above), i.e. the whole 16-direction block gets no more input variance than
genemean's top-2 directions, and well under round-1 donorpc's 0.61 (which
equals genemean's top-16 share).  Unwhitened, PC1 then carries 0.056 and
the minor components very little.  Values for a sweep: ``donorpc2@0.1`` and
``donorpc2@0.5`` (canonical '%g' text, 0 < share < 1; ``donorpc2@0.25`` is
refused: use ``donorpc2``).  The share is fixed in advance from these
fold-0 measurements of TRAIN bags; it is not tuned on held-out donors.  On
fake bags of the round-1 fits the default block has share 0.16-0.22.

Constraints.  Conflicts with ``donorpc`` and ``genemean`` (``genemean@s``):
like round 1, the donor-PC and gene-mean blocks are alternatives, and the
share is defined relative to the released 150 features only.  Composable
with ``pbulk`` and the other plug-ins (its share is then relative to the
released 150 + this block, not the total).

Device independence.  setup() computes everything (donor profiles, SVD,
reference-bag variances) in float64 numpy on CPU from the integer counts and
the cache's space.npz, then stores mu/V as float32 and alpha rounded to six
significant digits, so a GPU-trained fit and a CPU regeneration recompute the
same arm state (the exact consistency check of hook f).

State (hook f): K, alpha, share, mu [G], V [G,K]; block info has the measured
reference variances, explained-variance fractions and per-component score SDs.
"""
from __future__ import annotations

import math
import zlib

import numpy as np
import torch

import arms
from arms import Arm, FeatureBlock, register, canonical_float

NAME = 'donorpc2'
DEFAULT_SHARE = 0.25
REF_BAGS = 256
REF_BATCH = 32
BAG = 128
MAX_K = arms.DONORPC_MAX_K  # 16, as round 1


def r6(x):
    """Round to six significant digits (device/BLAS-independent stored scalars)."""
    return float('%.6g' % float(x))


def ref_bag_rows(by, name, n_bags=REF_BAGS, batch=REF_BATCH, bag=BAG):
    """Fixed reference real train bags: list of [batch, bag] row arrays (training-batch recipe)."""
    rng = np.random.default_rng(np.random.SeedSequence([20261109, zlib.crc32(name.encode()), 0]))
    out = []
    for _ in range(n_bags // batch):
        ds = rng.integers(len(by), size=batch)
        out.append((ds, np.stack([rng.choice(by[i], bag, replace=True) for i in ds])))
    return out


def counts64(ctx, rows):
    """Integer counts of the given train rows as float64 numpy (exact for any device)."""
    t = torch.as_tensor(np.asarray(rows), device=ctx.device)
    return ctx.X[t].double().cpu().numpy()


def std64(ctx, x):
    """standardize_counts in float64 numpy (same formula as pg_model)."""
    mean = ctx.space['mean'].astype(np.float64)
    sd = ctx.space['sd'].astype(np.float64)
    lib = np.maximum(x.sum(-1, keepdims=True), 1.0)
    return (np.log1p(x * (1e4 / lib)) - mean) / sd


def f150_64(ctx, z):
    """Released 150 bag features in float64 on CPU from standardised cells z [B, cells, G]."""
    W = ctx.space['W'].astype(np.float64) / ctx.space['pc_sd'].astype(np.float64)
    from released_update import bag_features
    return bag_features(torch.from_numpy(z @ W)).numpy()


def total_var(a):
    """Sum over columns of the (biased) variance across rows."""
    a = np.asarray(a, dtype=np.float64)
    return float(a.var(0).sum())


class DonorPCBlock(FeatureBlock):
    name = 'pc'

    def __init__(self, arm):
        super().__init__()
        self.arm = arm
        self.width = int(arm.K)
        self.info = dict(arm.block_info)

    def _f(self, gene_mean):
        return ((gene_mean - self.arm.mu) @ self.arm.V) * self.arm.alpha

    def real(self, ctx, idx, ds):
        return self._f(ctx.Z[idx].mean(1))

    def fake(self, ctx, cf, rate, counts):
        return self._f(cf.mean(1))

    def state(self):
        return {'width': self.width}


@register
class DonorPC2Arm(Arm):
    name = NAME
    value_policy = 'optional'
    conflicts = ('donorpc', 'genemean')
    needs_Z = True

    @classmethod
    def parse_value(cls, text):
        v, canon = canonical_float(text, lo=0.0, hi=1.0, lo_open=True, hi_open=True)
        if v == DEFAULT_SHARE:
            raise ValueError(f'{NAME}@{canon} is the default; write {NAME}')
        return v, canon

    def setup(self, ctx):
        self.share = DEFAULT_SHARE if self.value is None else float(self.value)
        n = len(ctx.by)
        K = min(MAX_K, n - 1)
        if K < 1:
            raise ValueError('donorpc2 needs at least two train donors')
        # donor mean standardised profiles, float64 on CPU
        P = np.stack([std64(ctx, counts64(ctx, r)).mean(0) for r in ctx.by])
        mu = P.mean(0)
        U, S, Vt = np.linalg.svd(P - mu, full_matrices=False)
        V = Vt[:K].T
        signs = np.sign(V[np.abs(V).argmax(0), np.arange(K)])
        V = V * signs
        score_sd = ((P - mu) @ V).std(0)
        if not (score_sd > 0).all():
            raise ValueError('degenerate donor PC')
        mu32, V32 = mu.astype(np.float32), V.astype(np.float32)
        # reference real bags: variance of the released 150 and of the unscaled block
        f150, raw, dsall = [], [], []
        for ds, rows in ref_bag_rows(ctx.by, NAME):
            z = std64(ctx, counts64(ctx, rows))
            f150.append(f150_64(ctx, z))
            raw.append((z.mean(1) - mu32.astype(np.float64)) @ V32.astype(np.float64))
            dsall.append(ds)
        f150, raw, dsall = np.concatenate(f150), np.concatenate(raw), np.concatenate(dsall)
        v150, vpc = total_var(f150), total_var(raw)
        alpha = r6(math.sqrt(self.share / (1.0 - self.share) * v150 / vpc))
        self.K = int(K)
        self.alpha = alpha
        dev = ctx.device
        self.mu = torch.as_tensor(mu32, device=dev)
        self.V = torch.as_tensor(V32, device=dev)
        # bag-sampling noise share per component on the reference bags (diagnostic)
        within = np.zeros(K)
        cnt = 0
        for d in np.unique(dsall):
            m = dsall == d
            if m.sum() > 1:
                within += raw[m].var(0, ddof=1) * m.sum(); cnt += m.sum()
        within = within / max(cnt, 1)
        self.block_info = {
            'K': self.K, 'alpha': alpha, 'share_target': self.share,
            'share_definition': 'alpha^2 V_pc / (V_150 + alpha^2 V_pc), total variances over 256 reference real train bags',
            'reference_bags': REF_BAGS, 'reference_seed': f'SeedSequence([20261109, crc32({NAME!r}), 0])',
            'V_150': r6(v150), 'V_pc_unscaled': r6(vpc),
            'E_norm2_150': r6(float((f150 ** 2).sum(1).mean())),
            'explained_variance_fraction': [r6(x) for x in (S[:K] ** 2 / (S ** 2).sum())],
            'train_donor_score_sd': [r6(x) for x in score_sd],
            'reference_bag_within_donor_var_share': [r6(x) for x in within / np.maximum(raw.var(0), 1e-300)],
            'whitening': 'none (single global scale alpha)',
        }
        self.info = {'share': self.share, 'alpha': alpha, 'K': self.K, 'V_150': r6(v150), 'V_pc_unscaled': r6(vpc)}

    def feature_blocks(self, ctx):
        return [DonorPCBlock(self)]

    def state(self):
        return {'share': self.share, 'K': self.K, 'alpha': self.alpha, 'mu': self.mu.cpu(), 'V': self.V.cpu()}
