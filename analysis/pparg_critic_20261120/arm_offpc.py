"""Study plug-in arm ``offpc``: the critic additionally observes a small off-PC gene-residual block.

Why.  The released critic observation is 150 bag summaries of 50 fixed PCs; the
default ``pbulk`` arm adds each bag's pseudobulk log1p CP10k (log of the MEAN of
counts) for all 2,000 genes.  The PPARG diagnostic (README, 0.9.7) found ~99% of
the Not-AD PPARG mean-log gap outside the PC column space.  This arm shows the
critic, for a fixed small gene set S chosen on TRAINING donors only
(select_genes.py -> selected_genes.json), the bag mean of each cell's off-PC
residual of the standardized log expression (a MEAN of LOGS, the scale of that
gap):

    z      = (log1p(1e4 x / sum x) - mean) / sd        per cell, [cells, G] (critic_features)
    r_S    = z_S - (z @ Wo) @ Wo[S]^T                   Wo = space.npz W (orthonormal [G,50])
    out    = s * (mean_cells r_S - m_S) / sd_S          [B, k]

m_S, sd_S: mean / SD of mean_cells r_S over 512 reference REAL train bags
(uniform train donor, 128 cells with replacement, numpy stream
SeedSequence([20261109, crc32('offpc'), 0])), sd floored at 0.25 x median.
s = sqrt(E||f150||^2 / k): the block has the same mean squared norm as the
released 150 features on the reference bags (the genemean/pbulk recipe).
Everything in setup is float64 numpy from the integer counts (device independent).

Real bags use ctx.Z (per-cell z of observed counts); fake bags use the trainer's
straight-through critic_features ``cf`` (forward = z of the Poisson counts,
backward through the Gamma-NB rate), so the forward function is identical for
real and fake.  The ParticleGAN update rule, recipe, optimizers, penalty and
streams are unchanged; only the critic input gains k columns.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

import arms
from arms import Arm, FeatureBlock, register

NAME = 'offpc'
GENES_FILE = Path(__file__).resolve().with_name('selected_genes.json')
REF_BAGS = 512
REF_BATCH = 32
BAG = 128
SD_FLOOR_FRAC = 0.25


def r6(x):
    return float('%.6g' % float(x))


def load_selection(path=None):
    path = GENES_FILE if path is None else path
    q = json.loads(Path(path).read_text())
    genes = [str(g) for g in q['genes']]
    if len(genes) != len(set(genes)) or not genes:
        raise ValueError('selected genes must be a nonempty list without duplicates')
    return genes, hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ref_bag_rows(by, name=NAME, n_bags=REF_BAGS, batch=REF_BATCH, bag=BAG):
    import zlib
    rng = np.random.default_rng(np.random.SeedSequence([20261109, zlib.crc32(name.encode()), 0]))
    out = []
    for _ in range(n_bags // batch):
        ds = rng.integers(len(by), size=batch)
        out.append((ds, np.stack([rng.choice(by[i], bag, replace=True) for i in ds])))
    return out


def z64(x, mean, sd):
    lib = np.maximum(x.sum(-1, keepdims=True), 1.0)
    return (np.log1p(x * (1e4 / lib)) - mean) / sd


def offpc_bag_mean(z, Wo, S):
    """[B, cells, G] standardized log expression -> [B, k] bag mean of the off-PC residual on genes S (torch or numpy)."""
    zm = z.mean(1)
    return zm[:, S] - (zm @ Wo) @ Wo[S].T


class OffpcBlock(FeatureBlock):
    name = 'offpc'

    def __init__(self, arm):
        super().__init__()
        self.arm = arm
        self.width = len(arm.S)
        self.info = dict(arm.block_info)

    def _f(self, z):
        return (offpc_bag_mean(z, self.arm.Wo, self.arm.S_t) - self.arm.m) * self.arm.w

    def real(self, ctx, idx, ds):
        return self._f(ctx.Z[idx])

    def fake(self, ctx, cf, rate, counts):
        return self._f(cf)

    def state(self):
        return {'width': self.width}


@register
class OffpcArm(Arm):
    name = NAME
    value_policy = 'none'
    needs_Z = True

    def setup(self, ctx):
        genes, digest = load_selection()
        index = {g: i for i, g in enumerate(ctx.genes)}
        missing = [g for g in genes if g not in index]
        if missing:
            raise ValueError(f'offpc genes not in the panel: {missing}')
        self.genes, self.genes_sha256 = genes, digest
        self.S = np.array([index[g] for g in genes], dtype=np.int64)
        W = ctx.space['W'].astype(np.float64)
        if np.abs(W.T @ W - np.eye(W.shape[1])).max() > 1e-4:
            raise ValueError('space W is not orthonormal')
        Wsc = W / ctx.space['pc_sd'].astype(np.float64)
        mean = ctx.space['mean'].astype(np.float64)
        sd_g = ctx.space['sd'].astype(np.float64)
        from released_update import bag_features
        ys, n150 = [], 0.0
        for ds, rows in ref_bag_rows(ctx.by):
            x = ctx.X[torch.as_tensor(rows, device=ctx.device)].double().cpu().numpy()
            z = z64(x, mean, sd_g)
            ys.append(offpc_bag_mean(z, W, self.S))
            n150 += float((bag_features(torch.from_numpy(z @ Wsc)).numpy() ** 2).sum())
        y = np.concatenate(ys)
        R, k = y.shape
        m = y.mean(0)
        sd = y.std(0)
        floor = SD_FLOOR_FRAC * float(np.median(sd))
        n_floored = int((sd < floor).sum())
        sd = np.maximum(sd, floor)
        m32, sd32 = m.astype(np.float32), sd.astype(np.float32)
        s = r6(math.sqrt((n150 / R) / k))
        self.s = s
        dev = ctx.device
        self.m = torch.as_tensor(m32, device=dev)
        self.sd = torch.as_tensor(sd32, device=dev)
        self.w = torch.as_tensor((s / sd32.astype(np.float64)).astype(np.float32), device=dev)
        self.Wo = torch.as_tensor(W.astype(np.float32), device=dev)
        self.S_t = torch.as_tensor(self.S, device=dev)
        # share of each selected gene's per-cell z variance that lies off the PC span (train reference bags)
        self.block_info = {
            'formula': 'out = s * (mean_cells(z_S - (z Wo) Wo_S^T) - m)/sd; fake z = straight-through critic_features',
            'genes': genes, 'genes_sha256': digest, 'k': int(k), 's': s,
            'reference_bags': R, 'reference_seed': f'SeedSequence([20261109, crc32({NAME!r}), 0])',
            'E_norm2_150_reference': r6(n150 / R), 'sd_floor': r6(floor), 'genes_floored': n_floored,
            'reference_bag_sd': [r6(v) for v in sd32],
        }
        self.info = {'k': int(k), 's': s, 'genes_sha256': digest}

    def feature_blocks(self, ctx):
        return [OffpcBlock(self)]

    def state(self):
        return {'s': self.s, 'genes': list(self.genes), 'genes_sha256': self.genes_sha256,
                'm': self.m.cpu(), 'sd': self.sd.cpu()}
