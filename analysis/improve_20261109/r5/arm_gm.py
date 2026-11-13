"""F6 genemean variants: ``genemean@s`` (core) and the plug-in ``gmlite@k`` (this file).

genemean@s (implemented in the core, r2/arms.py ``Features``; nothing is
registered here because plug-ins may not register the round-1 name)
    Round-1 genemean block with its scale multiplied by s: block =
    s * s_gm * mean_cells(F), s_gm and its 256 reference bags unchanged.
    ``genemean`` (no value) is bitwise round 1; ``genemean@1`` is refused.
    Values to screen (fixed here, before any round-2 fit, from the round-1
    diagnosis in r2/gm_diagnose.py; held-out donors were not used to choose
    them): s = 0.5 and s = 0.25.  Reason: in the trained round-1 genemean
    critics (6 per panel) the gene block carries 93% (hvg2000) / 92% (wide)
    of the across-bag variance of the critic's first-layer pre-activation on
    real bags and the released 150 PC features 6-7% (base: released 96%),
    although s_gm gives the two blocks equal second moments; the critic has
    shrunk its weights on the released columns to RMS 0.74 (base 2.7) and an
    integrated-gradient attribution puts 81% / 83% of the ~13-logit critic gap
    on the gene block.  With the trained first-layer weights held fixed the
    block's pre-activation share scales with s^2: 13:1 at s = 1 becomes about
    3:1 (77%) at s = 0.5 and about 0.8:1 (45%) at s = 0.25.  The critic
    re-weights during training (it did so against the released block), so
    the realised shares will lie between these and the s = 1 values.  s = 0.5
    is the mild step expected to keep most of genemean's per-gene gain;
    s = 0.25 is where the two blocks would weigh about equally.  s > 1 pushes
    further in the direction that already costs PC/cell fit.

gmlite@k (plug-in; value required, integer 2 <= k <= G/2; screen k = 64)
    Critic input = [150 released features, scale * (mean_cells(F) @ A)], where
    A [G,k] averages the genes within k fixed expression-ordered groups: genes
    are ranked by the TRAIN gene mean ``ctx.mean`` (space.npz, fitted on the
    fold's train cells; ties by gene index, stable sort) and rank r goes to
    group floor(r * k / G), so groups have G/k genes (+-1); A[g, j] =
    1/|group j| for g in group j.  F is the same straight-through
    critic_features tensor as the round-1 genemean block (real bags: the
    standardised train counts ctx.Z), so the block is linear in F with the
    rate gradient, exactly like genemean but k columns wide.
    scale = sqrt(E||f150||^2 / E||mean_cells(Z) @ A||^2) over the SAME 256
    reference bags as genemean (numpy default_rng(20261109), 8 x 32 bags of
    128 cells, uniform train donors; no training stream, no torch RNG).
    Conflicts with genemean and donorpc (one gene-level block at a time).
    Nothing changes at generation; the generator and update rule are the
    released ones.

    Why: in the 6 round-1 hvg2000 base fits, 34-40% of the squared per-gene
    error of mean per-cell log expression (train targets, standardised units)
    is explained by 16 expression-ordered gene groups and 40-46% by 64 groups
    (gene-permutation null 2-6%); genemean removes exactly that part (its
    residual error has no group structure beyond the null).  The rest of the
    diagnosis (r2/gm_diagnose.py on the 12 hvg2000 genemean fits, seeds 0-3)
    says the full-width block is where the instability comes from: the
    critic's mean gradient on the gene block at fake bags is uncorrelated with
    the per-gene mean error (Pearson -0.04) and with each seed's deviation
    from the other seeds (-0.04), so the 13-logit gap does not come from the
    critic steering per-gene means back.  The generator keeps wandering
    along those gene directions: per-gene means move between updates
    14k-18k and 20k by 0.9-1.0 times the seed-to-seed RMS difference (base:
    0.46-0.65 and shrinking).  gmlite@64 tests
    whether a 64-wide block keeps the structured part of the genemean gain
    while giving the critic 64 instead of 2000 extra inputs, i.e. whether the
    block's width drives the larger critic gap, the doubled seed-to-seed
    variance and the worse PC/cell fit.  It cannot reduce the idiosyncratic
    per-gene error (~55-60% of base's), so it is expected to sit between base
    and genemean on meanlog_rmse_gene.

Not proposed: a block mean-centred on the train grand mean.  The critic's
first layer is affine, so subtracting a constant vector from the block is
absorbed by the first-layer bias (W(x - m) + b = Wx + (b - Wm)); it changes the
parameterisation, not what the critic can see, and the released penalty acts
on input gradients, which centring leaves unchanged.

Tests: r2/test_arm_gm.py.  Diagnosis: r2/gm_diagnose.py.
"""
from __future__ import annotations

import math

import numpy as np
import torch

import arms
from arms import Arm, FeatureBlock, register, canonical_int

NAME = 'gmlite'
RECOMMENDED_GENEMEAN_S = (0.5, 0.25)
RECOMMENDED_GMLITE_K = 64


def expression_groups(train_gene_mean, k):
    """int64 [G] group index from expression rank (train gene mean, stable ties)."""
    m = np.asarray(train_gene_mean, dtype=np.float64)
    G = len(m)
    if not 2 <= k <= G // 2:
        raise ValueError(f'gmlite needs 2 <= k <= G/2 = {G // 2}, got {k}')
    order = np.argsort(m, kind='stable')
    rank = np.empty(G, dtype=np.int64)
    rank[order] = np.arange(G)
    return rank * k // G


def group_matrix(groups, k):
    G = len(groups)
    sizes = np.bincount(groups, minlength=k)
    if (sizes == 0).any():
        raise ValueError('empty gene group')
    A = np.zeros((G, k), dtype=np.float32)
    A[np.arange(G), groups] = (1.0 / sizes[groups]).astype(np.float32)
    return A, sizes


class GroupMeanBlock(FeatureBlock):
    name = 'groupmean'

    def __init__(self, arm):
        super().__init__()
        self.arm = arm
        self.width = int(arm.k)
        self.info = {'what': 'scale * mean_cells(F) @ A, A = expression-ordered gene-group averaging [G,k]',
                     'k': int(arm.k), 'scale': arm.scale}

    def real(self, ctx, idx, ds):
        return (ctx.Z[idx].mean(1) @ self.arm.A) * self.arm.scale

    def fake(self, ctx, cf, rate, counts):
        return (cf.mean(1) @ self.arm.A) * self.arm.scale

    def state(self):
        return {'width': self.width}


@register
class GmLiteArm(Arm):
    name = NAME
    value_policy = 'required'
    conflicts = ('genemean', 'donorpc')
    needs_Z = True

    @classmethod
    def parse_value(cls, text):
        return canonical_int(text, lo=2)

    @torch.no_grad()
    def setup(self, ctx):
        from released_update import bag_features
        self.k = int(self.value)
        groups = expression_groups(ctx.mean.double().cpu().numpy(), self.k)
        A, sizes = group_matrix(groups, self.k)
        self.groups = torch.as_tensor(groups)
        self.A = torch.as_tensor(A, device=ctx.device)
        # reference bags: identical draws to the round-1 genemean scale (arms.Features._genemean_scale)
        rng = np.random.default_rng(arms.GENEMEAN_REF_SEED)
        num = den = 0.0
        for _ in range(arms.GENEMEAN_REF_BAGS // 32):
            ds = rng.integers(len(ctx.by), size=32)
            idx = torch.as_tensor(np.stack([rng.choice(ctx.by[i], 128, replace=True) for i in ds]), device=ctx.device)
            num += float(bag_features(ctx.trainpcs[idx]).double().square().sum())
            den += float((ctx.Z[idx].mean(1) @ self.A).double().square().sum())
        self.scale = float(math.sqrt(num / den))
        self.info = {'k': self.k, 'scale': self.scale, 'group_size_min': int(sizes.min()), 'group_size_max': int(sizes.max()),
                     'reference_bags': arms.GENEMEAN_REF_BAGS, 'reference_seed': arms.GENEMEAN_REF_SEED,
                     'ordering': 'train gene mean (space.npz mean), stable argsort'}

    def feature_blocks(self, ctx):
        return [GroupMeanBlock(self)]

    def state(self):
        return {'k': self.k, 'scale': self.scale, 'groups': self.groups.clone()}
