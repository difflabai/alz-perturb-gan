"""Matched 50-direction random critic observation; evaluation PCA stays fixed.

Directions are independent of expression. Normalization and norm matching use
TRAIN cells/bags only. The pbulk block, feature count, critic and update rule
are unchanged. No default arm takes this path.
"""
import numpy as np
import torch
from arms import Arm, register

SEED = 20261113


def fixed_directions(n_genes, n_directions=50):
    rng = np.random.default_rng(SEED)
    q, r = np.linalg.qr(rng.normal(size=(n_genes, n_directions)))
    return q * np.where(np.diag(r) < 0, -1., 1.)


@register
class RandomProjection(Arm):
    name = 'd.randproj'
    needs_Z = True

    @torch.no_grad()
    def setup(self, ctx):
        from released_update import bag_features
        from arm_pbulk import ref_bag_rows
        directions = torch.as_tensor(fixed_directions(ctx.n_genes), dtype=torch.float32, device=ctx.device)
        scores = ctx.Z @ directions
        scale = scores.std(0, unbiased=False).clamp_min(1e-6)
        projection = directions / scale
        scores = ctx.Z @ projection
        num = den = 0.
        for _, rows in ref_bag_rows(ctx.by, 'pbulk'):
            idx = torch.as_tensor(rows, device=ctx.device)
            num += float(bag_features(ctx.trainpcs[idx]).double().square().sum())
            den += float(bag_features(scores[idx]).double().square().sum())
        norm_match = float(np.sqrt(num / den))
        # Match the full extracted feature norm, including Fourier features.
        ctx.critic_W = projection
        ctx.critic_trainpcs = scores
        # Features applies this factor after mean/SD/Fourier extraction.
        ctx.critic_feature_scale = norm_match
        self.projection = projection.detach().cpu()
        self.info = {'directions': 50, 'seed': SEED, 'normalization': 'train cells only',
                     'feature_norm_multiplier': norm_match, 'pbulk_scale': 'unchanged PCA reference',
                     'evaluation_projection': 'unchanged PCA'}

    def state(self):
        return {'projection': self.projection, **self.info}
