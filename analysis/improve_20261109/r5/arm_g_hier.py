"""Explicit donor expression, cell states, and within-state variation.

g.split separates donor logits from centered cell-logit residuals (initial
amplitude .5). g.hier adds four shared states and donor-specific mixture
weights. Both keep the existing NB dispersion head and ParticleGAN updates.
The centered residual is a bag decoder, like the existing 128-cell training
contract; it is not an iid single-cell generative claim.
"""
import torch
from torch import nn
import torch.nn.functional as F
from arms import Arm, register


def split_class(base, states=0):
    class SplitGenerator(base):
        def __init__(self, mean, theta, zd=8, zc=8, h=128):
            super().__init__(mean, theta, zd=zd, zc=zc, h=h)
            self.within_logit = nn.Parameter(torch.tensor(0.))
            self.n_states = states
            if states:
                private = torch.Generator().manual_seed(20261113)
                self.state_embedding = nn.Parameter(torch.randn(states, h, generator=private) * .1)
                self.mix_weight = nn.Parameter(torch.zeros(states, zd + 7))
                self.mix_bias = nn.Parameter(torch.zeros(states))

        def parameters_nb(self, z, c, k, library, gen=None):
            scale, shift = self.film(z, c)
            noise = torch.randn(len(z), k, self.zc, device=z.device, generator=gen)
            donor_logits = self.out(shift)[:, None]
            anchor = shift[:, None].expand(-1, k, -1)
            state_effect = 0.
            if self.n_states:
                mix_logits = F.linear(torch.cat([z, c], 1), self.mix_weight, self.mix_bias)
                uniform = torch.rand(len(z), k, self.n_states, device=z.device, generator=gen).clamp(1e-6, 1-1e-6)
                soft = torch.softmax(mix_logits[:, None] - torch.log(-torch.log(uniform)), -1)
                hard = F.one_hot(soft.argmax(-1), self.n_states).to(soft.dtype)
                assignment = hard + (soft - soft.detach())
                anchor = anchor + assignment @ self.state_embedding
                profiles = self.out(self.state_embedding)
                profiles = profiles - profiles.mean(0, keepdim=True)
                state_effect = assignment @ profiles
            state_logits = self.out(anchor)
            within = self.out(anchor + self.cell(noise) * (1 + scale[:, None])) - state_logits
            weights = library.clamp_min(1)[..., None]
            within = within - (within * weights).sum(1, keepdim=True) / weights.sum(1, keepdim=True)
            logits = donor_logits + state_effect + torch.sigmoid(self.within_logit) * within
            proportions = torch.softmax(logits, -1).clamp_min(1e-8)
            proportions = proportions / proportions.sum(-1, keepdim=True)
            return library.clamp_min(1)[..., None] * proportions, self.dispersion(z, c)

    return SplitGenerator


class _Split(Arm):
    states = 0

    def setup(self, ctx):
        other = {a.name for a in ctx.plugins} & {'g.split', 'g.hier'}
        if len(other) != 1:
            raise ValueError('g.split and g.hier are separate, exclusive ablations')
        self.info = {'states': self.states, 'within_initial_amplitude': .5,
                     'donor_path': 'separate logits', 'count_dispersion': 'unchanged shared NB',
                     'cell_residual': 'library-weighted centered logits'}

    def generator_class(self, ctx, cls):
        return split_class(cls, self.states)

    def generator_grad_params(self, G):
        params = [G.within_logit]
        if self.states:
            params += [G.state_embedding, G.mix_weight, G.mix_bias]
        return params


@register
class Split(_Split):
    name = 'g.split'


@register
class Hierarchical(_Split):
    name = 'g.hier'
    states = 4
