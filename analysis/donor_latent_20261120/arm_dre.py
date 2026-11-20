"""Donor random-effect generator variant ``g.dre`` (research/donor-latent-20261120).

Native EMPF generator (pg_model.CountGenerator via arms.HookedCountGenerator):
per generated donor a particle latent z (512 learned particles, z_dim 8) and the
stage condition c set a FiLM (scale, shift) of every cell's hidden state.

``g.dre`` adds an explicit continuous donor random effect, drawn once per
generated donor (= per bag) from an isotropic Gaussian prior:

    u_d ~ N(0, I_R),  R = 4                       (drawn after the native cell noise)
    noise_dc  <- noise_dc * exp(A u_d)           donor-level variance effect on the
                                                 cell latent (A: R -> zc)
    shift_d   <- shift_d + B u_d                 donor-level location effect on the
                                                 cell hidden state (B: R -> h)

A and B are bias-free linear maps, initialised to exactly zero, so at update 0 the
forward value equals the native generator for the same native noise draw (the extra
Gaussian draw is the only change to the random stream).  Their gradients are nonzero
(d/dB = u^T dL/dshift, d/dA = sum_c (noise * dL/dnoise) u^T), so the GAN can grow the
random-effect amplitude only if the critic rewards between-donor spread.  The
released parameters are constructed first under the same generator_init seed, so
every native initial tensor is unchanged.  The critic, EMPF penalty, particle prior,
recipe, streams and batch construction are exactly the native ones: the native
critic already scores one donor bag (128 cells) per row, conditioned on stage.
"""
from __future__ import annotations

import torch
from torch import nn

RE_DIM = 4
NAME = 'g.dre'


def dre_class(base):
    class DonorRandomEffectGenerator(base):
        def __init__(self, mean, theta, zd=8, zc=8, h=128):
            super().__init__(mean, theta, zd=zd, zc=zc, h=h)
            self.re_dim = RE_DIM
            self.re_scale = nn.Linear(RE_DIM, zc, bias=False)
            self.re_loc = nn.Linear(RE_DIM, h, bias=False)
            with torch.no_grad():
                self.re_scale.weight.zero_()
                self.re_loc.weight.zero_()

        def parameters_nb(self, z, c, k, library, gen=None):
            scale, shift = self.film(z, c)
            noise = torch.randn(len(z), k, self.zc, device=z.device, generator=gen)
            u = torch.randn(len(z), self.re_dim, device=z.device, generator=gen)
            noise = noise * torch.exp(self.re_scale(u))[:, None]
            shift = shift + self.re_loc(u)
            h = self.cell_hidden(noise, scale, shift, z, c)
            proportions = torch.softmax(self.logits(h, z, c), dim=-1).clamp_min(1e-8)
            proportions = proportions / proportions.sum(-1, keepdim=True)
            mu = library.clamp_min(1)[..., None] * proportions
            theta = self.dispersion(z, c)
            return mu, theta

        def random_effect_amplitude(self):
            """Frobenius norms of the variance (A) and location (B) random-effect maps."""
            return {'scale_A_fro': float(self.re_scale.weight.detach().norm()),
                    'loc_B_fro': float(self.re_loc.weight.detach().norm())}

    return DonorRandomEffectGenerator


def register(arms):
    """Register ``g.dre`` with the frozen r5 plug-in registry (no file in r5/ is changed)."""
    if NAME in arms._ARMS:
        return arms._ARMS[NAME]

    class DonorRandomEffect(arms.Arm):
        name = NAME

        def setup(self, ctx):
            self.info = {'random_effect_dim': RE_DIM, 'prior': 'N(0, I) per generated donor (bag)',
                         'effects': ['exp(A u) multiplies the cell latent noise (variance effect)',
                                     'B u shifts the donor FiLM shift (location effect)'],
                         'init': 'A = B = 0 (native forward at update 0)',
                         'critic': 'unchanged native bag critic'}

        def generator_class(self, ctx, cls):
            return dre_class(cls)

        def generator_grad_params(self, G):
            return [G.re_scale.weight, G.re_loc.weight]

    return arms.register(DonorRandomEffect)
