"""Round-3 critic-architecture arms (work item R4): ``d.tower``, ``d.sum``, ``d.h@N``, ``d.depth@L``.

Every arm here only replaces the critic MODULE handed to the released update
(hook d2, ``critic_module``).  The critic input x (the released 150 bag
features followed by the round-1 / plug-in blocks), the ParticleGAN 0.9.7
recipe, the optimizers (DualNorm, critic learning rate), the BISPF critic
penalty (``input_domain='continuous'``, evaluated on dD/dx of the whole x),
the non-saturating loss and the policy calls are unchanged.  The new module
is built in ``trainer.build`` under ``torch.manual_seed(critic_init)`` (the
released critic stream) and all its trainable parameters go to the released
critic optimizer.  No arm here draws from any other random stream, adds a
loss term or changes what the generator sees.  With no ``d.`` flag nothing
in this file runs (importing it registers classes only): training is bitwise
the round-1/2/3 trajectory (test_arm_dcritic.py ``neutral``).

Flags (each one overrides ``critic_module``; the trainer allows one such arm
per arm string, so these four are mutually exclusive):

``d.tower`` / ``d.tower@h``  (h = hidden width, 16..1024, default 128 = released)
    Block-wise ("tower") critic.  For every column block b of the layout
    (``released`` 0-149, then the round-1 genemean / donorpc block, then every
    plug-in block, e.g. pbulk, gmlite, donorpc2):

        x'_b = (x_b - m_b) / s_b                              fixed, per block
        t_b  = SiLU(Linear_b([x'_b, c]))                      [B, h]   (own first layer per block)
        f    = SiLU(Linear_mix([t_1, ..., t_n]))              [B, h]
        D    = Linear(SiLU(Linear([f, c])))                   [B]      (released rho)

    m_b: per-column mean of x_b over R = 256 reference REAL TRAIN bags; s_b:
    ONE scalar per block, sqrt(mean over the block's columns of the across-bag
    variance), so every block arrives with mean per-column variance 1 (a scalar,
    not a whitening: the within-block geometry is kept; whitening was the
    round-1 donorpc failure).  Reference bags: 8 x 32 bags of ctx.bag cells,
    uniform train donor, cells with replacement (the training batch recipe),
    numpy stream SeedSequence([20261109, crc32('d.tower'), fold]) (no training
    stream, no torch RNG), features from ``ctx.features.real`` exactly as in
    training.  Train donors only.  The statistics are part of the arm state
    (checked exactly at resume and generation) and are also stored in the
    critic as non-trainable buffers.
    With one block (``base,d.tower``) the module is the released VectorCritic
    structure (phi = Linear(150+7,h), SiLU, Linear(h,h), SiLU; rho) on the
    normalised input: a pure "normalisation only" control.  With k blocks the
    first layer is block-diagonal (k*(w_b+7)*h weights, about the same count as
    the released wide first layer) and Linear_mix is [h, k*h].

``d.sum`` / ``d.sum@h``
    Additive per-block critics: D(x, c) = sum_b D_b(x'_b, c), each D_b the
    released VectorCritic structure at width h on its own normalised block
    (same m_b, s_b as d.tower).  No parameter is shared between blocks, so no
    block can take another block's weights (the strongest separation; it also
    removes every cross-block interaction, e.g. the consistency of the
    genemean block with the per-PC bag means inside the released block).

``d.h@N``  (N in 16..1024, N != 128)
    The released critic (VectorCritic / WideVectorCritic layers on the raw x)
    with hidden width N in all three hidden layers.  Bitwise the released
    WideVectorCritic construction for N = 128 (refused: omit the flag).

``d.depth@L``  (L in 1..6, L != 2)
    The released critic with L hidden layers in phi (released: 2 =
    Linear(nf+7,h), Linear(h,h)); rho unchanged.  ``DeepVectorCritic`` with
    L = 2, h = 128 has exactly the released parameter list and initial tensors
    (tested), so only the depth changes.

Per-cell blocks (``cell_dim``) are refused by d.tower and d.sum (they need a
set encoder; see the cell-critic task) and are passed through unchanged by
d.h / d.depth (released-style critics flatten them, as WideVectorCritic does).

Why, and what the data say (test_arm_dcritic.py ``starve`` / ``penalty`` /
``short``; trained round-2 seed-0 critics, hvg2000, folds 0-2):
* Starvation is real in every gene-wide arm: released share of the critic's
  first-layer pre-activation variance on real bags is 0.96 (base), 0.06
  (genemean), 0.15 (pbulk), 0.05 (genemean,pbulk), 0.37 (genemean@0.5), 0.84
  (genemean@0.25), 0.33 (gmlite@64).  DualNorm's polar step on the shared
  first layer still gives the released columns 25-29% of the step (4x their
  7% column share), so the update itself is not what starves them.
* But it does not explain the worse PC/cell fit: genemean@0.25 restores the
  released share to 0.84 and has the WORST pc_energy_rep (2.61 vs 2.43
  genemean, 2.38 genemean@0.5, 2.19 base).  Within the gene-wide arms the
  released share is uncorrelated with pc_energy / cell_energy (Spearman
  about 0 without genemean,pbulk).
* What every gene-wide arm shares is a critic that the BISPF penalty no
  longer restrains: late RMS input gradient 0.21-0.60 against kappa 1 (base
  0.85-0.95, 44% of fake bags above kappa; gene-wide 0-7%), cap+floor loss
  10-50x smaller, critic gap 11-20 logits (base 4.4).  genemean,pbulk
  (x 4,150 wide): RMS 0.18-0.21, D_adversarial 0.05, gap 18-20: a nearly
  perfect classifier, and the worst arm.  The critic's mean push on gene-wide
  blocks is uncorrelated with the per-gene error (Pearson -0.13..0.005).
* So the block-wise critics here are tests of the architecture hypothesis,
  not a fix of the penalty regime.  Expect them to make the critic STRONGER:
  splitting one DualNorm matrix into k gives k full-size normalised steps,
  and the normalised inputs raise the initial gradient.  In 300-update GPU
  fits (fold 0) d.tower / d.sum reached gap 11.5-13 by update 300 vs 9.6 for
  genemean, with a larger released share of the gap (``short``).  The
  penalty-side knob that matches base's absolute Lipschitz cap for a 2,150-wide
  x is kappa = sqrt(150/2150) = 0.26 (``t.kappa``, core/integration).

How the BISPF penalty interacts with this (verified in test_arm_dcritic.py
``penalty``):
* The penalty is computed by ParticleGAN on dD/dx of the critic INPUT x (the
  feature vector); its gates (chords, continuous-flow, cooler query, RMS
  units) depend only on x's width and the batch sizes, which these arms do not
  change.  It caps the RMS gradient ||dD/dx|| / sqrt(dim x) at kappa (soft,
  softplus beta 30) and adds a small floor term 0.005 * ||dD/dx||^2 / dim.
* The fixed per-block normalisation is a re-parametrisation INSIDE the
  module: dD/dx_b = (1/s_b) dD/dx'_b.  The set of critic functions the
  penalty allows, as functions of x, is therefore exactly the same as for the
  released critic; a released-style critic whose first-layer columns are
  multiplied by 1/s_b gives the identical function, gradient and penalty
  (tested to float64 round-off with the native penalty object).  The
  normalisation changes initial scale and optimisation, NOT the penalty
  budget.  What changes the budget a block gets is its scale IN x (e.g.
  genemean@s, pbulk@s: scaling a block by s makes the same discrimination
  cost 1/s more gradient), and the width of x (the RMS cap scales with
  sqrt(dim)), neither of which a d. arm touches.

Repair note (round-3 adversarial review, finding E10; reproduced, numbers unchanged):
* The "starvation" is explained almost exactly by WIDTH SCALING of the shared
  first-layer step: DualNorm moves a matrix by a step whose spectral norm is
  fixed, so the RMS weight per released column shrinks as 1/sqrt(input width).
  Predicted from base's 2.74 at width 157 (150 + 7 condition columns):
  2.74 sqrt(157/2157) = 0.739 (measured genemean 0.74, pbulk 0.82) and
  2.74 sqrt(157/4157) = 0.532 (measured genemean,pbulk 0.50).  It is a
  consequence of the width, not a cause of the worse fit.
* "The penalty no longer restrains" holds for the CAP term only (ebi_loss
  base 0.0157 vs 0.0003-0.0018 for gene-wide arms).  The TOTAL penalty is
  LARGER for the gene-wide arms (pen 0.0209 base vs 0.025-0.031) because of
  the gauge term (0.0052 vs 0.024-0.030).  t.reg@4 (reg_coeff x 4) therefore
  mostly scales an already active gauge/logit term and is a weak test of
  "critic too strong"; t.kappa@0.25 is the critic arm with a mechanism.
  Expectation, stated before any round-3 score: d.tower and t.reg@4 near null.
"""
from __future__ import annotations

import math
import zlib

import numpy as np
import torch
from torch import nn

import arms
from arms import Arm, register, canonical_int

REF_BAGS = 256
REF_CHUNK = 32
REF_ROOT = 20261109
RELEASED_HIDDEN = 128
RELEASED_DEPTH = 2


def _width(text):
    v, t = canonical_int(text, lo=16, hi=1024)
    return v, t


# =========================================================================== modules
class DeepVectorCritic(nn.Module):
    """Released VectorCritic / WideVectorCritic layers with hidden width h and ``depth`` hidden layers in phi.

    depth = 2, h = 128: same modules, parameter names, order and initial draws as arms.WideVectorCritic.
    """

    def __init__(self, n_features, h=RELEASED_HIDDEN, depth=RELEASED_DEPTH, cond_dim=7):
        super().__init__()
        self.n_features, self.cond_dim = int(n_features), int(cond_dim)
        layers = [nn.Linear(n_features + cond_dim, h), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(h, h), nn.SiLU()]
        self.phi = nn.Sequential(*layers)
        self.rho = nn.Sequential(nn.Linear(h + cond_dim, h), nn.SiLU(), nn.Linear(h, 1))

    def forward(self, x, c):
        if x.ndim != 2 or x.shape[1] != self.n_features or c.shape != (len(x), self.cond_dim):
            raise ValueError(f'Expected donor vectors [B,{self.n_features}] and conditions [B,{self.cond_dim}]')
        f = self.phi(torch.cat([x, c], 1))
        return self.rho(torch.cat([f, c], 1)).squeeze(1)


class _BlockNorm(nn.Module):
    """Fixed per-block affine normalisation x'_b = (x_b - m_b) / s_b (buffers, no parameters)."""

    def __init__(self, blocks, center, scale):
        super().__init__()
        self.spans = [(int(b.lo), int(b.hi)) for b in blocks]
        for i, ((lo, hi), m, s) in enumerate(zip(self.spans, center, scale)):
            m = torch.as_tensor(m, dtype=torch.float32)
            if m.shape != (hi - lo,) or not (float(s) > 0 and math.isfinite(float(s))):
                raise ValueError('block normalisation needs a centre per column and a positive finite scale')
            self.register_buffer(f'center_{i}', m.clone())
            self.register_buffer(f'inv_scale_{i}', torch.tensor(1.0 / float(s), dtype=torch.float32))

    def forward(self, x):
        return [(x[:, lo:hi] - getattr(self, f'center_{i}')) * getattr(self, f'inv_scale_{i}') for i, (lo, hi) in enumerate(self.spans)]


class DTowerCritic(nn.Module):
    """Block-wise critic: own first layer per (normalised) block, concatenated, shared layer, released rho."""

    def __init__(self, layout, center, scale, h=RELEASED_HIDDEN):
        super().__init__()
        _no_cell_blocks(layout, 'd.tower')
        self.n_features, self.cond_dim = int(layout.n_features), int(layout.cond_dim)
        self.block_names = [f'{b.flag}/{b.name}' if b.flag else b.name for b in layout.blocks]
        self.norm = _BlockNorm(layout.blocks, center, scale)
        self.towers = nn.ModuleList([nn.Sequential(nn.Linear(b.width + self.cond_dim, h), nn.SiLU()) for b in layout.blocks])
        self.mix = nn.Sequential(nn.Linear(h * len(layout.blocks), h), nn.SiLU())
        self.rho = nn.Sequential(nn.Linear(h + self.cond_dim, h), nn.SiLU(), nn.Linear(h, 1))

    def forward(self, x, c):
        if x.ndim != 2 or x.shape[1] != self.n_features or c.shape != (len(x), self.cond_dim):
            raise ValueError(f'Expected [B,{self.n_features}] and [B,{self.cond_dim}]')
        t = [tw(torch.cat([xb, c], 1)) for tw, xb in zip(self.towers, self.norm(x))]
        f = self.mix(torch.cat(t, 1))
        return self.rho(torch.cat([f, c], 1)).squeeze(1)


class DSumCritic(nn.Module):
    """Additive per-block critics: D = sum_b VectorCritic_b(x'_b, c)."""

    def __init__(self, layout, center, scale, h=RELEASED_HIDDEN):
        super().__init__()
        _no_cell_blocks(layout, 'd.sum')
        self.n_features, self.cond_dim = int(layout.n_features), int(layout.cond_dim)
        self.block_names = [f'{b.flag}/{b.name}' if b.flag else b.name for b in layout.blocks]
        self.norm = _BlockNorm(layout.blocks, center, scale)
        self.critics = nn.ModuleList([DeepVectorCritic(b.width, h=h, depth=RELEASED_DEPTH, cond_dim=self.cond_dim) for b in layout.blocks])

    def block_logits(self, x, c):
        return torch.stack([D(xb, c) for D, xb in zip(self.critics, self.norm(x))], 1)

    def forward(self, x, c):
        if x.ndim != 2 or x.shape[1] != self.n_features or c.shape != (len(x), self.cond_dim):
            raise ValueError(f'Expected [B,{self.n_features}] and [B,{self.cond_dim}]')
        return self.block_logits(x, c).sum(1)


def _no_cell_blocks(layout, flag):
    cells = [b.name for b in layout.blocks if b.cell_dim is not None]
    if cells:
        raise ValueError(f'{flag} works on bag-level blocks only; per-cell blocks {cells} need a set encoder (cell-critic arms)')


# =========================================================================== reference statistics
@torch.no_grad()
def block_reference_stats(ctx, layout, name, n_bags=REF_BAGS):
    """Per-block column means and scalar scales over n_bags real TRAIN reference bags (float64; see docstring).

    Uses ctx.features.real (the training observation), numpy SeedSequence([20261109, crc32(name), fold]);
    no torch RNG, no training stream.
    """
    feats = ctx.features
    if feats.n_features != layout.n_features:
        raise ValueError('layout does not match ctx.features')
    rng = np.random.default_rng(np.random.SeedSequence([REF_ROOT, zlib.crc32(name.encode()), int(ctx.fold)]))
    xs = []
    for st in range(0, n_bags, REF_CHUNK):
        n = min(REF_CHUNK, n_bags - st)
        ds = rng.integers(len(ctx.by), size=n)
        idx = torch.as_tensor(np.stack([rng.choice(ctx.by[i], ctx.bag, replace=True) for i in ds]), device=ctx.device)
        xs.append(feats.real(idx, ds).double().cpu())
    X = torch.cat(xs)
    center, scale, info = [], [], []
    for b in layout.blocks:
        xb = X[:, b.lo:b.hi]
        m = xb.mean(0)
        v = float(xb.var(0, unbiased=True).mean())
        if not (v > 0 and math.isfinite(v)):
            raise ValueError(f'block {b.name}: degenerate reference variance {v}')
        s = math.sqrt(v)
        center.append(m.float())
        scale.append(s)
        info.append({'block': f'{b.flag}/{b.name}' if b.flag else b.name, 'columns': [b.lo, b.hi], 'scale': s,
                     'mean_sq_norm': float(xb.square().sum(1).mean()), 'total_variance': float(xb.var(0, unbiased=True).sum())})
    return center, scale, info


class _NormalisedCriticArm(Arm):
    """Shared code of d.tower / d.sum: reference statistics computed once ctx.features exists, checkpointed exactly.

    The statistics are computed lazily (ctx.features is built after the setup hook) the first time
    ``info``, ``state()`` or ``critic_module`` needs them; in a CLI fit that is make_config, before
    trainer.build.  The computation draws no torch random numbers (tested), so the critic_init
    stream is untouched even when it first happens inside critic_module.
    """
    value_policy = 'optional'
    module_cls = None

    @classmethod
    def parse_value(cls, text):
        v, t = _width(text)
        if v == RELEASED_HIDDEN:
            raise ValueError(f'{cls.name}@128 is the default width; write {cls.name}')
        return v, t

    def __init__(self, value, flag):
        self._ctx = None
        self._stats = None
        super().__init__(value, flag)

    @property
    def hidden(self):
        return RELEASED_HIDDEN if self.value is None else int(self.value)

    @property
    def info(self):
        if self._stats is None and self._ctx is not None and getattr(self._ctx, 'features', None) is not None:
            self.stats()
        return self._info

    @info.setter
    def info(self, value):
        self._info = value

    def setup(self, ctx):
        self._ctx = ctx
        self._stats = None
        self._info = {'hidden': self.hidden, 'module': self.module_cls.__name__, 'reference_bags': REF_BAGS,
                      'reference_seed': [REF_ROOT, f'crc32({self.name})', int(ctx.fold)], 'cells_per_bag': int(ctx.bag)}

    def stats(self):
        if self._stats is None:
            ctx = self._ctx
            if ctx is None or getattr(ctx, 'features', None) is None:
                raise RuntimeError(f'{self.flag}: reference statistics need ctx.features (available after Context.__init__)')
            self._stats = block_reference_stats(ctx, ctx.features.layout(), self.name)
            self._info['blocks'] = self._stats[2]
        return self._stats

    def critic_module(self, ctx, layout):
        center, scale, _ = self.stats()
        return self.module_cls(layout, center, scale, h=self.hidden)

    def state(self):
        center, scale, _ = self.stats()
        return {'hidden': self.hidden, 'center': [m.cpu() for m in center], 'scale': [float(s) for s in scale]}


@register
class DTowerArm(_NormalisedCriticArm):
    """d.tower[@h]: block-wise critic (docstring)."""
    name = 'd.tower'
    module_cls = DTowerCritic


@register
class DSumArm(_NormalisedCriticArm):
    """d.sum[@h]: additive per-block critics (docstring)."""
    name = 'd.sum'
    module_cls = DSumCritic


@register
class DWidthArm(Arm):
    """d.h@N: released critic layers on the raw x with hidden width N."""
    name = 'd.h'
    value_policy = 'required'

    @classmethod
    def parse_value(cls, text):
        v, t = _width(text)
        if v == RELEASED_HIDDEN:
            raise ValueError('d.h@128 is the released critic; omit the flag')
        return v, t

    def setup(self, ctx):
        self.info = {'hidden': int(self.value), 'depth': RELEASED_DEPTH, 'module': 'DeepVectorCritic'}

    def critic_module(self, ctx, layout):
        return DeepVectorCritic(layout.n_features, h=int(self.value), depth=RELEASED_DEPTH, cond_dim=layout.cond_dim)

    def state(self):
        return {'hidden': int(self.value)}


@register
class DDepthArm(Arm):
    """d.depth@L: released critic with L hidden layers in phi."""
    name = 'd.depth'
    value_policy = 'required'

    @classmethod
    def parse_value(cls, text):
        v, t = canonical_int(text, lo=1, hi=6)
        if v == RELEASED_DEPTH:
            raise ValueError('d.depth@2 is the released critic; omit the flag')
        return v, t

    def setup(self, ctx):
        self.info = {'hidden': RELEASED_HIDDEN, 'depth': int(self.value), 'module': 'DeepVectorCritic'}

    def critic_module(self, ctx, layout):
        return DeepVectorCritic(layout.n_features, h=RELEASED_HIDDEN, depth=int(self.value), cond_dim=layout.cond_dim)

    def state(self):
        return {'depth': int(self.value)}
