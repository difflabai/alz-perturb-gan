"""Round-5 arm ``pzero`` / ``pzero@s`` (work item 2, Fix B): per-bag, per-gene zero fraction as a critic block.

Why.  After ``pbulk`` every generator is still behind real-donor resampling on
the per-gene nonzero fraction (``nonzero_rmse_gene`` about 0.021-0.022 against
0.017, RESULTS.md 8e).  Nothing the critic sees pins it: the released 150
features are 50 PCs of standardised log1p CP10k (zeros of low genes are
nearly invisible there), ``genemean`` shows the mean of logs and ``pbulk``
the log of the summed counts.  For a fixed mean count the fraction of zero
cells is set by the cell-to-cell spread (NB dispersion, latent heterogeneity),
so a generator can match the pseudobulk and still get the zeros wrong.

Block.  For a bag of cells with counts x [B, cells, G] (cells = ctx.bag, 128):

    p_g   = mean_cells 1(x_g == 0)                       observed zero fraction
    out_g = s * s_pz * (p_g - m_g) / sd_g                [B, G]

Real bags: x = the bag's real train counts (ctx.X[idx]).

Fake bags, gradient path (straight-through).  The trainer passes the Poisson
counts (no gradient) and the Gamma-NB rate WITH gradient (the rate is a
reparameterised Gamma(theta, theta/mu) draw, ``pg_model.CountGenerator.rates``).
Given the rate, P(count = 0 | rate) = exp(-rate), so the block uses

    p_g = mean_cells[ 1(count_g == 0) + exp(-rate_g) - exp(-rate_g).detach() ]

FORWARD: exactly the observed zero fraction of the sampled counts (the added
term is exactly 0.0 in floating point, so fake(counts) == real(counts)
bitwise for the same counts).  BACKWARD: d out_g / d rate[b,c,g] =
-s s_pz / sd_g * exp(-rate[b,c,g]) / cells (only the same gene, every cell its
own term): the gradient of the expected zero fraction given the rates.  Through
the Gamma reparameterisation this is a pathwise, unbiased single-draw estimate
of the gradient of the closed-form NB zero probability
E_rate[exp(-rate)] = (theta / (theta + mu))^theta with respect to mu and theta
(implicit reparameterisation of torch's Gamma.rsample).

Why exp(-rate) and not the closed form (theta/(theta+mu))^theta.  Both have the
same expectation and the same expected gradient (verified in
test_arm_pzero.py ``unit``: the Monte-Carlo mean of the pathwise gradient over
many Gamma draws matches autograd of the closed form to Monte-Carlo error).
The closed form has lower variance, but (1) the FeatureBlock API (hook d)
receives only ``cf``, ``rate`` and ``counts``, not mu and theta, so using it
would need a new trainer hook and would break the "same inputs as every other
block" contract; (2) the rate path is correct for ANY generator head that
produces a Poisson rate (g.dispz, g.cdisp, latent heterogeneity at the cell
level), while the closed form would have to be re-derived per head; (3) it is
exactly the construction pbulk uses (straight-through through the rate), so
the two blocks get gradients of the same kind and noise level.  The extra
variance is small next to the Poisson count noise the forward value already
carries.  Library-0 cells have rate 0 (exp(0) = 1, count 0): forward value
1, gradient into a constant-zero rate (no parameter path), as for real
library-0 cells (none exist in the caches).

Standardisation (fixed, real TRAIN bags only).  m_g, sd_g: mean and SD of p_g
over R = 512 reference real train bags (uniform train donor, 128 cells with
replacement = the training batch recipe; numpy stream
SeedSequence([20261109, crc32('pzero'), 0]), the arm_pbulk.ref_bag_rows
recipe with this arm's name).  Bag-level statistics: they include the
128-cell binomial noise the critic actually sees.

SD floor.  sd_g >= max(SD_FLOOR_FRAC * median_g sd_g, BINOM_FLOOR) with
SD_FLOOR_FRAC = 0.25 (the pbulk rule) and BINOM_FLOOR = sqrt(q (1 - q) / 128)
at q = 1/128 (= 0.00778), the binomial SD of the zero fraction of a gene that
is seen in one cell per bag on average.  With this floor a one-cell change of a
bag (1/128 = 0.0078 in p_g) moves any gene's standardised value by at most
about 1.0.  Genes that are always zero in the reference bags (sd = 0) or never
zero (sd = 0) are floored and carry no gradient worth noting: never-zero genes
have exp(-rate) ~ 0, always-zero genes are rare in a 2,000-HVG panel.
Measured on hvg2000 folds 0-2 (smoke_r5/pzero/scripts/explore.py): mean zero
fraction per gene 0.23-0.994 (median 0.96-0.97), bag SD 0.0087-0.49 (median
0.024-0.028, minimum 1.1x binomial), so NO gene is floored on folds 0-2; the
floor is a guard for other panels and folds.  The between-donor share is high:
the SD of donor zero fractions is about 0.78 of the bag SD (median gene).

Scale.  s_pz = sqrt(E||f150||^2 / E||(p - m)/sd||^2) on the same reference
bags (the released 150 features recomputed in float64): the block has the
same mean squared norm as the released features on real train bags, the
genemean / pbulk recipe.  So on real bags the block holds 1/2 of the critic
input's mean squared norm alone (``pzero``: 150 + 2000 columns), and 1/3 with
``pbulk`` (``pbulk,pzero``: 150 + 2000 + 2000; same for genemean,pzero).
s = 1 for ``pzero``; ``pzero@s`` multiplies (s > 0, canonical '%g';
``pzero@1`` is refused).

Composable with ``genemean`` / ``genemean@s`` (round-1 block first), ``pbulk``
(columns: 150 released, then pbulk 2000, then pzero 2000 in sorted flag
order), ``donorpc2`` and any plug-in; conflicts with nothing.

Critic penalty.  The BISPF penalty acts on |dD/dx| / sqrt(input width) above
kappa; round 3 found that wide inputs make it stop binding.  Match it with
``t.kappa = sqrt(150 / width)``: ``pzero,t.kappa@0.25`` (2,150 wide: 0.264;
0.25 used as in round 3) and ``pbulk,pzero,t.kappa@0.19`` (4,150 wide:
0.190).  The arm does not set kappa itself (t. flags are core).

Does the block tell the critic something new?  Measured without training
(test_arm_pzero.py ``separability``; round-3 fits pbulk and
g.dispz,pbulk,t.kappa@0.25, seed 0, folds 0-2, states snap_10000,
snap_15000, snap_19000, final; 512 fake training-style bags against the 512
paired real train bags; d_g = per-gene mean offset in real-bag SD; means over
folds):
* Fake bags are offset along pzero about as much as along pbulk: RMS d 0.45-0.49
  (control) and 0.41-0.47 (g.dispz combination) against pbulk 0.45-0.56 and
  0.33-0.41; real-vs-real null 0.06.  ||mean offset||^2 / V_real: pzero
  0.21-0.25 / 0.17-0.22, pbulk 0.21-0.33 / 0.11-0.19, null 0.004.
* Most of it mirrors the pseudobulk offset (too little mass -> too many
  zeros): corr(d_pz, d_pb) -0.89 to -0.90 (control), -0.79 to -0.80 (g.dispz).
* The part NOT explained by the pbulk offset (residual of a per-quartile
  regression of d_pz on d_pb): RMS 0.20-0.22 (control) and 0.25-0.27
  (g.dispz), 7-10x the null (0.027), rising with expression: quartile 1
  0.13-0.16 / 0.19-0.21, quartile 4 0.27-0.29 / 0.33-0.35.  In the top
  quartile generated bags have too many zeros (mean d_pz +0.01 to +0.11; zero
  fraction +0.007 to +0.012 absolute) at a matched pseudobulk: over-dispersion
  of highly expressed genes.  Absolute RMS zero-fraction error 0.020-0.027
  (null 0.003), the scale of nonzero_rmse_gene.
So about 19% (control) to 34% (g.dispz) of the pzero offset energy is
information the pbulk block does not carry.  This is a train-bag
measurement: it says the critic could see a gap, not that training will close
it or that held-out scores improve.

Recommended arms: ``pzero,t.kappa@0.25`` (alone) and
``pbulk,pzero,t.kappa@0.19`` (with pbulk).  Variant
``g.dispz,pbulk,pzero,t.kappa@0.19``: the zero-fraction residual is largest
with donor-dependent dispersion, i.e. g.dispz moves exactly what pzero shows
and nothing in the round-3 critic constrains it; its comparator
g.dispz,pbulk,t.kappa@0.25 exists in fits_r3 (seeds 0-3).

The block is used only by the critic during training; generation is the
default procedure (no mode).

Device independence.  setup() computes m, sd, s_pz in float64 numpy on CPU
from the integer counts and space.npz; m and sd are stored as float32 and s_pz
rounded to six significant digits (pbulk's rule), so a GPU fit and a CPU
regeneration recompute an identical arm state.

State (hook f): mult, s_pz, m [G], sd [G] (after flooring).
"""
from __future__ import annotations

import math

import numpy as np
import torch

from arms import Arm, FeatureBlock, register, canonical_float
from arm_pbulk import ref_bag_rows, r6, REF_BAGS, BAG

NAME = 'pzero'
SD_FLOOR_FRAC = 0.25
BINOM_FLOOR = math.sqrt((1.0 / BAG) * (1.0 - 1.0 / BAG) / BAG)


def zero_frac64(x):
    """numpy [B, cells, G] counts -> [B, G] float64 zero fraction."""
    return (x == 0).mean(1)


def zero_frac(counts, rate=None):
    """torch [B, cells, G] -> [B, G] zero fraction; with ``rate`` the straight-through exp(-rate) gradient."""
    ind = (counts == 0).to(counts.dtype if rate is None else rate.dtype)
    if rate is None:
        return ind.mean(1)
    e = torch.exp(-rate)
    return (ind + (e - e.detach())).mean(1)


class PzeroBlock(FeatureBlock):
    name = 'pz'

    def __init__(self, arm, n_genes):
        super().__init__()
        self.arm = arm
        self.width = int(n_genes)
        self.info = dict(arm.block_info)

    def _f(self, p):
        return (p - self.arm.m) * self.arm.w

    def real(self, ctx, idx, ds):
        return self._f(zero_frac(ctx.X[idx]))

    def fake(self, ctx, cf, rate, counts):
        return self._f(zero_frac(counts, rate))

    def state(self):
        return {'width': self.width}


@register
class PzeroArm(Arm):
    name = NAME
    value_policy = 'optional'
    needs_Z = False

    @classmethod
    def parse_value(cls, text):
        v, canon = canonical_float(text, lo=0.0, lo_open=True)
        if v == 1.0:
            raise ValueError('pzero@1 is pzero; use the name without a value')
        return v, canon

    def setup(self, ctx):
        self.mult = 1.0 if self.value is None else float(self.value)
        W = ctx.space['W'].astype(np.float64) / ctx.space['pc_sd'].astype(np.float64)
        mean = ctx.space['mean'].astype(np.float64)
        sd_g = ctx.space['sd'].astype(np.float64)
        from released_update import bag_features
        ys, n150 = [], 0.0
        for ds, rows in ref_bag_rows(ctx.by, NAME):
            x = ctx.X[torch.as_tensor(rows, device=ctx.device)].double().cpu().numpy()
            ys.append(zero_frac64(x))
            lib = np.maximum(x.sum(-1, keepdims=True), 1.0)
            z = (np.log1p(x * (1e4 / lib)) - mean) / sd_g
            n150 += float((bag_features(torch.from_numpy(z @ W)).numpy() ** 2).sum())
        y = np.concatenate(ys)
        R = len(y)
        m = y.mean(0)
        sd = y.std(0)
        floor = max(SD_FLOOR_FRAC * float(np.median(sd)), BINOM_FLOOR)
        n_floored = int((sd < floor).sum())
        sd = np.maximum(sd, floor)
        m32, sd32 = m.astype(np.float32), sd.astype(np.float32)
        zz = (y - m32.astype(np.float64)) / sd32.astype(np.float64)
        s_pz = r6(math.sqrt((n150 / R) / float((zz ** 2).sum(1).mean())))
        self.s_pz = s_pz
        self.scale = s_pz * self.mult
        dev = ctx.device
        self.m = torch.as_tensor(m32, device=dev)
        self.sd = torch.as_tensor(sd32, device=dev)
        self.w = torch.as_tensor((self.scale / sd32.astype(np.float64)).astype(np.float32), device=dev)
        donor_p = np.stack([zero_frac64(ctx.X[torch.as_tensor(r, device=ctx.device)].double().cpu().numpy()[None])[0] for r in ctx.by])
        ratio = donor_p.std(0) / sd
        self.block_info = {
            'formula': 'out = s * s_pz * (mean_cells 1(x==0) - m)/sd; fake: mean_cells[1(count==0) + exp(-rate) - exp(-rate).detach()]',
            'reference_bags': R, 'reference_seed': f'SeedSequence([20261109, crc32({NAME!r}), 0])',
            's_pz': s_pz, 'multiplier': self.mult, 'scale_effective': r6(self.scale),
            'E_norm2_150_reference': r6(n150 / R),
            'sd_floor': r6(floor), 'sd_floor_rule': f'max({SD_FLOOR_FRAC} x median gene bag SD, binomial SD at q=1/{BAG})',
            'genes_floored': n_floored,
            'median_bag_sd': r6(float(np.median(sd))), 'median_zero_fraction': r6(float(np.median(m))),
            'median_donor_p0_sd_over_bag_sd': r6(float(np.median(ratio))),
        }
        self.info = {'multiplier': self.mult, 's_pz': s_pz, 'scale_effective': r6(self.scale), 'genes_floored': n_floored}

    def feature_blocks(self, ctx):
        return [PzeroBlock(self, ctx.n_genes)]

    def state(self):
        return {'mult': self.mult, 's_pz': self.s_pz, 'm': self.m.cpu(), 'sd': self.sd.cpu()}
