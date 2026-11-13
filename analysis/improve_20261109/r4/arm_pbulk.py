"""Round-2 arm ``pbulk`` / ``pbulk@s`` (failure F5): per-bag pseudobulk critic block, linear in counts.

Why.  After ``genemean`` the generators still have a per-gene level error in
pseudobulk log1p(CP10k) (the evaluator's ``pb_rmse`` quantity) that is not the
same as the error in mean per-cell log expression.  ``genemean`` shows the
critic mean_cells(standardised log1p CP10k of each cell): a mean of LOGS.  A
generator can match it while the mean of COUNTS (the pseudobulk) is off, e.g.
by getting the per-gene dispersion / zero fraction wrong: for a fixed mean of
log1p the mean of counts grows with the cell-to-cell spread.  Nothing in the
released 150 features or in genemean pins the bag's summed counts per gene.

Evidence on real data (test_arm_critic.py ``separability``; folds 0-2,
round-1 base and genemean seeds 0/1, stored generated.npz vs 32 real
256-cell bags per target donor; d = target-paired per-gene mean difference
in units of the real bag SD; null = two independent real draws):
* genemean cuts the mean-log error (train targets, all genes: d_rms_ml
  0.92-1.14 base -> 0.37-0.84 genemean) more than the pseudobulk error
  (d_rms_pb 0.86-1.00 -> 0.56-0.91); for genemean d_rms_pb > d_rms_ml in
  every fold and seed on train targets, most in the top expression
  quartile (e.g. fold 1: 0.49-0.50 vs 0.31-0.34).
* The part of the pseudobulk error not explained by the mean-log error
  (per-gene residual of d_pb on d_ml) is 0.25-0.63 per quartile (train and held-out
  targets) for both base and genemean, about 10x the real-vs-real null
  (0.02-0.05): genemean does
  not remove it.  Correlation of d_pb with d_ml falls to 0.64-0.88 in the top
  quartile for genemean (0.91-0.94 for base).
* Bags are separable along it: donor-grouped cross-validated logistic AUC on
  the per-gene residual of pb after a pb ~ ml map fitted on independent real
  bags is 0.97-1.00 in every quartile, fold, seed and split (null AUC for pb
  0.47-0.52).  (AUCs on ml or pb alone are also 0.93-1.00: generated bags
  are separable along many genes, so AUC is not a fine measure here.)
On fake 128-cell bags of the frozen fold-0 fits (``shares``) the mean
offset of the pbulk block, ||mean_fake - mean_real||^2 / V_real, is 0.76-0.79
(base), 0.39-0.49 (genemean), 1.57-1.94 (donorpc): the critic would see a
real gap there, also for genemean.

Block.  For a bag of cells with counts x [cells, G] (cells = 128):

    pb_g  = sum_cells x_g                                  (linear in counts)
    y_g   = log1p(1e4 * pb_g / max(sum_g' pb_g', 1))       bag pseudobulk log1p CP10k
    out_g = s * s_pb * (y_g - m_g) / sd_g                  [B, G]

m_g, sd_g: mean and SD of y_g over R = 512 reference REAL TRAIN bags (uniform
train donor, 128 cells with replacement = the training batch recipe; numpy
stream SeedSequence([20261109, crc32('pbulk'), 0])).  Bag-level, not
donor-pseudobulk, statistics: they include the 128-cell sampling noise the
critic actually sees, so low-count genes (whose bag pseudobulk is noisy) are
not inflated.  sd_g is floored at SD_FLOOR_FRAC * median_g(sd_g) (genes that
are almost never seen in a reference bag).  s_pb = sqrt(E||f150||^2 /
E||(y - m)/sd||^2) on the same reference bags (the released 150 features
recomputed in float64), i.e. the genemean recipe for s_gm: the block has the
same mean squared norm as the released features on real train bags.  s = 1
for ``pbulk``; ``pbulk@s`` multiplies (s > 0, canonical '%g'; ``pbulk@1`` is
refused).  Composable with ``genemean`` / ``genemean@s`` ("genemean,pbulk":
columns 0-149 released, 150-2149 genemean, 2150-4149 pbulk on hvg2000), with
``donorpc2`` and with any plug-in; conflicts with nothing.

Real bags: x = the bag's real train counts (ctx.X[idx]).

Fake bags, gradient path.  The trainer passes the Poisson counts (no
gradient) and the Gamma-NB rate WITH gradient.  The block uses

    x = counts + (rate - rate.detach())

so the FORWARD value is exactly the function above of the Poisson counts
(bitwise: rate - rate.detach() is exactly 0), the same function as for real
bags, and the BACKWARD is the straight-through estimator: d out_g / d
rate[b,c,g'] = d out_g / d pb_g' evaluated at the sampled pseudobulk, for
every cell c of the bag (pb is a sum, so every cell gets the same gradient):

    d out_g / d pb_g' = s s_pb / sd_g * (1e4/N) / (1 + 1e4 pb_g / N) * (delta_gg' - pb_g / N),   N = sum pb

(when N >= 1).  The rate is a reparameterised Gamma(theta, theta/mu) draw,
so the gradient reaches the generator's mean mu (and theta) of every gene:
the generator is pushed on its expected count MASS per gene (the mean of
counts), which is what genemean does not constrain.  test_arm_critic.py
checks: real/fake forward equality on the same counts; autograd gradient
w.r.t. the rate equals a central finite difference of the block taken in
the counts; nonzero gradient on the generator output layer through a real
generator; nonzero critic gradient on the block at update 0 (trainer
assertion) in tiny fits.

The block is used only by the critic during training; generation is the
default procedure (no mode).

Device independence.  setup() computes m, sd, s_pb in float64 numpy on CPU
from the integer counts and space.npz; m and sd are stored as float32 and
s_pb rounded to six significant digits, so a GPU fit and a CPU regeneration
recompute an identical arm state.

State (hook f): s, s_pb, m [G], sd [G] (after flooring).
"""
from __future__ import annotations

import math
import zlib

import numpy as np
import torch

import arms
from arms import Arm, FeatureBlock, register, canonical_float

NAME = 'pbulk'
REF_BAGS = 512
REF_BATCH = 32
BAG = 128
SD_FLOOR_FRAC = 0.25


def r6(x):
    return float('%.6g' % float(x))


def ref_bag_rows(by, name, n_bags=REF_BAGS, batch=REF_BATCH, bag=BAG):
    rng = np.random.default_rng(np.random.SeedSequence([20261109, zlib.crc32(name.encode()), 0]))
    out = []
    for _ in range(n_bags // batch):
        ds = rng.integers(len(by), size=batch)
        out.append((ds, np.stack([rng.choice(by[i], bag, replace=True) for i in ds])))
    return out


def log_cp10k_pb(x):
    """[B, cells, G] counts (torch, may carry a straight-through gradient) -> [B, G] log1p CP10k of the bag sum."""
    pb = x.sum(1)
    return torch.log1p(pb * (1e4 / pb.sum(-1, keepdim=True).clamp_min(1)))


def log_cp10k_pb64(x):
    """numpy float64 version of log_cp10k_pb."""
    pb = x.sum(1)
    return np.log1p(pb * (1e4 / np.maximum(pb.sum(-1, keepdims=True), 1.0)))


class PbulkBlock(FeatureBlock):
    name = 'pb'

    def __init__(self, arm, n_genes):
        super().__init__()
        self.arm = arm
        self.width = int(n_genes)
        self.info = dict(arm.block_info)

    def _f(self, x):
        return (log_cp10k_pb(x) - self.arm.m) * self.arm.w

    def real(self, ctx, idx, ds):
        return self._f(ctx.X[idx])

    def fake(self, ctx, cf, rate, counts):
        return self._f(counts + (rate - rate.detach()))

    def state(self):
        return {'width': self.width}


@register
class PbulkArm(Arm):
    name = NAME
    value_policy = 'optional'
    needs_Z = False

    @classmethod
    def parse_value(cls, text):
        v, canon = canonical_float(text, lo=0.0, lo_open=True)
        if v == 1.0:
            raise ValueError('pbulk@1 is pbulk; use the name without a value')
        return v, canon

    def setup(self, ctx):
        self.mult = 1.0 if self.value is None else float(self.value)
        W = ctx.space['W'].astype(np.float64) / ctx.space['pc_sd'].astype(np.float64)
        mean = ctx.space['mean'].astype(np.float64)
        sd_g = ctx.space['sd'].astype(np.float64)
        from released_update import bag_features
        ys, n150 = [], 0.0
        # trainer R4-6 (iii): reference over the target (SEA-AD) donors only; all donors for a round-1 cache
        by_ref = [ctx.by[i] for i in getattr(ctx, 'target_idx', range(len(ctx.by)))]
        for ds, rows in ref_bag_rows(by_ref, NAME):
            x = ctx.X[torch.as_tensor(rows, device=ctx.device)].double().cpu().numpy()
            ys.append(log_cp10k_pb64(x))
            lib = np.maximum(x.sum(-1, keepdims=True), 1.0)
            z = (np.log1p(x * (1e4 / lib)) - mean) / sd_g
            n150 += float((bag_features(torch.from_numpy(z @ W)).numpy() ** 2).sum())
        y = np.concatenate(ys)
        R = len(y)
        m = y.mean(0)
        sd = y.std(0)
        floor = SD_FLOOR_FRAC * float(np.median(sd))
        n_floored = int((sd < floor).sum())
        sd = np.maximum(sd, floor)
        m32, sd32 = m.astype(np.float32), sd.astype(np.float32)
        zz = (y - m32.astype(np.float64)) / sd32.astype(np.float64)
        s_pb = r6(math.sqrt((n150 / R) / float((zz ** 2).sum(1).mean())))
        self.s_pb = s_pb
        self.scale = s_pb * self.mult
        dev = ctx.device
        self.m = torch.as_tensor(m32, device=dev)
        self.sd = torch.as_tensor(sd32, device=dev)
        self.w = torch.as_tensor((self.scale / sd32.astype(np.float64)).astype(np.float32), device=dev)
        # donor-pseudobulk SD (all cells of each train donor) vs bag SD: how much of the bag variance is between donors
        donor_y = np.stack([log_cp10k_pb64(ctx.X[torch.as_tensor(r, device=ctx.device)].double().cpu().numpy()[None])[0] for r in by_ref])
        ratio = donor_y.std(0) / sd
        self.block_info = {
            'formula': 'out = s * s_pb * (log1p(1e4 pb/sum pb) - m)/sd, pb = sum over the 128 cells of counts; fake: counts + (rate - rate.detach())',
            'reference_bags': R, 'reference_seed': f'SeedSequence([20261109, crc32({NAME!r}), 0])',
            's_pb': s_pb, 'multiplier': self.mult, 'scale_effective': r6(self.scale),
            'E_norm2_150_reference': r6(n150 / R),
            'sd_floor': r6(floor), 'sd_floor_rule': f'{SD_FLOOR_FRAC} x median gene bag SD', 'genes_floored': n_floored,
            'median_bag_sd': r6(float(np.median(sd))),
            'median_donor_pb_sd_over_bag_sd': r6(float(np.median(ratio))),
        }
        self.info = {'multiplier': self.mult, 's_pb': s_pb, 'scale_effective': r6(self.scale), 'genes_floored': n_floored}

    def feature_blocks(self, ctx):
        return [PbulkBlock(self, ctx.n_genes)]

    def state(self):
        return {'mult': self.mult, 's_pb': self.s_pb, 'm': self.m.cpu(), 'sd': self.sd.cpu()}
