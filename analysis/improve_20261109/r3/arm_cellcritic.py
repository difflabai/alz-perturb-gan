"""Round-3 R5: a cell-level set critic (flag ``d.cellset`` / ``d.cellset@m``).

Question (PROTOCOL_R3.md R5): should the critic see cells, not only bag
summaries?  The released critic sees 150 bag summaries of the 50 frozen PCs
(mean and SD of each PC, 50 Fourier means) and, with ``genemean``, the bag
mean of every standardised gene.  It never sees an individual cell, so the
joint distribution of cells inside a bag is constrained only through those
summaries.  This arm adds a DeepSets critic branch over the cells of the bag.

Flags
=====

``d.cellset``     per-cell input = the cell's 50 frozen-PC coordinates (the
                  same coordinates the released features summarise:
                  ``standardize_counts(x) @ W / pc_sd``), cell_dim 50.
``d.cellset@m``   per-cell input = 50 PCs + m train-fitted RESIDUAL gene
                  components (m integer in 1..256): the top-m principal axes
                  of the train cells' standardised log1p CP10k expression
                  after projecting out the span of the 50 released PCs,
                  centred on the train-cell mean and scaled to unit train
                  variance.  cell_dim 50 + m.  This is the "compressed
                  gene-level view": information outside the 50-PC space,
                  which the released features cannot see per cell.
``d.cellwhite[@m]`` the same block and critic, but the cell branch first
                  standardises every cell coordinate WITHIN its bag (per-bag
                  mean and sqrt(var + 1e-3); no statistic across bags), so the
                  cell branch only sees structure beyond the per-coordinate bag
                  means and SDs (correlations between PCs, tails, modes); the
                  released 150 (and genemean) still enter phi unchanged.
                  Motivation (offline probe below): a set classifier on raw
                  per-cell PCs separates real from fake bags of the TRAINING
                  donors but not of new donors (it memorises donors), while
                  the same classifier on within-bag standardised cells
                  separates both.  Conflicts with d.cellset.

Combine freely with ``genemean``, ``genemean@s`` and plug-in blocks that are
not critic modules (e.g. ``pbulk``); at most one arm may define a critic
module (core rule), so ``d.cellset`` conflicts with ``d.example`` and any
other ``d.`` critic-module arm.

Train-only fitting (setup, hook "setup")
========================================

The residual basis (m > 0) is fitted on the fold's TRAIN cells only (ctx.X;
no held-out counts), deterministically on CPU in float64 (numpy eigh of the
residual covariance), with an exact sign convention (largest-|.| loading
positive).  It depends on no random stream.  Its tensors are the arm state
(hook f), so resume and generation recompute and compare them.

Critic input (hook d): one per-cell block
=========================================

Block ``cells`` with ``cell_dim = 50 + m`` and width ``ctx.bag * (50 + m)``,
flattened cell-major.  Real cells: ``standardize_counts(ctx.X[idx]) @ A``;
fake cells: the straight-through ``critic_features`` tensor ``cf @ A`` with
``A = [W/pc_sd | V/s]`` (and the residual centring), i.e. exactly the same
function of counts for real and fake bags, with the generator gradient
flowing through the rate (pg_model.critic_features).  No ``ctx.Z`` is built.

Critic module (hook d2): ``CellSetCritic``
==========================================

    phi(x_vec, c)   released VectorCritic phi: Linear(n_vec+7, 128) SiLU Linear(128,128) SiLU
                    on every NON-cell column (released 150, genemean/donorpc
                    block, other plug-in blocks) + condition;
    psi(cell)       per-cell MLP Linear(d, 64) SiLU Linear(64, 64) SiLU, shared over cells;
    pool            mean over the bag's cells and sqrt(var + 1e-6) over cells
                    (per bag; no statistics across bags, deterministic);
    rho             Linear(128 + 2*64 + 7, 128) SiLU Linear(128, 1).

This is the released VectorCritic with a pooled cell branch concatenated
before rho (and like pg_model.SetCritic's mean+SD pooling).  All parameters
live in the module and go to the released critic optimizer (DualNorm, critic
learning rate).  Built under the critic_init seed by the core.  Permutation
invariant in the cells of a bag.

What the native BISPF critic penalty needs, and what changes (measured from
particlegan 0.9.7 _bispf/penalty.py, gates.py, transport.py)
=============================================================================

The trainer calls ``pen(D, real, f, c)`` with the critic input matrices
``real``/``f`` [B, n_features]; the penalty differentiates D w.r.t. THOSE
tensors (``autograd.grad`` on detached copies, ``create_graph=True``).  So a
cell-level critic is under the penalty only if the cells are part of x, which
is why the cells are flattened into a per-cell block of x (no side channel).
With ``input_domain='continuous'`` and dim = n_features > 64 (base 150; this
arm 150 + 128 (50+m)), the penalty is, for every width used here, the same
branch as for the released critic: no simplex law, no transport chords
(chord_max_dim 64), no continuous-flow / curvature / rms-exempt gates
(dim > 4).  It is

    coeff * [ 0.5*0.01*(mean|g_r|^2 + mean|g_f|^2)/(2 dim)
              + 0.5*(mean softplus_30(|g_r|/sqrt(dim) - kappa)^2 + mean softplus_30(|g_f|/sqrt(dim) - kappa)^2)
              + 0.5*0.05*mean_logit^2 ],

evaluated at the real and fake endpoints only.  Consequences:

* Compatible as is: the cells' gradients are capped together with the bag
  features (an RMS-normalised gradient norm over the whole x).
* RMS dilution: the cap is on |dD/dx| / sqrt(dim).  Mean pooling gives each
  cell coordinate a gradient of order 1/bag, so the 6,400+ cell columns add
  little to |g| but raise sqrt(dim): the Lipschitz budget on the released
  columns grows from sqrt(150) to sqrt(150 + 128 (50+m)) (6.6x for
  ``d.cellset``; ``genemean`` alone already gives 3.8x).  This cannot be
  undone inside the arm (scaling inputs inside the critic does not change the
  constraint on dD/dx); it needs the recipe field ``reg_kappa`` (core flag
  ``t.kappa@M``, PROTOCOL_R3 addition 1) set to sqrt(150/dim) to restore the
  released budget.  Reported, not patched.
* The Sinkhorn coupling inside the penalty uses Euclidean distances between
  flattened x; cell-major flattening is NOT permutation invariant, so the
  cell columns add order noise to those distances.  Here the coupling only
  feeds the thermal cooler (EMA nearest-neighbour residual; the trainer's loss
  is ``non_saturating``, not the Sinkhorn relativistic loss), and the cooler
  only acts when the residual falls below tau 0.02; noise raises the
  residual, so the cooler stays at 1, as in base.  Measured in the tests
  (penalty_stats s_thermal).
* The update-0 core check (dD/dx nonzero on the cell block) applies.

Generation is unaffected (the critic is not used at generation).

Measured (2026-11-11, fold 0 hvg2000, train donors only; test_arm_cellcritic.py,
results in runs/improve_20261109/smoke_r3/cellcritic/results/)
=========================================================================

* Offline set classifiers, real vs generated 128-cell bags, AUC on held-out
  bags of the classifier-training donors ("same") / of 12 other train donors
  ("new"), mean of 3 restarts.  Generators: round-1 base, round-2 genemean
  and pbulk (fold 0, seed 0, final state):

      view                       base same/new   genemean same/new   pbulk same/new
      released150                1.000/0.533     1.000/0.962         1.000/0.907
      released150+genemean       1.000/1.000     1.000/1.000         1.000/0.982
      cells PC50 (raw)           1.000/0.564     1.000/0.858         1.000/0.879
      cells PC50 within-bag std  1.000/0.995     1.000/1.000         1.000/1.000
      cells residual 64          1.000/1.000     1.000/0.998         1.000/1.000
      released150+genemean+cells 1.000/1.000     1.000/1.000         1.000/0.968

  Every view separates every generator perfectly on held-out bags of the
  same donors, so the critic is not short of information in that sense.  On
  new donors, raw per-cell PCs (and summaries alone) mostly memorise donors;
  within-bag standardised cells separate across donors, i.e. a cell-level
  discrepancy exists that is not a per-PC mean/SD difference.  With 16-cell
  bags (genemean generator, new donors) released150+genemean reach 0.998
  and adding cells does not raise it (0.998); standardised cells ALONE also
  reach 0.998: a second, independent signal, not a stronger one.
* What the discrepancy is (512 bags, all 25 train donors, 50-PC coordinates
  within a bag): participation ratio of the within-bag PC correlation matrix
  real 25.0, base 22.9, genemean 12.4, pbulk 13.3; top-5 eigenvalue share
  0.34 / 0.37 / 0.54 / 0.54; mean |off-diagonal correlation| 0.11 / 0.12 /
  0.21 / 0.20; mean per-PC kurtosis 3.89 / 3.28 / 3.16 / 3.25.  Generated
  cell-to-cell variation is too low-rank and too light-tailed, worst for the
  genemean-family generators (cell variation comes from an 8-dim noise
  through one 128-wide layer, so part of this may be generator capacity).
* The 50 PCs hold 9.1% of the train cells' standardised variance; the top 64
  residual axes hold 6.5% of the rest (each about 0.1%, near isotropic).
* Cost per update vs base: CPU (heavily loaded machine, 10 updates) 1.14-1.28x;
  GPU (shared, 150 updates) 1.22x for d.cellset@64 and 1.19x for
  d.cellset@64,genemean vs genemean; peak GPU memory +110 MB (base) and
  +200 MB (genemean).
* Existing learning curves (late updates, fold 0-2): RMS critic input
  gradient 0.85 for base (cap 1, raw |g| 10.4 of a 12.2 budget) and 0.32-0.40
  for genemean (raw |g| 15-18, cap not binding, critic gap 12-14 vs 4-5):
  consistent with the RMS dilution described above.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

import arms
from arms import Arm, FeatureBlock, register, canonical_int

NAME = 'd.cellset'
N_PC = 50
CELL_HIDDEN = 64
POOL_EPS = 1e-6
WHITEN_EPS = 1e-3
MAX_EXTRA = 256


def _standardize(x, mean, sd):
    """pg_model.standardize_counts (copied to avoid importing the benchmark here; same arithmetic)."""
    return (torch.log1p(x * (1e4 / x.sum(-1, keepdim=True).clamp_min(1))) - mean) / sd


def fit_residual_basis(X, mean, sd, W, m, chunk=2048):
    """Top-m principal axes of standardised train cells after removing span(W).  CPU float64, deterministic.

    X: [cells, G] train counts (any device); mean/sd: [G]; W: [G, 50] (W / pc_sd).
    Returns (V [G,m] float32 with unit-variance scaling folded in, centre [m] float32, info dict).
    The projection of a standardised cell z is (z @ V) - centre.
    """
    Xc = X.detach().cpu()
    mean_c, sd_c = mean.detach().cpu(), sd.detach().cpu()
    Q, _ = np.linalg.qr(W.detach().cpu().double().numpy())          # [G,50] orthonormal basis of span(W)
    G = Xc.shape[1]
    s1 = np.zeros(G)
    s2 = np.zeros((G, G))
    n = 0
    for a in range(0, Xc.shape[0], chunk):
        z = _standardize(Xc[a:a + chunk], mean_c, sd_c).double().numpy()
        r = z - (z @ Q) @ Q.T
        s1 += r.sum(0)
        s2 += r.T @ r
        n += len(r)
    mu = s1 / n
    cov = s2 / n - np.outer(mu, mu)
    cov = 0.5 * (cov + cov.T)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1][:m]
    lam, V = evals[order], evecs[:, order]
    V = V * np.sign(V[np.abs(V).argmax(0), np.arange(m)])
    if not (lam > 1e-8).all():
        raise ValueError('degenerate residual component')
    scale = np.sqrt(lam)
    Vs = V / scale
    centre = (mu @ Vs)            # residual mean projected (z @ V == r @ V since V is orthogonal to span(W))
    info = {'m': int(m), 'residual_variance_total': float(np.trace(cov)),
            'explained_residual_fraction': (lam / np.trace(cov)).tolist()[:16], 'explained_residual_fraction_sum': float(lam.sum() / np.trace(cov)),
            'train_cells': int(n)}
    return torch.as_tensor(Vs, dtype=torch.float32), torch.as_tensor(centre, dtype=torch.float32), info


class CellBlock(FeatureBlock):
    """Per-cell block: 50 frozen PCs (+ m residual components) of every cell, cell-major [B, cells*(50+m)]."""
    name = 'cells'

    def __init__(self, arm, cells):
        super().__init__()
        self.arm = arm
        self.cell_dim = N_PC + arm.m
        self.width = int(cells) * self.cell_dim
        self.info = {'what': 'per-cell [PC1-50 (cf @ W/pc_sd), residual components 1-m ((cf @ V) - centre)]', 'cells': int(cells),
                     'cell_dim': self.cell_dim, 'm': arm.m}

    def _cells(self, z):
        parts = [z @ self.arm.W]
        if self.arm.m:
            parts.append(z @ self.arm.V - self.arm.centre)
        x = torch.cat(parts, -1) if len(parts) > 1 else parts[0]
        return x.reshape(len(x), -1)

    def real(self, ctx, idx, ds):
        return self._cells(_standardize(ctx.X[idx], ctx.mean, ctx.sd))

    def fake(self, ctx, cf, rate, counts):
        return self._cells(cf)

    def state(self):
        return {'cell_dim': self.cell_dim, 'width': self.width}


class CellSetCritic(nn.Module):
    """Released VectorCritic phi on the non-cell columns + DeepSets cell branch (mean and SD pooling) before rho."""

    def __init__(self, layout, cell_hidden=CELL_HIDDEN, whiten=False):
        super().__init__()
        self.layout = layout
        self.whiten = bool(whiten)
        cells = [b for b in layout.blocks if b.cell_dim is not None]
        if len(cells) != 1:
            raise ValueError(f'CellSetCritic expects exactly one per-cell block, got {len(cells)}')
        self.cb = cells[0]
        cols = [i for b in layout.blocks if b.cell_dim is None for i in range(b.lo, b.hi)]
        self.register_buffer('vector_cols', torch.as_tensor(cols, dtype=torch.long), persistent=False)
        h, k = layout.hidden, layout.cond_dim
        self.phi = nn.Sequential(nn.Linear(len(cols) + k, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU())
        self.psi = nn.Sequential(nn.Linear(self.cb.cell_dim, cell_hidden), nn.SiLU(), nn.Linear(cell_hidden, cell_hidden), nn.SiLU())
        self.rho = nn.Sequential(nn.Linear(h + 2 * cell_hidden + k, h), nn.SiLU(), nn.Linear(h, 1))

    def forward(self, x, c):
        if x.ndim != 2 or x.shape[1] != self.layout.n_features or c.shape != (len(x), self.layout.cond_dim):
            raise ValueError(f'Expected [B,{self.layout.n_features}] and [B,{self.layout.cond_dim}]')
        f = self.phi(torch.cat([x[:, self.vector_cols], c], 1))
        cells = x[:, self.cb.lo:self.cb.hi].reshape(len(x), -1, self.cb.cell_dim)
        if self.whiten:   # per bag and coordinate (no statistics across bags): keep only structure beyond bag means/SDs
            m = cells.mean(1, keepdim=True)
            cells = (cells - m) / ((cells - m).square().mean(1, keepdim=True) + WHITEN_EPS).sqrt()
        e = self.psi(cells)
        mu = e.mean(1)
        sd = ((e - mu[:, None]).square().mean(1) + POOL_EPS).sqrt()
        return self.rho(torch.cat([f, mu, sd, c], 1)).squeeze(1)


@register
class CellSetArm(Arm):
    """Cell-level DeepSets critic branch (hooks setup, d, d2, f).  See the module docstring."""
    name = NAME
    value_policy = 'optional'
    conflicts = ('d.example', 'd.cellwhite')
    whiten = False

    @classmethod
    def parse_value(cls, text):
        return canonical_int(text, lo=1, hi=MAX_EXTRA)

    @property
    def info(self):
        """Repair TR-5: penalty_dim (the critic-input width the native penalty normalises by) is filled in as soon as
        ctx.features exists, i.e. already in make_config (config.json departures), not only in critic_module (build)."""
        i = self._info
        ctx = getattr(self, '_ctx', None)
        if isinstance(i, dict) and i.get('penalty_dim', 0) is None and ctx is not None and getattr(ctx, 'features', None) is not None:
            n = int(ctx.features.n_features)
            i['penalty_dim'] = n
            i['rms_budget_vs_released'] = float(np.sqrt(n / 150.0))
        return i

    @info.setter
    def info(self, value):
        self._info = value

    def setup(self, ctx):
        self._ctx = ctx
        self.m = 0 if self.value is None else int(self.value)
        self.W = ctx.W
        self.info = {'m': self.m, 'cell_dim': N_PC + self.m, 'cells': int(ctx.bag), 'cell_hidden': CELL_HIDDEN, 'pooling': 'mean+sd',
                     'within_bag_whitening': self.whiten,
                     'penalty_dim': None}
        if self.m:
            V, centre, info = fit_residual_basis(ctx.X, ctx.mean, ctx.sd, ctx.W, self.m)
            self.V, self.centre = V.to(ctx.device), centre.to(ctx.device)
            self.info['residual_basis'] = info
        else:
            self.V = self.centre = None

    def feature_blocks(self, ctx):
        return [CellBlock(self, ctx.bag)]

    def critic_module(self, ctx, layout):
        if self.info.get('penalty_dim') not in (None, int(layout.n_features)):
            raise ValueError(f"d.cellset: penalty_dim {self.info['penalty_dim']} recorded in config != critic input width {layout.n_features}")
        self.info['penalty_dim'] = int(layout.n_features)
        self.info['rms_budget_vs_released'] = float(np.sqrt(layout.n_features / 150.0))
        return CellSetCritic(layout, whiten=self.whiten)

    def state(self):
        q = {'m': self.m, 'whiten': self.whiten}
        if self.m:
            q.update(V=self.V.detach().cpu(), centre=self.centre.detach().cpu())
        return q


@register
class CellWhiteArm(CellSetArm):
    """``d.cellwhite[@m]``: d.cellset with the cells standardised within each bag before psi (see module docstring)."""
    name = 'd.cellwhite'
    conflicts = ('d.example', 'd.cellset')
    whiten = True
