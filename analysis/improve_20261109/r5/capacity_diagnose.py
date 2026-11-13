"""Round-3 capacity diagnosis (work items R2 particles and R3 generator architecture; PROTOCOL_R3.md).

Questions: do we need more particles (or a bigger latent)?  Is the generator
architecture holding us back?  How much do the GAN fits leave on the table
relative to what the architecture can represent (optimisation gap versus
capacity gap)?  Everything here reads existing fits and the fold caches
(TRAIN donors for every fitted quantity; held-out donors only in clearly
labelled descriptive columns) and never writes outside
runs/improve_20261109/smoke_r3/capacity/.

Findings (2026-11-11; outputs in runs/improve_20261109/smoke_r3/capacity/:
particles_*.json, probe_f{k}_{variant}[_s{seed}].json, ganmetrics*.json)
------------------------------------------------------------------------
Particles are NOT limiting.  Every one of the 512 rows is drawn about 1250
times (CV 0.029 = the Poisson value), none is unused, 31 of 32 rows per
batch are distinct.  But the generator maps the cloud onto very few
distinguishable donors: at the real technical-noise radius (median
sigma_d 0.17) the 512 particles give 17 (base), 8 (genemean), 4-11
(round-2 genemean@s, pbulk, gmlite) distinct donor profiles per stage; the
donor map's Jacobian has participation ratio 1.3-2.3 (97-99% of its energy in 4
of the 8 latent directions); the generated cloud's effective dimension is
1.2-4.4 against 9.4 for the real within-stage donor residuals, while its
total spread is about right (0.94-1.29 x real).  Coverage of train donors
barely depends on the number of particles: excess distance of the nearest
of 8 / 32 / 512 particles 0.66 / 0.61 / 0.58 (base), and the nearest of 512
is no closer than the generated MEAN profile (0.595); the nearest other
real donor is at 0.475.  Optimising z freely for each train donor (no
particle constraint) still leaves 0.41-0.50 (latent 10-13 Mahalanobis units
outside the cloud, which sits at 2.6).  Generated stage-mean profiles are
0.37-0.76 RMS away from the real train stage means.  So more particles or a
larger z_dim cannot help: the trained generator's donor map, not the number
or placement of particles, is the limit.

The architecture's width, depth and cell latent are NOT limiting.  The same
architecture fitted by maximum likelihood (probe, folds 0-2) reproduces
every train donor's pseudobulk to the technical-noise level (own-latent
pb_rmse 0.16-0.20 vs sigma 0.17-0.18; grand mean 0.45).  Validation ELBO
relative to the released architecture (nats per cell; seed-to-seed SD about
0.2): h256 +6.8, h512 +27 (memorises cells), h64 -1.9 / -2.6, depth2 +0.9,
depth3 +1.6, zc16 -0.5, zc32 -0.6 to -1.0, zd16 +0.1 (zd2: donors no longer
representable), no cell latent at all +3.1 to +3.6 with no worse generated
cell metrics.  None of these moves any generated metric beyond noise.

The single dispersion per gene IS a capacity limit of the cell-level
distribution.  Donor-dependent dispersion (g.disp) improves the validation
ELBO by 4.9-7.1 nats per cell in all 3 folds and 3 seeds and cuts the
probe's generated per-gene errors (train donors): mean log 0.07 -> 0.02,
nonzero fraction 0.018 -> 0.006, log variance 0.10 -> 0.054-0.078, own-latent
cell energy 0.30-0.37 -> 0.20-0.25 (real half-vs-half floor 0.10),
population cell energy -0.08 in every fold.  Learned donor log-dispersion
offsets have SD 0.55-0.62 across donors.  Cell-dependent dispersion
(g.cdisp) gives -2.4 to -7.1 (less consistent, 8x the parameters).

GAN versus probe (train targets, in-sample, fold means): pb_rmse GAN 0.59-0.65
(single fit), 0.50-0.53 (snapens) vs probe population mode 0.44-0.46 and
real resampling 0.42-0.44.  CORRECTED by the repair (finding E1, below): the
probe ``pop`` mode and the ``resample`` reference include the target donor
itself (about 16% of draws), so the claimed 0.06-0.2 "optimisation gap" on
pb_rmse is NOT supported for snapens; see "Repair" below.

Repair (round-3 adversarial review, 2026-11-11; outputs smoke_r3/capacity/repair/probe_f{k}_{released,disp,dispz}.json)
----------------------------------------------------------------------------------------------------------------------
E1 (self-inclusion; confirmed).  New probe modes ``popx`` (same-stage train donors EXCLUDING the
target), ``popall`` (any train donor's latent, target's condition, self with p = 1/25: the
closest ML analogue of the GAN, whose particle cloud spans all train donors) and ``popallx``
(any OTHER donor), and reference ``resamplex`` (real cells of same-stage donors excluding the
target).  Released architecture, pb_rmse folds 0 / 1 / 2: pop 0.437 / 0.453 / 0.435; popx
0.506 / 0.529 / 0.523; popall 0.449 / 0.477 / 0.465; popallx 0.473 / 0.487 / 0.487; real
resample 0.423 / 0.445 / 0.428 vs resamplex 0.508 / 0.526 / 0.526; in-sample grand mean
0.451 / 0.460 / 0.453.  GAN snapens train-target pb_rmse 0.505-0.525 (fold means) is
therefore at the leave-self-out same-stage level (popx = resamplex = 0.51-0.53) and 0.02-0.05
above the GAN-like ML population (popallx 0.47-0.49).  Restated finding (c): on train-target
pb_rmse the snapens GAN shows at most a small gap (0.02-0.05) to what an ML fit of the same
architecture gives as a population generator; the large gap is between ONE GAN state (0.59-0.65)
and the averaged states, i.e. the late-training drift that snapens / EMA / annealing target.
The train-target probe cannot separate optimisation from capacity further.  The held-out gap
of the GAN (pbulk~r2#snapens 0.546 over 2 seeds) to ideal resampling (resample 0.506 same
stage, resample_blind 0.474 stage-blind; r3 evaluator) is real but this probe says nothing
about its cause.  The own-latent result (pb_rmse 0.16-0.20) is in-sample memorisation of 25
free latents: it shows the architecture can represent each train donor, not population
capacity.  The width / depth / cell-latent / dispersion conclusions use validation ELBO and
per-gene errors and do not depend on ``pop``; they stand.

TR-1 (g.disp's stage input).  Variant ``dispz`` (= flag g.dispz: dispersion from the donor
latent only) keeps the whole ML gain of ``disp``: validation NELBO relative to released
-6.9 / -4.8 / -4.9 nats per cell (disp -7.1 / -4.9 / -5.0); per-gene errors (popall) mean log
0.029 / 0.031 / 0.023 (disp 0.027 / 0.029 / 0.022; released 0.077 / 0.070 / 0.068), log variance
0.064 / 0.085 / 0.063 (disp 0.064 / 0.082 / 0.062; released 0.110 / 0.114 / 0.100); donor
log-dispersion offset SD 0.62 / 0.59 / 0.56.  The stage input contributes nothing measurable
to the ML gain, so the screen uses g.dispz.
Mean-log per-gene error: GAN genemean snapens 0.06, probe 0.07, probe with
g.disp 0.02.  Log-variance: GAN 0.16-0.29, probe 0.10-0.11, probe with g.disp
0.054-0.078.  These cell-level metrics are where g.disp can plausibly move the
T2 targets.

CLI (CPU, 3 threads; run from /tmp)::

    PY=/mnt/nvme/ai/conda/anaconda3/bin/python3
    $PY -B capacity_diagnose.py particles --fits GLOB [GLOB...] --out JSON [--invert] [--cells 512]
    $PY -B capacity_diagnose.py probe --fold K --variant NAME [--steps S] --out JSON
          [--device cuda --steps 300 --batch 4096 --lr 3e-3] [--seed N]
    $PY -B capacity_diagnose.py ganmetrics --fits GLOB [GLOB...] --out JSON [--mode snapens]
    $PY -B capacity_diagnose.py report --dir DIR

particles (a)
-------------
For every fit (final.pt on CPU; arms whose generator is plain
pg_model.CountGenerator and whose condition is stage only):

* particle table z [512, 8]: covariance spectrum, participation ratio
  PR = (sum l)^2 / sum l^2 (effective dimension), nearest-neighbour vs
  median pairwise distance; usage statistics from final.pt ``usage``
  (draws per row, rows moved, unique rows per batch).
* donor map.  For a particle z and stage s the generator's expected donor
  profile is  P(z, s) = log1p(1e4 * mean_k softmax(logits(z, s, eps_k)))
  over K fixed cell-noise draws eps_k (common to all particles; this is the
  noise-free version of the generated pseudobulk log1p CP10k that pb_rmse
  uses).  Real train-donor profiles R_d = log1p CP10k of the donor's summed
  counts; technical noise of R_d: sigma_d = rms(R_dA - R_dB) / 2 from a
  random half split of the donor's cells.  Distances are RMS over genes
  (pb_rmse units).  Per stage: between-particle variance trace versus the
  real pooled within-stage between-donor variance (raw and minus the
  technical part), PR of the generated cloud versus the real within-stage
  residuals, cumulative variance of the top components, number of
  distinguishable particles (greedy covering at radius = median sigma_d),
  nearest-real-donor assignment of particles (effective number of donors
  the cloud sits on), coverage curve of every train donor by its nearest
  particle among N = 8, 32, 128, 512 (random subsets, 10 repeats) with the
  technical noise removed: excess = sqrt(max(d^2 - sigma_d^2, 0)), and the
  same for the nearest other real donor; Jacobian dP/dz singular values at
  32 particles (how many of the 8 latent directions the generator uses).
* ``--invert``: per train donor, z optimised (Adam, 200 steps, from the
  nearest particle, fixed noise) to minimise rms(P(z, s_d) - R_d): how close
  the trained generator's donor manifold can get with a free latent, against
  the best of the 512 particles.

probe (b): maximum-likelihood capacity probe (NOT a study model)
-----------------------------------------------------------------
The SAME generator architecture (``arm_garch.make_generator_class``, i.e.
exactly what the g. flags build) fitted by maximum likelihood to the fold's
train-donor cells, without any GAN: a free per-donor latent table E [n, 8]
(the particle's role), the stage condition, and a per-cell latent eps
marginalised with an amortised encoder (variational autoencoder with the
generator as decoder; encoder 2000 -> 256 -> 256 -> 2 zc on standardised
log1p CP10k; ELBO = Gamma-Poisson (negative binomial) log-likelihood of the
observed counts with mu = library * softmax, theta = generator dispersion,
minus KL(q(eps|x) || N(0, I))).  Adam (cosine to lr/20), 10% of each
donor's cells held out for the validation ELBO.  Runs used for the round-3
report: ``--device cuda --steps 300 --batch 4096 --lr 3e-3`` (the GPU rule
allows test fits of at most 300 updates; 300 x 4096 cells is about 120
passes over the ~10,000 training cells; the validation ELBO plateaus by
update 200), generation on the same GPU, scoring on CPU.  Variants
(``VARIANTS``): released (h128 zc8), wider (h256, h512), larger cell latent
(zc16, zc32), deeper head (depth2, depth3), donor dispersion (disp), all
(h512 zc32 depth3), and ``nocell`` (eps = 0: no cell latent; cell
heterogeneity only from the Gamma-Poisson draw).  After fitting, cells are
generated exactly as generate.py does (eps ~ N(0, I), libraries resampled
from the target donor's own cells, Gamma-Poisson counts, M = 32 x K = 256)
in two modes: ``own`` (each train donor's own learned latent: what the
architecture can represent for that donor) and ``pop`` (latents of random
same-stage train donors: a stage-conditioned population generator like the
GAN's train-target generation).  The same train-target metrics are computed
for GAN fits by ``ganmetrics`` and for two references: ``split`` (half of
each donor's real cells against the other half: noise floor for ``own``) and
``resample`` (real cells of random same-stage train donors: the reference for
``pop``, like the evaluator's resample baseline).

Train-target metrics (``score_train``; same definitions as the evaluator but
on TRAIN donors, all in-sample by construction): pb_rmse (per-donor RMS of
mean-over-M generated pseudobulk log1p CP10k minus observed), meanlog /
nonzero / logvar per-gene errors averaged over donors (``*_rmse_gene``) and
per donor-gene (``*_rmse_dg``), and cell energy distance in the fold's
50-PC space (512 generated vs 512 real cells per donor, averaged).

ganmetrics (c)
--------------
The train-target metrics of existing GAN fits (generated.npz or a mode file,
train targets only), for comparison with probe ``pop`` (same objective
class of generator, maximum likelihood) and ``resample``.

All random draws use fixed numpy / torch generators seeded here; nothing
touches any training stream.  CPU only.
"""
from __future__ import annotations

import os
import sys

for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_k, '3')
sys.dont_write_bytecode = True

import argparse
import glob
import json
import math
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL = Path('/ml2/alz/analysis/faithful097/model')
BENCH = Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark')
for _p in (BENCH, MODEL):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
if str(HERE) in sys.path:
    sys.path.remove(str(HERE))
sys.path.insert(0, str(HERE))

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F
from scipy.spatial.distance import cdist

from release_env import activate
activate()
from pg_model import CountGenerator  # noqa: E402

import arms  # noqa: E402  (r3)
import arm_garch  # noqa: E402

RUNS = Path('/ml2/alz/runs/improve_20261109')
OUT_DIR = RUNS / 'smoke_r3' / 'capacity'
CACHES = RUNS / 'caches'
FOLDS_R1 = HERE.parent / 'folds.json'
SEED = 20261111
PLAIN_FLAGS = {'genemean', 'donorpc', 'pbulk', 'gmlite', 'donorpc2'}   # stage-only condition, plain CountGenerator
M_GEN, K_GEN, PC_CELLS = 32, 256, 64
torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '3')))


def _out_ok(p):
    p = Path(p).resolve()
    if OUT_DIR.resolve() not in p.parents and p.parent != OUT_DIR.resolve():
        raise ValueError(f'outputs go under {OUT_DIR}')
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def dump(p, obj):
    p = _out_ok(p)
    tmp = p.with_name('.' + p.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1, allow_nan=False, default=float) + '\n')
    tmp.replace(p)


def _dev_guard(donors):
    q = json.loads(FOLDS_R1.read_text())
    dev, sealed = set(q['development_donors']), set(q['sealed_evaluation_donors_fold0'])
    bad = [d for d in donors if d not in dev or d in sealed]
    if bad:
        raise PermissionError(f'non-development donors in a cache: {bad}')


# =========================================================================== data
def logcp(x):
    """log1p CP10k along the last axis (panel total, clamp_min(1)) - torch or numpy."""
    if isinstance(x, torch.Tensor):
        return torch.log1p(x * (1e4 / x.sum(-1, keepdim=True).clamp_min(1)))
    x = np.asarray(x, dtype=np.float64)
    return np.log1p(x * (1e4 / np.maximum(x.sum(-1, keepdims=True), 1)))


class Cache:
    """Train (and optionally held-out) cells of a fold cache with the fold space; per-donor real statistics."""

    def __init__(self, fold, panel='hvg2000', heldout=False):
        c = CACHES / panel / f'fold{fold}'
        self.fold, self.panel = int(fold), panel
        man = json.loads((c / 'manifest.json').read_text())
        _dev_guard(list(man['train_donors']) + list(man.get('heldout_donors', [])))
        s = np.load(c / 'space.npz', allow_pickle=False)
        self.mean = torch.as_tensor(s['mean']).float()
        self.sd = torch.as_tensor(s['sd']).float()
        self.W = torch.as_tensor(s['W'] / s['pc_sd']).float()
        self.genes = s['genes'].astype(str)
        parts = [('train', False)] + ([('heldout', True)] if heldout else [])
        self.sets = {}
        for name, held in parts:
            m = np.load(c / name / 'metadata.npz', allow_pickle=False)
            X = sp.load_npz(c / name / 'counts.npz').toarray().astype(np.float32)
            ids = m['donors'].astype(str)
            lab = m['donor'].astype(str)
            by = [np.flatnonzero(lab == d) for d in ids]
            stage = np.rint(m['raw_c'][:, 3] * 3).astype(int)
            self.sets[name] = {'X': torch.as_tensor(X), 'ids': ids, 'by': by, 'stage': stage, 'raw_c': m['raw_c'].astype(np.float64)}
        tr = self.sets['train']
        self.X, self.ids, self.by, self.stage = tr['X'], tr['ids'], tr['by'], tr['stage']
        self.n_genes = self.X.shape[1]
        self.mu = self.X.mean(0)
        v = self.X.var(0, unbiased=False)
        self.theta = (self.mu.square() / (v - self.mu).clamp_min(1e-4)).clamp(.1, 10000)

    def pcs(self, counts):
        return ((logcp(counts) - self.mean) / self.sd) @ self.W

    def donor_stats(self, name='train', seed=SEED):
        """Per donor: pb profile, meanlog, nonzero, logvar (ddof 1), cell PCs, and the same for a random half split."""
        S = self.sets[name]
        rng = np.random.default_rng(np.random.SeedSequence([seed, self.fold, 11 if name == 'train' else 12]))
        out = []
        for i, rows in enumerate(S['by']):
            x = S['X'][torch.as_tensor(rows)]
            perm = rng.permutation(len(rows))
            halves = [x[torch.as_tensor(perm[:len(rows) // 2])], x[torch.as_tensor(perm[len(rows) // 2:])]]
            d = {'donor': str(S['ids'][i]), 'stage': int(S['stage'][i]), 'n_cells': len(rows), 'lib': x.sum(-1).numpy()}
            d.update(_cell_stats(x))
            d['pcs'] = self.pcs(x).numpy()
            d['halves'] = []
            for hx in halves:
                hq = _cell_stats(hx)
                hq['pcs'] = self.pcs(hx).numpy()
                d['halves'].append(hq)
            d['sigma_pb'] = float(np.sqrt(np.mean((d['halves'][0]['pb'] - d['halves'][1]['pb']) ** 2)) / 2)
            out.append(d)
        return out


def _cell_stats(x):
    L = logcp(x).double()
    return {'pb': logcp(x.sum(0).double()).numpy(), 'meanlog': L.mean(0).numpy(), 'nonzero': (x > 0).double().mean(0).numpy(),
            'logvar': L.var(0, unbiased=True).numpy()}


# =========================================================================== particles (a)
def _fit_ok(cfg):
    names = {arms.split_flag(f)[0] for f in cfg['flags']}
    return names <= PLAIN_FLAGS


def load_plain_generator(sd, n_genes, zd=8, zc=8, h=128):
    G = CountGenerator(torch.zeros(n_genes), torch.ones(n_genes), zd=zd, zc=zc, h=h)
    G.load_state_dict(sd, strict=True)
    G.eval().requires_grad_(False)
    return G


def _hook_parts(G):
    """(film, cell_hidden, logits) for plain CountGenerator or HookedCountGenerator chains."""
    if hasattr(G, 'film'):
        return G.film, G.cell_hidden, G.logits
    return ((lambda z, c: G.donor(torch.cat([z, c], 1)).chunk(2, 1)),
            (lambda n, s, t, z, c: G.cell(n) * (1 + s[:, None]) + t[:, None]),
            (lambda h, z, c: G.out(h)))


def donor_profiles(G, z, stage, noise, chunk=16):
    """P(z, s) = log1p(1e4 * mean_k softmax(logits)) [N, G] for fixed cell noise [K, zc] (differentiable in z)."""
    film, cellh, logit = _hook_parts(G)
    out = []
    for a in range(0, len(z), chunk):
        zz = z[a:a + chunk]
        c = torch.zeros(len(zz), 7)
        c[:, 3] = float(np.float32(stage / 3))
        s, t = film(zz, c)
        h = cellh(noise[None].expand(len(zz), -1, -1), s, t, zz, c)
        p = torch.softmax(logit(h, zz, c), -1).clamp_min(1e-8)
        p = p / p.sum(-1, keepdim=True)
        out.append(torch.log1p(1e4 * p.mean(1)))
    return torch.cat(out)


def _rms(a, b):
    return np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).mean(-1))


def _pr(X):
    Xc = X - X.mean(0)
    s = np.linalg.svd(Xc, compute_uv=False) ** 2
    return float(s.sum() ** 2 / (s ** 2).sum()) if s.sum() > 0 else 0.0, (np.cumsum(s) / s.sum())[:10].tolist()


def greedy_cover(D, r):
    """Number of radius-r balls (greedy farthest point) needed to cover all points, D = pairwise distances."""
    n = len(D)
    centers = [0]
    dmin = D[0].copy()
    while dmin.max() > r:
        j = int(dmin.argmax())
        centers.append(j)
        dmin = np.minimum(dmin, D[j])
    return len(centers)


def particles_fit(fit, cache, real, *, cells=512, invert=False, heldreal=None):
    fit = Path(fit)
    cfg = json.loads((fit / 'config.json').read_text())
    q = torch.load(fit / 'final.pt', map_location='cpu', weights_only=False)
    mods = q['policy']['models']
    G = load_plain_generator(mods['generator'], cache.n_genes)
    Z = mods['prior']['z'].float()
    N, zd = Z.shape
    res = {'fit': str(fit), 'arm': arms.arm_name(tuple(cfg['flags'])), 'fold': cfg['fold'], 'seed': cfg['seed'], 'round': cfg.get('round', 1),
           'num_particles': N, 'z_dim': zd}
    # ---- particle table
    Zn = Z.double().numpy()
    ev = np.sort(np.linalg.eigvalsh(np.cov(Zn.T)))[::-1]
    Dz = cdist(Zn, Zn)
    np.fill_diagonal(Dz, np.inf)
    res['z'] = {'cov_eigs': ev.tolist(), 'trace': float(ev.sum()), 'pr': float(ev.sum() ** 2 / (ev ** 2).sum()),
                'nn_median': float(np.median(Dz.min(1))), 'pairwise_median': float(np.median(Dz[np.isfinite(Dz)])),
                'mean_norm': float(np.linalg.norm(Zn, axis=1).mean())}
    u = q['usage']
    draws = u['draws'].double().numpy()
    hist = np.asarray(u['unique_batch_histogram'], dtype=float)
    res['usage'] = {'steps': int(u['steps']), 'draws_mean': float(draws.mean()), 'draws_min': float(draws.min()), 'draws_max': float(draws.max()),
                    'draws_cv': float(draws.std() / draws.mean()), 'draws_cv_poisson': float(1 / np.sqrt(draws.mean())),
                    'never_drawn': int((draws == 0).sum()), 'moved_steps_mean': float(u['moved_steps'].double().mean()),
                    'moved_steps_min': float(u['moved_steps'].double().min()),
                    'unsampled_moved_steps_mean': float(u['unsampled_moved_steps'].double().mean()),
                    'unique_rows_per_batch_mean': float((hist * np.arange(len(hist))).sum() / max(hist.sum(), 1))}
    # ---- donor map
    gnoise = torch.Generator().manual_seed(SEED)
    noise = torch.randn(cells, G.zc, generator=gnoise)
    stages = sorted(set(int(d['stage']) for d in real))
    R = np.stack([d['pb'] for d in real])
    rstage = np.array([d['stage'] for d in real])
    sig = np.array([d['sigma_pb'] for d in real])
    with torch.no_grad():
        P = {s: donor_profiles(G, Z, s, noise).double().numpy() for s in stages}
    # real within-stage
    resid, tech = [], []
    for s in stages:
        m = rstage == s
        if m.sum() > 1:
            resid.append(R[m] - R[m].mean(0))
            tech += list(sig[m] ** 2)
    resid = np.concatenate(resid)
    dof = len(resid) - len([s for s in stages if (rstage == s).sum() > 1])
    Vr = float((resid ** 2).sum() / dof)               # sum over genes of pooled within-stage variance
    Vt = float(np.mean(tech) * cache.n_genes)         # technical part (sigma_d is an RMS over genes)
    out_st = {}
    for s in stages:
        Ps = P[s]
        Vg = float(Ps.var(0, ddof=1).sum())
        pr, cum = _pr(Ps)
        Dp = _rms(Ps, Ps)
        r_tech = float(np.median(sig))
        m = rstage == s
        Rs = R[m]
        Dpr = _rms(Ps, Rs)                              # particle x donor
        assign = np.bincount(Dpr.argmin(1), minlength=m.sum()) / N
        ent = float(np.exp(-(assign[assign > 0] * np.log(assign[assign > 0])).sum()))
        cov = {}
        rng = np.random.default_rng(np.random.SeedSequence([SEED, cfg['fold'], cfg['seed'], s]))
        for n_sub in (8, 32, 128, 512):
            if n_sub > N:
                continue
            reps = 1 if n_sub == N else 10
            dmins = np.mean([Dpr[rng.choice(N, n_sub, replace=False)].min(0) for _ in range(reps)], 0)
            cov[str(n_sub)] = {'dist': float(dmins.mean()),
                               'excess': float(np.sqrt(np.maximum(dmins ** 2 - sig[m] ** 2, 0)).mean())}
        dmean = np.sqrt(((Ps.mean(0)[None] - Rs) ** 2).mean(-1))
        cov['mean'] = {'dist': float(dmean.mean()), 'excess': float(np.sqrt(np.maximum(dmean ** 2 - sig[m] ** 2, 0)).mean())}
        if m.sum() > 1:
            Drr = _rms(Rs, Rs)
            np.fill_diagonal(Drr, np.inf)
            dnn = Drr.min(1)
            cov['real_nn'] = {'dist': float(dnn.mean()),
                              'excess': float(np.sqrt(np.maximum(dnn ** 2 - sig[m] ** 2 - np.array([sig[m][j] ** 2 for j in Drr.argmin(1)]), 0)).mean())}
        out_st[str(s)] = {'n_donors': int(m.sum()), 'gen_var_trace': Vg, 'gen_pr': pr, 'gen_cumvar_top10': cum,
                          'distinct_at_tech': greedy_cover(Dp, r_tech), 'distinct_at_2tech': greedy_cover(Dp, 2 * r_tech),
                          'nn_particle_median': float(np.median(np.where(np.eye(N, dtype=bool), np.inf, Dp).min(1))),
                          'assign_eff_donors': ent, 'assign_max_share': float(assign.max()),
                          'frac_particles_within_tech_of_a_donor': float((Dpr.min(1) < np.median(sig)).mean()),
                          'coverage': cov}
    rpr, rcum = _pr(resid)
    vg_mean = float(np.mean([out_st[str(s)]['gen_var_trace'] for s in stages if (rstage == s).sum() > 1]))
    res['donor_map'] = {'cells': cells, 'real_within_var_trace': Vr, 'real_tech_var_trace': Vt, 'real_resid_pr': rpr,
                        'real_resid_cumvar_top10': rcum, 'sigma_pb_median': float(np.median(sig)),
                        'spread_ratio_vs_real': float(np.sqrt(vg_mean / Vr)),
                        'spread_ratio_vs_real_minus_tech': float(np.sqrt(vg_mean / max(Vr - Vt, 1e-12))),
                        'stage_means_rms_from_real_stage_means': {str(s): float(np.sqrt(((P[s].mean(0) - R[rstage == s].mean(0)) ** 2).mean())) for s in stages},
                        'per_stage': out_st}
    # ---- Jacobian of the donor map in z (how many latent directions are used)
    smain = int(np.bincount(rstage).argmax())
    idx = np.random.default_rng(SEED).choice(N, 32, replace=False)
    svals = []
    for i in idx:
        zi = Z[i:i + 1].clone()
        J = torch.func.jacfwd(lambda v: donor_profiles(G, v, smain, noise[:128])[0])(zi)[:, 0, :]   # [G, zd], forward mode
        svals.append(np.linalg.svd(J.double().numpy(), compute_uv=False))
    sv = np.mean(svals, 0)
    res['jacobian'] = {'stage': smain, 'singular_values_mean': sv.tolist(), 'pr': float((sv ** 2).sum() ** 2 / (sv ** 4).sum()),
                       'frac_energy_top4': float((sv[:4] ** 2).sum() / (sv ** 2).sum())}
    # ---- held-out donors (descriptive only)
    if heldreal is not None:
        Hd = []
        for d in heldreal:
            s = d['stage']
            if s not in P:
                continue
            dist = np.sqrt(((P[s] - d['pb'][None]) ** 2).mean(-1))
            Rs = R[rstage == s]
            dr = np.sqrt(((Rs - d['pb'][None]) ** 2).mean(-1))
            Hd.append({'nn_particle': float(dist.min()), 'gen_mean': float(np.sqrt(((P[s].mean(0) - d['pb']) ** 2).mean())),
                       'nn_train_donor': float(dr.min()), 'sigma': d['sigma_pb']})
        res['heldout_descriptive'] = {k: float(np.mean([h[k] for h in Hd])) for k in Hd[0]} if Hd else None
    # ---- inversion
    if invert:
        inv = []
        nz = noise[:256]
        for j, d in enumerate(real):
            s = d['stage']
            Dpr_j = np.sqrt(((P[s] - d['pb'][None]) ** 2).mean(-1))
            z0 = Z[int(Dpr_j.argmin())].clone()
            zv = z0.clone()[None].requires_grad_(True)
            opt = torch.optim.Adam([zv], lr=.05)
            tgt = torch.as_tensor(d['pb']).float()
            for _ in range(200):
                opt.zero_grad()
                loss = (donor_profiles(G, zv, s, nz)[0] - tgt).square().mean()
                loss.backward()
                opt.step()
            with torch.no_grad():
                dd = float(np.sqrt(((donor_profiles(G, zv.detach(), s, noise).double().numpy()[0] - d['pb']) ** 2).mean()))
            zin = zv.detach().double().numpy()[0]
            maha = float(np.sqrt((zin - Zn.mean(0)) @ np.linalg.solve(np.cov(Zn.T), zin - Zn.mean(0))))
            inv.append({'nn_particle': float(Dpr_j.min()), 'inverted': dd, 'sigma': d['sigma_pb'], 'z_mahalanobis': maha})
        ex = lambda k: float(np.mean([np.sqrt(max(x[k] ** 2 - x['sigma'] ** 2, 0)) for x in inv]))
        res['inversion'] = {'nn_particle': float(np.mean([x['nn_particle'] for x in inv])), 'inverted': float(np.mean([x['inverted'] for x in inv])),
                            'nn_particle_excess': ex('nn_particle'), 'inverted_excess': ex('inverted'),
                            'z_mahalanobis_median': float(np.median([x['z_mahalanobis'] for x in inv])),
                            'z_mahalanobis_particles_median': float(np.median(np.sqrt(np.einsum('ij,ij->i', Zn - Zn.mean(0),
                                                                                                   np.linalg.solve(np.cov(Zn.T), (Zn - Zn.mean(0)).T).T))))}
    return res


def cmd_particles(a):
    fits = sorted({f for g in a.fits for f in glob.glob(g)})
    fits = [f for f in fits if (Path(f) / 'generated.npz').exists() and (Path(f) / 'final.pt').exists()]
    caches, out = {}, []
    if a.out and Path(a.out).exists():
        out = json.loads(Path(a.out).read_text())['fits']
    done = {r['fit'] for r in out}
    for f in fits:
        if str(Path(f)) in done:
            continue
        cfg = json.loads((Path(f) / 'config.json').read_text())
        if not _fit_ok(cfg) or cfg.get('panel') != 'hvg2000':
            continue
        k = cfg['fold']
        if k not in caches:
            c = Cache(k, heldout=a.heldout)
            caches[k] = (c, c.donor_stats('train'), c.donor_stats('heldout') if a.heldout else None)
        c, real, held = caches[k]
        t0 = time.time()
        r = particles_fit(f, c, real, cells=a.cells, invert=a.invert, heldreal=held)
        r['seconds'] = time.time() - t0
        out.append(r)
        print(f'[particles] {r["arm"]} fold{k} seed{cfg["seed"]} spread {r["donor_map"]["spread_ratio_vs_real"]:.3f} '
              f'zPR {r["z"]["pr"]:.2f} jacPR {r["jacobian"]["pr"]:.2f} {r["seconds"]:.0f}s', flush=True)
        dump(a.out, {'fits': out})
    dump(a.out, {'fits': out})


# =========================================================================== train-target metrics
def score_train(gen, real, cache, *, seed=SEED):
    """gen: dict of arrays for the train targets [n, M, ...] (pb_counts, mean_log, nonzero, sq_log or logvar, pcs [n,M,c,50]);
    real: Cache.donor_stats('train').  In-sample train-donor metrics (module docstring)."""
    n = len(real)
    pred = np.stack([logcp(gen['pb_counts'][i].astype(np.float64)).mean(0) for i in range(n)])
    obs = np.stack([d['pb'] for d in real])
    out = {'pb_rmse': float(np.sqrt(((pred - obs) ** 2).mean(1)).mean())}
    grand = obs.mean(0)
    out['pb_rmse_grand_mean_in_sample'] = float(np.sqrt(((grand[None] - obs) ** 2).mean(1)).mean())
    for key in ('meanlog', 'nonzero', 'logvar'):
        if key == 'logvar':
            if 'logvar' in gen:
                g = gen['logvar'].mean(1)
            elif 'sq_log' in gen:
                K = gen.get('K', K_GEN)
                g = ((gen['sq_log'].astype(np.float64) - gen['mean_log'].astype(np.float64) ** 2) * K / (K - 1)).mean(1)
            else:
                continue
        else:
            g = gen['mean_log' if key == 'meanlog' else 'nonzero'].astype(np.float64).mean(1)
        r = np.stack([d[key] for d in real])
        e = g - r
        out[key + '_rmse_gene'] = float(np.sqrt((e.mean(0) ** 2).mean()))
        out[key + '_rmse_dg'] = float(np.sqrt((e ** 2).mean()))
        out[key + '_bias'] = float(e.mean())
    rng = np.random.default_rng(np.random.SeedSequence([seed, cache.fold, 21]))
    ce = []
    for i, d in enumerate(real):
        gp = gen['pcs'][i].reshape(-1, 50).astype(np.float64)
        gp = gp[rng.choice(len(gp), min(512, len(gp)), replace=False)]
        rp = d['pcs'][rng.choice(len(d['pcs']), min(512, len(d['pcs'])), replace=False)]
        ce.append(2 * cdist(gp, rp).mean() - cdist(gp, gp).mean() - cdist(rp, rp).mean())
    out['cell_energy'] = float(np.mean(ce))
    return out


def reference_arrays(cache, real, kind, *, seed=SEED):
    """'split': half A of the donor's own cells as 'generated' (scored against the FULL donor, so it is optimistic;
    the half-vs-half numbers are in floors()); 'resample': cells of random same-stage train donors (incl. self), M x K;
    'resamplex' (repair E1): the same EXCLUDING the target donor itself (the honest population reference)."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, cache.fold, {'resample': 31, 'split': 32, 'resamplex': 33}[kind]]))
    n = len(real)
    stage = np.array([d['stage'] for d in real])
    arr = {k: np.zeros((n, M_GEN, cache.n_genes), np.float32) for k in ('pb_counts', 'mean_log', 'nonzero', 'logvar')}
    arr['pcs'] = np.zeros((n, M_GEN, PC_CELLS, 50), np.float32)
    for i in range(n):
        for m in range(M_GEN):
            if kind == 'resample':
                j = int(rng.choice(np.flatnonzero(stage == stage[i])))
            elif kind == 'resamplex':
                j = int(rng.choice(np.flatnonzero((stage == stage[i]) & (np.arange(n) != i))))
            else:
                j = i
            rows = cache.by[j]
            x = cache.X[torch.as_tensor(rng.choice(rows, K_GEN, replace=True))]
            _fill(arr, i, m, x, cache)
    return arr


def _fill(arr, i, m, x, cache):
    L = logcp(x)
    arr['pb_counts'][i, m] = x.sum(0).numpy()
    arr['mean_log'][i, m] = L.mean(0).numpy()
    arr['nonzero'][i, m] = (x > 0).float().mean(0).numpy()
    arr['logvar'][i, m] = L.var(0, unbiased=True).numpy()
    arr['pcs'][i, m] = cache.pcs(x[:PC_CELLS]).numpy()


def floors(cache, real):
    """Half-vs-half real-cell noise floors per metric (each half ~n/2 cells; upper bound of the own-mode floor)."""
    A = [d['halves'][0] for d in real]
    B = [d['halves'][1] for d in real]
    out = {'pb_rmse_half_vs_half': float(np.mean([np.sqrt(((a['pb'] - b['pb']) ** 2).mean()) for a, b in zip(A, B)])),
           'pb_sigma_full_donor': float(np.mean([d['sigma_pb'] for d in real]))}
    for key in ('meanlog', 'nonzero', 'logvar'):
        e = np.stack([a[key] - b[key] for a, b in zip(A, B)])
        out[key + '_rmse_gene_half'] = float(np.sqrt((e.mean(0) ** 2).mean()))
        out[key + '_rmse_dg_half'] = float(np.sqrt((e ** 2).mean()))
    rng = np.random.default_rng(np.random.SeedSequence([SEED, cache.fold, 41]))
    ce = []
    for a, b in zip(A, B):
        n = min(512, len(a['pcs']), len(b['pcs']))
        pa, pb = a['pcs'][rng.choice(len(a['pcs']), n, replace=False)], b['pcs'][rng.choice(len(b['pcs']), n, replace=False)]
        ce.append(2 * cdist(pa, pb).mean() - cdist(pa, pa).mean() - cdist(pb, pb).mean())
    out['cell_energy_half_vs_half'] = float(np.mean(ce))
    return out


def load_gen_train(path, n_train):
    with np.load(path, allow_pickle=False) as z:
        if z['is_heldout'][:n_train].any() or not z['is_heldout'][n_train:].all():
            raise ValueError('expected train targets first')
        g = {k: z[k][:n_train] for k in ('pb_counts', 'mean_log', 'nonzero', 'pcs') if k in z.files}
        if 'sq_log' in z.files:
            g['sq_log'] = z['sq_log'][:n_train]
        g['targets'] = z['targets'][:n_train]
    return g


def cmd_ganmetrics(a):
    fits = sorted({f for g in a.fits for f in glob.glob(g)})
    out = json.loads(Path(a.out).read_text()) if Path(a.out).exists() else {'fits': [], 'references': {}}
    done = {(r['fit'], r['file']) for r in out['fits']}
    caches = {}
    for f in fits:
        fn = 'generated.npz' if not a.mode else f'generated_{a.mode}.npz'
        p = Path(f) / fn
        if not p.exists() or (str(f), fn) in done:
            continue
        cfg = json.loads((Path(f) / 'config.json').read_text())
        if cfg.get('panel') != 'hvg2000':
            continue
        k = cfg['fold']
        if k not in caches:
            c = Cache(k)
            real = c.donor_stats('train')
            caches[k] = (c, real)
            if str(k) not in out['references']:
                out['references'][str(k)] = {'floors': floors(c, real),
                                             'split_vs_full': score_train(reference_arrays(c, real, 'split'), real, c),
                                             'resample': score_train(reference_arrays(c, real, 'resample'), real, c)}
                print('[ganmetrics] references fold', k, json.dumps(out['references'][str(k)]), flush=True)
        c, real = caches[k]
        g = load_gen_train(p, len(real))
        if list(g['targets']) != [d['donor'] for d in real]:
            raise ValueError('target order differs from cache train donors')
        r = {'fit': str(f), 'file': fn, 'arm': arms.arm_name(tuple(cfg['flags'])), 'fold': k, 'seed': cfg['seed'],
             'round': cfg.get('round', 1), **score_train(g, real, c)}
        out['fits'].append(r)
        print(f'[ganmetrics] {r["arm"]} {fn} fold{k} seed{r["seed"]} pb {r["pb_rmse"]:.3f} ml {r["meanlog_rmse_gene"]:.3f} '
              f'lv {r.get("logvar_rmse_gene", float("nan")):.3f} ce {r["cell_energy"]:.3f}', flush=True)
        dump(a.out, out)
    dump(a.out, out)


# =========================================================================== probe (b)
VARIANTS = {
    'released': {'h': 128, 'zc': 8, 'depth': 1, 'disp': False},
    'h256': {'h': 256, 'zc': 8, 'depth': 1, 'disp': False},
    'h512': {'h': 512, 'zc': 8, 'depth': 1, 'disp': False},
    'zc16': {'h': 128, 'zc': 16, 'depth': 1, 'disp': False},
    'zc32': {'h': 128, 'zc': 32, 'depth': 1, 'disp': False},
    'depth2': {'h': 128, 'zc': 8, 'depth': 2, 'disp': False},
    'depth3': {'h': 128, 'zc': 8, 'depth': 3, 'disp': False},
    'disp': {'h': 128, 'zc': 8, 'depth': 1, 'disp': True},
    'all': {'h': 512, 'zc': 32, 'depth': 3, 'disp': False},
    'nocell': {'h': 128, 'zc': 8, 'depth': 1, 'disp': False, 'nocell': True},
    'zd2': {'h': 128, 'zc': 8, 'depth': 1, 'disp': False, 'zd': 2},
    'zd16': {'h': 128, 'zc': 8, 'depth': 1, 'disp': False, 'zd': 16},
    'nocelldisp': {'h': 128, 'zc': 8, 'depth': 1, 'disp': True, 'nocell': True},
    'h64': {'h': 64, 'zc': 8, 'depth': 1, 'disp': False},
    'h32': {'h': 32, 'zc': 8, 'depth': 1, 'disp': False},
    'cdisp': {'h': 128, 'zc': 8, 'depth': 1, 'disp': False, 'cdisp': True},
    'dispzc32': {'h': 128, 'zc': 32, 'depth': 1, 'disp': True},
    'dispdepth2': {'h': 128, 'zc': 8, 'depth': 2, 'disp': True},
    'dispz': {'h': 128, 'zc': 8, 'depth': 1, 'disp': False, 'dispz': 1.0},   # repair TR-1: latent-only dispersion (g.dispz)
}


def nb_logpmf(x, mu, theta):
    """Gamma-Poisson (negative binomial, mean mu, shape theta) log pmf."""
    return (torch.lgamma(x + theta) - torch.lgamma(theta) - torch.lgamma(x + 1)
            + theta * (torch.log(theta) - torch.log(theta + mu)) + x * (torch.log(mu) - torch.log(theta + mu)))


class Probe(nn.Module):
    def __init__(self, cache, v, seed):
        super().__init__()
        torch.manual_seed(seed)
        cls = arm_garch.make_generator_class(depth=v['depth'], disp=v['disp'], cdisp=v.get('cdisp', False), dispz=v.get('dispz'))
        kw = {'zd': v.get('zd', 8), 'zc': v['zc'], 'h': v['h']}
        if v['depth'] > 1:
            kw['depth'] = v['depth']
        if v['disp']:
            kw['disp'] = True
        if v.get('cdisp'):
            kw['cdisp'] = True
        if v.get('dispz') is not None:
            kw['dispz'] = v['dispz']
        self.G = cls(cache.mu, cache.theta * 1.0, **kw)
        self.zc, self.zd = v['zc'], kw['zd']
        self.nocell = bool(v.get('nocell'))
        g = torch.Generator().manual_seed(seed + 1)
        self.E = nn.Parameter(torch.randn(len(cache.ids), self.zd, generator=g))
        Gn = cache.n_genes
        self.enc = nn.Sequential(nn.Linear(Gn, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 2 * self.zc))
        self.register_buffer('cond', torch.zeros(len(cache.ids), 7))
        self.cond[:, 3] = torch.as_tensor(np.float32(cache.stage / 3.0))
        self.register_buffer('mean', cache.mean.clone(), persistent=False)
        self.register_buffer('sd', cache.sd.clone(), persistent=False)

    def decode(self, z, c, eps):
        """[B] cells with their own donor latent z [B,zd], condition c [B,7], cell latent eps [B,zc] -> (p [B,G], theta)."""
        film, cellh, logit = self.G.film, self.G.cell_hidden, self.G.logits
        s, t = film(z, c)
        h = cellh(eps[:, None, :], s, t, z, c)
        p = torch.softmax(logit(h, z, c), -1).clamp_min(1e-8)[:, 0]
        p = p / p.sum(-1, keepdim=True)
        th = self.G.dispersion(z, c)
        if th.ndim == 3:
            th = th[:, 0]
        return p, th

    def elbo(self, x, d, *, sample=True):
        lib = x.sum(-1, keepdim=True)
        z, c = self.E[d], self.cond[d]
        if self.nocell:
            eps = x.new_zeros(len(x), self.zc)
            kl = x.new_zeros(len(x))
        else:
            q = self.enc((logcp(x) - self.mean) / self.sd)
            m, lv = q.chunk(2, -1)
            lv = lv.clamp(-12, 6)
            eps = m + (torch.randn_like(m) * (0.5 * lv).exp() if sample else 0)
            kl = 0.5 * (m.square() + lv.exp() - 1 - lv).sum(-1)
        p, th = self.decode(z, c, eps)
        ll = nb_logpmf(x, lib * p, th).sum(-1)
        return ll, kl

    @torch.no_grad()
    def generate(self, cache, real, mode, *, seed=SEED):
        """generate.py-like arrays for the train targets: eps ~ N(0,I), libraries from the target's own cells, Gamma-Poisson."""
        k = {'own': 0, 'pop': 1, 'popx': 2, 'popall': 3, 'popallx': 4}[mode]
        rng = np.random.default_rng(np.random.SeedSequence([seed, cache.fold, 51 + k]))
        dev = self.E.device
        tg = torch.Generator(device=dev).manual_seed(seed + 3 + k)
        n = len(real)
        stage = np.array([d['stage'] for d in real])
        arr = {k: np.zeros((n, M_GEN, cache.n_genes), np.float32) for k in ('pb_counts', 'mean_log', 'nonzero', 'logvar')}
        arr['pcs'] = np.zeros((n, M_GEN, PC_CELLS, 50), np.float32)
        for i in range(n):
            libs = rng.choice(real[i]['lib'], size=(M_GEN, K_GEN), replace=True)
            for m in range(M_GEN):
                if mode == 'own':
                    j = i
                elif mode == 'pop':      # legacy (round-3 capacity report): same stage INCLUDING the target itself
                    j = int(rng.choice(np.flatnonzero(stage == stage[i])))
                elif mode == 'popx':     # repair E1: same stage, target EXCLUDED
                    j = int(rng.choice(np.flatnonzero((stage == stage[i]) & (np.arange(n) != i))))
                elif mode == 'popall':   # any train donor's latent (incl. self, p = 1/n), target's condition (GAN-like)
                    j = int(rng.integers(n))
                else:                    # popallx: any OTHER train donor's latent, target's condition
                    j = int(rng.choice(np.flatnonzero(np.arange(n) != i)))
                z = self.E[j].expand(K_GEN, -1)
                c = self.cond[i].expand(K_GEN, -1)
                eps = torch.zeros(K_GEN, self.zc, device=dev) if self.nocell else torch.randn(K_GEN, self.zc, generator=tg, device=dev)
                p, th = self.decode(z, c, eps)
                mu = torch.as_tensor(libs[m], dtype=torch.float32, device=dev)[:, None] * p
                th = th.expand_as(mu)
                rate = torch._standard_gamma(th.contiguous(), generator=tg) * (mu / th)
                x = torch.poisson(rate, generator=tg)
                _fill(arr, i, m, x.cpu(), cache)
        return arr


def cmd_probe(a):
    v = VARIANTS[a.variant]
    cache = Cache(a.fold)
    real = cache.donor_stats('train')
    rng = np.random.default_rng(np.random.SeedSequence([SEED, a.fold, 61]))
    val = np.concatenate([rng.choice(r, max(1, len(r) // 10), replace=False) for r in cache.by])
    is_val = np.zeros(len(cache.X), bool)
    is_val[val] = True
    donor_of = np.zeros(len(cache.X), int)
    for j, r in enumerate(cache.by):
        donor_of[r] = j
    tr_idx = np.flatnonzero(~is_val)
    dev = torch.device(a.device)
    if dev.type == 'cuda':
        if a.steps > 300:
            raise ValueError('GPU probe fits are limited to 300 steps (PROTOCOL_R3 GPU rule)')
        torch.cuda.set_per_process_memory_fraction(a.gpu_mem_fraction)
    model = Probe(cache, v, SEED + a.fold + 1000 * a.seed).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps, eta_min=a.lr / 20)
    bg = np.random.default_rng(np.random.SeedSequence([SEED, a.fold, 62, a.seed]))
    torch.manual_seed(SEED + 7 + 1000 * a.seed)
    t0 = time.time()
    curve = []
    Xd = cache.X.to(dev)
    don = torch.as_tensor(donor_of, device=dev)
    for step in range(a.steps):
        b = torch.as_tensor(bg.choice(tr_idx, a.batch, replace=False), device=dev)
        x, d = Xd[b], don[b]
        ll, kl = model.elbo(x, d)
        loss = -(ll - kl).mean() + 0.5 * model.E.square().sum() / len(tr_idx)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 100 == 0 or step == a.steps - 1:
            with torch.no_grad():
                vb = torch.as_tensor(val, device=dev)
                vll, vkl = model.elbo(Xd[vb], don[vb])
                row = {'step': step, 'train_nelbo_per_cell': float(loss), 'val_nelbo_per_cell': float(-(vll - vkl).mean()),
                       'val_kl_per_cell': float(vkl.mean()), 'seconds': time.time() - t0}
            curve.append(row)
            print(f'[probe {a.variant} f{a.fold}] {json.dumps(row)}', flush=True)
    model.eval()
    res = {'device': str(dev), 'probe_seed': a.seed, 'variant': a.variant, 'arch': v, 'fold': a.fold, 'steps': a.steps, 'batch': a.batch, 'lr': a.lr, 'curve': curve,
           'n_params_generator': int(sum(p.numel() for p in model.G.parameters())),
           'final_val_nelbo_per_cell': curve[-1]['val_nelbo_per_cell'], 'final_val_kl_per_cell': curve[-1]['val_kl_per_cell'],
           'seconds_train': time.time() - t0}
    for mode in ('own', 'pop', 'popx', 'popall', 'popallx'):
        res[mode] = score_train(model.generate(cache, real, mode, seed=SEED), real, cache)
        print(f'[probe {a.variant} f{a.fold}] {mode} {json.dumps(res[mode])}', flush=True)
    E = model.E.detach().cpu().double().numpy()
    ev = np.sort(np.linalg.eigvalsh(np.cov(E.T)))[::-1]
    res['donor_latent'] = {'cov_eigs': ev.tolist(), 'pr': float(ev.sum() ** 2 / (ev ** 2).sum())}
    with torch.no_grad():   # how far the learned dispersion moves from one value per gene (log theta units)
        Gm = model.G
        if getattr(Gm, 'garch_disp', False):
            d = F.linear(torch.cat([model.E, model.cond], 1), Gm.disp_weight, Gm.disp_bias)
            res['dispersion_offsets'] = {'kind': 'donor', 'sd_over_donors_mean_over_genes': float(d.std(0).mean()),
                                         'mean_abs_offset': float(d.abs().mean())}
        if getattr(Gm, 'garch_dispz', None) is not None:
            d = Gm.garch_dispz * F.linear(model.E, Gm.disp_weight, Gm.disp_bias)
            res['dispersion_offsets'] = {'kind': 'donor-latent', 'sd_over_donors_mean_over_genes': float(d.std(0).mean()),
                                         'mean_abs_offset': float(d.abs().mean())}
        if getattr(Gm, 'garch_cdisp', False):
            dev = model.E.device
            eps = torch.randn(len(model.E), 256, model.zc, generator=torch.Generator(device=dev).manual_seed(SEED), device=dev)
            z, c = model.E, model.cond
            s_, t_ = Gm.film(z, c)
            h = Gm.cell_hidden(eps, s_, t_, z, c)
            for layer in getattr(Gm, 'extra', []):
                h = layer(h)
            d = F.linear(F.silu(h), Gm.cdisp_weight, Gm.cdisp_bias)           # [donors, cells, G]
            res['dispersion_offsets'] = {'kind': 'cell', 'sd_within_donor_mean_over_genes': float(d.std(1).mean()),
                                         'sd_of_donor_means_mean_over_genes': float(d.mean(1).std(0).mean()),
                                         'mean_abs_offset': float(d.abs().mean())}
    if a.references:
        res['references'] = {k: score_train(reference_arrays(cache, real, k), real, cache) for k in ('resample', 'resamplex')}
        res['references']['grand_mean_in_sample'] = res['own']['pb_rmse_grand_mean_in_sample']
    dump(a.out, res)


# =========================================================================== report
def cmd_report(a):
    d = Path(a.dir)
    rows = []
    for p in sorted(d.glob('probe_*.json')):
        q = json.loads(p.read_text())
        rows.append((q['fold'], q['variant'], q['final_val_nelbo_per_cell'], q['own'], q['pop']))
    keys = ['pb_rmse', 'meanlog_rmse_dg', 'nonzero_rmse_dg', 'logvar_rmse_dg', 'meanlog_rmse_gene', 'logvar_rmse_gene', 'cell_energy']
    print('fold variant   val_nelbo  | own: ' + ' '.join(k[:14] for k in keys) + ' | pop: ' + ' '.join(k[:14] for k in keys))
    for f, v, nl, o, pp in rows:
        print(f'{f} {v:9s} {nl:9.2f} | ' + ' '.join(f'{o.get(k, float("nan")):.4f}' for k in keys) + ' | ' +
              ' '.join(f'{pp.get(k, float("nan")):.4f}' for k in keys))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = p.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('particles')
    s.add_argument('--fits', nargs='+', required=True)
    s.add_argument('--out', required=True)
    s.add_argument('--cells', type=int, default=512)
    s.add_argument('--invert', action='store_true')
    s.add_argument('--heldout', action='store_true', help='descriptive held-out coverage columns')
    s = sub.add_parser('probe')
    s.add_argument('--fold', type=int, default=0)
    s.add_argument('--variant', choices=sorted(VARIANTS), required=True)
    s.add_argument('--steps', type=int, default=1500)
    s.add_argument('--batch', type=int, default=1024)
    s.add_argument('--lr', type=float, default=2e-3)
    s.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    s.add_argument('--seed', type=int, default=0, help='probe initialisation / batch seed')
    s.add_argument('--gpu-mem-fraction', type=float, default=0.04)
    s.add_argument('--references', action='store_true', help='also score the resample / resamplex real references (CPU, slow)')
    s.add_argument('--out', required=True)
    s = sub.add_parser('ganmetrics')
    s.add_argument('--fits', nargs='+', required=True)
    s.add_argument('--out', required=True)
    s.add_argument('--mode', default=None)
    s = sub.add_parser('report')
    s.add_argument('--dir', default=str(OUT_DIR))
    a = p.parse_args(argv)
    {'particles': cmd_particles, 'probe': cmd_probe, 'ganmetrics': cmd_ganmetrics, 'report': cmd_report}[a.cmd](a)


if __name__ == '__main__':
    main()
