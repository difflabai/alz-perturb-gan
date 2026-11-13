"""Composable application arms for the improve_20261109 ParticleGAN 0.9.7 trainer.

An arm changes only the *application* around the released update rule: the
generator head (``offset``), what the critic observes (``genemean`` /
``donorpc``) and the conditioning vector (``covars``).  The optimizer, penalty,
loss, policy and prior-regulariser calls are untouched (see trainer.advance).

Arm name = sorted flags joined by commas, ``base`` when there are none.

base
    Released formulation: CountGenerator (output bias = log pooled mean
    proportions, moment theta init), 150 bag features from 50 PCs
    (critic.bag_features), critic.VectorCritic(h=128), stage-only condition
    (stage_law.mask_adnc semantics), uniform donor sampling, Poisson forward
    with straight-through rate features (pg_model.critic_features).

offset
    ``OffsetCountGenerator``: a learnable table ``stage_offset`` [4, G] is
    added to the output logits before the softmax, row selected by the ADNC
    stage round(3*c[:,3]) and applied as ``one_hot(stage, 4) @ stage_offset``
    (forward bitwise equal to row indexing; the matmul backward is
    deterministic on multithreaded CPU, whereas the index backward is a
    scatter-add whose float summation order varies between runs).  It is a
    parameter of the generator module, so it is in the generator optimizer
    group (dense 2-D DualNorm/polar update, as for every 2-D generator
    weight).  Initialisation from TRAIN donors only (``offset_init``):

        C_d   = summed panel counts of donor d,  N_d = sum_g C_dg
        a     = 1 (pseudocount, counts per gene)
        q_g   = (sum_d C_dg + a) / (sum_d N_d + a G)          pooled proportions
        r_dg  = log((C_dg + a)/(N_d + a G)) - log q_g          donor log-ratios
        m_g   = mean over all n train donors of r_dg
        r_sg  = mean_{d in s} r_dg - m_g                       donor-averaged stage effect
        v_g   = pooled within-stage variance of r_dg
                = sum_s sum_{d in s} (r_dg - mean_{d in s} r_dg)^2 / (n - S)
                  (S = number of stages with >= 1 train donor)
        e_s   = 1/n_s - 1/n        (sampling variance of r_sg is v_g e_s under no effect)
        tau2  = max(1e-6, mean over (s with n_s>0, g) of (r_sg^2 - v_g e_s))
                (one global method-of-moments prior variance of true effects)
        w_sg  = tau2 / (tau2 + v_g e_s)                        (0 when n_s = 0)
        offset_sg = w_sg * r_sg

    so the shrinkage toward 0 is stronger for stages with fewer donors and
    for genes with larger between-donor noise.  The table carries stage
    contrasts only (sum_s n_s r_sg = 0); the gene level stays in the released
    output bias (log pooled mean).  Fixer revision (NOTES_fixer.md): the first
    version used the depth-weighted POOLED stage log-ratio for r_sg with
    sampling variance v_g/n_s; donor depth varies (CV ~0.6), so that variance
    was too small and tau2 was as large under permuted stage labels as with
    the real labels.  The donor-averaged r_sg has sampling variance v_g e_s
    exactly under exchangeable donors, so tau2 is ~0 under permutation.

genemean
    Critic input = [150 released features, s_gm * mean_cells(F)] where F is
    the SAME straight-through critic_features tensor [B, cells, G] whose PCA
    projection feeds the 150 features (so gradients reach the generator through
    both blocks); for real bags F = standardize_counts(real counts).  The fixed
    scale s_gm = sqrt(E||f150||^2 / E||mean_cells F||^2) is computed once from
    256 real train bags (128 cells, uniform donors, numpy default_rng(20261109),
    independent of every training stream) and stored in config/checkpoint.
    Critic = ``WideVectorCritic`` (VectorCritic architecture, h=128, wider
    input).

donorpc
    Critic input = [150 released features, ((mean_cells F) - mu) V / sd] where
    P_d = mean over donor d's cells of standardize_counts (train donors only),
    mu = mean_d P_d, V = top-K right singular vectors of (P - mu),
    K = min(16, n_train_donors - 1), sd = population SD (ddof=0) of the train
    donor scores (P - mu) V.  Mutually exclusive with genemean.

covars
    Condition slots: 3 = ADNC/3 (as released), 4 = Male, 5 = APOE4 dose (0/1/2,
    raw as in bench_core.encode_condition), 6 = (age - mean)/sd with train-donor
    population mean/SD (ddof=0, the bench_core.fit_space convention).  Braak,
    Thal, CERAD slots stay 0.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

FLAGS = ('covars', 'donorpc', 'genemean', 'offset')
OFFSET_PSEUDOCOUNT = 1.0
GENEMEAN_REF_BAGS = 256
GENEMEAN_REF_SEED = 20261109
DONORPC_MAX_K = 16


def parse_arm(text):
    """Return sorted flag tuple; 'base' -> ()."""
    if text is None or text == '' or text == 'base':
        return ()
    flags = [f.strip() for f in text.split(',') if f.strip()]
    if 'base' in flags:
        raise ValueError('base cannot be combined with other flags')
    bad = sorted(set(flags) - set(FLAGS))
    if bad:
        raise ValueError(f'unknown arm flags {bad}; known {FLAGS}')
    if len(set(flags)) != len(flags):
        raise ValueError('duplicate arm flag')
    if 'genemean' in flags and 'donorpc' in flags:
        raise ValueError('genemean and donorpc are mutually exclusive')
    return tuple(sorted(flags))


def arm_name(flags):
    return ','.join(sorted(flags)) if flags else 'base'


# --------------------------------------------------------------------------- conditions
def stage_of(raw_c):
    v = np.asarray(raw_c, dtype=np.float64)[..., 3] * 3
    s = np.rint(v).astype(int)
    if np.any((s < 0) | (s > 3)) or not np.allclose(v, s, atol=1e-6, rtol=0):
        raise ValueError('ADNC must be one of four ordinal codes')
    return s


def build_conditions(raw_c, flags, age_mean=None, age_sd=None):
    """Seven-slot condition per donor as the arm conditions."""
    x = np.asarray(raw_c, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 7 or not np.isfinite(x).all():
        raise ValueError('raw_c must be finite [n,7]')
    stage_of(x)
    out = np.zeros_like(x, dtype=np.float32)
    out[:, 3] = x[:, 3].astype(np.float32)  # identical to mask_adnc (float32 cast)
    if 'covars' in flags:
        if age_sd is None or not age_sd > 0:
            raise ValueError('covars arm needs a positive train age SD')
        out[:, 4] = x[:, 4]
        out[:, 5] = x[:, 5]
        out[:, 6] = (x[:, 6] - age_mean) / age_sd
    return out


# --------------------------------------------------------------------------- generator
def _count_generator_cls():
    from pg_model import CountGenerator  # imported lazily (sys.path set by trainer)
    return CountGenerator


def make_offset_generator_class():
    CountGenerator = _count_generator_cls()

    class OffsetCountGenerator(CountGenerator):
        """CountGenerator + learnable stage x gene logit offset (copied parameters_nb).

        Difference from pg_model.CountGenerator.parameters_nb: the logits are
        ``self.out(h) + (one_hot(stage, 4) @ self.stage_offset)[:, None, :]``; everything else
        (noise draw, FiLM, softmax, clamp/renormalise, theta) is identical.
        The table is registered after all CountGenerator parameters, so the
        constructor consumes exactly the same RNG draws as CountGenerator.
        """

        def __init__(self, mean, theta, offset, zd=4, zc=8, h=128):
            super().__init__(mean, theta, zd=zd, zc=zc, h=h)
            offset = torch.as_tensor(offset, dtype=torch.float32)
            if offset.shape != (4, len(mean)):
                raise ValueError('offset table must be [4,G]')
            self.stage_offset = nn.Parameter(offset.clone())

        def parameters_nb(self, z, c, k, library, gen=None):
            scale, shift = self.donor(torch.cat([z, c], 1)).chunk(2, 1)
            noise = torch.randn(len(z), k, self.zc, device=z.device, generator=gen)
            h = self.cell(noise) * (1 + scale[:, None]) + shift[:, None]
            stage = torch.round(c[:, 3] * 3).long()
            # one-hot matmul instead of stage_offset[stage]: identical forward, deterministic backward
            onehot = nn.functional.one_hot(stage, 4).to(self.stage_offset.dtype)
            logits = self.out(h) + (onehot @ self.stage_offset)[:, None, :]
            proportions = torch.softmax(logits, dim=-1).clamp_min(1e-8)
            proportions = proportions / proportions.sum(-1, keepdim=True)
            mu = library.clamp_min(1)[..., None] * proportions
            theta = nn.functional.softplus(self.log_theta) + .05
            return mu, theta

    return OffsetCountGenerator


def offset_init(X, by, stages, pseudocount=OFFSET_PSEUDOCOUNT):
    """Shrunken per-stage pseudobulk log-ratio table [4,G] (train donors only).

    X: cells x G count tensor/array; by: list of row-index arrays per train
    donor; stages: int per train donor.  Formula in the module docstring.
    """
    Xt = torch.as_tensor(X)
    C = torch.stack([Xt[torch.as_tensor(r, device=Xt.device)].sum(0) for r in by]).double().cpu().numpy()
    stages = np.asarray(stages, dtype=int)
    n, G = C.shape
    a = float(pseudocount)
    N = C.sum(1)
    logq = np.log(C.sum(0) + a) - np.log(N.sum() + a * G)
    r_d = np.log(C + a) - np.log(N[:, None] + a * G) - logq
    m = r_d.mean(0)
    n_s = np.array([(stages == s).sum() for s in range(4)])
    present = [s for s in range(4) if n_s[s] > 0]
    r_s = np.zeros((4, G))
    ss = np.zeros(G)
    for s in present:
        msk = stages == s
        r_s[s] = r_d[msk].mean(0) - m
        ss += ((r_d[msk] - r_d[msk].mean(0)) ** 2).sum(0)
    df = n - len(present)
    if df < 1:
        raise ValueError('offset init needs more train donors than stages')
    v = ss / df
    e = np.array([1.0 / n_s[s] - 1.0 / n if n_s[s] else 0.0 for s in range(4)])
    excess = [r_s[s] ** 2 - v * e[s] for s in present]
    tau2 = max(1e-6, float(np.mean(excess)))
    w = np.zeros((4, G))
    for s in present:
        w[s] = tau2 / (tau2 + v * e[s])
    table = (w * r_s).astype(np.float32)
    info = {
        'pseudocount': a, 'tau2': tau2, 'donors_per_stage': n_s.tolist(), 'within_stage_df': int(df),
        'mean_within_stage_variance': float(v.mean()),
        'mean_weight_per_stage': [float(w[s].mean()) if n_s[s] else 0.0 for s in range(4)],
        'raw_rms_per_stage': [float(np.sqrt((r_s[s] ** 2).mean())) for s in range(4)],
        'shrunk_rms_per_stage': [float(np.sqrt((table[s].astype(np.float64) ** 2).mean())) for s in range(4)],
        'formula': 'offset_sg = tau2/(tau2 + v_g (1/n_s - 1/n)) * r_sg, r_sg = donor-averaged stage log-ratio minus '
                   'all-donor average (fixer revision v2); see arms.py docstring',
        'version': 'offset_init-v2-donor-averaged',
    }
    return table, info


# --------------------------------------------------------------------------- critic
class WideVectorCritic(nn.Module):
    """critic.VectorCritic with an arbitrary input width (same layers, h=128)."""

    def __init__(self, n_features, h=128):
        super().__init__()
        self.n_features = int(n_features)
        self.phi = nn.Sequential(nn.Linear(n_features + 7, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU())
        self.rho = nn.Sequential(nn.Linear(h + 7, h), nn.SiLU(), nn.Linear(h, 1))

    def forward(self, x, c):
        if x.ndim != 2 or x.shape[1] != self.n_features or c.shape != (len(x), 7):
            raise ValueError(f'Expected donor vectors [B,{self.n_features}] and conditions [B,7]')
        f = self.phi(torch.cat([x, c], 1))
        return self.rho(torch.cat([f, c], 1)).squeeze(1)


# --------------------------------------------------------------------------- features
class Features:
    """Arm-specific critic observation.

    ``real(idx)`` maps a [B, cells] long tensor of train cell rows to [B, F];
    ``fake(cf)`` maps the straight-through critic_features tensor [B, cells, G]
    to [B, F].  The first 150 columns are always the released bag features.
    """

    def __init__(self, flags, ctx):
        from released_update import bag_features, CONFIG
        if CONFIG.get('fourier_feature_scale', 1.0) != 1.0:
            raise ValueError('released bag feature scale must be 1')
        self.bag_features = bag_features
        self.flags = flags
        self.ctx = ctx
        self.kind = 'genemean' if 'genemean' in flags else 'donorpc' if 'donorpc' in flags else 'base'
        self.W = ctx.W
        self.info = {'kind': self.kind}
        if self.kind == 'base':
            self.n_extra = 0
        elif self.kind == 'genemean':
            self.n_extra = ctx.n_genes
            self.scale = self._genemean_scale()
            self.info.update(genemean_scale=self.scale, reference_bags=GENEMEAN_REF_BAGS,
                             reference_seed=GENEMEAN_REF_SEED)
        else:
            self._fit_donorpc()
        self.n_features = 150 + self.n_extra

    def _extra(self, gene_mean):
        if self.kind == 'genemean':
            return gene_mean * self.scale
        return ((gene_mean - self.pc_mu) @ self.pc_V) / self.pc_sd

    def real(self, idx):
        base = self.bag_features(self.ctx.trainpcs[idx])
        if self.kind == 'base':
            return base
        return torch.cat([base, self._extra(self.ctx.Z[idx].mean(1))], dim=1)

    def fake(self, cf):
        base = self.bag_features(cf @ self.W)
        if self.kind == 'base':
            return base
        return torch.cat([base, self._extra(cf.mean(1))], dim=1)

    @torch.no_grad()
    def _genemean_scale(self):
        ctx = self.ctx
        rng = np.random.default_rng(GENEMEAN_REF_SEED)
        num = den = 0.0
        for _ in range(GENEMEAN_REF_BAGS // 32):
            ds = rng.integers(len(ctx.by), size=32)
            idx = torch.as_tensor(np.stack([rng.choice(ctx.by[i], 128, replace=True) for i in ds]), device=ctx.device)
            num += float(self.bag_features(ctx.trainpcs[idx]).double().square().sum())
            den += float(ctx.Z[idx].mean(1).double().square().sum())
        return float(math.sqrt(num / den))

    @torch.no_grad()
    def _fit_donorpc(self):
        ctx = self.ctx
        P = torch.stack([ctx.Z[torch.as_tensor(r, device=ctx.device)].double().mean(0) for r in ctx.by]).cpu().numpy()
        n = len(P)
        K = min(DONORPC_MAX_K, n - 1)
        mu = P.mean(0)
        U, S, Vt = np.linalg.svd(P - mu, full_matrices=False)
        V = Vt[:K].T
        # Deterministic sign: largest-|loading| gene positive.
        signs = np.sign(V[np.abs(V).argmax(0), np.arange(K)])
        V = V * signs
        scores = (P - mu) @ V
        sd = scores.std(0)
        if not (sd > 0).all():
            raise ValueError('degenerate donor PC')
        dev = ctx.device
        self.pc_mu = torch.as_tensor(mu, dtype=torch.float32, device=dev)
        self.pc_V = torch.as_tensor(V, dtype=torch.float32, device=dev)
        self.pc_sd = torch.as_tensor(sd, dtype=torch.float32, device=dev)
        self.n_extra = K
        self.info.update(K=int(K), explained_variance_fraction=(S[:K] ** 2 / (S ** 2).sum()).tolist(),
                         train_donor_score_sd=sd.tolist())

    def state(self):
        """Tensors to store in the checkpoint for an exact consistency check."""
        if self.kind == 'genemean':
            return {'genemean_scale': self.scale}
        if self.kind == 'donorpc':
            return {'pc_mu': self.pc_mu.cpu(), 'pc_V': self.pc_V.cpu(), 'pc_sd': self.pc_sd.cpu()}
        return {}
