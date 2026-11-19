"""Shared harness for ALZ paper-watch toy tests.

Everything here is deliberately small and CPU-bound: a conditional
ParticleGAN (EMPF recipe, company-standard configuration, unmodified)
with caller-side condition concatenation, the two standing control arms
(mean/linear baseline and shuffled-condition training), and distribution
metrics (energy distance, shift-direction cosine).

The GAN formulation is never touched by this harness or by the tests:
conditioning lives entirely in the caller-supplied networks, exactly the
way the production ALZ adapter wraps the trainer.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from particlegan import GANTrainer, get_recipe
from particlegan.init import deterministic_orthogonal_

HIDDEN = 96
Z_DIM = 16
NUM_PARTICLES = 64
BATCH = 64


class _CondGenerator(nn.Module):
    def __init__(self, z_dim: int, n_cond: int, data_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + n_cond, HIDDEN), nn.Tanh(),
            nn.Linear(HIDDEN, HIDDEN), nn.Tanh(),
            nn.Linear(HIDDEN, data_dim),
        )
        self.register_buffer("cond", torch.zeros(n_cond))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        c = self.cond.unsqueeze(0).expand(z.shape[0], -1)
        return self.net(torch.cat([z, c], dim=1))


class _CondCritic(nn.Module):
    def __init__(self, data_dim: int, n_cond: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(data_dim + n_cond, HIDDEN), nn.Tanh(),
            nn.Linear(HIDDEN, HIDDEN), nn.Tanh(),
            nn.Linear(HIDDEN, 1),
        )
        self.register_buffer("cond", torch.zeros(n_cond))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = self.cond.unsqueeze(0).expand(x.shape[0], -1)
        return self.net(torch.cat([x, c], dim=1))


def _onehot(idx: int, n_cond: int) -> torch.Tensor:
    v = torch.zeros(n_cond)
    v[idx] = 1.0
    return v


class ConditionalPGAN:
    """One conditional ParticleGAN over (condition -> distribution).

    Conditions are rotated step by step; each step trains on a real batch
    drawn from a single condition, with the condition one-hot fed to both
    networks by the wrappers above.
    """

    def __init__(self, data_dim: int, n_cond: int, seed: int = 0):
        self.n_cond = n_cond
        self.G = _CondGenerator(Z_DIM, n_cond, data_dim)
        self.D = _CondCritic(data_dim, n_cond)
        deterministic_orthogonal_(self.G, seed=seed)
        deterministic_orthogonal_(self.D, seed=seed + 1)
        recipe = get_recipe(
            "empf", z_dim=Z_DIM, num_particles=NUM_PARTICLES,
            batch_size=BATCH, lr=0.005,
        )
        self.prior = recipe.make_prior(sigma=0.025)
        self.trainer = GANTrainer(recipe, self.G, self.D, prior=self.prior,
                                  seed=seed, serial_backward=True)
        self._rng = np.random.default_rng(seed)

    def _set_cond(self, idx: int) -> None:
        v = _onehot(idx, self.n_cond)
        self.G.cond.copy_(v)
        self.D.cond.copy_(v)

    def fit(
        self, X: np.ndarray, cond: np.ndarray, steps: int = 300, *, min_condition_rows: int = 32
    ) -> None:
        if min_condition_rows < 1:
            raise ValueError("min_condition_rows must be positive")
        by_cond = [np.flatnonzero(cond == k) for k in range(self.n_cond)]
        order = [k for k, indices in enumerate(by_cond) if len(indices) >= min_condition_rows]
        if not order:
            raise ValueError(f"training needs at least {min_condition_rows} rows in one condition")
        for s in range(steps):
            if s % len(order) == 0:
                self._rng.shuffle(order)
            k = order[s % len(order)]
            ix = by_cond[k]
            pick = self._rng.choice(ix, size=BATCH, replace=len(ix) < BATCH)
            self._set_cond(k)
            batch = torch.tensor(X[pick], dtype=torch.float32)
            self.trainer.step(batch)

    @torch.no_grad()
    def sample(self, cond_idx: int, n: int, seed: int = 999) -> np.ndarray:
        self._set_cond(cond_idx)
        gen = torch.Generator().manual_seed(seed)
        out = self.trainer.sample(n, generator=gen, output_noise=False)
        return out.numpy().astype(np.float64)


def fit_unconditional(X: np.ndarray, steps: int = 300, seed: int = 0) -> ConditionalPGAN:
    """A ParticleGAN over a single distribution (condition count 1)."""
    model = ConditionalPGAN(X.shape[1], 1, seed=seed)
    model.fit(X, np.zeros(len(X), dtype=int), steps=steps)
    return model


# ---------------------------------------------------------------- metrics

def energy_distance(a: np.ndarray, b: np.ndarray, max_n: int = 512,
                    seed: int = 0) -> float:
    """Energy distance between two samples (subsampled for speed)."""
    rng = np.random.default_rng(seed)
    if len(a) > max_n:
        a = a[rng.choice(len(a), max_n, replace=False)]
    if len(b) > max_n:
        b = b[rng.choice(len(b), max_n, replace=False)]
    a = torch.tensor(a, dtype=torch.float64)
    b = torch.tensor(b, dtype=torch.float64)
    dab = torch.cdist(a, b).mean()
    daa = torch.cdist(a, a).mean()
    dbb = torch.cdist(b, b).mean()
    return float(2 * dab - daa - dbb)


def shift_cosine(gen_cond: np.ndarray, gen_ctrl: np.ndarray,
                 real_cond: np.ndarray, real_ctrl: np.ndarray) -> float:
    """Cosine between generated and real condition-vs-control mean shifts."""
    g = gen_cond.mean(0) - gen_ctrl.mean(0)
    r = real_cond.mean(0) - real_ctrl.mean(0)
    denom = np.linalg.norm(g) * np.linalg.norm(r)
    if denom == 0:
        return 0.0
    return float(g @ r / denom)


# ---------------------------------------------------------------- baselines

def mean_shift_prediction(ctrl: np.ndarray, train_cond: np.ndarray) -> np.ndarray:
    """Linear/moment baseline: control cells + the train-estimated mean shift.

    This is the moment-matching class of baseline that AMM-style results
    (Genes 2026) show is embarrassingly strong on distribution distances;
    a generative claim that cannot beat it is not a claim.
    """
    shift = train_cond.mean(0) - ctrl.mean(0)
    return ctrl + shift


def pca_fit_transform(X: np.ndarray, n_comp: int = 30, seed: int = 0):
    """Deterministic PCA via SVD on centered data. Returns (projected, parts)."""
    mean = X.mean(0)
    Xc = X - mean
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    comps = vt[:n_comp]
    return Xc @ comps.T, (mean, comps)


def standardize(X: np.ndarray) -> np.ndarray:
    mu, sd = X.mean(0), X.std(0) + 1e-8
    return (X - mu) / sd
