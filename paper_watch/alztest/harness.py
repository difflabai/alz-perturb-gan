"""Shared harness for ALZ paper-watch toy tests.

The small conditional ParticleGAN networks run on CPU by default and support
CUDA. The harness uses public trainer recipes, caller-side condition
concatenation, mean-shift and shuffled-condition controls, and distribution
metrics including energy distance and shift-direction cosine.

The GAN formulation is never touched by this harness or by the tests:
conditioning lives entirely in the caller-supplied networks, exactly the
way the production ALZ adapter wraps the trainer.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass

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


@dataclass(frozen=True)
class TrainingSettings:
    z_dim: int = Z_DIM
    num_particles: int = NUM_PARTICLES
    batch_size: int = BATCH
    d_lr_mult: float | None = None
    prior_lr_mult: float | None = None
    prior_sigma: float = 0.025
    optimizer_smoothing: float | None = None
    betas: tuple[float, float] | None = None
    total_steps: int | None = None
    lr_anneal_start: float | None = None
    lr_floor: float | None = None
    network_lr_floor: float | None = None
    prior_reg: float | None = None
    prior_reg_target_std: float | None = None
    reg_coeff: float | None = None
    rhbi_eps: float | None = None
    rhbi_iters: int | None = None
    rhbi_rho: float | None = None
    rhbi_tau: float | None = None
    rhbi_relativistic_weight: float | None = None
    rhbi_nn_quantile: float | None = None
    rhbi_self_coupling: str | None = None
    generator_output_init_gain: float = 1.0

    def __post_init__(self):
        if not np.isfinite(self.prior_sigma) or self.prior_sigma < 0:
            raise ValueError("prior sigma must be finite and nonnegative")
        if not np.isfinite(self.generator_output_init_gain) or self.generator_output_init_gain <= 0:
            raise ValueError("generator output initialization gain must be finite and positive")


class ConditionalPGAN:
    """One conditional ParticleGAN over (condition -> distribution).

    Conditions are rotated step by step; each step trains on a real batch
    drawn from a single condition, with the condition one-hot fed to both
    networks by the wrappers above.
    """

    def __init__(
        self, data_dim: int, n_cond: int, seed: int = 0, *,
        learning_rate: float | None = None, recipe_name: str = "empf",
        device: str | torch.device = "cpu", use_recipe_defaults: bool = False,
        settings: TrainingSettings | None = None,
    ):
        self.n_cond = n_cond
        self.device = torch.device(device)
        if use_recipe_defaults:
            if learning_rate is not None or settings is not None:
                raise ValueError("explicit training overrides cannot be used with recipe defaults")
            recipe = get_recipe(recipe_name)
        else:
            settings = settings or TrainingSettings()
            overrides = {key: value for key, value in asdict(settings).items()
                         if value is not None and key not in {"prior_sigma", "generator_output_init_gain"}}
            recipe = get_recipe(recipe_name, lr=0.005 if learning_rate is None else learning_rate, **overrides)
        self.G = _CondGenerator(recipe.z_dim, n_cond, data_dim)
        self.D = _CondCritic(data_dim, n_cond)
        deterministic_orthogonal_(self.G, seed=seed)
        deterministic_orthogonal_(self.D, seed=seed + 1)
        if settings is not None and settings.generator_output_init_gain != 1.0:
            with torch.no_grad():
                self.G.net[-1].weight.mul_(settings.generator_output_init_gain)
        self.G.to(self.device)
        self.D.to(self.device)
        prior = None if use_recipe_defaults else recipe.make_prior(sigma=settings.prior_sigma).to(self.device)
        self.trainer = GANTrainer(recipe, self.G, self.D, prior=prior, seed=seed, serial_backward=True)
        self.prior = self.trainer.prior
        self._rng = np.random.default_rng(seed)

    def _set_cond(self, idx: int) -> None:
        self.G.cond.zero_()
        self.G.cond[idx] = 1.0
        self.D.cond.copy_(self.G.cond)

    def fit(
        self, X: np.ndarray, cond: np.ndarray, steps: int = 300, *,
        min_condition_rows: int = 32, after_step: Callable[[int], None] | None = None,
        replay_batches: int = 0,
        on_batch: Callable[[int, np.ndarray], None] | None = None,
    ) -> None:
        if min_condition_rows < 1:
            raise ValueError("min_condition_rows must be positive")
        if replay_batches and replay_batches != self.trainer.completed_steps:
            raise ValueError("batch replay must match the restored trainer's update count")
        by_cond = [np.flatnonzero(cond == k) for k in range(self.n_cond)]
        order = [k for k, indices in enumerate(by_cond) if len(indices) >= min_condition_rows]
        if not order:
            raise ValueError(f"training needs at least {min_condition_rows} rows in one condition")
        data = torch.tensor(X, dtype=torch.float32, device=self.device)
        batch_size = self.trainer.recipe.batch_size
        # A restored trainer has its own RNG state. Replay the inexpensive NumPy
        # draws from the original seed to recover the data cursor and label order.
        for s in range(replay_batches + steps):
            if s % len(order) == 0:
                self._rng.shuffle(order)
            k = order[s % len(order)]
            ix = by_cond[k]
            pick = self._rng.choice(ix, size=batch_size, replace=len(ix) < batch_size)
            if s < replay_batches:
                continue
            if on_batch is not None:
                on_batch(k, pick)
            self._set_cond(k)
            rows = torch.as_tensor(pick, device=self.device)
            self.trainer.step(data.index_select(0, rows))
            if after_step is not None:
                after_step(self.trainer.completed_steps)

    @torch.no_grad()
    def sample(self, cond_idx: int, n: int, seed: int = 999) -> np.ndarray:
        self._set_cond(cond_idx)
        gen = torch.Generator(device=self.device).manual_seed(seed)
        out = self.trainer.sample(n, generator=gen, output_noise=False)
        return out.cpu().numpy().astype(np.float64)


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
