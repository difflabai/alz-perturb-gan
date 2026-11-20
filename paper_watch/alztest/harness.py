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

from collections.abc import Callable

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

    def __init__(
        self, data_dim: int, n_cond: int, seed: int = 0, *, learning_rate: float = 0.005
    ):
        self.n_cond = n_cond
        self.G = _CondGenerator(Z_DIM, n_cond, data_dim)
        self.D = _CondCritic(data_dim, n_cond)
        deterministic_orthogonal_(self.G, seed=seed)
        deterministic_orthogonal_(self.D, seed=seed + 1)
        recipe = get_recipe(
            "empf", z_dim=Z_DIM, num_particles=NUM_PARTICLES,
            batch_size=BATCH, lr=learning_rate,
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
        self, X: np.ndarray, cond: np.ndarray, steps: int = 300, *,
        min_condition_rows: int = 32, after_step: Callable[[int], None] | None = None
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
            if after_step is not None:
                after_step(self.trainer.completed_steps)

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

def energy_distance(a: np.ndarray, b: np.ndarray, max_n: int | None = 512,
                    seed: int = 0, *, unbiased: bool = True) -> float:
    """Energy distance between two samples (subsampled for speed).

    ``unbiased=True`` (the default) uses the U-statistic: the within-sample
    mean distances exclude the zero diagonal, ``sum / (n (n - 1))``. Its
    expectation is 0 when both samples come from one distribution, whatever
    the two sample sizes are. ``unbiased=False`` reproduces the earlier
    V-statistic, whose within-sample means include the diagonal. That
    estimator carries an extra ``E|X - X'| / n`` term per sample, so a
    donor with fewer cells reads as farther away. Recorded results that
    used the V-statistic pin ``unbiased=False``.

    ``max_n=None`` disables subsampling. Subsampling uses one generator
    seeded by ``seed``, so repeated calls draw the same rows.
    """
    rng = np.random.default_rng(seed)
    if max_n is not None:
        if len(a) > max_n:
            a = a[rng.choice(len(a), max_n, replace=False)]
        if len(b) > max_n:
            b = b[rng.choice(len(b), max_n, replace=False)]
    if unbiased and (len(a) < 2 or len(b) < 2):
        raise ValueError("the unbiased energy distance needs at least 2 rows per sample")
    a = torch.tensor(a, dtype=torch.float64)
    b = torch.tensor(b, dtype=torch.float64)
    dab = torch.cdist(a, b).mean()
    if unbiased:
        n, m = len(a), len(b)
        daa = torch.cdist(a, a).sum() / (n * (n - 1))
        dbb = torch.cdist(b, b).sum() / (m * (m - 1))
    else:
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
    """Deterministic PCA via SVD on centered data. Returns (projected, parts).

    This fits the basis on every row of ``X``. When some rows are scored
    as held-out donors, use ``pca_fit`` on the training rows and
    ``pca_apply`` on all rows instead.
    """
    parts = pca_fit(X, n_comp)
    return pca_apply(X, parts), parts


def pca_fit(X_train: np.ndarray, n_comp: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic PCA basis from training rows only. Returns (mean, comps).

    Mirrors the training-donor projection in ``psychad.project_profiles``:
    the centering mean and the SVD come from ``X_train`` alone.
    """
    mean = X_train.mean(0)
    _, _, vt = np.linalg.svd(X_train - mean, full_matrices=False)
    return mean, vt[:n_comp]


def pca_apply(X: np.ndarray, parts: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Project any rows onto a basis from ``pca_fit``."""
    mean, comps = parts
    return (X - mean) @ comps.T


def standardize(X: np.ndarray) -> np.ndarray:
    return standardize_apply(X, standardize_fit(X))


def standardize_fit(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-column (mean, sd) from training rows only."""
    return X_train.mean(0), X_train.std(0) + 1e-8


def standardize_apply(X: np.ndarray, parts: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    mu, sd = parts
    return (X - mu) / sd


# ---------------------------------------------------------------- inference

def _group_codes(groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    names, codes = np.unique(np.asarray(groups).astype(str), return_inverse=True)
    return names, codes


def grouped_permutations(labels, groups=None, *, within=None, n_perm: int = 2000,
                         seed: int = 0):
    """Yield label vectors permuted at the level of whole groups (donors).

    With ``groups=None`` each unit is permuted on its own. With ``groups``
    every unit of one donor receives the labels of one other donor, so the
    records of a donor (for example its AnG and LEC rows) stay together.

    If ``labels`` are constant within each group, donors are exchanged
    freely. If they differ within a group (for example a per-region
    target), pass ``within`` (for example the region of each unit): a donor
    is then exchanged only with donors that have the same set of ``within``
    keys, and unit (d, k) receives the label of unit (pi(d), k). Each
    (group, within) pair must identify one unit.
    """
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    if groups is None:
        for _ in range(n_perm):
            yield labels[rng.permutation(len(labels))]
        return
    _, codes = _group_codes(groups)
    n_groups = codes.max() + 1
    if within is None:
        _, first = np.unique(codes, return_index=True)
        group_label = labels[first]
        if not np.array_equal(group_label[codes], labels):
            raise ValueError(
                "labels differ within a group; pass `within` to permute per-key blocks")
        for _ in range(n_perm):
            yield group_label[rng.permutation(n_groups)][codes]
        return
    within = np.asarray(within).astype(str)
    lookup = {}
    for i, (g, k) in enumerate(zip(codes, within)):
        if (g, k) in lookup:
            raise ValueError("each (group, within) pair must identify one unit")
        lookup[g, k] = i
    pattern = {}
    for g in range(n_groups):
        pattern.setdefault(tuple(sorted(k for (h, k) in lookup if h == g)), []).append(g)
    strata = [np.array(members) for members in pattern.values()]
    for _ in range(n_perm):
        target = np.arange(n_groups)
        for members in strata:
            target[members] = members[rng.permutation(len(members))]
        source = np.array([lookup[target[g], k] for g, k in zip(codes, within)])
        yield labels[source]


def permutation_pvalue(stat_fn: Callable[[np.ndarray], float], labels, groups=None, *,
                       within=None, n_perm: int = 2000, seed: int = 0,
                       alternative: str = "greater") -> dict:
    """Monte Carlo permutation p-value for ``stat_fn(labels)``.

    Labels are permuted by ``grouped_permutations`` (whole donors when
    ``groups`` is given). The p-value counts the observed statistic as one
    of the permutations, ``(1 + #{null >= observed}) / (n_perm + 1)``, so it
    is never 0 and is valid at any ``n_perm``. ``alternative`` is
    "greater", "less" or "two-sided" (absolute values about 0).

    A non-finite observed statistic raises ValueError: every comparison
    with NaN is False, so it would otherwise give the smallest possible
    p-value. A non-finite null statistic is counted as a hit (the
    conservative choice) and reported in ``n_nonfinite_null``; the null
    summaries use the finite values only.
    """
    if alternative not in ("greater", "less", "two-sided"):
        raise ValueError(f"unknown alternative {alternative!r}")
    observed = float(stat_fn(np.asarray(labels)))
    if not np.isfinite(observed):
        raise ValueError(f"observed statistic is not finite ({observed})")
    null = np.array([float(stat_fn(perm)) for perm in grouped_permutations(
        labels, groups, within=within, n_perm=n_perm, seed=seed)])
    finite = np.isfinite(null)
    if alternative == "greater":
        hits = np.sum(null[finite] >= observed - 1e-12)
    elif alternative == "less":
        hits = np.sum(null[finite] <= observed + 1e-12)
    else:
        hits = np.sum(np.abs(null[finite]) >= abs(observed) - 1e-12)
    hits += int(np.sum(~finite))
    return {
        "observed": observed,
        "p": float((1 + hits) / (n_perm + 1)),
        "null_mean": float(null[finite].mean()) if finite.any() else None,
        "null_q95": float(np.quantile(null[finite], 0.95)) if finite.any() else None,
        "n_perm": int(n_perm),
        "n_nonfinite_null": int(np.sum(~finite)),
    }


def donor_bootstrap_ci(stat_fn: Callable[[np.ndarray], float], groups, *,
                       n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> dict:
    """Percentile bootstrap CI that resamples whole donors.

    Each replicate draws donors with replacement and passes ``stat_fn`` the
    unit indices of every drawn donor (a donor drawn twice contributes its
    units twice), so all records of a donor move together. Replicates where
    ``stat_fn`` returns None or a non-finite value (for example an AUROC
    with only one class drawn) are dropped and counted.
    """
    _, codes = _group_codes(groups)
    members = [np.flatnonzero(codes == g) for g in range(codes.max() + 1)]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        pick = rng.integers(len(members), size=len(members))
        value = stat_fn(np.concatenate([members[g] for g in pick]))
        if value is not None and np.isfinite(value):
            values.append(float(value))
    values = np.array(values)
    if not len(values):
        raise ValueError("no bootstrap replicate produced a finite statistic")
    lo, hi = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    return {"lo": float(lo), "hi": float(hi), "n_valid": int(len(values)),
            "n_boot": int(n_boot), "n_donors": len(members)}
