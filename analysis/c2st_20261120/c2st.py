"""Donor-held-out classifier two-sample test (C2ST).

Reusable core: no repository paths, no data loading. Inputs are an item
matrix ``X [n, d]``, binary labels ``y`` (1 = generated / class B), and an
integer donor group per item.  Every item of a donor is held out together
(grouped cross-validation), so a classifier can never score a donor whose
cells or bags it saw during training.  The statistic is the pooled
out-of-fold ROC AUC; 0.5 means indistinguishable.

Uncertainty: donor bootstrap of the out-of-fold scores (no refit), and
paired donor bootstrap for the difference between two C2STs that share the
same real items, donors and folds.  Attribution: per-feature univariate
within-donor AUC and out-of-fold permutation importance of the fitted
classifiers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

try:  # torch is optional; the linear classifier and all statistics are NumPy
    import torch
except ImportError:  # pragma: no cover
    torch = None


# ---------------------------------------------------------------- statistics
def _rankdata(x):
    """Average (mid) ranks, 1-based, vectorized."""
    order = np.argsort(x, kind='mergesort'); s = x[order]
    new = np.r_[True, s[1:] != s[:-1]]
    start = np.flatnonzero(new); end = np.r_[start[1:], len(s)]
    mid = (start + end - 1) / 2 + 1
    r = np.empty(len(s)); r[order] = np.repeat(mid, end - start)
    return r


def auc(score, y):
    """ROC AUC via the Mann-Whitney U statistic with midranks for ties."""
    score = np.asarray(score, np.float64); y = np.asarray(y).astype(bool)
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float('nan')
    r = _rankdata(score)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def within_group_auc(score, y, groups):
    """Mean of per-group AUCs over groups containing both classes."""
    vals = [auc(score[groups == g], y[groups == g]) for g in np.unique(groups)]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float('nan')


def donor_bootstrap(stat, groups, n_boot=2000, seed=0):
    """Resample donors with replacement; ``stat(index)`` gets item indices.

    Returns (estimates array, (2.5%, 97.5%) percentile CI)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in uniq}
    out = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([members[g] for g in pick])
        out.append(stat(idx))
    out = np.asarray(out, np.float64)
    ok = out[np.isfinite(out)]
    ci = (float(np.quantile(ok, .025)), float(np.quantile(ok, .975))) if len(ok) else (float('nan'),) * 2
    return out, ci


def bootstrap_auc(score, y, groups, n_boot=2000, seed=0):
    return donor_bootstrap(lambda i: auc(score[i], y[i]), groups, n_boot, seed)


def paired_bootstrap_diff(score_a, y_a, g_a, score_b, y_b, g_b, n_boot=2000, seed=0):
    """CI for AUC(b) - AUC(a), resampling the shared donors jointly."""
    uniq = np.unique(np.concatenate([g_a, g_b]))
    rng = np.random.default_rng(seed)
    ma = {g: np.flatnonzero(g_a == g) for g in uniq}
    mb = {g: np.flatnonzero(g_b == g) for g in uniq}
    out = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, len(uniq), replace=True)
        ia = np.concatenate([ma[g] for g in pick]); ib = np.concatenate([mb[g] for g in pick])
        out.append(auc(score_b[ib], y_b[ib]) - auc(score_a[ia], y_a[ia]))
    out = np.asarray(out)
    point = auc(score_b, y_b) - auc(score_a, y_a)
    return dict(diff=float(point), ci=[float(np.quantile(out, .025)), float(np.quantile(out, .975))],
                frac_negative=float((out < 0).mean()))


# ------------------------------------------------------------------- folds
def grouped_folds(groups, n_folds, seed=0, strata=None, pairs=None):
    """Assign whole donors to folds.  ``pairs`` (list of donor tuples) fixes
    which donors share a fold; otherwise donors are shuffled within ``strata``
    (dict donor->stratum) and dealt round-robin so strata spread over folds."""
    uniq = np.unique(groups)
    if pairs is not None:
        fold_of = {g: k for k, p in enumerate(pairs) for g in p}
        if set(fold_of) != set(uniq.tolist()):
            raise ValueError('pairs must cover every donor exactly once')
    else:
        rng = np.random.default_rng(seed)
        strata = strata or {g: 0 for g in uniq}
        ordered = []
        for s in sorted(set(strata[g] for g in uniq)):
            members = [g for g in uniq if strata[g] == s]
            ordered += list(rng.permutation(members))
        fold_of = {g: k % n_folds for k, g in enumerate(ordered)}
    return np.array([fold_of[g] for g in groups])


# ------------------------------------------------------------- classifiers
@dataclass
class MLPConfig:
    hidden: tuple = (128, 64)
    epochs: int = 40
    batch: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-3
    dropout: float = .1
    seeds: tuple = (0, 1, 2)
    device: str = 'cpu'
    batched: bool = True     # train every fold x seed network in one vectorized pass


def _standardize(train, *others):
    mu = train.mean(0); sd = train.std(0); sd = np.where(sd > 1e-8, sd, 1.)
    return [(a - mu) / sd for a in (train, *others)]


class _Fitted:
    """Ensemble of fitted MLPs for one fold (same standardization)."""

    def __init__(self, nets, mu, sd, device):
        self.nets, self.mu, self.sd, self.device = nets, mu, sd, device

    def predict(self, X):
        Z = torch.as_tensor(((X - self.mu) / self.sd).astype(np.float32), device=self.device)
        with torch.no_grad():
            p = [torch.sigmoid(n(Z)).squeeze(-1).cpu().numpy() for n in self.nets]
        return np.mean(p, 0)


def fit_mlp(X, y, cfg: MLPConfig):
    if torch is None:
        raise RuntimeError('torch is required for the MLP C2ST')
    mu = X.mean(0); sd = X.std(0); sd = np.where(sd > 1e-8, sd, 1.)
    Z = torch.as_tensor(((X - mu) / sd).astype(np.float32), device=cfg.device)
    T = torch.as_tensor(y.astype(np.float32), device=cfg.device)
    # balance classes within the training fold
    w1 = .5 / max(float(y.mean()), 1e-9); w0 = .5 / max(1 - float(y.mean()), 1e-9)
    W = torch.where(T > .5, torch.tensor(w1, device=cfg.device), torch.tensor(w0, device=cfg.device))
    nets = []
    for s in cfg.seeds:
        g = torch.Generator(device='cpu').manual_seed(int(s))
        torch.manual_seed(int(s))
        layers, d = [], X.shape[1]
        for h in cfg.hidden:
            layers += [torch.nn.Linear(d, h), torch.nn.ReLU(), torch.nn.Dropout(cfg.dropout)]; d = h
        layers += [torch.nn.Linear(d, 1)]
        net = torch.nn.Sequential(*layers).to(cfg.device)
        opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        n = len(Z)
        for _ in range(cfg.epochs):
            net.train()
            perm = torch.randperm(n, generator=g).to(cfg.device)
            for k in range(0, n, cfg.batch):
                b = perm[k:k + cfg.batch]
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    net(Z[b]).squeeze(-1), T[b], weight=W[b])
                opt.zero_grad(); loss.backward(); opt.step()
        net.eval(); nets.append(net)
    return _Fitted(nets, mu, sd, cfg.device)


class _BatchedFold:
    """Seed ensemble of one fold, sharing the batched parameter tensors."""

    def __init__(self, params, k, n_seeds, mu, sd, fold, device):
        self.params, self.k, self.n_seeds, self.mu, self.sd, self.fold, self.device = params, k, n_seeds, mu, sd, fold, device

    def predict(self, X):
        Z = torch.as_tensor(((X - self.mu) / self.sd).astype(np.float32), device=self.device)
        sl = slice(self.k * self.n_seeds, (self.k + 1) * self.n_seeds)
        with torch.no_grad():
            h = Z.unsqueeze(0).expand(self.n_seeds, -1, -1)
            for i, (W, b) in enumerate(self.params):
                h = torch.baddbmm(b[sl].unsqueeze(1), h, W[sl])
                if i < len(self.params) - 1:
                    h = torch.relu(h)
            return torch.sigmoid(h.squeeze(-1)).mean(0).cpu().numpy()


def fit_mlp_batched(X, y, folds, cfg: MLPConfig):
    """Train (fold x seed) MLPs simultaneously with batched matmuls.

    Each network sees only its fold's training donors (standardized with that
    fold's training statistics) and draws its own minibatches; an epoch is
    ``ceil(n_train / batch)`` steps of the largest training set.  Same
    architecture, AdamW, dropout and class balancing as :func:`fit_mlp`."""
    if torch is None:
        raise RuntimeError('torch is required for the MLP C2ST')
    dev = cfg.device
    ks = list(np.unique(folds)); S = len(cfg.seeds); Mn = len(ks) * S
    trs = [np.flatnonzero(folds != k) for k in ks]
    stats = []
    Zs = []
    for tr in trs:
        mu = X[tr].mean(0); sd = X[tr].std(0); sd = np.where(sd > 1e-8, sd, 1.)
        stats.append((mu, sd)); Zs.append(((X - mu) / sd).astype(np.float32))
    Z = torch.as_tensor(np.stack(Zs), device=dev)                      # [F, N, d]
    T = torch.as_tensor(y.astype(np.float32), device=dev)
    gen = torch.Generator(device='cpu').manual_seed(int(cfg.seeds[0]) * 7919 + len(y))
    dims = [X.shape[1], *cfg.hidden, 1]
    params = []
    for a, b in zip(dims[:-1], dims[1:]):
        Ws, bs = [], []
        for k in range(len(ks)):
            for s in cfg.seeds:
                torch.manual_seed(int(s) * 1000 + len(params))
                lin = torch.nn.Linear(a, b)
                Ws.append(lin.weight.detach().T.clone()); bs.append(lin.bias.detach().clone())
        params.append((torch.stack(Ws).to(dev).requires_grad_(), torch.stack(bs).to(dev).requires_grad_()))
    opt = torch.optim.AdamW([t for p in params for t in p], lr=cfg.lr, weight_decay=cfg.weight_decay)
    ntr = max(len(t) for t in trs)
    steps = cfg.epochs * int(np.ceil(ntr / cfg.batch))
    tr_t = [torch.as_tensor(t) for t in trs]
    weights = []
    for t in trs:
        p1 = float(y[t].mean()); weights.append((.5 / max(1 - p1, 1e-9), .5 / max(p1, 1e-9)))
    w0 = torch.tensor([w[0] for w in weights for _ in cfg.seeds], device=dev)[:, None]
    w1 = torch.tensor([w[1] for w in weights for _ in cfg.seeds], device=dev)[:, None]
    fold_of_model = torch.tensor([k for k in range(len(ks)) for _ in cfg.seeds], device=dev)
    # per-model permutation streams
    perms = [None] * Mn; pos = [0] * Mn
    def draw(m):
        t = tr_t[m // S]
        if perms[m] is None or pos[m] + cfg.batch > len(t):
            perms[m] = t[torch.randperm(len(t), generator=gen)]; pos[m] = 0
        b = perms[m][pos[m]:pos[m] + cfg.batch]; pos[m] += cfg.batch
        return b
    for _ in range(steps):
        idx = torch.stack([draw(m) for m in range(Mn)]).to(dev)          # [Mn, B]
        h = Z[fold_of_model[:, None], idx]                                # [Mn, B, d]
        for i, (W, b) in enumerate(params):
            h = torch.baddbmm(b.unsqueeze(1), h, W)
            if i < len(params) - 1:
                h = torch.nn.functional.dropout(torch.relu(h), cfg.dropout, training=True)
        logit = h.squeeze(-1); t = T[idx]
        w = torch.where(t > .5, w1, w0)
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(logit, t, reduction='none') * w).mean(1).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    frozen = [(W.detach(), b.detach()) for W, b in params]
    return [_BatchedFold(frozen, k, S, *stats[k], ks[k], dev) for k in range(len(ks))]


class _Linear:
    def __init__(self, coef, b, mu, sd):
        self.coef, self.b, self.mu, self.sd = coef, b, mu, sd

    def predict(self, X):
        return 1 / (1 + np.exp(-(((X - self.mu) / self.sd) @ self.coef + self.b)))


def fit_linear(X, y, l2=1.0, iters=300):
    """L2 logistic regression by Newton/IRLS-free gradient descent (NumPy)."""
    mu = X.mean(0); sd = X.std(0); sd = np.where(sd > 1e-8, sd, 1.)
    Z = (X - mu) / sd; n, d = Z.shape
    w1 = .5 / max(y.mean(), 1e-9); w0 = .5 / max(1 - y.mean(), 1e-9)
    sw = np.where(y > .5, w1, w0)
    coef = np.zeros(d); b = 0.
    lr = 1. / (np.linalg.norm(Z, 2) ** 2 / n / 4 + l2 / n) if n else 1.
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(Z @ coef + b)))
        r = sw * (p - y)
        coef -= lr * (Z.T @ r / n + l2 * coef / n); b -= lr * r.mean()
    return _Linear(coef, b, mu, sd)


# ------------------------------------------------------------------ C2ST
@dataclass
class C2STResult:
    score: np.ndarray            # out-of-fold P(class 1)
    y: np.ndarray
    groups: np.ndarray
    folds: np.ndarray
    models: list = field(default_factory=list)

    @property
    def auc(self):
        return auc(self.score, self.y)

    def summary(self, n_boot=2000, seed=0, strata=None):
        _, ci = bootstrap_auc(self.score, self.y, self.groups, n_boot, seed)
        out = dict(auc=self.auc, ci=list(ci), within_donor_auc=within_group_auc(self.score, self.y, self.groups),
                   n_items=int(len(self.y)), n_class1=int(self.y.sum()), n_donors=int(len(np.unique(self.groups))))
        if strata is not None:
            per = {}
            for s in sorted(set(strata.values())):
                m = np.isin(self.groups, [g for g, v in strata.items() if v == s])
                if not m.any():
                    continue
                _, c = bootstrap_auc(self.score[m], self.y[m], self.groups[m], n_boot, seed)
                per[str(s)] = dict(auc=auc(self.score[m], self.y[m]), ci=list(c),
                                   n_donors=int(len(np.unique(self.groups[m]))))
            out['per_stratum'] = per
        return out


def run_c2st(X, y, groups, folds, classifier='mlp', cfg: MLPConfig | None = None, l2=1.0, keep_models=False):
    """Grouped cross-fitting: each fold's donors are scored by a model trained
    on all other donors.  ``folds`` comes from :func:`grouped_folds`."""
    X = np.asarray(X, np.float64); y = np.asarray(y).astype(np.float64)
    for k in np.unique(folds):
        if len(np.unique(groups[folds == k])) and np.intersect1d(groups[folds == k], groups[folds != k]).size:
            raise ValueError('a donor appears in both training and test folds')
    score = np.full(len(y), np.nan); models = []
    if classifier == 'mlp' and (cfg or MLPConfig()).batched:
        models = fit_mlp_batched(X, y, folds, cfg or MLPConfig())
        for m in models:
            te = folds == m.fold
            score[te] = m.predict(X[te])
        return C2STResult(score, y.astype(int), np.asarray(groups), np.asarray(folds), models if keep_models else [])
    for k in np.unique(folds):
        te = folds == k; tr = ~te
        if classifier == 'mlp':
            m = fit_mlp(X[tr], y[tr], cfg or MLPConfig())
        elif classifier == 'linear':
            m = fit_linear(X[tr], y[tr], l2)
        else:
            raise ValueError(classifier)
        score[te] = m.predict(X[te])
        models.append(m if keep_models else None)
    return C2STResult(score, y.astype(int), np.asarray(groups), np.asarray(folds), models if keep_models else [])


def permutation_importance(result: C2STResult, X, n_repeats=3, seed=0, blocks=None):
    """Out-of-fold AUC drop when a feature (or block of features) is permuted
    among the test items of its fold.  Requires ``keep_models=True``."""
    rng = np.random.default_rng(seed)
    X = np.asarray(X, np.float64)
    blocks = blocks if blocks is not None else [[j] for j in range(X.shape[1])]
    base = result.auc
    drops = np.zeros(len(blocks))
    folds = np.unique(result.folds)
    for r in range(n_repeats):
        perms = {k: rng.permutation(np.flatnonzero(result.folds == k)) for k in folds}
        for b, cols in enumerate(blocks):
            s = result.score.copy()
            for i, k in enumerate(folds):
                te = np.flatnonzero(result.folds == k)
                Xt = X[te].copy(); Xt[:, cols] = X[perms[k]][:, cols]
                s[te] = result.models[i].predict(Xt)
            drops[b] += (base - auc(s, result.y)) / n_repeats
    return drops


def univariate_auc(X, y, groups):
    """Per-feature within-donor AUC (direction: >0.5 means class 1 larger)."""
    X = np.asarray(X, np.float64)
    out = np.zeros(X.shape[1])
    gs = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    gs = [i for i in gs if 0 < y[i].sum() < len(i)]
    for j in range(X.shape[1]):
        out[j] = np.mean([auc(X[i, j], y[i]) for i in gs])
    return out


def halves_null(X_real, groups_real, strata, n_splits=20, seed=0, classifier='mlp', cfg=None, l2=1.0):
    """Real-vs-real calibration: split donors into two disjoint stratified
    halves, label items by half, and run the same grouped C2ST with each fold
    holding one donor of each half.  Returns the list of pooled AUCs."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups_real)
    aucs = []
    for _ in range(n_splits):
        a, b = [], []
        ordered = []
        for s in sorted(set(strata[g] for g in uniq)):
            ordered += list(rng.permutation([g for g in uniq if strata[g] == s]))
        flip = rng.integers(2)
        for i, g in enumerate(ordered):
            (a if (i + flip) % 2 == 0 else b).append(g)
        if len(a) != len(b):
            # odd donor count: the extra donor joins the last fold
            pass
        pairs = [tuple(p) for p in zip(a, b)]
        extra = a[len(pairs):] + b[len(pairs):]
        if extra:
            pairs[-1] = pairs[-1] + tuple(extra)
        y = np.isin(groups_real, b).astype(int)
        folds = grouped_folds(groups_real, len(pairs), pairs=pairs)
        res = run_c2st(X_real, y, groups_real, folds, classifier, cfg, l2)
        aucs.append(res.auc)
    return aucs


def permutation_null(X, y, groups, folds, n_perm=20, seed=0, classifier='mlp', cfg=None, l2=1.0):
    """Within-donor label permutation: keeps each donor's class balance and
    the fold structure; returns the pooled AUCs of refitted C2STs."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_perm):
        yp = np.asarray(y).copy()
        for g in np.unique(groups):
            i = np.flatnonzero(groups == g); yp[i] = yp[rng.permutation(i)]
        out.append(run_c2st(X, yp, groups, folds, classifier, cfg, l2).auc)
    return out
