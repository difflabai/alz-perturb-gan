"""Training-only calibration split, row-softmax scoring and a frozen temperature calibrator.

The relativistic critic and the 50/50-trained head have no meaningful absolute level, so
scores are only ever compared within a target row (softmax over the common library).
Probabilities/Brier are reported only after a temperature fitted on held-out FIT identities.
"""
from __future__ import annotations
import math
import torch
from .gene_features import json_hash

CALIBRATION_SEED = 1729
CALIBRATION_COUNT = 16
MIN_FIT_FOR_CALIBRATION = 48   # real cohorts (FIT 96); small synthetic protocols keep no split


def calibration_split(fit, *, seed=CALIBRATION_SEED, count=CALIBRATION_COUNT):
    """Deterministic held-out FIT identities; excluded from G, D and J training."""
    fit = tuple(fit)
    if len(fit) < MIN_FIT_FOR_CALIBRATION: return ()
    perm = torch.randperm(len(fit), generator=torch.Generator().manual_seed(seed))
    return tuple(sorted(fit[i] for i in perm[:count].tolist()))


def split_hash(names, seed=CALIBRATION_SEED):
    return json_hash({'calibration_identities': list(names), 'seed': seed})


def row_softmax(logits, temperature=1.0):
    """Per-target distribution over candidates; invariant to any per-row offset."""
    return torch.softmax(logits.double() / temperature, dim=-1).float()


def relativistic_realness(fake_logits, real_logits):
    """Offset-free gate: sigma(D(generated) - D(matched real)); 0.5 at equilibrium."""
    return torch.sigmoid(fake_logits - real_logits)


def fit_temperature(logits, positives, *, lo=0.05, hi=1e4, iters=200):
    """Minimise held-out NLL of the true candidate over log-temperature (golden-section, 1 parameter)."""
    x = logits.double(); pos = torch.as_tensor(positives)
    def nll(log_t):
        return float(-torch.log_softmax(x / math.exp(log_t), -1)[torch.arange(len(pos)), pos].mean())
    a, b = math.log(lo), math.log(hi); g = (math.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    for _ in range(iters):
        if nll(c) < nll(d): b = d
        else: a = c
        c, d = b - g * (b - a), a + g * (b - a)
    t = math.exp((a + b) / 2)
    return {'method': 'temperature_row_softmax', 'temperature': t, 'nll': nll(math.log(t)),
            'uniform_nll': math.log(x.shape[1]), 'fitted_rows': int(len(pos))}


def calibrator_sha(calibrator):
    return json_hash({k: calibrator[k] for k in sorted(calibrator) if k != 'sha256'})


def reliability(probs, positives, bins=10):
    """Full-library Brier (mean over all cells) and ECE over all candidate probabilities."""
    labels = torch.zeros_like(probs); labels[torch.arange(len(positives)), torch.as_tensor(positives)] = 1.
    flat, y = probs.flatten().double(), labels.flatten().double()
    edges = torch.linspace(0, float(flat.max()) + 1e-12, bins + 1, dtype=torch.double)
    ece, table = 0., []
    for i in range(bins):
        m = (flat >= edges[i]) & ((flat < edges[i + 1]) if i < bins - 1 else (flat <= edges[i + 1]))
        n = int(m.sum())
        if n:
            ece += n / len(flat) * abs(float(flat[m].mean()) - float(y[m].mean()))
        table.append({'lower': float(edges[i]), 'count': n, 'mean_score': float(flat[m].mean()) if n else None,
                      'positive_rate': float(y[m].mean()) if n else None})
    k = probs.shape[1]
    return {'full_brier': float((flat - y).square().mean()), 'ece': ece, 'bins': table,
            'base_rate_brier': (1 / k) * (1 - 1 / k)}
