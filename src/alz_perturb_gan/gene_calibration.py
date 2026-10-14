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
    return torch.softmax(logits.double() / temperature, dim=-1)  # float64: huge T must not create rank ties


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


# ---- plan v4 §3: out-of-fold reporting, cap / near-uniform flags, exact thresholds ----
T_LO, T_HI = 0.05, 1e4
LOWER_HIT_FACTOR, UPPER_HIT_FACTOR = 1.05, 0.5      # T <= 1.05*lo or T >= 0.5*hi counts as a cap hit
NEAR_UNIFORM_SPREAD = 0.05                           # median row (max-min)/T below this => near-uniform
BRIER_FACTOR, NLL_MARGIN = 0.995, 0.01
FOLDS, FOLD_SEED = 4, 1729


def cap_hits(t):
    return {'lower': bool(t <= T_LO * LOWER_HIT_FACTOR), 'upper': bool(t >= T_HI * UPPER_HIT_FACTOR)}


def near_uniform(logits, t):
    spread = (logits.double().max(-1).values - logits.double().min(-1).values) / t
    return {'median_scaled_spread': float(spread.median()), 'near_uniform': bool(float(spread.median()) < NEAR_UNIFORM_SPREAD)}


def _nll(probs, positives):
    return float(-torch.log(probs[torch.arange(len(positives)), torch.as_tensor(positives)].clamp_min(1e-300)).mean())


def fold_assignment(n, *, folds=FOLDS, seed=FOLD_SEED):
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    return [sorted(perm[i::folds]) for i in range(folds)]


def crossfit_report(logits, positives, names):
    """4-fold OOF evaluation of the *procedure* (fit T on 12, score 4). Does not validate the all-16 T.

    16 dependent targets, not 2400 independent pairs: descriptive uncertainty only.
    """
    pos = torch.as_tensor(positives); n, k = logits.shape
    folds = fold_assignment(n); oof = torch.empty(n, k, dtype=torch.float64); per_fold = []
    for f, held in enumerate(folds):
        train = [i for i in range(n) if i not in held]
        cal = fit_temperature(logits[train], pos[train])
        oof[held] = row_softmax(logits[held], cal['temperature'])
        rel = reliability(oof[held], pos[held])
        per_fold.append({'fold': f, 'held_out': [names[i] for i in held], 'temperature': cal['temperature'],
                         'cap_hits': cap_hits(cal['temperature']), 'brier': rel['full_brier'], 'ece': rel['ece'],
                         'nll': _nll(oof[held], pos[held])})
    final = fit_temperature(logits, pos); final_probs = row_softmax(logits, final['temperature'])
    base = (1 / k) * (1 - 1 / k)
    o_rel, i_rel = reliability(oof, pos), reliability(final_probs, pos)
    report = {
        'unit': f'{n} dependent calibration targets (not {n * k} independent pairs); descriptive only',
        'fold_assignment_sha256': json_hash({'folds': [[names[i] for i in f] for f in folds], 'seed': FOLD_SEED}),
        'per_fold': per_fold,
        'fold_temperature_range': [min(p['temperature'] for p in per_fold), max(p['temperature'] for p in per_fold)],
        'oof': {'brier': o_rel['full_brier'], 'ece': o_rel['ece'], 'nll': _nll(oof, pos), 'bins': o_rel['bins'],
                'note': 'evaluates the fit-on-12 procedure; not an independent test of the all-16 temperature'},
        'final_all16': {'temperature': final['temperature'], 'cap_hits': cap_hits(final['temperature']),
                        **near_uniform(logits, final['temperature'])},
        'in_sample': {'brier': i_rel['full_brier'], 'ece': i_rel['ece'], 'nll': _nll(final_probs, pos),
                      'label': 'in_sample (all-16 fit scored on itself); never evidence'},
        'thresholds': {'base_brier': base, 'brier_cutoff': base * BRIER_FACTOR, 'nll_cutoff': math.log(k) - NLL_MARGIN,
                       'near_uniform_spread': NEAR_UNIFORM_SPREAD, 't_bounds': [T_LO, T_HI],
                       'cap_hit_rule': 'T <= 1.05*lo or T >= 0.5*hi'},
    }
    any_cap = any(p['cap_hits']['lower'] or p['cap_hits']['upper'] for p in per_fold) or any(report['final_all16']['cap_hits'].values())
    checks = {'no_cap_hit': not any_cap, 'not_near_uniform': not report['final_all16']['near_uniform'],
              'oof_brier_ok': report['oof']['brier'] <= report['thresholds']['brier_cutoff'],
              'oof_nll_ok': report['oof']['nll'] <= report['thresholds']['nll_cutoff']}
    report['checks'] = checks
    report['status'] = 'informative' if all(checks.values()) else 'did not meet prespecified informativeness thresholds'
    return report, final
