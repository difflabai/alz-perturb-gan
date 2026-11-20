"""Shared NumPy helpers: observed donor summaries, generated replicate summaries, calibration rule.

Nothing here imports torch or loads weights. Definitions follow PROTOCOL.md.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path('/ml2/alz')
CACHE = ROOT / 'runs/improve_20261109/caches/hvg2000/fold0'
RUNS = ROOT / 'runs/research_20261120/query-tool'
STAGES = ['Not AD', 'Low', 'Intermediate', 'High']
TYPES = ['Micro-PVM_1', 'Micro-PVM_2', 'Micro-PVM_2_1-SEAAD', 'Micro-PVM_2_3-SEAAD',
         'Micro-PVM_3-SEAAD', 'Micro-PVM_4-SEAAD']
GROUPS = [2, 4]                  # column k+1 = type k: Micro-PVM_2, Micro-PVM_2_3-SEAAD
MIN_SYN_CELLS = 100
MIN_REAL_CELLS = 20
MIN_REAL_DONORS = 3
K = 256
REPEATS = 24
ALPHA = .2
PILOT = ROOT / 'analysis/coverage_empf_20261117/completion/comparison.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def donor_seed(*parts):
    digest = hashlib.sha256('|'.join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], 'little')


def log_cp10k(X):
    lib = X.sum(1, keepdims=True)
    return np.log1p(X * (1e4 / np.maximum(lib, 1)))


def bag_summary(L):
    """(mean, SD with ddof=1) over the cells of a bag, per gene."""
    m = L.mean(0)
    sd = L.std(0, ddof=1) if len(L) > 1 else np.zeros(L.shape[1])
    return m, sd


def load_observed(cache=CACHE, bags=True, empirical_bags=32):
    """Per development donor summaries from the frozen fold cache (observed subtype labels)."""
    import scipy.sparse as sp
    cache = Path(cache)
    manifest = json.loads((cache / 'manifest.json').read_text())
    space = np.load(cache / 'space.npz', allow_pickle=False)
    genes = space['genes'].astype(str)
    out = dict(donors=[], stage=[], is_heldout=[], mean_log=[], detection=[], subtype_sums=[], subtype_n=[],
               bag_mean=[], bag_sd=[], n_cells=[])
    emp_mean, emp_sd = [], []
    for split in ('train', 'heldout'):
        C = sp.load_npz(cache / split / 'counts.npz').tocsr()
        meta = np.load(cache / split / 'metadata.npz', allow_pickle=False)
        if not np.array_equal(meta['genes'].astype(str), genes):
            raise ValueError('gene axis differs')
        cell_donor = meta['donor'].astype(str)
        types = meta['type'].astype(int)
        for j, d in enumerate(meta['donors'].astype(str)):
            rows = np.flatnonzero(cell_donor == d)
            X = C[rows].toarray().astype(np.float64)
            L = log_cp10k(X)
            stage = int(round(float(meta['raw_c'][j][3]) * 3))
            if stage != manifest['cells_per_donor'][split][d]['stage']:
                raise ValueError('stage encoding differs from the manifest')
            sums = np.zeros((7, len(genes)))
            n = np.zeros(7, np.int64)
            sums[0], n[0] = L.sum(0), len(L)
            for t in range(len(TYPES)):
                keep = types[rows] == t
                sums[t + 1], n[t + 1] = L[keep].sum(0), int(keep.sum())
            out['donors'].append(d); out['stage'].append(stage); out['is_heldout'].append(split == 'heldout')
            out['mean_log'].append(L.mean(0)); out['detection'].append((X > 0).mean(0))
            out['subtype_sums'].append(sums); out['subtype_n'].append(n); out['n_cells'].append(len(L))
            if bags:
                bm, bs = [], []
                for r in range(REPEATS):
                    rng = np.random.default_rng(donor_seed('query-tool-bag', 20261120, d, r))
                    pick = rng.choice(len(L), size=min(K, len(L)), replace=False)
                    m, s = bag_summary(L[pick]); bm.append(m); bs.append(s)
                out['bag_mean'].append(np.stack(bm)); out['bag_sd'].append(np.stack(bs))
                if split == 'train' and empirical_bags:
                    for r in range(empirical_bags):
                        rng = np.random.default_rng(donor_seed('query-tool-empirical', 20261120, d, r))
                        m, s = bag_summary(L[rng.choice(len(L), size=K, replace=True)])
                        emp_mean.append(m); emp_sd.append(s)
    result = {k: np.asarray(v) for k, v in out.items() if v}
    result['donors'] = np.asarray(out['donors'])
    result['genes'] = genes
    result['space_mean'] = space['mean'].astype(np.float64)
    result['space_sd'] = space['sd'].astype(np.float64)
    result['W'] = space['W'].astype(np.float64)
    result['train_donors'] = np.asarray(manifest['train_donors'])
    if emp_mean:
        result['empirical_mean'] = np.stack(emp_mean); result['empirical_sd'] = np.stack(emp_sd)
    return result


def generated_sd(mean_log, sq_log, k=K):
    m = np.asarray(mean_log, np.float64)
    var = np.maximum(np.asarray(sq_log, np.float64) - m * m, 0) * (k / (k - 1))
    return np.sqrt(var)


# ----------------------------------------------------------------------------- contrasts
def synthetic_stage_means(subtype_sums, subtype_n, stage, groups=GROUPS, min_cells=MIN_SYN_CELLS):
    """Per stage, equal-weight mean over ``groups`` of the pooled-cell subtype means.

    subtype_sums [T,7,G], subtype_n [T,7], stage [T]. Returns ([4,G] with NaN for invalid stages,
    per-stage cell counts [4,len(groups)])."""
    G = subtype_sums.shape[-1]
    out = np.full((4, G), np.nan)
    counts = np.zeros((4, len(groups)), np.int64)
    for s in range(4):
        keep = stage == s
        sums = subtype_sums[keep][:, groups].sum(0)
        n = subtype_n[keep][:, groups].sum(0)
        counts[s] = n
        if keep.any() and (n >= min_cells).all():
            out[s] = (sums / n[:, None]).mean(0)
    return out, counts


def observed_stage_means(subtype_sums, subtype_n, stage, groups=GROUPS, min_cells=MIN_REAL_CELLS,
                         min_donors=MIN_REAL_DONORS):
    """Equal donor weight within subtype/stage, equal subtype weight; NaN where a group lacks donors."""
    G = subtype_sums.shape[-1]
    out = np.full((4, G), np.nan)
    eligible = np.zeros((4, len(groups)), np.int64)
    for s in range(4):
        values = []
        for gi, g in enumerate(groups):
            keep = (stage == s) & (subtype_n[:, g] >= min_cells)
            eligible[s, gi] = int(keep.sum())
            if keep.sum() >= min_donors:
                values.append((subtype_sums[keep, g] / subtype_n[keep, g, None]).mean(0))
        if len(values) == len(groups):
            out[s] = np.mean(values, axis=0)
    return out, eligible


def within_contrast_observed(subtype_sums, subtype_n, stage, groups=GROUPS, min_cells=MIN_REAL_CELLS,
                             min_donors=MIN_REAL_DONORS):
    """0.9.7 rule: average over eligible groups (>=min_donors in BOTH Not AD and High) of High-minus-Not-AD."""
    effects, used = [], []
    for g in groups:
        keep = subtype_n[:, g] >= (1 if g == 0 else min_cells)
        if min(int((keep & (stage == s)).sum()) for s in (0, 3)) < min_donors:
            continue
        means = subtype_sums[:, g] / np.maximum(subtype_n[:, g, None], 1)
        effects.append(means[keep & (stage == 3)].mean(0) - means[keep & (stage == 0)].mean(0))
        used.append(g)
    if not effects:
        return None, used
    return np.mean(effects, axis=0), used


def bootstrap_observed_contrast(subtype_sums, subtype_n, stage, groups, n_boot=2000, seed=20261120,
                                min_cells=MIN_REAL_CELLS):
    """Stage-stratified donor bootstrap of the observed contrast for fixed groups. Returns [n_valid,G]."""
    rng = np.random.default_rng(seed)
    idx0, idx3 = np.flatnonzero(stage == 0), np.flatnonzero(stage == 3)
    draws = []
    for _ in range(n_boot):
        pick = np.concatenate([rng.choice(idx0, len(idx0)), rng.choice(idx3, len(idx3))])
        eff = []
        for g in groups:
            keep = subtype_n[pick, g] >= (1 if g == 0 else min_cells)
            st = stage[pick]
            if not (keep & (st == 0)).any() or not (keep & (st == 3)).any():
                break
            means = subtype_sums[pick, g] / np.maximum(subtype_n[pick, g, None], 1)
            eff.append(means[keep & (st == 3)].mean(0) - means[keep & (st == 0)].mean(0))
        if len(eff) == len(groups):
            draws.append(np.mean(eff, axis=0))
    return np.asarray(draws)


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return None
    from scipy.stats import rankdata  # average ranks for ties
    ra, rb = rankdata(a[ok]), rankdata(b[ok])
    return float(np.corrcoef(ra, rb)[0, 1])


def direction_agreement(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    return float((np.sign(a[ok]) == np.sign(b[ok])).mean()) if ok.any() else None


# ----------------------------------------------------------------------------- calibration (pilot rule)
def interval_score(low, high, y, alpha=ALPHA):
    return high - low + (2 / alpha) * (np.maximum(low - y, 0) + np.maximum(y - high, 0))


def fit_correction(low, high, observed, donors, training_donors, alpha=ALPHA):
    """Coverage pilot's nonnegative split-conformal rule on arbitrary coordinates.

    low/high/observed [donor, coordinate]; one observed bag per calibration donor."""
    low, high, observed = (np.asarray(x, float) for x in (low, high, observed))
    donors = [str(d) for d in donors]
    if low.shape != high.shape or low.shape != observed.shape or low.ndim != 2:
        raise ValueError('expected donor by coordinate arrays')
    if len(set(donors)) != len(donors) or len(donors) != len(low):
        raise ValueError('one unique identity per calibration donor')
    if set(donors) & {str(d) for d in training_donors}:
        raise ValueError('calibration donor was used in GAN training')
    n = len(donors)
    rank = math.ceil((n + 1) * (1 - alpha))
    if n == 0 or rank > n:
        raise ValueError('insufficient calibration donors for a finite correction')
    score = np.maximum(0, np.maximum(low - observed, observed - high))
    return dict(method='nonnegative_donor_split_CQR_pilot_rule', alpha=alpha, rank=rank,
                n_calibration_donors=n, calibration_donors=donors,
                correction=np.sort(score, axis=0)[rank - 1])


def apply_correction(low, high, correction, nonnegative):
    lo = np.asarray(low, float) - correction
    hi = np.asarray(high, float) + correction
    if nonnegative:
        lo = np.maximum(lo, 0)
    return lo, hi


def pilot_halves():
    swaps = json.loads(PILOT.read_text())['rows']['baseline']['swaps']
    return [(s['calibration_donors'], s['evaluation_donors']) for s in swaps]
