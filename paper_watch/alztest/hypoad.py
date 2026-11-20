"""Data handling and donor-level statistics for the HypoAD microglia readout.

No GAN code lives here. The readout test (tests/test_hypoad_microglia_readout.py)
uses these helpers to

* choose the analysed donors, with the paper's exclusions applied before
  grouping: the familial APOE4/4 donor AD_44 and the young reference donors;
* fit gene selection, the module-gene z-scoring, PCA and the
  standardization of the components on the fit-control cells only, then
  apply them unchanged to every cell, so held-out control and AD cells
  never shape the basis they are scored in;
* attach donor-level statistics to an AUROC: the exact label-permutation
  p-value over scored donors and a stratified donor-bootstrap interval.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy import sparse
from scipy.stats import rankdata

# Paper's exclusions. AD_44 is the familial APOE4/4 donor that Yu et al.
# leave out of their differential analyses. Matching ignores case and
# punctuation ("AD_44", "AD-44" and "ad44" are the same donor).
EXCLUDE_DONORS = frozenset({"AD_44"})

HOMEO = frozenset({"HEXB", "CST3", "CX3CR1", "CSF1R", "CTSS", "TMSB4X", "P2RY12", "C1QB"})
DAM = frozenset({"TYROBP", "CTSB", "APOE", "B2M", "FTH1", "TREM2",
                 "AXL", "CST7", "CTSL", "LPL", "CD9", "CSF1", "ITGAX", "CLEC7A",
                 "LILRB4", "TIMP2"})

DONOR_COLS = ("donor", "donor_id", "sample", "Sample", "individual", "Donor")
COND_COLS = ("condition", "diagnosis", "group", "Group", "disease", "status")

N_TOP_GENES = 1500
N_COMP = 30
EXACT_LIMIT = 200_000


def _norm_id(value: str) -> str:
    return re.sub(r"[^0-9a-z]", "", str(value).lower())


def classify_condition(cond_raw) -> np.ndarray:
    """Map condition strings to AD / CTRL / OTHER; young reference donors are OTHER."""
    out = []
    for value in np.asarray(cond_raw, dtype=str):
        low = value.lower()
        if "young" in low:
            out.append("OTHER")
        elif re.search(r"non[-_ ]?(ad|alzheimer)", low):
            out.append("CTRL")
        elif re.search(r"(^|[^a-z])ad([^a-z]|$)", low) or "alzheimer" in low:
            out.append("AD")
        elif ("control" in low or "healthy" in low or "normal" in low
              or re.search(r"(^|[^a-z])(ctl|ctrl|hc|nc)([^a-z]|$)", low)):
            out.append("CTRL")
        else:
            out.append("OTHER")
    return np.array(out, dtype=object)


def select_donors(donors, cond_raw, exclude=EXCLUDE_DONORS):
    """Apply donor exclusions, then group the remaining cells.

    Returns (keep_mask, group) over all cells. Raises ValueError when an
    excluded donor is not present, because the exclusion is a stated design
    property and a renamed donor would otherwise stay in the AD group silently.
    """
    donors = np.asarray(donors, dtype=str)
    normalized = np.array([_norm_id(d) for d in donors])
    excluded = np.zeros(len(donors), dtype=bool)
    for name in exclude:
        hit = normalized == _norm_id(name)
        if not hit.any():
            raise ValueError(
                f"excluded donor {name!r} not found; donors are {sorted(set(donors))}")
        excluded |= hit
    group = classify_condition(cond_raw)
    group[excluded] = "OTHER"
    return group != "OTHER", group


def check_groups(donors, group, cond_raw, min_ctrl: int = 3, min_ad: int = 2) -> None:
    """Fail clearly when grouping left too few donors for the readout.

    The readout fits on at least two control donors and scores the rest, so
    it needs at least three CTRL donors and two AD donors. When the
    condition labels are spelled in a way classify_condition does not
    recognize, this lists the raw labels per donor instead of letting a
    later sampling call fail with an opaque error.
    """
    donors = np.asarray(donors, dtype=str)
    group = np.asarray(group, dtype=object)
    cond_raw = np.asarray(cond_raw, dtype=str)
    n_ctrl = len(np.unique(donors[group == "CTRL"]))
    n_ad = len(np.unique(donors[group == "AD"]))
    if n_ctrl >= min_ctrl and n_ad >= min_ad:
        return
    pairs = sorted(set(zip(cond_raw.tolist(), group.tolist())))
    per_label = {f"{c!r}->{g}": len(np.unique(donors[(cond_raw == c) & (group == g)]))
                 for c, g in pairs}
    raise ValueError(
        f"too few donors after grouping: {n_ctrl} CTRL (need {min_ctrl}), "
        f"{n_ad} AD (need {min_ad}). Donors per raw condition label: {per_label}")


def find_column(columns, candidates, what: str) -> str:
    for c in candidates:
        if c in columns:
            return c
    raise KeyError(f"no {what} column among {list(columns)}; expected one of {candidates}")


def log_normalize(X):
    """CP10k + log1p for raw counts; values that are not integers are returned as is."""
    X = sparse.csr_matrix(X, dtype=np.float64)
    if X.nnz and np.allclose(X.data, np.round(X.data)):
        lib = np.asarray(X.sum(1)).ravel() + 1e-9
        X = sparse.diags(1e4 / lib) @ X
        X.data = np.log1p(X.data)
    return X.tocsr()


def _col_var(X) -> np.ndarray:
    if sparse.issparse(X):
        mean = np.asarray(X.mean(0)).ravel()
        sq = np.asarray(X.multiply(X).mean(0)).ravel()
        return sq - mean ** 2
    return np.asarray(X).var(0)


def _dense_cols(X, cols) -> np.ndarray:
    sub = X[:, cols]
    return np.asarray(sub.toarray() if sparse.issparse(sub) else sub, dtype=np.float64)


@dataclass(frozen=True)
class FeatureTransform:
    """Everything that is estimated from data, all of it from fit cells."""

    cols: np.ndarray        # selected gene indices into the full matrix, sorted
    mod_pos: np.ndarray     # positions of the module genes within cols
    gene_mu: np.ndarray     # per selected gene, for the module-gene z-scores
    gene_sd: np.ndarray
    pca_mean: np.ndarray
    comps: np.ndarray       # (n_comp, len(cols))
    proj_mu: np.ndarray
    proj_sd: np.ndarray


def fit_feature_transform(X, fit_mask, mod_idx, n_top: int = N_TOP_GENES,
                          n_comp: int = N_COMP) -> FeatureTransform:
    fit_mask = np.asarray(fit_mask, dtype=bool)
    mod_idx = np.asarray(sorted(set(int(i) for i in mod_idx)), dtype=int)
    Xf = X[np.flatnonzero(fit_mask)]
    var = _col_var(Xf)
    order = np.argsort(-var, kind="stable")
    top = [int(i) for i in order if i not in set(mod_idx)][:n_top]
    cols = np.sort(np.array(top + mod_idx.tolist(), dtype=int))
    D = _dense_cols(Xf, cols)
    gene_mu, gene_sd = D.mean(0), D.std(0) + 1e-9
    pca_mean = D.mean(0)
    _, _, vt = np.linalg.svd(D - pca_mean, full_matrices=False)
    comps = vt[:n_comp]
    proj = (D - pca_mean) @ comps.T
    return FeatureTransform(
        cols=cols, mod_pos=np.searchsorted(cols, mod_idx), gene_mu=gene_mu, gene_sd=gene_sd,
        pca_mean=pca_mean, comps=comps, proj_mu=proj.mean(0), proj_sd=proj.std(0) + 1e-8)


def apply_feature_transform(t: FeatureTransform, X):
    """Return (standardized PCA coordinates, module-gene z-scores by gene index)."""
    D = _dense_cols(X, t.cols)
    Xp = ((D - t.pca_mean) @ t.comps.T - t.proj_mu) / t.proj_sd
    Z = (D[:, t.mod_pos] - t.gene_mu[t.mod_pos]) / t.gene_sd[t.mod_pos]
    return Xp, dict(zip(t.cols[t.mod_pos].tolist(), Z.T))


def auroc(scores, labels):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    pos, neg = labels == 1, labels == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return None
    r = rankdata(scores)
    return float((r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2)
                 / (pos.sum() * neg.sum()))


def permutation_p(scores, labels, *, seed: int = 0, n_draws: int = 100_000):
    """One-sided p for AUROC >= observed under exchangeable donor labels.

    Exact (every assignment of the positive labels to scored donors) when
    there are at most EXACT_LIMIT assignments, otherwise Monte Carlo with
    (1 + hits) / (1 + draws). Returns (p, n_assignments, exact).
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    n, k = len(labels), int(labels.sum())
    observed = auroc(scores, labels)
    if observed is None:
        return None, 0, True
    total = math.comb(n, k)
    if total <= EXACT_LIMIT:
        hits = 0
        for pos in combinations(range(n), k):
            perm = np.zeros(n, dtype=int)
            perm[list(pos)] = 1
            hits += auroc(scores, perm) >= observed - 1e-12
        return hits / total, total, True
    rng = np.random.default_rng(seed)
    hits = sum(auroc(scores, rng.permutation(labels)) >= observed - 1e-12
               for _ in range(n_draws))
    return (1 + hits) / (1 + n_draws), total, False


def donor_bootstrap_ci(scores, labels, *, n_boot: int = 2000, seed: int = 0,
                       level: float = 0.95):
    """Percentile interval for AUROC, resampling donors within each class."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    pos, neg = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    if len(pos) == 0 or len(neg) == 0:
        return None, None
    rng = np.random.default_rng(seed)
    lab = np.r_[np.ones(len(pos), int), np.zeros(len(neg), int)]
    draws = np.array([
        auroc(scores[np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]], lab)
        for _ in range(n_boot)])
    tail = (1 - level) / 2
    return float(np.quantile(draws, tail)), float(np.quantile(draws, 1 - tail))
