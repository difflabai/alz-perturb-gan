"""Perez-Alcantara — expression conditioned on donor genotype x state.

Find: Washer & Perez-Alcantara et al., Nature Genetics 2026 (briefed
cycle 3; access solved cycle 8). 261 pooled iPSC donor lines
differentiated to microglia-like cells in three states (Untreated /
LPS / IFN); QTL mapping + pooled CRISPR KO of 251 disease genes. A
gene's effect depends on cellular context (TREM2 KO raises
phagocytosis in microglia, lowers it in macrophages).

Route: NEW DISTRIBUTION. Expression conditioned on donor GENOTYPE x
stimulation state in human microglia-like cells — fits must be
state-split, never pooled (the paper's own context-sign lesson).

Iteration 1 (this file, 2026-11-22): loader + sanity on the Untreated
arm, whose zip landed byte-exact this cycle. The metadata sanity runs
now; the h5ad counts-vs-normalized check skips cleanly until the
31.4 GB extraction completes locally. The conditional fit itself is
iteration 2: distribution conditioned on donor genotype group at one
AD locus vs shuffled-genotype arm + mean/linear baseline, donor as
the unit in every split, Untreated only.

Iteration log: notes/perez_alcantara_genotype.md
"""
from __future__ import annotations

import numpy as np
import pytest

from alztest.data import (DataUnavailable, load_perez_untreated_metadata,
                          perez_untreated_h5ad)


@pytest.fixture(scope="module")
def meta():
    try:
        return load_perez_untreated_metadata()
    except DataUnavailable as e:
        pytest.skip(str(e))


def test_perez_metadata_sanity(meta):
    # Verified 2026-11-22 from the landed file (cycle 11): 862,274
    # rows, 242 donors, mean demultiplexing prob_max 0.9993.
    assert len(meta) > 800_000
    for col in ("donor_id", "prob_max", "best_singlet", "treatment",
                "pool", "sequencing_batch", "merged_clusters"):
        assert col in meta.columns, f"missing metadata column {col}"
    assert set(meta["treatment"].astype(str).unique()) == {"untreated"}
    donors = meta["donor_id"].astype(str).value_counts()
    assert donors.shape[0] >= 200, donors.shape[0]
    # Donor-unit fits need depth: most donors must clear 100 cells.
    assert int((donors >= 100).sum()) >= 150
    # Demultiplexing must be near-certain or genotype conditioning is
    # built on sand.
    assert float(meta["prob_max"].mean()) > 0.99


def test_perez_h5ad_counts_check():
    try:
        p = perez_untreated_h5ad()
    except DataUnavailable as e:
        pytest.skip(str(e))
    import anndata as ad
    adata = ad.read_h5ad(p, backed="r")
    meta = load_perez_untreated_metadata()
    assert adata.shape[0] == len(meta), (adata.shape, len(meta))
    assert adata.shape[1] > 10_000  # transcriptome-wide, not a panel
    # Counts check on a row subset: raw counts are non-negative and
    # (near-)integer; normalized/log values are neither. This decides
    # which preprocessing the iteration-2 fit starts from.
    from scipy import sparse
    X = adata[:2000].X
    X = X.toarray() if sparse.issparse(X) else np.asarray(X)
    assert float(X.min()) >= 0.0
    frac_integer = float(np.mean(np.abs(X - np.round(X)) < 1e-4))
    # Recorded, not asserted to a fixed side: the loader docstring in
    # the notes states which regime the object is in once this runs;
    # the assertion is that the matrix is decisively one or the other.
    assert frac_integer > 0.9 or frac_integer < 0.1, frac_integer
