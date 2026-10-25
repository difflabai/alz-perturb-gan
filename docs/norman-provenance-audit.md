# Existing Norman source: bounded provenance audit

Audit date: 2026-10-24. This records a read-only audit of the existing practice
artifacts and official public metadata. No dataset was acquired, expression
matrix loaded into memory, GPU used, alias remapped or original artifact
changed. This document contains aggregate metadata and source identifiers;
raw cells, expression, guide lists and per-cell indices remain local. It is
not a canonical preparation handoff or an official benchmark specification.

## Exact source identity and citation

The existing `NormanWeissman2019_filtered.h5ad` has:

| Identifier | Independently checked value |
| --- | --- |
| Bytes | 698680199 |
| SHA256 | `efde6f5301fe256725dce1d980f37bd96a13481a9a16135515897368e631affc` |
| MD5 | `c870e6967d91c017d9da827bab183cd6` |
| Pseudobulk archive SHA256 | `04ec80816dae7ded499f04c91ea37e429314b0e4d9dc3ea1fb19ebdacb8c308a` |
| Target-delta archive SHA256 | `3cba0c0266c1cd90b72e867e90110f0b4895affdb1f77fe08ca09e8b7583d380` |

The raw SHA256 matches the existing `HASHES.txt`; the MD5 and byte count match
the exact filename in [Zenodo record 7041849, version 1.0](https://zenodo.org/records/7041849).
The original local download log is empty. Matching bytes support source-file
identity; they do not reconstruct when, how or by whom the file was obtained.
Do not replace this version with the record's newer version implicitly.

The [official record API](https://zenodo.org/api/records/7041849) declares
`license.id = cc-by-4.0`, open access, and describes a secondary upload of
scPerturb datasets made with scanpy 1.9.1. The retrieved JSON SHA256 is
`4cd265019649f1747d77ba23ca9294c9d79da916010ef1bdbce4c58181ec360f`.
This verifies the record-level declaration. [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
requires appropriate attribution, a license link and indication of changes;
it does not establish every underlying right or a blanket redistribution
clearance. The source-to-derived attribution/terms receipt still belongs in
Tau's handoff. The earlier statement that no primary terms receipt was available
has been narrowed by this API evidence; no dataset or model is redistributed here.

Cite Peidli and colleagues' dataset, *scPerturb Single-Cell Perturbation Data:
RNA and protein h5ad files*, version 1.0, DOI
[10.5281/zenodo.7041849](https://doi.org/10.5281/zenodo.7041849), together with
Norman and colleagues, *Exploring genetic interaction manifolds constructed
from rich single-cell phenotypes*, Science 365, 786–793 (2019), DOI
[10.1126/science.aax4438](https://doi.org/10.1126/science.aax4438).
The [author manuscript](https://pmc.ncbi.nlm.nih.gov/articles/PMC6746554/)
identifies CRISPRa activation in K562. The
[GEO GSE133344 series metadata](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE133344&targ=self&form=text&view=full)
links PMID 31395745 and describes single/combined overexpression. Its retrieved
text SHA256 is `a9a1698ae6240b2f85ce69283a760a65b9628eb596e45c6f8de283dbb2608a62`.
The H5AD's own `perturbation_type` says only `CRISPR`: CRISPRa is a study-level
annotation supported externally, not a more specific value present in that field.

## Observed local metadata and preparation

Metadata-only inspection confirms 111445 cells by 33694 expression genes;
237 labels including 105 singles, 131 doubles and 11855 control cells; 290
distinct `guide_id` values; and 111445 distinct observation IDs. All cells are
annotated K562, human, and cell-line tissue. There are eight `gemgroup` values,
1 through 8, each containing controls. These grouping values alone do not
establish eight independent biological replicates or a lane-to-replicate map.
No donor, sex, passage or experiment-date field or guide-sequence field is
present. Donor independence is unavailable; a donor value of N/A is not an
observed field. The file's `uns` group is empty.

The hash-matched existing preparation script canonicalizes underscores to
sorted `A+B` identities, applies per-cell CP10k then natural log1p, selects
5000 genes from pooled controls only, and computes per-label mean-control
deltas. It supplies a scoring-only Welch-style statistic and pools controls
without batch correction. Raw canonical labels and per-label cell counts match
the archived pseudobulk label order and `ncells`. This checks metadata alignment;
the expression summaries and control-only ranking were not independently
regenerated in this audit. The preparation-code SHA256 remains
`af1e96dac40ca9a37863fef11aaae59aee6c8be1968126c3570126be6fa3376e`.

`var['ensemble_id']` (the actual spelling) contains 33694 distinct, nonempty
ENSG-form strings. Every one of the 102 archived target-column Ensembl IDs
agrees with the corresponding exact-symbol source row. The 105-target panel's
three missing exact symbols are C19orf26, C3orf72 and KIAA1804. Historical alias
resolution and the Ensembl release remain unverified; no aliases were repaired.
The 185 PIE readout vocabulary misses are a separate issue: those source rows
have Ensembl IDs, but the existing PIE preparation selected by exact gene-text
symbol membership. This harness does not consume PIE features or remap that axis.

The target-delta archive still lacks embedded perturbation row labels. Its
row order is inferred from the hash-matched preparation script's common loop,
not independently attested by a canonical producer receipt. No frozen per-cell
index artifact was found among the relevant preparation/split artifacts.

## Consequences for splits

The five archived seed-20261022 label split lists match the harness exactly,
with test sizes 48/47/47/47/47. Their gene overlap remains intentional; those
scores address unseen labels, including some combinations of previously
targeted genes. They do not establish unseen-gene generalization.

Reconstructing source-row membership from canonical labels for all ten folds
(five label and five whole-gene, seed 20261022) gives no train/test row overlap
and no `guide_id` overlap. Every guide ID maps to one canonical label in this
file. This is an aggregate reconstruction from current metadata, not proof of
a historical frozen cell-index artifact or unseen guide-sequence generalization.
All eight `gemgroup` values occur in both train and test in every inspected
fold. Controls are shared and pooled across those groups. Neither split tests
batch-independent generalization. Batch-matched controls or grouped validation
would change the preparation/protocol and require a separate declared comparison.

The strict protocol still excludes bridging combinations and guarantees
disjoint **reported gene symbols**. Until aliases are resolved against a pinned
stable-ID namespace, biological-gene equivalence across historical names is
unverified. Missing measured target columns remain zero embeddings; their
perturbation labels are retained. These facts do not justify changing or
selectively dropping a label, silently repairing an alias, or moving a fold.

Remaining handoff gaps: immutable producer manifest binding raw source,
preparation code, output bytes and row order; stable-ID/alias map with release
and missing reasons; exact upstream filtering/QC lineage and sample/group map;
frozen cell membership or a reproducible derivation receipt; complete
source-to-derived attribution/terms evidence; and the source definition of the
historical `+/-0.01` uncertainty. The audit does not replace those receipts or
turn shared technical groups into independent biological replicates.
