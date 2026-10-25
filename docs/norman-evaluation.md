# Norman split and comparison harness

`python -m alz_perturb_gan.norman_eval` implements an isolated CPU comparison.
It starts from PR5 `fdaa6304694af1099c062467354b433fdee93939` and does not
change the existing Arc loaders, models, training code or results. Tau owns
feature/data preparation; this harness consumes frozen existing summaries.
It imports NumPy only, never initializes torch/CUDA, acquires no data, and
does not fit or publish a neural model.

## Protocol and interpretation

Predeclare at least three distinct split seeds (defaults: 20261022, 20261023,
20261024), ten PCA components and ridge 0.1. There are five folds per seed:

* `perturbation_label`: sort canonical identities, independently shuffle singles
  and doubles with NumPy `default_rng(seed)`, and round-robin each group into
  five folds. Each label is tested exactly once. Constituent genes can appear
  in training doubles/singles. This tests **unseen labels**, not necessarily
  unseen genes. With the same labels, seed 20261022 matches the legacy practice
  split algorithm.
* `whole_gene`: shuffle the sorted union of targeted genes and round-robin it
  into five blocks. For each block, train only labels whose members are all
  outside it, test only labels wholly inside it, and exclude bridging doubles.
  No reported targeted-gene symbol occurs in both train and test. Singles get one out-of-fold
  prediction; doubles in different blocks get none for that seed. Full coverage
  of every double and five disjoint gene blocks cannot both be claimed. Coverage
  and all excluded identities are reported per seed. These metrics concern a
  different test population from label holdout; compare arms within protocol.

Every test row records whether its label is unseen, the number of unseen genes,
single versus combination status, and whether all constituent singles were
available for the additive baseline. In label holdout, a single can be an
unseen label while its gene occurs in a training double. The strict protocol
means unseen **targeted identities**; a target gene's measured expression may
still occur in the training transcriptome. It does not hold out expression
columns, donors, guides, batches or cell contexts. The membership audit makes
these limits explicit. Raw cell/guide/donor provenance needs a preparer receipt.
The identity namespace is the reported label symbols. Without a reviewed alias
and stable-ID map, symbol disjointness does not establish disjoint biological
genes under every historical alias. No alias is silently repaired. The
[existing-source provenance audit](norman-provenance-audit.md) distinguishes
verified observations from unresolved preparation and source-history claims.

This predicts measured expression and retrieves the perturbation's recorded
identity. It does not create expression-distance-derived positive phenocopy
labels or infer therapeutic equivalence. All fixed hyperparameters are used
unchanged across seeds; there is no arm/seed selection on test scores. Any later
tuning requires an inner training-only protocol.

## Baselines and metrics

Each fit receives only training outcomes and the requested test identities:

* `no_change`: zero delta from control.
* `mean_effect`: the unweighted training-label mean delta.
* `additive`: sum observed training **single** deltas; missing singles contribute
  zero, with per-label missing counts. Training doubles never supply a single's
  profile. Under strict unseen-gene holdout this baseline is zero by construction.
* `pca_ridge`: SVD of centered training pseudobulk deltas. A target gene embedding
  is its centered expression across training perturbations, projected into the
  right singular vectors and divided by singular values. Double embeddings sum
  their members. The two-sided ridge fit is
  `(G'G + lambda I)^-1 G'Y P (P'P + lambda I)^-1`, with a training-mean intercept.
  Numerical null PCs are dropped; zero rank falls back to the mean. Missing
  measured target-gene columns use zero embeddings and are listed. No whole-data
  PCA, frozen outcome embedding, test DE statistics, or held-out outcomes enter
  the fit. This matches the legacy ridge formula on nonsingular training inputs.

The local **Systema-style** score is per-label Pearson correlation between
`prediction - training_perturbed_centroid` and
`truth - training_perturbed_centroid` on the full frozen expression axis. This
uses a training reference rather than an evaluation-cohort reference. It is
inspired by [Systema's custom-reference evaluation](https://github.com/mlbio-epfl/systema),
not a claim of reproducing that package or an official judging rubric.
Control-relative Pearson, MSE, and optional Pearson on the top 1,000 genes by
the test label's absolute supplied DE statistic are also reported. The DE
statistic is used only for scoring. All undefined correlations are JSON null,
and every summary reports defined/total counts. Mean-effect's centroid Pearson
is undefined, so a paired Pearson difference against it is also undefined.

Centroid accuracy compares each prediction to the fold's observed test
centroids by expression MSE, awarding 1 / number-of-ties if the correct identity
ties for nearest; otherwise zero. Normalized centroid rank is zero best, one
worst, and uses midranks for ties. These are identity-retrieval diagnostics,
not phenocopy training labels. Candidate-library size changes across folds.

All seeds and strata are retained. Paired comparisons require identical unique
seed/fold/label keys plus split/data/reference hashes. For each metric, average
paired seed differences within a label, then bootstrap labels 2,000 times with
seed 1729. Report the mean difference, percentile 95% interval, paired counts
and undefined exclusions. This clusters repeated labels across split seeds;
it conditions on these seeds, fitted references and candidate sets. It is not
uncertainty over independent donors or genes. Related combinations share genes,
and training sets overlap, so these intervals are descriptive rather than
confirmatory biological inference. Per-seed macro means are available separately.

The historical Norman ridge/PCA `0.668 +/- 0.01` is a **reproduction target**,
not an asserted outcome or official criterion. Compare it only with matching
QC (the existing practice packet has 105 singles + 131 doubles, rather than
100 + 124), normalization, gene axis, exact folds, baseline formula, top-1,000
DE definition and aggregation. Its uncertainty definition also needs a source
receipt. Full-axis Systema-style correlation and stricter gene holdout are
different quantities. A matched first seed alone is not a three-seed reproduction.

## Frozen packet handoff (Tau)

The CLI accepts schema 1 JSON, with these fields:

```json
{
  "schema_version": 1,
  "status": "frozen",
  "dataset": "exact Norman release and QC variant",
  "normalization": "per-cell CP10k then natural log1p, independent per label",
  "outcome_transform_scope": "per_label_only",
  "gene_selection": {"fit_scope": "controls_only", "description": "exact gene-axis rule"},
  "controls": {"composition": "exact labels/batches/cell counts and pooling"},
  "source": {"sha256": "raw-source byte hash", "revision": "exact source revision"},
  "preparation": {"code_sha256": "preparer code hash", "row_alignment": "how verified"},
  "terms": {"status": "review status", "citation": "primary receipt", "limitations": "restrictions"},
  "limitations": ["exact cell/guide/donor/QC limitations"],
  "target_row_labels": ["A", "A+B", "B"],
  "pseudobulk": {"path": "pseudobulk.npz", "sha256": "byte hash"},
  "target_deltas": {"path": "target_deltas.npz", "sha256": "byte hash"}
}
```

`pseudobulk.npz` has floating `D [labels, expression_genes]` (perturbation
mean minus control mean), Unicode `perts`, Unicode `genes`, and optionally
finite scoring-only `T` on the same axis. `target_deltas.npz` has floating
`Dt [labels, measured_target_genes]`, Unicode `targets`, and preferably Unicode
`perts`. Original raw measurements for each label may be summarized independently;
outcome-fitted whole-panel gene selection, scaling or projections are rejected.
For training-fitted/external features, a separate future fold-specific contract
is needed; this harness never consumes Tau's whole-cohort transformed features.

The manifest must attest the exact parallel target row order; if the legacy
target archive lacks embedded `perts`, this limitation is recorded explicitly
as preparer attestation only. Missing measured targets stay in the evaluation
population with zero embeddings. Supply stable-ID mapping and missing reasons
in preparation evidence; no ambiguous mapping is repaired by this evaluator.
NPZ hashes are verified before loading and pickle is disabled. A source `path`,
if supplied, is rehashed and checked; otherwise its claimed SHA is recorded as
`not_rehashed`. The manifest is itself hashed. Terms fields record evidence,
not blanket permission or a substitute for source review. Do not acquire new
data or use new unreviewed sources to satisfy this contract.
Source, preparation-code and archive SHA256 identifiers must be complete
lowercase 64-digit hexadecimal strings. Optional source `md5` must be a complete
32-digit identifier and is also checked when the raw source path is available.
Abbreviated identifiers and unknown placeholders are refused, even when the raw
source is unavailable. An unknown source remains a handoff gap.

For a fully auditable real-data handoff, Tau must supply the frozen manifest,
both archives and their byte hashes, embedded labels or row-alignment receipt,
raw-source hash/revision, normalization/QC/control composition, control-only
axis-selection evidence, preparation-code hash, target stable-ID map/missing
reasons, cell/guide/batch/donor grouping limitations, and terms/citation receipt.
Existing legacy packets can be diagnosed with candid limitations in the
manifest; those runs do not become a fully audited preparation or licensed
redistribution receipt.

## Run and evidence

No dependency installation or GPU capacity is needed in an existing NumPy
environment. Pin its version for bitwise hashes: SVD/BLAS implementations can
change low-order floats. Set single-thread limits for bounded CPU usage.

```bash
export ALZ_SCOPE_ROOT=/absolute/approved/root
export PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m pytest tests/test_norman_eval.py
python -m alz_perturb_gan.norman_eval \
  --manifest /absolute/approved/root/packet/manifest.json \
  --output /absolute/approved/root/new-comparison
```

The output directory must be new and inside `ALZ_SCOPE_ROOT`. Existing outputs
are never overwritten. `results.json` includes every split and membership audit,
split hashes, all per-label scores, baseline fit/prediction/reference hashes,
input byte and content hashes, manifest snapshot/hash, seed/configuration,
Python/NumPy versions, code hashes, Git commit/status, coverage, summaries and
paired intervals. `SHA256SUMS` binds its bytes. No environment secrets are logged.
The code creates no raw-data or model publication; keep real-data receipts local.

CPU tests cover deterministic folds, canonical duplicate rejection, mixed-block
purging, novelty strata, holdout-outcome poisoning, scoring-only DE independence,
historical ridge algebra, additive fallbacks, singular PCA, centroid/tie/null
behavior, paired-cluster alignment, hash/order guards, JSON receipts and overwrite
refusal. Synthetic checks establish engineering behavior, not biological efficacy.
