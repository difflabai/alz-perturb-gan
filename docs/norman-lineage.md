# Opt-in source-contributor audit

`python -m alz_perturb_gan.norman_lineage` audits an existing hash-bound Norman
evaluation receipt. It reads that receipt's exact protocol, seed, fold, train,
test and prediction-row identities. It never regenerates a different split,
fits a model, changes preprocessing, drops rows or recomputes a metric. The
existing `norman_eval` command and its default behavior are unchanged.

An aggregate delta is linked to **many treatment cells and many control cells**.
It is not assigned a fictitious single cell. Shared pools are stored once and
referenced by aggregate rows. The output links each evaluation row to its
aggregate membership, contributor counts/digests and hash-bound observation
and membership files. Those files resolve the source cells, guides and
gemgroups. Keep real contributor files and audit outputs local; they can
contain source cell identifiers.

## Adapter interface

The adapter JSON has `schema_version: 1`, `status` (`partial` while evidence is
incomplete), `provenance_kind` (`metadata_reconstruction` or producer-provided
`producer_attested`), `identity_scope`, `control_use`, `bindings`, and two
optional hash-bound artifacts:

```json
{
  "schema_version": 1,
  "status": "partial",
  "provenance_kind": "metadata_reconstruction",
  "identity_scope": {
    "namespace": "explicit source/capture namespace",
    "status": "export_local_only",
    "limitation": "physical library/cross-export mapping unavailable"
  },
  "control_use": ["response_construction", "control_only_gene_axis_selection"],
  "bindings": {
    "source_sha256": "complete raw-source SHA256",
    "data_content_sha256": "evaluated data content SHA256",
    "pseudobulk_sha256": "evaluated pseudobulk archive SHA256",
    "target_deltas_sha256": "evaluated target archive SHA256"
  },
  "observations": {"path": "observations.jsonl", "sha256": "complete SHA256"},
  "memberships": {"path": "memberships.json", "sha256": "complete SHA256"}
}
```

Each observation JSONL row has nonempty strings `cell_id`, `library_id`,
`label`, `guide_id` and `gemgroup`. `label` is a packet label or `control`.
Absent guide/group metadata must remain absent or null, never an invented ID.
Cell keys contain `(namespace, library_id, cell_id)`. An observation may supply
its own `namespace`; otherwise the adapter's namespace is used. Group member
references may be `[library_id, cell_id]` (adapter namespace) or
`[namespace, library_id, cell_id]`. Identity strings must be nonempty, trimmed,
and free of control characters. Gemgroups remain scoped by namespace/library.
Guide IDs use `(guide_namespace, guide_id)` **without a library component**, so
a declared guide reused across captures/libraries is still counted. A
`guide_namespace` may be supplied per observation or in `identity_scope`.
Absent guide namespaces fall back to the cell namespace for descriptive
counts and are explicitly unverified/partial. No sequence/alias equivalence
is inferred from matching names. Different exports can contain
the same physical cells: an export checksum is **not** a physical-cell
namespace. `canonical_physical_namespace`, `producer_attested` and `frozen`
are retained as producer declarations, never independently verified facts.
Changing those strings cannot produce complete evidence or a clean policy
pass. Artifact hashes verify the supplied bytes and binding to the evaluated
data; they do not authenticate the producer or establish production history.
This adapter has no independent attestation trust anchor, and reports that gap
even when every declaration and byte hash is present. No canonical lineage
manifest currently exists for the legacy packet.

Membership JSON uses explicit treatment/control roles:

```json
{
  "granularity": "aggregate",
  "cell_groups": {
    "treatment:A": [["library1", "barcode1"], ["library1", "barcode2"]],
    "controls": [["library1", "control1"], ["library1", "control2"]]
  },
  "rows": [{"label": "A", "treatment_groups": ["treatment:A"], "control_groups": ["controls"]}],
  "excluded_cells": []
}
```

Every expected aggregate needs both roles. Explicit source-cohort exclusions
may provide `library_id`, `cell_id`, and a nonempty `reason`; excluded cells
cannot also be contributors. This records producer selection rather than
performing new QC. Producer claims, namespace status and frozen status must
be supplied by the owner; do not upgrade reconstructed edges to canonical
evidence. Hash agreement verifies bytes and declared binding, not the truth
of an unsupported production history.

## Findings, coverage and policies

The audit reports duplicate source identities, repeated edges within groups
or an aggregate role, duplicate aggregates/evaluation keys, orphan cells,
groups and rows, role/label contradictions, cross-role reuse, and missing
contributors/fields/bindings. Ambiguous duplicate observations are not silently
selected. It also reports within-train/within-evaluation contributor reuse
separately from cross-split reuse. Repeated control edges across aggregates
can be intentional; they are not duplicate physical observation records.

For every exact saved fold it reports unique shared contributors for treatment
versus treatment, control versus control, each cross-role direction, and all
contributors. It counts affected evaluation aggregates with their denominator,
retained rows, observed role coverage, scoped gemgroup sharing and treatment
guide-ID sharing. A zero count with incomplete evidence is marked
`unknown_or_partial`, not certified clean. Gene isolation still depends on
the separate reported-symbol/alias contract.

* `report` (default): describe evidence and sharing without enforcing a new
  independence claim; CLI returns zero and `policy_result: not_enforced`.
* `source_disjoint`: require valid complete declared lineage and no shared
  source contributors, including shared reference/control contributors.
* `group_disjoint`: additionally require no scoped gemgroup sharing.

The stronger policies return exit code 2 for either a failed condition or
unknown/partial evidence. Without an independently verified producer history,
even fully populated declarations with no observed overlap remain `unknown`.
The report distinguishes `fail` from `unknown` and
preserves all missingness alongside observed overlaps. No policy certifies
biological independence, licensing, unseen guide sequences or canonical genes.
`evidence_validity` and `evidence_completeness` are separate. The headline
`evidence_status` prioritizes `invalid` whenever structural defects exist;
missing fields cannot mask invalidity. Report mode exits zero after emitting
the report even for invalid evidence: zero means **report produced**, not
evidence accepted. Invalid byte/split hashes are refused before reporting.
Invalid exclusions never suppress orphan counts. Repeated role pools,
including pools composed of multiple groups, reuse cached contributor sets
and digests without changing duplicate-edge detection.

Unmapped targets are read from the exact saved fit receipts, not guessed from
symbols. Known holes (C19orf26, C3orf72 and KIAA1804 in the legacy packet) are
explicit missing evidence, attached to affected aggregate/fold rows. Missing
fit mapping receipts are separately unknown. No aliases or embeddings are
changed and the original score remains untouched.

Shared gemgroups limit batch-generalization claims but do **not** automatically
constitute target leakage. A shared fixed reference can also be legitimate for
unseen-gene evaluation. It prevents a claim of fully independent source
populations. Record how controls are used: a fixed scoring reference is
different from controls used to construct training/evaluation response deltas,
select expression genes, normalize data, or fit transformations. In the existing
practice packet, pooled controls enter response construction and control-only
axis selection; CP10k/log1p themselves are per-cell operations. Do not call
their stronger-policy failure proof that the stated legacy score is invalid.

With fixed memberships containing shared pooled controls, source-disjoint
folds are infeasible without changing those memberships. The audit reports
that constraint and retained coverage; it does not drop affected aggregates,
move controls or silently turn the task into batch-held-out evaluation.

## Run and interpretation of retained scores

```bash
export ALZ_SCOPE_ROOT=/absolute/approved/root
export PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m alz_perturb_gan.norman_lineage \
  --evaluation-receipt /absolute/approved/root/results.json \
  --lineage-manifest /absolute/approved/root/partial-lineage.json \
  --policy report --output /absolute/approved/root/new-audit
```

Omit `--lineage-manifest` to obtain an explicit missing-evidence report. Output
must be a new scoped directory. Its JSON and byte ledger bind the exact input
receipt, adapter/artifacts and audit source. Inputs are not modified. This
module imports no torch, H5AD reader or GPU library and acquires nothing.

The retained existing-packet **ridge Pearson delta top-1000** scores are:

| Evaluation population | Score | Coverage |
| --- | ---: | --- |
| Label holdout, seed 20261022 matching the historical split | 0.668081 | 236 labels |
| Label holdout, three predeclared seeds pooled | 0.666332 | 236 labels per seed |
| Strict reported-gene-symbol holdout, the same three seeds pooled | 0.563597 | 128 / 135 / 122 labels |

The commonly shortened **.668 versus .564** compares different split/test
populations and, as quoted, different seed aggregation. It is not an improvement
or a new model result. Strict folds purge mixed-block combinations; labels,
gene availability and denominators differ. The historical +/-0.01 uncertainty
is still not reproduced. See [protocol](norman-evaluation.md) and
[source audit](norman-provenance-audit.md) for the remaining limitations.

Fixture tests exercise clean conditional disjointness, partial/missing evidence,
scoped IDs, shared controls versus batch sharing, duplicates/orphans, cross-role
reuse, exact split/row binding, coverage, exclusions, hash guards and exit
policies. Existing-data reconstruction remains partial until Tau supplies the
canonical producer manifest, physical namespace/mappings and aggregation
lineage. Record-level CC BY 4.0 evidence does not replace the complete
source-to-derived attribution receipt; no terms are accepted by this audit.

## Test scopes and the earlier 86 count

At PR8's initial `d9159e0` head, the minimal NumPy/pytest suite was **69 tests**:
28 in `test_norman_eval.py` and 41 in `test_norman_lineage.py`. That is the
GitHub workflow and the scope Tau reproduced. The previously reported **86**
was that same suite plus 17 existing regressions: 5 baselines, 5 metrics,
3 data and 4 path-hygiene tests. It was not 86 NumPy-only tests.

The full command used the already-existing local environment, without adding
packages or launching a GPU job:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 PYTHONPATH=src /ml2/altgan/.venv/bin/python -m pytest \
  -p no:cacheprovider --basetemp=/tmp/norman-lineage-publish-20261025 \
  tests/test_norman_lineage.py tests/test_norman_eval.py \
  tests/test_baselines.py tests/test_metrics.py tests/test_data.py \
  tests/test_path_hygiene.py
```

That environment had Python 3.11.15, NumPy 2.4.6, pytest 9.1.1, Torch
2.6.0+cu124, pandas 2.3.3, AnnData 0.12.19 and h5py 3.16.0. The extra
baseline/metric/data tests need existing Torch/AnnData dependencies; the
NumPy-only CI suite does not. No new installation is required for review.
The repaired minimal suite is **81 tests** (28 evaluator, 53 lineage). The
same 17 existing regressions bring the optional extended local scope to
**98 distinct tests**. CI runs the 81-test NumPy/pytest scope; it does not
require installing Torch, pandas or AnnData. These counts are not added across
repeated runs.
