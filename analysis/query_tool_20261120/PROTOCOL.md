# Query tool and EMPF recomputation of the headline biology: protocol

Declared 2026-11-20 before any replicate archive was scored and before any
EMPF biology number was computed. Only the bitwise adapter validation on two
targets (replicate 0, CPU) may precede this file.

## Scope

1. A usable query tool, `scripts/microglia_query.py` (new file; the pinned
   `scripts/microglia.py`, `scripts/microglia_runtime.py`, config and all
   manifest-pinned sources stay byte-identical). It reads precomputed, small
   query bundles and never loads torch or weights.
2. A recomputation of the README's headline biology on the selected native
   EMPF fold0/seed0 model, written beside, not over, the frozen 0.9.7 report.

No training update, no weight change, no change to the default, the frozen
report (`docs/`) or any earlier evidence. The running 17-fit confirmation
queue (`runs/microglia_empf/fits`, `analysis/coverage_empf_confirm_20261120`)
is not read, touched or waited on. Its fits are therefore **not** part of the
ensemble here; only one fit exists for this analysis. The tool accepts
several bundles (one per fit) so a later fits ensemble needs no code change.

The derived native host is imported from a private, hash-verified snapshot
(`native_host.py`) because `host.activate` rewrites shared derived files (the
recorded activation race).

## Generation Monte Carlo

Model: `runs/empf_comparison_20261117/fits/native/pbulk/hvg2000/fold0/seed0`
(weights and metadata checked against `selected_model.json`), readout
`cellshape@0.5:biascal:stagehalf`, six late states, the default contract of 37
development-donor conditions x M=32 generated donors x K=256 cells.

Replicate r uses streams `SeedSequence([20261114, 6, fold, dev_index, r])`;
r=0 is exactly the scored default archive. The adapter must reproduce that
archive bitwise on CPU for replicate 0 (first two targets; all six archive
arrays). The analysis replicates are **r = 1..8** (eight, matching the eight
stage replicates per 0.9.7 model), generated on a shared GPU (fraction <=.06,
3 threads, cores 3-23). They are statistically exchangeable draws, not a
bitwise CPU reproduction. Generated cells are assigned subtypes with the fold's
frozen TRAIN-only type classifier on `(z @ W) / pc_sd`, the same convention as
the 0.9.7 analysis; these are predictions, not observed labels.

## A. Frozen-metric Monte Carlo variability (descriptive)

Score each analysis replicate with the frozen
`analysis/improve_20261109/r5/architecture_score.py` (historical + interval
metrics, held-out donors). Report mean, SD and range over the 8 replicates for
profile RMSE, mean/detection/log-variance gene errors, held-out SD and
combined coverage, width and interval scores, PC and cell energy, alongside
the recorded single-archive EMPF, previous-default and stage-blind empirical
resampling (`empirical32`) values. Descriptive question: does each recorded
EMPF-vs-previous-default difference exceed 2 x the replicate SD? This uses one
fit; it cannot measure fit-to-fit variability.

## B. Gene-level donor intervals (the tool's calibration)

Coordinates per gene: a donor bag's mean of per-cell log1p(panel CP10k) and
its SD (K=256 cells; generated SD from `sq_log`, `mean_log`, with the K/(K-1)
correction). Raw stage interval: 10%/90% percentiles of all generated donors
of targets in that stage, pooled over the 8 replicates. Observed held-out bag:
256 cells drawn without replacement (all cells when fewer), 24 fixed repeats
seeded by donor and repeat; repeat 0 only is used for calibration.

Calibration is the coverage pilot's rule, unchanged except the coordinates:
nonnegative score `max(0, low-y, y-high)`, rank `ceil((n+1)*0.8)`, symmetric
expansion, SD lower bounds clipped at 0. Calibration donors must not be GAN
TRAIN donors. Cross-fitting uses the pilot's exact two six-donor halves
(`analysis/coverage_empf_20261117/completion/comparison.json`).
Primary: SD coordinates (the pilot's SD-only calibration; mean bounds
unchanged). Secondary, declared extension: the same rule on mean coordinates.

Score on the 12 held-out donors x 24 repeats x 2000 genes: coverage, below,
above, width, interval score (alpha .2). Donor bootstrap (2000 resamples of
held-out donors) gives 95% CIs for calibrated-minus-raw interval score.
Reference: stage-blind empirical resampling: 800 bags (32 per TRAIN donor,
256 cells with replacement from that donor) pooled regardless of stage, with
and without the same calibration. Calibration is called useful if, for the
coordinate group, coverage moves closer to 80% and the interval score falls.
The shipped bundle uses a correction refit on all 12 held-out donors (n=12,
rank 11); its own coverage cannot be evaluated on those donors and is labelled
as such.

## C. Headline biology on EMPF

Same definitions as the frozen 0.9.7 analysis
(`analysis/aggregate_primamente_gene_inference.py`,
`scripts/report/export_stage_profiles.py`):

* Within-subtype contrast: High minus Not AD of mean per-cell log1p(panel
  CP10k), equal weight on the two fixed subtypes Micro-PVM_2 and
  Micro-PVM_2_3-SEAAD (groups 2 and 4). Synthetic: per replicate and stage,
  pool the generated cells of that subtype over all targets of the stage;
  requires >=100 cells per subtype per stage replicate. Observed: equal donor
  weight among donors with >=20 cells of the subtype; >=3 donors per stage.
* Pooled contrast: same with all cells (synthetic equal-target weight,
  observed equal-donor weight, >=1 cell).
* Stage profile: per-stage within-subtype means; the peak is the argmax stage.
* Across-gene agreement: descriptive Spearman and direction agreement between
  model and observed within-subtype contrasts over all panel genes with both
  finite; also on the genes shared with the 0.9.7 panel.

Observed references on the EMPF fold0 panel (CP10k uses this panel's total):
primary = the same 33 donors as the frozen report; secondary = the 25 fold0
GAN TRAIN donors (in-sample analog of the 0.9.7 training-association
comparison) and all 37 development donors. The 12 held-out donors contain one
Not AD donor and cannot satisfy the >=3-donor rule; that reference is reported
as not estimable. Observed contrasts get donor-bootstrap 95% CIs (stage-
stratified, 2000 resamples). Stage-blind empirical resampling predicts a zero
contrast for every gene by construction and is reported as the reference.

The 0.9.7 values are recomputed from the frozen stage-replicate summaries and
must reproduce the published +0.454 / +0.489 / 0.502 / 65.9% before
side-by-side reporting. Reported per model: CLEC5A contrast and MC/seed
agreement; whether the Low-stage peak is recovered; Spearman and direction
agreement. A single EMPF fit is not compared to a five-model mean as if
equally certain; fit-level uncertainty is stated as unmeasured.

## D. Gene flags in the tool

* `not_in_panel`: gene absent from the 2000-gene panel.
* `poorly_fit`: held-out donor-level mean-log RMSE (stage-matched target means
  vs observed donor means) above the panel's 90th percentile, or above 1.5 x
  the stage-blind empirical reference's RMSE for that gene.
* `pc_blind_spot`: the gene's share of the TRAIN-donor stage gap (observed minus
  generated mean, standardized) outside the 50-PC column space, per stage;
  flagged when >0.5 in any stage where |gap| > 0.05 standardized units; also
  reported: static PC-captured fraction `||W_j||^2`.
* `low_expression`: observed mean detection < 1%.

Thresholds are fixed here. Results are reported whatever their direction.
