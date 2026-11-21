# GSE289721 Perturb-seq readout — iteration log

Find: Cardona et al., J Neuroinflammation 2025 (PMID 41340152);
GEO GSE289721 (public 2025-12-31). Cycle-5 top find, converted in
cycle 6 (the data landed 2026-11-18 23:35 UTC; the loop is run
under Mikkel's standing rule: run the loops every time).

Data facts (verified from the files): 4 libraries (SS001, SS002,
AS001, AS002), 49,568 cells total; matrices carry 36,601 genes +
72 CRISPR guide-capture features. Guide assignment (dominant guide
>= 50% of guide UMIs, >= 3 guide UMIs): SS libraries ~35%, AS
libraries ~15% — pooled 12,6xx assigned cells. Targets: AGFG2,
EED, INPP5D, MS4A6A, PVR, RABEP1 (each >= ~1,300 pooled assigned
cells) + Non-Targeting (~3,300). SS vs AS library arms are pooled
in v1 (library noted as a covariate; a library-stratified split is
the named iteration-2 candidate if batch structure shows).

Design: log-normalized counts -> top-2000 variable genes -> 30 PCs
(PCA + standardization fit on TRAIN only — the NeuroKO draft's
refit-on-test wart fixed here); 80/20 split per target; conditional
ParticleGAN (unmodified EMPF, 300 steps) vs shuffled-target,
mean-shift, control-only arms. Assertions: model ED < every arm
every seed; mean shift cosine > 0.3.

## Iteration log

- Iteration 0 (data) — DONE 2026-11-19: tar assembled byte-exact
  (619,325,440 B), extracted, guide assignment verified per library
  (counts in the cycle-6 ledger entry). Loader caches the built
  object at data/gse289721_perturbseq.h5ad.
- Iteration 1, attempt 0 — OOM (exit 137) at 207 s: _prepare
  densified all 49,568 x 36,601 cells (~14 GB float64). Seam fix
  (our plumbing, not a loop iteration): sparse-first preparation —
  subset to assigned cells, sparse log-normalize, top-2000 genes
  from sparse moments, densify only those (~200 MB).
- Iteration 1, attempt 1 — all seeds died in <1 min: scipy sparse
  multiply does not broadcast a 1-D row vector; needs a column
  vector. Fixed (plus Xn.power(2) -> multiply(Xn) for the sparse
  second moment). Plumbing, not a loop iteration.
- Iteration 1, attempt 2 — sparse multiply returned COO (not
  subscriptable); .tocsr() added. A _prepare-only smoke check now
  precedes gauntlet launches (three prep-stage slips in a row is
  the lesson: verify the plumbing cheaply before the fits).

## Iteration 1 — full pool (10,199 train cells), 300 steps
| seed | ED model | ED shuffled | ED mean-shift | ED control-only | shift cos |
| 0 | 0.306 | 0.317 | 0.035 | 0.096 | -0.118 |
| 1 | 0.300 | 0.334 | 0.034 | 0.110 | +0.081 |
| 2 | 0.316 | 0.344 | 0.039 | 0.112 | +0.014 |

Verdict: NOT PROVEN, decisively. The model is worse than even the
control-only arm (0.30 vs 0.10): at 10k training cells, 300 steps
x batch 64 is under 2 epochs — the generator has not learned the
marginal, let alone the conditional shifts (cosine ~0). The
perturbation effects themselves are real and large (control-only
0.10 vs mean-shift 0.035) and almost entirely first-moment at this
representation.

## Iteration 2 — fixed subsample (train cap 800/condition)
One change (data handling): cap the training pool per condition
at 800 cells (~5.6k total), matching the SEA-AD tests' scale where
the same 300-step budget converged. Test split unchanged.

## Iteration 2 — train cap 800/condition (~5.6k cells)
| seed | ED model | ED shuffled | ED mean-shift | ED control-only | shift cos |
| 0 | 0.341 | 0.326 | 0.036 | 0.093 | -0.046 |
| 1 | 0.335 | 0.315 | 0.034 | 0.106 | +0.144 |
| 2 | 0.368 | 0.318 | 0.040 | 0.109 | -0.071 |

Verdict: NOT PROVEN; the underfitting hypothesis is FALSIFIED —
less data at the same budget made the model slightly worse, and it
now sits above even the shuffled arm. The failure is not epochs.

## Iteration 3 — readout decomposition (diagnostic)
One change (readout): also score model-vs-real energy distance on
the CONTROL condition from the same fits. If control-condition ED
is ~0.3 like the targets, the generator is not matching even the
marginal distribution at this scale — a fit-quality fact for the
working-GANs effort (hand-off, not fix). If it is ~0.1, the gap is
in the conditioning.

## Iteration 2 — results (completed 2026-11-20, cycle 8)
| seed | ED model | ED shuffled | ED mean-shift | ED control-only | shift cos |
| 0 | 0.341 | 0.326 | 0.036 | 0.093 | -0.046 |
| 1 | 0.335 | 0.315 | 0.034 | 0.106 | +0.144 |
| 2 | 0.368 | 0.318 | 0.040 | 0.109 | -0.071 |

Verdict: NOT PROVEN. The 800/condition cap did not rescue the fit —
model ED is worse than shuffled on seeds 0/2 and ~3x worse than
control-only on every seed, the same failure shape as iteration 1.
Seeds 1-2 were killed by the 2026-11-20 20:52 UTC VM restart and
rerun to completion the same day (RESTART2, repo .venv python —
system python3 lacks anndata; use .venv/bin/python3 for all runs).

Iteration 3 (named, one change — evaluation/data handling only):
perturbation-specific scoring against library-matched controls +
treatment-stratified splits (vehicle vs poly(I:C); the four
libraries are SS/AS x vehicle/poly(I:C) and v1/v2 pooled treatment
blindly). Rationale: Ghajari et al., bioRxiv 2026-11-04 (cycle-8
plate-design brief) — absolute distances include the shared
library offset; perturbation-specific differences are the honest
score. Candidate later arm: guide-UMI-fraction strength-stratified
cosine (Sharma et al., cycle-8 brief).

## Iteration 3 — readout decomposition RESULT (completed 2026-11-21, cycle 9)
Same fits as iteration 2 (train cap 800/condition, 300 steps); the
one change is readout-only: model-vs-real energy distance also
scored on the CONTROL condition (ed_pgan_control_condition).

| seed | ED model (targets) | ED control-cond | ED shuffled | ED mean-shift | ED control-only | shift cos |
| 0 | 0.341 | 0.308 | 0.328 | 0.036 | 0.093 | -0.103 |
| 1 | 0.360 | 0.293 | 0.339 | 0.034 | 0.106 | -0.019 |
| 2 | 0.353 | 0.288 | 0.340 | 0.040 | 0.109 | +0.069 |

Verdict: the decomposition answers the question. Control-condition
ED (~0.29-0.31) is essentially the target-condition ED (~0.34-0.36):
the generator does not match even the CONTROL marginal at the
company-standard budget on this dataset — the failure is
marginal-fit failure, not conditioning failure. Conditioning adds
only ~0.05 ED on top of a ~0.30 marginal gap, while real-vs-real
control-only distance sits at ~0.10 and the mean-shift arm at
~0.035. LOOP CLOSED at iteration 3 of 10, NOT PROVEN (honest
negative). Remaining levers would be GAN-side (budget/architecture),
which this watch hands off, never performs: HAND-OFF to the
working-GANs effort — at 300-step EMPF in 30-PC space the
generator underfits the GSE289721 marginal by ~3x the real-vs-real
distance scale, on both pool sizes tried (iterations 1-2) — the
same shape as the SEA-AD/LL/resilience toy fits, where first-moment
arms won every gauntlet. Perturbation effects in this dataset are
real and large (mean-shift ED 0.035 vs control-only 0.10) and
almost purely first-moment in this representation.
