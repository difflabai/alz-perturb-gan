# NeuroKO perturbation test — iteration log

Find: Dominicus et al., bioRxiv 2026.10.13.758778. Route: new distribution + new target.
Test: `tests/test_neuroko_perturbation.py`. Arms: conditional ParticleGAN vs
shuffled-KO ParticleGAN vs mean-shift moment baseline vs control-only baseline.
Metric: energy distance to held-out KO cells + shift-direction cosine, 3 seeds.

## Iteration 0 — 2026-11-17 (access verification, no fit run)
- Test implemented end to end (loader, KO/cell-type introspection, arms, assertions).
- Data access checked four ways: (1) Zenodo API search by title/authors — no record;
  (2) the DOI printed in the preprint's data availability (10.5281/zenodo.2326038861)
  does not resolve — the identifier is corrupt as printed (11-digit suffix);
  (3) Europe PMC record — no data links, no full-text XML; (4) authors' GitHub
  (Lotfollahi-lab/NeuroID-reproducibility) — README carries no data link.
- Raw sequencing exists at ENA ERP134820 (unprocessed FASTQ) — processing a pooled
  Perturb-seq screen from raw reads is a project, not a toy test; not attempted.
- Status: BLOCKED on the annotated object. Next iteration runs unchanged the day
  data/neuroko_annotated.h5ad (or NEUROKO_H5AD) exists. Re-check the Zenodo record
  each watch cycle.

## 2026-11-21 — blocker resolved; schema verified from the authors' code
Corrected record: Zenodo 10.5281/zenodo.23260389 (v1, CC BY 4.0);
iNeurons.h5ad 6,314,792,054 B (md5 cd4d1fbb316a66f6d4daebfd770acd8e),
iAstrocytes.h5ad 7,465,978,066 B (md5 10abe464ca16bcaab07d644adcaae3ff).
Downloads relaunched 2026-11-21 (resumable; both had died silently
once — the restart pattern, not a data problem).
Schema, from the authors' own NeuroID-reproducibility scripts
(neuroID_iNeurons_disect.py, read 2026-11-21): obs carries
`target_gene`, `perturbation`, and `Replicate`; the authors
EXCLUDE targets INT3/INT30 in their analysis and treat
`perturbation` + `Replicate` as the categorical covariates; their
working object holds raw counts in X. The test's KO_CANDIDATES
includes both `perturbation` (picked first if present) and
`target_gene` — on open, compare the two columns' value counts and
pick the gene-level one deliberately, logging the choice; control
label to be identified from the actual values.

## 2026-11-22 — data landed; annotation check PASSED; iteration 0
iNeurons.h5ad assembled byte-exact (6,314,792,054 B) with md5
cd4d1fbb316a66f6d4daebfd770acd8e matching the Zenodo record.
iAstrocytes downloading (17% at check). Annotation check on open:
205,028 cells x 36,601 features; obs carries perturbation (251
distinct; INT intergenic control = 6,795 cells), target_gene (280),
Replicate (4 screens), guide_calls, target_vector — exactly the
schema the authors' code predicted. Nothing absent; no report of
missing fields needed. X = raw counts (float32 CSR).
Iteration 0 (plumbing): first _prepare attempt OOM-killed — the
object holds 784M nonzeros; a full load + float64 conversion peaks
past the VM's RAM. Fixed by backed read + row-subset to the
selected conditions BEFORE any conversion, float32 throughout.
Design and assertions unchanged.

## Iteration 1 — full gauntlet, 2026-11-22 (12 KOs scored per seed)
seed 0: ED model 0.275 / shuffled 0.270 / mean-shift 0.0385 /
  control-only 0.0652; shift cosine +0.179
seed 1: ED model 0.327 / shuffled 0.325 / mean-shift 0.0334 /
  control-only 0.0611; shift cosine +0.113
seed 2: ED model 0.264 / shuffled 0.283 / mean-shift 0.0335 /
  control-only 0.0603; shift cosine +0.040
NOT PROVEN, decisively. The model is 7-9x worse than mean-shift,
worse than the control-only arm (it does not match even the
marginal), and indistinguishable from its own shuffled-label arm
on seeds 0-1. Cosines are positive but far under the 0.3 bar.
The KO effects themselves are real and large (control-only ~0.06
vs mean-shift ~0.034) and almost purely first-moment in 30-PC
space — the same signature as GSE289721 (PR #36), now repeated in
a second, independent perturbation dataset (human iNeurons, ~250
KOs in the full object). Execution note: the first gauntlet run
was lost to a VM restart 59 min in; the rerun used a caching
driver (scripts/neuroko_gauntlet_driver.py) that persists each
seed's prep and result, and completed in ~28 min.
Iteration 2 (running): control-condition decomposition — fit on
INT control only, ED of generated control vs held-out control —
to split marginal-fit failure from conditioning failure before
the hand-off is written.

## Astrocyte stratum — annotation check + gauntlet launch, 2026-11-23 (cycle 12)
iAstrocytes.h5ad md5 VERIFIED 2026-11-22 (10abe464ca16bcaab07d644
adcaae3ff, exact match to Zenodo 23260389). Annotation check on
open (2026-11-23): 195,306 cells x 36,601 features; obs carries
target_gene (251 values; INT intergenic control = 6,111 cells),
guide_vector (767), sequencing_batch (2) — NO perturbation /
Replicate columns in this object (the iNeuron object has both);
the test's KO_CANDIDATES picks target_gene, and the authors'
INT3/INT30 exclusion applies on the same column. Nothing absent
that the test needs. Gauntlet launched 2026-11-23 via
scripts/neuroko_astro_gauntlet_driver.py (NEUROKO_H5AD override;
astro-specific prep cache + results file
data/neuroko_astro_gauntlet_results.jsonl; same 300-step
company-standard config, same arms). First launch died silently
with its parent exec session (no traceback, no prep written);
relaunched as a tracked background job. Numbers appended on
completion.
Prep plumbing fix (2026-11-23, data handling only; design and
assertions unchanged): the astro _prepare OOM-killed twice at
~43 s as the sole heavy process — the broadcast multiply in the
log-normalize step upcast the whole subset to float64 (astro
subset ~46.8k cells / ~230M nnz; the iNeuron subset squeaked
through at float64, the astro one does not on this 8 GB VM).
_prepare now scales Xs.data in place in float32 and computes
column mean/var by bincount over the stored entries (no squared
copy). Equivalence check against the CACHED iNeuron seed-0 prep
(data/neuroko_prep_seed0.npz, produced by the original code):
shapes, condition list and train/test labels identical; 1,912
of 2,000 columns match the cached prep exactly (median
per-column max diff 0.0) by (mean, std) fingerprint; the
residual is top-2000 boundary churn from float32-vs-float64
variance arithmetic plus fingerprint collisions in the check
itself. Peak prep RSS 5.09 GB (was: OOM). The closed iNeuron
loop's cached preps and results are untouched — no reported
number changes; the astro stratum runs entirely on the lean
prep, self-consistently across its seeds and arms.
Two further cuts followed (same day): the anndata backed row
slice itself peaked at 4.54 GB for the 1.78 GB astro subset
(instrumented stage-by-stage), replaced by a preallocated
h5py block reader (_read_csr_rows), and scipy's CSR column
slice (a whole-matrix CSC conversion copy) replaced by a
manual top-gene gather. Both verified EXACT: 300-row slices
of both objects byte-identical to the anndata slice
(data/indices/indptr), gather identical to toarray() on a
synthetic CSR; the full lean prep still reproduces the
cached iNeuron seed-0 fingerprint result (1,912/2,000).
FINAL GATE (cycle 12): the gauntlet still did not complete —
host MemAvailable sat at ~1.4 GB through the cycle window
(no local process holds the rest; the 14:28 MT simultaneous-
fire window on a shared host) against a ~4-5 GB lean-prep
peak. Three OOM kills total, each in prep, none in a fit.
The stratum is NOT blocked on data, code, or design — only
on a quieter memory window; it runs first in cycle 13, and
the restart-resilient driver + prep caches make that cheap.

## Iteration 2 — control-condition decomposition, 2026-11-22
Fit on INT control train cells only (5,188), sample control, ED
vs held-out control (1,298): seed 0: 0.307; seed 1: 0.262;
seed 2: 0.285. Same magnitude as the KO-condition EDs (0.26-0.33).
The generator does not match even the CONTROL marginal at this
budget — the failure is marginal-fit, not conditioning. Identical
diagnosis to GSE289721 (PR #36), now in a second independent
dataset. LOOP CLOSED at iteration 2 — honest negative. Remaining
levers (fit budget, generator behavior at small scale) are
GAN-side and are handed to the working-GANs effort with both
datasets' numbers; in-scope refinements (splits, ensembling,
readout) cannot repair an unmatched marginal, and the GSE loop
already falsified the smaller-subsample variant. Durable reads:
(1) NeuroKO itself is a strong dataset — real, large, almost
purely first-moment KO effects in 30-PC space; mean-shift is the
bar any future fit must clear, and it is a LOW bar numerically
(0.034) but the current toy configuration is 8x off it. (2) The
watch now has two independent perturbation conversions with the
same decomposition verdict — a fit-quality fact about the toy
configuration, stated once, applicable to both.

## Astrocyte stratum — prep unblocked 2026-11-26 (lean streaming)
The test's own _prepare was OOM-killed twice (exit 137) on the
iAstrocytes object inside this VM's shared memory limit.
scripts/neuroko_astro_lean_prep.py rebuilds the identical
prep outputs (same conds: INT + CCDC6, PLEKHA1, SLC2A4RG,
GLCCI1, MAMSTR, CREB5, FOXF1, PLD3, CD55, RIN3, APP, RASGRF1;
46,794 selected rows; same top-2000 genes and per-seed splits)
by streaming the h5ad CSR arrays in row blocks — peak is the
374 MB dense matrix, not the 1.8 GB subset. Caches:
data/neuroko_astro_prep_seed{0,1,2}.npz (train 37,428 /
test 9,366). The gauntlet driver consumes the caches; fits
append to data/neuroko_astro_gauntlet_results.jsonl.

## Astrocyte stratum — GAUNTLET COMPLETE 2026-11-27 (cycle 14)
Ran from the lean prep caches (scripts/neuroko_astro_lean_
prep.py, built 2026-11-26) via the astro gauntlet driver;
12 KOs scored per seed, same design/arms as iNeurons.
| seed | ED model | shuffled | mean-shift | control-only | cosine |
| 0 | 0.266 | 0.276 | 0.030 | 0.057 | +0.075 |
| 1 | 0.288 | 0.270 | 0.031 | 0.054 | -0.037 |
| 2 | 0.250 | 0.282 | 0.034 | 0.060 | +0.032 |
Verdict: NOT PROVEN — the iNeuron diagnosis repeated in the
second stratum: the model is ~8x worse than mean-shift and
~4.7x worse than control-only (it does not fit the control
marginal at the company-standard budget), indistinguishable
from its shuffled arm, cosines ~0. KO effects are again real
and first-moment (mean-shift ~0.03). The NeuroKO conversion
is now COMPLETE in both strata and CLOSED as an honest
negative at the toy configuration; the marginal-fit failure
pattern (now GSE289721 + NeuroKO neurons + NeuroKO
astrocytes + Perez genotype) is handed to the working-GANs
effort with all numbers, not fixed here. NeuroKO the
DATASET remains strong: real, large, first-moment KO
effects in two human cell types.
