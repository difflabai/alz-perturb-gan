# SEA-AD resilience contrast — iteration log

Find: Tosoni et al. 2026 (Cell Stem Cell) + Lu et al., bioRxiv
2026-10-28 (doi 10.64898/2026.10.26.761376). Route: NEW TARGET —
cognition at matched pathology. High-ADNC donors in local SEA-AD:
31 demented / 11 no-dementia. Compartment: cortical IT neurons.
Readouts: mito-program score (paper's direction: suppressed in
dementia) and control-fit model deviation, AUROC demented vs
resilient within High-ADNC, pooled AnG + LEC (donor x region units,
>=40 IT cells/donor). Arms: shuffled cognition, Gaussian,
centroid-linear, program-score-only (the bar both closed loops
point at: the model must beat the named program alone).
Assertions: model AUROC > 0.65 every seed, mean > shuffled + 0.15,
mean > program-only, mean > Gaussian.

Loader seam fixed on our end before iteration 1: seaad_annotations
did not carry Cognitive Status; the loader now also returns
`cognition` from disease.csv (additive).

## Iteration 1 RESULT (IT neurons; n=39 donor x region units)
Attempt 0 OOM-killed twice (exit 137 — IT compartment ~10x the
microglia one); seam fix = cap 600 cells/donor before densifying
(our plumbing, not a loop iteration). Gauntlet: model {0.574, 0.606,
0.560} mean **0.580**; Gaussian 0.546; linear 0.554; shuffled 0.426.
Model deviation was above every control arm on every seed, but far
under the 0.65 bar. See the inference revision below: these arms are
not calibrated, so this is not evidence of a signal.
**Program AUROC = 0.489 (deterministic, identical every seed): the
paper's mito-suppression direction does not reproduce in IT neurons
under this scoring at all.** The model signal is therefore NOT the
named program — something else in the IT distribution shifts with
dementia status at matched pathology.

## Iteration 2 RESULT (all-neuron compartment)
Model {0.540, 0.526, 0.574} mean **0.547** — WORSE than iteration
1's 0.580; program still a coin flip (0.497); seed 2's shuffled arm
hit 0.600 (small-n null noise at 14 resilient units). Widening the
compartment diluted the signal: the cognition-associated shift
concentrates in IT neurons, and interneuron/ET/CT distributions add
variance, not signal. Iteration 2's change is DISCARDED.

## Iteration 3 RESULT (IT restored + 3-fit pooled ensemble)
Model {0.571, 0.560, 0.540} mean **0.557** — ensembling bought
NOTHING here (iteration 1's single-fit mean was 0.580; seed 2 dips
below the linear arm). Unlike the SEA-AD ordering readout, this
signal is not fit-variance-limited.

## LOOP CLOSED at iteration 3 — honest partial/negative
Not proven (0.65/seed bar unmet in every iteration; assertions
unloosened). Three exploratory observations, pending a rerun under
the revised inference below:
1. Exploratory only: control-fit deviation ranked demented above
   resilient donors at AUROC ~0.56-0.58 across seeds. "Stable" here
   means stable across fit seeds on fixed data, not across donor
   samples. At 39 units the null AUROC has an SD of about 0.10, so
   this is within about one null SD of 0.5 even before accounting for
   donors that appear in both regions. It does not establish that
   cognition at matched pathology is an axis of the IT-neuron
   distribution.
2. The paper's named PROGRAM does not replicate in SEA-AD: the
   mito-suppression score is a coin flip in IT neurons (0.489) and
   in all neurons (0.497), deterministic in both. A cross-cohort
   caution, not a refutation — Lu et al.'s cohort, regions, and
   exact MBS gene set differ; mine is prefix-harvested.
3. The binding constraint is donors, not machinery: 11 resilient
   donors is the entire SEA-AD cohort. No data-handling lever
   creates more; more regions add correlated units, not donors.
   The contrast wants a bigger resilient-donor cohort (ROSMAP /
   PsychAD-class) — which loops back to the access blockers.

## Inference revision (2026-11-20, code only, not yet rerun)
The review of the closed loop found four problems with how iterations
1-3 were scored. The test now addresses each one; the recorded numbers
above were produced before these changes and have not been rerun,
because the SEA-AD inputs are not on the current machine.
- Depth bias: the harness energy distance was a V-statistic, so a donor
  with fewer cells reads as farther from any reference (in 30-d
  N(0, I), about +0.33 at 25 cells vs +0.03 at 600 for identical
  distributions). The centroid norm has the same small-n inflation.
  Donors had 40-600 IT nuclei. The harness now defaults to the
  unbiased U-statistic, and every scored donor is subsampled to
  MIN_CELLS_PER_DONOR cells. The AUROC of raw cell count is reported
  as a confound check.
- Shuffled arm: one label permutation per seed is one null draw (one
  such draw reached 0.600 in iteration 2). The arm is now a
  donor-grouped permutation p-value (2000 permutations; a donor's AnG
  and LEC units keep one label), and arm comparisons use
  donor-bootstrap CIs of the AUROC difference.
- Ensemble: iteration 3 pooled the three fits' 12,288 draws and the
  energy distance then subsampled 512 rows with a fixed seed (about
  170 per fit), so "ensembling bought NOTHING" is not a clean test of
  ensembling. Each fit's energy distance now uses all of its draws,
  and the three distances are averaged.
- ADNC: the AD/CTRL/other mapping used substring matches with a
  catch-all. It is now explicit and raises on a missing or unknown
  value.

## Iteration 3 (restore IT compartment + one refinement: ensemble)
Back to the iteration-1 config, plus the SEA-AD loop's proven fix:
deviation computed against the pooled samples of 3 independent
fits per region (pooled, not averaged; see the inference revision). The i1 model
signal (0.580, stable, clears all controls) is variance-limited at
39 units; ensembling bought +0.28 mean rho in the SEA-AD readout.
If the ensemble cannot lift AUROC materially toward 0.65, the
remaining gap is the effect size itself, not fit noise — and the
honest close is near.

## Iteration 2 (one change — data handling: compartment)
Widen IT-only -> ALL neuronal subclasses (exclude Astrocyte,
Endothelial, Ependymal, Immune, OPC, Oligodendrocyte,
VLMC & Perivascular). Lu et al. profiled neurons broadly; if the
program effect concentrates in interneurons / ET / CT types, the
IT-only compartment structurally misses it.

## Iteration 1 — running (3-seed gauntlet on the initial config)

## Use in ALZ

This is an opt-in cognition diagnostic for a neuronal cohort. The
committed runner reproduces the iteration 3 ensemble configuration
(recorded mean AUROC 0.557) with the revised inference, which has not
been rerun. The single-fit 0.580 is the historical iteration 1 result.
Both recorded values predate the depth-matched, donor-grouped scoring. A prospective comparison should fit and evaluate on
separate donors and keep records from the same donor together across
regions, including for the shuffled control.

From `paper_watch/` with the two local SEA-AD region inputs and a
`Cognitive Status` column in `data/disease.csv`:

```sh
python scripts/run_once.py test_seaad_resilience_contrast 0 300
```
