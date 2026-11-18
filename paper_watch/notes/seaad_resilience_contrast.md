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
Model deviation carries a real, stable, weak cognition signal
(clears every control on every seed) but is far under the 0.65 bar.
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
unloosened). Three durable findings:
1. The TARGET validates, weakly: cognition at matched pathology is
   a real axis in the IT-neuron distribution — control-fit deviation
   separates demented from resilient donors at AUROC ~0.58, stable
   across seeds, clearing Gaussian/linear/shuffled arms in the
   single-fit config. Worth conditioning on at production scale;
   at toy scale the effect is small.
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

## Iteration 3 (restore IT compartment + one refinement: ensemble)
Back to the iteration-1 config, plus the SEA-AD loop's proven fix:
deviation computed against the pooled samples of 3 independent
fits per region. The i1 model
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
