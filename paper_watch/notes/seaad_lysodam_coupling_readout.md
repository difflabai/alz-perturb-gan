# SEA-AD lysosome-DAM coupling readout — iteration log

Find: "Evolutionarily conserved coupling of lysosomal response
sensing and disease-associated microglial reprogramming scales with
pathological timescales", bioRxiv 2026-11-19 (preprint; ALZ paper
watch cycle 9 top find, 2026-11-21). Convergent: GPNMB lysosomal
program (bioRxiv 2026-11-20); MITF/TFE DAM signature under
lysosomal dysfunction (Immunity 2026).

Data: SEA-AD AnG + LEC, local (same objects as cycles 1-2).
Compartment: subclass "Immune" (microglia/PVM supertypes).
Quantity: per-donor coupling = Pearson r across the donor's
microglia between MiT/TFE lysosomal-biogenesis module score and
DAM module score (both z-scored vs CTRL donors). Amplification =
donor coupling − CTRL mean coupling. Paper's SEA-AD Delta: +0.072.

Design: unconditional ParticleGAN (unmodified EMPF, 300 steps) on
CTRL microglia per region, 30-PC space (top-1500 var genes + 22
program genes forced in). Readout: per-donor energy distance to
model samples vs coupling amplification, Spearman over donor x
region units. Arms: Gaussian sampler, linear centroid distance,
shuffled coupling. Bar (house standard): rho_model > 0.3 every
seed and clear of every arm. Replication check reported, not
asserted: AD-minus-CTRL mean coupling should be small positive.

## Iteration log
- Iteration 0 (data/design) — DONE 2026-11-21: test written
  (tests/test_seaad_lysodam_coupling_readout.py); program genes
  asserted present at load (>=16 of 22 or DataUnavailable skip).
- Iteration 1 — queued behind the GSE289721 iteration-3 gauntlet
  on the shared local VM; numbers appended on completion.

## Iteration 1 — single fit per region (completed 2026-11-21, cycle 9)
n = 77 donor x region units (AnG + LEC).
Replication check: AD-minus-CTRL mean coupling Delta = **+0.046**
(paper's SEA-AD Delta +0.072 — same sign, same order; toy
compartment is the Immune subclass, theirs microglia/PVM nuclei).
| seed | rho model | rho gauss | rho lin | rho shuffled |
| 0 | +0.255 | +0.432 | +0.030 | +0.014 |
| 1 | +0.341 | +0.403 | +0.030 | +0.057 |
| 2 | +0.400 | +0.400 | +0.030 | -0.063 |
NOT PROVEN: model below the 0.3 bar on seed 0 and below the
Gaussian on seeds 0-1. Two structural facts stand out: the linear
centroid arm is DEAD (+0.030 — this signal is not a mean shift,
a first for this watch), and the null is calibrated (~0).

## Iteration 2 — 3-fit ensemble (one change: fit averaging)
| seed | rho model | rho gauss | rho lin | rho shuffled |
| 0 | +0.401 | +0.432 | +0.030 | +0.093 |
| 1 | +0.417 | +0.403 | +0.030 | +0.035 |
| 2 | +0.375 | +0.400 | +0.030 | -0.167 |
Ensembling lifted the model mean +0.332 -> +0.398 (the SEA-AD
iteration-9 effect again); model now clears 0.3, linear and
shuffled on every seed, and clears the Gaussian on seed 1 only.

## LOOP CLOSED at iteration 2 of 10 — honest partial (closest yet)
Not proven: the assertions require clearing EVERY arm every seed;
the Gaussian sampler stays ~+0.01-0.03 ahead on seeds 0 and 2,
unloosened. In hindsight the Gaussian's edge is structural:
coupling IS covariance, and the Gaussian arm is the closed-form
covariance model — the honest summary is that at toy scale the
fitted generator recovers the second-order structure about as
well as the covariance it would need to beat, and no better.
Closing early: remaining levers (more fits, more steps) are
variance/budget levers against a variance-free closed-form arm.
Durable findings: (1) the paper's coupling Delta replicates in
sign and order on our local SEA-AD objects (+0.046 vs +0.072);
(2) coupling amplification is the first target in this watch
whose signal is genuinely second-order (linear arm +0.03, where
SEA-AD/LL/resilience/GSE signals were all first-moment);
(3) a control-fit ParticleGAN's deviation tracks it at rho ~0.40
— real, but not beyond a Gaussian at this scale. Flagged for
domain-expert review per house rules (toy test, no biological
claim rests on it alone); production-scale fits are a different
regime the toy cannot speak to.
