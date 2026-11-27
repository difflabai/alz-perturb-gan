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

## Iteration 3 — split control-vs-AD Delta readout + GPNMB arm
(2026-11-22 deep-pass design; run completed 2026-11-23, cycle 12)
Design change (readout only): coupling estimated SEPARATELY in
control vs AD donors; the disease Delta is reported against a
donor-label permutation interval (200 perms, p95 of |Delta|).
AD-only readout: per-AD-donor deviation (3-fit ensemble, as in
iteration 2) vs that donor's coupling CENTRED on the AD mean —
removing the pooled baseline coupling the deep pass warned the
iteration 1-2 rho mostly re-measured. A GPNMB-only coupling arm
(lyso score vs GPNMB alone) is reported alongside. n = 39 AD
donor x region units. Raw: data/lysodam_split_results.jsonl.
| seed | Delta AD-CTRL | perm p95 | rho model | rho gauss | rho lin | rho shuffled | rho GPNMB-coupling |
| 0 | +0.049 | 0.059 | +0.279 | +0.303 | +0.140 | -0.115 | +0.546 |
| 1 | +0.049 | 0.064 | +0.240 | +0.364 | +0.139 | +0.008 | +0.510 |
| 2 | +0.049 | 0.064 | +0.291 | +0.288 | +0.139 | +0.293 | +0.513 |
(Delta is fit-independent, hence identical across seeds.)
Verdict: NOT PROVEN, and the split design explains the pooled
result. (1) The disease Delta itself (+0.049) sits INSIDE the
permutation interval (p95 0.059-0.064): at n=39 AD units the
Delta is not resolvable from donor-label noise — a null here
AGREES with the paper (their SEA-AD Delta is +0.072, small by
design), it does not contradict it. (2) AD-only model rho drops
to 0.24-0.29 (from pooled 0.38-0.42): most of the pooled signal
was baseline coupling, exactly the deep-pass correction.
(3) The GPNMB-only coupling is the surprise: model deviation
tracks it at rho 0.51-0.55 on every seed, well clear of the
full-program coupling — consistent with the cycle-9 GPNMB brief
(the lysosomal subprogram, not DAM as a whole, is the severity
signal). Hypothesis-generating only: GPNMB coupling is a
single-gene correlation (noisier per-donor), the shuffled arm
is unstable at this n (seed 2: +0.293), and no assertion was
pre-registered for this arm. Named follow-up, not a claim: a
pre-registered GPNMB-coupling readout on the Perez/LPS-IFN
microglia or a larger SEA-AD compartment, if the queue opens.
Loop status: the LysoDAM follow-up is COMPLETE as designed;
the target stays an honest partial. No biological claim rests
on this toy test alone (house rule).

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

## Iteration 4 — PRE-REGISTERED 2026-11-26 (before any run)
Question: does the GPNMB-only DAM-proxy coupling, which tracked
model deviation at rho {0.546, 0.510, 0.513} as a REPORTED arm in
iteration 3, hold as an ASSERTED readout under the iteration-1
bars? Estimand: pooled Spearman rho(control-fit model deviation,
donor GPNMB-coupling amplification = donor coupling minus CTRL
mean coupling), same compartment (SEA-AD Immune, AnG+LEC), same
3-fit ensemble deviations as iterations 1-3, coupling computed
with the DAM score replaced by GPNMB alone (the iteration-3
reported arm, unchanged). Bars, every seed: rho_model > 0.3,
rho_model > rho_gauss, rho_model > rho_lin, rho_model >
rho_shuffled. External support logged before the run: the
2026-11-20 GPNMB lysosomal-program preprint and Luquez et al.
Nature 2026 (cycle 13) making GPNMB a cross-population signal.
No other change from iterations 1-3. If it clears, the loop's
verdict upgrades to PROVEN for the GPNMB estimand only; the
full-DAM-program estimand stays at its iteration 1-3 verdicts.

## Iteration 4 — RESULT 2026-11-26 (pre-registered estimand)
Pooled GPNMB-coupling amplification, 77 donor x region units:
seed 0: rho_model 0.152 vs gauss 0.179, lin -0.039, shuffled
0.007 — FAILS the 0.3 bar and the Gaussian arm. Seed 1:
rho_model 0.375 vs gauss 0.172, lin -0.039, shuffled 0.159 —
clears every bar. Seed 2 pending at log time. Verdict: NOT
PROVEN under the pre-registered every-seed bars; the estimand
is seed-unstable (0.152 / 0.375). The iteration-3 reported
0.51-0.55 was a different, weaker estimand (AD-subset RAW
coupling, not pooled amplification) — it does not survive the
stricter pre-registered form. Logged as the loop's iteration-4
entry; the GPNMB question stays open, the claim does not
advance.

## Iteration 4 — seed 2 COMPLETE 2026-11-27 (cycle 14)
Seed 2: rho_model 0.046 vs gauss 0.127, lin -0.039, shuffled
-0.051 — fails every bar. Full pre-registered gauntlet:
0.152 / 0.375 / 0.046. Verdict stands NOT PROVEN and is now
complete: the pooled GPNMB-coupling-amplification estimand
is seed-unstable (one seed in three clears), and the
iteration-3 AD-subset raw-coupling arm (0.51-0.55) does not
survive the stricter pre-registered form. The GPNMB question
is closed as a pre-registered negative for THIS estimand on
THIS compartment; the external target validation (cycle-9
preprint, Luquez Nature 2026) is untouched — what failed is
the toy readout's stability at n=77, not the biology. No
further LysoDAM iterations: the loop is CLOSED (iterations
1-4 executed; honest partial for the full-program estimand,
honest negative for the GPNMB estimand).
