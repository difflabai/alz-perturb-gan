# SEA-AD LL microglia readout — iteration log

Find: Arif et al. 2026 (Research Square, doi 10.21203/rs.3.rs-10721231/v1).
Data: SEA-AD AnG, local (annotations + expression already on disk from
cycle 1). Compartment: subclass "Immune" (ABC taxonomy home of the
microglia/PVM supertypes) as the toy stand-in for the paper's
Microglia/PVM-lineage object. LL burden per donor from the paper's
operational rule (joint high-high module scores, q0.75/q0.75 thresholds
taken from the CTRL distribution).

## Iteration 1, attempt 0 — loader seam (fixed on our end, not a loop iteration)
First launch died instantly: "only 0 module genes found". SEA-AD var
index is Ensembl IDs; gene symbols live in var["gene_symbol"]. Fixed
the loader to read symbols from that column; relaunched.

## Iteration 1 RESULT (seed 0, 43 donors)
model rho **+0.193**, Gaussian +0.120, linear +0.151, shuffled −0.361.
Right direction, model clears every arm — but below the 0.3 bar and
the margins over Gaussian/linear are thin (+0.07/+0.04). Only 43 of 84
donors pass the >=40-microglia threshold in AnG: donor count is the
power bottleneck, not the estimator.

## Iteration 2 (one change — data handling)
MIN_CELLS_PER_DONOR 40 -> 25. More donors in the Spearman at slightly
noisier per-donor ED; no change to the readout, fit, or thresholds.
RESULT (seed 0): model rho **+0.134**, Gaussian +0.120, linear +0.151,
shuffled −0.413, n_donors **43 — unchanged**. No AnG donor has 25–39
Immune cells (counts are bimodal), so the refinement was a no-op on
the donor set; Gaussian/linear rhos are bit-identical to iteration 1,
confirming it. The model rho moved 0.193 -> 0.134 on an identical
donor set = fit noise of roughly ±0.06 at this scale. Standing read:
direction persists (model >> shuffled, ≈ Gaussian/linear), effect is
weak and noisy at n=43. Next lever when the loop resumes: pool LEC
donors with per-region burden/deviation (donor n is the wall), and run
seeds 1–2 before any verdict.

## Iteration 3 (one change — data handling: pool LEC donors)
AnG-only iterations 1–2 were donor-starved (n=43 at any threshold).
Iteration 3 pools LEC: burden, PCA and fits stay per-region; the
Spearman unit is donor x region (both quantities are region-specific
measurements). Expected pooled n ~80. If rho still wobbles at the
larger n, iteration 4 imports the SEA-AD loop's fix (3-fit ensemble).
RESULT — GAUNTLET (n=77 donor x region units; AnG 43 + LEC 34):
model {+0.223, +0.142, +0.241} mean **+0.202**; Gaussian mean
**+0.228**; linear centroid **+0.248** (deterministic — identical
every seed); shuffled mean +0.038. Every real arm was above the single
shuffled draw on every seed, and the no-fit linear baseline matched or
beat both fitted arms on all three seeds. See the inference revision
below: the scored donors included the CTRL fit donors, deviations were
not depth-matched, and the shuffled arm was one unit-level draw, so
these numbers do not establish an association.

## LOOP CLOSED at iteration 3 — honest partial/negative
Not proven as a model readout (assertions require model > 0.3/seed
and > linear mean; neither holds, assertions unloosened). Closing
early rather than spending iterations 4–10: the remaining candidate
levers don't survive scrutiny — ensembling attacks fit variance, but
the arm to beat (linear) has no variance; a module-subspace deviation
readout would be semi-circular (burden is computed from the same
module genes). The non-circular version — burden from the lipid half
of the module, deviation read on the lysosome half, and vice versa
(cross-module split) — changes the claim under test and is queued as
a possible future conversion, not an iteration of this one.
Exploratory observation, pending a rerun: donor LL burden tracked
centroid distance at rho ~ +0.25 over n=77 donor x region units
(43 + 34, so many donors are counted twice). That value was computed
with in-sample CTRL donors and without depth matching, so it does not
validate the axis as a target. As a ParticleGAN readout at toy scale,
the model added nothing over a centroid. Production-scale fits (20k steps, stage conditioning) are
a different regime the toy cannot speak to.

## Inference revision (2026-11-20, code only, not yet rerun)
The SEA-AD inputs are not on the current machine, so the recorded
numbers above have not been rerun under these changes.
- Held-out CTRL donors: iterations 1-3 fit the PCA, burden thresholds,
  ParticleGAN, Gaussian and centroid on all CTRL cells and then scored
  those same CTRL donors. Their deviation is low by construction (the
  arms were fit on them) and the burden thresholds were set on their own
  cells. Half of the CTRL donors (a seeded hash of the donor
  label, the same side in both regions) are now fit donors. Gene
  selection, PCA, standardization, burden thresholds and every arm use
  fit donors only. Scored units are held-out CTRL plus all non-CTRL
  donors, and rho over non-CTRL donors only is also reported.
- Depth bias: the harness energy distance was a V-statistic that grows
  as a donor's cell count falls. It now defaults to the unbiased
  U-statistic, and every scored donor is subsampled to
  MIN_CELLS_PER_DONOR (25) cells for the energy-distance and centroid
  arms. The Spearman of raw cell count with burden is reported.
- Shuffled arm: one permutation per seed is replaced by a donor-grouped
  permutation p-value (2000 permutations; a donor is exchanged only
  with donors present in the same regions, and its AnG and LEC burdens
  move together). Arm comparisons use donor-bootstrap CIs of the rho
  difference.
- ADNC is mapped explicitly ({Not AD, Low} -> CTRL, Intermediate ->
  EARLY, High -> AD) and a missing or unknown value raises.

## Iteration 1 — running
Config: PGAN fit on CTRL microglia only (unmodified EMPF recipe,
300 steps), PC space (top-1500 var genes + 16 module genes, 30 PCs).
Readout: per-donor energy distance to model samples (donors with
>= 40 microglia), Spearman vs donor LL burden. Arms: shuffled burden,
Gaussian sampler, linear centroid distance.

## Applicability to the default microglia model

The default EMPF archive and all nine development `hvg2000` fold
panels contain **PLIN2 only (1/7 lipid markers, 0/9 lysosome markers)**.
This makes the cross-module analysis unavailable for the retained model.
The exploratory centroid result above comes from a separate SEA-AD
analysis with its own gene panel.

From the repository root, check the marker coverage of any development
cache or generated archive with the panel diagnostic:

```sh
python -m paper_watch.alztest.ll_panel \
  runs/improve_20261109/caches/hvg2000/fold*/space.npz \
  runs/microglia_empf/verification/final_generated.npz
```

A prospective test needs a panel with both modules, a training-only
reference, donor-level holdouts, and the centroid control. Define burden
with one module and measure deviation in the other to avoid scoring
against the genes used to construct the target.
