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

## Iteration 1 — running
Config: PGAN fit on CTRL microglia only (unmodified EMPF recipe,
300 steps), PC space (top-1500 var genes + 16 module genes, 30 PCs).
Readout: per-donor energy distance to model samples (donors with
>= 40 microglia), Spearman vs donor LL burden. Arms: shuffled burden,
Gaussian sampler, linear centroid distance.
