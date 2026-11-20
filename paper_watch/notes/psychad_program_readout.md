# PsychAD lipid and lysosome program readout

An expression generator can make two gene programs appear more tightly
linked than they are in the public data. This diagnostic checks whether
EMPF reproduces how lipid and lysosome expression vary together across
held-out microglia donors. It also shows how a researcher can use named
gene programs as a readable check on a generative model.

## What the command does

1. Use the published PsychAD MSSM microglia extract and sum counts per
   donor. Convert to log counts per 10,000 and use training donors to
   standardize the markers. Average the standardized marker values
   within each program to get two scores per donor.
2. Choose the EMPF configuration without the held-out donors. Within
   each split's training donors, hold out a validation set: a quarter
   of each diagnosis, capped so that 20 fitting donors remain. Lewy
   body disease has 25 training donors, so validation keeps 5 of them.
   Standardize on the fitting donors only. Fit the small CPU EMPF
   wrapper at generator rates 0.0025, 0.005, and 0.012. Critic and
   prior rates start at 1.5 and 2.5 times the generator rate, on the
   7,000-update cosine schedule. Record each fit at 300, 1,200, and
   2,400 updates, and verify the regularizer counter with `lazy_k=1`.
   Select the rate and checkpoint with the lowest mean validation
   correlation error, breaking ties by energy distance.
3. Refit only the selected configuration on all training donors of
   the split. Score it on the held-out donors next to three control
   arms. The shuffled-label arm is EMPF with the same rate,
   checkpoint, and seed, fit on permuted diagnosis labels. The
   mean-shift arm adds each diagnosis's training mean shift to
   training control donors. The empirical arm resamples training
   donors. Each arm gives 256 score pairs per group. Energy distance
   measures the overall joint fit. Correlation error is the absolute
   difference between generated and observed Spearman correlations.
   Both summaries weight the three diagnoses and three splits equally.
4. Bootstrap the held-out donors 2,000 times, stratified by diagnosis.
   Each replicate draws donor multiplicities once for the cohort, so a
   donor held out in two splits is resampled consistently. Generated
   draws stay fixed. The intervals describe held-out donor sampling
   only, not refitting or training-donor variation. Each arm gets a
   bootstrap SE. Each EMPF-minus-control difference also gets a 95%
   percentile interval. Repeated donors shrink within-sample
   distances and inflate both metrics within a replicate, so marginal
   percentile intervals would sit above the point estimates. Paired
   differences share the observed side, so that bias largely cancels.
5. As an observed-versus-observed reference, split each held-out group
   into random halves 200 times and score one half against the other.

The input covered 64,816 nuclei from 619 donors. The diagnostic uses
the seven lipid markers in the [panel check](../alztest/ll_panel.py)
and eight lysosome markers, with `GBA1` matched by Ensembl ID. The
pinned extract lacks `CTSD`. This is a public-data computation on the
[PsychAD MSSM extract](psychad_cross_disorder.md#protocol).

## Run

From the repository root in the [paper-watch environment](../README.md#running-locally),
use the public H5AD produced by the
[PsychAD fetch script](../scripts/fetch_psychad.py).

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m paper_watch.alztest.psychad_programs \
  --input paper_watch/data/psychad-mssm-microglia.h5ad \
  > paper_watch/data/psychad-program-results.json
```

The command prints the per-split selection records, aggregate and
per-split scores for each arm, the bootstrap records, the split-half
reference, the input SHA-256, and the rate and regularizer records.
The per-group spread ratio compares the root sum of the two
generated score variances with the same quantity in held-out
donors. Use `--learning-rates 0.005`, `--checkpoints 300 1200`, or
`--seeds 0` to narrow the selection grid or run fewer splits, and
`--bootstrap-replicates` to change the bootstrap size.

## What the run showed

The inner validation split chose a different configuration in each
split. It chose generator rate 0.012 at 1,200 updates for split 0,
0.0025 at 1,200 for split 1, and 0.0025 at 2,400 for split 2.

| Held-out arm | Energy distance ↓ (SE) | Correlation error ↓ (SE) |
|---|---:|---:|
| EMPF, selected configuration | 0.081 (0.023) | 0.293 (0.062) |
| Shuffled-label EMPF | 0.086 (0.025) | 0.229 (0.057) |
| Mean shift | 0.059 (0.023) | 0.177 (0.057) |
| Empirical training-donor sampling | 0.073 (0.019) | 0.207 (0.059) |
| Observed split-half reference | 0.156 | 0.323 |

| EMPF minus | Energy distance [95% interval] | Correlation error [95% interval] |
|---|---:|---:|
| Shuffled-label EMPF | −0.005 [−0.030, 0.020] | 0.064 [−0.068, 0.140] |
| Mean shift | 0.022 [−0.016, 0.063] | 0.116 [0.027, 0.200] |
| Empirical sampling | 0.008 [−0.012, 0.029] | 0.087 [0.039, 0.137] |

With the configuration chosen on training donors, EMPF did not match
the observed relationship between the two programs better than any
control. Its correlation error was larger than the mean-shift and
empirical arms, and those intervals exclude zero. The intervals
cover only held-out donor resampling. They leave out refitting, and
nine Lewy body disease donors per split dominate the correlation
term. The shuffled-label and energy distance differences are within
donor uncertainty. The split-half reference compares half-size
groups, so it is noisier than any arm. Read it as a scale for
donor-level variation, not as a floor that an arm must clear.

An earlier version of this diagnostic scored the full rate and
checkpoint grid on the held-out donors. At rate 0.005, 1,200
updates had the lowest correlation error (0.196), and 2,400 updates
had the lowest energy distance (0.067). Those comparisons used the
evaluation donors to choose between configurations. Their gaps to
empirical sampling (0.207 and 0.073) are smaller than the donor
uncertainty above, so they do not identify a better checkpoint.
The selected-configuration result above replaces them.

These are exploratory readouts in a two-score representation. The
three splits reuse donors, and inner validation sees only five Lewy
body disease donors, so selection is itself noisy. Use the values to
diagnose the readout and compare training choices on this input.
[Per-split results and input provenance](psychad_program_readout_results.json)
