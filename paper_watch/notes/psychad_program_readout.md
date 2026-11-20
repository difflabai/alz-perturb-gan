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
2. Fit the small CPU EMPF wrapper on the two scores. Match donor
   splits, initialization seeds, draw counts, and the 7,000-update
   cosine schedule across rates. The grid uses generator rates
   0.0025, 0.005, and 0.012. Critic and prior rates start at 1.5 and
   2.5 times the generator rate. Record scores during a continuous
   fit at 300, 1,200, and 2,400 updates. Verify the regularizer
   counter at each checkpoint with `lazy_k=1`.
3. Compare 256 generated score pairs per group with the observed
   held-out pairs. Compare the same held-out data with 256 pairs
   resampled from training donors. Energy distance measures the
   overall joint fit. Correlation error is the absolute difference
   between generated and observed Spearman correlations. Both
   summaries weight the three diagnoses and three splits equally.

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

The command prints aggregate and per-split scores, the input SHA-256,
and the rate and regularizer records. The per-group spread ratio
compares the root sum of the two generated score variances with the
same quantity in held-out donors. Use `--learning-rates 0.005`,
`--checkpoints 300 1200`, or `--seeds 0` to run a subset.

## What the run showed

| Generator rate 0.005 | Energy distance ↓ | Correlation error ↓ |
|---|---:|---:|
| 300 updates | 0.159 | 0.496 |
| 1,200 updates | 0.074 | **0.196** |
| 2,400 updates | **0.067** | 0.251 |
| Empirical training-donor sampling | 0.073 | 0.207 |

The observed correlation in controls averaged 0.004. The generated
correlation fell from 0.617 at 300 updates to 0.014 at 1,200. In the
tested grid, 1,200 updates at 0.005 gave the best relationship
score, while 2,400 at the same rate gave the best joint distribution
score. Reporting both helps distinguish a better match overall from
a more faithful relationship between the two features.

These are exploratory checkpoint comparisons in a two-score
representation. The three splits reuse donors and each holds out
nine Lewy body disease donors. Use the values to diagnose the
readout and compare training choices on this input.
[Per-split results and input provenance](psychad_program_readout_results.json)
