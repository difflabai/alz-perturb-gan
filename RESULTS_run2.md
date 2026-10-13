# Real-data run 2: decision-label fix (GPU, pop-os)

Run 2 used the same command, data, split (seed 42), and cell caps as run 1. The only change is the decision-head label and inputs from this branch. It ran 20,000 steps on an RTX A6000, with a peak of 1.47 GiB allocated. The run took 51 min in total: load 19.5 min and GAN training 30.3 min. Training was about 2.5× slower than run 1 because the decision step now pushes 8 generator draws per candidate through the model.

The code was imported via `PYTHONPATH=/ml2/altgan/decision-label-fix/src`. The shared venv has no `pip`, so `pip install -e` was impossible, and its editable install still points at real-data-run-1.

## Held-out phenocopy ranking (30 held-out targets × 150 library candidates)

| model | top-1 r1 | top-1 r2 | MRR r1 | MRR r2 | recall@3 r1 | recall@3 r2 |
|---|---|---|---|---|---|---|
| no_change | 0 | 0 | 0.0132 | 0.0132 | 0 | 0 |
| mean_effect | 0 | 0 | 0.0132 | 0.0132 | 0 | 0 |
| additive | 0 | 0 | 0.0459 | 0.0459 | 0 | 0 |
| linear | 0 | 0 | 0.0239 | 0.0239 | 0 | 0 |
| gan_generator (L2 to target) | 0 | 0 | 0.0185 | 0.0194 | 0 | 0 |
| **decision_noul** | 0 | **0.0333** (1/30) | 0.0132 (all-tied) | **0.0534** | 0 | **0.0333** |

Random ranking among 150 candidates gives MRR ≈ 0.037 and recall@3 = 0.02.

## Decision-label diagnostics

| | run 1 (old geometric label) | run 2 (identity label) |
|---|---|---|
| held-out positive rate | **0.000** (degenerate) | full library 0.0067 (1/150); balanced 0.50 |
| held-out accuracy | 1.000 (meaningless) | balanced 0.50 |
| held-out AUROC | n/a (one class) | full 0.533; balanced 0.506 |
| held-out Brier | 1e-14 | balanced 0.319 |
| mean noul, positives vs negatives | n/a | 0.375 vs 0.346 |
| train-batch positive rate (mean) | ~0 | 0.500 |
| train-batch accuracy (mean) | ~1.0 (trivial) | 0.996 |

## Expression (local pseudobulk, held-out)

| model | MAE r1 | MAE r2 | DE overlap@10 r1 | DE overlap@10 r2 | Pearson Δ r2 |
|---|---|---|---|---|---|
| no_change | 0.0141 | 0.0141 | 0.000 | 0.000 | – |
| mean_effect / additive / linear | 0.0143 | 0.0143 | 0.0289 | 0.0289 | 0.161 |
| gan_generator | 0.0145 | 0.0144 | 0.0333 | 0.0289 | 0.157 |

Official cell-eval (vcc profile, mean): GAN overlap_at_N 0.0034 vs 0.0023 for mean_effect; MAE 0.01784 vs 0.01779; discrimination_l1 0.524 for both.

## Read

The label is now well posed. Both classes appear in training (positive rate 0.50) and in evaluation, so the numbers measure something. On training perturbations the head separates the identities almost perfectly (accuracy 0.996). On held-out perturbations it sits at chance: AUROC 0.53 and balanced accuracy 0.50. Its MRR of 0.053 is only slightly above random (0.037) and the additive baseline (0.046), and that is based on one top-1 hit out of 30. This is consistent with the bottleneck: candidates are scored through the generator's predicted profiles. The generator is no better than the mean-effect baseline (MAE 0.0144 vs 0.0143), because unseen perturbations have one-hot embeddings with no training signal, so the generator predicts roughly the same shift for every held-out gene. Next lever (outside this PR's scope): informative perturbation features, so the generator can tell held-out perturbations apart.
