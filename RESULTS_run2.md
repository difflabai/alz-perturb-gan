# Real-data run 2: decision-label fix (GPU, pop-os)

**What ran.** The exact command is in `runs/gpu-run-2/launch.sh`, and the raw artifacts are in `runs/gpu-run-2/` (command.json, hardware.json, split.json, results.json, cell-eval results, and the full main.log). It ran on pop-os on 2026-10-13, starting about 14:20 MT, from `decision-label-fix` at d5f7da5. It used 20,000 steps, batch size 256, hidden 128, lr 2e-4, seed 42, 128 training and 128 eval cells per perturbation, eval fraction 0.2, 64 draws, `--device cuda` (RTX A6000), and `--cell-eval`. The code was imported via `PYTHONPATH=/ml2/altgan/decision-label-fix/src`. The shared venv has no `pip`, and its editable install points at `real-data-run-1`. A first launch without PYTHONPATH, which imported the old code, was killed during data loading and its outputs were deleted. Peak GPU memory was 1.47 GiB allocated. The run took 51 min in total: load 19.5 min and GAN training 30.3 min, about 2.5× slower per step than gpu-run-1 because the decision step now runs 8 generator draws per candidate.

**Which "run 1".** The "r1" columns below are **gpu-run-1** (`/ml2/altgan/runs/gpu-run-1`). That is an unpublished 20,000-step GPU run on `real-data-run-1`. Its saved command (`runs/gpu-run-1/command.json`) is identical to run 2's except for `run_dir`. It is **not** the run documented in RESULTS.md and `results/real_data_run_1/`. That one was a 2,000-step **CPU** run with batch size 64 (`results/real_data_run_1/command.json`), and its GAN was weaker: MAE 0.0172, DE overlap 0.0273, decision MRR 0.0132, decision positive rate 0.0. Run 2 differs from the saved CPU run 1 config in device (cuda vs cpu), steps (20,000 vs 2,000), and batch size (256 vs 64). All data, split, cell-cap, hidden, lr, and draw settings are the same.

## Held-out phenocopy ranking (30 held-out targets × 150 library candidates)

| model (r1 = gpu-run-1) | top-1 r1 | top-1 r2 | MRR r1 | MRR r2 | recall@3 r1 | recall@3 r2 |
|---|---|---|---|---|---|---|
| no_change | 0 | 0 | 0.0132 | 0.0132 | 0 | 0 |
| mean_effect | 0 | 0 | 0.0132 | 0.0132 | 0 | 0 |
| additive | 0 | 0 | 0.0459 | 0.0459 | 0 | 0 |
| linear | 0 | 0 | 0.0239 | 0.0239 | 0 | 0 |
| gan_generator (L2 to target) | 0 | 0 | 0.0185 | 0.0194 | 0 | 0 |
| **decision_noul** | 0 | **0.0333** (1/30) | 0.0132 (all-tied) | **0.0534** | 0 | **0.0333** |

Random ranking among 150 candidates gives MRR ≈ 0.037 and recall@3 = 0.02.

## Decision-label diagnostics

| | gpu-run-1 (old geometric label) | run 2 (identity label) |
|---|---|---|
| held-out positive rate | **0.000** (degenerate) | full library 0.0067 (1/150); balanced 0.50 |
| held-out accuracy | 1.000 (meaningless) | balanced 0.50 |
| held-out AUROC | n/a (one class) | full 0.533; balanced 0.506 |
| held-out Brier | 1e-14 | balanced 0.319 |
| mean noul, positives vs negatives | n/a | 0.375 vs 0.346 |
| train-batch positive rate (mean; real-cell candidates) | not recorded here | 0.500 |
| train-batch accuracy (mean; real-cell candidates) | not recorded here | 0.996 |

## Expression (local pseudobulk, held-out)

| model | MAE r1 | MAE r2 | DE overlap@10 r1 | DE overlap@10 r2 | Pearson Δ r2 |
|---|---|---|---|---|---|
| no_change | 0.0141 | 0.0141 | 0.000 | 0.000 | – |
| mean_effect / additive / linear | 0.0143 | 0.0143 | 0.0289 | 0.0289 | 0.161 |
| gan_generator | 0.0145 | 0.0144 | 0.0333 | 0.0289 | 0.157 |

Official cell-eval (vcc profile, mean): GAN overlap_at_N 0.0034 vs 0.0023 for mean_effect; MAE 0.01784 vs 0.01779; discrimination_l1 0.524 for both.

## Read

The label is now well posed. Both classes appear in training (positive rate 0.50) and in evaluation, so the numbers measure something. On training perturbations the head separates the identities almost perfectly (accuracy 0.996). On held-out perturbations it sits at chance: AUROC 0.53 and balanced accuracy 0.50. Its MRR of 0.053 is only slightly above random (0.037) and the additive baseline (0.046), and that is based on one top-1 hit out of 30. This is consistent with the bottleneck: candidates are scored through the generator's predicted profiles. The generator is no better than the mean-effect baseline (MAE 0.0144 vs 0.0143), because unseen perturbations have one-hot embeddings with no training signal, so the generator predicts roughly the same shift for every held-out gene. Next lever (outside this PR's scope): informative perturbation features, so the generator can tell held-out perturbations apart.

## Like-for-like: training vs held-out targets, same inputs and metric

The 0.996 above is training-batch accuracy with **real-cell** candidate profiles. The 0.53 is held-out AUROC with **generator-averaged** candidate profiles, so the two are not comparable. To compare like with like, `scripts/like_for_like.py` loads the run-2 checkpoint (`runs/gpu-run-2/main/checkpoint.pt`) on CPU without using the GPU and re-derives the same split. It scores the 120 training targets and the 30 held-out targets with identical inputs:
- every library candidate is the generator's 64-draw averaged prediction, using seed 45
- the label is candidate == target
- both splits go through `harness.decision_pair_report`, the function behind the existing training check `decision_label_report` and behind `decision_heldout_report`

Raw output is in `runs/gpu-run-2/like_for_like/`.

| targets | n | AUROC (all 150 candidates) | balanced AUROC | balanced accuracy | balanced Brier | mean noul pos / neg |
|---|---|---|---|---|---|---|
| training perturbations | 120 | **0.895** | 0.893 | **0.804** | 0.144 | 0.725 / 0.365 |
| held-out perturbations | 30 | **0.534** | 0.512 | **0.517** | 0.318 | 0.376 / 0.347 |

The held-out row reproduces the GPU run's in-run report (AUROC 0.533, balanced accuracy 0.50). The small differences come from CPU instead of GPU generator sampling.

**Read:** the generator is the main bottleneck. With the same generator-averaged inputs, the head still separates training identities well (AUROC 0.90, even though it was trained on real-cell means). It falls to chance only when the generator has to predict for unseen one-hot identities, which received no training signal. These numbers don't rule out some overfitting by the head, but the 0.90 → 0.53 drop follows from the generator's inputs collapsing for unseen identities.
