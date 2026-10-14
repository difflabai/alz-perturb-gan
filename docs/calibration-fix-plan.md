# Calibration-fix plan (before seeds 43/44) — draft, no code changed

Commit 051a50df. Evidence: seed-42 run `runs/seed42-20261014T065738935604Z`. CPU-only until §5 passes.

## 1. Diagnosis
- **Loss is offset-free.** `particlegan/gan_loss.py:17,23` (RpGAN): `softplus(-(D(real)-D(fake)))`. Adding any constant to D leaves both losses unchanged, so the absolute logit level drifts freely. Seed 42 shows it at step 1000: the real and fake logit means are *equal* within each arm, but the shared level differs:
  - B0/H: −4.44
  - G/GH: +0.41
  - shuffle: −0.93
- **Offset-free critic used as an absolute gate.** `gene_training.py:161`: `realness = d(generated_cells, cond).sigmoid()...mean(1)`. At `:165` that realness multiplies the head's fake-pair BCE. So the gate weight is set by the drifting offset (B0 0.0124 vs G 0.5997 at step 1000; both ≈0.52 by step 2000), not by realism. Arms therefore get different effective amounts of generated-pair training. That confounds the G-vs-B0 comparison.
- **Head sigmoid reported as a probability.**
  - `gene_models.py:70-71`: `GeneHead.forward = logits(...).sigmoid()`.
  - `gene_evaluation.py:171`: builds the score matrix from that sigmoid.
  - `gene_evaluation.py:178,191`: `probabilities=True`.
  - `score_summary`, `gene_evaluation.py:62-64,70-73`: computes balanced accuracy at 0.5, Brier and calibration bins on it.
  - The head is not relativistic, but it is trained on 50/50 pairs (`gene_training.py:140-142`) and scored against a 1/150 library prior. The raw sigmoid is therefore miscalibrated by construction: every full Brier is above the 0.00662 base rate, and H/GH balanced Brier is ≈0.48.
- Rank metrics (AUROC, MRR, top-1) are per-target and invariant to a per-row offset, so they are not invalidated. They are confounded only through the gate above.

## 2. Option A: relative scores only
- **Scoring:** within each target row, use the head logit minus the row mean, or a softmax over the 150 candidates, `p = softmax(logits/τ=1)`. Report full Brier and reliability on that softmax; it is a proper distribution against the 1/150 prior. Ranks are unchanged. Drop balanced accuracy and balanced Brier at the 0.5 threshold.
- **Gate:** use a relativistic realness, `realness = σ(D(gen) − D(real_ref))` averaged over `decision_cells`. Here `real_ref` is the critic on the already-sampled real candidate cells for the same identity and condition (`rows`, `:143`). The gate is offset-invariant and sits at ≈0.5 at equilibrium. The raw `σ(D)` is never used as a probability or gate.

## 3. Option B: separately calibrated head
- Hold out 16 of the 96 FIT identities: `torch.randperm(96, generator=seed 1729)[:16]`, sorted and hashed into the receipt. They are excluded from **head (J) pair sampling only**. Ambiguous: whether G and D should also drop them. Default is J only, so G is unchanged; flag this for Ember.
- After training, freeze all models and score the 16 calibration targets against the full library.
- Fit temperature or Platt scaling (2 params) on the row-softmax or logits. Use isotonic regression only if Platt fails reliability. Freeze the calibrator and record its sha in the receipt.
- Then evaluate on inner-24 and exploratory-30. The calibration split is never inner or exploratory, and is never used for selection.

## 4. Recommendation
- **A is the default** for all scoring and for the gate.
- **B is required** for any probability, Brier, or reliability claim; raw-sigmoid Brier is removed from summaries.
- **Gate change:** `σ(D(fake))` → `σ(D(fake) − D(real_ref))`. Keep `realness = 1` for G_frozen, as now. Add a no-gate ablation (`realness ≡ 1`) as a CPU check only.

## 5. CPU validation before any GPU
1. **Unit tests:**
   - The gate and row-softmax scores are invariant to adding a constant to D or J logits.
   - Softmax rows sum to 1.
   - The calibration split is disjoint from inner, exploratory and J pairs, and is deterministic under seed 1729.
   - The calibrator is fitted only on calibration rows.
   - All existing tests still pass (75 tests, plus these).
2. **Short CPU training** (synthetic plus the real bundle, about 200 steps, B0 and G): gate means at matched steps differ by ≤ 0.10 and stay in [0.3, 0.7].
3. **On the calibration split:** reliability bins and Brier. The calibrated full Brier must be ≤ the 1/150 base rate (0.00662), and ECE ≤ 0.02.
4. **Re-score the saved seed-42 checkpoints under A on CPU.** Rank metrics must be identical (scoring sanity check); only the probability fields change.

## 6. Minimal diff scope
- `gene_training.py`: gate (`:158-165`), calibration-split exclusion from J pair sampling, gate diagnostics.
- `gene_evaluation.py`: row-softmax scorer, `probabilities` semantics, calibrator application and reliability on softmax.
- New `gene_calibration.py`: split and Platt/temperature fit.
- `gene_protocol.py`: optional `calibration` group inside FIT.
- `tests/test_gene_harness.py`.
- No model, loss, or particlegan change.
- **Run receipt additions** (supervisor):
  - GPU released: no owned PID on `GPU-cb4ce47d…` after cleanup.
  - The llama-server (`work`) PID plus its `/proc` start time before and after, which must be identical.
  - The full list of signals the supervisor sent, all to owned pgids; no other process killed.
  - The calibration-split hash and the calibrator sha.

## 7. Success criteria, seeds 42/43/44 after the fix (selection on inner only)
- **Calibration:** per seed, the calibration-split full Brier is ≤ 0.00662 and ECE ≤ 0.02, and gate-mean |B0 − G| ≤ 0.10 at steps 1000 and 2000.
- **Selection:** an arm is selected only if:
  - its inner-24 head-AUROC paired delta is > 0 against **both** B0 and shuffle in all 3 seeds, and
  - the 3-seed pooled paired bootstrap (`all_seed_deltas`, 1000 draws) has a lower bound > 0.
  - MRR is reported but not used to select.
- **After selection is frozen:** report exploratory-30 once, as non-final.
- **Otherwise:** the result is reported as "no generalizing feature effect". No extra seeds, steps, or arms are added after seeing results.
