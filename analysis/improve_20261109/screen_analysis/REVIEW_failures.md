# REVIEW_failures: adversarial review of the "why three arms got worse" analysis

Reviewer scope. `ANALYSIS_failures.md` **did not exist** when I did this review. I reviewed the analyst's returned
summary (headline, 8 claims, 3 suspected bugs) and their scripts `failures_{common,table,curves}.py`. I recomputed
every key number with my own code, which reads `generated.npz`, the fold caches, `final.pt` (map_location='cpu') and
`learning_curve.jsonl` directly. My code does not import `evaluate.py`, `metrics.py` or the analyst's scripts.
The counterfactual generation imports `trainer.build/Context` read-only to rebuild the modules.

All work is CPU only (CUDA_VISIBLE_DEVICES='', 6 threads) and uses development donors only: the caches hold only the
37 development donors, and `review_failures_common.load_real` asserts this. No existing file was modified.

Scripts (in `screen_analysis/`), with outputs in `/ml2/alz/runs/improve_20261109/screen_analysis/`:

| script | output | what it checks |
|---|---|---|
| `review_failures_common.py` | - | own loaders: real per-donor pb, mean_log and library from the CSR caches; generated arrays reordered to cache order |
| `review_failures_pb.py` | `review_failures_pb.json/.log` | train/held-out pb_rmse, dev-Pearson, mean_log rmse and Q4 bias, library ratio, **train-donor identification**, **held-out nearest-train-donor correlation**, CRN-matched per-target diversity vs base, 256-cell noise floor, covariate uniqueness |
| `review_failures_pc.py` | `review_failures_pc.json` | PC-space coverage80, within-condition spread and energy, with my own real 64-cell subsample |
| `review_failures_state.py` | `review_failures_state.json/.log` | learning-curve windows; final.pt particle covariance; **latent vs condition variance in the generator's first donor layer**; offset table init vs final |
| `review_failures_donorpc.py` | `review_failures_donorpc.json/.log` | donorpc bag-noise share per PC, block magnitudes, critic first-layer variance share per input block, gene-mean error inside the donor-PC and W50 spans |
| `review_failures_covars_cf.py` | `review_failures_covars_cf.json/.log` | held-out regeneration of covars arms under own, nearest-train, mixed (marginalised) and train-mean covariates |
| `review_failures_regen_check.py` | `review_failures_regen_check.json` | fresh CPU generation vs the stored generated.npz for the fits re-generated after the 12:39 trainer.py edit |

Fold-balanced means: seeds are averaged within a fold, then the fold means are averaged (hvg2000). n = 3 inner folds
whose training sets overlap, x 2 seeds. I count signs over the 6 paired fits. I make no significance claims.

## Verdicts on the claims

**C1 covars is overfitting, not optimisation failure: CONFIRMED (numbers), with a better supporting statistic.**
My recomputation gives train pb_rmse 0.647 vs base 0.650 (paired -0.003, worse in 3/6) and held-out 0.927 vs 0.689
(paired +0.238, worse in 6/6). The held-out minus train gap is +0.280 vs +0.039.

The analyst's evidence has a weakness. Train pb_rmse parity does *not* show memorisation, and memorisation should
improve train fit. Two statistics do show it:
- Train-donor identification: the share of train targets whose prediction deviation correlates best with their own
  real deviation, among same-stage train donors. covars 0.85 (per fit 0.72-0.96) and the combined arm 0.68, against
  base 0.15, for a chance level of 0.16.
- Train dev-Pearson: covars 0.43 and the combined arm 0.46, against base 0.09.

On held-out donors the deviation correlates with the covariate-nearest same-stage *train* donor (r = 0.25 for covars,
0.17 for the combined arm, 0.08 for base). The correlation with the donor's own deviation is about 0 for every arm.
The pb_rmse 256-cell sampling floor is 0.043-0.048, so the pb_rmse gaps are model error, not depth noise.

**C2 combined arm fits train better and generalises worst: CONFIRMED.**
- Train pb_rmse: 0.603 (paired -0.048, better in 6/6).
- Train meanlog_rmse: 0.058 vs 0.204.
- Held-out pb_rmse: 0.974 (paired +0.286, worse in 6/6).
- Held-out PC coverage80: my own subsample gives 0.248 vs 0.603; the evaluator gives 0.256 vs 0.600.

**C3 generated donors collapse to one per condition, memorisation: QUALIFIED.**
My numbers reproduce the analyst's:

| quantity | covars | base |
|---|---|---|
| train PC within-condition spread | 0.407 | 0.846 |
| train PC energy | 1.33 | 2.20 |
| held-out PC energy | 3.46 | 2.31 |
| held-out coverage80 | 0.276 | 0.603 |

But the within-condition spread compares unlike things. covars conditions on a donor-unique covariate vector, while
base conditions on stage only, so lower spread at a fixed full condition is partly expected by design. "One per
condition" also overstates it. Better evidence for a real collapse:
- With common random numbers, the per-target variance across the 32 generated donors is 0.30x base for covars and
  0.25x for the combined arm. This variance includes the identical Poisson and cell noise.
- Two independent CPU generations of covars fold 0 seed 0 agree per donor at r = 0.9999 (base-like arms: 0.93).
  So the latent hardly changes the output.

**C4 particle prior collapses: QUALIFIED.**
Over updates 15k-20k my numbers match:

| quantity | covars | combined | base |
|---|---|---|---|
| particle covariance trace | 0.324 | 0.172 | 4.304 |
| prior regulariser | 0.762 | 0.816 | 0.228 |
| critic gap | 2.31 | - | 4.55 |

There are no non-finite rows. The particle trace by itself depends on scale, because the generator can rescale z.
I measured the generator side directly. In final.pt's `donor.0.weight`, the latent-driven pre-activation variance
tr(Wz Cov z Wz^T) is 3.1 for covars and 0.9 for the combined arm, against 54.7 for base. The covariates carry 98-99%
of the donor-layer variance, against 18% (stage only) in base. So the collapse is real.

The analyst also wrote "critic gap lower, so the critic can no longer separate". That over-reads it:
- A gap of 2.3 logits still separates.
- The combined arm collapses just as much but ends with a gap of 8.7 logits.
- Critic gaps are not comparable across arms whose critics see different inputs. The covars critic puts 23-41% of
  its first-layer variance on the condition block, against about 4% in base.

**C5 covariates identify single donors; extrapolation is not the main cause: CONFIRMED** (now in a saved script).
The train sets have 23, 24 and 22 unique (stage, male, APOE4, age) tuples. Only fold 2 has held-out ages outside the
train range: 65 and 102, against a train range of 68-99. Held-out tuples that also occur in train: 2, 3 and 1.

The counterfactual supports the claim. Feeding the nearest train donor's exact covariates still scores 0.827. So
point-conditioning on *any* donor key is the problem, not only out-of-range values. Own covariates (0.926) are
somewhat worse than the nearest train key, so off-support interpolation adds some damage.

**C6 no covariate bug: CONFIRMED.**
`Context` computes age mean and SD from the train `raw_c` (ddof=0), and these equal `space.npz` and the config
(87.96 / 8.28). `generate_arrays` applies `build_conditions` to the target's own `raw_c` with the same values.
Held-out raw_c comes from `heldout/metadata.npz`. My independent regeneration with own covariates gives a
fold-balanced held-out pb_rmse of 0.926, against the evaluator's 0.927, and per-fit values match to ±0.002.
The generated/target library median ratio is 1.000 for every arm.

**C7 donorpc is an objective or scaling failure, not overfitting: QUALIFIED.**
All the numbers are confirmed:
- train pb_rmse 0.770 vs 0.650 (+0.119, worse in 6/6); held-out 0.793
- meanlog_rmse 0.255 vs 0.204; Q4 bias -0.218 vs -0.176
- coverage 0.62 vs 0.60; pb within-condition spread 1.29 vs 1.05
- final critic gap 5.88 vs 4.55, though that comparison is unlike-for-like

The analyst's lead hypothesis was that low donor PCs are dominated by 128-cell sampling noise. That is **refuted**:
the within-donor bag variance is at most 1.5% of the feature variance for every one of the 16 PCs in every fold.

What I do find:
- The whitened 16-dim block takes 74-77% of the critic's first-layer variance. The 150 released features drop from
  about 96% to 20-23%.
- Whitening gives PC16 (about 2% of the donor variance) the same unit scale as PC1 (about 20%).
- The generator's gene-mean error grows: squared norm 45-148, against 33-48 for base.
- Only 2-10% of that error lies inside the donor-PC span.

So the extra block redirects the critic without constraining the gene means outside its 16 directions. Per-target
generated diversity rises to 1.32x base, so there is no collapse. This favours "mis-scaled or redundant objective"
over "noise", but one screen cannot separate them.

**C8 offset gives nothing; table about zero: QUALIFIED.**
The confirmed parts:
- held-out 0.690 vs 0.689 (+0.001); train 0.658 vs 0.650
- final critic gap 4.58 vs 4.55
- tau2 = 0.00313 on fold 0 and 1e-6 on folds 1 and 2

Question (e) asked whether the table is still about zero after training. **It is not.**
- Final RMS per stage is 0.08-0.17. On fold 0 that is 6-7x the initial RMS of 0.010-0.024. On folds 1 and 2 the
  table grew from about 1e-5.
- The two seeds learn similar rows (row correlation 0.24-0.81 for offset, 0.12-0.74 for the combined arm), so the drift is reproducible, not pure noise.
- The final rows correlate only -0.07 to 0.60 with the unshrunk donor-averaged stage log-ratio, against 0.73-0.83
  for the initial table.

So the table moves substantially but mostly learns something other than the stage effect. The base network can
already express stage effects through the ADNC condition slot, so the table is a redundant parametrisation.
"Init about zero" is not the explanation.

## Suspected bugs

1. "None confirmed": **agree.** I found no train/generation mismatch in covariate standardisation or conditioning.
2. "DualNorm polar step on the [4,G] table amplifies noise": **plausible, not shown; not a code bug.**
   The drift has a flat singular spectrum (max/min 1.4-3.1 across the 12 offset-carrying fits), but any 4x2000 matrix of noise has one too, so that is
   not diagnostic. The reproducibility across seeds argues against pure noise amplification.
3. "trainer.py edited after the screen": **the edit is real, but the timeline is wrong and the effect is benign.**
   - The fits did not finish at 04:00-06:00. final.pt times run from 03:31 to 12:30.
   - `logs/screen_main.out` shows 18 fits failing at *generation*: offset, donorpc and the combined arm, all with
     tensor arm state. The failure is in `restore._same`, a "tensors on cuda:0 vs cpu" RuntimeError (for example
     in `logs/offset_hvg2000_fold0_seed0.log`).
   - trainer.py was saved at 12:39:24, and its `_same` now moves both sides to CPU. At 12:39:29
     `logs/screen_regen.out` re-ran generation only for those 18 fits ("final.pt present without generated.npz;
     generation only").
   - So all 48 trainings used the pre-edit code, and 18 generations used the edited `restore`.
   - I cannot diff the untracked file. But a fresh CPU regeneration from final.pt matches the stored arrays (fold 0
     seed 0) within Monte-Carlo error:

     | arm | fresh held-out pb_rmse | stored |
     |---|---|---|
     | donorpc | 0.822 | 0.819 |
     | offset | 0.691 | 0.696 |
     | combined | 0.927 | 0.927 |

## Additional issues and alternative explanations

- **Marginalising the covariates removes the covars damage.** At evaluation, give each of the 32 generated donors the
  covariates of a random same-stage train donor:
  - covars held-out pb_rmse becomes 0.657, better than base (0.689);
  - the combined arm becomes 0.598, better than single-seed genemean (0.661) and near its seed ensemble (0.583);
  - train-mean covariates give 0.821 and 0.764.

  So the failure lies in point-conditioning a donor-key model on one covariate vector, not in the networks
  themselves. This is evidence that the evaluation protocol for covars is a poor match for this model. It is not a
  win: no arm predicts held-out donor-specific deviations (own dev-Pearson about 0 everywhere), and all arms remain
  well above the baselines (about 0.46).
- **Lookup is the optimum of the objective.** With about 24 train donors whose condition vectors are nearly unique,
  the conditional target p(bag | c) *is* one donor's cell distribution. The critic sees c too. So the observed lookup
  is the correct solution of the training objective, not an optimiser pathology.
- **Critic-gap comparisons across arms** (genemean 13.1, combined 8.7, donorpc 5.9, base 4.5, covars 2.3) compare
  critics with different inputs. They should not be used as evidence of fit quality between arms.
- **Train pb_rmse does not detect memorisation.** It is insensitive to donor-specific fit; identification and
  dev-Pearson are. Future reports should show both.
- **Robustness.** The held-out direction is 6/6 for covars, the combined arm and donorpc, and the effect sizes
  (+0.10 to +0.29) are large next to the fit-to-fit spread. The causal mechanisms are supported by within-fit
  counterfactuals, but they rest on 3 overlapping folds x 2 seeds. "High confidence" fits the direction, not the
  mechanism details.

## What a fair retest needs

- **covars:** stop covariates acting as a donor key. Options are coarse bins, slot dropout or noise in training,
  or not showing covariates to the critic. Report held-out results with point and marginalised covariates, plus
  train identification and latent-variance diagnostics.
- **donorpc:** do not whiten. Scale the block to the released features' per-dimension variance, or weight PCs by
  explained variance. Track critic first-layer block shares and the gene-mean error outside the span.
- **offset:** the current table learns reproducible non-stage content. To test the idea, fix the table, use a
  separate optimizer group, or regularise it toward the shrunk init. With 2-3 Not AD train donors, expect no
  detectable held-out gain.
