# PPARG critic-observation test (`offpc`): fold 0, two seeds, 5,000 updates

**Verdict: negative.** Under the application readout, adding an off-PC
gene-residual block to the EMPF critic did not measurably improve held-out mean
recovery for PPARG or for the 16 selected genes. Both primary CIs include 0. No
secondary metric regressed under the declared rule. A few secondaries moved
nominally in the variant's favor, but those moves are small, their CI bounds sit
near 0, and they are not adjusted for multiplicity. Stage-blind empirical
resampling still has clearly better SD and combined interval scores than either
GAN arm.

The protocol, declared before any score, is in [PROTOCOL.md](PROTOCOL.md).
Amendment 1 cut the matched budget from 10k to 5k updates because of CPU
contention, also before any score. All numbers are in
[results.json](results.json), produced by `score.py`.

## Protocol summary

- **Arms (matched).** The control is `pbulk`, native EMPF on ParticleGAN 0.10.0
  with the default recipe. The variant is `offpc,pbulk`: the same setup plus 16
  extra critic input columns ([arm_offpc.py](arm_offpc.py)). For each gene in S,
  a column holds the bag mean of the per-cell residual of standardized log
  expression that lies outside the 50-PC span. Real and fake bags go through the
  same forward function, and fake bags get a straight-through gradient. The
  block is normalized on 512 real training reference bags so that its mean
  squared norm matches that of the 150 released features. The update rule,
  optimizers, penalty (`lazy_k=1`) and streams are unchanged.
- **Genes S** ([selected_genes.json](selected_genes.json)): PPARG, fixed in
  advance, plus the 15 genes with the largest stage-RMS off-PC mean gap on
  fold-0 **training** donors. The gap is measured in the raw readout of the
  retained 20k default fit. No held-out rows were read.
- **Budget:** fold 0, seeds 0 and 1, 5,000 updates per fit, for four fits in
  total. This is a quarter of the 20k default.
- **Readouts:** `app` is `cellshape@0.5:biascal:stagehalf`, the application
  default, calibrated on training donors only. It decides the outcome. `raw` is
  `snapens` and serves as a diagnostic only. Both use the application generation
  streams with M=32 donors of 256 cells.
- **Scoring:** the 12 held-out development donors of fold 0, one of them Not-AD.
  Metrics come from the frozen `architecture_score` / `interval_metrics`.
  Per-donor errors are averaged over the two seeds. CIs come from a paired donor
  bootstrap with 10,000 resamples, percentile 95%. Decision rule: "improves" or
  "worsens" only when the whole CI is on one side of 0.
- **References:** stage-blind empirical resampling is required. Stage-aware
  resampling is shown in results.json.

## Primary and secondary outcomes (application readout)

| Outcome | Variant | Control | Blind empirical | Variant − control [95% CI] | Rule outcome |
| --- | ---: | ---: | ---: | ---: | --- |
| **P1** PPARG mean_log RMSE | .4321 | .4458 | .4148 | −.0137 [−.0351, +.0062] | inconclusive |
| **P2** 16-gene block RMSE | .8301 | .8260 | .8686 | +.0042 [−.0183, +.0265] | inconclusive |
| P2 without PPARG | .8501 | .8453 | .8906 | +.0048 [−.0187, +.0281] | inconclusive |
| All-gene mean_log RMSE | .2189 | .2175 | .2194 | +.0014 [−.0036, +.0058] | no regression |
| SD interval score ↓ | .8060 | .8318 | **.6133** | −.0259 [−.0448, −.0058] | nominal improvement |
| Combined interval score ↓ | .9463 | .9511 | **.8360** | −.0047 [−.0246, +.0149] | no change |
| Mean interval score ↓ | 1.0867 | 1.0703 | 1.0587 | +.0164 [−.0134, +.0445] | no regression (point estimate worse) |
| SD coverage (80% target) | 58.2% | 57.0% | 71.1% | +1.2 [−0.7, +2.8] pts | no change |
| Combined coverage | 69.7% | 68.0% | 78.8% | +1.7 [+0.2, +3.3] pts | nominal improvement |
| Mean coverage | 81.2% | 79.0% | 86.4% | +2.3 [+0.5, +4.3] pts | nominal improvement |

The mean interval score was declared as a secondary. The first version of
`score.py` left it out by mistake. It has now been added, and the frozen score
files were reused with matching fingerprints. All other CIs are unchanged,
because the bootstrap uses a fixed seed per comparison.

**Per seed** (control → variant), from `per_seed` in results.json:

- SD interval score: .829 → .803 and .835 → .809. Lower in both seeds.
- Combined interval score: .952 → .935 and .950 → .958. Mixed.
- PPARG RMSE: .444 → .429 and .447 → .436.
- Frozen profile RMSE, not bootstrapped: control .4457/.4598, variant
  .4542/.4523. Mixed.

Gene, energy and spread metrics are mixed or near-equal across seeds. Details
are under `readouts.app.frozen`.

### Arms vs stage-blind empirical (paired donor bootstrap, arm − blind)

| Metric | Control − blind | Variant − blind |
| --- | ---: | ---: |
| PPARG RMSE | +.031 [−.035, +.090] | +.017 [−.037, +.062] |
| 16-gene block RMSE | −.043 [−.085, +.002] | −.038 [−.071, −.003] |
| SD interval score | +.219 [+.096, +.361] | +.193 [+.073, +.330] |
| Combined interval score | +.115 [+.055, +.183] | +.110 [+.053, +.176] |
| SD coverage | −14.1 [−19.0, −9.7] pts | −13.0 [−18.9, −7.9] pts |

Stage-blind resampling has a **lower point estimate** for PPARG RMSE than
either GAN arm, but the CIs include 0, so neither arm is shown to be worse on
PPARG. On SD and combined interval scores, and on coverage, empirical
resampling is clearly better than both arms. On the 16-gene block mean, both
arms have lower point estimates than empirical resampling. For the variant the
CI just excludes 0.

## Raw readout diagnostic (not used for the decision)

| Metric | Variant | Control | Variant − control [95% CI] |
| --- | ---: | ---: | ---: |
| SD coverage | 60.2% | 38.7% | +21.4 [+17.3, +25.0] pts |
| SD interval score | .743 | 1.094 | −.351 [−.425, −.276] |
| Combined interval score | .970 | 1.197 | −.227 [−.292, −.168] |
| 16-gene block RMSE | .892 | 1.050 | −.158 [−.273, −.012] |
| All-gene mean_log RMSE | .2382 | .2487 | −.0106 [−.0194, −.0013] |
| PPARG RMSE | .496 | .455 | +.041 [+.007, +.080] |

Without calibration the variant has much better spread, and better block and
all-gene means. The SD interval score is lower in all 12 donors. The raw PPARG
worsening comes mostly from one seed: seed 0 goes .420 → .424 and seed 1
goes .487 → .558. The training-only bias and dispersion calibration of the
application readout removes most of the raw difference.

This test cannot separate two explanations for the raw spread effect: the
specific off-PC genes, or any strongly weighted extra critic block. The block
is normalized to the full norm of the 150 released features. Telling them apart
needs a matched random-gene or random-direction block control.

## Premise check: the 0.9.7 PPARG gap mostly isn't there in the current default

The README's PPARG blind-spot diagnostic described the frozen 0.9.7 models. In
those, generated Not-AD PPARG was 0.237 vs 0.827 observed, with about 99% of the
gap off the PC span. The current default is EMPF with the `pbulk` arm, and its
critic already sees every gene's bag pseudobulk, PPARG included. The check below
uses the retained 20k default fit on fold-0 **training** donors with the raw
`snapens` readout (`pparg_training_gap` in selected_genes.json):

| Stage (train donors) | Real mean_log | Generated | z gap | Off-PC z component | Off-PC share |
| --- | ---: | ---: | ---: | ---: | ---: |
| Not AD (3) | 0.889 | 1.094 | +0.105 | +0.018 | 17% |
| Low (3) | 1.377 | 1.322 | −0.028 | +0.091 | −321% |
| Intermediate (6) | 1.168 | 1.391 | +0.114 | +0.035 | 30% |
| High (13) | 1.105 | 1.017 | −0.045 | +0.042 | −93% |

In Low and High the total gap is small, but the off-PC component is not zero
and points the opposite way from the total. Low's off-PC component is about
5 times Not-AD's. Shares with a negative sign or a magnitude above 100% mean the
on-PC and off-PC parts partly cancel. PPARG ranks 501 of 2,000 genes by off-PC
training gap.

Two caveats apply. The check is **in-sample**: it uses the training donors the
fit was trained on, and the same readout that selected S. It also follows a
different protocol from the 0.9.7 matched-depth diagnostic, so the two are not
strictly comparable. Even so, the Not-AD undershoot of about 99% off-PC in
0.9.7 does not reproduce here. Not-AD is a small overshoot that is mostly
on-PC. On held-out donors, PPARG error is dominated by donor spread: observed
PPARG ranges from 0.61 to 1.92, while both arms predict about 1.0 to 1.2. A
stage-conditioned mean cannot capture that.

## Harness verification

- [harness_check.json](harness_check.json): after 5,000 updates, the final
  weights of control seed 0 are **bitwise identical** to the retained default
  fit's `snap_5000.pt`, for all 15 generator and prior tensors. The study host
  ([host_pparg.py](host_pparg.py)) therefore reproduces the native EMPF training
  path exactly.
- All four fits recorded `penalty_calls == completed_updates == 5000`
  (`lazy_k=1`), `native_recipe='empf'`, and the same recipe as the retained fit.
  The `study_audit.json` files under the run directory still say
  `recipe_equals_default_fit=false`, because that comparison skipped JSON
  normalization. The bug is fixed in `run_fit.py`. A normalized check, which an
  independent audit reproduced, gives True for all four fits.
- Tests: `python -B -m pytest analysis/pparg_critic_20261120/test_offpc.py`
  passes 8 of 8. The tests cover real/fake forward equality, the
  straight-through gradient, the off-PC projection, reference normalization,
  determinism, and that selection ignores held-out rows.
- An independent audit re-derived P1, P2 and the interval metrics from raw
  counts and archives. Point estimates matched exactly, and CIs agreed within
  Monte Carlo error. It found no donor leakage.
- Incidents, none of which change the scored math:
  - The first launch of variant seed 1 hit a physical GPU0 OOM at update 0. It
    was rerun from scratch on GPU1.
  - Variant seed 0 logged two recovered allocator OOM warnings.
  - The four aborted 10k partial runs are kept in `aborted_10k/`.

## Limitations

- **Scope.** One fold, two seeds and 12 held-out development donors, only one
  of them Not-AD. That one donor is the only held-out case directly relevant to
  the original Not-AD PPARG premise. All 12 donors were used in earlier
  development selection, and no sealed or external donors were used.
- **CIs.** The bootstrap resamples donors only. Training-seed variance (2 seeds)
  is not part of the intervals. A percentile bootstrap on 12 donors probably
  under-covers.
- **Multiplicity.** There are 2 primaries and about 8 bootstrapped secondaries,
  with no adjustment. The "nominal improvement" labels should be read
  accordingly.
- **Budget.** 5k updates, not 20k. The 5k control has an application SD
  interval score of .832, against .806 for the 20k default.
- **Selection.** S was chosen from the retained 20k default fit's training-donor
  readout. The block's weight is tied to the norm of the released features, and
  no weight sweep was run.
- The count-gradient ablation was not run.

## Next steps

1. Do not pursue `offpc` as a PPARG fix. Its premise largely fails for the
   EMPF+pbulk default, and held-out PPARG error is mostly donor variation. That
   supports the donor-latent direction.
2. To find out whether the raw-readout spread gain is real, run a matched
   random-gene or random-direction block of the same norm, plus a block-weight
   sweep (for example 0.25x and 0.5x), on 2 or 3 confirmation folds.
3. Re-check at 20k updates once contention eases.
4. `host_pparg.py`, which is bitwise verified, can serve as a native control
   harness for other critic-input studies.

## Reproduce

```bash
PY=/mnt/nvme/ai/conda/anaconda3/bin/python3; cd analysis/pparg_critic_20261120
$PY -B gen.py --fit <retained fold0/seed0 fit> --mode snapens --out <run>/selection/default_fit_snapens.npz
$PY -B select_genes.py --generated <run>/selection/default_fit_snapens.npz
CUDA_VISIBLE_DEVICES=1 taskset -c 3-23 $PY -B run_fit.py --arm pbulk --seed 0 --updates 5000        # x seeds 0,1
CUDA_VISIBLE_DEVICES=1 taskset -c 3-23 $PY -B run_fit.py --arm offpc,pbulk --seed 0 --updates 5000  # x seeds 0,1
$PY -B check_harness.py 5000
$PY -B score.py --seeds 0 1 --updates 5000
```

Large outputs (fits, archives and frozen score JSONs) are in
`/ml2/alz/runs/research_20261120/pparg-critic/`, outside git.
