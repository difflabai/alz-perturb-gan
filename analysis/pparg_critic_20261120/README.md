# PPARG critic-observation test (`offpc`): fold 0, two seeds, 5,000 updates

**Result.** The extra critic block did not improve held-out mean recovery for
PPARG or for the 16 selected genes under the application readout. Both primary
outcomes are inconclusive, with 95% CIs that include 0. No secondary metric
regressed. The application SD interval score improved slightly, and
combined/mean coverage rose 1.7 and 2.3 points. Stage-blind empirical resampling
still has the best interval scores.

The protocol, declared before any score, is in [PROTOCOL.md](PROTOCOL.md).
Amendment 1 shortened the matched budget from 10k to 5k updates because of CPU
contention, also before any score. All numbers are in [results.json](results.json).

## Premise check: the PPARG gap mostly isn't there in the current default

The README's PPARG blind-spot diagnostic described the frozen 0.9.7 models. In
those, generated Not-AD PPARG was 0.237 vs 0.827 observed, with about 99% of the
gap off the PC span. The current default is EMPF with the `pbulk` arm, and its
critic already sees every gene's bag pseudobulk, PPARG included. We checked
the retained default fit on fold-0 **training** donors, raw `snapens` readout
([selected_genes.json](selected_genes.json)):

| Stage | Real mean_log | Generated | z gap | Off-PC share |
| --- | ---: | ---: | ---: | ---: |
| Not AD | 0.889 | 1.094 | +0.105 | 17% |
| Low | 1.377 | 1.322 | −0.028 | (gap ≈ 0) |
| Intermediate | 1.168 | 1.391 | +0.114 | 30% |
| High | 1.105 | 1.017 | −0.045 | (gap ≈ 0) |

PPARG ranks 501 of 2,000 genes by off-PC training gap. The 0.9.7 undershoot has
become a small overshoot. Most of PPARG's held-out error comes from donor
variation: observed held-out PPARG ranges from 0.61 to 1.92, while both arms
predict about 1.0 to 1.2. Stage-blind resampling has a lower PPARG RMSE (0.415)
than either GAN arm.

## Primary and secondary outcomes (application readout, 12 held-out donors)

Arms: control `pbulk` (native EMPF) vs variant `offpc,pbulk`. Both run 5,000
updates on fold 0 with seeds 0 and 1. The readout is
`cellshape@0.5:biascal:stagehalf` with the application generation streams.
Variant − control CIs come from a paired donor bootstrap: 10,000 resamples,
seed-pooled per-donor errors.

| Outcome | Variant | Control | Blind empirical | Variant − control [95% CI] | Verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| **P1** PPARG mean_log RMSE | .4321 | .4458 | .4148 | −.0137 [−.0351, +.0062] | inconclusive |
| **P2** 16-gene block RMSE | .8301 | .8260 | .8686 | +.0042 [−.0183, +.0265] | inconclusive |
| P2 without PPARG | .8501 | .8453 | .8906 | +.0048 [−.0187, +.0281] | inconclusive |
| All-gene mean_log RMSE | .2189 | .2175 | .2194 | +.0014 [−.0036, +.0058] | no regression |
| SD interval score ↓ | .8060 | .8318 | **.6133** | −.0259 [−.0448, −.0058] | improved |
| Combined interval score ↓ | .9463 | .9511 | **.8360** | −.0047 [−.0246, +.0149] | no change |
| SD coverage (80% target) | 58.2% | 57.0% | 71.1% | +1.2 [−0.7, +2.8] pts | no change |
| Combined coverage | 69.7% | 68.0% | 78.8% | +1.7 [+0.2, +3.3] pts | improved |
| Mean coverage | 81.2% | 79.0% | 86.4% | +2.3 [+0.5, +4.3] pts | improved |

Frozen headline metrics are reported per seed without bootstrapping
(`architecture_score.py`). On profile RMSE, seed 0 is worse for the variant and
seed 1 is better: control .4457/.4598, variant .4542/.4523. Gene, energy and
spread metrics are mixed or near-equal across seeds. Details are in
`readouts.app.frozen`. In both seeds the variant's SD interval score is lower:
.803/.809 vs .829/.835.

## Raw readout diagnostic (not used for the decision)

The uncalibrated `snapens` readout shows a much larger difference. SD coverage
rises from 38.7% to 60.2% and the SD interval score falls from 1.094 to .743,
with CI [−.425, −.276]. Block-gene RMSE falls from 1.050 to .892, CI excluding 0.
Raw PPARG RMSE gets *worse*: .455 to .496, CI [+.007, +.080].

The training-only bias/dispersion calibration of the application readout
removes most of that difference. This test cannot say whether the raw-spread
effect comes from the specific off-PC genes or from any strongly weighted extra
critic block. The block is normalized to the full norm of the 150 released
features. Separating the two needs a matched random-gene or random-direction
block control.

## Harness verification

- [harness_check.json](harness_check.json): the control seed 0 final weights
  after 5,000 updates are **bitwise identical** to the retained default fit's
  `snap_5000.pt`, for generator and prior. The study host (`host_pparg.py`)
  therefore reproduces the native EMPF training path exactly.
- Every fit recorded `penalty_calls == completed_updates == 5000` (`lazy_k=1`),
  `native_recipe='empf'`, and a recipe equal to the retained fit's.
  `study_audit.json` reports `recipe_equals_default_fit=false` because the
  comparison skipped JSON normalization. That bug is fixed in `run_fit.py`. A
  separate normalized check gives True for all four fits.
- Tests: `python -B -m pytest analysis/pparg_critic_20261120/test_offpc.py`
  passes 8 of 8. The tests cover real/fake forward equality, the
  straight-through gradient reaching the rate, the off-PC projection, reference
  normalization, determinism, and that selection ignores held-out rows.
- Incidents, none of which affect the scored runs:
  - The first variant seed 1 launch hit a physical GPU0 OOM at update 0
    (GPU0 was full from other jobs). It was rerun from scratch on GPU1.
  - Variant seed 0 logged two recovered allocator OOM warnings.
  - The four aborted 10k partial runs are kept under `aborted_10k/`.

## Limits

This is one fold with two seeds and 12 held-out development donors, one of them
Not AD. These donors were used in earlier development selection. The 5k budget
is not the 20k default: control 5k application SD score is .832 against .806
for the 20k default. Gene selection used the 20k default fit's training-donor
readout. The count-gradient ablation was not run.

None of this supports a default change. The single positive signal is the
raw-readout spread improvement, which disappears under calibration. Its next
step would be the random-block control above.

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
