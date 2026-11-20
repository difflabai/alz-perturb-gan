# Donor random-effect generator: negative result (no improvement)

**Result.** Adding this donor random effect to the native EMPF generator did **not**
improve donor-level variability coverage. At a matched 10,000-update budget on folds
0 and 1, the variant failed the declared improvement rule in both training seeds. The
rule was declared in [PROTOCOL.md](PROTOCOL.md) before any score existed, and the
seed replicate was declared in
[PROTOCOL_ADDENDUM_SEED1.md](PROTOCOL_ADDENDUM_SEED1.md) before any seed-1 output
existed. Under those rules the variant does not advance.

With seed 0 alone, the variant looked significantly **worse**: SD interval score
+.034 [+.014, +.054]. That CI covers donor sampling only. The seed-1 replicate
**does not reproduce it**: −.009 [−.025, +.006]. The native control alone moves by
+.035 between seeds, which is as large as the seed-0 arm difference. Over the two
seeds the difference is +.012 [−.001, +.026]. "Worse than control" is therefore
**not established**; the honest reading is "no improvement, and the difference is
within training-seed noise". Neither arm beats stage-blind empirical resampling on
the pooled SD score.

## What was tested

The native generator already has a per-donor latent: a particle `z`, one of 512
learned particles, that sets a FiLM of each cell. The variant `g.dre`
([arm_dre.py](arm_dre.py)) adds `u ~ N(0, I_4)` per generated donor. This scales the
cell-latent noise by `exp(A u)` (a donor variance effect) and shifts the donor FiLM
by `B u` (a donor location effect). A and B start at zero, and everything else is
native: critic, EMPF penalty, particle prior, recipe and batches. Both arms were
trained with the same derived native EMPF trainer ([dl_host.py](dl_host.py)) for
10,000 updates. Every update was regularized: 10,000 penalty calls in 10,000 updates
for all 8 fits.

At seed 0 the control's trajectory reproduces the selected 20k default fit
(fold 0) **bitwise**. Critic and generator losses are equal at all logged steps
compared, so the seed-0 fold-0 control is the first half of the default run. Parity
was checked on fold 0 only. The readout `cellshape@0.5:biascal:stagehalf` uses the six
late states (7,500 through 9,500 plus final), M=32 and K=256. Both arms were generated
on CUDA with the application's named generation streams. Scoring used the frozen r5
scorer, plus per-donor and per-PC-block intervals, with a paired donor bootstrap
(10,000 resamples) ([score.py](score.py), [seed_replicate.py](seed_replicate.py)).

## Held-out development donors (80% target; lower interval score is better)

| Fold | Arm | Seed | SD coverage | SD interval score | Combined coverage | Combined score |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | native control | 0 | 59.2% | .7936 | 68.5% | .947 |
| 0 | donor random effect | 0 | 56.7% | .8415 | 68.0% | 1.008 |
| 0 | native control | 1 | 58.3% | .8428 | 68.3% | .977 |
| 0 | donor random effect | 1 | 58.3% | .8332 | 70.5% | .967 |
| 0 | stage-blind resampling | – | 69.8% | .6119 | 78.2% | .832 |
| 0 | same-stage resampling | – | 65.9% | .6804 | 71.7% | .934 |
| 1 | native control | 0 | 68.2% | .5579 | 70.5% | .829 |
| 1 | donor random effect | 0 | 65.1% | .5769 | 71.1% | .829 |
| 1 | native control | 1 | 62.5% | .5794 | 70.9% | .823 |
| 1 | donor random effect | 1 | 60.7% | .5707 | 68.2% | .828 |
| 1 | stage-blind resampling | – | 65.2% | .6335 | 74.8% | .836 |
| 1 | same-stage resampling | – | 63.3% | .6763 | 71.1% | .916 |

Fold 0 has 12 held-out donors and fold 1 has 13. The statistic below is the mean of the
fold means, with a paired donor bootstrap (resampling held-out donors within each fold).
**These CIs reflect donor sampling only.** None of them include training-seed variance;
the seed rows show how large that variance is.

| Contrast (held-out) | SD interval score | SD coverage |
| --- | ---: | ---: |
| variant − control, seed 0 | +.034 [+.014, +.054] | −2.8 pp [−4.7, −0.8] |
| variant − control, seed 1 | −.009 [−.025, +.006] | −0.9 pp [−2.6, +0.9] |
| variant − control, mean of 2 seeds | +.012 [−.001, +.026] | −1.9 pp [−3.1, −0.6] |
| control seed 1 − control seed 0 (native seed noise) | +.035 [+.017, +.053] | −3.3 pp [−5.0, −1.7] |
| variant seed 1 − variant seed 0 | −.007 [−.021, +.007] | −1.4 pp [−3.4, +0.6] |
| variant − stage-blind empirical, seed 0 | +.087 [−.007, +.182] | −6.6 pp [−11.7, −1.2] |
| control − stage-blind empirical, seed 0 | +.053 [−.037, +.142] | −3.8 pp [−9.4, +2.0] |

Decision rule outcomes:

* **Seed 0.** The CI is above 0, not below; SD coverage moved further from 80% in both
  folds (59.2 → 56.7%, 68.2 → 65.1%). Guards failed: fold 0 cell energy +3.0%; fold 1
  pb_rmse +3.9%, detection +4.6%, log-variance +4.1%, PC energy +3.5% and cell energy
  +7.5%. Fold 0 gene mean and log-variance errors improved (meanlog .074 → .065).
* **Seed 1.** The CI includes 0. SD coverage was unchanged in fold 0 (58.3%) and lower
  in fold 1 (62.5 → 60.7%). Guards failed in fold 0: pb_rmse +0.9% and cell energy
  +3.3%.
* **Replication rule from the addendum.** "Worse" would count as replicated only if
  seed 1 also had a positive difference, the two-seed CI lay above 0, and the native
  seed-to-seed difference were smaller than the seed-0 effect. None of the three
  holds (seed-to-seed .035 vs effect .034).

## Diagnostics (descriptive, declared secondary)

* **Generated donors are about half as variable as real donors, and the random effect
  did not change that.** For the SD summaries, the spread across the 32 generated
  donors per target is .092–.098 for the control and .085–.095 for the variant (both
  seeds, both folds). The real spread across donors within a stage is .169/.175, and
  empirical resampling gives .17–.18. The real-donor figure uses one 64-cell subsample
  over all train and held-out donors per stage, so it is only descriptive. The learned
  amplitudes stayed small. At 10k updates, ||A|| was .27/.31 at seed 0 and .19/.33 at
  seed 1, and ||B|| was .72/.99 at seed 0 and .91/.93 at seed 1 (Frobenius norms,
  folds 0/1). At seed 0, ||B|| had reached 1.5/1.9 at update 2,500 and then shrank.
* **Undercoverage is concentrated on PCs 1–10.** In the control, 56–78% of held-out
  PC 1–10 SD observations lie above the interval, and at most 2.6% lie below; coverage
  is 22–42% across seeds and folds. The generated interval on these PCs is
  .19–.24 wide, less than half the empirical width (.52–.53). Both problems are
  visible: the generated within-donor SD sits too low, and the interval across
  generated donors is too narrow, matching the between-donor spread deficit above.
  This result only shows that **this particular zero-initialised, dimension-4 random
  effect, trained against the native per-bag critic, did not help**. It does not
  rule out a donor latent in general. The native PC 1–10 score also moves a lot with
  the training seed (control seed 1 − seed 0: +.128 [+.082, +.176]).
* **On PCs 31–50 the native model beats both empirical references.** At seed 0 its
  interval score is lower than stage-blind resampling by .23 [.17, .29]. One
  possible explanation is the documented out-of-sample PCA shrinkage of held-out
  targets on those PCs. That explanation is an **untested interpretation**; this
  study did not test it. The headline SD average mixes this block with the PC 1–10
  failure.
* The motivating train/held-out gap persists in both arms at seed 0: TRAIN SD coverage
  is 29–33%, against 75% for resampling, and the misses are mostly real TRAIN SDs
  above the generated interval.

Descriptive, not a declared endpoint: on fold 1 the 10k native control has a lower SD
score than stage-blind resampling at both seeds (.558 and .579 vs .634). On fold 0 it
is much higher (.794 and .843 vs .612). The pooled CI includes zero, so the "beat
empirical" goal remains open.

## Protocol deviations and corrections

* **Wrong fold-0 reference value.** The protocol said the fold-0 stage-blind reference
  "must reproduce .61856". That number belongs to the covariance challenge's separate
  stream namespace. The archive used here is from the declared
  `empirical_challenge_20261114` `empirical32` namespace, replicate 0. It is
  **byte-identical** to that study's fold-0 archive (sha256 `79541774…`), and it
  reproduces the recorded scores exactly: .61186 for fold 0 and .63345 for fold 1. No
  decision depended on this value.
* **Wrong donor count.** The protocol said there were 12 held-out donors per fold.
  Fold 1 has **13** held-out donors, and the bootstrap uses all 13.
* **Training launched before the commit.** The four seed-0 training jobs started at
  about 03:55 UTC, before the protocol commit at 03:57. Training writes no evaluation
  output, and the first score file is from 03:59.
* **Seed replicate added after the seed-0 result.** It was added in response to an
  audit, after the seed-0 result was known. It was declared, and its scoring code
  committed, before any seed-1 generation or score existed.

## Limits

* Two folds with 12–13 reused development donors each. This is not an independent test.
* Two training seeds per fold. That is enough to show that seed noise is as large as
  the seed-0 effect, but not enough to estimate a seed-variance-aware CI. The donor
  bootstrap percentile CIs use few clusters and may be somewhat anti-conservative.
* 10k updates, not the 20k default, and one variant configuration: dimension 4 with
  zero initialisation. A larger initialisation, a critic that sees several bags of
  the same stage together, or longer training could behave differently. None of these
  were tested.
* The `biascal` calibration is inherited from the frozen application readout, and the
  same readout was used for every arm. Empirical references exclude the target donor.
  The PCA basis is the TRAIN-fitted cache basis.
* Fits ran under heavy shared-host load, on cores 3–23 with GPU fraction .04. The
  default config, released ParticleGAN and every file in `/ml2/alz` are unchanged.
* Smoke outputs (`u40`, `smoke_scores`, `context_scores`) under the runs directory are
  not part of the comparison.

## Reproduce

```bash
PY=/mnt/nvme/ai/conda/anaconda3/bin/python3
D=analysis/donor_latent_20261120
for S in 0 1; do for K in 0 1; do for A in pbulk g.dre,pbulk; do
  CUDA_VISIBLE_DEVICES=1 taskset -c 3-23 $PY -B $D/run.py train    --fold $K --seed $S --arm $A --updates 10000 --device cuda --gpu-mem-fraction .04
  CUDA_VISIBLE_DEVICES=1 taskset -c 3-23 $PY -B $D/run.py generate --fold $K --seed $S --arm $A --updates 10000 --device cuda --gpu-mem-fraction .04
done; done; done
$PY -B $D/score.py empirical --fold 0; $PY -B $D/score.py score --fold 0      # and fold 1
$PY -B $D/score.py compare                                                     # seed 0 -> completion/comparison.json
$PY -B $D/seed_replicate.py score --fold 0; $PY -B $D/seed_replicate.py score --fold 1
$PY -B $D/seed_replicate.py compare                                            # -> completion/seed_replicate.json
PYTHONPATH=/ml2/alz/artifacts/native-releases/particlegan-0.10.0/site-packages $PY -B -m pytest -q $D/test_donor_latent.py
```

Fits, archives and scores are under `/ml2/alz/runs/research_20261120/donor-latent/`
(outside git). [completion/artifacts.json](completion/artifacts.json) binds the seed-0
hashes. [completion/comparison.json](completion/comparison.json) holds every seed-0
contrast, block and guard. [completion/seed_replicate.json](completion/seed_replicate.json)
holds the seed-1 scores, the seed contrasts and the seed-1 archive and score hashes.
