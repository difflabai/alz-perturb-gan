# Donor random-effect generator: negative result

**Result.** An explicit continuous donor random effect did **not** improve donor-level
variability coverage. At a matched 10,000-update budget on folds 0 and 1, it made the
held-out SD interval score **worse** than the native EMPF control, by +0.034 (95% CI
+0.014 to +0.054). SD coverage was lower in both folds. Historical guards failed,
in fold 1 especially. Neither the variant nor the control beats stage-blind empirical
resampling on the declared pooled SD score. The decision rule was declared in
[PROTOCOL.md](PROTOCOL.md) before any score existed. Under that rule the variant
does not advance.

## What was tested

The native generator already has a per-donor latent: a particle `z` from 512 learned
particles that sets a FiLM of each cell. The variant `g.dre` ([arm_dre.py](arm_dre.py))
adds `u ~ N(0, I_4)` per generated donor. This scales the cell-latent noise by `exp(A u)`
(a donor variance effect) and shifts the donor FiLM by `B u` (a donor location effect).
A and B start at zero, and everything else is native. Both arms were trained with the
same derived native EMPF trainer ([dl_host.py](dl_host.py)) and the same streams for
10,000 updates. Every update was regularized: 10,000 penalty calls in 10,000 updates.

The control trajectory reproduces the selected 20k default fit **bitwise**. Critic and
generator losses are equal at updates 1, 250 and 251, so the control is the first half
of the default run. The readout `cellshape@0.5:biascal:stagehalf` uses the six late
states (7,500 through 9,500 plus final), M=32 and K=256. Both arms were generated on
CUDA with the application's named generation streams. Scoring used the frozen r5
scorer, plus per-donor and per-PC-block intervals, with a paired donor bootstrap
(10,000 resamples) ([score.py](score.py)).

## Held-out development donors (80% target; lower interval score is better)

| Fold | Arm | SD coverage | SD interval score | Combined coverage | Combined score |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | native control (10k) | 59.2% | .7936 | 68.5% | .947 |
| 0 | donor random effect (10k) | 56.7% | .8415 | 68.0% | 1.008 |
| 0 | stage-blind resampling | 69.8% | .6119 | 78.2% | .832 |
| 0 | same-stage resampling | 65.9% | .6804 | 71.7% | .934 |
| 1 | native control (10k) | 68.2% | .5579 | 70.5% | .829 |
| 1 | donor random effect (10k) | 65.1% | .5769 | 71.1% | .829 |
| 1 | stage-blind resampling | 65.2% | .6335 | 74.8% | .836 |
| 1 | same-stage resampling | 63.3% | .6763 | 71.1% | .916 |

Paired donor bootstrap, mean of the fold means (fold 0 has 12 held-out donors, fold 1 has 13):

| Contrast | SD interval score | SD coverage |
| --- | ---: | ---: |
| variant − control | **+.034 [+.014, +.054]** | **−2.8 pp [−4.7, −0.8]** |
| variant − stage-blind empirical | +.087 [−.007, +.182] | −6.6 pp [−11.7, −1.2] |
| control − stage-blind empirical | +.053 [−.037, +.142] | −3.8 pp [−9.4, +2.0] |

Guards relative to control: fold 0 failed cell energy (+3.0%). Fold 1 failed pb_rmse
(+3.9%), detection, log-variance, PC energy and cell energy errors. Fold 0 variant
gene mean/log-variance errors improved (meanlog .074 → .065).

## Why it did not help

* **Generated donors are too similar to each other, and the random effect did not change that.**
  For the SD summaries, the spread across the 32 generated donors per target is .095/.098
  for the control and .091/.095 for the variant (folds 0/1). The real spread across donors
  within a stage is .169/.175, and empirical resampling gives .17-.18. The learned
  amplitudes stayed small: ||A|| .27/.31 and ||B|| .72/.99 at 10k. ||B|| rose to
  1.5/1.9 by update 2,500, then fell. The native bag critic gives no incentive to keep
  that spread.
* **On PCs 1-10, almost every miss has the real SD above the interval.** For the control,
  70% (fold 0) and 59% (fold 1) of held-out PC 1-10 SD observations lie above the
  interval, and fewer than 2% lie below. Coverage on those PCs is 29%/39%, compared
  with 76%/86% on PCs 31-50. The leading
  PCs need more within-donor dispersion. The variant lowered PC 1-10 coverage further,
  to 22% in fold 0. On PCs 31-50 the native model *beats* both empirical references.
  Its interval score is lower by .23 (CI .17-.29), consistent with the documented
  out-of-sample PCA shrinkage of held-out targets on those PCs. The headline SD average
  therefore mixes two opposite failures.
* The motivating train/held-out gap persists in both arms: TRAIN SD coverage is 29-33%,
  compared with 75% for resampling. It mostly reflects real TRAIN SDs above the
  generated interval, not a donor-spread effect the random effect could fix.

Descriptive, not a declared endpoint: on fold 1 the 10k native control has a lower SD
score than stage-blind resampling (.558 vs .634). On fold 0 it is much higher (.794 vs
.612). The pooled CI includes zero, so the "beat empirical" goal remains open.

## Protocol notes and limits

* The protocol said the fold-0 stage-blind reference "must reproduce .61856". That
  number belongs to the covariance challenge's separate stream namespace. The archive
  here uses the declared `empirical_challenge_20261114` `empirical32` namespace,
  replicate 0. It is **byte-identical** to that study's fold-0 archive (sha256
  `79541774…`), and it reproduces the recorded fold-0 and fold-1 scores, .61186 and
  .63345, exactly. No decision depended on this number.
* The study used 10k updates rather than the 20k default, one seed per fold, and two
  folds of 12-13 reused development donors. It is not an independent test. A larger
  random-effect initialisation, a critic that sees several bags of the same stage
  together (between-donor statistics), or longer training could behave differently;
  none were tested. The fits ran under heavy shared-host load, at about 0.8 s per
  update, on cores 3-23 with GPU fraction .04.
* Default config, released ParticleGAN and every file in `/ml2/alz` are unchanged.

## Reproduce

```bash
PY=/mnt/nvme/ai/conda/anaconda3/bin/python3
D=analysis/donor_latent_20261120
CUDA_VISIBLE_DEVICES=0 taskset -c 3-23 $PY -B $D/run.py train --fold 0 --arm pbulk --updates 10000 --device cuda --gpu-mem-fraction .04
CUDA_VISIBLE_DEVICES=1 taskset -c 3-23 $PY -B $D/run.py train --fold 0 --arm g.dre,pbulk --updates 10000 --device cuda --gpu-mem-fraction .04
# ... same for --fold 1; then generate each fit (same flags, command `generate`)
$PY -B $D/score.py empirical --fold 0; $PY -B $D/score.py score --fold 0   # and fold 1
$PY -B $D/score.py compare
PYTHONPATH=/ml2/alz/artifacts/native-releases/particlegan-0.10.0/site-packages $PY -B -m pytest -q $D/test_donor_latent.py
```

Fits, archives and scores are under `/ml2/alz/runs/research_20261120/donor-latent/`
(outside git). [completion/artifacts.json](completion/artifacts.json) binds their hashes,
and [completion/comparison.json](completion/comparison.json) holds every contrast,
block and guard.
