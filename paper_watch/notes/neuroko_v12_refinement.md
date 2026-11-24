# Can tuned v12 outperform v11 and EMPF on NeuroKO?

The comparison uses the published 1,200-update neuronal benchmark with the same network, 64 × 16 learned particle table, batch size 64, training cells, 256 generated draws per target and four-target mean energy distance. The incumbent v12 uses hollow self-coupling, relativistic weight 2 and constant generator, critic and particle learning rates of 0.014, 0.021 and 0.0406. [Previous twelve-seed comparison](neuroko_v12_1200.md)

## Results at 1,200 updates

On 16 paired seeds (84–99), EMPF with the same latent noise and learning rates had the lowest mean held-out energy, 1.3531. Matched v11 averaged 1.3584 and the selected v12 averaged 1.3665. V12 scored lower on 6 of 16 seeds against each matched control. Its mean difference from EMPF was +0.0134 with a paired 95% interval of −0.0257 to +0.0525. This cohort leaves the direction of the formulation effect uncertain.

The primary v12 configuration increased latent noise from 0.025 to 0.15. Its average across the nine training fold–seed pairs fell from 0.9782 to 0.9417. Its held-out average also fell, from 1.3773 to 1.3665, with a paired 95% interval on the change of −0.0545 to +0.0329. Increasing the prior penalty from 0.02 to 0.1 instead reached 0.9641 in training validation and 1.3993 on the holdout.

All 208 held-out fits passed the constant-rate and matched-input checks. The primary and its two matched controls used generator 0.014, critic 0.021 and particles 0.0406. Timings are per fit with concurrent jobs on two RTX A6000 cards. The intervals describe variation across training seeds on one biological group. [Per-seed and per-target results, selected configurations and input hashes](neuroko_v12_fixed_1200.json)

| Model and setting | Mean energy ↓ | Mean spread ratio (ideal 1) | Median training time | Selected v12 minus arm, 95% paired interval |
| --- | ---: | ---: | ---: | ---: |
| V12, selected latent noise | 1.3665 | 1.051 | 125 s | Reference |
| V11, same noise and settings | 1.3584 | 1.042 | 136 s | +0.0081 [-0.0310, +0.0472] |
| EMPF, same noise and rates | 1.3531 | 1.043 | 201 s | +0.0134 [-0.0257, +0.0525] |
| V12, selected prior penalty | 1.3993 | 1.035 | 122 s | -0.0329 [-0.0672, +0.0015] |
| V11, same prior-penalty settings | 1.3574 | 1.035 | 134 s | +0.0090 [-0.0241, +0.0421] |
| EMPF, same prior penalty and rates | 1.3747 | 1.060 | 210 s | -0.0083 [-0.0383, +0.0218] |
| V11, previous same-wheel control | 1.3615 | 1.038 | 137 s | +0.0050 [-0.0297, +0.0397] |
| EMPF, previous matched-rate control | 1.3678 | 1.055 | 180 s | -0.0014 [-0.0400, +0.0372] |
| EMPF, independently tuned rates | 1.3777 | 1.059 | 181 s | -0.0113 [-0.0556, +0.0330] |
| V11, previously tuned on 0.11.1 | 1.3846 | 1.036 | 168 s | -0.0182 [-0.0581, +0.0217] |
| V12, previous selection | 1.3773 | 1.029 | 123 s | -0.0108 [-0.0545, +0.0329] |
| V12, previous v11 role rates | 1.3826 | 1.080 | 127 s | -0.0161 [-0.0469, +0.0146] |
| V12, release defaults at fixed capacity | 1.7905 | 0.649 | 123 s | -0.4240 [-0.4594, -0.3887] |

## Selection and confirmation

All eligible fits use constant generator, critic and particle learning rates throughout. The training search compares the incumbent with 16 alternatives. Six trials vary the constant rate of just one role while holding the other two fixed. Other trials vary optimizer momentum and smoothing, transport entropy and mass relaxation, active-pair critic weight, critic penalty and prior penalty. Each held-out receipt verifies the actual rates at initialization and at updates 300, 600, 900 and 1,200. The fixed-rate drivers also check the released scheduler at every update before each new fit and compare live optimizer group rates after each update. Cached receipts are checked against their recorded optimizer rates.

An initial screen started before the fixed-rate requirement. Five decay configurations were removed from the protocol before finalist selection. Only fixed-rate measurements enter the results. The superseded screen has a separate output directory.

A second search varies one quantity, the standard deviation of Gaussian noise around each sampled latent particle. The original value is 0.025. The four alternatives are 0.15, 0.4, 0.8 and 1.6. The released prior applies the specified value during both training and generation. The particle table has the same size and the same initialization. The matched v11 and EMPF controls receive the same noise as their v12 candidate.

Both searches use the same training groups. Each screen starts with seed 0 and holds out each of `Screen1_1`, `Screen1_3` and `Screen1_4` in turn. The four best training configurations and the two best noise configurations advance within their respective searches, alongside the incumbent. Seeds 1 and 2 repeat all three folds. The mean of the nine fold–seed scores determines each finalist and which finalist is the primary candidate. Ties break lexically by trial name. The selection runs use no expression measurements from the outer group for fitting or scoring.

We evaluate both finalists against the previously reported arms and against same-wheel v11 and EMPF matched to each candidate. The 16 paired seeds are 84–99. Each seed runs all arms on the same physical GPU, with rotated order and matching data, network initialization and minibatch sequence. We compare mean held-out energy, report paired 95% Student intervals over seeds and identify whether the primary v12 candidate has the lowest mean. The biological holdout `Screen1_0` has been used in earlier experiments. Its new training seeds quantify seed variability on that group.

## Reproduction

Use the [NeuroKO setup and 0.12.0 wheel installation](neuroko_v12_1200.md#reproduce-on-two-gpus). From the project root, with both original H5ADs and the shared Python environment available, run the following on two CUDA cards.

```bash
data=paper_watch/data/neuroko
base=(--data "$data" --window-cells 10000 --wheels "$data/wheels" --previous "$data/v12-1200")
python paper_watch/scripts/retune_neuroko_v12.py "${base[@]}" --study training --output "$data/v12-fixed"
python paper_watch/scripts/retune_neuroko_v12.py "${base[@]}" --study latent-noise --output "$data/v12-latent-noise"
python paper_watch/scripts/retune_neuroko_v12.py "${base[@]}" --stage confirm \
  --output "$data/v12-fixed" --companion "$data/v12-latent-noise"
```

For a previously extracted pilot, pass its directory as `--data` and omit `--window-cells`. If the previous v12 run directory is absent, the search computes the incumbent afresh. The joint output directory contains `freeze.json` with the selected parameters and controls, `heldout/` with individual receipts, and `result.json` with paired scores.

## Particle sampling comparison

All 108 training fits finished. Both v12 finalists scored above matched EMPF on average across the nine fold–seed pairs, so the prespecified rule ended the experiment before another holdout evaluation.

| Training sampler and v12 setting | V12 energy ↓ | Matched v11 | Matched EMPF |
| --- | ---: | ---: | ---: |
| Every particle once, hollow coupling, distance quantile 0.75 | 1.0018 | 1.0419 | 0.9761 |
| Every particle once, Feydy discrete self-coupling | 1.0032 | 0.9737 | 0.9761 |

With uniform sampling, the v12 references scored 0.9718 at noise 0.025 and 0.9966 at noise 0.15. The existing uniform v11 reference scored 0.9467, the matched-rate uniform EMPF reference 0.9645, and independently tuned uniform EMPF 0.9546. The EMPF column is shared because its loss has neither of the two v12 transport options. [All training-fold scores and selection decision](neuroko_v12_stratified_validation.json)

With 64 independent uniform particle draws from a table of 64, the expected number of distinct rows is about 41. ParticleGAN 0.12.0 offers a training sampler that draws a random permutation and uses all 64 table rows every update. Its evaluation mode samples with replacement. An integration test verifies both live training and the equality of the generated evaluation draws before fitting. All formulations share the sampler in the matched comparisons, the initialization, real minibatches and constant learning rates. [Upstream sampler](https://github.com/HyperGAN/ParticleGAN/blob/3256a05867688ab7ec0e5b4f50354c14d6972e6f/particlegan/priors.py#L473-L575)

At seed 3 the sampler experiment screens eight stratified v12 configurations on the three training groups. The top two face matched v11 and EMPF, the existing v11 and EMPF reference configurations, independently tuned EMPF and both uniform-sampling v12 references on seeds 3–5. A finalist qualifies for a new held-out comparison only if its average nine-fold–seed energy is lower than all seven controls. If both qualify, the lower v12 mean wins. The qualification and outputs are independent of the 84–99 holdout scores. [Configurations and qualification rule](../scripts/neuroko_stratified.py)

```bash
python paper_watch/scripts/neuroko_stratified.py --data "$data" --window-cells 10000 \
  --wheels "$data/wheels" --output "$data/v12-stratified"
```

If `selection.json` records a qualifying candidate, add `--stage confirm` to run its paired evaluation on seeds 100–115.

## Transport distance scale with uniform particle sampling

The 0.75 quantile with latent noise 0.025 led both matched controls on the three seed-6 folds. The result reversed on independent seeds 7–9. Across nine fold–seed pairs, v12 scored 0.9864, matched v11 scored 0.9594, and matched EMPF scored 0.9803. V12 won four of the nine fold–seed pairs against each matched control. Independently tuned EMPF scored 0.9503. The equivalent EMPF controls with explicit and default noise 0.025 reproduced all per-target scores exactly on all nine pairs. The study ended after 120 training fits. [Complete screening and independent validation](neuroko_v12_transport_scale_validation.json)

| Seed-6 distance quantile | Latent noise | V12 energy ↓ | Matched v11 | Matched EMPF |
| ---: | ---: | ---: | ---: | ---: |
| 75% | 0.025 | 0.9598 | 0.9842 | 0.9866 |
| 75% | 0.15 | 0.9575 | 0.9314 | 0.9480 |
| 95% | 0.025 | 0.9995 | 0.9356 | 0.9866 |
| 95% | 0.15 | 0.9483 | 0.9781 | 0.9480 |

In the stratified screen the higher nearest-neighbor distance quantile improved v12 across the three seed-3 folds. A separate experiment keeps uniform sampling, crosses quantile 0.75 or 0.95 with Gaussian latent noise 0.025 or 0.15, and ranks the four v12 settings by their difference from the better matched v11 or EMPF control. Each fit uses the same 1,200 updates and constant role rates. Seed 6 screens on the three training folds. The candidate with the largest advantage must then beat the matched controls and the previous uniform-sampling references on separate training seeds 7–9 before a new holdout comparison on seeds 116–131. [Selection rule and comparison arms](../scripts/neuroko_transport_scale.py)

```bash
python paper_watch/scripts/neuroko_transport_scale.py --data "$data" --window-cells 10000 \
  --wheels "$data/wheels" --output "$data/v12-transport-scale"
```

If a candidate qualifies, add `--stage confirm` to run the frozen holdout comparison. The holdout also includes two runs per seed on the original 0.10.0 EMPF wheel, one at the v12 candidate’s role rates and latent noise and one at the historical rates (0.005, 0.0075 and 0.0125).

A diagnostic replay checked whether the generator's adaptive gain could explain the v12 result on `Screen1_3` at seed 6, quantile 0.75 and latent noise 0.025. Both optimizers reported generator `ema_func_gain` and `last_step_gain` of 1.0 at updates 300, 600, 900 and 1,200. Each replay reproduced its original per-target scores exactly. A separate first-50-update check showed 50 generator probes for each formulation. V11 also ran 50 critic probes, while v12 ran zero, consistent with the upstream v12 optimizer. [Upstream generator and critic probe rules](https://github.com/HyperGAN/ParticleGAN/blob/3256a05867688ab7ec0e5b4f50354c14d6972e6f/particlegan/_unified_v12/optimizer.py#L702-L714)

## Critic gradient penalty threshold

Neither threshold qualified on training seed 11. At 0.75, v12 edged v11 and lost to EMPF. At 1.25, it scored above v11. All 18 fits used the intended native threshold and fixed learning rates.

| Critic threshold | V12 energy ↓ | Matched v11 | Matched EMPF |
| ---: | ---: | ---: | ---: |
| 0.75 | 1.0013 | 1.0043 | 0.9934 |
| 1.25 | 1.0193 | 0.9495 | 0.9805 |

[Per-fold energies and training decision](neuroko_v12_critic_cap_validation.json)

The transport-scale finalist scored 0.9864 across the nine independent training fold–seed pairs. Its matched v11 control scored 0.9594. This rules it out under the selection criterion. A separate fixed-rate experiment changes the native critic penalty threshold (`reg_kappa`), which sets the gradient norm above which the critic incurs a larger penalty. Both matched controls receive the same threshold.

Across 44 native penalty measurements after update 100 on training seed 10, the median reported norm was 0.835 for v12 and 0.718 for v11 under a common threshold of 1.0. This measurement used `Screen1_3` for validation and the default distance quantile 0.25. The new experiment compares thresholds 0.75 and 1.25 while keeping the quantile, latent noise, model, data and all three learning rates fixed. Seed 11 screens on three training folds. A setting that leads both matched controls must repeat against the matched and established controls on training seeds 12–14 to qualify for holdout seeds 132–147. [Native measurements and input hashes](neuroko_v12_critic_cap_diagnostic.json) · [Paired selection rule](../scripts/neuroko_transport_scale.py)

```bash
python paper_watch/scripts/neuroko_transport_scale.py --study critic-cap --data "$data" --window-cells 10000 \
  --wheels "$data/wheels" --output "$data/v12-critic-cap"
```

## Joint generator, critic and particle learning rates

Four of eight v12 combinations improved on the centered v12 score. The top three were then compared with v11 and EMPF at identical rates. Independently tuned EMPF reached 0.9174, lower than all three v12 scores. The training-only comparison covered 51 distinct fits.

| Generator LR | Critic LR | Particle LR | V12 energy ↓ | Matched v11 | Matched EMPF |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.02 | 0.042 | 0.0203 | 0.9433 | 0.9832 | 0.9932 |
| 0.01 | 0.0105 | 0.0812 | 0.9446 | 0.9444 | 0.9366 |
| 0.02 | 0.0105 | 0.0203 | 0.9746 | 0.9710 | 0.9682 |

The centered v12 and v11 controls scored 0.9772 and 0.9554. All fits used a fixed latent noise of 0.025. [Screen and per-fold paired results](neuroko_v12_role_factorial_validation.json)

The first fixed-rate search changed one role at a time around the v12 incumbent. To measure interactions, a separate experiment crosses a generator rate of 0.010 or 0.020 with a critic rate of 0.0105 or 0.042 and a particle rate of 0.0203 or 0.0812. All eight combinations use uniform sampling, latent noise 0.025, hollow self-coupling for v12 and v11, relativistic weight 2 and the default critic penalty. Each fit keeps its assigned rates constant through update 1,200.

The three v12 settings with the lowest mean energy at seed 15 face v11 and EMPF at the same three rates and three reference configurations. An eligible setting must beat all five controls. The lowest eligible v12 score advances to independent training seeds 16–18. Holdout seeds 148–163 are reserved for a candidate that beats the matched and previously tuned controls across these nine training fold–seed pairs. [Candidate grid and selection rule](../scripts/neuroko_role_factorial.py)

```bash
python paper_watch/scripts/neuroko_role_factorial.py --data "$data" --window-cells 10000 \
  --wheels "$data/wheels" --output "$data/v12-role-factorial"
```

## Replicated comparison of learning rates and latent noise

The two lowest seed-15 v12 scores came from different rate combinations. The first used generator 0.020, critic 0.042 and particles 0.0203. The second used 0.010, 0.0105 and 0.0812. Independently tuned EMPF scored 0.9174 on this seed, below both v12 scores. A follow-up tests each combination with Gaussian latent noise 0.075 or 0.15. All three formulations receive the same noise and rates within each comparison.

The screen uses six paired training fold–seed combinations. Each of three biological groups and each of seeds 19–21 occurs twice. The candidate with the lowest mean difference from its strongest competitor advances to three fresh seeds regardless of its screening rank relative to EMPF. This allows a promising setting to be assessed across seeds when one seed favors a control. The independent nine-pair validation compares the selected candidate with matched and established controls, including v12 at the same rates and its original noise of 0.025. Only a candidate below all these on validation goes to the held-out biological group on seeds 164–179. [Rate combinations, exact folds and ranking rule](../scripts/neuroko_rate_noise.py)

```bash
python paper_watch/scripts/neuroko_rate_noise.py --data "$data" --window-cells 10000 \
  --wheels "$data/wheels" --output "$data/v12-rate-noise"
```

The high-generator, high-critic, low-particle finalist also merits direct replication at its original noise 0.025. On seed 15 its mean of 0.9433 was lower than both exactly matched controls (v11 at 0.9832 and EMPF at 0.9932), while independently tuned EMPF still scored 0.9174. We froze that one configuration for an independent comparison on every training group at seeds 22–24. It shares the predeclared reference fits with the new noise comparison. The readout reports whether the original configuration beats both matched controls and whether it beats every reference. The latter is required for the separate holdout cohort on seeds 180–195. [Exact-rate replication rule](../scripts/neuroko_matched_rate_replication.py)

```bash
python paper_watch/scripts/neuroko_matched_rate_replication.py --data "$data" --window-cells 10000 \
  --wheels "$data/wheels" --output "$data/v12-matched-rate-replication" --shared-validation "$data/v12-rate-noise"
```

On three fresh training seeds (22–24), the original setting again had the lowest mean energy of the three exactly rate-matched formulations. V12 beat EMPF on seven of the nine fold–seed pairs and v11 on five. Averaged within each seed, v12 led both at seed 22 and both controls led at seeds 23 and 24. The biological holdout is excluded from these 27 fits. [Paired energies, target scores, rates and input hashes](neuroko_v12_matched_rates_1200.json)

| Formulation at identical rates | Mean energy ↓ | Spread ratio (ideal 1) | Effective dimensions | Median fit time |
| --- | ---: | ---: | ---: | ---: |
| V12 | 0.9774 | 1.121 | 16.89 | 146 s |
| V11 | 0.9867 | 1.127 | 16.39 | 164 s |
| EMPF | 0.9836 | 1.172 | 15.42 | 304 s |

The observed neurons averaged 17.44 effective dimensions on these groups, calculated from each target's covariance participation ratio. Each generated target contributed 256 draws. The fits ran concurrently on two RTX A6000 cards and timings include contention from other workloads.

The nine v12 fits at the original noise on seeds 22–24 held their generator, critic and particle rates at 0.020, 0.042 and 0.0203 through every one of the 1,200 updates. The optimizer also measures each generator step and can adjust its size in response. Its recorded `last_step_gain` was 1.0 on all nine fits at updates 300 and 600, and ranged from 0.856 to 0.996 at update 1,200. This adaptive gain is part of the released optimizer. [Per-update learning-rate check](../alztest/neuroko_tune.py#L249-L253) · [Upstream gain calculation and application](https://github.com/HyperGAN/ParticleGAN/blob/3256a05867688ab7ec0e5b4f50354c14d6972e6f/particlegan/_unified_v12/optimizer.py#L1205-L1228)

### All three formulations at EMPF’s tuned rates

The original matched-rate finalist moved the critic rate above and the particle rate below the tuned EMPF values. A second fixed comparison assigns all three models EMPF’s previously tuned rates of 0.020, 0.030 and 0.058. The v12 and v11 arms keep hollow coupling and relativistic weight 2. All models use latent noise 0.025. This comparison shares the exact tuned EMPF receipts with the other validation on seeds 22–24. The same selection threshold applies to the nine independent fold–seed pairs, with outer seeds 196–211 reserved in advance. [Comparison arms and decision rule](../scripts/neuroko_matched_rate_replication.py)

```bash
python paper_watch/scripts/neuroko_matched_rate_replication.py --rates empf --data "$data" --window-cells 10000 \
  --wheels "$data/wheels" --output "$data/v12-at-empf-rates" --shared-validation "$data/v12-rate-noise"
```
