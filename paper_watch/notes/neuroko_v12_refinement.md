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

## Six configurations on the same nine training pairs

The stronger-prior v12 had the lowest mean energy of all 18 entries below on screening seeds 22–24. Its 0.9423 was 0.0038 below the lowest EMPF mean and 0.0039 below the lowest v11 mean. At penalty weight 0.1 it beat matched EMPF on four of nine pairs and matched v11 on six. Only seed 22 favored v12 when the three groups were averaged within each seed against matched EMPF. The setting qualified for independent training seeds 251–253. [All six configurations, paired energies, timings and input hashes](neuroko_v12_six_settings_1200.json) · [Prior-weight screening decision](neuroko_v12_prior_weight_screen.json)

| Generator / critic / particle LR | Latent noise | Prior weight | V12 energy ↓ | V11 | EMPF |
| --- | ---: | --- | ---: | ---: | ---: |
| 0.014 / 0.021 / 0.0406 | 0.025 | Release default | 0.9729 | 0.9462 | 0.9722 |
| 0.014 / 0.021 / 0.0406 | 0.150 | Release default | 0.9734 | 0.9506 | 0.9461 |
| 0.014 / 0.021 / 0.0406 | 0.150 | 0.1 for all | **0.9423** | 0.9675 | 0.9518 |
| 0.020 / 0.042 / 0.0203 | 0.025 | Release default | 0.9774 | 0.9867 | 0.9836 |
| 0.020 / 0.042 / 0.0203 | 0.150 | Release default | 0.9797 | 0.9778 | 0.9900 |
| 0.020 / 0.030 / 0.0580 | 0.025 | Release default | 0.9639 | 0.9531 | 0.9659 |

Every entry covers nine 1,200-update runs on the three training groups. The three formulations share the learning rates, noise, network size, initial parameters and real minibatches in each row. All rates stayed constant at every update. The released prior weights are 0.02 for v12 and v11 and 0.01 for EMPF. Shared control fits appear once in this table. [Earlier finalist with exact rates](neuroko_v12_matched_rate_replication_validation.json) · [All controls at EMPF’s tuned rates](neuroko_v12_at_empf_rates_validation.json)

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

Across the six screening pairs on seeds 19–21, v12 at latent noise 0.15 and fixed generator, critic and particle rates of 0.020, 0.042 and 0.0203 scored 0.9466. It beat matched EMPF on five pairs and independently tuned EMPF on four. V11 at the reference rates of 0.014, 0.021 and 0.0406 had the lowest mean, 0.9339. The frozen ranking rule selected the 0.9466 v12 configuration for independent training validation on seeds 22–24. [All 84 fit scores and the selection](neuroko_v12_rate_noise_screen.json)

| Generator LR | Critic LR | Particle LR | Latent noise | V12 energy ↓ | Matched v11 | Matched EMPF |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.020 | 0.042 | 0.0203 | 0.075 | 0.9735 | 1.0015 | 1.0053 |
| 0.020 | 0.042 | 0.0203 | 0.150 | 0.9466 | 0.9654 | 1.0125 |
| 0.010 | 0.0105 | 0.0812 | 0.075 | 0.9700 | 0.9734 | 0.9940 |
| 0.010 | 0.0105 | 0.0812 | 0.150 | 0.9785 | 0.9792 | 0.9774 |

Independently tuned EMPF scored 0.9605 on the same six pairs at rates of 0.020, 0.030 and 0.058 with latent noise 0.025. In the direct formulation comparisons, every model in a row had identical learning rates and noise.

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

### Effect of increasing latent noise on fresh training seeds

At the fixed rates of 0.020, 0.042 and 0.0203, raising the latent noise from 0.025 to 0.15 changed mean v12 energy from 0.9774 to 0.9797 on seeds 22–24. The higher noise improved energy on four of nine pairs. V11 at its original rates and noise 0.15 scored 0.9506 and led v12 when averaged within each of the three seeds. At identical rates and noise, matched v11 averaged 0.9778 and matched EMPF 0.9900. The full 11-arm validation ended before the biological holdout. The nine low-noise v12 replays reproduced every original per-target score exactly. [Independent decision and all 99 scores](neuroko_v12_rate_noise_validation.json)

The extra noise increased the mean number of effective covariance dimensions from 16.89 to 17.42, with an increase on seven of nine pairs. The same target groups in the observed data averaged 17.44. The generated-to-observed spread ratio rose from 1.121 to 1.134. Thus the model used more covariance directions while producing slightly more total spread than the observed cells. [Paired target energies and effective dimensions](neuroko_v12_noise_ablation_1200.json)

### All three formulations at EMPF’s tuned rates

The original matched-rate finalist moved the critic rate above and the particle rate below the tuned EMPF values. A second fixed comparison assigns all three models EMPF’s previously tuned rates of 0.020, 0.030 and 0.058. The v12 and v11 arms keep hollow coupling and relativistic weight 2. All models use latent noise 0.025. This comparison shares the exact tuned EMPF receipts with the other validation on seeds 22–24. The same selection threshold applies to the nine independent fold–seed pairs, with outer seeds 196–211 reserved in advance. [Comparison arms and decision rule](../scripts/neuroko_matched_rate_replication.py)

```bash
python paper_watch/scripts/neuroko_matched_rate_replication.py --rates empf --data "$data" --window-cells 10000 \
  --wheels "$data/wheels" --output "$data/v12-at-empf-rates" --shared-validation "$data/v12-rate-noise"
```

At EMPF’s tuned rates, v12 averaged 0.9639 and EMPF averaged 0.9659 on the same nine pairs. V12 won five of nine by energy and two of three when averaged within seed. V11 had the lowest mean energy at 0.9531, which rules out this setting under its training qualification. [Paired scores, diversity, timings and input hashes](neuroko_v12_at_empf_rates_1200.json)

| Formulation at rates 0.020 / 0.030 / 0.058 | Mean energy ↓ | Spread ratio (ideal 1) | Effective dimensions | Median fit time |
| --- | ---: | ---: | ---: | ---: |
| V11 | 0.9531 | 1.146 | 18.00 | 161 s |
| V12 | 0.9639 | 1.118 | 17.50 | 152 s |
| EMPF | 0.9659 | 1.174 | 17.16 | 310 s |

At noise 0.025, moving from the earlier rates of 0.020, 0.042 and 0.0203 to these rates improved v12’s mean by 0.0135 and v11’s by 0.0336 on the same seed–group pairs. This larger change in v11 reversed their ranking. The higher-noise v12 finalist at the earlier rates scored 0.9797 against this independently tuned EMPF reference of 0.9659.

### Checking the original releases at both rate settings

To test whether the matched comparisons also hold for EMPF from 0.10.0 and Unified v11 from 0.11.1, run these two commands after the corresponding same-wheel comparisons. Both repeat seeds 22–24 on the same three training groups and retain the native prior weights of 0.02 for v12 and v11 and 0.01 for EMPF. The experiments with weight 0.1 apply that same coefficient to all three formulations. The scores of all three 0.12.0 formulations must reproduce the published values, and each original-release fit must match the corresponding initial weights, training data, minibatches, physical GPU and fixed learning rates. The biological holdout decisions above use the original qualification rules.

```bash
python paper_watch/scripts/neuroko_matched_rate_replication.py --stage compare-releases --rates seed15 \
  --data "$data" --window-cells 10000 --wheels "$data/wheels" --shared-validation "$data/v12-rate-noise" \
  --output "$data/v12-matched-rate-replication"
python paper_watch/scripts/neuroko_matched_rate_replication.py --stage compare-releases --rates empf \
  --data "$data" --window-cells 10000 --wheels "$data/wheels" --shared-validation "$data/v12-rate-noise" \
  --output "$data/v12-at-empf-rates"
```

## Prior penalty at latent noise 0.15

At the release role rates, two individual changes lowered v12's mean energy on earlier training folds. Increasing latent noise to 0.15 moved the mean from 0.9782 to 0.9417. Increasing the native prior penalty weight from 0.02 to 0.1 at the original noise moved it to 0.9641. The follow-up combines those two changes at constant generator, critic and particle rates of 0.014, 0.021 and 0.0406. It tests whether the stronger penalty helps with the higher noise. A training qualification for the preceding rate-and-noise candidate gives its biological holdout priority.

A 32-update CUDA check on the 30-dimensional neuronal projection verified that the released v12 training loss actually applies this coefficient. At noise 0.15 and rates 0.020, 0.042 and 0.0203, the gradient arriving at the native prior penalty equaled its configured weight on every update in both runs. The models started from identical weights, saw the same minibatches and finished with different parameter hashes. [Measured gradients and input hashes](neuroko_v12_prior_weight_diagnostic.json)

Across the nine screening pairs, increasing the v12 prior weight from 0.02 to 0.1 lowered mean energy from 0.9734 to 0.9423. The higher weight improved six of nine pairs, two of three in each training group. All four targets improved on average, with the largest reductions for MAMSTR (0.0678) and GLCCI1 (0.0400). The average spread ratio barely moved, from 1.1262 to 1.1252.

Six deterministic replays covered both weights on one seed per training group. Each replay reproduced every original per-target score. The higher weight reduced mean distance to the observed target centroid from 2.280 to 2.263 and to the training target centroid from 0.543 to 0.507. After recentering generated samples on the observed centroid for diagnosis, energy was 0.1174 at weight 0.02 and 0.1226 at weight 0.1. This pattern points to better centroid estimates as a source of the energy gain in these three pairs. The benchmark uses the original generated samples for all qualification scores. [Nine paired results and three geometry comparisons](neuroko_v12_prior_weight_geometry.json)

V12, v11 and EMPF all receive weight 0.1 in the matched arms. The release-default reference arms retain each recipe’s own weight (0.02 for v12 and v11, 0.01 for EMPF). On all three training groups at seeds 22–24, v12 must beat both weight-matched controls and v12 at the same noise and rates with weight 0.02. A passing configuration goes to three new training seeds, 251–253. Only a lower mean than every reference there qualifies for the biological holdout on seeds 212–227. [Fixed settings and selection rule](../scripts/neuroko_rate_noise.py)

```bash
python paper_watch/scripts/neuroko_rate_noise.py --study prior-weight --data "$data" --window-cells 10000 \
  --wheels "$data/wheels" --output "$data/v12-noise-prior" --shared-screen "$data/v12-rate-noise"
```

Add `--stage confirm` after a qualifying independent result.

For the comparison with the original releases, run the same prior-weight command with `--stage compare-releases` after the screen passes. It fits EMPF from ParticleGAN 0.10.0 and Unified v11 from 0.11.1 at the exact candidate rates, noise and penalty weight on seeds 251–253. It also scores those recipes and v12 from the 0.12.0 wheel on the same nine pairs. The results go to `release_comparison.json`. The independent training qualification still uses the controls in the frozen prior-weight study.

On fresh training seeds 251–253, v12 averaged 0.94813 and matched 0.12.0 EMPF averaged 0.93927. V12 won three of nine pairs and one of three seed means. The largest gap was on `Screen1_4`, where EMPF won at each seed. V12 led on `Screen1_3` by 0.0320 in the three-seed mean. These results rule out the stronger-prior candidate at noise 0.15 under its predefined biological holdout criterion.

Matched v11 from the 0.12.0 wheel averaged 0.93629. V12 beat it on three of nine pairs and one of the three seed means. Within v12 itself, increasing the penalty from 0.02 to 0.1 also helped on the fresh seeds. Mean energy fell from 0.97650 to 0.94813, an improvement of 2.9%. The higher weight improved seven of nine pairs, all three seed means and each of the four target means.

The prior weight had different effects across formulations at noise 0.15. In both cohorts, weight 0.1 lowered v12 energy and raised v11 energy. Across the 18 fold–seed pairs, the stronger prior improved v12 on 13 and v11 on five. EMPF's mean moved by less than 0.006 in either cohort. On fresh seeds, v11 at its default weight scored 0.91437, the lowest of the ten arms and 0.03376 below the tuned v12. [All 90 independent scores and the qualification decision](neuroko_v12_prior_weight_validation.json)

| Change in energy from release-default weight to 0.1 | Seeds 22–24 | Seeds 251–253 |
| --- | ---: | ---: |
| V12 (default 0.02) | −0.03113 | −0.02837 |
| V11 from 0.12.0 (default 0.02) | +0.01696 | +0.02192 |
| EMPF from 0.12.0 (default 0.01) | +0.00570 | −0.00141 |

| Fresh seed, mean over three training folds | V12 energy ↓ | Matched v11 (0.12.0) | Matched EMPF (0.12.0) |
| --- | ---: | ---: | ---: |
| 251 | 0.97204 | 0.93236 | 0.93740 |
| 252 | 0.93266 | 0.91767 | 0.97112 |
| 253 | 0.93969 | 0.95885 | 0.90930 |

The original 0.10.0 EMPF reproduced the 0.12.0 EMPF per-target measurements exactly across all nine pairs. The original 0.11.1 v11 averaged 0.93856. V12 scored lower on three of nine pairs against each original release. The median fit took 124 s for v12, 147 s for original EMPF and 155 s for original v11 under concurrent GPU workloads. Every one of the 45 fits across the five recipe and release combinations verified all 1,200 updates and matched its corresponding input, initialization and minibatch hashes. [Pinned settings, paired scores, rate checks and timings](neuroko_v12_prior_weight_original_releases.json)

Across all five combinations the generator, critic and particle rates held at 0.014, 0.021 and 0.0406. V12's release recipe sets both `lr_floor` and `network_lr_floor` to 1.0. We checked its schedule at every update through 40,000 and both multipliers stayed at 1.0. In the nine 1,200-update v12 runs, its generator and critic adaptive gains also read 1.0 at each of the 36 logged checkpoints per role. For fixed-rate comparisons, the runner replays older cached fits that lack the per-update verification count.

To measure the learning curves and sensitivity to the 256 generated samples per target, replay the nine fixed training pairs for the three original releases. The diagnostic checks that each replay reproduces every published target score and input hash. It measures energy, spread and covariance participation ratio at 300, 600, 900 and 1,200 updates using the original draw for each fit. At 1,200 it also samples the fitted generator with eight additional seed sets shared across all folds and formulations. The average over draws gives one score per fitted model. The original benchmark scores determine holdout eligibility. [Replay and aggregation code](../scripts/neuroko_evaluation_draws.py)

```bash
python paper_watch/scripts/neuroko_evaluation_draws.py --data "$data" --window-cells 10000 \
  --wheels "$data/wheels" --output "$data/v12-evaluation-draws"
```

All 27 GPU replays reproduced the original per-target scores. V12 improved between 300 and 1,200 updates on six of nine training pairs. Both v12 and EMPF had a small rebound in mean energy at update 900 and reached their lowest measured mean at 1,200. The fixed v12 learning rates held through that rebound.

| Original release | Energy at 300 ↓ | 600 | 900 | 1,200 | 1,200 averaged over eight additional draw sets |
| --- | ---: | ---: | ---: | ---: | ---: |
| EMPF 0.10.0 | 1.0656 | 0.9577 | 0.9694 | 0.9393 | 0.9435 |
| Unified v11 0.11.1 | 0.9853 | 0.9737 | 0.9680 | 0.9386 | 0.9338 |
| Unified v12 0.12.0 | 0.9958 | 0.9652 | 0.9740 | 0.9481 | 0.9466 |

Averaging the eight new draw sets within each of the same nine fits per arm put v12 0.0030 above EMPF and 0.0127 above v11. V12 beat EMPF on five of the nine averaged pairs and v11 on two. The v12-minus-EMPF mean ranged from −0.0053 to +0.0128 across draw sets. These repeated draws measure sampling variation within the trained models.

V12's effective covariance dimensions grew from 12.13 at update 300 to 17.80 at 1,200 on the original draws. EMPF grew from 9.65 to 16.75 and v11 from 11.07 to 17.96. The observed cells averaged 17.44. On the eight additional final draw sets, v12 averaged 17.66 dimensions with 1.130 times the observed spread, compared with 16.64 and 1.174 for EMPF and 17.86 and 1.122 for v11. [Paired learning curves, evaluation draws and diversity measurements](neuroko_v12_evaluation_draws.json)

### Stronger prior at the replicated matched rates

At generator 0.020, critic 0.042, particles 0.0203 and latent noise 0.025, v12 had lower mean energy than both matched formulations on seed 15 and again on seeds 22–24. We froze a single change to that configuration, increasing the prior weight to 0.1 and giving both controls the same coefficient. On seeds 22–24, v12 must score below both controls and its own previous weight of 0.02. A passing setting must then beat all references on fresh seeds 254–256, including the preceding v12 settings at both noise levels, before a biological holdout comparison on seeds 228–243. All runs keep the original network, particle table, draws and 1,200 constant-rate updates.

```bash
python paper_watch/scripts/neuroko_rate_noise.py --study matched-prior-weight --data "$data" --window-cells 10000 \
  --wheels "$data/wheels" --output "$data/v12-matched-prior" --shared-screen "$data/v12-rate-noise"
```

At these rates, raising the prior weight from 0.02 to 0.1 increased v12's nine-pair mean energy from 0.97744 to 0.98734. The stronger weight improved four of nine pairs and one of three seed means, which rules out independent validation under the screening criterion. All 18 v12 fits verified constant learning rates at every update. At update 1,200, the native generator step gain averaged 0.9190 with weight 0.02 and 0.9289 with weight 0.1.

At weight 0.1, matched EMPF averaged 0.96680, v12 averaged 0.98734 and matched v11 averaged 0.98923. V12 beat v11 on five of nine pairs and EMPF on four. The completed 12-arm screen ended at the training folds. [All 108 paired energies and the screening decision](neuroko_v12_matched_prior_screen.json)

### Doubling the prior penalty at latent noise 0.15

At noise 0.15 and generator, critic and particle rates of 0.014, 0.021 and 0.0406, the within-v12 improvement at weight 0.1 repeated on the fresh cohort. We chose a single further increase to 0.2, with all learning rates fixed and the larger weight assigned to all three formulations. On seeds 22–24, weight-0.2 v12 must beat matched v11 and EMPF as well as all three models at weight 0.1. Independent training seeds 257–259 and biological holdout seeds 300–315 were reserved before fitting. Only a lower independent mean than every frozen reference qualifies for that holdout.

```bash
python paper_watch/scripts/neuroko_rate_noise.py --study double-prior-weight --data "$data" --window-cells 10000 \
  --wheels "$data/wheels" --output "$data/v12-double-prior" --shared-screen "$data/v12-rate-noise"
```

Across all nine screening pairs, doubling the weight from 0.1 to 0.2 raised v12's mean energy from 0.94231 to 0.98036. The higher weight improved one pair and increased energy in each of the three seed means. At weight 0.2, EMPF scored 0.95944 and v11 scored 0.96545. V12 beat each matched control on four of nine pairs. The study ended on the training folds. [All 117 paired scores and the screening decision](neuroko_v12_double_prior_screen.json)

### Separating the effects of latent noise and role rates

Weight 0.1 helped v12 at noise 0.15 and release rates. It hurt at noise 0.025 and the higher critic rate. Both noise and rates changed between those comparisons. To measure the dependence on each, the next study fills the two missing v12 combinations in a grid of two rate settings, two noise levels (0.025 and 0.15) and two weights (0.02 and 0.1). Each combination uses the same nine seed–group pairs and verifies all 1,200 learning rates.

The one new candidate for independent validation uses rates 0.020, 0.042 and 0.0203, noise 0.15 and weight 0.1. To advance, it must beat its lower-weight counterpart and all v11 and EMPF controls in the screen. Because the ranking changed between earlier three-seed cohorts, this study reserves six fresh training seeds (260–265) for 18 paired comparisons, then outer seeds 316–331 for a qualifying result. The low-noise, release-rate corner completes the comparison of training settings.

```bash
python paper_watch/scripts/neuroko_rate_noise.py --study prior-interaction --data "$data" --window-cells 10000 \
  --wheels "$data/wheels" --output "$data/v12-prior-interaction" --shared-screen "$data/v12-rate-noise"
```

Across the 72 v12 fits, increasing the prior weight from 0.02 to 0.1 raised mean energy slightly at noise 0.025 and lowered it at noise 0.15 under both rate settings. The change in the effect of weight between the two noise levels was −0.03495 at the release rates and −0.03575 at the higher critic rate. At both settings the stronger weight improved six of nine pairs with higher noise and four of nine with lower noise. [Paired contrasts for all eight combinations](neuroko_v12_prior_effects.json)

| Generator / critic / particle LR | Latent noise | V12 at weight 0.02 | V12 at weight 0.1 | Change in energy ↓ |
| --- | ---: | ---: | ---: | ---: |
| 0.014 / 0.021 / 0.0406 | 0.025 | 0.97295 | 0.97676 | +0.00381 |
| 0.014 / 0.021 / 0.0406 | 0.150 | 0.97344 | 0.94231 | −0.03113 |
| 0.020 / 0.042 / 0.0203 | 0.025 | 0.97744 | 0.98734 | +0.00990 |
| 0.020 / 0.042 / 0.0203 | 0.150 | 0.97967 | 0.95381 | −0.02586 |

At noise 0.15 and weight 0.1, the higher-critic candidate scored 0.95381, matched v11 scored 0.94923, and matched EMPF scored 0.98896. An earlier EMPF reference scored 0.94613 on the same nine pairs. V12 beat its matched EMPF control and missed the v11 and earlier EMPF controls, ending this trial at the first screen. [Full comparison and selection decision](neuroko_v12_prior_interaction_screen.json)

### Increasing noise with the stronger prior

At weight 0.1, increasing noise from 0.025 to 0.15 improved v12 at both rate settings. The next comparison increases noise one more time, to 0.25, between the previously tested values 0.15 and 0.4. It keeps weight 0.1 and the release learning rates of 0.014, 0.021 and 0.0406 for all three matched formulations.

The candidate must beat every reference, including all three lower-noise arms, first on seeds 22–24 and then separately on seeds 251–253. Both cohorts have informed earlier experiments. Passing both qualifies the candidate for six fresh training seeds (266–271), where it must also beat the original 0.10.0 EMPF and 0.11.1 v11 at the candidate settings. A qualifying independent result goes to the same biological holdout on paired seeds 332–347. Every fit uses the original architecture and 1,200 fixed-rate updates.

```bash
python paper_watch/scripts/neuroko_rate_noise.py --study strong-prior-noise --data "$data" --window-cells 10000 \
  --wheels "$data/wheels" --output "$data/v12-strong-prior-noise" \
  --shared-screen "$data/v12-rate-noise" --second-screen "$data/v12-noise-prior"
```

At fixed generator, critic and particle rates of 0.014, 0.021 and 0.0406 and a shared prior weight of 0.1, increasing noise helped v11 and EMPF and raised v12's mean energy. The higher-noise v12 failed its first development screen. [All paired scores and selection decision](neuroko_v12_strong_prior_noise_screen.json)

| Formulation from 0.12.0 | Energy at noise 0.15 ↓ | Energy at noise 0.25 ↓ | Pairs improved at 0.25 |
| --- | ---: | ---: | ---: |
| Unified v12 | 0.94231 | 0.94988 | 4/9 |
| Unified v11 | 0.96751 | 0.92511 | 8/9 |
| EMPF | 0.95183 | 0.92240 | 6/9 |

### Separating training noise from sampling noise

The native Gaussian noise affects both the training updates and the generated samples used for scoring. To distinguish their contributions to the 0.97676 to 0.94231 change in v12 energy, replay the 18 fits at the release rates and prior weight 0.1. For each of the nine seed–group pairs, train once at noise 0.025 and once at 0.15. Score each fitted generator at both sampling widths with identical particle and Gaussian draws. The original samples must reproduce exactly after restoring the native width. The diagnostic checks every original target score and input hash.

At a fixed sampling width, the energy difference measures the effect of training at higher noise. For a given fitted generator, it measures the effect of changing sampling noise alone. Averaging the two training differences and the two sampling differences gives contributions that sum to the change between the two original configurations. The report includes both the benchmark draws and eight additional draw sets shared across training settings. Qualification uses the original training and sampling widths.

```bash
python paper_watch/scripts/neuroko_noise_crossover.py --data "$data" --window-cells 10000 \
  --wheels "$data/wheels" --reference "$data/v12-rate-noise" --output "$data/v12-noise-crossover"
```

Across the nine paired fits and eight additional draw sets, raising both widths lowered mean energy by 0.03401. Training accounted for 0.02419 of that decrease (71%) and sampling accounted for 0.00982 (29%). The training contribution improved seven of nine pairs, and the sampling contribution improved all nine. On the original benchmark draws the corresponding decreases were 0.02418 and 0.01027. Every one of the 18 replays reproduced its original target scores. [Full paired measurements and decomposition](neuroko_v12_noise_crossover.json)

| Training noise | Energy at sampling noise 0.025 ↓ | Energy at sampling noise 0.15 ↓ |
| ---: | ---: | ---: |
| 0.025 | 0.98028 | 0.97074 |
| 0.15 | 0.95636 | 0.94627 |

### Calibrating the critic learning rate from its measured updates

At generator, critic and particle rates of 0.014, 0.021 and 0.0406, v12's first update moved the critic's scores 5.20 times as far as same-wheel v11's. The median cosine between the changes was 0.996. The optimizer received the same initialized model, training cells, minibatches and loss configuration in each pair. The probes comprised 32 fixed real training cells per condition. We subtracted each condition's mean score change before computing the RMS, so an additive critic offset drops out.

Across the first 64 updates, the median v12-to-v11 ratio was 6.06. It ranged from 5.52 to 6.40 across the nine seed–group pairs. Each ratio compares the median size of an update in that run. The trajectories diverge after the first update. Replaying both formulations without the probes on one pair reproduced their final weight hashes. [Per-update movements, paired hashes and probe settings](neuroko_v12_critic_motion.json)

The released optimizers transform dense critic gradients differently. V12's fast path scales a normalized gradient using the matrix dimensions. V11 applies a five-step matrix orthogonalization. The installed source files for both optimizers were byte-identical to the tagged GitHub source. The sixfold ratio is a measurement on this network and these training cells. [V12 dense critic path](https://github.com/HyperGAN/ParticleGAN/blob/3256a05867688ab7ec0e5b4f50354c14d6972e6f/particlegan/_unified_v12/optimizer.py#L917-L943) · [V11 dense matrix update](https://github.com/HyperGAN/ParticleGAN/blob/3256a05867688ab7ec0e5b4f50354c14d6972e6f/particlegan/_unified_v11/optimizer.py#L1112-L1139)

Dividing 0.021 by the measured ratio and rounding to four decimal places gives a critic LR of **0.0035**. The 1,200-update study gives that fixed rate to all three matched formulations, with generator 0.014, particles 0.0406, latent noise 0.15 and prior weight 0.1. It retains the original-rate controls. The candidate must beat the controls separately on development seeds 22–24 and 251–253, then on fresh seeds 272–277 including the two original releases. A qualifying result goes to the biological holdout on seeds 348–363. A short repeat measures the actual critic movement at the new rate.

With the v12 critic reduced to 0.0035 and the reference v11 critic at 0.021, the first-update ratio fell to 0.865 and the median 64-update ratio fell to 0.722 (range 0.702–0.743). All nine reference v11 traces reproduced exactly. The formula slightly overcorrected the measured 64-update movement. Interpolating the two measured ratios, 0.722 and 6.060, against their respective learning rates in log space gives **0.0046** for a ratio of 1. A separate 64-step probe tests this prediction. The 1,200-update comparison above keeps its original fixed rate of 0.0035. [Paired rate-check traces](neuroko_v12_critic_motion_check.json)

The repeat at 0.0046 produced a median 64-update ratio of **0.974**, with a range of 0.941–1.004 across the nine pairs. On the first step alone the ratio was 1.137. The training score comparison at 0.0046 includes v11 and EMPF at the same 0.0046 rate, and all three formulations at 0.0035 and 0.021. It uses the same two development cohorts followed by fresh qualification seeds 278–283. Only a qualifying setting goes to the biological holdout on seeds 364–379. [Per-update measurements at 0.0046](neuroko_v12_critic_motion_interpolated.json)

```bash
PYTHONPATH="$data/wheels/0.12.0:." python paper_watch/scripts/neuroko_critic_motion.py \
  --data "$data" --window-cells 10000 --reference "$data/v12-rate-noise" --output "$data/v12-critic-motion"
PYTHONPATH="$data/wheels/0.12.0:." python paper_watch/scripts/neuroko_critic_motion.py \
  --data "$data" --window-cells 10000 --reference "$data/v12-rate-noise" \
  --output "$data/v12-critic-motion-check" --v12-critic-lr 0.0035
python paper_watch/scripts/neuroko_rate_noise.py --study critic-motion --data "$data" --window-cells 10000 \
  --wheels "$data/wheels" --motion-result "$data/v12-critic-motion/result.json" \
  --output "$data/v12-motion-calibrated" --shared-screen "$data/v12-rate-noise" \
  --second-screen "$data/v12-noise-prior"
python paper_watch/scripts/neuroko_rate_noise.py --study matched-critic-motion --data "$data" --window-cells 10000 \
  --wheels "$data/wheels" --motion-result "$data/v12-critic-motion/result.json" \
  --motion-check "$data/v12-critic-motion-check/result.json" --output "$data/v12-motion-matched" \
  --shared-screen "$data/v12-rate-noise" --second-screen "$data/v12-noise-prior"
```

At a critic LR of 0.0035, v12's mean energy at 1,200 updates rose from 0.94231 to 0.95185. Four of nine pairs improved. The lower rate helped on average on `Screen1_1` and `Screen1_4` and increased mean energy on `Screen1_3` from 0.5665 to 0.6138. The generated spread ratio rose from 1.125 to 1.200 and the effective covariance dimensions rose from 17.49 to 22.60. The observed cells averaged 17.44 dimensions. The original-rate v12 comparison rules out 0.0035 under this study's criterion.

At 0.0046 with the same prior weight of 0.1, v12 averaged **0.94547** across the nine pairs, compared with **0.94231** at 0.021. It had lower energy at seeds 22 and 23 and higher energy at seed 24 (0.97746 versus 0.93755). Both reduced critic rates missed the original-rate v12 reference at 1,200 updates. [Nine paired scores for each fixed critic rate](neuroko_v12_critic_rate_screen.json)

### Particle gradients under the smaller critic update

The generator objective has an adversarial term and a weighted particle prior penalty. Reducing the critic rate can change the relative strengths of their gradients on the learned 64 × 16 particle table even when the nominal prior weight stays fixed. We recorded both native gradients separately at each of the first 64 updates. The 36 instrumented fits reproduced the earlier final weight hashes and critic traces with the same initialization, data and minibatches.

| Training recipe and critic LR | Median adversarial gradient norm | Median weighted prior gradient norm | Median prior/adversarial norm ratio |
| --- | ---: | ---: | ---: |
| V11, 0.021 | 0.02030 | 0.00558 | 0.281 |
| V12, 0.021 | 0.04236 | 0.00560 | 0.136 |
| V12, 0.0046 | 0.01726 | 0.00559 | 0.325 |

The table takes a median within each fit, then across the nine seed–group pairs. The particle penalty had **2.41 times** its previous strength relative to the adversarial gradient under the smaller critic rate. This ratio increased on all nine pairs, by factors from 1.99 to 2.53. Dividing the original weight of 0.1 by the median paired factor gives **0.041** to three decimal places. [Paired measurements and weight calculation](neuroko_v12_particle_balance.json)

A second 64-update measurement at weight 0.041 put the median relative strength at **1.009 times** the original v12 value, with a paired range of 0.829–1.054. Critic motion stayed at 0.974 times the v11 reference. The 1,200-update energy score measures whether this also improves the generated cells. [Measured gradient balance at weight 0.041](neuroko_v12_particle_balance_check.json)

The 1,200-update study tests this one weight for v12, v11 and EMPF at fixed generator, critic and particle rates of 0.014, 0.0046 and 0.0406, with latent noise 0.15. The candidate must beat every reference separately on development seeds 22–24 and 251–253, including the three formulations with weight 0.1 at each of the three critic rates. Fresh seeds 284–289 include the original 0.10.0 EMPF and 0.11.1 v11. A qualifying result goes to biological holdout seeds 380–395.

```bash
for setting in original matched; do
  if [ "$setting" = original ]; then critic=0.021; else critic=0.0046; fi
  PYTHONPATH="$data/wheels/0.12.0:." python paper_watch/scripts/neuroko_critic_motion.py \
    --data "$data" --window-cells 10000 --reference "$data/v12-rate-noise" --particle-gradients \
    --v12-critic-lr "$critic" --output "$data/v12-particle-gradients-$setting"
done
PYTHONPATH="$data/wheels/0.12.0:." python paper_watch/scripts/neuroko_critic_motion.py \
  --data "$data" --window-cells 10000 --reference "$data/v12-rate-noise" --particle-gradients \
  --v12-critic-lr 0.0046 --v12-prior-weight 0.041 --output "$data/v12-particle-gradients-check"
python paper_watch/scripts/neuroko_particle_balance.py --original "$data/v12-particle-gradients-original/result.json" \
  --calibrated "$data/v12-particle-gradients-matched/result.json" --output "$data/particle-balance.json"
python paper_watch/scripts/neuroko_rate_noise.py --study balanced-particle-prior --data "$data" --window-cells 10000 \
  --wheels "$data/wheels" --gradient-result "$data/particle-balance.json" --output "$data/v12-balanced-particles" \
  --shared-screen "$data/v12-rate-noise" --second-screen "$data/v12-noise-prior"
```

At 1,200 updates the corrected weight of 0.041 scored **0.94752** across all nine pairs. At the same critic LR of 0.0046, weight 0.1 scored 0.94547. With the original critic LR of 0.021 and weight 0.1, v12 scored 0.94231. The correction improved four of nine pairs against each reference. Its largest gain against the original configuration was on `Screen1_4` (1.06092 to 0.95675), while `Screen1_3` worsened (0.56650 to 0.66575). The overall result ends this candidate's qualification. [Paired energy scores, group means, spread and effective dimensions](neuroko_v12_particle_weight_energies.json)

### Isolating the generator and particle learning rates

The earlier generator-only and particle-only trials used noise 0.025 and the default v12 prior penalty. This comparison repeats a single-role search at noise 0.15, prior weight 0.1 and critic LR 0.021. The reference generator rate is 0.014 and the reference particle rate is 0.0406. Four alternatives bracket each rate by approximately a factor of √2. Two use generator rates 0.010 and 0.020 with particles fixed at 0.0406. Two use particle rates 0.0287 and 0.0574 with the generator fixed at 0.014.

Rank the four alternatives by mean four-target energy across nine training pairs on seeds 22–24. Take only the lowest, breaking ties by trial name. If it fails to improve the original v12 mean, stop. Otherwise the same candidate must beat the matched v11 and EMPF arms and all established controls on each development cohort (22–24 and 251–253), then on six fresh seeds (290–295) including both original releases. A qualifying candidate goes to the biological holdout (396–411). The same two GPUs, network, 64 × 16 particle table and 1,200 constant-rate updates apply throughout.

```bash
python paper_watch/scripts/neuroko_rate_noise.py --study individual-role-rates --data "$data" --window-cells 10000 \
  --wheels "$data/wheels" --output "$data/v12-individual-role-rates" --shared-screen "$data/v12-rate-noise" \
  --second-screen "$data/v12-noise-prior"
```

The noise-0.25 controls finished while the individual-rate trials were running. They had lower first-cohort energies than the earlier v11 and EMPF references. An additional comparison fixes these two controls and measures them on seeds 251–253. It reports each of the four rate trials against them on seeds 22–24 once the shortlist exists. If the selected trial has completed the second cohort under its original protocol, it reports the same paired comparison there. Run the command again after the shortlist and any second-cohort fits finish to include those results.

```bash
python paper_watch/scripts/neuroko_rate_noise.py --stage compare-noise-controls --study individual-role-rates \
  --data "$data" --window-cells 10000 --wheels "$data/wheels" --output "$data/v12-individual-role-rates" \
  --shared-screen "$data/v12-rate-noise" --second-screen "$data/v12-noise-prior"
```
