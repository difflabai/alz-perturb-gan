# ParticleGAN formulation comparison at 1,200 updates

The controlled benchmark keeps the generator at 21 → 96 → 96 → 30, the critic at 35 → 96 → 96 → 1, the learned particle table at 64 × 16, the per-condition batch at 64 and the latent noise standard deviation at 0.025. Both recipes run from the same 0.11.1 wheel. The benchmark hashes the starting generator, critic and prior parameters, the projected data and the complete sequence of training minibatches. In the shared-rate comparisons, both arms started with identical hashes and optimizer-group rates. Targets and evaluation draws follow the [original NeuroKO study](neuroko_cuda_tuning_1200.md).

## Shared settings

Across 16 predeclared learning-rate and regularization configurations at validation seed 0, EMPF won 15 and Unified v11 won one. In that one configuration the critic rate was 0.042 against a generator rate of 0.005. EMPF scored 1.825, substantially above its best score of 1.097. The strongest v11 score in the shared grid was 1.490. Lower four-target mean energy distance is better.

The first numeric v11 candidate to outperform the matched EMPF average across the three inner replicates changed two public settings, `rhbi_eps` from 0.1 to 1 and `rhbi_rho` from 3 to 12. Both recipes used generator, critic and particle rates of 0.005, 0.0075 and 0.0125. The candidate retained the released `feydy_discrete` self-coupling rule and all three v11 formulation components.

| Inner replicate | EMPF | V11 with both numeric changes |
|---|---:|---:|
| Screen1_1 | 1.153 | 1.183 |
| Screen1_3 | 0.657 | 0.583 |
| Screen1_4 | 1.065 | 1.089 |
| Mean | 0.958 | 0.952 |

This candidate was frozen before opening its outer scores. Held-out results on neuronal replicate `Screen1_0` use the same hardware model, initial state and real batches within each seed.

| Configuration | Seed 5 | Seed 6 | Seed 7 | Mean | Generated / observed spread |
|---|---:|---:|---:|---:|---:|
| 0.11.1 EMPF | 1.307 | 1.496 | 1.353 | 1.385 | 0.999 |
| 0.11.1 Unified v11 at the same rates | 1.818 | 1.819 | 1.819 | 1.818 | 0.623 |
| 0.11.1 Unified v11, two numeric changes | 1.436 | 1.370 | 1.421 | 1.409 | 0.945 |
| Historical 0.10.0 EMPF | 1.287 | 1.517 | 1.283 | 1.363 | 1.003 |

The numeric changes closed approximately 95% of the held-out deficit against same-wheel EMPF, leaving a 1.7% gap. The candidate won seed 6. EMPF also received the strongest shared validation rate setting, with rates 0.014, 0.021 and 0.0406. Its mean held-out energy was 1.398.

The next comparison repeated the two-setting candidate against same-wheel EMPF on all 12 prespecified training seeds from 8 through 19. Mean held-out energy was **1.403 for v11 and 1.396 for EMPF**. V11 won five seeds and one of four targets on average. The paired mean difference was +0.0069, with a two-sided 95% paired t interval of −0.0286 to +0.0424 over training seeds. The spread ratios were 0.945 and 0.992, respectively. All seeds evaluated the same held-out biological replicate. This interval measures variation across training runs within that replicate. The study did not prespecify a practical equivalence margin.

## Validation with optimizer momentum

The numeric candidate also ran with generator, critic and particle rates of 0.014, 0.021 and 0.0406. Raising v11’s first-moment coefficient from 0 to 0.9 gave the following comparisons. Its second-moment coefficient stayed at 0.999. Both same-wheel EMPF arms ran the same zero-momentum DualNorm optimizer. Its native control is `optimizer_momentum`. The released EMPF direction-blend geometry requires this value to be zero and ignores the nominal `betas[0]` override. The distinction was verified in the 0.11.1 optimizer factory and by exact deterministic replay. [Release optimizer factory](https://github.com/HyperGAN/ParticleGAN/blob/d5dec2fe005305082acf382f0939a4bdb51c7131/particlegan/optim/dualnorm.py) Both learning-rate floors were 1.0 throughout the confirmations, so the configured group rates stayed constant.

| Inner replicate | Unified v11, β₁ = 0.9 | EMPF, repeat A | EMPF, repeat B |
|---|---:|---:|---:|
| Screen1_1 | 1.138 | 1.178 | 1.097 |
| Screen1_3 | 0.546 | 0.626 | 0.588 |
| Screen1_4 | 1.095 | 1.023 | 1.050 |
| Mean | 0.926 | 0.942 | 0.912 |

The momentum candidate and all three EMPF controls were fixed before opening the eight new seed results. Separate validation trajectories sampled every 150 updates showed rebounds in both formulations. V11 finished at 1.138 and 1.151 on seeds 0 and 4 after scoring 1.158 and 1.251 at update 1,050. The seed-0 EMPF trajectory ended at 1.201 in the replay and 1.178 in the original fit, despite identical data and initialization hashes. The search included more formulation-specific trials for v11. Scores on the same validation groups therefore have selection bias.

### Eight fresh training seeds on the held-out replicate

| Arm | Mean energy | V11 seed wins against arm | Mean generated / observed spread |
|---|---:|---:|---:|
| Unified v11, two transport settings and β₁ = 0.9 | 1.415 | — | 0.941 |
| 0.11.1 EMPF, same rates, repeat A | 1.396 | 4/8 | 1.054 |
| 0.11.1 EMPF, same rates, repeat B | 1.345 | 2/8 | 1.059 |
| Historical 0.10.0 EMPF | 1.386 | 4/8 | 0.974 |

The paired mean differences for v11 against those three controls were +0.018, +0.070 and +0.029. The paired 95% t intervals across seeds were [−0.049, +0.085], [−0.011, +0.151] and [−0.070, +0.128], respectively. Averaging the two equivalent same-wheel EMPF repeats within each seed gave 1.371, with v11 0.044 higher. In both repeats EMPF averaged a lower score for each target. Median synchronized fit times on the shared cards were 196 seconds for v11, 259 and 207 seconds for the two same-wheel EMPF repeats and 203 seconds for historical EMPF. Concurrent workloads differed during the runs, limiting what these times say about isolated training speed.

Four additional frozen seeds (28–31) used deterministic CUDA operations in every arm. Momentum v11 averaged **1.4090**, the same-wheel EMPF reference averaged **1.4041** and historical 0.10.0 EMPF averaged **1.3806**. The two nominal same-wheel EMPF configurations produced identical target metrics on every seed. V11 won one of four seeds against same-wheel EMPF and two of four against historical EMPF. The paired difference against same-wheel EMPF was +0.0049, with a 95% t interval of −0.152 to +0.161.

### Within-group shape and held-out centroid shift

On a deterministic replay of seed 28, the mean distance between each target’s training centroid and held-out centroid was 2.848 in the shared 30-dimensional space. V11’s generated centroids were 0.441 from the training centroids, and EMPF’s were 0.541 away. Their held-out energy distances were 1.463 and 1.422. Recentring each set of generated samples on its observed held-out centroid reduced these scores to 0.149 and 0.169. Matching total observed variance as well gave 0.149 and 0.146. These transformations use evaluation data after training solely to diagnose the score. The primary benchmark always scores the original clean samples. The result shows how strongly this replicate’s centroid shift influences the endpoint.

Two independent default-mode EMPF reruns on validation seed 0 scored 1.2164 and 1.2286. Seeding the global PyTorch streams still produced 1.1881 and 1.1207. Requiring deterministic CUDA operations produced exactly 1.100383 in two untraced runs and one run with intermediate scoring. In both the validation and held-out energy diagnostics, the deterministic EMPF controls with nominal `betas[0]` equal to 0 and 0.9 produced identical target metrics.

## Source of the inward gradient

For 30-dimensional vectors, this released v11 loss computes a real–fake transport plan and a fake–fake plan. It computes the self plan first and then removes its diagonal from the repulsive term. Consequently, affinity assigned to a sample's own row can consume much of the self plan without contributing a repulsive gradient. The default v11 vector path applies its learned-critic term to unmatched mass. The historical EMPF recipe uses the non-saturating adversarial loss for this vector input.

The radial probe used disjoint 64-cell samples from the same training population, a constant critic and fixed noise draws. A positive derivative with respect to log spread means gradient descent shrinks the sample. At the correct spread, the default self plan put 94.7% of its active mass on the diagonal, and the derivative averaged +2.521 over 32 draws. The library's `hollow` option shifted that derivative to +0.043. On independent 30-dimensional Gaussian samples, the corresponding derivatives were +2.571 and +0.012.

A second probe mimicked finite particle reuse by drawing 64 rows with replacement from 64 prototypes and adding small jitter. With the `hollow` option, 56.7% of the self plan's mass still paired draws from the same prototype on the neuronal probe. Its radial derivative was +1.564. In an isolated counterfactual, excluding all same-prototype pairs moved it to +0.006. This corrects the isolated matched-spread gradient. Training with the full exclusion overshot the observed spread and did not establish a better GAN.

In the actual final 600 training updates at the matched historical rates, sampled batches averaged 40.6 distinct prior IDs. With the default self rule, 47.2% of the self plan's active mass paired draws from the same particle, including 27.6% on the diagonal. With `hollow`, 42.5% still paired draws from the same particle. Almost all the real–fake mass was active in both runs, leaving little unmatched mass for the default critic term. A read-only backward hook measured the L1 norm of the generator-loss gradient with respect to the actual fake critic logits. The late-update mean was 3.4 × 10⁻¹¹ for the released default and 0.062 for the two-setting candidate. Setting the released active-pair critic weight to 2 at the middle rate configuration raised the norm to 1.193. Each of the three instrumented runs reproduced its uninstrumented energy distance exactly. The measured gradient concerns the direct generator-loss dependence on critic logits. The endpoint scores include the rest of the dynamics.

## Reproduction

Use the [GPU environment and public H5AD instructions](../README.md#neuroko-gpu-comparison). From the repository root, set `data=paper_watch/data/neuroko`. The function below runs the confirmations. Seeds 8–19 compare the two-setting candidate to same-wheel EMPF. Seeds 20–27 run the momentum candidate, two repeats of same-wheel EMPF at the same rates and the historical 0.10.0 reference. Seeds 28–31 repeat the same four arms with deterministic CUDA operations. Each seed runs its arms on one physical card. The two groups of seeds run concurrently.

```bash
data=paper_watch/data/neuroko
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
fit1200() {
  local gpu=$1 seed=$2 version=$3 recipe=$4 trial=$5
  local -a kernel_options=()
  if ((seed >= 28)); then kernel_options=(--deterministic); fi
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$data/wheels/$version" \
    python -m paper_watch.alztest.neuroko_tune heldout --data "$data" \
      --window-cells 10000 --seed "$seed" --recipe "$recipe" --release "$version" \
      --trial "$trial" "${kernel_options[@]}" --output "$data/results1200/$recipe-$version"
}
card1200() (
  set -e
  gpu=$1
  for ((seed=8+gpu; seed<32; seed+=2)); do
    if ((seed < 20)); then
      fit1200 "$gpu" "$seed" 0.11.1 unified_v11 '{"name":"transport-1-12-legacy-rates","lr":0.005,"d_lr_mult":1.5,"prior_lr_mult":2.5,"rhbi_eps":1,"rhbi_rho":12}'
      fit1200 "$gpu" "$seed" 0.11.1 empf '{"name":"legacy-rates","lr":0.005,"d_lr_mult":1.5,"prior_lr_mult":2.5}'
    else
      fit1200 "$gpu" "$seed" 0.11.1 unified_v11 '{"name":"settle-b9-mid","lr":0.014,"d_lr_mult":1.5,"prior_lr_mult":2.9,"rhbi_eps":1,"rhbi_rho":12,"betas":[0.9,0.999]}'
      fit1200 "$gpu" "$seed" 0.11.1 empf '{"name":"control-b9-mid","lr":0.014,"d_lr_mult":1.5,"prior_lr_mult":2.9,"betas":[0.9,0.999]}'
      fit1200 "$gpu" "$seed" 0.11.1 empf '{"name":"mid","lr":0.014,"d_lr_mult":1.5,"prior_lr_mult":2.9}'
      fit1200 "$gpu" "$seed" 0.10.0 empf '{"name":"legacy-rates","lr":0.005,"d_lr_mult":1.5,"prior_lr_mult":2.5}'
    fi
  done
)
card1200 0 & first=$!
card1200 1 & second=$!
rc=0
wait "$first" || rc=1
wait "$second" || rc=1
test "$rc" -eq 0
```

Use `validation --validation-group Screen1_3` in place of `heldout` to reproduce a training-side comparison. Matched validation uses seed 0. To reproduce the default radial probe, run the following with the same CUDA environment.

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$data/wheels/0.11.1" \
  python -m paper_watch.alztest.neuroko_radial --data "$data" --window-cells 10000 \
    --eps 0.1 --out "$data/results1200/radial-default.json"
```

Run the radial probe with `--self-coupling hollow` to exclude the sample diagonal. Add `--repeat-particles` for repeated IDs and `--exclude-siblings` to test the explicit same-prototype mask. In a training run, `--self-plan observe` records the native plans every 25 updates. The other diagnostic modes change self-pair affinities on the instrumented trainer. All benchmark confirmations use the released implementation. [Paired scores, solver diagnostics and original fit receipts](neuroko_cuda_formulation_1200.json)
