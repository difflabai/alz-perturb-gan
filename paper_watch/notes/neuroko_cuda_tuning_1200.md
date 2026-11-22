# NeuroKO at 1,200 CUDA updates

The 0.11.1 Unified v11 configuration selected on validation scored **1.567** on the neuronal holdout. ParticleGAN 0.10.0 EMPF scored **1.390** with the original small network and **1.292** with the same capacity and batch size as the selected v11. Lower mean energy distance is better. EMPF won each of the three paired seeds and each of the four targets averaged over seeds. [Full trial and confirmation receipts](neuroko_cuda_tuning_1200.json)

| Formulation and configuration | Seed 1 | Seed 2 | Seed 3 | Mean | Generated / observed spread | Mean fit time, shared GPUs |
|---|---:|---:|---:|---:|---:|---:|
| EMPF, original | 1.456 | 1.411 | 1.302 | 1.390 | 0.953 | 306 s |
| Unified v11, selected | 1.533 | 1.581 | 1.587 | 1.567 | 0.761 | 215 s |
| EMPF, matched capacity and batch | 1.309 | 1.239 | 1.329 | 1.292 | 1.056 | 196 s |

All three arms received 1,200 updates on RTX A6000 GPUs. The original processed 76,800 sampled cell rows and each larger-batch arm processed 307,200. Individual synchronized fit times varied with GPU contention. The selected model used generator, critic and particle learning rates of 0.028, 0.042 and 0.0812. Both EMPF controls used 0.005, 0.0075 and 0.0125. The library supplied each formulation's loss, optimizer and prior regularizer.

The follow-up locks the original network and particle dimensions in both arms. [Controlled formulation comparison and diagnosis](neuroko_formulation_1200.md)

## What improved validation

The benchmark kept the original 30 training-whitened expression coordinates, condition rotation and four-target mean energy metric. The search fitted on neuronal replicates `Screen1_3` and `Screen1_4` and scored `Screen1_1`. It tested 33 v11 configurations at seed 0 and repeated the best three at seed 4. The lowest two-seed mean selected the final configuration before scoring it on `Screen1_0` at seeds 1, 2 and 3. Both inner and outer evaluations used 256 clean draws per condition.

| V11 change from original | Batch | Particles | Latent dimensions | Validation energy, seed 0 |
|---|---:|---:|---:|---:|
| Original | 64 | 64 | 16 | 1.513 |
| Batch size alone | 256 | 64 | 16 | 1.298 |
| Particle count and latent dimensions | 64 | 512 | 32 | 1.375 |
| All three | 256 | 512 | 32 | 1.144 |

The winning v11 averaged 1.173 on its two validation seeds versus 1.180 for the original EMPF. The generator input widened from 21 features to 37 after adding the five condition flags. This added 1,536 weights to its first layer. Both hidden layers retained 96 units each. The output had 30 coordinates and the critic was identical across sizes. The particle table grew from 64 × 16 to 512 × 32. On the outer holdout v11's spread averaged 0.761 times the observed within-target spread.

A separate replay logged validation energy every 150 updates using the same evaluation draws at each checkpoint. The final two-seed means were 1.206 at update 1050 and 1.173 at update 1200. Seed 4 reached 1.178 at update 300 and later rebounded. The traces record nonmonotonic convergence over this interval.

### Reproduce the 1,200-update comparison

Follow the [GPU setup](../README.md#neuroko-gpu-comparison) through downloading the public H5ADs and installing the two wheels. From the repository root, the following commands refit all three arms on the same held-out replicate. Each JSON receipt records the actual optimizer rates, number of updates, hardware, projected-data fingerprint and per-target metrics.

```bash
data=paper_watch/data/neuroko
run1200() {
  local release="$1" recipe="$2" trial="$3" seed
  for seed in 1 2 3; do
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$data/wheels/$release" OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
      python -m paper_watch.alztest.neuroko_tune heldout --recipe "$recipe" \
      --data "$data" --window-cells 10000 --seed "$seed" --trial "$trial" --output "$data/results1200" || return 1
  done
}
run1200 0.10.0 empf '{"name":"empf","lr":0.005}'
run1200 0.11.1 unified_v11 '{"batch_size":256,"lr":0.028,"name":"batch256-support","num_particles":512,"z_dim":32}'
run1200 0.10.0 empf '{"batch_size":256,"lr":0.005,"name":"empf-matched","num_particles":512,"z_dim":32}'
```

To replay a validation fit, use the same command with `validation`, ParticleGAN 0.11.1, and a trial from `configurations` in the results JSON. The search used seed 0, then repeated the three finalists at seed 4. When using the extracted pilot H5ADs, pass their directory to `--data` and omit `--window-cells`.
