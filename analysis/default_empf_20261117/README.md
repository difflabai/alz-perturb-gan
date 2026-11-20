# Owner-selected EMPF application default

The owner selected the released native EMPF preset after the completed ParticleGAN 0.10.0 comparison and requested that it become the default and be pushed. The application config is [`configs/microglia_default.json`](../../configs/microglia_default.json), and the public entrypoint is [`scripts/microglia.py`](../../scripts/microglia.py).

The selected model is the complete 20,000-update fold0/seed0 fit at `runs/empf_comparison_20261117/fits/native/pbulk/hvg2000/fold0/seed0`. It trained on 25 of the 37 development donors; 12 donors were held out. This is the retained pilot model, with no full-cohort refit. Its eight weight files, metadata and every-step regularization audit are bound in [`selected_model.json`](selected_model.json). All 20,000 updates applied the regularizer (`lazy_k=1`).

This is an **owner-selected experimental default**. Native EMPF improves profile RMSE and both interval scores against the previous default in one fixed fold/seed. Gene errors and coverage regress, empirical resampling still wins both interval scores, and prospective population/default qualification did not pass. The [selection record](selection.json) preserves these distinctions; no old score, grade or confirmation result was changed.

The final public CLI generation reproduced the scored native pilot archive byte for byte. All 46 automated tests passed. The [verification record](verification.json), [generation receipt](application_generation.receipt.json) and [independent application review](independent_application_review.json) bind the selected model, cache payloads and current runtime sources.

## Use the completed model

In the recorded research environment with development caches available:

```bash
PY=/mnt/nvme/ai/conda/anaconda3/bin/python3
$PY -B scripts/microglia.py config
$PY -B scripts/microglia.py generate --dry-run
$PY -B scripts/microglia.py generate
```

The default generation command selects the retained native EMPF model and writes to `runs/microglia_empf/generated/selected/fold0/seed0/`. Newly trained fits generate into `runs/microglia_empf/generated/fold{f}/seed{s}/`, so a fold0/seed0 retrain never shares an output with the retained model. `--fit-source trained` generates from the newly trained fit at the selected fold/seed, and `--fit` explicitly selects another compatible, complete native EMPF fit, with matching `--fold` and `--seed`. Dry runs and real runs print the chosen fit and output. An existing output or receipt is refused before any work starts; `--force` replaces both, and only after the new output passes its checks: the receipt and then the NPZ are each replaced atomically, so the previous pair stays in place until then. The NPZ appears only after its receipt is written. If a run is interrupted between those two steps, a receipt without an NPZ remains; a rerun refuses it, and `--force` recovers. Hidden staging files (`.<stem>.pending.npz` and the generator's sidecars) are removed whether the run succeeds or fails. Supply `--out /path/outside/the/fit/generated.npz` for another output. Frozen cache payloads, recipe/version/regularizer audits and all checkpoint hashes are checked before inference. Receipts bind the starting source manifest, and changes during execution are rejected. The separately tested RHBI-matched EMPF variant is rejected.

The six late states, `cellshape@0.5:biascal:stagehalf` readout, training-only calibration, 32 generated donors per target, 256 cells per donor and named generation streams match the completed native EMPF pilot. `--profile count-shape` explicitly selects that same readout. `--profile profile-rmse` selects the bias-only alternative on EMPF; old BISPF results do not qualify this alternative.

## Train or resume

```bash
CUDA_VISIBLE_DEVICES=1 $PY -B scripts/microglia.py train --fold 0 --seed 0 --device cuda --dry-run
CUDA_VISIBLE_DEVICES=1 $PY -B scripts/microglia.py train --fold 0 --seed 0 --device cuda
```

Training uses fresh weights or its own compatible resume state under `runs/microglia_empf/fits/`, then generates outside the fit directory. A fresh fit records the source-manifest digest it started under in `train_start.json`; resuming under a different digest is refused, and an existing `no_skip_audit.json` is never relabelled with a different digest. `train` on a fit that is already complete and audited skips straight to generation. It preserves the selected pilot. A completed newly trained fit can be selected explicitly with `--fit`; the committed default model pointer continues to name the retained pilot. CPU threads/affinity are limited to three and GPU allocation to 4%. On a shared host each process claims a free block of three CPUs (preferring idle ones) under a machine-wide lock in `/tmp/microglia-cpu-slots-<uid>`; an inherited mask of three or fewer CPUs is kept, and `MICROGLIA_CPU_SLOT=k` selects the k-th block (an out-of-range k is an error, not wrapped). The slot locks are per user and per `/tmp`; jobs of other users, or with an isolated `/tmp`, do not take part and are avoided only through the measured CPU load. Receipts record the CPUs used. Host activation is serialized with a file lock, so processes may start together. Set `CUDA_VISIBLE_DEVICES` explicitly when using CUDA. No GPU scheduler admission is required.

## Runtime setup

The root `pyproject.toml` installs the older synthetic smoke experiment. It is not the environment recipe for human microglia. Use the recorded dependencies in [`runtime_environment.json`](../improve_20261109/runtime_environment.json), and the cache/model dependencies described in the [original study protocol](../improve_20261109/PROTOCOL.md). On a fresh environment, install the exact native release into its isolated target:

```bash
$PY -m pip install --no-deps --no-compile \
  --target artifacts/native-releases/particlegan-0.10.0/site-packages \
  'particlegan @ git+https://github.com/HyperGAN/ParticleGAN.git@952c3b62ba434f7da194067923351ff5748ef8dd'
```

This command is for an empty target on a fresh setup; this workspace already contains the verified release. The loader checks the imported version/origin and pinned native source hashes. Research caches and checkpoints remain local artifacts, as in the repository's existing contract. Git contains the config, source, metadata hashes and evidence, rather than patient cells or binary weights. `config` and `--dry-run` remain available without Torch, data or checkpoints.

### Model sources in the repository

The generator, critic and data helpers that the runtime imports used to be read from two absolute directories under `/ml2/codex-storage`. Six of them are now [vendored](vendored/) byte for byte: `pg_model.py`, `critic.py`, `usage.py`, `contracts.py`, `stage_law.py` and `bench_core.py`. [`vendored/SOURCES.json`](vendored/SOURCES.json) records each original path, size, modification time and sha256. The first three keep the digests they were pinned with. The last three were imported at runtime but had never been pinned, and they are now added to the manifest. An import trace of a CPU generation showed only these six files being loaded from `/ml2/codex-storage`. None of the six carries a license header; they are assumed to be covered by the repository license, which the owner should confirm.

[`vendored_host.py`](vendored_host.py) derives the native host the same way as the concluded study's [`host.py`](../empf_comparison_20261117/host.py), which stays unchanged because its study manifest pins it. It reuses that host's rewrites and adds five location rewrites, so that every location the derived files name is in the current checkout: `MODEL`, `SCIENCE`, `BENCH` and `RUNS` in the trainer, `SITE` in `release_env.py`, and `B` in a derived `contracts.py` (the released model adapter puts `contracts.B` first on `sys.path`). The derived files go to `artifacts/native-releases/particlegan-0.10.0/alz-host-empf-native-vendored/`, beside the earlier `alz-host-empf-native/`, which is no longer written. A derived file that already exists and differs is an error and is never overwritten. The model code does not change; only where it is imported from.

### Local artifacts the runtime still needs

These are gitignored and are not fetched by any script:

| Path | Needed for | Checked by |
| --- | --- | --- |
| `artifacts/native-releases/particlegan-0.10.0/site-packages/` | every run (imported) | release pin and runtime manifest |
| `artifacts/native-releases/particlegan-0.9.7/site-packages/` | the manifest check only (pinned, not imported) | runtime manifest |
| `runs/improve_20261109/caches/hvg2000/fold{f}/` | `train` and `generate` | `analysis/empirical_challenge_20261114/cache_and_fit_anchors.json` |
| `runs/empf_comparison_20261117/fits/native/pbulk/hvg2000/fold0/seed0/` | `generate` from the selected model | [`selected_model.json`](selected_model.json) and the fit receipts |

`artifacts/` must be a real directory that belongs to this checkout. Do not share it between checkouts or reach it through a symlink: the derived host files name this checkout's absolute paths, so a second checkout writing into a shared `artifacts/` is refused with "existing derived EMPF host differs", and `release_env` compares the resolved `particlegan` path with the unresolved `SITE` path, so a symlinked `artifacts/` fails its import-origin check.

The derived host directory is created on first use. In a checkout without these artifacts the manifest check reports the native releases as missing, and `train`/`generate` stop before any work starts.

## Historical source preservation

The previous 0.9.7 config remains unchanged at [`analysis/improve_20261109/default_config.json`](../improve_20261109/default_config.json). The previous public CLI is archived byte for byte in [`previous/microglia.py`](previous/microglia.py), with the old config also copied alongside it. Original 326/330/345-source manifests, training/evaluation code and receipts are unchanged.

Those historical manifests included a nonexecuted guard requiring the public CLI/default to stay unchanged during their experiments. That guard now correctly rejects the later owner-selected application CLI. For an exact historical rerun use source commit `77b2d7d`. The current runtime's [source manifest](runtime_source_manifest.json) checks the unchanged model/native sources, the archived old CLI and the new application config/entrypoints. It does not silently rewrite old pins or claim that new selection passed old promotion gates.

## Re-pinning the runtime sources

After an intended change to a pinned source, re-pin with `scripts/refreeze_runtime_manifest.py --reason '...'`. The tool keeps the pinned key set (missing files are an error, never dropped), archives the previous manifest under `previous/`, and records the reason and changed keys. `--check` verifies the checkout without writing. In a worktree without the gitignored native releases, `--artifact-root /path/to/main/checkout` reads those files from there; they must keep their pinned digests.
