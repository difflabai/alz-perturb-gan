# improve_20261109: gene-level accuracy development study

Current status (2026-11-13): the source and experiment history are preserved
with a selected default in [`default_config.json`](default_config.json).
Use [`../../scripts/microglia.py`](../../scripts/microglia.py) for the current
training/generation entry point; its `config` and `--dry-run` commands require
no dataset or GPU. The historical commands below describe the original
round-1 harness. Later experiments are under `r2/` through `r5/`; see
[`RESULTS.md`](RESULTS.md) and [`TAKEOVER_20261113.txt`](TAKEOVER_20261113.txt).
The owner explicitly authorized committing this study and its default in
[`PROTOCOL_DEFAULT_PASS_20261113.txt`](PROTOCOL_DEFAULT_PASS_20261113.txt),
superseding the earlier no-commit restriction for this pass.

An exploratory development study of the stage-conditioned ParticleGAN 0.9.7
microglia model (released update rule, candidate t005). It tests whether
application-level changes ("arms") improve gene-level accuracy on held-out
donors, compared honestly against the `base` arm and against simple baselines.
It uses only the 37 development donors in `folds.json` and three inner folds.
Its numbers are not comparable with benchmark097 or any sealed/audited result,
and it makes no SOTA or held-out claim. The binding contract is
[`PROTOCOL.md`](PROTOCOL.md); read it first.

Hard rules (summary; PROTOCOL.md wins): only the 37 development donors; write
only under this directory and `/ml2/alz/runs/improve_20261109/`; no git
commits/stashes/checkouts; CPU at most 6 threads per process; GPU only when a
task explicitly authorises it; keep disk use small. Python:
`/mnt/nvme/ai/conda/anaconda3/bin/python3` (always run with `-B`; the scripts
also set `sys.dont_write_bytecode`, so nothing is written into the read-only
released source directories).

## Layout

Code and docs (this directory, `/ml2/alz/analysis/improve_20261109/`):

| file | role |
|---|---|
| `PROTOCOL.md`, `folds.json` | contract: hard rules, interfaces, donors and inner folds |
| `build_cache.py`, `test_cache.py` | Interface 1: per-fold caches (gene panel, space, counts) |
| `trainer.py`, `arms.py`, `generate.py`, `test_trainer.py` | Interface 2: training (released update rule), arms, `generated.npz` |
| `evaluate.py`, `metrics.py`, `baselines.py`, `test_evaluate.py` | Interface 3: scoring, baselines, peer-variability reference (`noise_ceiling`), aggregate tables |
| `run_screen.sh` | launcher for a list of fits (devices, parallelism, logs, skip completed) |
| `NOTES_*.md` | per-component interpretation choices; `NOTES_integration.md` for cross-component decisions; `NOTES_fixer.md` for the post-review fixes |
| `DATA_EXPANSION.md`, `data_expansion/` | metadata-only inventory of further data sources |

Outputs (`/ml2/alz/runs/improve_20261109/`, untracked):

```
caches/{hvg2000,wide}/fold{0,1,2}/   manifest.json, space.npz, train/, heldout/   (~49 MB total)
fits/{arm}/{panel}/fold{k}/seed{s}/  config.json, learning_curve.jsonl, final.pt,
                                     fit_receipt.json, generated.npz (~16 MB hvg2000, ~40 MB wide)
results/                             evaluate.py output: fits/, ensembles/, baselines/, real/,
                                     aggregate.json, aggregate.csv, summary.md
logs/                                run_screen.sh logs, one per fit
smoke_fits_v2/, smoke_results_v2/    post-review smoke test (300 updates, fold 0; not study results)
smoke_logs_v2/                       its launcher/fit/test logs
```

Arms (`--arm`): `base`, or a comma-joined set of `offset`, `genemean`,
`donorpc`, `covars` (`genemean` and `donorpc` are mutually exclusive). The fit
directory uses the sorted flag list, e.g. `covars,genemean,offset`.
Panels: `hvg2000` (2,000 HVGs) and `wide` (5,000 HVGs plus forced AD/microglia
risk genes; G = 5028/5029/5028 by fold).

## Commands

All commands below assume:

```bash
cd /ml2/alz/analysis/improve_20261109
PY=/mnt/nvme/ai/conda/anaconda3/bin/python3
RUNS=/ml2/alz/runs/improve_20261109
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6
```

### 1. Build and check the caches (about 30 s, 2.9 GB RSS; deterministic)

```bash
CUDA_VISIBLE_DEVICES='' $PY -B build_cache.py              # all 2 panels x 3 folds; add --overwrite to rebuild
CUDA_VISIBLE_DEVICES='' $PY -B build_cache.py --panel hvg2000 --fold 0 --overwrite   # one cache
CUDA_VISIBLE_DEVICES='' $PY -B test_cache.py               # must print ALL CACHE TESTS PASSED
```

### 2. One fit

CPU (measured 2.7-3.4 s/update on hvg2000 and 6.7-8.3 s/update on wide at 6 threads on the
shared machine, so a 20,000-update fit takes about 15-19 h or 37-46 h on CPU):

```bash
CUDA_VISIBLE_DEVICES='' $PY -B trainer.py --cache $RUNS/caches/hvg2000/fold0 \
    --arm base --seed 0 --out-root $RUNS/fits --updates 20000 --device cpu --threads 6
```

CUDA (only when a task authorises GPU use; untested so far; a fit needs < 1 GB):

```bash
CUDA_VISIBLE_DEVICES=0 $PY -B trainer.py --cache $RUNS/caches/hvg2000/fold0 \
    --arm covars,genemean,offset --seed 0 --out-root $RUNS/fits --updates 20000 \
    --device cuda --threads 6 --gpu-mem-fraction 0.1
```

The trainer writes `recovery.pt` every 2000 updates and on SIGINT/SIGTERM, and
resumes automatically when rerun with the same arguments. It refuses to
overwrite a completed fit. When `final.pt` exists without `generated.npz` it
only generates, after checking `config.json` and that `final.pt` was trained to
the requested `--updates` (a different `--updates` is refused). It writes `generated.npz` itself after training;
`generate.py --fit FIT_DIR --cache CACHE_DIR [--force]` regenerates it standalone.

### 3. Many fits

```bash
cat > $RUNS/jobs_screen.txt <<'EOF'
# arm                    panel    fold seed
base                     hvg2000  0    0
covars,genemean,offset   hvg2000  0    0
offset                   wide     0    0
EOF
./run_screen.sh -d cpu -p 3 -j $RUNS/jobs_screen.txt               # CPU, 3 concurrent fits x 6 threads
./run_screen.sh -d cuda:0,cuda:1 -p 4 -j $RUNS/jobs_screen.txt -g 0.1   # GPU (authorised tasks only)
./run_screen.sh -d cpu -p 3 -j $RUNS/jobs_screen.txt -n             # dry run
```

Options: `-t THREADS` (default 6, max 6), `-u UPDATES` (default 20000), `-o OUT_ROOT`
(default `$RUNS/fits`), `-c CACHES_ROOT`, `-l LOG_DIR` (default `$RUNS/logs`).
Logs go to `$RUNS/logs/{arm with + for ,}_{panel}_fold{k}_seed{s}.log`.
Completed fits are skipped, interrupted fits resume, and the script exits
non-zero if any fit fails. SIGINT/SIGTERM to the script asks running fits to
checkpoint and stop.

### 4. Evaluate

```bash
$PY -B evaluate.py --fits-root $RUNS/fits --caches-root $RUNS/caches --out $RUNS/results
#   [--folds 0] [--panels hvg2000] [--no-baselines] [--no-ensembles]
```

First run with baselines on all 2 panels x 3 folds takes about 30 min; reruns
are incremental (only changed fits are rescored). Read `results/summary.md`
first, then `aggregate.csv` / `aggregate.json` (paired comparisons vs `base` and
vs the reference baselines `stage_mean`, `stage_mean_log`, `shrunk_stage_mean`
and `ridge`, pooled-across-folds contrasts, the peer-variability reference).
Row values are fold-balanced means with SE over fold means; paired sign counts
are per fold. Baseline metrics that `baselines.MEANINGFUL` marks "NOT
meaningful" (PC space and cell energy for the profile baselines) are blanked.
`stage_mean` (log of mean proportions) is a weak bar: compare with
`stage_mean_log`, `shrunk_stage_mean` and `ridge` as well. For cross-panel
claims use the `shared_<panel>_renorm` subset only; `shared_<panel>_withinpanel`
keeps each panel's CP10k denominator (only its nonzero metrics compare across
panels). `noise_ceiling` is a peer-variability reference, not a ceiling.
Primary metrics are on held-out inner-fold donors.

### 5. Tests

```bash
CUDA_VISIBLE_DEVICES='' $PY -B test_cache.py
CUDA_VISIBLE_DEVICES='' $PY -B test_evaluate.py                 # 34 synthetic planted-truth and unit checks
CUDA_VISIBLE_DEVICES='' $PY -B test_trainer.py devcache          # then: parity grads arms timing resume resume_offset cleanup
```

`test_trainer.py all` runs every trainer test except cleanup (about 70 min on the shared CPU;
`arms` is 50 min of it). `test_evaluate.py` takes 6-8 min.

## Status

See `NOTES_integration.md` for the integration smoke test (real caches, three
short fits, evaluator with baselines), the measured CPU cost and open issues.
See `NOTES_fixer.md` for the fixes made after the two independent reviews
(offset initialisation v2, deterministic offset arm, evaluator v2 / baselines
v5, metadata streaming) and the post-review smoke test in `smoke_fits_v2/` and
`smoke_results_v2/`. Old `smoke_fits/` and `smoke_results/` were removed.
The original smoke-test status above is historical. The completed screens,
confirmation runs and default-selection pass are recorded in `RESULTS.md`.
