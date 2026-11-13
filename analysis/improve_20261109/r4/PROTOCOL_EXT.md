# improve_20261109 external donors (round 4, data side)

Owner mandate (Mikkel, 2026-11-11): "keep trying different approaches until
you get top scores". The round-3 open-data inventory
(`../r3/open_data_inventory.json`) found GEO GSE268599 (MGH-AbbVie MADRC
cohort; Serrano-Pozo 2024, Wachter 2024): 32 donors with published per-donor
A, B, C scores, Braak, CERAD, APOE, sex and age; computed ADNC 8 Not AD / 5 Low
/ 6 Intermediate / 13 High; inferior temporal gyrus (ITG, next to MTG) for 31
donors (7 Not AD); 10x v3; same 36,601-gene GRCh38-2020-A axis as SEA-AD; open
access. The 37 SEA-AD development donors hold 4 Not AD donors, and no method
recovers held-out stage contrasts from them. This workstream tests whether
open external donors fix that.

Read `../PROTOCOL.md`, `../r2/PROTOCOL_R2.md`, `../r3/PROTOCOL_R3.md` and
`../RESULTS.md` first. All earlier hard rules bind unless changed here.

## Rules

- External counts allowed: GSE268599 (ITG libraries; other regions only for
  quality checks) and GSE264648 (microglia object, MTG, 42 MB; cohort-offset
  check only). Nothing else. Requests to outside services carry no personal
  details of the repository owner; use a generic client name.
- SEA-AD: still ONLY the 37 development donors. Sealed donors stay sealed.
- External donors are TRAINING donors only. Every score is computed on SEA-AD
  held-out inner-fold donors, with the same gene panel, space and metrics as
  `hvg2000` fold k, so results are directly comparable with rounds 1-3.
- Gene selection, scaling and PCA stay fitted on the SEA-AD train donors of
  the fold (copy `caches/hvg2000/fold{k}/space.npz`). Any cohort correction is
  fitted on train donors (SEA-AD train + external) only.
- `r3/` is FROZEN (the round-3 screen is training with it). Code goes in
  `/ml2/alz/analysis/improve_20261109/r4/`, starting as a copy of `r3/`.
  Outputs: `/ml2/alz/runs/improve_20261109/ext/` (downloads and derived
  external data), `caches/hvg2000x/fold{0,1,2}/`, `fits_r4/`, `results_r4/`,
  `smoke_r4/`, `logs_r4/`. Never write into another run folder.
- Disk is 99% full: at most 10 GB of transient downloads at any time, at most
  3 GB kept. Delete raw matrices once per-donor microglia counts are written
  and verified.
- At most 3 CPU threads per process. GPUs are shared with the round-3 screen:
  test fits of at most 300 updates and generation only, one process at a time
  per task, `--gpu-mem-fraction 0.04`. Never kill processes you did not start.
- No `.md` files from subagents; no git operations. Python:
  `/mnt/nvme/ai/conda/anaconda3/bin/python3 -B`.
- Honesty: cross-cohort differences (region, nuclei sorting, lab, depth) may
  be larger than stage effects. If external donors do not help, say so.

## Interfaces

- External donor store, same layout as the SEA-AD rebuilt store
  (`/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark/rebuilt/`):
  `ext/GSE268599/counts/{donor}.npz` (cells x 36,601 genes, CSR),
  `{donor}_names.npy`, and `ext/GSE268599/donors.json` with per donor: cohort,
  region, A, B, C, ADNC (NIA-AA 2012 table, computed and cross-checked), Braak,
  CERAD, sex, age (and how "90+" is coded), APOE4 dose, cells kept, median UMIs
  per cell, and the microglia selection receipt.
- Microglia definition: as close as possible to the SEA-AD "Microglia-PVM"
  supertype used for the 37 donors, documented and validated (marker genes,
  doublet and ambient checks, agreement with any author annotation).
- Caches `caches/hvg2000x/fold{k}/`: Interface 1 of `../PROTOCOL.md`; genes and
  `space.npz` identical to `hvg2000/fold{k}`; `heldout/` identical; `train/` =
  SEA-AD train donors followed by external donors; `raw_c` has the usual seven
  slots, and `metadata.npz` adds `cohort` (0 = SEA-AD MTG, 1 = MADRC ITG) per
  donor. The manifest lists both donor sets.
- Arm `cohort` (plug-in `r4/arm_cohort.py`): the generator and critic receive a
  cohort indicator; generation for SEA-AD targets uses cohort 0. Composable
  with `pbulk`, `genemean`, `nostage` and the `t.`/`g.`/`d.` flags.
- Evaluation: `r4/evaluate.py` scores `hvg2000x` fits and baselines on the same
  SEA-AD held-out donors as `hvg2000`, with baselines fitted with and without
  the external donors, and with and without a cohort correction.
