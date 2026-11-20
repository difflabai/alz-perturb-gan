# Protocol addendum (post hoc, 2026-11-20)

`PROTOCOL.md` is left byte-identical so that `protocol_sha256` in
`results.json` still verifies against commit e300bb7. This file records
corrections and deviations found by an independent audit after the first
scoring run. Nothing here was declared before scores existed.

1. **Declaration time.** `PROTOCOL.md` says "Declared 2026-11-20 (UTC 03:55)".
   Its commit (e300bb7) is timestamped 03:53:09 UTC, so the stated time is
   about two minutes later than the commit itself. The commit time is the
   authoritative declaration time. The first scoring log line is at 04:10 UTC.
   The research question was motivated by already-known pilot numbers
   (G0 raw SD coverage 58.3%), as the protocol states.

2. **(b1) recipe deviation, repaired.** The protocol specifies "the frozen
   space recipe" for the leave-one-TRAIN-donor-out refits. The first run
   (aa99dda) drew the 20,000-cell SVD subsample *with* replacement and used a
   float64 log transform. The frozen `build_cache.fit_space` samples
   *without* replacement over the kept cells in cache order and uses the
   float32 `bench_core.log_hvg`. `run.py` now matches the frozen recipe
   (`replace=False`, float32 `log_hvg32`, kept cells in cache donor order) and
   records a full-TRAIN recipe check against the frozen `space.npz`. The b1
   factors, the b1 view and the per-PC alignment numbers were recomputed.
   All other views do not depend on this function and are unchanged.
   The first run's results are kept outside git at
   `runs/research_20261120/pca-shrinkage/results_v1_replace_true.json`.

3. **Exploratory comparisons (not declared).** All paired conformal
   comparisons are exploratory. The matched comparison for calibrated G0 is
   calibrated E_train (same global leave-one-donor-out procedure), not raw
   E_train. The comparison with the earlier 6/6 split is an unpaired point
   comparison without uncertainty.
