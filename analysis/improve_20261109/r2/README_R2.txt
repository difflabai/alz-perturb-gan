improve_20261109 ROUND 2 - how to run it
========================================

Plain-text operating notes for the round-2 code in this folder.  The rules
are in PROTOCOL_R2.md (binding) and ../PROTOCOL.md.  Design documentation is
in each module's docstring (arms.py = plug-in hook API).

Conventions used below
  PY   = /mnt/nvme/ai/conda/anaconda3/bin/python3 -B
  R2   = /ml2/alz/analysis/improve_20261109/r2
  RUNS = /ml2/alz/runs/improve_20261109
  Every CPU command: export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3
  and run from /tmp (cd /tmp) so no stray module shadows the r2 ones.
  Round-1 trees (RUNS/fits, results, results_screen48, screen_analysis,
  caches/*/fold{0,1,2}) are read-only; every r2 tool refuses to write there.

READ FIRST - repair notes (2026-11-10, after two adversarial reviews)
  * SELECTION HISTORY (review S3).  Several round-2 choices were informed by
    round-1 HELD-OUT results on the same 37 development donors:
      (a) condmix@1 on every screen arm (best generation-only held-out value;
          the train-chosen lam was 0.65-0.9);
      (b) condmix preferred over condshrink (it beat condshrink on held-out genemean);
      (c) genemean as the partner of both combinations and of the extra-fold jobs;
      (d) pbulk motivated partly by held-out pseudobulk-error quartiles;
      (e) donorpc2's share kept below the 0.61 that hurt; covdrop, covbin,
          stageanchor and stagedrop exist because of held-out failures.
    Folds 3-8 re-split the same donors, each of which was held out in round 1.
    So every round-2 held-out number is development evidence CONDITIONAL on
    round-1 selection; a confirmed fix needs donors used in neither round
    (sealed / external).  condmix@1 and condmix@0.5 are both pre-specified
    candidates on every arm and both are reported.
  * STAGE-BLIND modes (FID-3/S4).  condmix@1 (also condunk@1, condshrink@1,
    anchorgain@0) gives every target the same stage mixture: a generator-made
    grand-mean predictor.  Its pb_rmse is a CALIBRATION result (compare with
    baseline:grand_mean and posthoc direct:train_mean), never an F3 fix.  The
    evaluator labels these units [STAGE-BLIND] and blanks their stage metrics;
    judge stage fixes with condmix@0.5 and the trained arms (stageanchor,
    stagedrop@0.5, genemean,stageanchor).
  * DAM score (S1/S2).  dam_signed_perm_z (stage-label permutation null) is the
    stage test; the matched-random-gene dam_signed_z is miscalibrated (p<0.05 in
    ~20% of no-signal splits) and descriptive only.  hvg2000 has 3-5 listed
    genes, all up-genes: every dam_* is descriptive on hvg2000; use xfit_*.
  * Common random numbers (FID-1).  CPU and CUDA generations use different
    random draws.  Round-1 comparator modes are now generated with --device
    cuda (bitwise CRN with round-1 GPU defaults: condmix@0 on CUDA reproduces
    base fold0 seed0 generated.npz on all arrays) into RUNS/posthoc_r2/modes_gpu.
    The CPU files in posthoc_r2/cov and posthoc_r2/stage are NOT CRN-paired;
    do not pass them to the evaluator next to modes_gpu (duplicate units).
    The evaluator marks cross-device pairs 'xdev' (crn=False).
  * Repeats share donors (S7): paired sign counts are shown per fold AND per
    repeat ('b/n f; r/R rep'); fold counts over 0-8 are not 9 replications.
  * Posthoc verdict (S5/S6) is now like-for-like: a calibrated generator is
    compared with min(baseline:grand_mean with the same method,
    direct:train_mean); the old raw-reference booleans are renamed
    *_raw_round1_grand_mean.  The seed curve shows calibrated generators get
    worse as seeds are added.
  * trainer.py / run_screen.sh refuse an output root inside the frozen round-1
    trees (FID-2).  anchorgain@v is final-state only (FID-4).

Files
  trainer.py arms.py generate.py run_screen.sh    core (round-1 copy + plug-in hooks, snapshots, modes)
  arm_covdrop.py arm_covbin.py mode_margcov.py    F1  covdrop@p, covbin; modes margcov, owncov
  arm_stageanchor.py arm_stagedrop.py             F2/F3 stageanchor, stagedrop@p; modes anchorgain@v, condunk@lam
  arm_condshrink.py mode_condshrink.py            F3  modes condshrink@lam, condmix@lam (+ posthoc analysis)
  arm_donorpc2.py arm_pbulk.py                    F4/F5 donorpc2[@share], pbulk[@s]
  arm_gm.py gm_diagnose.py                        F6  gmlite@k (genemean@s is in the core); diagnosis
  posthoc.py                                      F5/F7 post-hoc calibration (recal, stage shrink, seed ensembles)
  evaluate.py metrics.py baselines.py             F8  round-2 evaluator (r2 copies of round 1 + new metrics)
  folds_r2.json make_folds_r2.py build_cache_r2.py F9 inner folds 3-8 + caches
  arm_example.py                                  hook API example only (flag 'example', mode exshrink@v)
  test_*.py                                       tests (see the end of this file)

Arm strings: comma-joined flags, any order on the command line (the fit
folder uses Python-sorted order), values in canonical %g text: covdrop@0.5,
stagedrop@0.5, genemean@0.5, genemean@0.25, gmlite@64, donorpc2@0.1, pbulk@0.5.
Bare 'genemean', 'donorpc2', 'pbulk', 'covbin', 'stageanchor' take their
defaults.  'genemean@1', 'donorpc2@0.25', 'pbulk@1', 'gmlite' (no value),
'covdrop' (no value) and non-canonical text (0.50) are refused on purpose.
Conflicts are refused: covdrop/covbin vs covars; stageanchor vs offset and
stagedrop; donorpc2 vs donorpc and genemean; gmlite vs genemean and donorpc.

Generation modes (run with generate.py --mode, or in the 5th job-file column):
  snapens            32 donors split 7,7,6,6,6 over snap_16000..19000 + final (r2 fits only)
  snap@U             all donors from snap_U (r2 fits only)
  margcov / owncov   covariate arms (covars, covdrop, covbin): same-stage train donors' covariates / own covariates
  condmix@lam        last floor(32*lam+.5) donors get stages in train stage proportions (any arm)
  condshrink@lam     c3 <- (1-lam)c3 + lam*mean (not on offset fits)
  condunk@lam        stagedrop fits only: trained 'unknown stage' condition
  anchorgain@v       stageanchor fits only: stage gains times v
Each mode writes generated_{mode}.npz (+ .receipt.json) next to generated.npz
(r2 fits) or to --out (round-1 fits; --out is REQUIRED and must lie outside
RUNS/fits).

-----------------------------------------------------------------------------
1. Launch the round-2 screen (72 fits, hvg2000, folds 0-2 x seeds 0-1)
-----------------------------------------------------------------------------
Job file: RUNS/jobs_r2.txt (seed-major, then fold).  Start after the round-1
confirmation run has finished (it holds both GPUs until about 19:00 UTC).

  mkdir -p /ml2/alz/runs/improve_20261109/logs_r2
  cd /ml2/alz/analysis/improve_20261109
  nohup r2/run_screen.sh -d cuda:0,cuda:1 -p 12 -j /ml2/alz/runs/improve_20261109/jobs_r2.txt \
        -t 2 -g 0.06 > /ml2/alz/runs/improve_20261109/logs_r2/screen_r2.out 2>&1 &

  dry run first:   r2/run_screen.sh -n -d cuda:0,cuda:1 -p 12 -j /ml2/alz/runs/improve_20261109/jobs_r2.txt -t 2 -g 0.06
  per-fit logs:    RUNS/logs_r2/{arm with + for ,}_{panel}_fold{k}_seed{s}.log
  fits:            RUNS/fits_r2/{arm}/hvg2000/fold{k}/seed{s}/
  stop:            kill -TERM <run_screen pid>   (fits checkpoint at the next update and exit)
  resume:          run the same command again: complete fits are skipped, interrupted fits
                   resume from recovery.pt (bitwise, tested on GPU), missing modes are run.
-t 2 keeps 12 x 2 = 24 threads on the 24-core machine (the round-2 cap is 3).
Budget (measured 2026-11-10 with 14 fits sharing the GPUs): 0.45-0.54 s per
update -> about 2.5-3 h per 20000-update fit; 72 fits at 12 concurrent = 6
waves = about 15-18 h.  Modes add about 10-15 s each on GPU (258 modes after
the repair added condmix@0.5 to every arm).
Disk: about 8.5 GB in total (generated.npz 23-25 MB, each mode file 23-25 MB,
final.pt 2.8-4.9 MB, 4 snapshots 5.1-5.7 MB per fit; recovery.pt is removed
when a fit completes).

Later (F9, after the screen is evaluated):
  r2/run_screen.sh -d cuda:0,cuda:1 -p 12 -j /ml2/alz/runs/improve_20261109/jobs_r2_extra_folds.txt -t 2 -g 0.06
  (base and genemean,stageanchor on inner folds 3-8, seeds 0-1, 24 fits, 84 modes; swap
  the second arm if the screen picks another winner - that pick is then itself selected
  on folds 0-2 held-out results of the same donors.)

-----------------------------------------------------------------------------
2. Generation modes on round-1 fits (no retraining; CPU or GPU)
-----------------------------------------------------------------------------
Round-1 fits have no snapshots, so only final-state modes apply.  --out is
required.  GPU-trained round-1 genemean/donorpc fits now load on CPU: float
arm-state values are compared with rtol 1e-5 (see generate._close); accepted
differences are listed in the receipt under arm_state_tolerance.

Stage comparators and covariate marginalisation for the screen, ON CUDA (common
random numbers with the round-1 GPU defaults and with the r2 modes, which
run_screen.sh generates on CUDA).  DONE by the repair task (2026-11-10, 36 files,
RUNS/posthoc_r2/modes_gpu/{arm}/hvg2000_fold{k}_seed{s}_{mode}.npz); the exact
command (rerun only to add files; existing outputs are refused without --force):
  cd /tmp; export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3
  for k in 0 1 2; do for s in 0 1; do
    for am in base:condmix@1 base:condmix@0.5 genemean:condmix@1 genemean:condmix@0.5 \
              covars:margcov covars,genemean,offset:margcov; do arm=${am%%:*}; m=${am#*:}
      CUDA_VISIBLE_DEVICES=1 /mnt/nvme/ai/conda/anaconda3/bin/python3 -B -c "
import sys, runpy, torch; torch.cuda.set_per_process_memory_fraction(0.06)
sys.argv = ['generate.py', '--fit', '/ml2/alz/runs/improve_20261109/fits/$arm/hvg2000/fold$k/seed$s',
            '--cache', '/ml2/alz/runs/improve_20261109/caches/hvg2000/fold$k', '--mode', '$m', '--device', 'cuda',
            '--out', '/ml2/alz/runs/improve_20261109/posthoc_r2/modes_gpu/$arm/hvg2000_fold${k}_seed${s}_$m.npz']
runpy.run_path('/ml2/alz/analysis/improve_20261109/r2/generate.py', run_name='__main__')"
  done; done; done
  (about 5 s per file on GPU; generate.py has no --gpu-mem-fraction flag, hence the wrapper.)
The older CPU files (posthoc_r2/cov/*, made by the cov task) are kept for the
record but are not CRN-paired with the GPU defaults.

-----------------------------------------------------------------------------
3. Evaluate (r2/evaluate.py)
-----------------------------------------------------------------------------
  cd /tmp; export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3
  /mnt/nvme/ai/conda/anaconda3/bin/python3 -B /ml2/alz/analysis/improve_20261109/r2/evaluate.py \
     --fits-root /ml2/alz/runs/improve_20261109/fits \
     --fits-root r2=/ml2/alz/runs/improve_20261109/fits_r2 \
     --modes-root /ml2/alz/runs/improve_20261109/posthoc_r2/modes_gpu \
     --out /ml2/alz/runs/improve_20261109/results_r2 --panels hvg2000 --folds 0,1,2
(--modes-root is posthoc_r2/modes_gpu ONLY: the whole posthoc_r2 tree also holds the
CPU-made cov files, which would create duplicate, non-CRN 'covars#margcov' units.)
The 'r2=' tag (integration addition) names r2 units 'arm~r2' and
'arm~r2#mode'.  It is needed because the screen refits 'genemean' (with
snapshots): untagged, the round-1 fits/genemean seeds 0-1 would shadow it and
genemean#snapens would be paired vs_parent with a different training run.
vs_base still pairs every unit with round-1 'base'.
Units: arm (default generated.npz), arm#mode (each mode file),
'arm [seed-ensemble n=N]' and '... M=32', baseline:name.  Outputs:
results_r2/summary.txt (markdown text), aggregate.json/csv, per-unit JSON.
Add --seeds 0,1 for like-for-like seed sets against round-1 base/genemean
(which have seeds 0-5).  Baselines take 6-13 min per (panel, fold) the first
time and are cached in --out.  Parallel option: separate processes per
--folds with the same --out, then one final single-process run to write
aggregate.json.  Primary columns: pb_rmse, pb_rmse_recal,
meanlog_rmse_gene(_recal), nonzero_rmse_gene, logvar_rmse_gene,
pc_energy_rep, cell_energy_rep, xfit_score(_z), dam_signed_perm_z; dam_* only
descriptive on hvg2000 (3-5 listed up-genes present, no down gene), so on hvg2000
xfit_* is the stage metric.  [STAGE-BLIND] units have no stage metrics.  Paired
cells: 'diff (b/n f; r/R rep)', 'xdev' = no common random numbers.

-----------------------------------------------------------------------------
4. Post-hoc calibration (r2/posthoc.py)
-----------------------------------------------------------------------------
  cd /tmp; export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3
  GEN=""; for f in /ml2/alz/runs/improve_20261109/posthoc_r2/modes_gpu/*/*.npz; do
    [ -f "${f%.npz}.receipt.json" ] && GEN="$GEN --gen-file $f"; done
  mkdir -p /ml2/alz/runs/improve_20261109/posthoc_r2/calib_r2
  cp -rn /ml2/alz/runs/improve_20261109/posthoc_r2/calib_repair/baseline_preds /ml2/alz/runs/improve_20261109/posthoc_r2/calib_r2/
  /mnt/nvme/ai/conda/anaconda3/bin/python3 -B /ml2/alz/analysis/improve_20261109/r2/posthoc.py run \
     --fits-root /ml2/alz/runs/improve_20261109/fits --fits-root r2=/ml2/alz/runs/improve_20261109/fits_r2 \
     $GEN --panels hvg2000 --folds 0,1,2 --out /ml2/alz/runs/improve_20261109/posthoc_r2/calib_r2
Round-1-only calibration after the repair (seeds 0-5 for base/genemean, GPU mode
files): RUNS/posthoc_r2/calib_repair (results.json, rows.csv, summary.txt).
'posthoc.py reverdict --out DIR' recomputes rows/verdict/summary from results.json.
Read the verdict's *_like_for_like keys; *_raw_round1_grand_mean is the round-1 rule.
To reuse the regenerated-baseline cache, first copy
RUNS/posthoc_r2/calib/baseline_preds into the new --out (3 MB per panel/fold).
For folds 3-8 build baselines first (one process per fold, about 6 min each):
  ... posthoc.py baselines --panel hvg2000 --fold 3 --out <same --out>
Keep the round-1 root UNTAGGED: the pre-specified primary unit is 'genemean'
and the results_screen48 verification only matches untagged round-1 units;
r2 units are then named 'r2:{arm}' (posthoc uses ':', the evaluator '~').
Always read calibrated rows next to direct:train_mean (a calibration that
shrinks the stage term to 0 becomes the train mean).

-----------------------------------------------------------------------------
5. Diagnostics
-----------------------------------------------------------------------------
  gm diagnosis on r2 fits:
  ... gm_diagnose.py all --roots /ml2/alz/runs/improve_20261109/fits_r2 --panels hvg2000 --states final,snap \
      --out-dir /ml2/alz/runs/improve_20261109/results_r2/gm_diag
  stage-inflation measurement: ... mode_condshrink.py measure | gen | report (docstring)

-----------------------------------------------------------------------------
6. Tests (CPU only, 3 threads, from /tmp)
-----------------------------------------------------------------------------
  test_core.py {parity|example|cli|screen|round1|cleanup}    parity keeps base/genemean bitwise = round 1
  test_arm_cov.py {unit|cli|cleanup}       test_arm_stage.py {unit|cli|round1|cleanup}
  test_arm_critic.py {unit|shares|separability|fits|cleanup}
  test_arm_gm.py {flags|parity|gmlite|cli|cleanup}
  test_posthoc.py {unit|e2e}    test_splits.py all    test_evaluate_r2.py {units|synthetic|reproduce OUT}
All passed on 2026-11-10 after the integration edits (see generate.py docstring
of _close) and again after the repair edits (test_evaluate_r2 units 23/23 with the
new DAM-permutation / stage-blind / device checks; synthetic includes a stage-blind,
cross-device mode unit; test_posthoc e2e checks the like-for-like verdict, seed curve
and reverdict).  cli/fits tests take 10-30 min each on the loaded machine.
Repair smoke (2026-11-10): base and stageanchor, 300 updates on GPU through
run_screen.sh into RUNS/smoke_r2/fits_repair, modes condmix@1, condmix@0.5,
(anchorgain@0.5), snapens; generated.npz bitwise equal to the integration smoke.
Evaluator / posthoc smoke outputs: RUNS/smoke_r2/repair/{results,posthoc}.

Integration smoke (2026-11-10): every arm and proposed combination ran 300
updates on GPU through run_screen.sh (RUNS/smoke_r2/integ/smoke_jobs.txt),
with every registered mode; numbers at 300 updates are plumbing checks only.
