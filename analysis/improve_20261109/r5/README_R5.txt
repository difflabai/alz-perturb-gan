improve_20261109 round 5: README (plain text)
=============================================

TAKEOVER UPDATE 2026-11-13: Block A finished 21/21 fits, no failures, and was
scored by Codex after Claude reached its weekly limit. The pzero block fails:
snapens profile RMSE 0.6854 against same-kappa control 0.5211, with worse
mean-log and nonzero errors in all six pairs. Do not relaunch Block A. The
optional 12-fit Block B remains unstarted after this negative result.
See ../TAKEOVER_20261113.txt and ../RESULTS.md sections 8f/8h. Recovered
extra-fold PRIMARY profile confirmation also fails for both tested factors;
its full distribution analysis is still outstanding.

Binding protocol: PROTOCOL_R5.md (this folder). Owner decision 2026-11-12: "yes do 1 and 2".
Work item 1 = short training with averaging (no new training).
Work item 2 = the per-gene mean-log and nonzero-fraction gaps: diagnosis, Fix A (zcal), Fix B (pzero).
This file was written after the adversarial review of round 5 (findings F1-F7) and gives the CORRECTED numbers.
Every comparison is paired by fold and seed. "SE" means seed-noise SE. "fold t" is the t on the per-fold differences.

Python: /mnt/nvme/ai/conda/anaconda3/bin/python3 -B. Run every command below in BASH, from /tmp.
The login shell is zsh, and zsh does not word-split "$PY", so run the blocks with bash.
GPU commands need CUDA_VISIBLE_DEVICES=0 or 1 set explicitly, because the environment exports it as empty.


1. WORK ITEM 1: SHORT TRAINING WITH AVERAGING  (r5/shortavg.py, r5/test_shortavg.py)
------------------------------------------------------------------------------------
Outputs: runs/improve_20261109/smoke_r5/shortavg/{analysis.txt, analysis.json, scores/, kept/}

Selection (pre-declared: control pbulk, folds 0-2, seed 0, held-out pb_rmse):
  snapema@5000:0.9995 is chosen at 0.506 (folds 0.4965/0.4984/0.5238).
  References: snapens 0.543, ema@0.9995 0.550, final 0.645.
  This was recomputed independently by the reviewer and matches exactly.

Confirmation, chosen minus snapens (negative = better):
  control seed 1      -0.045, SE 0.026, z -1.74, 3/3 folds
  combination         -0.027 +- 0.004, 12/12 pairs
  folds 3-5           -0.035 +- 0.009, 5/6 pairs
  folds 6-8           -0.021 +- 0.015, z -1.4, 4/6 pairs
DECISION: snapens is NOT replaced. Control seed 1 and the folds 6-8 repeat do not clear 2 SE.

What the gain is: shrinkage toward the pooled-mean profile, not a better generator.
  - The EMA keeps 8.2% of the untrained initial weights (0.9995^5000).
  - The untrained generator alone scores pb_rmse 0.4985. That beats snapens by 0.043 (control, 6/6 pairs)
    and by 0.019 (combination, 11/12).
  - Removing the initial-weight share costs +0.013 (control, 0/6 pairs better) and +0.010 (combination, 0/12).
  - At the stage-blind and calibrated stage-blind read-outs the gain disappears.
    Calibrated stage-blind: chosen minus snapens +0.003 (control); ema@0.9995 minus snapens -0.004 (control, 6/6).

CORRECTION F4: the guard metrics also reward collapse.
  pc_energy_rep and cell_energy_rep favour collapse in the same way pb_rmse does.
  The untrained generator, compared with snapens (control, folds 0-2, seeds 0-1):
    pb_rmse           -0.043 (SE 0.012, 6/6)
    cell_energy_rep   -0.198 (SE 0.019, 6/6)
    pc_energy_rep     -0.020 (SE 0.016, 4/6), a tie
  For the combination, the untrained generator's pc_energy_rep is worse (+0.148, 2/12 better), but its
  cell_energy_rep is still better (-0.121, 12/12).
  So the chosen read-out's "better" pc_energy (2.10 against 2.33) and cell_energy (0.78 against 0.99) are NOT
  evidence of a better generator. They come with:
    pc_spread_rep      -0.59 (SE 0.06, 6/6)
    pc_coverage80_rep  -0.11 (0.50 against 0.61; nominal 0.80)
    spread ratio       0.53 against 0.99 for snapens (well-expressed genes)
  The spread ratio is the SD over a target's 32 generated donors divided by the real held-out between-donor SD
  at the same depth.
  analysis.txt now has a "Collapse check" section (init, chosen, final, ema@0.9995 and snapens next to spread
  and coverage). pzero_compare.py flags a guard "gain" that comes with a loss of spread and coverage
  (collapse_flag).
  Rule from now on: never read pb_rmse, pc_energy_rep or cell_energy_rep without pc_spread_rep,
  pc_coverage80_rep and the spread ratio next to them. Conversely, a WORSE pc or cell energy (for example
  control zcal, below) may partly reflect more spread rather than a worse fit.

CORRECTION F6: every spread number now names its gene subset.
  "all"  = all 2,000 panel genes.
  "well" = the 256 genes with the highest train grand mean.
  Control, folds 0-2, seeds 0-1; columns are init / chosen / final / snapens; the real value follows:
    SD over the 32 generated donors of a held-out target (K-cell log pseudobulk):
      well  0.195 / 0.321 / 0.486 / 0.599   real held-out 0.603, real train 0.624
      all   0.212 / 0.291 / 0.424 / 0.525   real held-out 0.434, real train 0.449
    Between-target SD of the predicted profile:
      all   0.037 / 0.082 / 0.288 / 0.222   observed 0.404
      well  0.034 / 0.103 / 0.332 / 0.281   observed 0.586
  The chosen read-out roughly halves the generated donor spread in both subsets.
  Note: on all genes, snapens OVER-shoots real spread (ratio 1.21). On the well-expressed genes it matches (0.99).

Training length:
  Single-state pb_rmse is flat from 5k to 20k updates. For example, 10k minus 20k is -0.005 +- 0.040 (control),
  so the round-3 0.595-versus-0.645 lead was single-state noise.
  The 60k gene-block runs degrade after 20k updates, with no consistent optimum.
  Recommendation: keep snapens (or ema@0.9995) as the default. Do not schedule shorter runs. Never select a
  read-out on pb_rmse alone.


2. WORK ITEM 2: THE COUNT-SHAPE GAP  (r5/shape_diagnose.py, r5/mode_zcal.py, r5/test_mode_zcal.py)
-------------------------------------------------------------------------------------------------
Outputs: runs/improve_20261109/smoke_r5/shape/{diagnosis_summary.json, zeval_report.json, real_fold*.json, zeval/}

Diagnosis (seed 0, both arms, folds 0-2):
  At matched mean and depth, the generated cells have too many zeros and too heavy tails, i.e. they are
  over-dispersed.
  The main cause is the network's cell-to-cell spread of mu, not the theta parameter.
  At matched mean, 38-50% of genes have a real nonzero fraction above the generator's Poisson limit.
  No zero-inflation term is needed.

CORRECTION F5: NB reference.
  nb_donor fits per-donor means in-sample on the same 25 train donors it is scored on. It is an in-sample UPPER
  bound (nonzero 0.0036-0.0042, meanlog 0.016), not a comparator.
  The like-for-like comparator for the generators is nb_pop (one mean per gene, also in-sample):
    nonzero 0.0060-0.0072 (bias -0.0017 to +0.0007)
    meanlog 0.028-0.033
  The conclusions stand: a single NB per gene reproduces the train nonzero fraction to about 0.006-0.007,
  against the generators' 0.015 in-sample, and needs no zero inflation.

CORRECTION F7: shape share of the per-gene MSE, quoted per fit.
  biascal@stagehalf, seed 0, 6 fits:
    train    meanlog 0.65-1.03, nonzero 0.82-1.22
    held-out meanlog 0.46-0.63, nonzero 0.61-0.74
  (The fold-averaged "82-106%" quoted before was the train set only.)
  The held-out shape residual is about the same size as the train-mean predictor's error, which is the
  donor-to-donor part: shape rmse minus train-mean error is -0.011 to +0.004 (meanlog) and 0.000 to +0.004
  (nonzero). On held-out donors, most of the remaining error therefore cannot be separated from donor variation.

Fix A, zcal (zcal@biascal:stagehalf against biascal@stagehalf, held-out, folds 0-2):
  control, seeds 0-1:
    meanlog          -0.0116 (SE 0.0011, 3/3)
    nonzero          -0.0032 (SE 0.0003, 3/3)
    pb_rmse           0.0000
    pc_energy_rep    +0.040 (SE 0.006, 0/3)   cost beyond 2 SE
    cell_energy_rep  +0.019 (SE 0.004, 0/3)   cost beyond 2 SE
  combination, seeds 0-3:
    meanlog          -0.0072 (3/3)
    nonzero          -0.0026 (3/3)
    pc_energy_rep    -0.086
    cell_energy_rep  -0.039
    pb_rmse          +0.0008 (SE 0.0002)

CORRECTION F2: resampling is not a floor, and zcal is a pull onto train statistics.
  The direct train-mean predictor predicts each held-out donor's per-gene statistic by the train-donor mean.
  It uses the same train statistic zcal calibrates to, and it beats both resampling and zcal.
  Folds 0-2 means:
                               meanlog_rmse_gene   nonzero_rmse_gene
    train-mean predictor       0.0728              0.0160
    resampling (M = 32)        0.0790              0.0172
    zcal, control              0.0837              0.0177
    zcal, combination          0.0941              0.0201
    biascal, control           0.0953              0.0209
    biascal, combination       0.1013              0.0227
  Fold detail: resampling minus train-mean is +0.010/+0.010/-0.001 for meanlog and +0.002/+0.002/-0.001 for
  nonzero (folds 0/1/2), so in fold 2 resampling beats the train-mean predictor. Neither is a floor.
  zcal minus train-mean predictor:
    control      meanlog +0.0109 (SE 0.0009), nonzero +0.0017 (SE 0.0002); worse in 3/3 folds
    combination  meanlog +0.0213, nonzero +0.0040; worse in 3/3 folds
  zcal minus resampling: control +0.0047 / +0.0006, combination +0.0151 / +0.0029; neither reaches resampling.
  "zcal closes 71% (meanlog) / 85% (nonzero) of the control's gap" is measured against resampling only, and
  against the train-mean predictor it closes 52% / 65% of the control's gap.
  Like-for-like after the evaluator's train-only additive per-gene recalibration, applied alike to models and
  resampling, zcal changes nothing:
    zcal minus biascal, recal meanlog   +0.0001 (control), -0.0002 (combination, 3/3)
    zcal minus biascal, recal nonzero   +0.00004 (control), -0.00007 (combination)
    gap to resampling after recal       0.0018-0.0024 meanlog and 0.0003-0.0006 nonzero, with or without zcal
  Verdict: zcal is a train-statistic pull with no gain beyond a per-gene recalibration. It does not meet the
  PROTOCOL_R5 targets for either arm, and for the control it also costs pc_energy_rep and cell_energy_rep.

Fix B, pzero (r5/arm_pzero.py, r5/test_arm_pzero.py):
  Built and tested. Its straight-through gradient was checked against the closed-form NB zero probability.
  Its standardisation uses train bags only. Nothing has been trained beyond tests.

CORRECTION F3: like-for-like controls.
  pbulk,pzero,t.kappa@0.19 differs from pbulk,t.kappa@0.25 in BOTH the block and kappa.
  runs/improve_20261109/jobs_r5.txt now adds the kappa-matched controls:
    Block A: pbulk,t.kappa@0.19, folds 0-2, seeds 0-1 (6 fits)
    Block B: g.dispz,pbulk,t.kappa@0.19 (6 fits)
  The comparisons, fixed in advance, are in r5/pzero_compare.py:
    block effect                  pbulk,pzero,t.kappa@0.19 - pbulk,t.kappa@0.19
    block + width-matched kappa   pbulk,pzero,t.kappa@0.19 - pbulk,t.kappa@0.25
    vs the r3 control             pbulk,pzero,t.kappa@0.19 - pbulk
    swap (not an add)             pzero,t.kappa@0.25 - pbulk,t.kappa@0.25
    kappa alone                   pbulk,t.kappa@0.19 - pbulk,t.kappa@0.25
    Block B                       g.dispz,pbulk,pzero,t.kappa@0.19 - g.dispz,pbulk,t.kappa@0.19
                                  and - g.dispz,pbulk,t.kappa@0.25
  fits_r3/g.dispz,genemean,pbulk,t.kappa@0.19 is a width- and kappa-matched third-block reference.
  Job count: 33 (Block A 21, Block B 12). run_screen.sh -n parses all 33 jobs into fits_r5 and logs_r5.

CORRECTION F1 (code): r5/generate.py could write into the frozen fits_r3.
  generate._check_out now applies trainer.check_out_root to the fit folder whenever --out is not given
  (docstring R5-G1).
  A fits_r3 or fits_r4 fit with no --out, or with --out inside a frozen tree, is refused.
  Fits under fits_r5 (the trainer's post-training generation and run_screen.sh modes) and scratch fits
  outside RUNS are still written into. Generation output itself is unchanged.
  Verified through the CLI: the call is refused with no file written into fits_r3. test_core.py guards checks
  8 cases.


3. TESTS RUN AFTER THE REPAIR
-----------------------------
test_core.py parity                PASS: bitwise round 1 (base, genemean, offset, donorpc, covars),
                                   round 2 (pbulk, genemean@0.5), frozen r3 trainer (pbulk,
                                   g.dispz,pbulk,t.kappa@0.25), t.ema
test_core.py guards (new)          PASS: 8 out-root cases
test_shortavg.py unit, score       PASS: score equals results_r3 in 49 metrics, max diff 0
test_mode_zcal.py parse analytic observed   PASS
test_arm_pzero.py unit, neutral    PASS
GPU smoke, 300 updates             2 completed fits, all 4 modes finite.
                                   Arms: pbulk,t.kappa@0.19 and g.dispz,pbulk,t.kappa@0.19.
                                   Command: run_screen.sh -d cuda:1 -p 1 -u 300 -g 0.04 -o smoke_r5/repair/fits.
                                   This exercises the new guard path. Peak CUDA about 504 MB.
                                   Record: smoke_r5/repair/gpu_smoke.json. Weights deleted.
No arm code changed, so the pzero GPU smoke (test_arm_pzero.py gpu) was not repeated.
shortavg analyze and shape_diagnose summary and zreport were rerun from stored scores. All earlier numbers are
unchanged; only labels and the new sections were added.


4. COMMANDS
-----------
(a) Launch the pzero screen (Block A, then Block B). Run in bash.
    To stay under 3 GB kept, run Block A, score it, delete its mode files, then run Block B.

    R=/ml2/alz/runs/improve_20261109; R5=/ml2/alz/analysis/improve_20261109/r5
    mkdir -p $R/logs_r5
    sed -n '1,/^# --- Block B/p' $R/jobs_r5.txt > $R/logs_r5/jobs_r5_blockA.txt
    { grep '^#' $R/jobs_r5.txt | head -1; sed -n '/^# --- Block B/,$p' $R/jobs_r5.txt; } > $R/logs_r5/jobs_r5_blockB.txt
    cd /tmp && nohup $R5/run_screen.sh -d cuda:0,cuda:1 -p 12 -t 2 -g 0.04 -j $R/logs_r5/jobs_r5_blockA.txt \
        > $R/logs_r5/screen_pzero_A.out 2>&1 &
    # after Block A is scored and its mode files deleted:
    cd /tmp && nohup $R5/run_screen.sh -d cuda:0,cuda:1 -p 12 -t 2 -g 0.04 -j $R/logs_r5/jobs_r5_blockB.txt \
        > $R/logs_r5/screen_pzero_B.out 2>&1 &
    # dry run of either file: add -n

(b) Evaluate the pzero screen (after it has finished). Run in bash.

    R=/ml2/alz/runs/improve_20261109; R5=/ml2/alz/analysis/improve_20261109/r5
    PY="/mnt/nvme/ai/conda/anaconda3/bin/python3 -B"
    export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3
    OUT=$R/results_r5/pzero; mkdir -p $OUT/fits
    cp -rn $R/results_r3/baselines $R/results_r3/floor.json $R/results_r3/real $OUT/
    (cd $R/results_r3/fits && cp -rn pbulk~r3* 'pbulk,t.kappa@0.25~r3'* 'g.dispz,pbulk,t.kappa@0.25~r3'* $OUT/fits/)
    SEL="--fits-root r3=$R/fits_r3 --fits-root r3=$R/fits_r5 --modes-root $R/smoke_r5/pzero/modes_r3 --out $OUT \
      --panels hvg2000 --seeds 0,1 --arm pbulk~r3* --arm pbulk,t.kappa@0.25~r3* --arm pbulk,t.kappa@0.19~r3* \
      --arm *pzero* --arm g.dispz,pbulk,t.kappa@0.25~r3* --arm g.dispz,pbulk,t.kappa@0.19~r3* \
      --ensemble-arm *#snapens --ensemble-arm *#stagehalf --ensemble-arm *#stageblind --ensemble-arm *#biascal@stagehalf"
    cd /tmp; set -f
    for k in 0 1 2; do $PY $R5/evaluate.py $SEL --folds $k > $R/logs_r5/eval_pzero_fold$k.log 2>&1 & done; wait
    $PY $R5/evaluate.py $SEL --folds 0,1,2 > $R/logs_r5/eval_pzero_all.log 2>&1; set +f
    $PY $R5/pzero_compare.py --results $OUT --folds 0,1,2 --seeds 0,1 > $R/logs_r5/pzero_compare.txt
    # then delete the scored mode files:
    find $R/fits_r5 -name 'generated_*.npz' -delete; rm -rf $R/smoke_r5/pzero/modes_r3

    How to read pzero_compare.txt:
    - The headline is "block effect (kappa matched)" at snapens and at biascal@stagehalf.
    - Targets: meanlog and nonzero against resampling AND against the train-mean predictor.
    - Guards: pb_rmse, pc_energy_rep and cell_energy_rep no worse than 2 SE. Always read them together with
      pc_spread_rep, pc_coverage80_rep and collapse_flag.
    - The evaluator's own dCtrl column pairs pbulk,t.kappa@0.19 with pbulk but cannot pair the pzero arms;
      use pzero_compare.py for those.

(c) Short-training analysis on folds 3-8: stage-blind and calibrated variants. The plain read-outs are done.

    R5=/ml2/alz/analysis/improve_20261109/r5; PY="/mnt/nvme/ai/conda/anaconda3/bin/python3 -B"
    cd /tmp && CUDA_VISIBLE_DEVICES=0 $PY $R5/shortavg.py run --set main --arms pbulk --folds 3,4,5,6,7,8 --seeds 0,1 \
        --variants sb,bc --bcsb 'snapema@5000:0.9995;snapens;ema@0.9995'
    cd /tmp && $PY $R5/shortavg.py analyze      # writes smoke_r5/shortavg/analysis.{txt,json}, incl. the collapse check

(d) zcal.
    One fit (writes only under smoke_r5):
      R=/ml2/alz/runs/improve_20261109; R5=/ml2/alz/analysis/improve_20261109/r5; PY="/mnt/nvme/ai/conda/anaconda3/bin/python3 -B"
      cd /tmp && CUDA_VISIBLE_DEVICES=0 $PY $R5/mode_zcal.py --fit $R/fits_r3/pbulk/hvg2000/fold0/seed0 \
          --cache $R/caches/hvg2000/fold0 --mode zcal@biascal:stagehalf \
          --out $R/smoke_r5/zcal/pbulk_f0_s0.npz --device cuda --gpu-mem-fraction 0.04
    The folds 0-2 evaluation (generates, scores, deletes mode files), then the report:
      cd /tmp && $PY $R5/shape_diagnose.py zeval --folds 0,1,2
      cd /tmp && $PY $R5/shape_diagnose.py zreport  # zeval_report.json incl. the trainmean reference and the "headline" block
    Folds 3-8 (control pbulk, seeds 0-1; the fits_r3 biascal@stagehalf files there are the paired reference):
      for k in 3 4 5 6 7 8; do for s in 0 1; do
        CUDA_VISIBLE_DEVICES=0 $PY $R5/mode_zcal.py --fit $R/fits_r3/pbulk/hvg2000/fold$k/seed$s \
          --cache $R/caches/hvg2000/fold$k --mode zcal@biascal:stagehalf \
          --out "$R/smoke_r5/zcal_f38/pbulk_f${k}_s${s}/generated_zcal@biascal:stagehalf.npz" \
          --device cuda --gpu-mem-fraction 0.04; done; done
      Then score these with evaluate.py --modes-root $R/smoke_r5/zcal_f38 (fits root r3=$R/fits_r3, arm pbulk~r3*,
      folds 3-8) and pair them against pbulk~r3#biascal@stagehalf by fold and seed. The real_fold{k}.json train-mean
      reference must first be made with: shape_diagnose.py real --fold k
      NOT YET EXERCISED: scoring zcal mode files through evaluate.py --modes-root. zeval scored them in-process
      with evaluate.score_generated. Delete the npz files once they are scored.

(e) Tests.
      cd /tmp && $PY $R5/test_core.py parity; $PY $R5/test_core.py guards
      cd /tmp && $PY $R5/test_shortavg.py unit; $PY $R5/test_shortavg.py score
      cd /tmp && $PY $R5/test_mode_zcal.py parse analytic observed
      cd /tmp && $PY $R5/test_arm_pzero.py unit; $PY $R5/test_arm_pzero.py neutral


5. OPEN ISSUES
--------------
- Disk: Block A is about 3.6 GB at peak with modes, A+B about 5.6 GB. Score and delete mode files between blocks.
- r5/test_evaluate_r3.py still expects results_r3/smoke_r3 as the writable evaluator trees, so its out-dir check
  will fail against the r5 evaluator guard. It was not edited here.
- Babel onboarding (user CLAUDE.md) was not performed by this workflow subagent: the babel_account_wake MCP
  server failed to connect.
