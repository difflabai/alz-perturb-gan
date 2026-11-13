improve_20261109 ROUND 3: how to launch, generate and evaluate
===============================================================

Written by the integration task, 2026-11-11.  Binding rules are in
r3/PROTOCOL_R3.md, r2/PROTOCOL_R2.md and PROTOCOL.md.  Code: this folder (r3/).
Outputs: /ml2/alz/runs/improve_20261109/{fits_r3,results_r3,logs_r3,smoke_r3}.

Conventions used below (bash; zsh does not split "$PY" into two words, so run
these in bash or write the interpreter path out)
  R3=/ml2/alz/analysis/improve_20261109/r3
  RUNS=/ml2/alz/runs/improve_20261109
  PY="/mnt/nvme/ai/conda/anaconda3/bin/python3 -B"
  export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3
  Run Python from /tmp (cd /tmp) so no stray module in the cwd shadows r3/.


1. WHAT IS IN THE SCREEN
------------------------
Job files (one line "arm panel fold seed modes"; '#' lines are comments and
list every arm with its units and the reason it is there).  Both were
rewritten by the repair task (2026-11-11) with
$RUNS/smoke_r3/repair/make_jobs_repair.py:

  $RUNS/jobs_r3.txt          stage 1: seed 0, folds 0-2, 18 arms, 68.0 units
  $RUNS/jobs_r3_stage2.txt   stage 2, PROVISIONAL superset, 77.8 units unpruned
                             (seed 1 of the control and of every stage-1 arm
                             except plain base t.steps@60000, plus three
                             placeholder combinations at seeds 0 and 1).
                             Prune it after stage 1 is scored (section 7).

Control observation: pbulk.  Round 2 is now complete: pbulk~r2#snapens scores
0.549 / 0.542 (seed 0 / seed 1, folds 0-2), mean 0.546, against
genemean@0.25 0.549 / 0.568 (0.558), genemean 0.562 / 0.577, genemean@0.5
0.566 / 0.571 (smoke_r3/repair/eval_seed; r3 evaluator).  Every single-factor
arm is "pbulk,<one flag>" and is compared with the round-3 control "pbulk".

Stage-1 arms (units): pbulk,t.steps@60000 (9); t.steps@60000 (9); pbulk (3,
control); pbulk,t.anneal@0.1 (3); pbulk,t.anneal@0.01 (3); pbulk,t.glr@0.5
(3); pbulk,t.dlr@0.5 (3); pbulk,t.reg@4 (3); pbulk,t.kappa@0.25 (3);
pbulk,t.particles@1024 (3); g.dispz,pbulk (3.2); g.dispz@0.1,pbulk (3.2);
g.h@256,pbulk (3.4); d.tower,pbulk (3); d.cellwhite,pbulk (3.6);
d.cellwhite,pbulk,t.kappa@0.5 (3.6, two factors); nostage,pbulk (3);
fixedstage,pbulk (3).  Changes by the repair (section 7): g.disp -> g.dispz
and g.dispz@0.1; pbulk,t.ema@0.9999 dropped; the three combinations moved to
stage 2; the 60k lines gained window-matched and in-run 20k read-outs.

Modes on every stage-conditional line: snapens,stagehalf,stageblind,ema@0.9995.
The control also gets ema@0.999,ema@0.9998,condmix@1,condmix@0.5,
biascal@stagehalf.  The 60,000-update lines also get snapens@5, ema@0.9998,
snapema@20000:0.9995, snapema@20000:0.9998 and snap@10000 ... snap@50000.
nostage gets snapens,ema@0.9995,biascal@snapens.  Fewer modes than asked for
because of the 25 GB disk rule (addition 7): each mode file is about 25 MB.
Any mode can be added later (the snapshots stay).

Primary read-outs (addition 4, implemented in r3/evaluate.py primary_readout):
constant learning rate -> #snapens; t.anneal / t.ema -> the default
generated.npz; stage-conditional arms additionally #stagehalf (T1a) and
#biascal@stagehalf (T1b); #stageblind is listed with the stage-blind units.
Secondary read-outs declared by the repair before any round-3 score: section 7.


2. LAUNCH
---------
Wait until the round-2 screen (r2/run_screen.sh, jobs_r2.txt) has finished:
both screens share the two GPUs.  Then:

  mkdir -p $RUNS/logs_r3 && cd /tmp
  nohup $R3/run_screen.sh -d cuda:0,cuda:1 -p 12 -t 2 -g 0.04 \
      -j $RUNS/jobs_r3.txt > $RUNS/logs_r3/screen_stage1.out 2>&1 &

  (-o defaults to $RUNS/fits_r3, -l to $RUNS/logs_r3; each fit runs to its own
  horizon: t.steps@N or 20000.  Dry run: add -n.)

Stop: kill -TERM <run_screen.sh pid>; every running fit checkpoints at an
update boundary.  Restart: the same command; complete fits are skipped,
interrupted fits resume bitwise from recovery.pt, missing modes are generated.
trainer.py refuses to overwrite a completed fit.

Stage 2 (after stage 1 is scored and the file is pruned):

  nohup $R3/run_screen.sh -d cuda:0,cuda:1 -p 12 -t 2 -g 0.04 \
      -j $RUNS/jobs_r3_stage2.txt > $RUNS/logs_r3/screen_stage2.out 2>&1 &

Single fit by hand:

  $PY $R3/trainer.py --cache $RUNS/caches/hvg2000/fold0 --arm pbulk,t.anneal@0.1 \
      --seed 0 --device cuda --gpu-mem-fraction 0.04

Arm strings: comma-joined flags in Python sort order (run_screen.sh sorts
them for the folder name).  Flags: round-1/2 names, t.* (arms.py TRAIN_OPTIONS:
steps anneal annealstart netfloor nethorizon ema glr dlr plr particles zdim
batch bag reg kappa priorreg), g.* (arm_garch.py: g.h g.zc g.depth g.disp
g.cdisp; arm_example.py: g.example), d.* (arm_dcritic.py: d.tower d.sum d.h
d.depth; arm_cellcritic.py: d.cellset d.cellwhite; arm_example.py: d.example),
stage arms nostage and fixedstage[@w].  t.reg, t.kappa and t.priorreg were
added by integration (PROTOCOL_R3 addition 1): reg_coeff = M, reg_kappa = M,
prior_reg = .01 M through get_recipe's own fields.


3. GENERATION MODES (also on round-1/2 fits)
---------------------------------------------
  $PY $R3/generate.py --fit FIT --cache $RUNS/caches/hvg2000/foldK --mode MODE \
      [--out OUT.npz] --device cuda --gpu-mem-fraction 0.04

Round-3 fits write FIT/generated_MODE.npz.  Round-1 (fits/) and round-2
(fits_r2/) fits are frozen: --out is REQUIRED and must lie outside fits/ and
fits_r2/, e.g. --out $RUNS/results_r3/modes/<arm>/hvg2000_foldK_seedS_MODE.npz
and then pass --modes-root $RUNS/results_r3/modes to the evaluator.  Any file
name works: the evaluator pairs every *.receipt.json that has "fit" and
"mode" keys and a sibling .npz with the fit, and the unit inherits the tag of
the --fits-root holding that fit (arm~r2#MODE for a round-2 fit).

Modes:
  raw            live final weights (differs from default only for t.ema fits)
  ema@D          shadow EMA at D in {0.999, 0.9995, 0.9998}       (round 3 only)
  snapens        snapshots at u >= 0.75 U plus raw final (round 2: 16000-19000 + final)
  snapens@K      last K snapshots + raw final
  snap@U         snapshot U (round 2: 16000..19000; round 3: config snapshot_updates)
  snapema@U:D    shadow EMA D stored in snapshot U                (round 3 only)
  condmix@L, condunk@L, condshrink@L, margcov, owncov, anchorgain@g   (round 2)
  stagehalf[@R]  condmix@0.5 conditions on the primary read-out states
                 (R = final | raw | snapens | ema<D>, e.g. stagehalf@ema0.9995)
  stageblind[@R] condmix@1 conditions on the same states (stage-blind)
  biascal[@INNER] train-only per-gene output-bias calibration, INNER written
                 with ':' for its '@' (biascal@stagehalf, biascal@ema:0.9995).
                 Its calibration record is stored in the mode receipt under
                 mode_extra.biascal (integration edit of generate.py); the
                 stand-alone CLI $PY $R3/mode_biascal.py ... also writes
                 OUT.biascal.json.

Which snapshots exist: 20,000 updates -> 5000, 10000, 15000-19000 (7 files,
5.1 MB each); 60,000 updates -> 5000-40000 every 5000 and 45000-59000 every
1000 (23 files, about 117 MB).  For a run of U updates both intervals scale with
U/20000.


4. EVALUATE
-----------
The result cache is ALREADY SEEDED: integration moved the evaluator task's
baselines/ (folds 0-8) and per-unit fits/ cache (round-1 and round-2 units,
EVALUATOR_VERSION r3-v1) from smoke_r3/eval/results_now into $RUNS/results_r3
and copied floor.json there.  This saves about 2 h of baselines and
re-scoring.  The round-2 snap@U units in that cache lost their .npz files
(deleted by the evaluator task for disk), so they drop out of new aggregates.
To recompute the floor (about 9 min, only if baselines change):

  $PY $R3/floor.py --folds 0,1,2,3,4,5,6,7,8 --boot 1000 \
      --baseline-root $RUNS/results_r3 --out $RUNS/results_r3/floor.json

Score (one process per fold in parallel, then one aggregate run):

  cd /tmp
  for k in 0 1 2; do
    $PY $R3/evaluate.py --fits-root $RUNS/fits --fits-root r2=$RUNS/fits_r2 \
        --fits-root r3=$RUNS/fits_r3 --modes-root $RUNS/posthoc_r2/modes_gpu \
        --out $RUNS/results_r3 --panels hvg2000 --folds $k \
        --ensemble-arm base --ensemble-arm genemean --ensemble-arm '*~r2#snapens' \
        > $RUNS/logs_r3/eval_fold$k.log 2>&1 &
  done; wait
  $PY $R3/evaluate.py --fits-root $RUNS/fits --fits-root r2=$RUNS/fits_r2 \
      --fits-root r3=$RUNS/fits_r3 --modes-root $RUNS/posthoc_r2/modes_gpu \
      --out $RUNS/results_r3 --panels hvg2000 --folds 0,1,2 \
      --ensemble-arm base --ensemble-arm genemean --ensemble-arm '*~r2#snapens' \
      > $RUNS/logs_r3/eval_all.log 2>&1

Read: results_r3/leaderboard.txt (ranking, T1/T2, P column = primary read-out,
dCtrl = difference to the round-3 control pbulk~r3 with the MATCHED read-out
(evaluate.control_mode: same mode; snap@20000 -> control default;
snapema@20000:D -> control ema@D; snapens@5 -> control snapens; nostage ->
control stageblind / condmix@1), dCtrlP = the unit's PRIMARY read-out minus
the control's primary read-out, +- 2 SE of seed noise),
targets.json (T1a/T1b/T2/T3 per unit), curves.json (score versus updates:
series raw = snap@U + raw final, ema@D = snapema@U:D + ema@D), summary.txt and
aggregate.csv.  About 11 s of CPU per scored unit; stage 1 has about 360 units (about 25 min with 3 fold processes).

Stage-2 selection (addition 6) from targets.json:
units['<arm>~r3#<primary>|hvg2000'].groups['folds0-2'].T1.pb_rmse and
.T2.n_pass against the control's; for combinations also
.vs_control_primary (mean_difference, n_better, se); training length from
.vs_length_reference of the t.steps@60000 units (section 7).


5. TESTS
--------
  cd /tmp
  $PY $R3/test_core.py {parity|recipes|anneal|capacity|hooks|cli|oldfits|all}
  $PY $R3/test_arm_garch.py {flags|neutral|init|probe|cli|cleanup}
  $PY $R3/test_arm_dcritic.py {unit|penalty|neutral|fits|cli|cleanup}
  $PY $R3/test_arm_cellcritic.py {unit|fit|cost|cleanup}
  $PY $R3/test_arm_stage3.py {unit|cli|cleanup}
  $PY $R3/test_evaluate_r3.py {units|floor|synthetic ...|reproduce ...}
  copied round-2 tests (now writing under smoke_r3/r2copy_tests):
  test_arm_gm.py test_arm_stage.py test_arm_critic.py test_arm_cov.py
  test_posthoc.py test_splits.py test_evaluate_r2.py


6. BUDGET (measured by integration, 2026-11-11)
-----------------------------------------------
Units (1 = one 20,000-update fit, scaled by measured s/update): stage 1 68.0,
stage 2 (provisional, unpruned) 77.8, total 145.8 (repair; integration had
77.2 + 68.2).
Wall time: round-2 fits at 12 concurrent on the two shared GPUs took 2.2-2.7 h
per 20,000 updates (0.39-0.49 s/update).  Stage 1 is about 16 h (the six
60,000-update fits, about 7.5 h each, start first); stage 2 about 14 h.
Smoke s/update at 300 updates (2 concurrent next to the 12 round-2 fits;
+-10% noise): control 0.50; d.cellwhite 1.2x (cellcritic task: 1.22x at 180
updates); g.h@256 1.12x; g.disp 1.07x; t.particles@1024 1.08x; d.tower 0.97x;
everything else within noise.  Peak CUDA allocation 0.40-0.61 GB per fit
(fraction 0.04 = 1.9 GB).
Disk (repair estimate): stage 1 about 10.8 GB, stage 2 unpruned about 12.5 GB,
together about 23 GB of the 25 GB cap; prune stage 2.
Disk (integration): 20,000-update fit = 7 snapshots (35.7 MB) + final.pt (7.7 MB) +
generated.npz (24.3 MB) + 24.8 MB per mode file; g.h@256 snapshots are 81 MB,
t.ema adds the native EMA (44.6 MB of snapshots).  60,000-update fit: 23
snapshots, about 125 MB of model files.  Estimated: stage 1 12.4 GB, stage 2
(unpruned) 10.7 GB, together about 23 GB of the 25 GB cap (addition 7).
Snapshot density was kept (a 20,000-update fit holds 43 MB of model files).


7. REPAIR AFTER THE ADVERSARIAL REVIEW (2026-11-11), READ-OUT RULES, SELECTION HISTORY
-------------------------------------------------------------------------------------
Code changes (each documented in the module docstring):
  arm_garch.py      new flag g.dispz[@s]: donor-LATENT-only dispersion (no
                    condition input, so no stage path); s = fixed
                    re-parametrisation scale (effective DualNorm step x s).
                    g.disp kept unchanged.  disp_stats / "arm_garch.py
                    dispstats FIT" report the dispersion-map norms (stage
                    column for g.disp) from every snapshot.
  arm_fixedstage.py refuses g.disp; info.stage_blind_generator when the offset
                    table is numerically zero (folds 1-2).
  arm_cellcritic.py penalty_dim filled in config.json (was null).
  capacity_diagnose.py  probe population modes popx (same stage, target
                    excluded), popall / popallx (any train donor's latent);
                    resamplex reference; variant dispz.
  evaluate.py       matched dCtrl, dCtrlP (primary vs control primary) with a
                    seed-noise interval, vs_length_reference for t.steps arms,
                    T2 'missing', fixedstage stage-blind listing, T1
                    reference note, selection history (S3).
  arm_dcritic.py    docstring note (starvation = width scaling; t.reg).

Read-out rules declared now, before any round-3 score:
  * Training length: judge pbulk,t.steps@60000 and t.steps@60000 on
    snapens@5 (last 5 snapshots + final: the same 6-state, 5,000-update
    window as the control's snapens), ema@0.9995 / ema@0.9998 and snap@U.
    vs_length_reference pairs ema@D at 60k with snapema@20000:D of the SAME
    run and the default with snap@20000 (constant rates: the long run's first
    20,000 updates are the control's trajectory).  #snapens of a 60k run
    (16 states over 15,000 updates) is reported only as a wider-window
    read-out, never as the length answer.
  * Annealing: the primary is the final state (addition 4), but annealed
    arms are ALSO judged on #snapens and #ema@0.9995 against the control's
    same read-outs (dCtrl), because the round-2 evidence (random-walk drift,
    snapens beats every single state by 0.033-0.091) supports averaging,
    not necessarily freezing one point of the walk.
  * EMA: the control's shadow ema@0.999/0.9995/0.9998 answer it (initial-
    weight share 0.9998^20000 = 1.8%, 0.9995^20000 = 4.5e-5).  The native
    t.ema@0.9999 arm was dropped: its average keeps 0.9999^20000 = 13.5% on
    the untrained initial weights and its training is bitwise the control's.
    In curves.json, snapema@U:D points with D^U > 0.01 (e.g. 0.9998 at
    U <= 23,000) carry a non-negligible initial-weight share.
  * Seed noise: one seed on three folds resolves only differences beyond
    about 2 SE.  Round-2 seed pairs (5 arms x 3 folds, #snapens) give a
    seed SD of 0.0156 pb_rmse (pbulk alone 0.026, 3 pairs), so a paired
    single-seed 3-fold difference has SE about 0.013 (2 SE about 0.025).
    A single raw final state is far noisier (pbulk default SD 0.067), so
    intervals for single-state primaries (annealed arms) are optimistic.
    Stage-1 "nulls" (t.particles, g.h, d.tower) are uninformative, not
    confirmations.
  * Stage-2 combinations (stricter than addition 6, a subset of it): each
    combined single factor must beat the control on the primary read-out in
    3 of 3 folds AND by more than 2 SE (targets.json vs_control_primary).
    The combinations in jobs_r3_stage2.txt are placeholders.
  * Two-factor arm d.cellwhite,pbulk,t.kappa@0.5 is compared with
    d.cellwhite,pbulk and pbulk,t.kappa@0.25 as well as the control.
  * t.particles@1024 also halves draws per particle (about 625 instead of
    1250 moves per row); g.h@256 changes the initial draws (confounded with
    seed).  Expected near null: d.tower, t.reg@4 (scales mostly the active
    gauge term), t.particles@1024, g.h@256.
  * fixedstage: in folds 1-2 its offset table is about 3e-6 (tau2 floored),
    i.e. a stage-blind generator against a stage-aware critic; its units
    are listed STAGE-BLIND.  nostage is paired with the control's
    stageblind read-out.  A T1a pass via stagehalf is half stage-blind.
  * g.dispz / g.dispz@0.1: in a 300-update GPU test the latent-only map
    still grows at about the DualNorm maximum rate (Frobenius 4.9 / 18.9 /
    33 / 47 at u = 75 / 150 / 225 / 300; per-gene SD of the offsets across
    particles 0.98 at u = 300 against the probe's ML optimum 0.55-0.62).
    Whether it saturates is unknown without full training; @0.1 moves 10x
    slower.  Read with arm_garch.py dispstats on the snapshots.

Selection history (round 3).  Every round-3 number is selection-conditioned
development evidence on the same 37 donors:
  * Control pbulk: chosen from round-2 HELD-OUT primary read-outs (at
    integration time 0.547 vs genemean@0.25 0.555, decided by one extra seed
    on one fold; complete round 2 now 0.546 vs 0.558) and the stage task's
    held-out biascal@stagehalf (0.487).
  * The stage-1 arm list: from round-2 held-out results and train-donor
    diagnostics; the repair changed it (g.dispz, t.ema dropped, combos to
    stage 2) on train-side and mechanical evidence only (no held-out score
    of any round-3 arm exists).
  * stagehalf's 0.5: declared in PROTOCOL addition 4 (05:31) and coded
    (06:25) before any stagehalf score, informed by round-1/2 condmix
    held-out results.
  * biascal: v1 (plain unit steps) was scored on held-out donors (06:54;
    biascal@stagehalf 0.4918 genemean@0.25, 0.4882 pbulk); the rule was then
    changed to Newton steps and re-scored (07:13; 0.4894 / 0.4866), i.e.
    about 0.002 better, chosen after held-out scores had been seen (the
    stated reason was the train residual, which stalled with unit steps).
  * The T1 reference grand_mean (v4 residual centring) was kept in round 1
    over v3 on held-out scores (0.464 vs 0.500): this makes T1 stricter.
  * T1 may be unattainable for a faithful generator at M = 32 (ideal
    stage-blind generator emulation resample_blind 0.474 > 0.469).
