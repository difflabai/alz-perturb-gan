#!/usr/bin/env python3
"""Tests for the round-3 evaluator and floor (r3/evaluate.py, r3/baselines.py, r3/floor.py).

Subcommands (CPU only, 3 threads; scratch under runs/improve_20261109/smoke_r3/eval/, deleted at the end
except the test result JSONs in smoke_r3/eval/test_results/):

  units       pure-function checks: label parsing with t./g./d. flags, '+'-composed modes and tags;
              stage-kind rules (round-2 name rule unchanged for every round-2 mode, composites,
              receipt/config 'stage_blind', generated-condition slope); single-factor control labels;
              primary read-out rule (PROTOCOL_R3 addition 4); paired deltas; T1/T2 blocks with planted
              values (pass, miss, ties, coverage rule, missing metric); score-versus-updates curves with
              a planted curve (series assignment, best update, late slope); output guard; scan_fits tag
              inheritance for --modes-root.
  floor       planted-truth checks of r3/floor.py on simulated donors with a known population: route A
              and route B de-biased floors recover the Monte Carlo truth (the uncorrected train-mean
              error is above it), the stage oracle equals the grand oracle when there is no stage effect
              and is lower when there is one, and the cell-bootstrap technical variance matches the
              Poisson delta-method value.
  synthetic   end-to-end planted truth on the REAL fold-0 hvg2000 cache: synthetic round-3 fit folders
              whose generated files are exact copies of baseline generations (grand_mean, resample) or
              known perturbations of them.  Checks: a copy of grand_mean has pb_rmse equal to the
              baseline and passes T1 with margin exactly -0.005; a +0.15 log-shifted copy fails T1 by
              the measured amount; a copy of resample passes T2 6/6 with margins exactly 0; snap@U /
              snapema@U:D / ema@D / raw / default units land on the right curve series and the planted
              monotone curve is recovered (late slope < 0, best update = last); a mode file with constant
              generated ADNC is labelled stage-blind, a half-shrunk one stage-shrunk, 'stagehalf'
              stage-shrunk, 'biascal+snapens' T1b; t.ema default is the native-EMA series; vs_control
              pairs 'genemean,t.anneal@0.1~r3' with 'genemean~r3'; leaderboard and targets written.
              Baselines are copied from ``--baselines-from`` (an evaluator output with fold-0
              baselines of the same version, e.g. smoke_r3/eval/repro) when given, else computed.
  reproduce OUT [REF]
              compares a round-3 evaluation OUT of the round-1 + round-2 seed-0 fits (same inputs as
              REF, default results_r2_interim) with REF: every numeric value of every per-unit fit,
              baseline, noise-ceiling and real-reference JSON (keys present in REF), and every
              aggregate row value, paired block, xfit and pooled entry present in REF, to 1e-8.

Run: cd /tmp; OMP_NUM_THREADS=3 python3 -B /ml2/alz/analysis/improve_20261109/r3/test_evaluate_r3.py units
"""
from __future__ import annotations

import os
for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[_v] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import json
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
if str(HERE) in sys.path:
    sys.path.remove(str(HERE))
sys.path.insert(0, str(HERE))
import metrics as MX  # noqa: E402
import evaluate as EV  # noqa: E402
import baselines as BL  # noqa: E402
import floor as FL  # noqa: E402

for _m in (MX, EV, BL, FL):
    assert Path(_m.__file__).resolve().parent == HERE, _m.__file__

RUNS = Path("/ml2/alz/runs/improve_20261109")
SCRATCH = RUNS / "smoke_r3" / "eval" / "test_scratch"
RESULTS = RUNS / "smoke_r3" / "eval" / "test_results"
CHECKS = []


def check(name, cond, detail=""):
    CHECKS.append({"check": name, "pass": bool(cond), "detail": str(detail)})
    print(("PASS " if cond else "FAIL ") + name + " :: " + str(detail), flush=True)


def finish(tag, t0):
    n_fail = sum(not c["pass"] for c in CHECKS)
    RESULTS.mkdir(parents=True, exist_ok=True)
    EV.write_json(RESULTS / f"{tag}.json", {"checks": CHECKS, "n_fail": n_fail, "seconds": time.monotonic() - t0})
    print(f"\n{tag}: {len(CHECKS) - n_fail}/{len(CHECKS)} checks passed; {time.monotonic() - t0:.1f}s")
    return n_fail


# ==========================================================================
# units
# ==========================================================================
def _u(fold, seed, **m):
    return (fold, seed, dict(m), {"kind": "fit"})


def units():
    t0 = time.monotonic()
    lp = EV.label_parts("genemean,t.anneal@0.1,t.steps@40000~r3#biascal+snapens [seed-ensemble n=2 M=32]")
    check("label_parts: arm, tag, composed mode, suffix",
          lp["arm"] == "genemean,t.anneal@0.1,t.steps@40000" and lp["tag"] == "r3" and lp["mode"] == "biascal+snapens"
          and lp["components"] == ["biascal", "snapens"] and lp["suffix"] == " [seed-ensemble n=2 M=32]"
          and lp["base_unit"] == "genemean,t.anneal@0.1,t.steps@40000~r3", lp)
    lp = EV.label_parts("snapema~r3#snapema@15000:0.999")
    check("label_parts: snapema value keeps ':'", lp["mode"] == "snapema@15000:0.999" and lp["components"] == ["snapema@15000:0.999"], lp)
    # stage kinds: round-2 rule unchanged for every round-2 mode name
    r2modes = ["condmix@1", "condmix@0.5", "condmix@0", "condunk@1", "condunk@0.5", "condshrink@1", "condshrink@0.5",
               "anchorgain@0", "anchorgain@0.5", "margcov", "owncov", "snapens", "snap@16000", None, "condmix@0.98", "condmix@0.97"]
    check("stage_blind_mode_r3 == round-2 stage_blind_mode on every round-2 mode",
          all(EV.stage_blind_mode_r3(m) == EV.stage_blind_mode(m) for m in r2modes),
          [(m, EV.stage_blind_mode(m)) for m in r2modes])
    check("composite modes: blind component wins, then mixture, stagehalf -> stage-shrunk",
          EV.stage_blind_mode_r3("biascal+condmix@1") == "stage-blind" and EV.stage_blind_mode_r3("condmix@0.5+snapens") == "stage-mixture"
          and EV.stage_blind_mode_r3("stagehalf") == "stage-shrunk" and EV.stage_blind_mode_r3("biascal+stagehalf+snapens") == "stage-shrunk"
          and EV.stage_blind_mode_r3("biascal+snapens") is None and EV.stage_blind_mode_r3("ema@0.999") is None)
    fr = [{"unit": "a~r3#x", "mode": "x", "mode_receipt": {"stage_blind": True}},
          {"unit": "b~r3", "mode": None, "config": {"stage_blind": True}},
          {"unit": "c~r3#y", "mode": "y", "r3_condition_stage": {"adnc_slope_on_target_stage": 0.01}},
          {"unit": "d~r3#z", "mode": "z", "r3_condition_stage": {"adnc_slope_on_target_stage": 0.5}},
          {"unit": "e~r3#w", "mode": "w", "r3_condition_stage": {"adnc_slope_on_target_stage": 1.0}},
          {"unit": "f~r3", "mode": None, "r3_condition_stage": {"adnc_slope_on_target_stage": 0.0}},
          {"unit": "g~r3#condmix@1", "mode": "condmix@1", "r3_condition_stage": {"adnc_slope_on_target_stage": 1.0}}]
    k = EV.unit_stage_kinds(fr)
    check("unit_stage_kinds: receipt / config flag, condition slope (blind < 0.05 <= shrunk < 0.95), defaults never by slope, name rule first",
          k == {"a~r3#x": "stage-blind", "b~r3": "stage-blind", "c~r3#y": "stage-blind", "d~r3#z": "stage-shrunk",
                "e~r3#w": None, "f~r3": None, "g~r3#condmix@1": "stage-blind"}, k)
    # condition slope from a file
    SCRATCH.mkdir(parents=True, exist_ok=True)
    st = np.array([0, 1, 2, 3] * 5)
    cond = np.zeros((20, 32, 7), np.float32)
    cond[:, :, 3] = (st / 3.0)[:, None]
    np.savez(SCRATCH / "c1.npz", conditions=cond, stage=st)
    c2 = cond.copy(); c2[:, :, 3] = 0.5 * c2[:, :, 3] + 0.25
    np.savez(SCRATCH / "c2.npz", conditions=c2, stage=st)
    c3 = cond.copy(); c3[:, :, 3] = np.tile(np.array([0, 1 / 3, 2 / 3, 1.0])[None, :].repeat(8, 0).ravel()[None, :], (20, 1))
    np.savez(SCRATCH / "c3.npz", conditions=c3, stage=st)
    s1, s2, s3 = (EV.condition_stage_info(SCRATCH / f"c{i}.npz")["adnc_slope_on_target_stage"] for i in (1, 2, 3))
    check("condition_stage_info: slope 1 (own stage), 0.5 (half shrink), 0 (same stage mixture for every target)",
          abs(s1 - 1) < 1e-6 and abs(s2 - .5) < 1e-6 and abs(s3) < 1e-6, (s1, s2, s3))
    np.savez(SCRATCH / "c4.npz", stage=st)
    check("condition_stage_info: None without conditions", EV.condition_stage_info(SCRATCH / "c4.npz") is None)
    # control labels
    cl = {x: EV.control_label(x) for x in ["genemean,t.anneal@0.1~r3#snapens", "t.particles@1024~r3", "genemean,g.width@256,d.cell~r3 [seed-ensemble n=2]",
                                            "genemean~r3", "genemean@0.5,t.reg@4~r3#ema@0.999"]}
    check("control_label strips t./g./d. flags only, keeps tag/mode/suffix, empty -> base",
          cl == {"genemean,t.anneal@0.1~r3#snapens": "genemean~r3#snapens", "t.particles@1024~r3": "base~r3",
                 "genemean,g.width@256,d.cell~r3 [seed-ensemble n=2]": "genemean~r3 [seed-ensemble n=2]",
                 "genemean~r3": None, "genemean@0.5,t.reg@4~r3#ema@0.999": "genemean@0.5~r3#ema@0.999"}, cl)
    pr = {(x, r): EV.primary_readout(x, r) for x, r in [("genemean~r3#snapens", 3), ("genemean~r3", 3), ("genemean,t.anneal@0.1~r3", 3),
                                                       ("genemean,t.anneal@0.1~r3#snapens", 3), ("genemean,t.ema@0.999~r3", 3),
                                                       ("base", 1), ("base#condmix@1", 1), ("pbulk~r2#snapens", 2),
                                                       ("genemean~r3#stagehalf+snapens", 3), ("genemean~r3#biascal+snapens", 3),
                                                       ("genemean~r3#snapens@10", 3), ("genemean~r3#snapens [seed-ensemble n=2]", 3)]}
    want = {("genemean~r3#snapens", 3): "P", ("genemean~r3", 3): None, ("genemean,t.anneal@0.1~r3", 3): "P",
            ("genemean,t.anneal@0.1~r3#snapens", 3): None, ("genemean,t.ema@0.999~r3", 3): "P", ("base", 1): "P",
            ("base#condmix@1", 1): None, ("pbulk~r2#snapens", 2): "P", ("genemean~r3#stagehalf+snapens", 3): "P+stagehalf",
            ("genemean~r3#biascal+snapens", 3): "P+biascal", ("genemean~r3#snapens@10", 3): None,
            ("genemean~r3#snapens [seed-ensemble n=2]", 3): "P"}
    check("primary_readout follows PROTOCOL_R3 addition 4", pr == want, {k: v for k, v in pr.items() if want[k] != v})
    # paired delta and T1/T2 blocks on a planted units dict
    U = {("baseline:grand_mean", "p"): [_u(0, None, pb_rmse=.46), _u(1, None, pb_rmse=.47), _u(2, None, pb_rmse=.45)],
         ("baseline:resample", "p"): [_u(f, None, pb_rmse=.5, pc_energy_rep=2.0, cell_energy_rep=.8, pc_coverage80_rep=.7,
                                         meanlog_rmse_gene=.08, nonzero_rmse_gene=.03, logvar_rmse_gene=.4) for f in (0, 1, 2)],
         ("good", "p"): [_u(f, s, pb_rmse=v, pc_energy_rep=2.0, cell_energy_rep=.7, pc_coverage80_rep=.9, meanlog_rmse_gene=.07,
                            nonzero_rmse_gene=.03, logvar_rmse_gene=.3)
                         for f, s, v in [(0, 0, .46), (0, 1, .47), (1, 0, .47), (1, 1, .48), (2, 0, .46), (2, 1, .45)]],
         ("bad", "p"): [_u(f, 0, pb_rmse=.6, pc_energy_rep=2.5, cell_energy_rep=.9, pc_coverage80_rep=.6, meanlog_rmse_gene=.1,
                           nonzero_rmse_gene=.02, logvar_rmse_gene=None) for f in (0, 1)],
         ("ref", "p"): [_u(f, s, pb_rmse=.5) for f in (0, 1, 2) for s in (0, 1)]}
    b = EV._target_block(U, "good", "p", [0, 1, 2])
    gm = np.mean([.46, .47, .45]); val = np.mean([.465, .475, .455])
    check("T1: fold-balanced unit vs grand mean + 0.005 on common folds",
          abs(b["T1"]["pb_rmse"] - val) < 1e-12 and abs(b["T1"]["margin"] - (val - gm - .005)) < 1e-12 and b["T1"]["pass"], b["T1"])
    t2 = b["T2"]
    check("T2: ties pass, coverage |v-.8| rule (0.9 vs 0.7 tie), 6/6",
          t2["n_pass"] == 6 and t2["pass"] and abs(t2["metrics"]["pc_coverage80_rep"]["margin"]) < 1e-12
          and abs(t2["metrics"]["cell_energy_rep"]["margin"] + .1) < 1e-12 and t2["worst_miss"] is None, t2)
    bb = EV._target_block(U, "bad", "p", [0, 1, 2])
    check("T1 on the unit's common folds (0,1) only; miss margin",
          bb["T1"]["folds"] == [0, 1] and abs(bb["T1"]["margin"] - (.6 - .465 - .005)) < 1e-12 and not bb["T1"]["pass"], bb["T1"])
    check("T2: 1 of 6 (nonzero), missing logvar counted as not passed, worst miss = pc_energy +0.5",
          bb["T2"]["n_pass"] == 1 and bb["T2"]["n_available"] == 5 and not bb["T2"]["metrics"]["logvar_rmse_gene"]["available"]
          and bb["T2"]["worst_miss"][0] == "pc_energy_rep" and abs(bb["T2"]["worst_miss"][1] - .5) < 1e-12, bb["T2"])
    bs = EV._target_block(U, "good", "p", [0, 1, 2], {1})
    check("per-seed T1 uses only that seed", abs(bs["T1"]["pb_rmse"] - np.mean([.47, .48, .45])) < 1e-12, bs["T1"])
    d = EV.paired_delta(U, "good", "ref", "p", [0, 1, 2])
    check("paired_delta: seed-paired per fold, fold-balanced; n_better counts folds lower",
          abs(d["mean_difference"] - (val - .5)) < 1e-12 and d["n_folds"] == 3 and d["n_pairs"] == 6 and d["n_better"] == 3, d)
    tg = EV.build_targets(U, {"good": None, "bad": "stage-blind"}, {"good": 3, "bad": 3}, ["p"])
    check("build_targets: T1a lists split stage model vs stage-blind; complete flag",
          tg["lists"]["T1a_pass_folds0-2_stage_model"] == ["good|p"] and tg["units"]["good|p"]["groups"]["folds0-2"]["complete"]
          and not tg["units"]["bad|p"]["groups"]["folds0-2"]["complete"] and tg["units"]["bad|p"]["stage_blind"]
          and tg["units"]["good|p"]["groups"]["folds0-2"]["per_seed"]["0"]["T2_n_pass"] == 6, tg["lists"])
    # curves: planted
    cfg = {"updates": 20000, "round": 3, "flags": ["genemean"], "snapshot_updates": [5000, 10000, 15000, 16000, 17000, 18000, 19000]}
    fits = []
    planted = {5000: .70, 10000: .65, 15000: .62, 16000: .61, 17000: .60, 18000: .59, 19000: .58, 20000: .57}

    def fres(mode, pb, cfgx=cfg, unit="gm~r3"):
        return {"kind": "fit", "unit": unit + (f"#{mode}" if mode else ""), "mode": mode, "panel": "p", "fold": 0, "seed": 0,
                "config": cfgx, "subsets": {"all": {"heldout": {"pseudobulk": {"rmse": pb}}}}}
    for u, v in planted.items():
        if u < 20000:
            fits.append(fres(f"snap@{u}", v))
            fits.append(fres(f"snapema@{u}:0.999", v - .05))
    fits += [fres(None, planted[20000]), fres("ema@0.999", .51), fres("snapens", .5), fres("condmix@1", .45)]
    ecfg = dict(cfg, flags=["genemean", "t.ema@0.9995"], native_ema_decay=0.9995)
    fits += [fres(None, .4, ecfg, "em~r3"), fres("raw", .57, ecfg, "em~r3"), fres("snap@19000", .58, ecfg, "em~r3")]
    cv = EV.build_curves(fits, {}, ["p"])
    s = cv["per_fit"]["gm~r3|p|fold0|seed0"]["series"]
    late = [15000, 16000, 17000, 18000, 19000, 20000]       # u >= 0.75 x run length
    want_slope = np.polyfit(np.array(late) / 1000, [planted[u] for u in late], 1)[0]
    check("curves: raw series = snap@U + default (no native EMA); best update, late slope per 1000",
          s["raw"]["updates"] == sorted(planted) and s["raw"]["values"]["pb_rmse"] == [planted[u] for u in sorted(planted)]
          and s["raw"]["best_update_pb_rmse"] == 20000 and abs(s["raw"]["late_slope_per_1000_pb_rmse"] - want_slope) < 1e-12
          and s["raw"]["late_updates"] == late, s["raw"].get("late_slope_per_1000_pb_rmse"))
    check("curves: ema@0.999 series = snapema@U:0.999 + ema@0.999 at the run length; snapens/condmix are read-outs",
          s["ema@0.999"]["updates"][-1] == 20000 and s["ema@0.999"]["values"]["pb_rmse"][-1] == .51
          and set(cv["per_fit"]["gm~r3|p|fold0|seed0"]["readouts"]) == {"snapens", "condmix@1"}, s.keys())
    se = cv["per_fit"]["em~r3|p|fold0|seed0"]["series"]
    check("curves: t.ema fit default -> native_ema@D series; raw mode -> raw series at the run length",
          se["native_ema@0.9995"]["updates"] == [20000] and se["raw"]["updates"] == [19000, 20000]
          and se["raw"]["values"]["pb_rmse"] == [.58, .57], se.keys())
    check("curves: aggregate per (unit, panel, series)", cv["aggregate"]["gm~r3|p|raw"]["metrics"]["pb_rmse"]["value"][-1] == .57)
    # output guard
    bad = []
    for o in [RUNS / "results_r2", RUNS / "results_r2_interim" / "x", RUNS / "fits_r3", RUNS / "posthoc_r2", RUNS / "results", RUNS]:
        try:
            EV._check_out_dir(o); bad.append(str(o))
        except PermissionError:
            pass
    ok = []
    for o in [RUNS / "results_r3", RUNS / "smoke_r3" / "eval" / "x", Path("/tmp/x")]:
        try:
            EV._check_out_dir(o); ok.append(True)
        except PermissionError:
            ok.append(False)
    check("output guard: only results_r3 / smoke_r3 inside RUNS", not bad and all(ok), bad)
    # scan_fits tag inheritance
    sc = SCRATCH / "scan"
    fd = sc / "fitsX" / "genemean" / "hvg2000" / "fold0" / "seed0"; fd.mkdir(parents=True)
    (fd / "generated.npz").write_text("x"); (fd / "fit_receipt.json").write_text("{}")
    md = sc / "modes"; md.mkdir()
    (md / "a.npz").write_text("x"); (md / "a.receipt.json").write_text(json.dumps({"fit": str(fd), "mode": "snap@16000"}))
    md2 = sc / "modes2"; md2.mkdir()
    (md2 / "b.npz").write_text("x"); (md2 / "b.receipt.json").write_text(json.dumps({"fit": str(fd), "mode": "snap@17000"}))
    found, _, _ = EV.scan_fits([f"r2={sc / 'fitsX'}"], [str(md), f"zz={md2}"])
    names = sorted(r["unit"] for r in found)
    check("scan_fits: --modes-root inherits the fits-root tag; TAG=DIR overrides",
          names == ["genemean~r2", "genemean~r2#snap@16000", "genemean~zz#snap@17000"], names)
    found, _, _ = EV.scan_fits([str(sc / "fitsX")], [str(md)])
    check("scan_fits: untagged root -> untagged mode unit (round-2 behaviour)",
          sorted(r["unit"] for r in found) == ["genemean", "genemean#snap@16000"], [r["unit"] for r in found])
    shutil.rmtree(SCRATCH, ignore_errors=True)
    return finish("units", t0)


# ==========================================================================
# repair (round-3 adversarial review): matched control pairing, primary pairing, length reference, seed noise,
# fixedstage stage-blind generator, T2 'missing'
# ==========================================================================
def repair():
    t0 = time.monotonic()
    cl = EV.control_label
    pairs = {"pbulk,t.anneal@0.1~r3": "pbulk~r3", "pbulk,t.anneal@0.1~r3#snapens": "pbulk~r3#snapens",
             "pbulk,t.steps@60000~r3#snap@20000": "pbulk~r3", "pbulk,t.steps@60000~r3#snapema@20000:0.9995": "pbulk~r3#ema@0.9995",
             "pbulk,t.steps@60000~r3#snapens@5": "pbulk~r3#snapens", "pbulk,t.steps@60000~r3#snap@30000": "pbulk~r3#snap@30000",
             "nostage,pbulk~r3#snapens": "pbulk~r3#stageblind", "nostage,pbulk~r3": "pbulk~r3#condmix@1",
             "nostage,pbulk~r3#ema@0.9995": None, "nostage,pbulk~r3#biascal@snapens": None,
             "g.dispz,pbulk~r3#stagehalf [seed-ensemble n=2]": "pbulk~r3#stagehalf [seed-ensemble n=2]", "pbulk~r3#snapens": None,
             "genemean,t.anneal@0.1~r3#snap@15000": "genemean~r3#snap@15000"}
    got = {k: cl(k) for k in pairs}
    check("control_label: matched read-outs (snap@20000->default, snapema@20000:D->ema@D, snapens@5->snapens, nostage->blind)",
          got == pairs, got)
    prim = {("pbulk,t.anneal@0.1~r3", 3): "pbulk~r3#snapens", ("pbulk,t.anneal@0.1~r3#snapens", 3): None,
            ("g.dispz,pbulk~r3#snapens", 3): "pbulk~r3#snapens", ("g.dispz,pbulk~r3#stagehalf", 3): "pbulk~r3#stagehalf",
            ("g.dispz,pbulk~r3#biascal@stagehalf", 3): "pbulk~r3#biascal@stagehalf", ("nostage,pbulk~r3#snapens", 3): "pbulk~r3#stageblind",
            ("pbulk,t.ema@0.999~r3", 3): "pbulk~r3#snapens", ("pbulk~r3#snapens", 3): None}
    got = {k: EV.control_primary_label(*k) for k in prim}
    check("control_primary_label: arm primary vs control primary (snapens / stagehalf / biascal@stagehalf / stageblind)",
          got == prim, got)
    lr = {"pbulk,t.steps@60000~r3#ema@0.9995": "pbulk,t.steps@60000~r3#snapema@20000:0.9995",
          "t.steps@60000~r3": "t.steps@60000~r3#snap@20000", "t.steps@60000~r3#raw": "t.steps@60000~r3#snap@20000",
          "pbulk,t.steps@60000~r3#snapens@5": None, "pbulk~r3#ema@0.9995": None, "pbulk,t.steps@10000~r3": None}
    got = {k: EV.length_reference_label(k) for k in lr}
    check("length_reference_label: in-run 20k references of t.steps arms", got == lr, got)
    U = {("a", "p"): [_u(0, 0, pb_rmse=0.50), _u(1, 0, pb_rmse=0.52), _u(2, 0, pb_rmse=0.54)],
         ("c", "p"): [_u(0, 0, pb_rmse=0.55), _u(1, 0, pb_rmse=0.56), _u(2, 0, pb_rmse=0.57)]}
    se, src = EV.seed_noise_se(U, "a", "c", "p", [0, 1, 2])
    want = EV.SEED_SD_PRIOR["pb_rmse"] * np.sqrt(2 / 3)
    check("seed_noise_se: prior when the reference has one seed per fold", src == "round-2 prior" and abs(se - want) < 1e-12, (se, want, src))
    U[("c", "p")] += [_u(0, 1, pb_rmse=0.57), _u(1, 1, pb_rmse=0.58)]
    se, src = EV.seed_noise_se(U, "a", "c", "p", [0, 1, 2])
    s2 = np.mean([np.var([0.55, 0.57], ddof=1), np.var([0.56, 0.58], ddof=1)])
    check("seed_noise_se: pooled control seed variance once >= 2 seeds exist", src.startswith("control seeds") and abs(se - np.sqrt(3 * 2 * s2) / 3) < 1e-12,
          (se, src))
    fr = [{"unit": "fixedstage,pbulk~r3#snapens", "mode": "snapens",
           "config": {"r2_arms": {"fixedstage": {"info": {"stage_blind_generator": True}}}}},
          {"unit": "fixedstage,pbulk~r3#stagehalf", "mode": "stagehalf",
           "config": {"r2_arms": {"fixedstage": {"info": {"stage_blind_generator": True}}}}},
          {"unit": "fixedstage,pbulk~r3#stagehalf", "mode": "stagehalf",
           "config": {"r2_arms": {"fixedstage": {"info": {"stage_blind_generator": False}}}}},
          {"unit": "fixedstage,pbulk~r3#snapens", "mode": "snapens",
           "config": {"r2_arms": {"fixedstage": {"info": {"stage_blind_generator": False}}}}}]
    k = EV.unit_stage_kinds(fr)
    check("unit_stage_kinds: fixedstage with a zero table makes its units STAGE-BLIND (any fold), even under stagehalf",
          k == {"fixedstage,pbulk~r3#snapens": "stage-blind", "fixedstage,pbulk~r3#stagehalf": "stage-blind"}, k)
    U2 = {("x", "p"): [_u(0, 0, pb_rmse=0.5, pc_energy_rep=1.0, cell_energy_rep=1.0, pc_coverage80_rep=0.8,
                          meanlog_rmse_gene=0.01, nonzero_rmse_gene=0.01)],
          ("baseline:grand_mean", "p"): [_u(0, 0, pb_rmse=0.46)],
          ("baseline:resample", "p"): [_u(0, 0, pb_rmse=0.5, pc_energy_rep=2.0, cell_energy_rep=2.0, pc_coverage80_rep=0.7,
                                         meanlog_rmse_gene=0.1, nonzero_rmse_gene=0.1, logvar_rmse_gene=0.1)]}
    b = EV._target_block(U2, "x", "p", [0])
    check("T2 'missing' lists metrics the unit lacks (not passed)", b["T2"]["missing"] == ["logvar_rmse_gene"] and b["T2"]["n_pass"] == 5
          and b["T2"]["worst_miss"] is None, b["T2"]["missing"])
    return finish("repair", t0)


# ==========================================================================
# floor (planted truth)
# ==========================================================================
def _sim_world(rng, n_donors, G, sd_bio, stage_eff, n_cells=None, lib=2000.0, base=None):
    q0 = base if base is not None else np.exp(rng.normal(0, 1.2, G))
    stage = rng.integers(0, 4, n_donors)
    eff = rng.normal(0, 1, G)
    logq = np.log(q0)[None, :] + rng.normal(0, sd_bio, (n_donors, G)) + stage_eff * (stage[:, None] / 3.0) * eff[None, :]
    q = np.exp(logq); q /= q.sum(1, keepdims=True)
    n = np.full(n_donors, 400 if n_cells is None else n_cells)
    pb = rng.poisson(n[:, None] * lib * q).astype(np.float64)
    return pb, stage, q0, eff


def floor_test():
    t0 = time.monotonic()
    rng = np.random.default_rng(7)
    G = 400
    for stage_eff, label in [(0.0, "no stage effect"), (1.2, "stage effect")]:
        base = np.exp(np.random.default_rng(11).normal(0, 1.2, G))
        big, bst, _, eff0 = _sim_world(np.random.default_rng(99), 1, G, .3, stage_eff, base=base)
        # truth: Monte Carlo over a very large population of the same world (same eff vector)
        rng_t = np.random.default_rng(5)
        R = 20000
        st = rng_t.integers(0, 4, R)
        logq = np.log(base)[None, :] + rng_t.normal(0, .3, (R, G)) + stage_eff * (st[:, None] / 3.0) * eff0[None, :]
        q = np.exp(logq); q /= q.sum(1, keepdims=True)
        Y = MX.log_cp10k(rng_t.poisson(400 * 2000.0 * q).astype(np.float64))
        mu = Y.mean(0)
        truth_g = float(np.mean(np.sqrt(((Y - mu) ** 2).mean(1))))
        mus = {s: Y[st == s].mean(0) for s in range(4)}
        truth_s = float(np.mean([np.sqrt(((Y[i] - mus[st[i]]) ** 2).mean()) for i in range(R)]))
        A, Ar, As, Bg, Bs = [], [], [], [], []
        for rep in range(150):
            r = np.random.default_rng(1000 + rep)
            sti = np.r_[np.repeat(np.arange(4), 4), r.integers(0, 4, 21)]
            logq = np.log(base)[None, :] + r.normal(0, .3, (37, G)) + stage_eff * (sti[:, None] / 3.0) * eff0[None, :]
            qq = np.exp(logq); qq /= qq.sum(1, keepdims=True)
            Yd = MX.log_cp10k(r.poisson(400 * 2000.0 * qq).astype(np.float64))
            held = np.zeros(37, bool); held[r.permutation(37)[:12]] = True
            for s in range(4):          # each stage present in train and held-out
                idx = np.where(sti == s)[0]
                held[idx[0]] = False; held[idx[1]] = True
            fd = SimpleNamespace(Y=Yd, X=Yd, stage=sti, held=held)
            a = FL.route_a(fd, np.ones(37))
            A.append(a["pb_grand"]); Ar.append(a["pb_raw"]); As.append(a["pb_stage"])
            raw, g, s_ = FL.route_b(fd)
            Bg.append(np.nanmean(g[held])); Bs.append(np.nanmean(s_[held]))
        mA, mAr, mAs, mBg, mBs = map(np.mean, (A, Ar, As, Bg, Bs))
        se = np.std(A, ddof=1) / np.sqrt(len(A))
        check(f"floor route A recovers the oracle grand floor ({label})",
              abs(mA - truth_g) < max(3 * se, .25 * (mAr - truth_g)) and mAr > truth_g,
              f"truth {truth_g:.4f} A {mA:.4f} raw {mAr:.4f} se {se:.4f}")
        check(f"floor route B (LOO) recovers the oracle grand floor ({label})", abs(mBg - truth_g) < max(3 * se, .25 * (mAr - truth_g)),
              f"truth {truth_g:.4f} B {mBg:.4f}")
        ses = np.std(As, ddof=1) / np.sqrt(len(As))
        check(f"floor stage oracle (routes A and B) recovers the within-stage truth ({label})",
              abs(mAs - truth_s) < max(4 * ses, .01) and abs(mBs - truth_s) < max(4 * ses, .01),
              f"truth_stage {truth_s:.4f} A {mAs:.4f} B {mBs:.4f} (grand truth {truth_g:.4f})")
    # technical variance: cell bootstrap vs Poisson delta method
    r = np.random.default_rng(3)
    Gs, n = 200, 500
    q = np.exp(r.normal(0, 1, Gs)); q /= q.sum()
    lib = r.integers(800, 3000, n).astype(float)
    C = r.poisson(lib[:, None] * q[None, :]).astype(np.float32)
    real = SimpleNamespace(panel="sim", fold=0, cells=[sp.csr_matrix(C)])
    fd = FL.FoldData.__new__(FL.FoldData)
    fd.real, fd.donors, fd.tau2 = real, np.array(["simdonor"]), None
    tau2, _ = fd.technical(n_boot=200)
    pb = C.sum(0).astype(np.float64)
    cp = pb * 1e4 / pb.sum()
    # delta method (ratio estimator) for y_g = log1p(1e4 * pb_g / S) of a cell-resampled sum, S = sum_g pb_g:
    # var(pb_g / S) ~ sum over cells of (x_cg - r_g * L_c)^2 / S^2 with r_g = pb_g / S, L_c = cell library
    S = pb.sum(); rg = pb / S; Lc = C.sum(1).astype(np.float64)
    vr = ((C - rg[None, :] * Lc[:, None]) ** 2).sum(0) / S ** 2
    approx = float(np.mean(vr * 1e8 / (1 + cp) ** 2))
    check("floor technical: cell-bootstrap variance of y_dg matches the delta-method value (within 25%)",
          abs(tau2[0] / approx - 1) < .25, f"bootstrap {tau2[0]:.5f} delta {approx:.5f}")
    return finish("floor", t0)


# ==========================================================================
# synthetic end-to-end on the real fold-0 cache
# ==========================================================================
def _write_gen(path, arr, mode=None, extra=None, fit=None, receipt=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    a = {k: v for k, v in arr.items()}
    if extra:
        a.update(extra)
    if mode:
        a["mode"] = np.array(mode)
    np.savez(path, **a)
    if mode and receipt:
        EV.write_json(path.with_name(path.stem + ".receipt.json"), {"mode": mode, "fit": str(fit), "generation_device": "cpu"})


def _shift(arr, gshift):
    """Multiply every generated count of gene g by exp(gshift_g) (pseudobulk only; other arrays unchanged)."""
    out = dict(arr)
    out["pb_counts"] = (arr["pb_counts"].astype(np.float64) * np.exp(gshift)[None, None, :]).astype(np.float32)
    return out


def synthetic(baselines_from=None):
    t0 = time.monotonic()
    root = SCRATCH / "synth"
    shutil.rmtree(root, ignore_errors=True)
    fits = root / "fits_r3"; out = root / "results"
    real = EV.load_real(RUNS / "caches" / "hvg2000" / "fold0", "hvg2000", 0)
    gm, _ = BL.generate("grand_mean", real)
    rs, _ = BL.generate("resample", real)
    G = len(real.genes)
    zg = np.random.default_rng(1).normal(0, 1, G)

    def fit_dir(arm, cfg):
        d = fits / arm / "hvg2000" / "fold0" / "seed0"
        d.mkdir(parents=True, exist_ok=True)
        EV.write_json(d / "config.json", {"round": 3, "arm": arm, "flags": arm.split(","), "panel": "hvg2000", "fold": 0, "seed": 0,
                                          "updates": 20000, "recipe_total_steps": 20000,
                                          "snapshot_updates": [5000, 10000, 15000, 16000, 17000, 18000, 19000], **cfg})
        EV.write_json(d / "fit_receipt.json", {"device": "cpu", "status": "completed"})
        return d
    d = fit_dir("gmcopy", {})
    _write_gen(d / "generated.npz", gm)
    sched = {5000: .30, 10000: .20, 15000: .12, 17000: .08, 19000: .04}
    for u, a in sched.items():
        _write_gen(d / f"generated_snap@{u}.npz", _shift(gm, a * zg), f"snap@{u}", fit=d)
        _write_gen(d / f"generated_snapema@{u}:0.999.npz", _shift(gm, .5 * a * zg), f"snapema@{u}:0.999", fit=d)
    _write_gen(d / "generated_ema@0.999.npz", gm, "ema@0.999", fit=d)
    st = gm["stage"].astype(int)
    blind = np.zeros((len(st), 32, 7), np.float32); blind[:, :, 3] = 1.0 / 3
    half = np.zeros((len(st), 32, 7), np.float32); half[:, :, 3] = (0.5 * st / 3 + .25)[:, None]
    _write_gen(d / "generated_mixall.npz", gm, "mixall", extra={"conditions": blind}, fit=d)
    _write_gen(d / "generated_halfmix.npz", gm, "halfmix", extra={"conditions": half}, fit=d)
    _write_gen(d / "generated_stagehalf.npz", gm, "stagehalf", fit=d)
    _write_gen(d / "generated_biascal+snapens.npz", gm, "biascal+snapens", fit=d)
    d2 = fit_dir("gmshift", {})
    _write_gen(d2 / "generated.npz", _shift(gm, .3 * zg))      # per-gene log shift (a constant would cancel in CP10k)
    d3 = fit_dir("rscopy", {})
    _write_gen(d3 / "generated.npz", rs)
    d4 = fit_dir("genemean", {})
    _write_gen(d4 / "generated.npz", _shift(gm, .1 * zg))
    d5 = fit_dir("genemean,t.anneal@0.1", {})
    _write_gen(d5 / "generated.npz", _shift(gm, .05 * zg))
    d6 = fit_dir("genemean,t.ema@0.9995", {"native_ema_decay": 0.9995})
    _write_gen(d6 / "generated.npz", gm)
    _write_gen(d6 / "generated_raw.npz", _shift(gm, .1 * zg), "raw", fit=d6)
    if baselines_from and (Path(baselines_from) / "baselines" / "hvg2000" / "fold0").exists():
        shutil.copytree(Path(baselines_from) / "baselines" / "hvg2000" / "fold0", out / "baselines" / "hvg2000" / "fold0")
    EV.main(["--fits-root", f"r3={fits}", "--out", str(out), "--panels", "hvg2000", "--folds", "0", "--no-ensembles"])
    agg = json.loads((out / "aggregate.json").read_text())
    rows = {r["arm"]: r for r in agg["rows"]}
    tg = json.loads((out / "targets.json").read_text())
    cv = json.loads((out / "curves.json").read_text())
    gmr, rsr = rows["baseline:grand_mean"], rows["baseline:resample"]
    t = tg["units"]["gmcopy~r3|hvg2000"]["groups"]["folds0-2"]
    check("synthetic: copy of grand_mean -> pb_rmse equal to the baseline, T1 pass with margin -0.005",
          abs(rows["gmcopy~r3"]["pb_rmse"] - gmr["pb_rmse"]) < 1e-12 and t["T1"]["pass"] and abs(t["T1"]["margin"] + .005) < 1e-12,
          (rows["gmcopy~r3"]["pb_rmse"], gmr["pb_rmse"], t["T1"]["margin"]))
    t = tg["units"]["gmshift~r3|hvg2000"]["groups"]["folds0-2"]
    check("synthetic: per-gene shifted copy (0.3 z_g) fails T1 by (its pb_rmse - grand mean - 0.005)",
          not t["T1"]["pass"] and abs(t["T1"]["margin"] - (rows["gmshift~r3"]["pb_rmse"] - gmr["pb_rmse"] - .005)) < 1e-12
          and rows["gmshift~r3"]["pb_rmse"] > gmr["pb_rmse"] + .02, t["T1"])
    t = tg["units"]["rscopy~r3|hvg2000"]["groups"]["folds0-2"]["T2"]
    check("synthetic: copy of resample passes T2 6/6 with every margin exactly 0",
          t["n_pass"] == 6 and all(abs(m["margin"]) < 1e-12 for m in t["metrics"].values()), t)
    t = tg["units"]["gmcopy~r3|hvg2000"]["groups"]["folds0-2"]["T2"]
    check("synthetic: grand_mean copy (independent-gene NB cells) does not pass T2; margins reported per metric",
          t["n_pass"] < 6 and all("margin" in v for v in t["metrics"].values()), {m: (v.get("pass"), v.get("margin")) for m, v in t["metrics"].items()})
    s = cv["per_fit"]["gmcopy~r3|hvg2000|fold0|seed0"]["series"]
    pbr = s["raw"]["values"]["pb_rmse"]
    check("synthetic curves: raw series 5000..20000 monotone decreasing as planted, best = 20000, late slope < 0",
          s["raw"]["updates"] == [5000, 10000, 15000, 17000, 19000, 20000] and all(a > b for a, b in zip(pbr, pbr[1:]))
          and s["raw"]["best_update_pb_rmse"] == 20000 and s["raw"]["late_slope_per_1000_pb_rmse"] < 0, pbr)
    pe = s["ema@0.999"]["values"]["pb_rmse"]
    check("synthetic curves: snapema series below the raw one at every snapshot (half the planted shift)",
          s["ema@0.999"]["updates"] == [5000, 10000, 15000, 17000, 19000, 20000] and all(a < b for a, b in zip(pe[:-1], pbr[:-1])), pe)
    se6 = cv["per_fit"]["genemean,t.ema@0.9995~r3|hvg2000|fold0|seed0"]["series"]
    check("synthetic curves: t.ema fit default -> native_ema@0.9995, raw mode -> raw",
          set(se6) == {"native_ema@0.9995", "raw"}, list(se6))
    sm = {r["arm"]: r.get("stage_mode") for r in agg["rows"]}
    check("synthetic stage kinds: constant ADNC -> stage-blind, half -> stage-shrunk, stagehalf -> stage-shrunk, resample_blind blind",
          sm.get("gmcopy~r3#mixall") == "stage-blind" and sm.get("gmcopy~r3#halfmix") == "stage-shrunk"
          and sm.get("gmcopy~r3#stagehalf") == "stage-shrunk" and sm.get("baseline:resample_blind") == "stage-blind"
          and sm.get("gmcopy~r3") is None, sm)
    check("synthetic: stage-blind unit has no stage metrics; listed separately in targets",
          rows["gmcopy~r3#mixall"].get("xfit_score") is None and rows["gmcopy~r3#mixall"].get("hn_spearman_vs_train") is None
          and "gmcopy~r3#mixall|hvg2000" in tg["lists"]["T1a_pass_folds0-2_stage_blind"], tg["lists"]["T1a_pass_folds0-2_stage_blind"])
    check("synthetic: biascal+snapens is a T1b unit", tg["units"]["gmcopy~r3#biascal+snapens|hvg2000"]["T1_variant"] == "T1b")
    pc = agg["paired"].get("genemean,t.anneal@0.1~r3|hvg2000", {}).get("vs_control")
    check("synthetic: vs_control pairs the t.anneal arm with genemean~r3",
          pc is not None and pc["reference"] == "genemean~r3" and pc["metrics"]["pb_rmse"]["n"] == 1
          and abs(pc["metrics"]["pb_rmse"]["raw_mean_difference"] - (rows["genemean,t.anneal@0.1~r3"]["pb_rmse"] - rows["genemean~r3"]["pb_rmse"])) < 1e-12,
          pc and pc["metrics"]["pb_rmse"])
    lb = (out / "leaderboard.txt").read_text()
    check("synthetic: leaderboard lists units with T1/T2/stage columns and references",
          "gmcopy~r3" in lb and "baseline:grand_mean" in lb and "BLIND" in lb and "T1a PASS" in lb, lb[:200])
    check("synthetic: primary flags (t.anneal default P, t.ema default P, gmcopy default not P)",
          tg["units"]["genemean,t.anneal@0.1~r3|hvg2000"]["primary_readout"] == "P"
          and tg["units"]["genemean,t.ema@0.9995~r3|hvg2000"]["primary_readout"] == "P"
          and tg["units"]["gmcopy~r3|hvg2000"]["primary_readout"] is None)
    shutil.rmtree(root, ignore_errors=True)
    return finish("synthetic", t0)


# ==========================================================================
# reproduce results_r2_interim
# ==========================================================================
IGNORE = {"evaluator_version", "score_seconds", "generation_seconds", "seconds"}


def compare(a, b, path, stats, tol=1e-8):
    if isinstance(a, dict):
        if not isinstance(b, dict):
            stats["mismatch"].append(path); return
        for k, v in a.items():
            if k in IGNORE:
                continue
            if k not in b:
                stats["missing"].append(f"{path}/{k}"); continue
            compare(v, b[k], f"{path}/{k}", stats, tol)
    elif isinstance(a, list):
        if not isinstance(b, list) or len(a) != len(b):
            stats["mismatch"].append(path); return
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, f"{path}[{i}]", stats, tol)
    elif isinstance(a, bool) or a is None or isinstance(a, str):
        if a != b:
            stats["mismatch"].append(f"{path}: {a!r} vs {b!r}")
    elif isinstance(a, (int, float)):
        if not isinstance(b, (int, float)) or isinstance(b, bool):
            stats["mismatch"].append(f"{path}: {a!r} vs {b!r}"); return
        dd = abs(float(a) - float(b))
        stats["n"] += 1
        stats["max"] = max(stats["max"], dd)
        if dd > tol:
            stats["mismatch"].append(f"{path}: {a} vs {b} (diff {dd:.2e})")
    else:
        stats["mismatch"].append(f"{path}: type {type(a)}")


def reproduce(out, ref=None):
    t0 = time.monotonic()
    out = Path(out); ref = Path(ref or RUNS / "results_r2_interim")
    tot = {}
    for sub, kind in [("fits", "fit"), ("baselines", "baseline"), ("real", "real")]:
        for a in sorted((ref / sub).rglob("*.json")):
            b = out / a.relative_to(ref)
            st = tot.setdefault(kind, {"files": 0, "n": 0, "max": 0.0, "mismatch": [], "missing": [], "absent": []})
            if not b.exists():
                st["absent"].append(str(b)); continue
            st["files"] += 1
            compare(json.loads(a.read_text()), json.loads(b.read_text()), str(a.relative_to(ref)), st)
        for a in sorted((ref / sub).rglob("*.npz")):
            b = out / a.relative_to(ref)
            st = tot.setdefault(kind + "_npz", {"files": 0, "n": 0, "max": 0.0, "mismatch": [], "missing": [], "absent": []})
            if not b.exists():
                st["absent"].append(str(b)); continue
            st["files"] += 1
            with np.load(a, allow_pickle=False) as fa, np.load(b, allow_pickle=False) as fb:
                for k in fa.files:
                    if k not in fb.files:
                        st["missing"].append(f"{a}:{k}"); continue
                    x, y = fa[k], fb[k]
                    if x.dtype.kind in "fc":
                        dd = float(np.nanmax(np.abs(x.astype(np.float64) - y.astype(np.float64)))) if x.size else 0.0
                        st["n"] += x.size; st["max"] = max(st["max"], dd)
                        if x.shape != y.shape or dd > 1e-8 or not np.array_equal(np.isnan(x), np.isnan(y)):
                            st["mismatch"].append(f"{a}:{k}")
                    elif not np.array_equal(x, y):
                        st["mismatch"].append(f"{a}:{k}")
    for kind, st in tot.items():
        check(f"reproduce {kind}: every interim value equal to 1e-8",
              st["files"] > 0 and not st["absent"] and not st["mismatch"] and not st["missing"],
              f"files={st['files']} values={st['n']} max_abs_diff={st['max']:.2e} absent={len(st['absent'])} {st['absent'][:2]} "
              f"mismatch={st['mismatch'][:3]} missing={st['missing'][:3]}")
    a1 = json.loads((ref / "aggregate.json").read_text()); a2 = json.loads((out / "aggregate.json").read_text())
    rows2 = {(r["arm"], r["panel"]): r for r in a2["rows"]}
    st = {"n": 0, "max": 0.0, "mismatch": [], "missing": []}
    for r in a1["rows"]:
        r2 = rows2.get((r["arm"], r["panel"]))
        if r2 is None:
            st["missing"].append(r["arm"]); continue
        compare(r, r2, f"row {r['arm']}", st)
    check("reproduce aggregate rows: every interim row value equal to 1e-8", not st["mismatch"] and not st["missing"],
          f"values={st['n']} max={st['max']:.2e} mismatch={st['mismatch'][:3]} missing={st['missing'][:3]}")
    for key in ["paired", "xfit", "pooled_across_folds", "noise_ceiling", "dam_paired_panels", "dam_present_range"]:
        st = {"n": 0, "max": 0.0, "mismatch": [], "missing": []}
        compare(a1[key], a2.get(key), key, st)
        check(f"reproduce aggregate {key}: every interim value equal to 1e-8", not st["mismatch"] and not st["missing"],
              f"values={st['n']} max={st['max']:.2e} mismatch={st['mismatch'][:3]} missing={st['missing'][:3]}")
    for key in ["stage_blind_units", "stage_mixture_units"]:
        check(f"reproduce {key}: same labels", set(a1[key]) <= set(a2[key]) and
              not (set(a2[key]) - set(a1[key])) & {r["arm"] for r in a1["rows"]}, (a1[key], a2[key]))
    return finish("reproduce", t0)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "units"
    if cmd == "units":
        sys.exit(1 if units() else 0)
    if cmd == "floor":
        sys.exit(1 if floor_test() else 0)
    if cmd == "repair":
        sys.exit(1 if repair() else 0)
    if cmd == "synthetic":
        bf = None
        if "--baselines-from" in sys.argv:
            bf = sys.argv[sys.argv.index("--baselines-from") + 1]
        sys.exit(1 if synthetic(bf) else 0)
    if cmd == "reproduce":
        sys.exit(1 if reproduce(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None) else 0)
    raise SystemExit(f"unknown subcommand {cmd}")
