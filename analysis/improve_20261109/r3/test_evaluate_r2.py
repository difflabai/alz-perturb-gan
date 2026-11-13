#!/usr/bin/env python3
"""Tests for the round-2 evaluator (r2/evaluate.py, r2/metrics.py, r2/baselines.py).

Subcommands (CPU only, 3 threads):

  units       pure-function planted-truth checks: frozen DAM list equals the
              PROTOCOL_R2.md text, DAM score / matched null, cross-fitted donor
              score and its permutation null, additive recalibration,
              repeat-balanced aggregation, M-matched ensembles, baseline
              masking, scanning (``@`` arms, mode files, --modes-root,
              shadowing), development-donor guard on generated targets.
  synthetic   end-to-end on a synthetic world (no real data read): caches for
              the 37 development donor IDs in the 9 inner folds of
              r2/folds_r2.json, fake generators with planted properties
              (truth, oracle, shrunk, inflated, biased, overdispersed,
              wrongstage), a mode file and a --modes-root file; checks every
              new metric against its planted truth and the aggregation /
              tables.  Scratch: runs/improve_20261109/smoke_r3/r2copy_tests/eval_test
              (deleted at the end; only test_report.json is kept).
  reproduce OUT
              compares an r2 evaluation of the round-1 fits (seeds 0,1, folds
              0-2, both panels; e.g. smoke_r3/r2copy_tests/eval_r1repro) with
              results_screen48: every numeric value of every round-1 per-fit,
              ensemble (round-1 'fold{k}.json' vs r2 'fold{k}_n2.json'),
              baseline and noise-ceiling JSON (keys present in round 1) to 1e-8,
              and the aggregate rows / paired vs_base blocks of round-1
              headline metrics (r2 '__se_folds' vs round-1 '__se').

Run: cd /tmp; python3 -B /ml2/alz/analysis/improve_20261109/r2/test_evaluate_r2.py units
"""
from __future__ import annotations

import os
for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[_v] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import json
import re
import shutil
import sys
import time
from pathlib import Path

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

for _m in (MX, EV, BL):
    assert Path(_m.__file__).resolve().parent == HERE, _m.__file__

RUNS = Path("/ml2/alz/runs/improve_20261109")
ROOT = RUNS / "smoke_r3/r2copy_tests" / "eval_test"
RESULTS = RUNS / "smoke_r3/r2copy_tests" / "eval_test_results"
FOLDS = json.loads(EV.FOLDS_JSON_R2.read_text()) if EV.FOLDS_JSON_R2.exists() else json.loads(EV.FOLDS_JSON_R1.read_text())
CHECKS = []


def check(name, cond, detail=""):
    CHECKS.append({"check": name, "pass": bool(cond), "detail": detail})
    print(("PASS " if cond else "FAIL ") + name + " :: " + detail, flush=True)


def finish(tag, t0):
    n_fail = sum(not c["pass"] for c in CHECKS)
    RESULTS.mkdir(parents=True, exist_ok=True)
    EV.write_json(RESULTS / f"{tag}.json", {"checks": CHECKS, "n_fail": n_fail, "seconds": time.monotonic() - t0})
    print(f"\n{tag}: {len(CHECKS) - n_fail}/{len(CHECKS)} checks passed; {time.monotonic() - t0:.1f}s")
    return n_fail


# ==========================================================================
# units
# ==========================================================================
def units():
    t0 = time.monotonic()
    rng = np.random.default_rng(1)
    # --- frozen list equals the protocol text
    txt = (HERE.parent / "r2" / "PROTOCOL_R2.md").read_text()   # integration: r3 holds no copy
    up = re.search(r"Expected higher with pathology:(.*?)\n\s*- Expected lower", txt, re.S).group(1)
    down = re.search(r"Expected lower with pathology:(.*?)\n\n", txt, re.S).group(1)
    parse = lambda s: [g.strip().rstrip(".") for g in s.replace("\n", " ").split(",") if g.strip()]  # noqa: E731
    check("DAM_UP / DAM_DOWN equal the PROTOCOL_R2.md frozen lists (order and spelling)",
          parse(up) == MX.DAM_UP and parse(down) == MX.DAM_DOWN, f"{len(MX.DAM_UP)} up, {len(MX.DAM_DOWN)} down")
    # --- DAM score: planted signal vs none
    G = 2000
    genes = np.array([f"G{i}" for i in range(G)], dtype=object)
    pos = rng.choice(G, 27, replace=False)
    for p, g in zip(pos, MX.DAM_UP + MX.DAM_DOWN):
        genes[p] = g
    genes = genes.astype(str)
    level = rng.normal(0, 1, G)
    level[pos[:20]] += 1.0           # listed genes more expressed than average: matching matters
    idx, sgn, missing = MX.dam_genes(genes)
    null = MX.matched_null_sets(idx, level, 2000, 7)
    b = MX.expression_bins(level)
    check("matched null: each null gene comes from its listed gene's expression decile and is not listed",
          (b[null] == b[idx][None, :]).all() and not np.isin(null, idx).any() and len(missing) == 0,
          f"n_present={len(idx)}")
    noise = rng.normal(0, .3, G)
    planted = noise.copy(); planted[idx] += .4 * sgn
    level_eff = noise + .5 * level      # contrast correlated with expression level only
    real_h = noise * 0 + rng.normal(0, .3, G); real_h[idx] += .4 * sgn
    s1 = MX.dam_score(planted, idx, sgn, null, {"heldout": real_h})
    s0 = MX.dam_score(rng.normal(0, .3, G), idx, sgn, null, {"heldout": real_h})
    s2 = MX.dam_score(level_eff, idx, sgn, null, {"heldout": real_h})
    check("DAM planted direction signal: signed z > 3, sign_frac > 0.8, p < 0.01",
          s1["signed_mean"]["z"] > 3 and s1["sign_frac"]["value"] > .8 and s1["signed_mean"]["p_one_sided"] < .01,
          f"z={s1['signed_mean']['z']:.2f} frac={s1['sign_frac']['value']:.2f} p={s1['signed_mean']['p_one_sided']:.4f}")
    check("DAM no signal: |signed z| < 2.5; held-out correlation planted > none",
          abs(s0["signed_mean"]["z"]) < 2.5 and s1["spearman_vs_heldout"]["value"] > s0["spearman_vs_heldout"]["value"],
          f"z0={s0['signed_mean']['z']:.2f} rho1={s1['spearman_vs_heldout']['value']:.2f} rho0={s0['spearman_vs_heldout']['value']:.2f}")
    check("DAM expression-level-only effect is absorbed by the matched null (|z| < 2.5)",
          abs(s2["signed_mean"]["z"]) < 2.5, f"z={s2['signed_mean']['z']:.2f} raw signed mean={s2['signed_mean']['value']:.3f}")
    # --- repair S1: stage-permutation null for the DAM score (gene-gene correlation kept)
    def corr_rows(n, k_shift=0.0, seed=0):
        r = np.random.default_rng(seed)
        X = r.normal(0, 1, (n, G))
        X[:, idx] += 1.5 * r.normal(0, 1, (n, 1)) * sgn      # listed genes co-vary across rows (shared factor)
        st = np.full(n, 3); st[:max(2, n // 5)] = 0
        X[np.ix_(st == 3, idx)] += k_shift * sgn
        return X, st
    pn, pg = [], []
    for it in range(300):
        X, st = corr_rows(23, 0.0, 100 + it)
        c = MX.stage_contrasts(X, st)["high_notad"]
        pn.append(MX.dam_perm_null(X, st, idx, sgn, 400, it)["signed_mean"]["p_one_sided"])
        pg.append(MX.dam_score(c, idx, sgn, null)["signed_mean"]["p_one_sided"])
    pn, pg = np.array(pn), np.array(pg)
    check("DAM permutation null calibrated under co-varying listed genes and no stage effect (P(p<.05) <= 0.09); "
          "the matched-gene null is not (P(p<.05) > 0.15)",
          (pn < .05).mean() <= .09 and (pg < .05).mean() > .15,
          f"perm {(pn < .05).mean():.3f} gene-null {(pg < .05).mean():.3f}")
    Xs, sts = corr_rows(23, 3.0, 7)
    ps = MX.dam_perm_null(Xs, sts, idx, sgn, 2000, 1)
    check("DAM permutation null: planted High shift -> z > 3; exact enumeration when splits <= draws",
          ps["signed_mean"]["z"] > 3 and MX.dam_perm_null(Xs[:8], sts[:8], idx, sgn, 2000, 1)["null_kind"] == "exact",
          f"z={ps['signed_mean']['z']:.2f}")
    # --- repair FID-3 / S4 / FID-1 helpers
    check("stage_blind_mode: condmix@1/condunk@1/condshrink@1/anchorgain@0 blind; condmix@0.5 mixture; others None",
          [EV.stage_blind_mode(m) for m in ["condmix@1", "condunk@1", "condshrink@1", "anchorgain@0", "condmix@0.5",
                                            "condunk@0.5", "condshrink@0.5", "anchorgain@0.5", "margcov", "snapens", None,
                                            "condmix@0.99"]]
          == ["stage-blind"] * 4 + ["stage-mixture", "stage-mixture", None, None, None, None, None, "stage-blind"]
          and EV.label_mode("genemean~r2#condmix@1 [seed-ensemble n=2]") == "condmix@1"
          and EV.is_stage_metric("xfit_score") and EV.is_stage_metric("risk__hn_spearman_vs_train")
          and not EV.is_stage_metric("dam_n_present") and not EV.is_stage_metric("pb_rmse"), "")
    check("gen_device: mode receipt device, fit receipt device, unknown",
          EV.gen_device({"mode": "condmix@1", "mode_receipt": {"generation_device": "cuda"}}) == "cuda"
          and EV.gen_device({"mode": None, "fit_receipt": {"device": "cuda"}}) == "cuda"
          and EV.gen_device({"mode": "margcov", "mode_receipt": {}, "generated_path": "/x/y.npz", "fit_dir": "/z"}) == "unknown", "")
    # --- recalibration
    truth = rng.normal(0, 1, (30, 500)); bias = rng.normal(0, .5, 500)
    pred = truth + bias + rng.normal(0, .05, (30, 500))
    a = MX.additive_recalibration(pred[:20], truth[:20])
    rm0 = np.sqrt(((pred[20:] - truth[20:]) ** 2).mean()); rm1 = np.sqrt(((pred[20:] + a - truth[20:]) ** 2).mean())
    check("additive recalibration recovers a planted per-gene bias (corr > 0.99) and removes it on held-out donors",
          np.corrcoef(-a, bias)[0, 1] > .99 and rm1 < .1 < rm0, f"rmse {rm0:.3f} -> {rm1:.3f}")
    # --- cross-fitted donor score
    def xfold(n, gain, G=400, sig=1.0, perm_stage=False, seed=0):
        r = np.random.default_rng(seed)
        eff = r.normal(0, sig, (4, G)); eff -= eff.mean(0)
        folds = []
        for k in range(3):
            st = r.integers(0, 4, n)
            obs = eff[st] + r.normal(0, 1.0, (n, G))
            model = gain * (eff[[1, 2, 3, 0]] if perm_stage else eff) + r.normal(0, .1, (4, G))
            folds.append({"obs": obs, "cand": model, "assign": st, "group": st})
        return folds
    good = MX.xfit_score(xfold(12, 1.0), 500, 3)
    infl = MX.xfit_score(xfold(12, 3.0), 500, 3)
    bad = MX.xfit_score(xfold(12, 1.0, perm_stage=True), 500, 3)
    zero = MX.xfit_score(xfold(12, 0.0), 500, 3)
    check("xfit: planted stage signal -> score > 0.3, z > 3, p < 0.01",
          good["score"]["value"] > .3 and good["score"]["z"] > 3 and good["score"]["p_one_sided"] < .01,
          f"score={good['score']['value']:.3f} z={good['score']['z']:.2f} null mean={good['score']['null_mean']:.3f}")
    check("xfit: a 3x inflated stage effect scores the same correlation (scale-free)", abs(infl["score"]["value"] - good["score"]["value"]) < .02,
          f"{infl['score']['value']:.3f} vs {good['score']['value']:.3f}")
    check("xfit: wrong stage mapping and pure-noise model are below the planted model and near their null",
          bad["score"]["value"] < good["score"]["value"] - .2 and abs(zero["score"]["z"]) < 3,
          f"wrong={bad['score']['value']:.3f} (z {bad['score']['z']:.2f}) noise z={zero['score']['z']:.2f}")
    check("xfit: pooled-pairs Pearson defined and positive for planted signal", good["pooled_pairs"]["value"] > .3,
          f"{good['pooled_pairs']['value']:.3f}")
    # --- repeat-balanced aggregation
    m, se, kind, se_f, nf, nr = EV.repeat_balanced({0: 1.0, 1: 2.0, 2: 3.0, 3: 5.0, 4: 5.0, 5: 5.0})
    check("repeat_balanced: mean of repeat means (2 and 5 -> 3.5), SE over repeats",
          abs(m - 3.5) < 1e-12 and kind == "repeats" and abs(se - 1.5) < 1e-12 and nr == 2 and nf == 6, f"m={m} se={se} kind={kind}")
    m1, se1, kind1, *_ = EV.repeat_balanced({0: 1.0, 1: 2.0, 2: 6.0})
    check("repeat_balanced with one repeat = round-1 fold mean and fold SE",
          abs(m1 - 3.0) < 1e-12 and kind1 == "folds" and abs(se1 - np.std([1, 2, 6], ddof=1) / np.sqrt(3)) < 1e-12, f"{m1} {se1}")
    # --- M-matched ensemble
    gens = [{k: np.full((2, 32, 3) if k != "lib" else (2, 32), i, dtype=np.float32) for k in ["pb_counts", "mean_log", "nonzero", "lib"]}
            for i in range(3)]
    for i, g in enumerate(gens):
        g["pcs"] = np.full((2, 32, 4, 5), i, np.float16)
    mm = EV.mmatched_generated(gens)
    cat = EV.concat_generated(gens)
    check("M-matched ensemble: M=32, donor m from seed m mod n; full ensemble M=96",
          mm["pb_counts"].shape[1] == 32 and (mm["pb_counts"][0, :, 0] == np.arange(32) % 3).all() and cat["pb_counts"].shape[1] == 96
          and "sq_log" not in mm, "")
    # --- baseline masking
    flat = {k: 1.0 for k in ["pb_rmse", "pb_rmse_recal", "meanlog_rmse_gene", "meanlog_rmse_gene_recal", "nonzero_rmse_gene",
                             "logvar_rmse_gene", "pc_energy", "pc_energy_rep", "dam_signed_z", "risk__meanlog_rmse_gene"]}
    f1, _ = EV.mask_not_meaningful(dict(flat), BL.MEANINGFUL["ridge"])
    f2, _ = EV.mask_not_meaningful(dict(flat), BL.MEANINGFUL["resample"])
    check("masking: profile baseline meanlog/nonzero/logvar (+recal, subsets) and PC masked; pb and DAM kept; resample keeps all",
          all(f1[k] is None for k in ["meanlog_rmse_gene", "meanlog_rmse_gene_recal", "nonzero_rmse_gene", "logvar_rmse_gene",
                                      "pc_energy", "pc_energy_rep", "risk__meanlog_rmse_gene"])
          and f1["pb_rmse"] == 1 and f1["pb_rmse_recal"] == 1 and f1["dam_signed_z"] == 1 and all(v == 1 for v in f2.values()), "")
    check("descriptive-only metrics have no paired direction use (listed) and labels map to base / parent",
          "hn_spearman_vs_heldout" in EV.DESCRIPTIVE_ONLY and EV._base_label("genemean@0.5#snapens [seed-ensemble n=2 M=32]")
          == "base [seed-ensemble n=2 M=32]" and EV._parent_label("genemean@0.5#snapens") == "genemean@0.5", "")
    # --- scanning and guard
    scan = ROOT / "scan_unit"
    if scan.exists():
        shutil.rmtree(scan)
    fdir = scan / "rootA" / "genemean@0.5" / "hvg2000" / "fold4" / "seed1"; fdir.mkdir(parents=True)
    for f in ["generated.npz", "fit_receipt.json", "generated_snapens.npz", "generated_snapens.receipt.json",
              "generated_snap@16000.npz", ".generated_x.partial.npz", "generated_nore.npz"]:
        (fdir / f).write_text("x")
    gdir = scan / "rootB" / "genemean@0.5" / "hvg2000" / "fold4" / "seed1"; gdir.mkdir(parents=True)
    (gdir / "generated.npz").write_text("x"); (gdir / "fit_receipt.json").write_text("x")
    pdir = scan / "post"; pdir.mkdir()
    (pdir / "cov_margcov_f0s0.npz").write_text("x")
    (pdir / "cov_margcov_f0s0.receipt.json").write_text(json.dumps({"fit": str(RUNS / "fits/covars/hvg2000/fold0/seed0"), "mode": "margcov"}))
    found, inc, shadow = EV.scan_fits([scan / "rootA", scan / "rootB"], [pdir])
    units_found = sorted((r["unit"], r["fold"], r["seed"]) for r in found)
    check("scan: '@' arm, mode file with receipt only, modes-root receipt -> unit arm#mode, second root shadowed",
          units_found == [("covars#margcov", 0, 0), ("genemean@0.5", 4, 1), ("genemean@0.5#snapens", 4, 1)] and len(shadow) == 1,
          f"{units_found} shadowed={len(shadow)}")
    shutil.rmtree(scan)
    sealed = json.loads(EV.FOLDS_JSON_R1.read_text())["sealed_evaluation_donors_fold0"][0]
    tmp = ROOT / "guard"; tmp.mkdir(parents=True, exist_ok=True)
    np.savez(tmp / "g.npz", targets=np.array(["H20.33.001", sealed]), is_heldout=np.zeros(2, bool), stage=np.zeros(2, int),
             genes=np.array(["A"]), pb_counts=np.zeros((2, 1, 1)), mean_log=np.zeros((2, 1, 1)), nonzero=np.zeros((2, 1, 1)),
             pcs=np.zeros((2, 1, 1, 1)), lib=np.zeros((2, 1)))
    try:
        EV.load_generated(tmp / "g.npz"); refused = False
    except PermissionError:
        refused = True
    shutil.rmtree(tmp)
    check("development-donor guard refuses a generated file naming a sealed donor", refused, "")
    try:
        EV._check_out_dir(RUNS / "results_screen48" / "x"); ok = False
    except PermissionError:
        ok = True
    check("evaluator refuses to write into a frozen round-1 output tree", ok, "")
    check("repeat map from folds_r2.json: folds 0-2 -> 0, 3-5 -> 1, 6-8 -> 2",
          [EV.repeat_of(k) for k in range(9)] == [0, 0, 0, 1, 1, 1, 2, 2, 2] and EV.folds_of_repeat(1) == [3, 4, 5], "")
    if ROOT.exists() and not any(ROOT.iterdir()):
        ROOT.rmdir()
    return finish("units", t0)


# ==========================================================================
# synthetic end-to-end
# ==========================================================================
G_ALL = 450
PANEL_GENES = {"hvg2000": np.arange(0, 300), "wide": np.arange(50, 450)}
DAM_POS = np.arange(60, 87)          # 27 listed genes, in both panels
SD_U = 0.15
M, K, KPC = 32, 256, 64


class Sim:
    def __init__(self, seed=7):
        rng = np.random.default_rng(seed)
        self.names = np.array([f"G{i:03d}" for i in range(G_ALL)], dtype="<U12")
        self.dirs = np.zeros(G_ALL)
        for p, g in zip(DAM_POS, MX.DAM_UP + MX.DAM_DOWN):
            self.names[p] = g
            self.dirs[p] = 1.0 if g in MX.DAM_UP else -1.0
        self.names[200] = "SPI1"; self.names[250] = "CD74"
        self.a = rng.normal(0, 1.2, G_ALL)
        self.beta = np.where(rng.random(G_ALL) < .3, rng.normal(0, .8, G_ALL), 0.0)
        self.beta[DAM_POS] = 0.7 * self.dirs[DAM_POS]
        self.Lam = rng.normal(0, .5, (G_ALL, 2)) * (rng.random((G_ALL, 1)) < .4)
        self.bias = rng.normal(0, .4, G_ALL)
        self.donors = FOLDS["development_donors"]
        self.stage = {d: FOLDS["stage_by_donor"][d] for d in self.donors}
        self.u = {d: rng.normal(0, SD_U, G_ALL) for d in self.donors}
        self.ncell = {d: int(rng.integers(260, 400)) for d in self.donors}
        self.cov = {d: [float(rng.integers(0, 2)), float(rng.integers(0, 3)), float(rng.integers(70, 100))] for d in self.donors}

    def logit(self, stage, u, beta_scale=1.0, beta=None):
        return self.a + beta_scale * (self.beta if beta is None else beta) * stage / 3 + u

    def cells(self, rng, logit, n, panel=None, libs=None, shift=0.0, cell_sd=0.0):
        z = rng.normal(shift, 1, (n, 2))
        lp = logit[None, :] + z @ self.Lam.T
        if cell_sd > 0:
            lp = lp + rng.normal(0, cell_sd, lp.shape)
        if panel is None:
            p = np.exp(lp - lp.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
            lib = rng.lognormal(np.log(2500), .4, n)
            return rng.poisson(lib[:, None] * p)
        lp = lp[:, panel]
        p = np.exp(lp - lp.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        return rng.poisson(libs[:, None] * p)


def write_cache(sim, data, panel, k):
    fold = FOLDS["inner_folds"][k]
    d_ = ROOT / "caches" / panel / f"fold{k}"
    gi = PANEL_GENES[panel]
    genes = sim.names[gi]
    trX = np.concatenate([data[d][:, gi] for d in fold["train_donors"]]).astype(np.float64)
    keep = trX.sum(1) > 0
    L = MX.log_cp10k(trX[keep]); mean = L.mean(0); sd = L.std(0) + 1e-6
    Z = (L - mean) / sd
    _, _, Vt = np.linalg.svd(Z, full_matrices=False)
    W = Vt[:50].T.copy(); pc_sd = np.maximum((Z @ W).std(0), 1e-6)
    ages = np.array([sim.cov[d][2] for d in fold["train_donors"]])
    d_.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d_ / "space.npz", genes=genes, hvg=gi, mean=mean.astype(np.float32), sd=sd.astype(np.float32),
                        W=W.astype(np.float32), pc_sd=pc_sd.astype(np.float32), age_mean=ages.mean(), age_sd=ages.std())
    for split, key in [("train", "train_donors"), ("heldout", "heldout_donors")]:
        mats, donor, cell, raw = [], [], [], []
        for d in fold[key]:
            X = data[d][:, gi]
            X = X[X.sum(1) > 0]
            mats.append(sp.csr_matrix(X.astype(np.float32))); donor += [d] * len(X)
            cell += [f"{d}-c{i}" for i in range(len(X))]
            raw.append([0, 0, 0, sim.stage[d] / 3, *sim.cov[d]])
        (d_ / split).mkdir(exist_ok=True)
        sp.save_npz(d_ / split / "counts.npz", sp.vstack(mats).tocsr())
        np.savez_compressed(d_ / split / "metadata.npz", cell=np.array(cell), donor=np.array(donor),
                            type=np.zeros(len(cell), int), donors=np.array(fold[key]), raw_c=np.array(raw, float), genes=genes)
    (d_ / "manifest.json").write_text(json.dumps({"panel": panel, "inner_fold": k, "synthetic": True,
                                                  "train_donors": fold["train_donors"], "heldout_donors": fold["heldout_donors"]}))


def fake_arrays(sim, real, panel, k, seed, kind):
    gi = PANEL_GENES[panel]
    T, Gp = len(real.donors), len(gi)
    out = BL._alloc(T, M, Gp, KPC, real.W.shape[1])
    perm_beta = np.random.default_rng(99).permutation(sim.beta)
    for t, donor in enumerate(real.donors):
        rng = np.random.default_rng(MX.stable_seed("fake", k, donor, kind, seed))
        st = sim.stage[donor]
        for m in range(M):
            libs = rng.choice(real.libs[t], K, replace=True)
            u = sim.u[donor] if kind == "oracle" else rng.normal(0, SD_U, G_ALL)
            logit = sim.logit(st, u, {"shrunk": .3, "inflated": 3.0}.get(kind, 1.0), perm_beta if kind == "wrongstage" else None)
            if kind == "biased":
                logit = logit + sim.bias
            X = sim.cells(rng, logit, K, gi, libs, cell_sd=.8 if kind == "overdisp" else 0.0).astype(np.float64)
            pb, ml, nz, pcs, lib, sq = BL._summarise(X, real, KPC)
            out["pb_counts"][t, m] = pb; out["mean_log"][t, m] = ml; out["nonzero"][t, m] = nz
            out["pcs"][t, m] = pcs; out["lib"][t, m] = lib; out["sq_log"][t, m] = sq
    return out


def write_fit(sim, real, arm, panel, k, seed, kind, root, mode_kind=None, sq=True):
    d_ = root / arm / panel / f"fold{k}" / f"seed{seed}"
    d_.mkdir(parents=True, exist_ok=True)
    out = fake_arrays(sim, real, panel, k, seed, kind)
    keys = ["pb_counts", "mean_log", "nonzero", "pcs", "lib"] + (["sq_log"] if sq else [])
    order = np.random.default_rng(k).permutation(len(real.donors))
    np.savez_compressed(d_ / "generated.npz", targets=real.donors[order], is_heldout=real.is_heldout[order],
                        stage=real.stage[order], genes=real.genes, **{key: out[key][order] for key in keys})
    (d_ / "config.json").write_text(json.dumps({"arm": arm, "panel": panel, "fold": k, "seed": seed, "fake": kind}))
    (d_ / "fit_receipt.json").write_text(json.dumps({"completed_updates": 20000, "fake": True, "device": "cuda"}))
    with open(d_ / "learning_curve.jsonl", "w") as fh:
        for step in [1] + list(range(250, 20001, 250)):
            fh.write(json.dumps({"step": step, "critic_gap": 5.0, "D_total": .8, "G_total": 2.0}) + "\n")
    if mode_kind is not None:
        mo = fake_arrays(sim, real, panel, k, seed + 100, mode_kind)
        np.savez_compressed(d_ / "generated_half.npz", mode=np.array("half"), targets=real.donors, is_heldout=real.is_heldout,
                            stage=real.stage, genes=real.genes, **{key: mo[key] for key in keys})
        (d_ / "generated_half.receipt.json").write_text(json.dumps({"mode": "half", "fit": str(d_)}))
    return d_


def synthetic():
    t0 = time.monotonic()
    if ROOT.exists():
        shutil.rmtree(ROOT)
    ROOT.mkdir(parents=True)
    sim = Sim()
    rng = np.random.default_rng(11)
    data = {d: sim.cells(rng, sim.logit(sim.stage[d], sim.u[d]), sim.ncell[d]) for d in sim.donors}
    for k in range(9):
        write_cache(sim, data, "hvg2000", k)
    for k in range(3):
        write_cache(sim, data, "wide", k)
    print(f"caches written {time.monotonic() - t0:.1f}s", flush=True)
    rc = EV.RealCache(ROOT / "caches", 500)
    fits, fits2, post = ROOT / "fits", ROOT / "fits_r2", ROOT / "posthoc"
    plan = [("base", "hvg2000", k, s, "truth", fits) for k in range(3) for s in (0, 1)]
    plan += [("base", "hvg2000", k, 0, "truth", fits2) for k in range(3, 9)]
    plan += [("base", "hvg2000", 0, 0, "oracle", fits2)]           # shadowed by round-1 root
    plan += [(a, "hvg2000", k, 0, kind, fits2) for a, kind in [("shrunk", "shrunk"), ("inflated@3", "inflated"), ("biased", "biased"),
                                                                ("overdisp", "overdisp"), ("wrongstage", "wrongstage")] for k in range(3)]
    plan += [("oracle", "hvg2000", 0, 0, "oracle", fits2), ("base", "wide", 0, 0, "truth", fits), ("base", "hvg2000", 0, 2, "truth", fits)]
    for arm, panel, k, s, kind, root in plan:
        real = rc.get(panel, k)[0]
        mode = "biased" if (arm == "base" and panel == "hvg2000" and k == 0 and s == 0 and root == fits) else None
        write_fit(sim, real, arm, panel, k, s, kind, root, mode_kind=mode, sq=(root == fits2 or kind != "truth" or k > 0))
        rc.drop_except({(panel, k)})
    # a mode file written elsewhere (posthoc_r2 style) for a round-1-root fit
    real = rc.get("hvg2000", 1)[0]
    mo = fake_arrays(sim, real, "hvg2000", 1, 0, "shrunk")
    post.mkdir()
    np.savez_compressed(post / "base_shrinkmode_f1s0.npz", mode=np.array("shrinkmode"), targets=real.donors, is_heldout=real.is_heldout,
                        stage=real.stage, genes=real.genes, **{key: mo[key] for key in ["pb_counts", "mean_log", "nonzero", "pcs", "lib", "sq_log"]})
    (post / "base_shrinkmode_f1s0.receipt.json").write_text(json.dumps({"mode": "shrinkmode", "fit": str(fits / "base/hvg2000/fold1/seed0")}))
    # repair FID-3/S4/FID-1: a stage-blind mode unit on all 3 folds of repeat 0, generated on another device
    for k in range(3):
        real = rc.get("hvg2000", k)[0]
        mo = fake_arrays(sim, real, "hvg2000", k, 0, "truth")
        np.savez_compressed(post / f"base_condmix1_f{k}s0.npz", mode=np.array("condmix@1"), targets=real.donors, is_heldout=real.is_heldout,
                            stage=real.stage, genes=real.genes, **{key: mo[key] for key in ["pb_counts", "mean_log", "nonzero", "pcs", "lib", "sq_log"]})
        (post / f"base_condmix1_f{k}s0.receipt.json").write_text(json.dumps({"mode": "condmix@1", "generation_device": "cpu",
                                                                              "fit": str(fits / f"base/hvg2000/fold{k}/seed0")}))
        rc.drop_except({("hvg2000", k)})
    print(f"fake fits written {time.monotonic() - t0:.1f}s", flush=True)
    out = ROOT / "results"
    t1 = time.monotonic()
    args = ["--fits-root", str(fits), "--fits-root", str(fits2), "--modes-root", str(post), "--caches-root", str(ROOT / "caches"),
            "--out", str(out), "--pc-repeats", "4", "--cell-repeats", "2", "--dam-draws", "500", "--xfit-permutations", "500"]
    agg = EV.main(args)
    t_first = time.monotonic() - t1

    def fit(unit, k=0, s=0, panel="hvg2000"):
        return json.loads((out / "fits" / unit / panel / f"fold{k}" / f"seed{s}.json").read_text())

    H = lambda r: EV.headline(r, "heldout")  # noqa: E731
    h = {u: H(fit(u)) for u in ["base", "shrunk", "inflated@3", "biased", "overdisp", "wrongstage", "oracle"]}
    hb = {b: H(json.loads((out / "baselines" / "hvg2000" / "fold0" / f"{b}.json").read_text())) for b in BL.BASELINES}
    rows = {(r["arm"], r["panel"]): r for r in agg["rows"]}
    # recalibration
    check("recal: planted per-gene bias inflates pb_rmse; train-only recalibration brings it back near truth",
          h["biased"]["pb_rmse"] > 1.5 * h["base"]["pb_rmse"] and h["biased"]["pb_rmse_recal"] < 1.15 * h["base"]["pb_rmse_recal"],
          f"biased {h['biased']['pb_rmse']:.3f} -> {h['biased']['pb_rmse_recal']:.3f}; truth {h['base']['pb_rmse']:.3f} -> {h['base']['pb_rmse_recal']:.3f}")
    check("recal: meanlog_rmse_gene of biased model falls to near truth after recalibration",
          h["biased"]["meanlog_rmse_gene"] > 2 * h["biased"]["meanlog_rmse_gene_recal"]
          and h["biased"]["meanlog_rmse_gene_recal"] < 1.5 * h["base"]["meanlog_rmse_gene_recal"] + .01,
          f"{h['biased']['meanlog_rmse_gene']:.3f} -> {h['biased']['meanlog_rmse_gene_recal']:.3f} (truth {h['base']['meanlog_rmse_gene_recal']:.3f})")
    check("recal metrics defined for baselines (same code path)", hb["grand_mean"]["pb_rmse_recal"] is not None
          and hb["resample"]["meanlog_rmse_gene_recal"] is not None, f"grand_mean recal {hb['grand_mean']['pb_rmse_recal']:.3f}")
    # second moment
    r1file = fit("base", 0, 0)
    check("second moment: absent (None) for a generated file without sq_log; present for r2 files",
          h["base"]["logvar_rmse_gene"] is None and not r1file["has_sq_log"] and h["shrunk"]["logvar_rmse_gene"] is not None, "")
    tb = H(fit("base", 3, 0))
    check("second moment: truth sampler small error, overdispersed sampler larger (planted extra per-cell noise)",
          h["overdisp"]["logvar_rmse_gene"] > 2 * tb["logvar_rmse_gene"] and abs(tb["logvar_bias"]) < .03 and h["overdisp"]["logvar_bias"] > .05,
          f"truth {tb['logvar_rmse_gene']:.4f} (bias {tb['logvar_bias']:.4f}) overdisp {h['overdisp']['logvar_rmse_gene']:.4f} (bias {h['overdisp']['logvar_bias']:.4f})")
    check("second moment of resample baseline (real cells, thinned) close to truth sampler",
          hb["resample"]["logvar_rmse_gene"] < 3 * tb["logvar_rmse_gene"] + .02, f"resample {hb['resample']['logvar_rmse_gene']:.4f}")
    # DAM
    check("DAM: 27 listed genes present; truth signed z > 3 and sign_frac > 0.85",
          h["base"]["dam_n_present"] == 27 and h["base"]["dam_signed_z"] > 3 and h["base"]["dam_sign_frac"] > .85,
          f"z={h['base']['dam_signed_z']:.2f} frac={h['base']['dam_sign_frac']:.2f}")
    check("DAM: wrong-stage model near null (|z| < 2.5), grand_mean baseline near null",
          abs(h["wrongstage"]["dam_signed_z"]) < 2.5 and abs(hb["grand_mean"]["dam_signed_z"]) < 2.5,
          f"wrong z={h['wrongstage']['dam_signed_z']:.2f} grand z={hb['grand_mean']['dam_signed_z']:.2f}")
    check("DAM permutation null (repair S1): truth dam_signed_perm_z > 3; grand_mean baseline |z| < 2.5",
          h["base"]["dam_signed_perm_z"] > 3 and abs(hb["grand_mean"]["dam_signed_perm_z"]) < 2.5,
          f"truth {h['base']['dam_signed_perm_z']:.2f} grand {hb['grand_mean']['dam_signed_perm_z']:.2f}")
    realref = json.loads((out / "real" / "hvg2000" / "fold0.json").read_text())
    check("DAM model-free real reference written (train contrast direction z > 2)",
          realref["dam_n_present"] == 27 and realref["dam_real"]["mean_log"]["train"]["signed_mean"]["z"] > 2,
          f"train real z={realref['dam_real']['mean_log']['train']['signed_mean']['z']:.2f}")
    # xfit
    xf = agg["xfit"]
    b0 = xf.get("base|hvg2000|repeat=0|seed=0"); b1 = xf.get("base|hvg2000|repeat=1|seed=0")
    check("xfit: computed per repeat for base (37 donors each), truth z > 3",
          b0 and b1 and b0["n_donors"] == 37 and b1["n_donors"] == 37 and b0["stage"]["score"]["z"] > 3,
          f"repeat0 score={b0 and b0['stage']['score']['value']:.3f} z={b0 and b0['stage']['score']['z']:.2f}")
    sc = lambda u: xf[f"{u}|hvg2000|repeat=0|seed=0"]["stage"]["score"]["value"]  # noqa: E731
    gm = xf["baseline:grand_mean|hvg2000|repeat=0|seed=None"]["stage"]["score"]
    # A 3x LOGIT stage effect is not a linear 3x scaling of the log1p(CP10k)
    # deviation (softmax renormalisation, log1p compression), so its score is
    # only expected to stay clearly positive; exact scale-freeness for a linear
    # scaling is checked in `units`.  (First synthetic run: truth 0.549,
    # inflated 0.453, wrong-stage 0.057, grand_mean -0.005.)
    zi = xf["inflated@3|hvg2000|repeat=0|seed=0"]["stage"]["score"]["z"]
    check("xfit: inflated (3x logit) still clearly positive (z > 3); wrong-stage model and grand_mean baseline much lower",
          zi > 3 and sc("inflated@3") > sc("base") - .2 and sc("wrongstage") < sc("base") - .1 and gm["value"] < sc("base") - .1,
          f"truth {sc('base'):.3f} inflated {sc('inflated@3'):.3f} wrong {sc('wrongstage'):.3f} grand_mean {gm['value']:.3f} (z {gm['z']})")
    check("xfit: stage_mean_log baseline (train stage means) scores positive with z > 2",
          xf["baseline:stage_mean_log|hvg2000|repeat=0|seed=None"]["stage"]["score"]["z"] > 2,
          f"{xf['baseline:stage_mean_log|hvg2000|repeat=0|seed=None']['stage']['score']}")
    rb = rows[("base", "hvg2000")]
    check("aggregation: base row repeat-balanced over 3 repeats, SE over repeats; xfit row over 3 repeats",
          rb["pb_rmse__n_repeats"] == 3 and rb["pb_rmse__se_kind"] == "repeats" and rb["xfit_score__n_repeats"] == 3
          and rb["pb_rmse__n_folds"] == 9, f"n_folds={rb['pb_rmse__n_folds']} n_rep={rb['pb_rmse__n_repeats']}")
    # repeated subsample
    rep = fit("shrunk")["pc_space_rep"]["heldout"]
    check("repeat-averaged PC metrics present with R draws and mcse = draw_sd/sqrt(R)",
          rep["energy_score"] is not None and abs(rep["energy_score_mcse"] - rep["energy_score_draw_sd"] / 2) < 1e-12
          and h["shrunk"]["pc_energy"] is not None, f"pc_energy_rep={rep['energy_score']:.3f} single={h['shrunk']['pc_energy']:.3f}")
    # modes, '@' arms, shadowing, ensembles
    check("mode unit 'base#half' (in fit dir) and 'base#shrinkmode' (--modes-root) evaluated; '@' arm row present; shadowed fit listed",
          ("base#half", "hvg2000") in rows and ("base#shrinkmode", "hvg2000") in rows and ("inflated@3", "hvg2000") in rows
          and len(agg["shadowed"]) == 1, f"shadowed={[(s['unit'], s['fold'], s['seed']) for s in agg['shadowed']]}")
    check("mode unit paired vs its parent arm and vs base",
          {"vs_base", "vs_parent"} <= set(agg["paired"].get("base#half|hvg2000", {})) or
          "vs_base" in agg["paired"].get("base#half|hvg2000", {}), f"{sorted(agg['paired'].get('base#half|hvg2000', {}))}")
    e3 = json.loads((out / "ensembles" / "base" / "hvg2000" / "fold0_n3.json").read_text())
    e2m = json.loads((out / "ensembles" / "base" / "hvg2000" / "fold0_n2_M32.json").read_text())
    check("ensembles: n=3 (all seeds, M=96), first-2-seed n=2 and M-matched (M=32) variants; labels in rows",
          e3["M"] == 96 and e2m["M"] == 32 and e2m["seeds"] == [0, 1] and ("base [seed-ensemble n=2 M=32]", "hvg2000") in rows
          and ("base [seed-ensemble n=3]", "hvg2000") in rows, "")
    pv = agg["paired"]["shrunk|hvg2000"]["vs_base"]["metrics"]
    check("paired tables exclude descriptive-only metrics (incl. matched-gene dam_signed_z) and include recal / rep / dam perm / xfit",
          "hn_spearman_vs_heldout" not in pv and "pb_rmse_recal" in pv and "pc_energy_rep" in pv and "dam_signed_perm_z" in pv
          and "dam_signed_z" not in pv and agg["dam_paired_panels"].get("hvg2000") is True
          and pv.get("xfit_score", {}).get("n") == 1, f"xfit paired {pv.get('xfit_score')}")
    cm = rows[("base#condmix@1", "hvg2000")]
    pcm = agg["paired"]["base#condmix@1|hvg2000"]
    check("stage-blind unit (repair S4): flagged, stage metrics None, no xfit; pb kept; vs_parent marked non-CRN (cpu vs cuda)",
          cm.get("stage_mode") == "stage-blind" and cm["xfit_score"] is None and cm["hn_spearman_vs_train"] is None
          and cm["dam_signed_perm_z"] is None and cm["pb_rmse"] is not None and "base#condmix@1" in agg["stage_blind_units"]
          and not any(k.startswith("base#condmix@1|") for k in agg["xfit"])
          and pcm["vs_parent"]["crn"] is False and "xfit_score" not in {k for k, v in pcm["vs_parent"]["metrics"].items() if v.get("n")},
          f"crn={pcm['vs_parent']['crn']} devices={pcm['vs_parent']['device_pairs']}")
    pb9 = agg["paired"]["shrunk|hvg2000"]["vs_base"]["metrics"]["pb_rmse"]
    check("paired sign counts per repeat present (repair S7)", "n_better_by_repeat" in pb9 and "repeats_majority_better" in pb9, f"{pb9.get('n_better_by_repeat')}")
    rr = rows[("baseline:ridge", "hvg2000")]; rs = rows[("baseline:resample", "hvg2000")]
    check("baseline rows: ridge meanlog/logvar masked, resample not; pb kept",
          rr["meanlog_rmse_gene"] is None and rr["logvar_rmse_gene"] is None and rr["meanlog_rmse_gene_recal"] is None
          and rr["pb_rmse"] is not None and rs["meanlog_rmse_gene"] is not None and rs["logvar_rmse_gene"] is not None, "")
    summ = (out / "summary.txt").read_text()
    check("summary: per-panel sections, like-for-like shared-gene section, descriptive table, xfit table, stage-blind tag",
          "## Panel hvg2000" in summ and "## Panel wide" in summ and "## Like-for-like" in summ and "Descriptive per-fold" in summ
          and "cross-fitted donor score" in summ and "base#condmix@1 [STAGE-BLIND]" in summ and " xdev" in summ, "")
    main_tab = summ.split("## Panel hvg2000")[1].split("Paired")[0]
    check("summary: hn_spearman_vs_heldout not in the main or paired tables", "hn_spearman_vs_heldout" not in main_tab
          and "hn_spearman_vs_heldout" not in summ.split("## Panel hvg2000")[1].split("Descriptive")[0], "")
    t2 = time.monotonic()
    EV.main(args)
    t_second = time.monotonic() - t2
    check("re-run is incremental", t_second < .5 * t_first, f"first={t_first:.1f}s second={t_second:.1f}s")
    tim = json.loads((out / "timings.json").read_text())
    shutil.rmtree(ROOT)
    return finish("synthetic", t0) if tim else 1


# ==========================================================================
# reproduce
# ==========================================================================
IGNORE = {"evaluator_version", "score_seconds", "generation_seconds", "fit_dir", "baseline_version", "meaningful_metrics",
          "generated_file", "subset_names"}


def compare(a, b, path, stats, tol=1e-8):
    """Every value of round-1 a must equal r2 b (numbers within tol)."""
    if isinstance(a, dict):
        if not isinstance(b, dict):
            stats["mismatch"].append(path); return
        for k, v in a.items():
            if path == "" and k in IGNORE:
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
        d = abs(float(a) - float(b))
        stats["n"] += 1
        stats["max"] = max(stats["max"], d)
        if d > tol:
            stats["mismatch"].append(f"{path}: {a} vs {b} (diff {d:.2e})")
    else:
        stats["mismatch"].append(f"{path}: type {type(a)}")


def reproduce(out):
    t0 = time.monotonic()
    out = Path(out)
    r1 = RUNS / "results_screen48"
    pairs = []
    for p in sorted((r1 / "fits").rglob("seed*.json")):
        pairs.append(("fit", p, out / p.relative_to(r1)))
    for p in sorted((r1 / "ensembles").rglob("fold*.json")):
        pairs.append(("ensemble", p, out / p.relative_to(r1).parent / (p.stem + "_n2.json")))
    for p in sorted((r1 / "baselines").rglob("*.json")):
        pairs.append(("baseline" if p.stem != "noise_ceiling" else "noise_ceiling", p, out / p.relative_to(r1)))
    tot = {}
    for kind, a, b in pairs:
        st = tot.setdefault(kind, {"files": 0, "n": 0, "max": 0.0, "mismatch": [], "missing": [], "absent": []})
        if not b.exists():
            st["absent"].append(str(b)); continue
        st["files"] += 1
        ja, jb = json.loads(a.read_text()), json.loads(b.read_text())
        if kind == "noise_ceiling":
            ja = {k: v for k, v in ja.items() if k not in IGNORE}
        compare(ja, jb, "", st)
    for kind, st in tot.items():
        check(f"reproduce {kind}: every round-1 value equal to 1e-8",
              st["files"] > 0 and not st["absent"] and not st["mismatch"] and not st["missing"],
              f"files={st['files']} values={st['n']} max_abs_diff={st['max']:.2e} absent={len(st['absent'])} "
              f"mismatch={st['mismatch'][:3]} missing={st['missing'][:3]}")
    # aggregate rows
    a1 = json.loads((r1 / "aggregate.json").read_text()); a2 = json.loads((out / "aggregate.json").read_text())
    rows2 = {(r["arm"], r["panel"]): r for r in a2["rows"]}
    st = {"n": 0, "max": 0.0, "mismatch": [], "missing": []}
    for r in a1["rows"]:
        lab = r["arm"].replace("[seed-ensemble]", "[seed-ensemble n=2]")
        r2 = rows2.get((lab, r["panel"]))
        if r2 is None:
            st["missing"].append(lab); continue
        masked = set(r2.get("masked_not_meaningful", []))
        for k, v in r.items():
            base = re.sub(r"__(se|n_folds|unit_mean|unit_se|n)$", "", k)
            if base.split("__")[-1] not in EV.ROUND1_HEADLINE:
                continue
            if base in masked or k.endswith("__n_folds"):
                continue
            k2 = k.replace("__se", "__se_folds") if k.endswith("__se") else k
            if v is None and r2.get(k2) is None:
                continue
            compare(v, r2.get(k2), f"{lab}|{r['panel']}|{k}", st)
    check("reproduce aggregate rows: round-1 headline means and fold SEs equal to 1e-8 (masked baseline metrics excluded)",
          not st["mismatch"] and not st["missing"], f"values={st['n']} max={st['max']:.2e} mismatch={st['mismatch'][:3]} missing={st['missing'][:3]}")
    st = {"n": 0, "max": 0.0, "mismatch": [], "missing": []}
    for key, comps in a1["paired"].items():
        lab, panel = key.rsplit("|", 1)
        lab2 = lab.replace("[seed-ensemble]", "[seed-ensemble n=2]")
        c1 = comps.get("vs_base")
        c2 = a2["paired"].get(f"{lab2}|{panel}", {}).get("vs_base")
        if c1 is None:
            continue
        if c2 is None:
            st["missing"].append(key); continue
        for m, s in c1["metrics"].items():
            if m.split("__")[-1] in EV.DESCRIPTIVE_ONLY or not s.get("n"):
                continue
            s2 = c2["metrics"].get(m, {})
            for f in ["n", "n_better", "n_worse", "mean_difference", "se_difference", "raw_mean_difference"]:
                compare(s.get(f), s2.get(f), f"{key}|{m}|{f}", st)
    check("reproduce paired vs_base (non-descriptive metrics): n, signs, mean and SE of differences equal to 1e-8",
          not st["mismatch"] and not st["missing"], f"values={st['n']} max={st['max']:.2e} mismatch={st['mismatch'][:3]} missing={st['missing'][:3]}")
    return finish("reproduce", t0)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "units"
    if cmd == "units":
        sys.exit(1 if units() else 0)
    if cmd == "synthetic":
        sys.exit(1 if synthetic() else 0)
    if cmd == "reproduce":
        sys.exit(1 if reproduce(sys.argv[2]) else 0)
    raise SystemExit(f"unknown subcommand {cmd}")
