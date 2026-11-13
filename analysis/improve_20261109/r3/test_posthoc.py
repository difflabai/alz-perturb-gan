#!/usr/bin/env python3
"""Tests for r2/posthoc.py on synthetic planted-truth data (CPU, 3 threads, no real counts).

  python3 -B r2/test_posthoc.py unit   # numerics, planted truth, leakage, 1/S law (~20 s)
  python3 -B r2/test_posthoc.py e2e    # synthetic cache + fit tree -> posthoc.run; raw and ensemble
                                       # numbers equal ../evaluate.py score_generated (~1-2 min)
  python3 -B r2/test_posthoc.py all

Planted truth (unit): G genes, true profile O_t = g + delta_s + donor noise; model
P_t = g + beta + kappa_s * delta_s + seed noise + small target noise.  Checks: recal recovers
-beta; fitted lam_s ~ 1/kappa_s and rho ~ 0; calibrated held-out error drops in the planted
order; the subset-averaged raw MSE follows a + b/S EXACTLY (a = ||mean_n - O||^2 - s2/(n-1),
b = n s2/(n-1)); changing held-out observations never changes any calibrated prediction
(leakage); recal of the train mean is an exact no-op; metric functions equal ../metrics.py;
box_ls equals scipy's bounded least squares; and a memorising model (train predictions = the
train donors) gets rho ~ 1 in-sample while held-out error does not improve (documents the
in-sample-optimism limit that no train-only calibration can remove).
The e2e test writes a synthetic cache whose donor IDS are development donors (synthetic counts,
no real data read), because ../evaluate.load_real refuses any other donor id.
"""
from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "3"
sys.dont_write_bytecode = True
R2 = os.path.dirname(os.path.abspath(__file__))
if R2 not in sys.path:
    sys.path.insert(0, R2)

import json  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402

import posthoc as PH  # noqa: E402

MX, E = PH.MX, PH.E


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print("  ok:", msg, flush=True)


# --------------------------------------------------------------------------
def planted(seed=0, G=400, n_tr_per=6, n_h_per=3, kappa=(3.0, 3.0, 3.0, 3.0), n_seeds=6, seed_sd=0.08,
            beta_sd=0.2, donor_sd=0.3):
    rng = np.random.default_rng(seed)
    g = rng.normal(1.0, 1.0, G)
    w = rng.normal(0, 0.15, (4, G))
    w -= w.mean(0)                                    # independent stage effects, centred over stages
    delta = {s: w[s] for s in range(4)}
    stage = np.array([s for s in range(4) for _ in range(n_tr_per)] + [s for s in range(4) for _ in range(n_h_per)])
    train = np.array([True] * (4 * n_tr_per) + [False] * (4 * n_h_per))
    T = len(stage)
    O = np.stack([g + delta[s] for s in stage]) + rng.normal(0, donor_sd, (T, G))
    RML = 0.5 * O + rng.normal(0, 0.05, (T, G))
    beta = rng.normal(0, beta_sd, G)
    Ps, MLs = [], []
    for k in range(n_seeds):
        eta = {s: rng.normal(0, seed_sd, G) for s in range(4)}
        P = np.stack([g + beta + kappa[s] * delta[s] + eta[s] for s in stage]) + rng.normal(0, 0.02, (T, G))
        Ps.append(P); MLs.append(0.5 * P + 0.1)
    return dict(g=g, delta=delta, stage=stage, train=train, O=O, RML=RML, beta=beta, Ps=Ps, MLs=MLs, kappa=kappa)


def test_box_ls():
    from scipy.optimize import lsq_linear
    rng = np.random.default_rng(1)
    worst = 0.0
    for trial in range(200):
        p = rng.integers(1, 6); n = 40
        A = rng.normal(0, 1, (n, p)) * rng.uniform(0.1, 3, p)
        y = A @ rng.normal(0.5, 1.5, p) + rng.normal(0, 0.5, n)
        x = PH.box_ls(A.T @ A, A.T @ y)
        ref = lsq_linear(A, y, bounds=(0, 1), method="bvls").x
        f1 = np.sum((y - A @ x) ** 2); f2 = np.sum((y - A @ ref) ** 2)
        worst = max(worst, f1 - f2)
        assert np.all(x >= 0) and np.all(x <= 1)
    check(worst < 1e-8, f"box_ls objective <= scipy bvls on 200 random problems (worst excess {worst:.1e})")
    x = PH.box_ls(np.zeros((2, 2)), np.zeros(2))
    check(np.array_equal(x, [0.0, 0.0]), "dead columns fixed at 0")


def test_metrics_equal_round1():
    d = planted()
    P, O, st, tr = d["Ps"][0], d["O"], d["stage"], d["train"]
    held = ~tr
    donors = np.array([f"d{i}" for i in range(len(st))])
    ref = MX.pseudobulk_metrics(P[held], O[held], O[tr].mean(0), st[held], donors[held])["rmse"]
    check(abs(PH.pb_rmse(P, O, held) - ref) < 1e-12, "pb_rmse equals metrics.pseudobulk_metrics rmse")
    ML, RML = d["MLs"][0], d["RML"]
    ref = MX.gene_error_metrics(ML[held], RML[held], RML[tr].mean(0), [0.1])["rmse_gene"]
    check(abs(PH.meanlog_rmse_gene(ML, RML, held) - ref) < 1e-12, "meanlog_rmse_gene equals metrics.gene_error_metrics")
    dec = PH.decompose(P, O, st, held)
    check(abs(dec["shared"] + dec["stage"] + dec["donor"] - dec["mse"]) < 1e-12, "MSE = shared + stage + donor exactly")
    # stage-constant predictor: donor part equals the in-sample floor
    sm = np.stack([O[held & (st == s)].mean(0) * 0 + P[tr & (st == s)].mean(0) for s in st])
    dec = PH.decompose(sm, O, st, held)
    Oh, sh = O[held], st[held]
    floor = np.mean((Oh - np.stack([Oh[sh == s].mean(0) for s in sh])) ** 2)
    check(abs(dec["donor"] - floor) < 1e-12, "stage-constant predictor: donor part == held-out within-stage floor")


def test_planted_recovery():
    d = planted(kappa=(2.0, 2.0, 4.0, 8.0), seed_sd=0.0)
    P = np.mean(d["Ps"], axis=0)
    O, st, tr = d["O"], d["stage"], d["train"]
    Pc, prm = PH.calibrate(P, O, tr, st, "recal")
    b = Pc[0] - P[0]
    # planted: b = -beta - mean_tr((kappa_s - 1) delta_s) + train donor-noise mean
    want_b = -d["beta"] - np.mean([(d["kappa"][s] - 1) * d["delta"][s] for s in st[tr]], axis=0)
    r = np.corrcoef(b, want_b)[0, 1]
    check(r > 0.95 and np.sqrt(np.mean((b - want_b) ** 2)) < 0.1, f"recal b recovers the planted level bias (r={r:.3f})")
    _, prm = PH.calibrate(P, O, tr, st, "recal_shrink_s")
    lam = prm["lam"]
    # planted optimum: the model stage deviation is D_s = kappa_s delta_s - c (c = train mean of
    # kappa delta), the true one delta_s, so lam_s = <delta_s, D_s> / ||D_s||^2 (~ 1/kappa_s)
    c = np.mean([d["kappa"][s] * d["delta"][s] for s in st[tr]], axis=0)
    want = {}
    for s in range(4):
        Ds = d["kappa"][s] * d["delta"][s] - c
        want[s] = float(np.clip(d["delta"][s] @ Ds / (Ds @ Ds), 0, 1))
    err = max(abs(lam[str(s)] - want[s]) for s in range(4))
    check(err < 0.05, f"per-stage lam recovers the planted optimum (max err {err:.3f}): "
          f"{ {k: round(v, 3) for k, v in lam.items()} } vs { {k: round(v, 3) for k, v in want.items()} }")
    _, prr = PH.calibrate(P, O, tr, st, "recal_shrinkr_s")
    check(prr["rho"] < 0.3, f"rho ~ 0 for pure target noise (rho={prr['rho']:.3f})")
    check(prm["rho"] == 0.0, "shrink_* methods keep rho = 0")
    d2 = planted(kappa=(4.0,) * 4, seed_sd=0.0)
    P2 = np.mean(d2["Ps"], axis=0)
    _, prm = PH.calibrate(P2, d2["O"], d2["train"], d2["stage"], "recal_shrink_g")
    check(abs(prm["lam"]["0"] - 0.25) < 0.06, f"global lam ~ 1/kappa = 0.25 (got {prm['lam']['0']:.3f})")
    held = ~d2["train"]
    e = {m: PH.pb_rmse(PH.calibrate(P2, d2["O"], d2["train"], d2["stage"], m)[0], d2["O"], held)
         for m in ["raw", "recal", "recal_shrink_g", "recal_pool"]}
    check(e["recal_shrink_g"] < e["recal"] < e["raw"], f"held-out error raw > recal > recal_shrink_g: {e}")
    _, p_noshr = PH.calibrate(P2, d2["O"], d2["train"], d2["stage"], "shrink_g")
    check(abs(p_noshr["lam"]["0"] - prm["lam"]["0"]) < 1e-9, "lam identical with and without recal (sum_tr D = 0)")


def test_leakage_and_noop():
    d = planted()
    P, O, st, tr = d["Ps"][0], d["O"], d["stage"], d["train"]
    O2 = O.copy(); O2[~tr] = np.random.default_rng(5).normal(0, 9, O2[~tr].shape)
    for m in PH.METHODS:
        a, _ = PH.calibrate(P, O, tr, st, m); b, _ = PH.calibrate(P, O2, tr, st, m)
        assert np.array_equal(a, b), m
    check(True, "no calibrated prediction depends on held-out observations (all 12 methods, bitwise)")
    P3 = P.copy(); P3[~tr] += 5.0
    for m in PH.METHODS:
        a, pa = PH.calibrate(P, O, tr, st, m); b, pb_ = PH.calibrate(P3, O, tr, st, m)
        assert np.allclose(a[tr], b[tr]) and pa == pb_, m
    check(True, "calibration parameters depend on train-target predictions only")
    tm = np.tile(O[tr].mean(0), (len(st), 1))
    out, prm = PH.calibrate(tm, O, tr, st, "recal")
    check(np.array_equal(out, tm + 0.0) or np.max(np.abs(out - tm)) < 1e-14, "recal of the train mean is a no-op (b = 0)")
    out, _ = PH.calibrate(P, O, tr, st, "recal_shrink_g")
    # lam = rho = 0 limit equals the train mean
    lvl = PH.calibrate(P, O, tr, st, "recal")[0][tr].mean(0)
    check(np.allclose(lvl, O[tr].mean(0)), "recal level equals the train mean of observed profiles")


def test_inverse_s_law():
    d = planted(n_seeds=5, seed_sd=0.15)
    O, st, tr = d["O"], d["stage"], d["train"]
    cur = PH.ensemble_curve(d["Ps"], d["MLs"], O, d["RML"], tr, st, ["raw", "recal_shrink_g"], loo=False)
    held = ~tr; n = 5
    Pst = np.stack(d["Ps"])[:, held]                    # [n, H, G]
    mu = Pst.mean(0)
    s2 = ((Pst - mu) ** 2).mean(0).mean(-1)            # per donor, ddof=0 over seeds
    base = ((mu - O[held]) ** 2).mean(-1)
    worst = 0.0
    for S in range(1, n + 1):
        pred = base + (n - S) / (S * (n - 1)) * s2
        worst = max(worst, np.max(np.abs(cur["S"][S]["raw"]["pb_mse_donor"] - pred)))
    check(worst < 1e-12, f"raw subset-averaged MSE == ||mean-O||^2 + (n-S)/(S(n-1)) s2 exactly (max diff {worst:.1e})")
    a_exact = base - s2 / (n - 1)
    a_fit = cur["inv_s_fit"]["raw"]["pb_mse_inf"]
    check(abs(a_fit - a_exact.mean()) < 1e-12, "1/S least-squares intercept equals the unbiased infinite-seed MSE")
    # planted infinite-seed truth (seed noise removed) is recovered roughly
    truth = np.mean([((np.stack([d["g"] + d["beta"] + 3.0 * d["delta"][s] for s in st[held]]) - O[held]) ** 2).mean()])
    truth += 0.02 ** 2                                # the per-target noise is not removed by seeds
    check(abs(a_fit - truth) < 0.02, f"projected MSE {a_fit:.4f} ~ planted noiseless-model MSE {truth:.4f}")
    check(cur["S"][5]["recal_shrink_g"]["pb_rmse"] < cur["S"][1]["recal_shrink_g"]["pb_rmse"], "ensembling helps the calibrated predictor too")


def test_memorisation_trap():
    d = planted()
    O, st, tr = d["O"], d["stage"], d["train"]
    P = np.stack([O[t] if tr[t] else O[tr & (st == st[t])].mean(0) for t in range(len(st))])
    s1 = PH.score_unit(P, None, O, None, tr, st, "recal_shrinkr_g", loo=True)
    s0 = PH.score_unit(P, None, O, None, tr, st, "recal_pool")
    check(s1["pb_params"]["rho"] > 0.9, f"memorising model: in-sample rho ~ 1 ({s1['pb_params']['rho']:.3f})")
    check(s1["pb_rmse_train"] < 0.05 and s1["pb_rmse_train_loo"] < 0.05,
          "memorising model: in-sample and LOO train scores both ~0 (LOO cannot remove model optimism)")
    check(s1["pb_rmse"] >= s0["pb_rmse"] - 1e-9, "memorising model: rho gives no held-out gain (rho only acts within stage)")


def unit():
    print("unit tests")
    test_box_ls(); test_metrics_equal_round1(); test_planted_recovery(); test_leakage_and_noop()
    test_inverse_s_law(); test_memorisation_trap()


# --------------------------------------------------------------------------
# End-to-end on a synthetic cache + fit tree
# --------------------------------------------------------------------------
def write_cache(cdir, donors_tr, donors_h, G=60, seed=0):
    rng = np.random.default_rng(seed)
    genes = np.array([f"G{i}" for i in range(G)])
    base = rng.dirichlet(np.ones(G) * 0.5)
    stage_eff = rng.normal(0, 0.3, (4, G))
    info = {}
    for split, ds in [("train", donors_tr), ("heldout", donors_h)]:
        rows, cell, donor, raw = [], [], [], []
        for j, dn in enumerate(ds):
            s = j % 4
            q = base * np.exp(stage_eff[s] + rng.normal(0, 0.2, G)); q /= q.sum()
            n = int(rng.integers(40, 90))
            lib = rng.integers(300, 900, n)
            X = rng.poisson(lib[:, None] * q[None, :] * rng.gamma(4, 0.25, (n, G)))
            X[X.sum(1) == 0, 0] = 1
            rows.append(X); cell += [f"{dn}_c{i}" for i in range(n)]; donor += [dn] * n
            raw.append([s / 6, s / 5, s / 3, s / 3, j % 2, j % 3 % 2, 70 + j])
            info[dn] = (s, q)
        C = sp.csr_matrix(np.vstack(rows).astype(np.float32))
        (cdir / split).mkdir(parents=True)
        sp.save_npz(cdir / split / "counts.npz", C)
        np.savez(cdir / split / "metadata.npz", cell=np.array(cell), donor=np.array(donor), type=np.array(["mg"] * len(cell)),
                 donors=np.array(ds), raw_c=np.array(raw, dtype=np.float64), genes=genes)
    W = np.linalg.qr(rng.normal(0, 1, (G, 50)))[0]
    np.savez(cdir / "space.npz", genes=genes, hvg=np.arange(G), mean=np.full(G, 0.5), sd=np.ones(G), W=W,
             pc_sd=np.ones(50), age_mean=np.array(80.0), age_sd=np.array(8.0))
    (cdir / "manifest.json").write_text(json.dumps({"train_donors": list(donors_tr), "heldout_donors": list(donors_h)}))
    return genes, info


def write_gen(path, real, info, seed, M=8, K=64, bias=0.0, amp=2.0):
    rng = np.random.default_rng(1000 + seed)
    T, G = len(real.donors), len(real.genes)
    pb = np.zeros((T, M, G), np.float32); ml = np.zeros_like(pb); nz = np.zeros_like(pb)
    pcs = np.zeros((T, M, 64, 50), np.float16); lib = np.zeros((T, M), np.float32)
    for t, dn in enumerate(real.donors):
        s, q = info[dn]
        for m in range(M):
            qq = q * np.exp(bias + amp * 0.1 * s + rng.normal(0, 0.1, G)); qq /= qq.sum()
            X = rng.poisson(rng.choice(real.libs[t], K)[:, None] * qq[None, :]).astype(np.float64)
            L = MX.log_cp10k(X)
            pb[t, m] = X.sum(0); ml[t, m] = L.mean(0); nz[t, m] = (X > 0).mean(0); lib[t, m] = X.sum(1).mean()
            pcs[t, m] = (((L[:64] - real.mean) / real.sd) @ real.W / real.pc_sd)[:64] if K >= 64 else 0
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, targets=real.donors, is_heldout=real.is_heldout, stage=real.stage, genes=real.genes,
                        pb_counts=pb, mean_log=ml, nonzero=nz, pcs=pcs, lib=lib)
    (path.parent / "fit_receipt.json").write_text("{}")


def e2e():
    print("e2e test")
    folds = json.loads((Path(R2).parent / "folds.json").read_text())
    dev = folds["development_donors"]
    with tempfile.TemporaryDirectory(prefix="posthoc_e2e_") as tmp:
        tmp = Path(tmp)
        cdir = tmp / "caches" / "hvg2000" / "fold0"
        _, info = write_cache(cdir, dev[:12], dev[12:20])
        real = E.load_real(cdir, "hvg2000", 0)
        fits = tmp / "fits"
        for s in range(3):
            write_gen(fits / "toy" / "hvg2000" / "fold0" / f"seed{s}" / "generated.npz", real, info, s)
        write_gen(fits / "toy" / "hvg2000" / "fold0" / "seed0" / "generated_snapens.npz", real, info, 7, amp=1.0)
        a = ["run", "--fits-root", str(fits), "--caches-root", str(tmp / "caches"), "--out", str(tmp / "out"),
             "--panels", "hvg2000", "--folds", "0", "--screen48", "none"]
        res = PH.main(a)
        subsets = {"all": (np.arange(len(real.genes)), False)}
        gens = [E.align_generated(E.load_generated(fits / "toy" / "hvg2000" / "fold0" / f"seed{s}" / "generated.npz"), real)
                for s in range(3)]
        u = res["units"]["hvg2000"]["toy"][0]
        # per-seed raw (S=1 average) vs evaluate.score_generated
        refs = [E.score_generated(g, real, subsets)[0]["subsets"]["all"]["heldout"] for g in gens]
        want_pb = np.mean([x["pseudobulk"]["rmse"] for x in refs]); want_ml = np.mean([x["mean_log"]["rmse_gene"] for x in refs])
        check(abs(u["S"][1]["raw"]["pb_rmse"] - want_pb) < 1e-10, "S=1 raw pb_rmse == evaluate.score_generated (seed mean)")
        check(abs(u["S"][1]["raw"]["meanlog_rmse_gene"] - want_ml) < 1e-10, "S=1 raw meanlog_rmse_gene == evaluate.score_generated")
        r = E.score_generated(E.concat_generated(gens), real, subsets)[0]["subsets"]["all"]["heldout"]
        check(abs(u["S"][3]["raw"]["pb_rmse"] - r["pseudobulk"]["rmse"]) < 1e-9, "S=3 raw == evaluate ensemble (concat_generated)")
        check(abs(u["S"][3]["raw"]["meanlog_rmse_gene"] - r["mean_log"]["rmse_gene"]) < 1e-9, "S=3 raw meanlog == evaluate ensemble")
        check("toy#snapens" in res["units"]["hvg2000"], "generation-mode file is its own unit (arm#mode)")
        for b in PH.BASELINE_NAMES:
            bl = res["units"]["hvg2000"][f"baseline:{b}"][0]
            gen, _ = PH.BL.generate(b, real)
            rr = E.score_generated(E.align_generated(gen, real), real, subsets)[0]["subsets"]["all"]["heldout"]
            assert abs(bl["S"][1]["raw"]["pb_rmse"] - rr["pseudobulk"]["rmse"]) < 1e-6, b
        check(True, "every baseline raw pb_rmse == evaluate.score_generated(baselines.generate) (float32 cache, 1e-6)")
        tm = res["units"]["hvg2000"]["direct:train_mean"][0]["S"][1]
        check(abs(tm["raw"]["pb_rmse"] - tm["recal"]["pb_rmse"]) < 1e-12, "direct:train_mean recal is a no-op")
        for f in ["results.json", "rows.csv", "summary.txt"]:
            check((tmp / "out" / f).exists(), f"wrote {f}")
        j = json.loads((tmp / "out" / "results.json").read_text())
        check(any(r["S"] == "inf" for r in j["rows"]), "1/S projection rows present")
        # repair S5/S6: like-for-like verdict keys, seed curve, reverdict reproduces the run's rows/verdict
        v = res["verdict"]["hvg2000"]
        check("any_beats_raw_round1_grand_mean" in v and "any_beats_like_for_like" in v and "any_beats_grand_mean" not in v,
              "verdict: raw-reference booleans renamed; like-for-like booleans present")
        bl = v["best_generator_row_like_for_like"]
        ref_vals = [PH._row(res, "hvg2000", "baseline:grand_mean", bl["method"], 1)["pb_rmse"],
                    PH._row(res, "hvg2000", "direct:train_mean", "raw", 1)["pb_rmse"]]
        check(abs(bl["reference_pb_rmse_same_folds"] - min(ref_vals)) < 1e-12
              and v["any_beats_like_for_like"] == (bl["diff_vs_reference"] < 0),
              "like-for-like reference = min(grand_mean same method, direct:train_mean raw)")
        sc = v["seed_curve"].get("toy")
        check(sc is not None and [q["S"] for q in sc["points"]] == [1, 2, 3] and all(q["lam"] is not None for q in sc["points"]),
              "seed curve over S=1..3 with lambda for the 3-seed unit")
        check(PH.stage_mode_of("x#condmix@1") == "stage-blind" and PH.stage_mode_of("x#condmix@0.5") == "stage-mixture"
              and PH.stage_mode_of("x") is None, "stage-blind tagging of mode units")
        rv = PH.main(["reverdict", "--out", str(tmp / "out")])
        check(json.dumps(PH._clean(rv["verdict"]), sort_keys=True) == json.dumps(PH._clean(res["verdict"]), sort_keys=True)
              and len(rv["rows"]) == len(res["rows"]), "reverdict from results.json reproduces the run's verdict")
        # second run reuses baseline caches
        res2 = PH.main(a)
        check(res2["rows"] == j["rows"] or json.loads((tmp / "out" / "results.json").read_text())["rows"] == j["rows"],
              "rerun is deterministic (baseline cache reused)")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("unit", "all"):
        unit()
    if what in ("e2e", "all"):
        e2e()
    print("PASS")
