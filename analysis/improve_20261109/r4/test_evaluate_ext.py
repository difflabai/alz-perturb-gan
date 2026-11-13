#!/usr/bin/env python3
"""Planted-truth tests for r4/evaluate_ext.py.

  python3 -B test_evaluate_ext.py            # all tests (the real-data pipeline test takes ~2-3 min)
  python3 -B test_evaluate_ext.py --fast     # synthetic tests only
  python3 -B test_evaluate_ext.py --ens      # confirmation extension only: seed ensembles (synthetic + real data:
                                             #   x ensembles vs the lead's smoke_r4/ens/x_ensembles.json, reference
                                             #   ensembles and --ref-arm units vs results_r3), by-repeat aggregation

Scratch goes to runs/improve_20261109/smoke_r4/eval/test_tmp (deleted at the end).
"""
from __future__ import annotations

import json
import shutil
import sys
import types
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_ext as X  # noqa: E402

EV, MX, BL = X.EV, X.MX, X.BL
TMP = X.RUNS / "smoke_r4" / "eval" / "test_tmp"
FOLDS = json.loads(EV.FOLDS_JSON_R1.read_text())


def _template(G=30, D=5, seed=0):
    rng = np.random.default_rng(seed)
    return EV.RealFold("hvg2000", 0, TMP, np.array([f"G{i}" for i in range(G)]), np.zeros(G), np.ones(G),
                       rng.normal(size=(G, D)), np.ones(D), 80.0, 8.0, None, None, None, None, None, None, None,
                       manifest_sha256="synthetic")


def _entry(tpl, donor, held, stage, cohort, q, n_cells, lib, rng, male=0, apoe=0, age=85.0):
    lam = rng.gamma(20.0, lib / 20.0, size=n_cells)[:, None] * q[None, :]
    C = sp.csr_matrix(rng.poisson(lam).astype(np.float32))
    arr = X._donor_arrays(tpl, C)
    rc = np.array([0, 0, 0, stage / 3.0, male, apoe, age], np.float64)
    return {"donor": donor, "held": held, "raw_c": rc, "stage": stage, "cohort": cohort, **arr}


def _profiles(G, rng, stage_eff=0.3):
    base = rng.lognormal(0, 1.2, G); base /= base.sum()
    eff = rng.normal(0, stage_eff, G)
    return base, eff


def _q(base, eff, stage, extra=None):
    lq = np.log(base) + eff * stage / 3.0 + (0 if extra is None else extra)
    q = np.exp(lq)
    return q / q.sum()


# --------------------------------------------------------------------------
def test_guard():
    dev = FOLDS["development_donors"]; sealed = FOLDS["sealed_evaluation_donors_fold0"]
    sea, ext = X.check_targets([dev[0], "MADRC-01", dev[1]], "t")
    assert sea == [dev[0], dev[1]] and ext == ["MADRC-01"]
    for bad in ([sealed[0]], ["H99.99.999"], ["MADRC-27"]):       # sealed, unknown, external donor without ITG
        try:
            X.check_targets(bad, "t")
        except PermissionError:
            continue
        raise AssertionError(f"guard accepted {bad}")
    for out in [X.RUNS / "results_r3", X.RUNS / "smoke_r4", X.RUNS / "caches" / "x", Path("/tmp/x")]:
        try:
            X.check_out_dir(out)
        except PermissionError:
            continue
        raise AssertionError(f"out guard accepted {out}")
    X.check_out_dir(X.RUNS / "results_r4"); X.check_out_dir(TMP)
    print("PASS test_guard")


def test_strip_cohort():
    cases = {"cohort": "base", "cohort,pbulk": "pbulk", "cohortfix,genemean,t.reg@4": "genemean,t.reg@4",
             "pbulk": "pbulk", "base": "base", "cohort@2,pbulk": "pbulk", "cohortfix,extperm,pbulk": "pbulk"}
    for a, b in cases.items():
        assert X.strip_cohort(a) == b, (a, X.strip_cohort(a), b)
    assert X.x_label({"arm": "cohort,pbulk", "mode": "snapens"}) == "cohort,pbulk~x#snapens"
    print("PASS test_strip_cohort")


def _synthetic_world(seed=1, G=40, cohort_shift=None, n_sea_tr=(2, 2, 3, 6), n_held=(1, 1, 1, 2), n_ext=(6, 3, 3, 4)):
    rng = np.random.default_rng(seed)
    tpl = _template(G, seed=seed)
    base, eff = _profiles(G, rng)
    shift = np.zeros(G) if cohort_shift is None else cohort_shift
    ents = []
    k = 0
    for s, n in enumerate(n_sea_tr):
        for _ in range(n):
            ents.append(_entry(tpl, f"S{k:02d}", False, s, 0, _q(base, eff, s), 150, 3000, rng, male=k % 2, age=70 + k)); k += 1
    ext = []
    for s, n in enumerate(n_ext):
        for _ in range(n):
            ext.append(_entry(tpl, f"E{k:02d}", False, s, 1, _q(base, eff, s, shift), 150, 3000, rng, male=k % 2, age=70 + k)); k += 1
    held = []
    for s, n in enumerate(n_held):
        for _ in range(n):
            held.append(_entry(tpl, f"H{k:02d}", True, s, 0, _q(base, eff, s), 150, 3000, rng, male=k % 2, age=70 + k)); k += 1
    real = X.build_realfold(tpl, ents + held, "sea")
    return tpl, real, ext, base, eff


def test_cohort_offset_planted():
    """Planted per-gene cohort shift with a stage effect and DIFFERENT stage composition per cohort:
    the stage-matched estimate recovers the shift; a naive pooled ratio is confounded by stage."""
    G = 40
    rng = np.random.default_rng(7)
    shift = rng.normal(0, 0.7, G)
    tpl, real, ext, base, eff = _synthetic_world(seed=3, G=G, cohort_shift=shift)
    pool = X.build_realfold(tpl, X.pool_entries(real, ext), "pool")
    o, c, info = X.cohort_offset(pool)
    truth = shift - np.log(np.sum(np.exp(shift) * base))        # identifiable up to a constant: compare centred
    oc, tc = o - o.mean(), truth - truth.mean()
    err = np.sqrt(np.mean((oc - tc) ** 2))
    tr = ~pool.is_heldout
    P = pool.pb / pool.pb.sum(1, keepdims=True)
    naive = np.log(P[tr & (pool.cohort == 1)].mean(0)) - np.log(P[tr & (pool.cohort == 0)].mean(0))
    nerr = np.sqrt(np.mean(((naive - naive.mean()) - tc) ** 2))
    assert MX.pearson(oc, tc) > 0.97, MX.pearson(oc, tc)
    assert err < 0.5 * nerr or err < 0.08, (err, nerr)
    # correction: the corrected external donors no longer differ from SEA-AD within stage
    corr = X.build_realfold(tpl, X.apply_offset(pool, o, c), "poolc")
    o2, _, _ = X.cohort_offset(corr)
    assert np.sqrt(np.mean(o2 ** 2)) < 0.25 * np.sqrt(np.mean(o ** 2)), (np.sqrt(np.mean(o2 ** 2)), np.sqrt(np.mean(o ** 2)))
    # SEA-AD donors untouched; counts stay integers
    for i in np.where(corr.cohort == 0)[0]:
        assert np.array_equal(corr.pb[i], pool.pb[i])
    for i in np.where(corr.cohort == 1)[0]:
        d = corr.cells[i].data
        assert np.allclose(d, np.round(d))
    print(f"PASS test_cohort_offset_planted (rms err {err:.3f} vs naive {nerr:.3f}; residual after correction "
          f"{np.sqrt(np.mean(o2 ** 2)):.3f} of {np.sqrt(np.mean(o ** 2)):.3f})")


def test_ridge_cohort():
    """Planted cohort shift in log profiles: ridge with the cohort column predicts SEA-AD held-out targets
    (cohort 0) better than ridge without it; the module patch is restored."""
    G = 40
    rng = np.random.default_rng(11)
    shift = rng.normal(0, 0.8, G)
    tpl, real, ext, base, eff = _synthetic_world(seed=5, G=G, cohort_shift=shift)
    pool = X.build_realfold(tpl, X.pool_entries(real, ext), "pool")
    orig = BL.ridge_design
    mu0, _, _ = BL.fit_profile_baseline("ridge", pool)
    with X.ridge_with_cohort():
        assert BL.ridge_design is not orig
        mu1, _, info = BL.fit_profile_baseline("ridge", pool)
    assert BL.ridge_design is orig
    h = pool.is_heldout
    obs = MX.log_cp10k(pool.pb[h])
    e0 = np.sqrt(np.mean((mu0[h] - obs) ** 2)); e1 = np.sqrt(np.mean((mu1[h] - obs) ** 2))
    assert e1 < e0, (e1, e0)
    # SEA-AD-only ridge unaffected by the patch (cohort column constant 0 -> standardised to 0)
    m_a, _, _ = BL.fit_profile_baseline("ridge", real)
    with X.ridge_with_cohort():
        m_b, _, _ = BL.fit_profile_baseline("ridge", real)
    assert np.allclose(m_a, m_b, atol=1e-10)
    print(f"PASS test_ridge_cohort (held-out rmse {e1:.4f} with cohort column vs {e0:.4f} without)")


def _fake_gen(real, rng, M=4, K=50, extra=None, extra_scale=None):
    """Generated dict for real.donors (+ extra entries): K-cell pseudobulks drawn from each donor's own proportions
    (extra entries' proportions multiplied by extra_scale: a cohort-blind model)."""
    donors = list(real.donors) + ([e["donor"] for e in extra] if extra else [])
    sc = 1.0 if extra_scale is None else extra_scale
    pbs = list(real.pb) + ([e["pb"] * sc for e in extra] if extra else [])
    T, G = len(donors), real.pb.shape[1]
    pb = np.zeros((T, M, G), np.float32); ml = np.zeros_like(pb); nz = np.zeros_like(pb)
    for t in range(T):
        q = pbs[t] / pbs[t].sum()
        for m in range(M):
            C = rng.poisson(np.full((K, 1), 3000.0) * q[None, :])
            pb[t, m] = C.sum(0); L = MX.log_cp10k(C); ml[t, m] = L.mean(0); nz[t, m] = (C > 0).mean(0)
    held = list(real.is_heldout) + ([False] * len(extra) if extra else [])
    stage = list(real.stage) + ([e["stage"] for e in extra] if extra else [])
    return {"targets": np.array(donors), "is_heldout": np.array(held), "stage": np.array(stage), "genes": real.genes,
            "pb_counts": pb, "mean_log": ml, "nonzero": nz, "lib": pb.sum(-1) / K,
            "pcs": np.zeros((T, M, 8, real.W.shape[1]), np.float16)}


def test_pooled_recal_and_split():
    """External targets do not change SEA-AD-view inputs; pooled recalibration equals the hand formula."""
    G = 40
    rng = np.random.default_rng(13)
    shift = rng.normal(0, 0.6, G)
    tpl, real, ext, base, eff = _synthetic_world(seed=9, G=G, cohort_shift=shift)
    ext_rf = X.build_realfold(tpl, ext, "ext")
    gen = _fake_gen(real, rng, extra=ext, extra_scale=np.exp(-shift))
    sub = X.subset_generated(gen, real.donors.tolist())
    assert np.array_equal(sub["targets"], real.donors) and sub["pb_counts"].shape[0] == len(real.donors)
    al = EV.align_generated(sub, real)
    assert np.array_equal(al["pb_counts"], gen["pb_counts"][:len(real.donors)])
    blk = X.pooled_recal_block(gen, real, ext_rf)
    pred = MX.predicted_profiles(gen["pb_counts"].astype(np.float64))
    obs = np.concatenate([MX.log_cp10k(real.pb), MX.log_cp10k(ext_rf.pb)])
    nt = len(real.donors); tr = np.where(~real.is_heldout)[0]; h = np.where(real.is_heldout)[0]
    pool_idx = np.concatenate([tr, nt + np.arange(len(ext))])
    for nm, idx in [("seaad", tr), ("pooled", pool_idx)]:
        a = (obs[idx] - pred[idx]).mean(0)
        want = np.mean([np.sqrt(np.mean((pred[i] + a - obs[i]) ** 2)) for i in h])
        assert abs(blk[f"pb_rmse_recal_{nm}"] - want) < 1e-10, (nm, blk[f"pb_rmse_recal_{nm}"], want)
    # cohort shift present -> pooled calibration is pulled away from the SEA-AD one
    assert blk["shift_rms_pooled_minus_seaad"] > 0.05
    assert blk["pb_rmse_recal_pooled"] > blk["pb_rmse_recal_seaad"]
    print(f"PASS test_pooled_recal_and_split (recal seaad {blk['pb_rmse_recal_seaad']:.4f}, pooled {blk['pb_rmse_recal_pooled']:.4f})")


def _write_hoc_npz(path, rng, G, contrast, model_contrast, n_held=(1, 1, 2, 3), noise=1.0):
    st = np.concatenate([[s] * n for s, n in enumerate(n_held)])
    obs = rng.normal(0, noise, (len(st), G)) + np.outer(st == 3, contrast)
    sdev = np.zeros((4, G)); sdev[3] = model_contrast
    n_tr = 5
    np.savez(path, is_heldout=np.r_[np.zeros(n_tr, bool), np.ones(len(st), bool)], stage=np.r_[np.zeros(n_tr, int), st],
             xf_obs_dev=obs.astype(np.float32), xf_stage_dev=sdev.astype(np.float32))


def test_hoc_planted():
    """Held-out contrast agreement: a model with the true contrast is detected; random models have a calibrated null."""
    TMP.mkdir(parents=True, exist_ok=True)
    G = 300
    rng = np.random.default_rng(17)
    truth = rng.normal(0, 1, G)
    paths = []
    for k in range(3):
        p = TMP / f"hoc_true_{k}.npz"; _write_hoc_npz(p, rng, G, truth, truth); paths.append(p)
    r = X.heldout_contrast_agreement(paths, 500, 1)
    assert r["spearman"]["value"] > 0.3 and r["spearman"]["p_one_sided"] < 0.01, r["spearman"]
    ps = []
    for j in range(40):
        paths = []
        model = rng.normal(0, 1, G)
        for k in range(3):
            p = TMP / f"hoc_null_{k}.npz"; _write_hoc_npz(p, rng, G, truth, model); paths.append(p)
        ps.append(X.heldout_contrast_agreement(paths, 200, 2 + j)["spearman"]["p_one_sided"])
    frac = float(np.mean(np.array(ps) < 0.05))
    assert frac <= 0.15, frac
    # a model with the opposite sign is not rewarded
    paths = []
    for k in range(3):
        p = TMP / f"hoc_neg_{k}.npz"; _write_hoc_npz(p, rng, G, truth, -truth); paths.append(p)
    rn = X.heldout_contrast_agreement(paths, 300, 3)
    assert rn["spearman"]["value"] < 0 and rn["spearman"]["p_one_sided"] > 0.9
    print(f"PASS test_hoc_planted (true model z {r['spearman']['z']:.1f}; random models P(p<0.05) = {frac:.3f})")


def test_compare_json():
    a = {"x": 1.0, "y": [1, 2, {"z": 0.5}], "generation_seconds": 3}
    b = {"x": 1.0 + 1e-10, "y": [1, 2, {"z": 0.5}], "generation_seconds": 9}
    assert X.compare_json(a, b) == []
    b["y"][2]["z"] = 0.5001
    assert len(X.compare_json(a, b)) == 1
    print("PASS test_compare_json")


def test_x_fit_pipeline_real():
    """Real data, fold 0: a fake hvg2000x unit whose SEA-AD targets are the round-2 pbulk fit's generated
    targets, plus four external targets, scores EXACTLY like its hvg2000 reference on every SEA-AD metric
    (paired difference 0), its pooled-recal block is consistent with score_generated's SEA-AD recal, and
    the reference search finds fits_r2/pbulk."""
    src = X.RUNS / "fits_r2" / "pbulk" / "hvg2000" / "fold0" / "seed0"
    if not (src / "generated.npz").exists():
        print("SKIP test_x_fit_pipeline_real (no fits_r2 pbulk fold0 seed0)")
        return
    fits = TMP / "fits_fake"
    fd = fits / "cohort,pbulk" / "hvg2000x" / "fold0" / "seed0"
    fd.mkdir(parents=True, exist_ok=True)
    g = dict(np.load(src / "generated.npz", allow_pickle=False))
    ext_ids = ["MADRC-01", "MADRC-13", "MADRC-02", "MADRC-09"]
    pick = [0, 1, 2, 3]            # copy four SEA-AD train targets' arrays as stand-in external predictions
    for k in ["pb_counts", "mean_log", "nonzero", "pcs", "lib", "sq_log"]:
        g[k] = np.concatenate([g[k], g[k][pick]])
    g["targets"] = np.concatenate([g["targets"].astype(str), ext_ids]); g["is_heldout"] = np.r_[g["is_heldout"], [False] * 4]
    g["stage"] = np.r_[g["stage"], [1, 0, 3, 2]]
    np.savez(fd / "generated.npz", **g)
    (fd / "fit_receipt.json").write_text(json.dumps({"device": "cpu", "note": "test fake"}))
    (fd / "config.json").write_text(json.dumps({"arm": "cohort,pbulk"}))
    out = TMP / "out"
    opts = types.SimpleNamespace(pc_repeats=2, cell_repeats=1, dam_draws=200, ext_cap=1000, ext_pc_repeats=1,
                                 ext_cell_repeats=1, xfit_permutations=50, hoc_permutations=50, agg_folds="0")
    log = lambda m: None  # noqa: E731
    ctx = X.FoldCtx(X.RUNS / "caches", 0, opts)
    recs, _ = X.scan_x_fits([f"t={fits}"], (), {0})
    assert len(recs) == 1, recs
    res, _ = X.evaluate_x_fit(recs[0], ctx, out, log)
    ref_label, ref_recs = X.find_reference(recs[0], ["r2=" + str(X.RUNS / "fits_r2")])
    assert ref_label == "pbulk~r2", ref_label
    rres = [X.evaluate_ref(r, ctx, out, log) for r in ref_recs if r["seed"] == 0][0]
    fa, fb = X._flat(res), X._flat(rres)
    for m in ["pb_rmse", "pb_rmse_recal", "meanlog_rmse_gene", "nonzero_rmse_gene", "logvar_rmse_gene", "pc_energy_rep",
              "cell_energy_rep", "pc_coverage80_rep", "train_seaad_pb_rmse"]:
        assert fa[m] is not None and abs(fa[m] - fb[m]) < 1e-12, (m, fa[m], fb[m])
    pr = res["ext"]["pooled_recal"]
    assert abs(pr["pb_rmse_recal_seaad"] - fa["pb_rmse_recal"]) < 1e-9, (pr["pb_rmse_recal_seaad"], fa["pb_rmse_recal"])
    assert abs(pr["meanlog_rmse_gene_recal_seaad"] - fa["meanlog_rmse_gene_recal"]) < 1e-9
    assert res["ext"]["n_ext_targets"] == 4 and res["ext"]["ext_view"]["external"]["pb_rmse"] is not None
    X.EV.write_json(out / "ref_map_fold0.json", {X.x_label(recs[0]): ref_label})
    agg = X.aggregate(out, opts, log)
    c = agg["comparison_vs_hvg2000"]["cohort,pbulk~x"]
    for m in ["pb_rmse", "meanlog_rmse_gene", "pc_energy_rep", "pc_coverage80_rep"]:
        assert c[m]["mean_difference"] == 0.0, (m, c[m])
    assert c["pb_rmse"]["seed_noise_se"] is not None      # round-2 prior (one seed)
    print(f"PASS test_x_fit_pipeline_real (pb_rmse {fa['pb_rmse']:.4f} both; pooled recal {pr['pb_rmse_recal_pooled']:.4f} "
          f"vs SEA-AD recal {pr['pb_rmse_recal_seaad']:.4f}; ext-view pb_rmse {res['ext']['ext_view']['external']['pb_rmse']:.4f})")


def test_x_ensemble_synthetic():
    """x_ensemble_generated = r3 concat_generated of the members' SEA-AD targets (external targets dropped,
    targets reordered to the RealFold), pooled along M in seed order."""
    G = 30
    rng = np.random.default_rng(21)
    tpl, real, ext, base, eff = _synthetic_world(seed=5, G=G)
    gens = {}
    for s in (0, 1):
        g = _fake_gen(real, rng, M=3, extra=ext[:3])
        perm = rng.permutation(len(g["targets"]))           # members store targets in another order
        gens[s] = {k: (v[perm] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == len(perm) and k != "genes" else v)
                   for k, v in g.items()}
    orig = X.load_generated_ext
    X.load_generated_ext = lambda path, store=None: dict(gens[int(str(path)[-1])])
    try:
        ctx = types.SimpleNamespace(real=real, store=None, fold=0)
        recs = [{"gen": f"fake{s}", "seed": s} for s in (0, 1)]
        ens, members = X.x_ensemble_generated(recs, ctx)
    finally:
        X.load_generated_ext = orig
    want = EV.concat_generated([EV.align_generated(X.subset_generated(gens[s], real.donors.tolist()), real) for s in (0, 1)])
    assert ens["pb_counts"].shape == (len(real.donors), 6, G)
    for k in ["pb_counts", "mean_log", "nonzero", "lib", "pcs"]:
        assert np.array_equal(ens[k], want[k]), k
    pos = {d: i for i, d in enumerate(gens[1]["targets"])}
    i = pos[str(real.donors[2])]
    assert np.array_equal(ens["pb_counts"][2, 3:], gens[1]["pb_counts"][i])        # member 1 occupies M slots 3..5
    assert list(ens["targets"]) == list(real.donors)
    print("PASS test_x_ensemble_synthetic")


def test_ensembles_real():
    """Real data, fold 0 (skipped if the inputs are absent):
    (a) two hvg2000x units of the lead's ad hoc script (smoke_r4/ens/x_ensembles.py) re-scored through
        evaluate_x_ensemble reproduce every value it stored in x_ensembles.json exactly;
    (b) the reference ensemble pbulk~r3#snapens (pooled and M-matched) and the --ref-arm unit pbulk~r3#snapens
        seed 0 reproduce results_r3 exactly (JSON identical except timings);
    (c) collect/aggregate pick the ensembles up; aggregate --by-repeat writes by_repeat/rep0 with the same rows."""
    lead = X.RUNS / "smoke_r4" / "ens" / "x_ensembles.json"
    have_x = all((X.RUNS / "fits_r4" / a / "hvg2000x" / "fold0" / f"seed{s}" / "generated_snapens.npz").exists()
                 for a in ("pbulk", "g.dispz,pbulk,t.kappa@0.25") for s in (0, 1))
    have_r3 = (X.RUNS / "results_r3" / "ensembles" / "pbulk~r3#snapens" / "hvg2000" / "fold0_n2.json").exists()
    if not (lead.exists() and have_x and have_r3):
        print("SKIP test_ensembles_real (inputs absent)")
        return
    out = TMP / "ens_out"
    opts = types.SimpleNamespace(pc_repeats=EV.DEFAULT_PC_REPEATS, cell_repeats=EV.DEFAULT_CELL_REPEATS, dam_draws=EV.DAM_NULL_DRAWS,
                                 ext_cap=1000, ext_pc_repeats=1, ext_cell_repeats=1, xfit_permutations=50,
                                 hoc_permutations=50, agg_folds="0")
    log = lambda m: None  # noqa: E731
    ctx = X.FoldCtx(X.RUNS / "caches", 0, opts)
    want = json.loads(lead.read_text())
    fits = ["r4=" + str(X.RUNS / "fits_r4")]
    refs = ["r3=" + str(X.RUNS / "fits_r3")]
    labels = ["pbulk~x#snapens", "g.dispz,pbulk,t.kappa@0.25~x#biascal@stagehalf", "pbulk~r3#snapens"]
    done, missing = X.run_ensembles(labels, ctx, out, log, opts, fits, refs, [0, 1])
    assert not missing and len(done) == 3, (done, missing)
    n = 0
    for lab, key in [(labels[0], "pbulk#snapens|fold0"), (labels[1], "g.dispz,pbulk,t.kappa@0.25#biascal@stagehalf|fold0")]:
        res = json.loads((out / "ensembles" / lab / "hvg2000" / "fold0_n2.json").read_text())
        h = res["subsets"]["all"]["heldout"]
        mine = {"pb_rmse": h["pseudobulk"]["rmse"]}

        def dig(o, pre=""):
            if isinstance(o, dict):
                for k, v in o.items():
                    dig(v, pre + k + ".")
            elif isinstance(o, (int, float)) and any(t in pre for t in ["meanlog", "nonzero", "logvar", "energy", "coverage80", "recal"]):
                mine[pre[:-1]] = o
        dig(h); dig(res.get("pc_space_rep", {}).get("heldout", {}), "pcrep.")
        assert set(mine) == set(want[key]), key
        for k, v in want[key].items():
            assert mine[k] == v, (key, k, mine[k], v)
            n += 1
        assert res["label"] == lab + " [seed-ensemble n=2]" and res["seeds"] == [0, 1] and res["kind"] == "seed_ensemble"
    skip = X.SKIP_KEYS | {"fit_dir", "root", "score_seconds", "generation_seconds"}
    for suf in ("_n2", "_n2_M32"):
        a = json.loads((out / "ref" / "ensembles" / "pbulk~r3#snapens" / "hvg2000" / f"fold0{suf}.json").read_text())
        b = json.loads((X.RUNS / "results_r3" / "ensembles" / "pbulk~r3#snapens" / "hvg2000" / f"fold0{suf}.json").read_text())
        d = X.compare_json(a, b, skip=skip)
        assert not d, d[:5]
    arm = [r for r in X.scan_ref_arm_fits(refs, ["pbulk"], {0}) if r["unit"] == "pbulk~r3#snapens" and r["seed"] == 0]
    assert len(arm) == 1
    ra = X.evaluate_ref(arm[0], ctx, out, log)
    rb = json.loads((X.RUNS / "results_r3" / "fits" / "pbulk~r3#snapens" / "hvg2000" / "fold0" / "seed0.json").read_text())
    assert not X.compare_json(json.loads(json.dumps(ra, default=str)), rb, skip=skip)
    units = X.collect(out)
    assert "pbulk~x#snapens [seed-ensemble n=2]" in units and "pbulk~r3#snapens [seed-ensemble n=2 M=32]" in units
    agg = X.aggregate(out, types.SimpleNamespace(**{**vars(opts), "agg_folds": "0,1,2"}), log)
    byr = X.aggregate_by_repeat(out, opts, log)
    assert list(byr) == [0] and (out / "by_repeat" / "rep0" / "aggregate_ext.json").exists()
    assert json.dumps(byr[0]["rows"], sort_keys=True, default=str) == json.dumps(agg["rows"], sort_keys=True, default=str)
    assert not agg["missing_references"]
    print(f"PASS test_ensembles_real ({n} x_ensembles.json values identical; reference ensembles and --ref-arm unit "
          f"identical to results_r3; by-repeat aggregation equal)")


def main():
    fast = "--fast" in sys.argv
    if "--ens" in sys.argv:
        TMP.mkdir(parents=True, exist_ok=True)
        try:
            test_x_ensemble_synthetic(); test_ensembles_real()
        finally:
            shutil.rmtree(TMP, ignore_errors=True)
        print("ALL PASS")
        return
    TMP.mkdir(parents=True, exist_ok=True)
    try:
        test_guard(); test_strip_cohort(); test_compare_json()
        test_cohort_offset_planted(); test_ridge_cohort(); test_pooled_recal_and_split(); test_hoc_planted()
        test_x_ensemble_synthetic()
        if not fast:
            test_x_fit_pipeline_real()
            test_ensembles_real()
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    print("ALL PASS")


if __name__ == "__main__":
    main()
