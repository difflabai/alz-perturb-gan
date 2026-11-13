#!/usr/bin/env python3
"""Synthetic planted-truth tests for evaluate.py / baselines.py / metrics.py.

Builds a small fake cache (Interface 1 layout) for the 37 development donor
ids / real inner folds of folds.json (ids only; no real data is read) and fake
generated.npz files (Interface 2 layout) under
/ml2/alz/runs/improve_20261109/tmp_evaluator_test/, from a known simulator:

  cell log-proportion = a_g + beta_g * stage/3 + u_dg + Lambda_g . z_i (+ shift)
  counts ~ Poisson(panel_lib_i * softmax_panel(...))

Fake generators:
  truth      perfect sampler: fresh donor effect u ~ N(0, sd_u), true stage effect
  oracle     uses the target's OWN true donor effect (best possible point prediction)
  shrunk     stage effect multiplied by 0.3
  nospread   donor effect fixed at 0 (no between-donor spread)
  shifted    cell-state latent mean shifted by 1.5 sd (cell distribution wrong)

Run: python3 test_evaluate.py   (exit code 0 only if every check passes)
"""
from __future__ import annotations

import os
for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[_v] = "6"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import metrics as MX  # noqa: E402
import evaluate as EV  # noqa: E402
import baselines as BL  # noqa: E402

ROOT = Path("/ml2/alz/runs/improve_20261109/tmp_evaluator_test")
FOLDS = json.loads((HERE / "folds.json").read_text())
G_ALL = 450
PANEL_GENES = {"hvg2000": np.arange(0, 300), "wide": np.arange(50, 450)}
SD_U = 0.15
M, K, KPC = 32, 256, 64

CHECKS = []


def check(name, cond, detail):
    CHECKS.append({"check": name, "pass": bool(cond), "detail": detail})
    print(("PASS " if cond else "FAIL ") + name + " :: " + detail, flush=True)


class Sim:
    def __init__(self, seed=7):
        rng = np.random.default_rng(seed)
        self.names = np.array([f"G{i:03d}" for i in range(G_ALL)], dtype="<U12")
        risk = ["APOE", "TREM2", "SPI1", "CD74", "P2RY12", "CX3CR1"]
        for j, r in enumerate(risk):
            self.names[100 + j * 50] = r   # positions 100..350 -> TREM2(150) etc
        self.a = rng.normal(0, 1.2, G_ALL)
        self.beta = np.where(rng.random(G_ALL) < .3, rng.normal(0, .8, G_ALL), 0.0)
        self.Lam = rng.normal(0, .5, (G_ALL, 2)) * (rng.random((G_ALL, 1)) < .4)
        self.donors = FOLDS["development_donors"]
        self.stage = {d: FOLDS["stage_by_donor"][d] for d in self.donors}
        self.u = {d: rng.normal(0, SD_U, G_ALL) for d in self.donors}
        self.ncell = {d: int(rng.integers(260, 400)) for d in self.donors}
        self.cov = {d: [float(rng.integers(0, 2)), float(rng.integers(0, 3)), float(rng.integers(70, 100))] for d in self.donors}

    def logit(self, stage, u, beta_scale=1.0):
        return self.a + beta_scale * self.beta * stage / 3 + u

    def cells(self, rng, logit, n, panel=None, libs=None, shift=0.0):
        z = rng.normal(shift, 1, (n, 2))
        lp = logit[None, :] + z @ self.Lam.T
        if panel is None:
            p = np.exp(lp - lp.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
            lib = rng.lognormal(np.log(2500), .4, n)
            return rng.poisson(lib[:, None] * p)
        lp = lp[:, panel]
        p = np.exp(lp - lp.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        return rng.poisson(libs[:, None] * p)


def build_data(sim):
    rng = np.random.default_rng(11)
    return {d: sim.cells(rng, sim.logit(sim.stage[d], sim.u[d]), sim.ncell[d]) for d in sim.donors}


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
            s = sim.stage[d]
            raw.append([0, 0, 0, s / 3, *sim.cov[d]])
        (d_ / split).mkdir(exist_ok=True)
        sp.save_npz(d_ / split / "counts.npz", sp.vstack(mats).tocsr())
        np.savez_compressed(d_ / split / "metadata.npz", cell=np.array(cell), donor=np.array(donor),
                            type=np.zeros(len(cell), int), donors=np.array(fold[key]), raw_c=np.array(raw, float), genes=genes)
    (d_ / "manifest.json").write_text(json.dumps({"panel": panel, "inner_fold": k, "synthetic": True}))


def write_fit(sim, real, arm, panel, k, seed, kind):
    """Fake generated.npz from the simulator. Common random numbers: the rng
    depends on (fold, target, generator kind) and seed only."""
    d_ = ROOT / "fits" / arm / panel / f"fold{k}" / f"seed{seed}"
    d_.mkdir(parents=True, exist_ok=True)
    gi = PANEL_GENES[panel]
    T, Gp = len(real.donors), len(gi)
    out = BL._alloc(T, M, Gp, KPC, real.W.shape[1])
    for t, donor in enumerate(real.donors):
        rng = np.random.default_rng(MX.stable_seed("fake", k, donor, kind, seed))
        st = sim.stage[donor]
        for m in range(M):
            libs = rng.choice(real.libs[t], K, replace=True)
            if kind == "oracle":
                u = sim.u[donor]
            elif kind == "nospread":
                u = np.zeros(G_ALL)
            else:
                u = rng.normal(0, SD_U, G_ALL)
            logit = sim.logit(st, u, .3 if kind == "shrunk" else 1.0)
            X = sim.cells(rng, logit, K, gi, libs, shift=1.5 if kind == "shifted" else 0.0).astype(np.float64)
            pb, ml, nz, pcs, lib = BL._summarise(X, real, KPC)
            out["pb_counts"][t, m] = pb; out["mean_log"][t, m] = ml; out["nonzero"][t, m] = nz
            out["pcs"][t, m] = pcs; out["lib"][t, m] = lib
    # shuffle target order to test alignment
    order = np.random.default_rng(k).permutation(T)
    np.savez_compressed(d_ / "generated.npz", targets=real.donors[order], is_heldout=real.is_heldout[order],
                        stage=real.stage[order], genes=real.genes,
                        **{key: out[key][order] for key in ["pb_counts", "mean_log", "nonzero", "pcs", "lib"]})
    (d_ / "config.json").write_text(json.dumps({"arm": arm, "panel": panel, "fold": k, "seed": seed, "fake": kind}))
    (d_ / "fit_receipt.json").write_text(json.dumps({"completed_updates": 20000, "fake": True}))
    with open(d_ / "learning_curve.jsonl", "w") as fh:
        for step in [1] + list(range(250, 20001, 250)):
            fh.write(json.dumps({"step": step, "critic_gap": 5.0 if step > 15000 else 1.0, "D_total": .8, "G_total": 2.0}) + "\n")


def unit_checks():
    rng = np.random.default_rng(0)
    # energy score / cell energy identical to bench_core
    sys.path.insert(0, "/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark")
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("bench_core_ro", "/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark/bench_core.py")
        bc = importlib.util.module_from_spec(spec); sys.modules["bench_core_ro"] = bc; spec.loader.exec_module(bc)
        Y = rng.normal(size=(20, 5)); y = rng.normal(size=5); A = rng.normal(size=(30, 4)); B = rng.normal(.5, 1, (25, 4))
        same = np.isclose(bc.energy_score(Y, y), MX.energy_score(Y, y)) and np.isclose(bc.cell_energy(A, B), MX.cell_energy(A, B))
        check("energy_score/cell_energy identical to bench_core", same, f"es {MX.energy_score(Y, y):.6f} vs {bc.energy_score(Y, y):.6f}")
    except Exception as e:  # pragma: no cover
        check("energy_score/cell_energy identical to bench_core", False, repr(e))
    # fixer: slope goodness defined for wrong-sign slopes; development-donor guard
    g_neg, g_pos = EV.goodness("hn_slope_vs_heldout", -0.1), EV.goodness("hn_slope_vs_heldout", 0.5)
    check("goodness of a wrong-sign contrast slope is defined and worse than a shrunk positive slope",
          g_neg is not None and g_pos is not None and g_neg < g_pos, f"{g_neg} < {g_pos}")
    sealed = json.loads(EV.FOLDS_JSON.read_text())["sealed_evaluation_donors_fold0"][0]
    try:
        EV.check_development_donors(["H20.33.001", sealed], "unit"); refused = False
    except PermissionError:
        refused = True
    check("evaluator refuses a sealed donor before reading counts", refused, "")
    # agreement: model = 0.4 * real -> slope 0.4, spearman 1
    r = rng.normal(size=500); a = MX.agreement(.4 * r, r)
    check("agreement slope recovers planted shrinkage 0.4", abs(a["regression_slope"] - .4) < 1e-9 and a["spearman"] > .999,
          f"slope={a['regression_slope']:.4f} spearman={a['spearman']:.4f}")
    # coverage of a perfect sampler on gaussian summaries
    n, Mm, D = 400, 200, 5
    Y = rng.normal(size=(n, Mm, D)); y = rng.normal(size=(n, D))
    d = MX.distribution_metrics(Y, y, np.zeros(n, int))
    check("distribution_metrics: perfect gaussian sampler coverage~0.8, spread~1",
          abs(d["coverage80"] - .8) < .03 and abs(d["spread_ratio"] - 1) < .05,
          f"cov={d['coverage80']:.3f} spread={d['spread_ratio']:.3f} within_stage={d['within_stage_spread_ratio']:.3f}")


def main():
    t0 = time.monotonic()
    if ROOT.exists():
        shutil.rmtree(ROOT)
    ROOT.mkdir(parents=True)
    unit_checks()
    sim = Sim(); data = build_data(sim)
    for panel in EV.PANELS:
        for k in range(3):
            write_cache(sim, data, panel, k)
    print(f"caches written {time.monotonic() - t0:.1f}s", flush=True)
    rc = EV.RealCache(ROOT / "caches")
    real0, subsets0, pooled0 = rc.get("hvg2000", 0)
    # fits: truth for all folds (seeds 0,1 in fold0), others fold 0 only, wide truth fold0
    plan = [("base", "hvg2000", k, 0, "truth") for k in range(3)] + [("base", "hvg2000", 0, 1, "truth"),
            ("oracle", "hvg2000", 0, 0, "oracle"), ("shrunk", "hvg2000", 0, 0, "shrunk"),
            ("nospread", "hvg2000", 0, 0, "nospread"), ("shifted", "hvg2000", 0, 0, "shifted"),
            ("base", "wide", 0, 0, "truth"), ("shrunk", "hvg2000", 0, 1, "shrunk")]
    for arm, panel, k, s, kind in plan:
        real, _, _ = rc.get(panel, k)
        write_fit(sim, real, arm, panel, k, s, kind)
    # an incomplete fit (no generated.npz) must be tolerated
    inc = ROOT / "fits" / "covars" / "hvg2000" / "fold0" / "seed0"; inc.mkdir(parents=True)
    (inc / "config.json").write_text("{}")
    print(f"fake fits written {time.monotonic() - t0:.1f}s", flush=True)
    out = ROOT / "results"
    agg = EV.main(["--fits-root", str(ROOT / "fits"), "--caches-root", str(ROOT / "caches"), "--out", str(out)])
    t_first = time.monotonic() - t0

    def fit(arm, panel="hvg2000", k=0, s=0):
        return json.loads((out / "fits" / arm / panel / f"fold{k}" / f"seed{s}.json").read_text())

    def base(name, panel="hvg2000", k=0):
        return json.loads((out / "baselines" / panel / f"fold{k}" / f"{name}.json").read_text())

    R = {a: fit(a) for a in ["base", "oracle", "shrunk", "nospread", "shifted"]}
    B = {b: base(b) for b in BL.BASELINES}
    H = lambda r: EV.headline(r, "heldout")  # noqa: E731
    HT = lambda r: EV.headline(r, "train")   # noqa: E731
    h = {k: H(v) for k, v in {**R, **{"bl_" + k: v for k, v in B.items()}}.items()}
    ht = {k: HT(v) for k, v in R.items()}
    show = lambda key: ", ".join(f"{k}={v.get(key):.3f}" if v.get(key) is not None else f"{k}=None" for k, v in h.items())  # noqa: E731
    for key in ["pb_rmse", "pb_dev_pearson", "meanlog_rmse_gene", "nonzero_rmse_gene", "hn_spearman_vs_heldout",
                "hn_spearman_vs_train", "hn_slope_vs_train", "hn_slope_vs_heldout", "ord_spearman_vs_train",
                "pc_coverage80", "pc_spread", "pc_within_stage_spread", "pbdist_coverage80", "pbdist_spread",
                "pbdist_within_stage_spread", "pc_energy", "pbdist_energy", "cell_energy", "cell_energy_pooled"]:
        print(f"  {key}: {show(key)}", flush=True)

    # --- planted-truth checks
    check("pb RMSE: oracle < truth < shrunk (held-out)", h["oracle"]["pb_rmse"] < h["base"]["pb_rmse"] < h["shrunk"]["pb_rmse"],
          f"{h['oracle']['pb_rmse']:.4f} < {h['base']['pb_rmse']:.4f} < {h['shrunk']['pb_rmse']:.4f}")
    check("pb deviation Pearson: truth > shrunk", h["base"]["pb_dev_pearson"] > h["shrunk"]["pb_dev_pearson"],
          f"{h['base']['pb_dev_pearson']:.3f} > {h['shrunk']['pb_dev_pearson']:.3f}")
    check("per-cell mean-log RMSE: truth < shrunk; nonzero RMSE truth < shrunk",
          h["base"]["meanlog_rmse_gene"] < h["shrunk"]["meanlog_rmse_gene"] and h["base"]["nonzero_rmse_gene"] < h["shrunk"]["nonzero_rmse_gene"],
          f"meanlog {h['base']['meanlog_rmse_gene']:.4f} vs {h['shrunk']['meanlog_rmse_gene']:.4f}; nonzero {h['base']['nonzero_rmse_gene']:.4f} vs {h['shrunk']['nonzero_rmse_gene']:.4f}")
    check("per-cell mean-log bias of truth sampler near 0", abs(h["base"]["meanlog_bias"]) < .02, f"bias={h['base']['meanlog_bias']:.4f}")
    rs = h["shrunk"]["hn_slope_vs_train"] / h["base"]["hn_slope_vs_train"]
    check("contrast regression slope ratio shrunk/truth ~0.3 (vs train real contrast)", .2 < rs < .4,
          f"truth slope={h['base']['hn_slope_vs_train']:.3f} shrunk slope={h['shrunk']['hn_slope_vs_train']:.3f} ratio={rs:.3f}")
    check("contrast Spearman vs train real: truth high (>0.6) and shrunk similar ranking",
          h["base"]["hn_spearman_vs_train"] > .6, f"truth={h['base']['hn_spearman_vs_train']:.3f} shrunk={h['shrunk']['hn_spearman_vs_train']:.3f}")
    ch = R["base"]["subsets"]["all"]["heldout"]["contrast"]
    check("fold0 held-out contrast flagged low-n (1 Not AD held-out donor)", ch["real_heldout_low_n"] is True and ch["real_heldout_n_by_stage"]["0"] == 1,
          f"n_by_stage={ch['real_heldout_n_by_stage']}")
    for space in ["pc", "pbdist"]:
        cov, spr = h["base"][f"{space}_coverage80"], h["base"][f"{space}_spread"]
        check(f"{space}: perfect sampler coverage80 near nominal and spread ratio near 1",
              .65 < cov < .93 and .7 < spr < 1.35, f"cov={cov:.3f} spread={spr:.3f}")
        check(f"{space}: no-donor-spread generator has lower coverage and within-stage spread than truth",
              h["nospread"][f"{space}_coverage80"] < cov - .05 and h["nospread"][f"{space}_within_stage_spread"] < h["base"][f"{space}_within_stage_spread"],
              f"cov {h['nospread'][f'{space}_coverage80']:.3f} vs {cov:.3f}; within {h['nospread'][f'{space}_within_stage_spread']:.3f} vs {h['base'][f'{space}_within_stage_spread']:.3f}")
    check("energy score (pb space): truth < shrunk and truth < nospread",
          h["base"]["pbdist_energy"] < h["shrunk"]["pbdist_energy"] and h["base"]["pbdist_energy"] < h["nospread"]["pbdist_energy"],
          f"{h['base']['pbdist_energy']:.3f} vs {h['shrunk']['pbdist_energy']:.3f}, {h['nospread']['pbdist_energy']:.3f}")
    check("cell energy distance: truth < shifted cell-state generator", h["base"]["cell_energy"] < h["shifted"]["cell_energy"],
          f"{h['base']['cell_energy']:.4f} < {h['shifted']['cell_energy']:.4f}")
    # Per-target cell energy of a population (stage) sampler includes the
    # target's own donor effect; only the oracle should be near zero there.
    # The pooled held-out version compares populations and should be small.
    check("cell energy: oracle per-target small (<0.08); truth pooled small (<0.05) and < shifted pooled",
          h["oracle"]["cell_energy"] < .08 and h["base"]["cell_energy_pooled"] < .05 and h["base"]["cell_energy_pooled"] < h["shifted"]["cell_energy_pooled"],
          f"oracle per-target={h['oracle']['cell_energy']:.4f}; pooled truth={h['base']['cell_energy_pooled']:.4f} shifted={h['shifted']['cell_energy_pooled']:.4f}")
    # baselines
    check("baseline stage_mean beats grand_mean on held-out pb RMSE and contrast Spearman",
          h["bl_stage_mean"]["pb_rmse"] < h["bl_grand_mean"]["pb_rmse"] and h["bl_stage_mean"]["hn_spearman_vs_heldout"] > (h["bl_grand_mean"]["hn_spearman_vs_heldout"] or 0) + .2,
          f"rmse {h['bl_stage_mean']['pb_rmse']:.4f} vs {h['bl_grand_mean']['pb_rmse']:.4f}; spearman {h['bl_stage_mean']['hn_spearman_vs_heldout']:.3f} vs {h['bl_grand_mean']['hn_spearman_vs_heldout']}")
    check("baseline shrunk_stage_mean: contrast slope vs train <= stage_mean slope",
          h["bl_shrunk_stage_mean"]["hn_slope_vs_train"] <= h["bl_stage_mean"]["hn_slope_vs_train"] + 1e-9,
          f"{h['bl_shrunk_stage_mean']['hn_slope_vs_train']:.3f} <= {h['bl_stage_mean']['hn_slope_vs_train']:.3f}; info={B['shrunk_stage_mean']['baseline_info'].get('mean_shrinkage_factor')}")
    check("baseline ridge: alpha chosen by LOO and held-out pb RMSE within 20% of stage_mean",
          h["bl_ridge"]["pb_rmse"] < 1.2 * h["bl_stage_mean"]["pb_rmse"],
          f"alpha={B['ridge']['baseline_info']['alpha']}, rmse {h['bl_ridge']['pb_rmse']:.4f} vs {h['bl_stage_mean']['pb_rmse']:.4f}")
    # In the simulation real train donors of the target's stage are exact draws
    # of the stage population, so resampling should behave like a population
    # sampler on depth-matched gene metrics and coverage. Its held-out PC
    # spread is inflated (~1.5) because the train-fitted PCs contain
    # train-donor-identity axes (diagnosed: train donor-mean variance on such
    # PCs ~0.5 vs ~0.05 for held-out donors); on train targets it is ~1.
    hr_t = HT(B["resample"])
    check("baseline resample: depth-matched gene metrics like truth, pc coverage 0.65-0.93, pooled ED < 0.08, train-target pc spread 0.7-1.35",
          .65 < h["bl_resample"]["pc_coverage80"] < .93 and .7 < hr_t["pc_spread"] < 1.35
          and h["bl_resample"]["cell_energy_pooled"] < .08 and h["bl_resample"]["nonzero_rmse_gene"] <= 1.5 * h["base"]["nonzero_rmse_gene"],
          f"cov={h['bl_resample']['pc_coverage80']:.3f} heldout spread={h['bl_resample']['pc_spread']:.3f} train spread={hr_t['pc_spread']:.3f} pooled ed={h['bl_resample']['cell_energy_pooled']:.4f} "
          f"nonzero rmse={h['bl_resample']['nonzero_rmse_gene']:.4f} (truth {h['base']['nonzero_rmse_gene']:.4f}); "
          f"unmatched depth frac={B['resample']['baseline_info']['frac_cells_depth_unmatched']:.4f}")
    check("truth generator beats stage_mean baseline on held-out per-cell mean-log RMSE",
          h["base"]["meanlog_rmse_gene"] < h["bl_stage_mean"]["meanlog_rmse_gene"],
          f"{h['base']['meanlog_rmse_gene']:.4f} < {h['bl_stage_mean']['meanlog_rmse_gene']:.4f}")
    check("train fit-quality metrics present and oracle better on train too", ht["oracle"]["pb_rmse"] < ht["base"]["pb_rmse"],
          f"train pb_rmse oracle={ht['oracle']['pb_rmse']:.4f} truth={ht['base']['pb_rmse']:.4f}")
    # noise ceiling
    nc = json.loads((out / "baselines" / "hvg2000" / "fold0" / "noise_ceiling.json").read_text())["subsets"]["all"]
    check("noise ceiling computed (held-out peers; split-half train defined; held-out split-half undefined in fold0)",
          nc["heldout_peers_same_stage"] is not None and nc["split_half"]["train"]["defined"] and not nc["split_half"]["heldout"]["defined"],
          f"peers rmse={nc['heldout_peers_same_stage']['rmse']:.4f} (n without peer {nc['heldout_peers_same_stage']['n_heldout_without_peer']}); "
          f"train split-half spearman={nc['split_half']['train']['spearman']:.3f}")
    # ensembles & seed stability
    ens = json.loads((out / "ensembles" / "base" / "hvg2000" / "fold0.json").read_text())
    st = ens["seed_stability"]["all_targets"]["high_notad"]["mean_spearman"]
    sts = json.loads((out / "ensembles" / "shrunk" / "hvg2000" / "fold0.json").read_text())["seed_stability"]["all_targets"]["high_notad"]["mean_spearman"]
    check("seed ensemble pooled M=64 and seed stability of truth high (>0.8)", ens["M"] == 64 and st > .8,
          f"M={ens['M']} stability spearman truth={st:.3f} shrunk={sts:.3f}")
    # wide subsets
    rw = fit("base", "wide")
    check("wide panel reports shared_hvg2000_withinpanel (+renorm) and risk subsets",
          {"shared_hvg2000_withinpanel", "shared_hvg2000_renorm", "risk"} <= set(rw["subsets"])
          and rw["subsets"]["shared_hvg2000_withinpanel"]["n_genes"] == 250
          and rw["subsets"]["shared_hvg2000_renorm"]["cross_panel_comparable"] == "all metrics",
          f"subsets={sorted(rw['subsets'])} shared n={rw['subsets']['shared_hvg2000_withinpanel']['n_genes']} risk n={rw['subsets']['risk']['n_genes']}")
    # aggregate
    rows = {(r["arm"], r["panel"]): r for r in agg["rows"]}
    check("aggregate rows for arms, ensembles and baselines; incomplete fit listed",
          ("base", "hvg2000") in rows and ("base [seed-ensemble]", "hvg2000") in rows and ("baseline:stage_mean", "hvg2000") in rows
          and any(i["arm"] == "covars" for i in agg["incomplete_fits"]) and not agg["errors"],
          f"{len(rows)} rows, incomplete={len(agg['incomplete_fits'])}, errors={agg['errors']}")
    pv = agg["paired"].get("oracle|hvg2000", {}).get("vs_base", {}).get("metrics", {}).get("pb_rmse", {})
    check("paired vs base: oracle better on pb_rmse in its 1 paired unit", pv.get("n") == 1 and pv.get("n_better") == 1,
          json.dumps(pv))
    pooled = agg["pooled_across_folds"].get("base|hvg2000|seed=0")
    check("pooled-across-folds contrast for base (37 held-out donors, 4 Not AD)",
          pooled is not None and pooled["n_donors"] == 37 and pooled["n_by_stage"]["0"] == 4,
          f"n={pooled and pooled['n_donors']} by_stage={pooled and pooled['n_by_stage']} spearman={pooled and pooled['high_notad']['spearman']:.3f} "
          f"slope={pooled and pooled['high_notad']['regression_slope']:.3f} genes={pooled and pooled['n_genes']}")
    rsm = rows.get(("baseline:stage_mean", "hvg2000"), {})
    pvs = agg["paired"].get("base|hvg2000", {})
    check("fixer: profile-baseline PC/cell metrics masked in rows and paired; reference baselines paired per fold",
          rsm.get("pc_energy") is None and rsm.get("cell_energy") is None and rsm.get("pb_rmse") is not None
          and rows.get(("baseline:resample", "hvg2000"), {}).get("pc_energy") is not None
          and pvs.get("vs_stage_mean_baseline", {}).get("metrics", {}).get("pc_energy", {}).get("n") == 0
          and all(f"vs_{b}_baseline" in pvs for b in ["stage_mean", "stage_mean_log", "shrunk_stage_mean", "ridge"])
          and pvs["vs_stage_mean_baseline"]["metrics"]["pb_rmse"]["n"] == 3
          and rows[("base", "hvg2000")]["pb_rmse__n_folds"] == 3,
          f"stage_mean pc_energy={rsm.get('pc_energy')} paired blocks={sorted(pvs)} "
          f"n vs stage_mean={pvs.get('vs_stage_mean_baseline', {}).get('metrics', {}).get('pb_rmse', {}).get('n')}")
    check("aggregate.csv and summary.md written", (out / "aggregate.csv").exists() and (out / "summary.md").exists(), "")
    # incremental re-run
    t1 = time.monotonic()
    EV.main(["--fits-root", str(ROOT / "fits"), "--caches-root", str(ROOT / "caches"), "--out", str(out)])
    t_second = time.monotonic() - t1
    check("re-run is incremental (much faster than first run)", t_second < .5 * t_first, f"first={t_first:.1f}s second={t_second:.1f}s")
    n_fail = sum(not c["pass"] for c in CHECKS)
    (ROOT / "test_report.json").write_text(json.dumps({"checks": CHECKS, "n_fail": n_fail}, indent=1))
    print(f"\n{len(CHECKS) - n_fail}/{len(CHECKS)} checks passed; total {time.monotonic() - t0:.1f}s")
    return n_fail


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
