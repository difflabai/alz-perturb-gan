#!/usr/bin/env python3
"""improve_20261109 round 3, R6 (stage task): failure analysis of the round-2 stage and covariate
arms and an error decomposition of the best round-2 read-outs.  CPU only, existing fits only.

Usage (from /tmp, 3 threads)::

    python3 -B r3/stage_failures.py anchor   [--out DIR]   # stageanchor, genemean,stageanchor
    python3 -B r3/stage_failures.py cov      [--out DIR]   # covbin, covdrop@0.5 (memorisation)
    python3 -B r3/stage_failures.py decomp   [--out DIR]   # T1 gap: level bias vs stage vs donor vs seed
    python3 -B r3/stage_failures.py tau2     [--out DIR]   # offset_init tau2 with a stage-permutation null
    python3 -B r3/stage_failures.py all      [--out DIR]

Default --out: /ml2/alz/runs/improve_20261109/smoke_r3/stage/analysis (small JSON files only).

Inputs (read-only): round-1 fits ``runs/improve_20261109/fits`` (base, genemean), round-2 fits
``fits_r2`` (only fits with generated.npz), the hvg2000 fold caches (37 development donors; evaluate.
load_real refuses anything else).  Every prediction is built EXACTLY as the evaluator does
(``posthoc.predictions_from_gen``: P[t] = mean over the M generated donors of log1p CP10k of the
generated pseudobulk; O[t] = log1p CP10k of the observed full pseudobulk).

Definitions
-----------
* Held-out error decomposition (``posthoc.decompose``; d = P - O over held-out donors x genes):
  MSE = shared + stage + donor, shared = mean_g (mean_t d)^2 (per-gene LEVEL bias common to every
  held-out donor), stage = what the held-out stage means of d add, donor = the rest.  For a
  predictor that is constant within stage, donor >= floor = mean_t mean_g (O_t - Obar_{s(t)})^2 with
  held-out stage means (``posthoc.floors``: the donor-specific error no stage-level predictor can
  remove).  ``donor_excess`` = donor - floor (the model's own between-donor scatter: latent, depth,
  Monte Carlo).
* Seed noise (fold 0, where seed 1 exists): delta = (P_seed0 - P_seed1)/sqrt(2) estimates one seed's
  deviation from the infinite-seed mean (it also contains the part of the Monte Carlo noise that the
  common random numbers do not share).  Its shared/stage/donor parts (same partition) are the seed
  share of each term of a single-seed MSE.
* Read-out variants computed from the SAME predictions (no regeneration; train targets only):
  ``raw``; ``blind`` = level_P = mean over train targets of P (stage-blind: the model's train-weighted
  stage marginal, what condmix@1 generates up to library/Monte Carlo noise); ``half`` = level_P + D/2
  (D = model stage profile minus level_P on train targets; the analogue of condmix@0.5 = stagehalf);
  ``recal`` = P + b (b = mean_tr (O - P), the per-gene level shift biascal targets);
  ``recal_half`` = level_P + b + D/2.
* Identification rate (round-1 review definition): TRAIN targets; the fraction whose predicted
  deviation (P - train grand mean of O) correlates best with its OWN observed deviation among the
  train donors of the same stage (chance = mean 1/n_stage).
* Stage-deviation size: RMS over train targets x genes of the model's D and of the real train stage
  deviation (O stage means minus O train mean).

Findings are returned in the stage task's structured output; the JSON files hold every number.
"""
from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "3"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.append(str(HERE))          # appended: round-1 metrics/evaluate must win inside posthoc
import posthoc as PH  # noqa: E402  (r3 copy; imports the frozen round-1 metrics/evaluate/baselines)

E, MX = PH.E, PH.MX
RUNS = Path("/ml2/alz/runs/improve_20261109")
R1 = RUNS / "fits"
R2 = RUNS / "fits_r2"
CACHES = RUNS / "caches" / "hvg2000"
DEFAULT_OUT = RUNS / "smoke_r3" / "stage" / "analysis"
PANEL = "hvg2000"

_REAL = {}
_FOLD = {}


def fold_data(k):
    if k not in _FOLD:
        real = E.load_real(CACHES / f"fold{k}", PANEL, k)
        real.cells = real.pcs = real.libs = None          # keep memory small
        _REAL[k] = real
        _FOLD[k] = PH.Fold(real)
    return _REAL[k], _FOLD[k]


def gen_path(root, arm, k, s, mode=None):
    return Path(root) / arm / PANEL / f"fold{k}" / f"seed{s}" / ("generated.npz" if mode is None else f"generated_{mode}.npz")


def predictions(path, k):
    real, _ = fold_data(k)
    gen = E.align_generated(E.load_generated(path), real)
    P, ML = PH.predictions_from_gen(gen)
    return P, ML


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def corr(a, b):
    a = np.ravel(a) - np.mean(a); b = np.ravel(b) - np.mean(b)
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else None


def stage_parts(P, f):
    """level_P, model stage deviation D (per target) and within-stage residual R on TRAIN targets."""
    tr = f.train
    level = P[tr].mean(0)
    Sm = {int(s): P[tr & (f.stage == s)].mean(0) for s in np.unique(f.stage[tr])}
    D = np.stack([Sm[int(s)] - level if int(s) in Sm else np.zeros_like(level) for s in f.stage])
    return level, D, P - level - D


def variants(P, f):
    level, D, _ = stage_parts(P, f)
    b = (f.O[f.train] - P[f.train]).mean(0)
    T = len(P)
    return {"raw": P, "blind": np.tile(level, (T, 1)), "half": level + 0.5 * D, "recal": P + b,
            "recal_half": level + b + 0.5 * D}


def decomp_row(P, f, floor_mse):
    held = ~f.train
    d = PH.decompose(P, f.O, f.stage, held)
    d["pb_rmse"] = PH.pb_rmse(P, f.O, held)
    d["pb_rmse_train"] = PH.pb_rmse(P, f.O, f.train)
    d["floor"] = floor_mse
    d["donor_excess"] = d["donor"] - floor_mse
    dt = PH.decompose(P, f.O, f.stage, f.train)
    d["train_shared"] = dt["shared"]
    return d


def seed_parts(P0, P1, f):
    held = ~f.train
    delta = (P0 - P1) / np.sqrt(2.0)
    st = f.stage[held]
    x = delta[held]
    eb = x.mean(0)
    es = {s: x[st == s].mean(0) for s in np.unique(st)}
    stage_part = np.stack([es[s] - eb for s in st])
    donor_part = x - np.stack([es[s] for s in st])
    return {"seed_mse": float((x ** 2).mean()), "seed_shared": float((eb ** 2).mean()),
            "seed_stage": float((stage_part ** 2).mean()), "seed_donor": float((donor_part ** 2).mean())}


def ident(P, f):
    tr = np.flatnonzero(f.train)
    grand = f.O[f.train].mean(0)
    hits, chance = [], []
    for i in tr:
        cand = tr[f.stage[tr] == f.stage[i]]
        if len(cand) < 2:
            continue
        c = [corr(P[i] - grand, f.O[j] - grand) for j in cand]
        hits.append(cand[int(np.argmax(c))] == i); chance.append(1 / len(cand))
    return float(np.mean(hits)), float(np.mean(chance))


def lc_gap(fitdir, window=5000):
    p = Path(fitdir) / "learning_curve.jsonl"
    if not p.exists():
        return None
    r = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    last = max(x["step"] for x in r)
    v = [x["critic_gap"] for x in r if x["step"] > last - window and np.isfinite(x.get("critic_gap", np.nan))]
    return float(np.mean(v)) if v else None


def lc_series(fitdir, key="critic_gap", every=2000):
    p = Path(fitdir) / "learning_curve.jsonl"
    r = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    return {int(x["step"]): round(float(x[key]), 3) for x in r if x["step"] % every == 0 and key in x}


# ---------------------------------------------------------------------------- stage log-ratio (offset_init)
def stage_logratio(C, stages, a=1.0, tau2_floor=1e-6):
    """numpy copy of arms.offset_init returning its parts: r_s [4,G], v [G], e [4], tau2 (and unfloored)."""
    stages = np.asarray(stages, int)
    n, G = C.shape
    N = C.sum(1)
    logq = np.log(C.sum(0) + a) - np.log(N.sum() + a * G)
    r_d = np.log(C + a) - np.log(N[:, None] + a * G) - logq
    m = r_d.mean(0)
    n_s = np.array([(stages == s).sum() for s in range(4)])
    present = [s for s in range(4) if n_s[s] > 0]
    r_s = np.zeros((4, G)); ss = np.zeros(G)
    for s in present:
        msk = stages == s
        r_s[s] = r_d[msk].mean(0) - m
        ss += ((r_d[msk] - r_d[msk].mean(0)) ** 2).sum(0)
    v = ss / (n - len(present))
    e = np.array([1.0 / n_s[s] - 1.0 / n if n_s[s] else 0.0 for s in range(4)])
    tau2_raw = float(np.mean([r_s[s] ** 2 - v * e[s] for s in present]))
    tau2 = max(tau2_floor, tau2_raw)
    w = np.zeros((4, G))
    for s in present:
        w[s] = tau2 / (tau2 + v * e[s])
    return {"r_s": r_s, "v": v, "e": e, "tau2_raw": tau2_raw, "tau2": tau2, "w": w, "table": w * r_s, "n_s": n_s}


def train_donor_counts(real):
    tr = ~real.is_heldout
    return real.pb[tr].astype(np.float64), real.stage[tr]


def run_tau2(out, n_perm=500):
    res = {}
    for k in range(3):
        real, f = fold_data(k)
        C, st = train_donor_counts(real)
        obs = stage_logratio(C, st)
        rng = np.random.default_rng(np.random.SeedSequence([20261109, 6, k]))
        null = np.array([stage_logratio(C, rng.permutation(st))["tau2_raw"] for _ in range(n_perm)])
        res[k] = {"n_train_by_stage": obs["n_s"].tolist(), "tau2_raw_observed": obs["tau2_raw"],
                  "tau2_used": obs["tau2"], "perm_null_mean": float(null.mean()), "perm_null_sd": float(null.std()),
                  "perm_p_ge": float((1 + (null >= obs["tau2_raw"]).sum()) / (1 + n_perm)),
                  "mean_weight_per_stage": [float(obs["w"][s].mean()) for s in range(4)],
                  "raw_rms_per_stage": [rms(obs["r_s"][s]) for s in range(4)],
                  "shrunk_rms_per_stage": [rms(obs["table"][s]) for s in range(4)],
                  "mean_within_stage_var": float(obs["v"].mean())}
        print("tau2 fold", k, {a: (round(b, 6) if isinstance(b, float) else b) for a, b in res[k].items()}, flush=True)
    write(out, "tau2.json", res)
    return res


# ---------------------------------------------------------------------------- anchor
def run_anchor(out):
    import torch
    res = {}
    for arm in ("stageanchor", "genemean,stageanchor"):
        ref_arm = "base" if arm == "stageanchor" else "genemean"
        for k, s in ((0, 0), (0, 1), (1, 0), (2, 0)):
            fit = R2 / arm / PANEL / f"fold{k}" / f"seed{s}"
            if not (fit / "generated.npz").exists():
                continue
            real, f = fold_data(k)
            floor = PH.floors(f)["floor_insample_mse"]
            q = torch.load(fit / "final.pt", map_location="cpu", weights_only=False)
            g = q["policy"]["models"]["generator"]
            A = g["stage_anchor"].double().numpy(); gain = g["stage_gain"].double().numpy()
            C, st = train_donor_counts(real)
            lr = stage_logratio(C, st)
            snaps = {}
            for u in (16000, 17000, 18000, 19000):
                sp_ = fit / f"snap_{u}.pt"
                if sp_.exists():
                    snaps[u] = [round(float(x), 3) for x in torch.load(sp_, map_location="cpu", weights_only=False)["generator"]["stage_gain"]]
            eff = gain[:, None] * A
            # held-out stage deviation in log1p CP10k (evaluation only) and train real stage deviation
            O = f.O; tr = f.train; held = ~tr
            grand_tr = O[tr].mean(0)
            real_dev_tr = {int(x): O[tr & (f.stage == x)].mean(0) - grand_tr for x in np.unique(f.stage[tr])}
            real_dev_ho = {int(x): O[held & (f.stage == x)].mean(0) - O[held].mean(0) for x in np.unique(f.stage[held])}
            row = {"gain_final": gain.round(3).tolist(), "gain_snapshots": snaps,
                   "anchor_equals_offset_init": bool(np.allclose(A, lr["table"], atol=1e-6, rtol=1e-4)),
                   "tau2": lr["tau2"], "tau2_raw": lr["tau2_raw"],
                   "anchor_rms_per_stage": [rms(A[x]) for x in range(4)],
                   "raw_logratio_rms_per_stage": [rms(lr["r_s"][x]) for x in range(4)],
                   "effective_offset_rms_per_stage": [rms(eff[x]) for x in range(4)],
                   "effective_over_raw_rms": [rms(eff[x]) / rms(lr["r_s"][x]) if rms(lr["r_s"][x]) > 0 else None for x in range(4)],
                   "corr_anchor_vs_heldout_stage_dev": {x: corr(A[x], real_dev_ho[x]) for x in real_dev_ho},
                   }
            units = {"raw": gen_path(R2, arm, k, s), "condmix@1": gen_path(R2, arm, k, s, "condmix@1"),
                     "snapens": gen_path(R2, arm, k, s, "snapens")}
            refp = gen_path(R1 if ref_arm == "base" else R2, ref_arm, k, s)
            Pref, _ = predictions(refp, k)
            lvl_ref, D_ref, R_ref = stage_parts(Pref, f)
            b_ref = (O[tr] - Pref[tr]).mean(0)
            for name, p in units.items():
                if not p.exists():
                    continue
                P, ML = predictions(p, k)
                lvl, D, Rw = stage_parts(P, f)
                b = (O[tr] - P[tr]).mean(0)
                hit, ch = ident(P, f)
                row[name] = {**decomp_row(P, f, floor),
                             "model_stage_dev_rms_train": rms(D[tr]), "real_stage_dev_rms_train": rms(np.stack([real_dev_tr[int(x)] for x in f.stage[tr]])),
                             "within_stage_pred_spread_rms_train": rms(Rw[tr]),
                             "level_bias_rms_train": rms(b), "corr_level_bias_with_ref_arm": corr(b, b_ref),
                             "meanlog_rmse_gene_heldout": PH.meanlog_rmse_gene(ML, f.RML, held),
                             "ident_train": hit, "ident_chance": ch}
            row["ref_arm"] = ref_arm
            row["ref_raw"] = {**decomp_row(Pref, f, floor), "model_stage_dev_rms_train": rms(D_ref[tr]),
                              "within_stage_pred_spread_rms_train": rms(R_ref[tr]), "level_bias_rms_train": rms(b_ref)}
            row["critic_gap_last5000"] = lc_gap(fit)
            row["ref_critic_gap_last5000"] = lc_gap(refp.parent)
            row["critic_gap_curve"] = lc_series(fit)
            res[f"{arm}|fold{k}|seed{s}"] = row
            print(arm, k, s, "gain", row["gain_final"], "tau2", round(lr["tau2"], 7), "eff/raw", [None if v is None else round(v, 2) for v in row["effective_over_raw_rms"]],
                  "pb", round(row["raw"]["pb_rmse"], 3), "blind", round(row["condmix@1"]["pb_rmse"], 3) if "condmix@1" in row else None,
                  "ref", round(row["ref_raw"]["pb_rmse"], 3), "shared", round(row["raw"]["shared"], 3), "ref shared", round(row["ref_raw"]["shared"], 3),
                  "gap", row["critic_gap_last5000"], row["ref_critic_gap_last5000"], flush=True)
    write(out, "anchor.json", res)
    return res


# ---------------------------------------------------------------------------- covariates
def cov_groups(real):
    """(stage, sex, APOE4 carrier, age tertile) tuple sizes among train donors (covbin coding, train thresholds)."""
    tr = ~real.is_heldout
    raw = real.raw_c
    t1, t2 = np.quantile(raw[tr, 6], [1 / 3, 2 / 3])
    tup = [(int(real.stage[i]), int(raw[i, 4]), int(raw[i, 5] >= 1), int(raw[i, 6] > t1) + int(raw[i, 6] > t2)) for i in range(len(raw))]
    trs = [tup[i] for i in np.flatnonzero(tr)]
    size = {t: trs.count(t) for t in set(trs)}
    return tup, size


def run_cov(out):
    res = {}
    for k, s in ((0, 0), (0, 1), (1, 0), (2, 0)):
        real, f = fold_data(k)
        floor = PH.floors(f)["floor_insample_mse"]
        tup, size = cov_groups(real)
        tr_idx = np.flatnonzero(f.train)
        single = np.array([size[tup[i]] == 1 for i in tr_idx])
        held = ~f.train
        seen = [size.get(tup[i], 0) for i in np.flatnonzero(held)]
        fold_info = {"train_unique_stage_tuples": len(size), "train_singleton_donors": int(single.sum()), "n_train": int(len(tr_idx)),
                     "heldout_tuple_train_count": seen}
        Pb, _ = predictions(gen_path(R1, "base", k, s), k)
        units = {"base": Pb}
        for arm, modes in (("covbin", (None, "margcov", "condmix@1", "snapens")),
                           ("covdrop@0.5", (None, "owncov", "margcov", "condmix@1", "snapens"))):
            for m in modes:
                p = gen_path(R2, arm, k, s, m)
                if p.exists():
                    units[arm + ("" if m is None else "#" + m)] = predictions(p, k)[0]
        for u, P in units.items():
            hit, ch = ident(P, f)
            _, D, Rw = stage_parts(P, f)
            # identification restricted to singleton (stage, tuple) donors vs shared-tuple donors
            tri = tr_idx
            grand = f.O[f.train].mean(0)
            hs = {True: [], False: []}
            for j, i in enumerate(tri):
                cand = tri[f.stage[tri] == f.stage[i]]
                if len(cand) < 2:
                    continue
                c = [corr(P[i] - grand, f.O[x] - grand) for x in cand]
                hs[bool(single[j])].append(cand[int(np.argmax(c))] == i)
            r = {**decomp_row(P, f, floor), "ident_train": hit, "ident_chance": ch,
                 "ident_singleton_tuple_donors": float(np.mean(hs[True])) if hs[True] else None,
                 "ident_shared_tuple_donors": float(np.mean(hs[False])) if hs[False] else None,
                 "within_stage_pred_spread_rms_train": rms(Rw[f.train]), "model_stage_dev_rms_train": rms(D[f.train]),
                 "train_heldout_gap": PH.pb_rmse(P, f.O, held) - PH.pb_rmse(P, f.O, f.train)}
            res.setdefault(f"fold{k}|seed{s}", {"fold": fold_info})[u] = r
            print(k, s, u.ljust(22), "pb %.3f tr %.3f ident %.2f (single %s shared %s, chance %.2f) spread %.3f" % (
                r["pb_rmse"], r["pb_rmse_train"], hit, None if r["ident_singleton_tuple_donors"] is None else round(r["ident_singleton_tuple_donors"], 2),
                None if r["ident_shared_tuple_donors"] is None else round(r["ident_shared_tuple_donors"], 2), ch, r["within_stage_pred_spread_rms_train"]), flush=True)
        print("fold info", fold_info, flush=True)
    write(out, "cov.json", res)
    return res


# ---------------------------------------------------------------------------- decomposition
DECOMP_UNITS = [  # (label, root, arm, mode)
    ("base", R1, "base", None),
    ("genemean(r1)", R1, "genemean", None),
    ("genemean@0.25#snapens", R2, "genemean@0.25", "snapens"),
    ("pbulk#snapens", R2, "pbulk", "snapens"),
    ("genemean@0.5#snapens", R2, "genemean@0.5", "snapens"),
    ("genemean~r2#snapens", R2, "genemean", "snapens"),
    ("genemean@0.25#condmix@1", R2, "genemean@0.25", "condmix@1"),
    ("pbulk#condmix@1", R2, "pbulk", "condmix@1"),
    ("genemean@0.5#condmix@1", R2, "genemean@0.5", "condmix@1"),
    ("genemean@0.25#condmix@0.5", R2, "genemean@0.25", "condmix@0.5"),
    ("pbulk#condmix@0.5", R2, "pbulk", "condmix@0.5"),
    ("genemean@0.5", R2, "genemean@0.5", None),
    ("stagedrop@0.5", R2, "stagedrop@0.5", None),
]
TERMS = ("mse", "shared", "stage", "donor", "floor", "donor_excess")


def run_decomp(out):
    res = {"per_fold": {}, "fold_mean": {}, "seed_fold0": {}}
    for k in range(3):
        real, f = fold_data(k)
        floor = PH.floors(f)["floor_insample_mse"]
        tm = np.tile(f.O[f.train].mean(0), (len(f.O), 1))
        rows = {"train_mean": decomp_row(tm, f, floor)}
        for label, root, arm, mode in DECOMP_UNITS:
            p = gen_path(root, arm, k, 0, mode)
            if not p.exists():
                continue
            P, _ = predictions(p, k)
            for vname, V in variants(P, f).items():
                rows[f"{label}|{vname}"] = decomp_row(V, f, floor)
            if k == 0:
                p1 = gen_path(root, arm, k, 1, mode)
                if p1.exists():
                    P1, _ = predictions(p1, k)
                    sv = {}
                    for vname in ("raw", "blind", "half", "recal", "recal_half"):
                        V0 = variants(P, f)[vname]; V1 = variants(P1, f)[vname]
                        sv[vname] = {**seed_parts(V0, V1, f), "seed1": decomp_row(V1, f, floor),
                                     "two_seed_mean": decomp_row((V0 + V1) / 2, f, floor)}
                    res["seed_fold0"][label] = sv
        res["per_fold"][k] = rows
        for name, r in rows.items():
            if name == "train_mean" or name.endswith("|raw") or name.endswith("|half") or name.endswith("|blind") or name.endswith("|recal_half"):
                print(k, name.ljust(36), "pb %.3f mse %.3f shared %.3f stage %.3f donor %.3f (floor %.3f excess %.3f)" % tuple(
                    r[x] for x in ("pb_rmse", "mse", "shared", "stage", "donor", "floor", "donor_excess")), flush=True)
    names = set.intersection(*[set(v) for v in res["per_fold"].values()])
    for n in sorted(names):
        res["fold_mean"][n] = {t: float(np.mean([res["per_fold"][k][n][t] for k in range(3)])) for t in TERMS + ("pb_rmse", "pb_rmse_train", "train_shared")}
    tm = res["fold_mean"]["train_mean"]
    gap = {}
    for n, r in res["fold_mean"].items():
        gap[n] = {"pb_rmse": r["pb_rmse"], "pb_gap_to_train_mean": r["pb_rmse"] - tm["pb_rmse"],
                  "mse_gap": r["mse"] - tm["mse"], "shared_gap": r["shared"] - tm["shared"],
                  "stage_gap": r["stage"] - tm["stage"], "donor_excess_gap": r["donor_excess"] - tm["donor_excess"]}
    res["gap_vs_train_mean"] = gap
    print("\nfold-mean gap vs train mean (MSE units): unit | pb | mse_gap = shared + stage + donor_excess")
    for n in sorted(gap, key=lambda x: gap[x]["pb_rmse"]):
        g = gap[n]
        print(n.ljust(38), "%.3f | %+.4f = %+.4f %+.4f %+.4f" % (g["pb_rmse"], g["mse_gap"], g["shared_gap"], g["stage_gap"], g["donor_excess_gap"]))
    print("\nfold 0 seed noise share (single seed): unit|variant  mse  seed_mse (shared/stage/donor)")
    for lab, sv in res["seed_fold0"].items():
        for vn in ("raw", "half", "blind", "recal_half"):
            x = sv[vn]
            r0 = res["per_fold"][0][f"{lab}|{vn}"]
            print(f"{lab}|{vn}".ljust(38), "mse %.4f shared %.4f stage %.4f donor %.4f | seed %.4f (%.4f/%.4f/%.4f) | 2-seed pb %.3f" % (
                r0["mse"], r0["shared"], r0["stage"], r0["donor"], x["seed_mse"], x["seed_shared"], x["seed_stage"], x["seed_donor"], x["two_seed_mean"]["pb_rmse"]))
    write(out, "decomp.json", res)
    return res


def write(out, name, obj):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / name).write_text(json.dumps(PH._clean(obj), indent=1))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("what", choices=["anchor", "cov", "decomp", "tau2", "all"])
    p.add_argument("--out", default=str(DEFAULT_OUT))
    a = p.parse_args(argv)
    out = Path(a.out)
    if RUNS in out.resolve().parents and not str(out.resolve()).startswith(str(RUNS / "smoke_r3")) and not str(out.resolve()).startswith(str(RUNS / "results_r3")):
        raise ValueError("--out must be under smoke_r3/ or results_r3/")
    todo = ["tau2", "anchor", "cov", "decomp"] if a.what == "all" else [a.what]
    for w in todo:
        {"tau2": run_tau2, "anchor": run_anchor, "cov": run_cov, "decomp": run_decomp}[w](out)


if __name__ == "__main__":
    main()
