"""Recover the completed r4 confirmation's profile-error analysis, on CPU.

Reads only existing generated SEA-AD summaries and the 37-donor development
caches. No external or sealed donor counts, training, calibration or selection.
The float32 log-profile intermediate matches evaluate.score_generated exactly;
the pre-existing r4 confirmation functions supply its fixed factor analysis.
This is the profile component only: distributional adoption checks remain
separate. Outputs live in results_r5/confirmation_profiles, never frozen trees.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import evaluate as EV

spec = importlib.util.spec_from_file_location("confirmation_analysis", HERE.parent / "r4/confirm_analysis.py")
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)
RUNS = EV.RUNS
MODES = ("snapens", "stagehalf", "biascal@stagehalf", "stageblind")
ARMS = ("pbulk", "g.dispz,pbulk")
OUT = RUNS / "results_r5/confirmation_profiles"


def load_profiles(path, real):
    with np.load(path, allow_pickle=False) as f:
        targets = f["targets"].astype(str).tolist()
        EV.check_development_donors(targets, path)
        assert sorted(targets) == sorted(real.donors.tolist()), path
        assert np.array_equal(f["genes"].astype(str), real.genes), path
        order = [targets.index(d) for d in real.donors]
        assert np.array_equal(f["is_heldout"][order], real.is_heldout), path
        assert np.array_equal(f["stage"][order], real.stage), path
        pb = f["pb_counts"][order].astype(np.float64)
    assert pb.shape[1] == 32, path
    assert np.isfinite(pb).all() and (pb >= 0).all(), path
    gl = np.empty(pb.shape, dtype=np.float32)
    for t in range(len(pb)):
        gl[t] = EV.MX.log_cp10k(pb[t])
    return gl


def score(gl, real):
    pred = gl.astype(np.float64).mean(1)
    obs = EV.MX.log_cp10k(real.pb)
    err = pred[real.is_heldout] - obs[real.is_heldout]
    return float(np.sqrt(np.mean(err ** 2, axis=1)).mean())


def secondary_analysis(table, labels, mode):
    block = C.factor_analysis(table, labels, mode, "pb_rmse", C.CONFIRM_REPEATS)
    for factor in C.FACTORS:
        block[factor].pop("decision", None)
    block["status"] = "Secondary or reference read-out; no confirmation decision rule."
    return block


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    table, sources, reproduced = {}, [], []
    for k in range(9):
        real = EV.load_real(RUNS / f"caches/hvg2000/fold{k}")
        for root, panel, tag, oldtree in (("fits_r3", "hvg2000", "r3", "results_r3"),
                                         ("fits_r4", "hvg2000x", "x", "results_r4")):
            for arm in ARMS:
                for mode in MODES:
                    pairs = []
                    for seed in (0, 1):
                        path = RUNS / root / arm / panel / f"fold{k}/seed{seed}" / f"generated_{mode}.npz"
                        if not path.exists():
                            continue
                        gl = load_profiles(path, real)
                        value = score(gl, real)
                        label = f"{arm}~{tag}#{mode}"
                        table[(label, k, seed)] = {"pb_rmse": value}
                        pairs.append(gl)
                        sources.append({"path": str(path), "sha256": EV.sha256(path), "pb_rmse": value,
                                        "fold": k, "seed": seed, "unit": label})
                        oldpath = RUNS / oldtree / "fits" / label / "hvg2000" / f"fold{k}/seed{seed}.json"
                        if oldpath.exists():
                            oldvalue = EV.headline(json.loads(oldpath.read_text()))["pb_rmse"]
                            diff = abs(value - oldvalue)
                            assert diff < 1e-12, (path, value, oldvalue)
                            reproduced.append({"reference": str(oldpath), "absolute_difference": diff})
                    if len(pairs) == 2:
                        pooled = np.concatenate(pairs, axis=1)
                        table[(label + C.ENS_SUFFIX, k, None)] = {"pb_rmse": score(pooled, real)}
                        matched = np.stack([pairs[m % 2][:, m] for m in range(32)], axis=1)
                        table[(label + " [seed-ensemble n=2 M=32]", k, None)] = {"pb_rmse": score(matched, real)}
        for name in C.REF_BASELINES:
            path = RUNS / "results_r3/baselines/hvg2000" / f"fold{k}/{name}.json"
            if path.exists():
                table[(f"baseline:{name}", k, None)] = {"pb_rmse": EV.headline(json.loads(path.read_text()))["pb_rmse"]}
        print(f"fold {k} scored", flush=True)
    assert reproduced, "Require reproduction against pre-existing evaluator scores"
    labels = C.cell_labels()
    res = {
        "protocol": "r4/PROTOCOL_CONFIRM.md",
        "scope": "Profile errors only; generated SEA-AD summaries only. Distribution checks remain unscored here.",
        "primary": C.factor_analysis(table, labels, "snapens", "pb_rmse", C.CONFIRM_REPEATS),
        "secondary": {m: secondary_analysis(table, labels, m) for m in MODES[1:]},
        "ensemble_M64_secondary": {m: C.ensemble_analysis(table, labels, m, "pb_rmse", C.CONFIRM_REPEATS) for m in MODES},
        "references": C.baseline_rows(table, C.CONFIRM_REPEATS, C.REF_BASELINES),
        "reproduction": reproduced,
        "sources": sources,
        "table": [{"unit": u, "fold": f, "seed": s, **v} for (u, f, s), v in table.items()],
    }
    assert res["primary"]["complete"], res["primary"]["missing"]
    EV.write_json(OUT / "analysis.json", C._jsonable(res))
    lines = [res["scope"], f"Reproduced {len(reproduced)} existing scores to 1e-12."]
    for mode, block in [("snapens (primary)", res["primary"]), *res["secondary"].items()]:
        lines.append(mode)
        for factor in C.FACTORS:
            lines.append(C._effect_lines(factor, block[factor], C.CONFIRM_REPEATS))
    txt = "\n".join(lines) + "\n"
    (OUT / "analysis.txt").write_text(txt)
    print(txt)


if __name__ == "__main__":
    EV.set_torch_threads()
    main()
