#!/usr/bin/env python3
"""Planted-truth tests for r4/confirm_analysis.py (PROTOCOL_CONFIRM primary analysis) and the confirmation
extension of r4/evaluate_ext.py that it reads (collect of fits, references, ensembles, baselines).

    cd /tmp && python3 -B /ml2/alz/analysis/improve_20261109/r4/test_confirm.py

CPU only, seconds.  Scratch: runs/improve_20261109/smoke_r4/eval/test_confirm_tmp (deleted at the end).
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import confirm_analysis as C  # noqa: E402

TMP = Path("/ml2/alz/runs/improve_20261109/smoke_r4/eval/test_confirm_tmp")
LAB = C.cell_labels("r3")
LEVELS = {"A": (0, 0), "B": (1, 0), "C": (0, 1), "D": (1, 1)}   # (dispersion, external)


def planted_table(rng, disp, ext, inter, sigma, folds=range(9), seeds=C.SEEDS, modes=("snapens", "stagehalf", "biascal@stagehalf"),
                  ens=True, drop=(), base_sd=0.03, disp_by_fold=None):
    """pb_rmse = fold base + disp*d + ext*e + inter*d*e + N(0, sigma) per fit; ensembles without seed noise."""
    t = {}
    base = {f: 0.5 + rng.normal(0, base_sd) for f in folds}
    for mode in modes:
        for c, (d, e) in LEVELS.items():
            for f in folds:
                dd = disp if disp_by_fold is None else disp_by_fold[f]
                mu = base[f] + dd * d + ext * e + inter * d * e
                for s in seeds:
                    if (c, mode, f, s) in drop:
                        continue
                    t[(C.mode_label(LAB[c], mode), f, s)] = {"pb_rmse": mu + rng.normal(0, sigma),
                                                              "pc_coverage80_rep": 0.75, "pc_energy_rep": 1.9 + 0.1 * d}
                if ens:
                    t[(C.mode_label(LAB[c], mode) + C.ENS_SUFFIX, f, None)] = {"pb_rmse": mu - 0.02, "pc_coverage80_rep": 0.78,
                                                                              "pc_energy_rep": 1.8}
    for f in folds:
        t[("baseline:resample", f, None)] = {"pb_rmse": 0.506, "pc_coverage80_rep": 0.70, "pc_energy_rep": 2.08}
        t[("baseline:grand_mean", f, None)] = {"pb_rmse": 0.464}
        t[("baseline:resample+ext", f, None)] = {"pb_rmse": 0.52, "pc_coverage80_rep": 0.72, "pc_energy_rep": 2.0}
    return t


def test_exact_effects():
    """No seed noise: effects recovered exactly, s2 = 0 -> SE 0, interaction exact."""
    rng = np.random.default_rng(0)
    t = planted_table(rng, disp=-0.03, ext=-0.01, inter=0.008, sigma=0.0)
    r = C.analyse(t)
    p = r["primary"]
    assert p["complete"] and not p["missing"]
    assert abs(p["dispersion"]["pooled"]["mean"] - (-0.03 + 0.004)) < 1e-12, p["dispersion"]["pooled"]
    assert abs(p["external"]["pooled"]["mean"] - (-0.01 + 0.004)) < 1e-12
    assert abs(p["interaction"]["pooled"]["mean"] - 0.008) < 1e-12
    for rep in (1, 2):
        assert abs(p["dispersion"]["per_repeat"][rep]["mean"] - (-0.026)) < 1e-12
        assert p["dispersion"]["per_repeat"][rep]["n_blocks"] == 6
    assert p["dispersion"]["s2"] == 0.0 and p["dispersion"]["decision"]["verdict"].startswith("NOT CONFIRMED") is False
    # ensembles: same planted structure (offset cancels)
    e = r["secondary_ensembles"]["snapens"]
    assert abs(e["dispersion"]["pooled"]["mean"] - (-0.026)) < 1e-12 and e["dispersion"]["pooled"]["n_folds_favourable"] == 6
    # exploratory folds 0-2 computed separately
    ex = r["exploratory_original_folds"]["primary_mode"]
    assert abs(ex["dispersion"]["pooled"]["mean"] + 0.026) < 1e-12 and list(ex["dispersion"]["per_repeat"]) == [0]
    # distribution: |0.75-0.8| - |0.70-0.8| = -0.05 for every fit cell
    d = r["distribution_single"]
    key = [k for k in d if k.endswith(LAB["A"] + "#snapens")][0]
    cov = d[key]["pc_coverage80_rep"]["minus_resample"]
    assert all(abs(v + 0.05) < 1e-12 for v in cov["per_repeat"].values()) and cov["n_folds_better"] == 6
    keyD = [k for k in d if k.endswith(LAB["D"] + "#snapens")][0]
    assert "minus_resample+ext" in d[keyD]["pc_energy_rep"] and abs(d[keyD]["pc_energy_rep"]["minus_resample+ext"]["per_repeat"][1] - 0.0) < 1e-12
    assert abs(r["references"]["baseline:grand_mean"]["pooled"] - 0.464) < 1e-12
    print("PASS test_exact_effects")


def test_se_calibration_and_error_rates():
    """Seed noise sigma: the seed-noise SE matches the Monte-Carlo SD of the pooled effect; under the null the
    rule confirms rarely; a 3-SE effect is confirmed most of the time."""
    rng = np.random.default_rng(1)
    sigma = 0.0156
    est, ses, conf_null = [], [], []
    for _ in range(400):
        t = planted_table(rng, disp=0.0, ext=0.0, inter=0.0, sigma=sigma, modes=("snapens",), ens=False)
        p = C.factor_analysis(t, LAB, "snapens", "pb_rmse", C.CONFIRM_REPEATS)
        est.append(p["dispersion"]["pooled"]["mean"]); ses.append(p["dispersion"]["pooled"]["seed_noise_se"])
        conf_null.append(bool(p["dispersion"]["decision"]["confirmed"]))
    true_se = sigma / np.sqrt(12)               # 12 blocks, each term var sigma^2
    sd = float(np.std(est))
    assert abs(sd / true_se - 1) < 0.12, (sd, true_se)
    assert abs(float(np.mean(ses)) / true_se - 1) < 0.12, (np.mean(ses), true_se)
    fp = float(np.mean(conf_null))
    # one-sided 2-SE rule with s2 estimated from 12 fold-cells (12 df): nominal P(t_12 < -2) = 0.034
    assert fp < 0.065, fp
    power = []
    for _ in range(200):
        t = planted_table(rng, disp=-3 * true_se, ext=0.0, inter=0.0, sigma=sigma, modes=("snapens",), ens=False)
        p = C.factor_analysis(t, LAB, "snapens", "pb_rmse", C.CONFIRM_REPEATS)
        power.append(bool(p["dispersion"]["decision"]["confirmed"]))
        assert p["external"]["pooled"]["mean"] is not None
    pw = float(np.mean(power))
    assert pw > 0.6, pw
    # interaction SE is twice the main-effect SE (variance 4 sigma^2 vs sigma^2 per block)
    p = C.factor_analysis(planted_table(rng, 0, 0, 0, sigma, modes=("snapens",), ens=False), LAB, "snapens", "pb_rmse",
                          C.CONFIRM_REPEATS)
    s_int = p["interaction"]["pooled"]["seed_noise_se"]; s_all = np.sqrt(p["interaction"]["s2"])
    assert abs(s_int - 2 * s_all / np.sqrt(12)) < 1e-12
    print(f"PASS test_se_calibration_and_error_rates (MC SD {sd:.5f}, mean SE {np.mean(ses):.5f}, true {true_se:.5f}; "
          f"false confirmations {fp:.3f}; power at 3 SE {pw:.2f})")


def test_rule_needs_both_repeats():
    """Large pooled gain carried by one repeat only is NOT confirmed."""
    rng = np.random.default_rng(2)
    by = {f: (-0.06 if f in (3, 4, 5) else 0.004) for f in range(9)}
    t = planted_table(rng, disp=0, ext=0, inter=0, sigma=0.002, modes=("snapens",), ens=False, disp_by_fold=by)
    p = C.factor_analysis(t, LAB, "snapens", "pb_rmse", C.CONFIRM_REPEATS)
    dz = p["dispersion"]
    assert dz["pooled"]["mean"] < -2 * dz["pooled"]["seed_noise_se"]
    assert dz["per_repeat"][1]["mean"] < 0 < dz["per_repeat"][2]["mean"]
    assert dz["decision"]["confirmed"] is False and dz["decision"]["verdict"] == "NOT CONFIRMED"
    assert dz["decision"]["beyond_2se"] and not dz["decision"]["favourable_in_both_repeats"]
    print("PASS test_rule_needs_both_repeats")


def test_missing_and_provisional():
    rng = np.random.default_rng(3)
    drop = {("D", "snapens", 7, 1), ("B", "snapens", 3, 0)}
    t = planted_table(rng, disp=-0.05, ext=0, inter=0, sigma=0.002, modes=("snapens",), ens=False, drop=drop)
    p = C.factor_analysis(t, LAB, "snapens", "pb_rmse", C.CONFIRM_REPEATS)
    miss = {(m["cell"], m["fold"], m["seed"]) for m in p["missing"]}
    assert miss == {("D", 7, 1), ("B", 3, 0)}, miss
    assert not p["complete"] and p["dispersion"]["n_complete_blocks"] == 10
    assert p["dispersion"]["decision"]["verdict"].startswith("PROVISIONAL CONFIRMED")
    assert p["dispersion"]["decision"]["confirmed"] is None
    # fold 3 / 7 contribute one seed only -> SE uses k_f = 1 there
    s2 = p["dispersion"]["s2"]
    want = np.sqrt(np.sqrt(s2 / 1 + s2 / 2 + s2 / 2) ** 2 / 9 + np.sqrt(s2 / 2 + s2 / 1 + s2 / 2) ** 2 / 9) / 2
    assert abs(p["dispersion"]["pooled"]["seed_noise_se"] - want) < 1e-12
    # nothing at all for repeat 2 -> not evaluable
    t2 = {k: v for k, v in t.items() if not (k[1] in (6, 7, 8))}
    p2 = C.factor_analysis(t2, LAB, "snapens", "pb_rmse", C.CONFIRM_REPEATS)
    assert p2["dispersion"]["decision"]["verdict"].startswith("NOT EVALUABLE")
    r = C.analyse(t)                     # the full analysis runs on a partial table and lists what is missing
    assert len(r["missing_summary"]["primary_fits"]) == 2 and r["missing_summary"]["ensembles"]["snapens"]
    txt = C.report(r)
    assert "PROVISIONAL" in txt and "MISSING" in txt
    print("PASS test_missing_and_provisional")


def _write(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj))


def _res(pb, cov=0.75, pce=1.9, **kw):
    return {"subsets": {"all": {"heldout": {"pseudobulk": {"rmse": pb}, "mean_log": {"rmse_gene": 0.09},
                                            "nonzero": {"rmse_gene": 0.02}, "second_moment": {"rmse_gene": 0.2}}}},
            "pc_space_rep": {"heldout": {"energy_score": pce, "cell_energy_distance": 0.8, "coverage80": cov}}, **kw}


def test_loader_end_to_end():
    """Synthetic evaluate_ext output tree -> evaluate_ext.collect -> confirm_analysis: same numbers as the table."""
    import evaluate_ext as X
    rng = np.random.default_rng(4)
    t = planted_table(rng, disp=-0.02, ext=-0.01, inter=0.0, sigma=0.004, folds=(3, 4, 5, 6, 7, 8), modes=("snapens",))
    shutil.rmtree(TMP, ignore_errors=True)
    for (lab, f, s), flat in t.items():
        if lab.startswith("baseline:"):
            name = lab.removeprefix("baseline:")
            _write(TMP / "baselines" / "hvg2000" / f"fold{f}" / f"{name}.json",
                   _res(flat["pb_rmse"], flat.get("pc_coverage80_rep", 0.7), flat.get("pc_energy_rep", 2.0), kind="baseline",
                        unit=lab, fold=f, meaningful_metrics=X.BL.MEANINGFUL.get(name.split("+")[0])))
        elif C.ENS_SUFFIX in lab:
            unit = lab.partition(" [")[0]
            sub = "ensembles" if "~x" in unit else "ref/ensembles"
            _write(TMP / sub / unit / "hvg2000" / f"fold{f}_n2.json",
                   _res(flat["pb_rmse"], flat["pc_coverage80_rep"], flat["pc_energy_rep"], kind="seed_ensemble", unit=unit,
                        label=lab, fold=f))
        else:
            sub = "fits" if "~x" in lab else "ref/fits"
            _write(TMP / sub / lab / "hvg2000" / f"fold{f}" / f"seed{s}.json",
                   _res(flat["pb_rmse"], flat["pc_coverage80_rep"], flat["pc_energy_rep"], kind="fit", unit=lab, fold=f, seed=s))
    tab = C.load_table(TMP)
    a = C.factor_analysis(t, LAB, "snapens", "pb_rmse", C.CONFIRM_REPEATS)
    b = C.factor_analysis(tab, LAB, "snapens", "pb_rmse", C.CONFIRM_REPEATS)
    for name in C.FACTORS:
        assert abs(a[name]["pooled"]["mean"] - b[name]["pooled"]["mean"]) < 1e-12
        assert abs(a[name]["pooled"]["seed_noise_se"] - b[name]["pooled"]["seed_noise_se"]) < 1e-12
    ea = C.ensemble_analysis(tab, LAB, "snapens", "pb_rmse", C.CONFIRM_REPEATS)
    assert abs(ea["dispersion"]["pooled"]["mean"] + 0.02) < 1e-12 and not ea["missing"]
    res = C.main(["--results", str(TMP), "--json", str(TMP / "a.json"), "--txt", str(TMP / "a.txt")])
    assert res["primary"]["complete"] and json.loads((TMP / "a.json").read_text())["primary"]["complete"]
    # resample masks: baseline distribution metrics stay; grand_mean's are masked by MEANINGFUL
    assert tab[("baseline:grand_mean", 3, None)].get("pc_energy_rep") is None
    assert tab[("baseline:resample", 3, None)].get("pc_energy_rep") is not None
    print(f"PASS test_loader_end_to_end ({len(tab)} records; dispersion {b['dispersion']['pooled']['mean']:+.4f} "
          f"+- {b['dispersion']['pooled']['seed_noise_se']:.4f}: {b['dispersion']['decision']['verdict']})")


def test_evaluate_ext_ensemble_labels():
    """Ensemble label parsing / member search and the confirm preset."""
    import evaluate_ext as X
    assert X.parse_unit_label("g.dispz,pbulk~x#snapens") == ("g.dispz,pbulk", "x", "snapens")
    assert X.parse_unit_label("pbulk,g.dispz~r3") == ("g.dispz,pbulk", "r3", None)
    for bad in ("pbulk", "~x#snapens"):
        try:
            X.parse_unit_label(bad)
        except ValueError:
            continue
        raise AssertionError(bad)
    assert len(X.ENSEMBLE_PRESETS["confirm"]) == 12 and len(X.ENSEMBLE_PRESETS["x_lead"]) == 10
    fits = TMP / "fits_x"
    for s in (0, 1):
        d = fits / "pbulk" / "hvg2000x" / "fold3" / f"seed{s}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "generated_snapens.npz").write_bytes(b"")
        if s == 0:
            (d / "generated_snapens.receipt.json").write_text("{}")
    recs, miss = X.ensemble_members("pbulk~x#snapens", 3, [f"r4={fits}"], [], [0, 1])
    assert recs is None and miss == [1]
    (fits / "pbulk" / "hvg2000x" / "fold3" / "seed1" / "generated_snapens.receipt.json").write_text("{}")
    recs, miss = X.ensemble_members("pbulk~x#snapens", 3, [f"r4={fits}"], [], [0, 1])
    assert [r["seed"] for r in recs] == [0, 1] and recs[0]["unit"] == "pbulk~x#snapens" and recs[0]["panel"] == "hvg2000x"
    try:
        X.ensemble_members("pbulk~r3#snapens", 3, [], ["r2=/nonexistent"], [0, 1])
        raise AssertionError("missing ref tag accepted")
    except ValueError:
        pass
    print("PASS test_evaluate_ext_ensemble_labels")


def main():
    TMP.mkdir(parents=True, exist_ok=True)
    try:
        test_exact_effects()
        test_se_calibration_and_error_rates()
        test_rule_needs_both_repeats()
        test_missing_and_provisional()
        test_loader_end_to_end()
        test_evaluate_ext_ensemble_labels()
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    print("ALL PASS")


if __name__ == "__main__":
    main()
