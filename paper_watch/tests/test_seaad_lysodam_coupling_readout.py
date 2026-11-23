"""Cycle-9 top find — lysosomal-DAM coupling readout on SEA-AD.

Find: "Evolutionarily conserved coupling of lysosomal response
sensing and disease-associated microglial reprogramming scales with
pathological timescales", bioRxiv 2026-11-19 (preprint). Unit-
corrected reanalysis of six datasets (>320,000 human microglial
nuclei): the MiT/TFE lysosomal-biogenesis program temporally
precedes DAM reprogramming and stays coupled to it at baseline;
disease amplifies the coupling (Delta) with pathological timescale
— 5xFAD Delta +0.252, human SEA-AD Delta +0.072 (baseline
maintenance across decades-long asynchronous progression). AD GWAS
risk anchors downstream at APOE, not on the lysosomal program.
Convergent same-week evidence: GPNMB lysosomal program on the DAM
continuum (bioRxiv 2026-11-20) and MITF/TFE driving the DAM
signature under lysosomal dysfunction (Immunity 2026).

Route: NEW TARGET + READOUT. The per-donor coupling amplification
(donor lyso-DAM coupling minus the control mean coupling) is a
named, quantified, donor-level quantity computed on SEA-AD — data
already local. Question for a ParticleGAN fit on CONTROL microglia
only: does a donor's distance-from-the-control-model track that
donor's coupling amplification? If the model has learned the
healthy joint program structure, donors whose microglia have
re-coupled lyso->DAM should sit farther from it.

Salient claim under test (toy scale):
    Spearman rho(model deviation, donor coupling amplification)
    > 0.3 across >= 3 seeds, clear of a shuffled-coupling arm, a
    Gaussian sampler arm, and a linear centroid-distance baseline.
Replication check (no fit, reported not asserted at the 0.3 bar):
    mean coupling in AD donors minus CTRL donors should be small
    and positive (paper: SEA-AD Delta +0.072).

Compartment: subclass "Immune" (ABC taxonomy home of the
microglia/PVM supertypes), as in the LL readout (cycle 2).
Programs (paper-named constituents; asserted present in the data):
  lysosomal biogenesis (MiT/TFE CLEAR core):
    LAMP1 LAMP2 CTSD CTSB CTSS NPC1 NPC2 LIPA GBA PSAP ATP6V1H
  DAM:
    APOE TREM2 TYROBP LPL CST7 SPP1 ITGAX CLEC7A AXL CD9 GPNMB

Iteration log: notes/seaad_lysodam_coupling_readout.md
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import spearmanr

from alztest.data import DataUnavailable, load_seaad_obs, seaad_annotations
from alztest.harness import energy_distance, fit_unconditional, pca_fit_transform, standardize

REGIONS = ("AnG", "LEC")
LYSO = ["LAMP1", "LAMP2", "CTSD", "CTSB", "CTSS", "NPC1", "NPC2",
        "LIPA", "GBA", "PSAP", "ATP6V1H"]
DAM = ["APOE", "TREM2", "TYROBP", "LPL", "CST7", "SPP1", "ITGAX",
       "CLEC7A", "AXL", "CD9", "GPNMB"]
MODULE = LYSO + DAM
N_COMP = 30
MIN_CELLS_PER_DONOR = 25


def _load_microglia(region: str = "AnG"):
    adata = load_seaad_obs(region)
    ann = seaad_annotations(region)
    obs = adata.obs
    key = (obs["cell_barcode"].astype(str) + "-"
           + obs["barcoded_cell_sample_label"].astype(str))
    pos = {label: i for i, label in enumerate(key)}
    common = [label for label in ann.index if label in pos]
    ann = ann.loc[common]
    rows_all = np.array([pos[label] for label in common])
    keep = ann["subclass"].astype(str).to_numpy() == "Immune"
    ann = ann[keep]
    rows = rows_all[keep]
    adnc = ann["adnc"].astype(str)
    group = np.where(adnc.str.contains("High"), "AD",
                     np.where(adnc.str.contains("Not AD|Low"), "CTRL",
                              "OTHER")).astype(object)
    donors = ann["donor_label"].astype(str).to_numpy()
    order = np.argsort(rows)
    sel = adata[rows[order]]
    S = sel.X
    var_names = np.array(sel.var["gene_symbol"].astype(str))
    present = [g for g in MODULE if (var_names == g).any()]
    if len(present) < 16:
        raise DataUnavailable(f"only {len(present)} program genes found")
    mod_idx = [int(np.where(var_names == g)[0][0]) for g in present]
    n = S.shape[0]
    s1 = np.zeros(S.shape[1]); s2 = np.zeros(S.shape[1])
    for lo in range(0, n, 512):
        C = np.asarray(S[lo:lo + 512].todense(), dtype=np.float64)
        s1 += C.sum(0); s2 += (C * C).sum(0)
    var = s2 / n - (s1 / n) ** 2
    top = [i for i in np.argsort(var)[::-1][:1500] if i not in mod_idx]
    cols = np.sort(np.array(top + mod_idx))
    X = np.empty((n, len(cols)), dtype=np.float64)
    for lo in range(0, n, 512):
        X[lo:lo + 512] = np.asarray(S[lo:lo + 512][:, cols].todense(),
                                    dtype=np.float64)
    X = X[np.argsort(order)]
    mod_pos = [int(np.where(cols == i)[0][0]) for i in mod_idx]
    n_lyso = len([g for g in LYSO if g in set(present)])
    return X, group, donors, mod_pos[:n_lyso], mod_pos[n_lyso:]


def _coupling(X, group, donors, lyso_pos, dam_pos):
    """Per-donor lyso-DAM coupling: Pearson r across the donor's
    cells between the two module scores (z-scored vs CTRL)."""
    ctrl = group == "CTRL"
    def zscore(cols):
        mu, sd = X[ctrl][:, cols].mean(0), X[ctrl][:, cols].std(0) + 1e-9
        return ((X[:, cols] - mu) / sd).mean(1)
    lyso, dam = zscore(lyso_pos), zscore(dam_pos)
    out = {}
    for d in np.unique(donors):
        m = donors == d
        if m.sum() >= MIN_CELLS_PER_DONOR:
            out[d] = float(np.corrcoef(lyso[m], dam[m])[0, 1])
    return out


def run_once(seed: int = 0, steps: int = 300) -> dict:
    devs = {"model": [], "gauss": [], "lin": []}
    amps, groups = [], []
    rng = np.random.default_rng(seed)
    for region in REGIONS:
        X, group, donors, lyso_pos, dam_pos = _load_microglia(region)
        coup = _coupling(X, group, donors, lyso_pos, dam_pos)
        ctrl_coup = np.mean([c for d, c in coup.items()
                             if group[donors == d][0] == "CTRL"])
        ctrl_mask = group == "CTRL"
        Xc = X[ctrl_mask]
        if len(Xc) > 6000:
            Xc = Xc[rng.choice(len(Xc), 6000, replace=False)]
        Pc, (mu, comps) = pca_fit_transform(Xc, n_comp=N_COMP, seed=seed)
        Pc = standardize(Pc)
        # Iteration 2 (one change, readout/fit-averaging only):
        # 3-fit ensemble — average each donor's ED over 3 fits.
        # Iteration 1 was single-fit: rho_model 0.255/0.341/0.400
        # vs Gaussian 0.432/0.403/0.400 (model variance-dominated,
        # the SEA-AD loop's iteration-9 lesson).
        gens = [fit_unconditional(Pc, steps=steps, seed=seed + 100 * k
                                  ).sample(0, 1024, seed=seed + 100 * k)
                for k in range(3)]
        gmu, gsd = Pc.mean(0), Pc.std(0) + 1e-9
        for d, c in coup.items():
            m = donors == d
            Pd = standardize((X[m] - mu) @ comps.T)
            if len(Pd) > 512:
                Pd = Pd[rng.choice(len(Pd), 512, replace=False)]
            devs["model"].append(float(np.mean(
                [energy_distance(g, Pd, seed=seed) for g in gens])))
            g = rng.normal(gmu, gsd, size=(1024, Pc.shape[1]))
            devs["gauss"].append(energy_distance(g, Pd, seed=seed))
            devs["lin"].append(float(np.linalg.norm(
                Pd.mean(0) - Pc.mean(0))))
            amps.append(c - ctrl_coup)
            groups.append(str(group[m][0]))
    amps = np.array(amps)
    res = {"seed": seed, "n_units": int(len(amps)),
           "delta_coupling_ad_minus_ctrl": float(
               amps[np.array(groups) == "AD"].mean()
               - amps[np.array(groups) == "CTRL"].mean())}
    for k, v in devs.items():
        res[f"rho_{k}"] = float(spearmanr(v, amps).statistic)
    shuf = rng.permutation(amps)
    res["rho_shuffled"] = float(spearmanr(devs["model"], shuf).statistic)
    return res


def test_seaad_lysodam_coupling_readout():
    try:
        results = [run_once(seed=s) for s in (0, 1, 2)]
    except DataUnavailable as e:
        pytest.skip(str(e))
    for r in results:
        assert r["rho_model"] > 0.3, r
        assert r["rho_model"] > r["rho_gauss"], r
        assert r["rho_model"] > r["rho_lin"], r
        assert r["rho_model"] > r["rho_shuffled"], r


# ------------------------------------------------------------------
# Iteration 3 (2026-11-22, deep-pass correction): the SPLIT readout.
# Iterations 1-2 pooled all donors: rho(deviation, coupling minus
# CTRL mean) mostly re-measures BASELINE coupling, which the source
# paper itself establishes as the large term. The disease-specific
# quantity is the Delta: coupling estimated separately in CTRL and
# AD donors. This readout (a) reports Delta with a donor-level
# permutation interval (paper: SEA-AD +0.072), and (b) asserts on
# the AD arm only: within AD donors, does model deviation track
# within-AD coupling amplification (c_d - mean AD coupling), clear
# of Gaussian / linear / shuffled arms? A GPNMB-single-gene variant
# of the DAM score is reported alongside (the 2026-11-20 GPNMB
# lysosomal-program paper). Same fits/deviations pipeline as
# run_once; only the grouping and estimand change.
# ------------------------------------------------------------------

def run_split(seed: int = 0, steps: int = 300) -> dict:
    per = {"model": [], "gauss": [], "lin": []}
    amps_ad, amps_ctrl = [], []
    coup_ad, coup_ctrl = [], []
    gpnmb_rhos = []
    rng = np.random.default_rng(seed)
    for region in REGIONS:
        X, group, donors, lyso_pos, dam_pos = _load_microglia(region)
        coup = _coupling(X, group, donors, lyso_pos, dam_pos)
        # GPNMB-only DAM proxy coupling (reported arm)
        g_pos = [dam_pos[-1]] if len(dam_pos) else []
        coup_g = (_coupling(X, group, donors, lyso_pos, g_pos)
                  if g_pos else {})
        ctrl_mask = group == "CTRL"
        Xc = X[ctrl_mask]
        if len(Xc) > 6000:
            Xc = Xc[rng.choice(len(Xc), 6000, replace=False)]
        Pc, (mu, comps) = pca_fit_transform(Xc, n_comp=N_COMP, seed=seed)
        Pc = standardize(Pc)
        gens = [fit_unconditional(Pc, steps=steps, seed=seed + 100 * k
                                  ).sample(0, 1024, seed=seed + 100 * k)
                for k in range(3)]
        gmu, gsd = Pc.mean(0), Pc.std(0) + 1e-9
        dev_by_donor = {}
        for d, c in coup.items():
            m = donors == d
            Pd = standardize((X[m] - mu) @ comps.T)
            if len(Pd) > 512:
                Pd = Pd[rng.choice(len(Pd), 512, replace=False)]
            dev_by_donor[d] = (
                float(np.mean([energy_distance(g, Pd, seed=seed)
                               for g in gens])),
                energy_distance(rng.normal(gmu, gsd,
                                           size=(1024, Pc.shape[1])),
                                Pd, seed=seed),
                float(np.linalg.norm(Pd.mean(0) - Pc.mean(0))),
                str(group[m][0]), c,
                coup_g.get(d, np.nan))
        ad = {d: v for d, v in dev_by_donor.items() if v[3] == "AD"}
        ct = {d: v for d, v in dev_by_donor.items() if v[3] == "CTRL"}
        if len(ad) >= 10 and len(ct) >= 10:
            ma = np.mean([v[4] for v in ad.values()])
            mc = np.mean([v[4] for v in ct.values()])
            for d, v in ad.items():
                per["model"].append(v[0]); per["gauss"].append(v[1])
                per["lin"].append(v[2]); amps_ad.append(v[4] - ma)
                coup_ad.append(v[4])
                if not np.isnan(v[5]):
                    gpnmb_rhos.append((v[0], v[5]))
            for d, v in ct.items():
                coup_ctrl.append(v[4])
    amps_ad = np.array(amps_ad)
    res = {"seed": seed, "n_ad_units": int(len(amps_ad)),
           "delta_coupling_ad_minus_ctrl": float(
               np.mean(coup_ad) - np.mean(coup_ctrl)),
           "mean_coupling_ad": float(np.mean(coup_ad)),
           "mean_coupling_ctrl": float(np.mean(coup_ctrl))}
    for k, v in per.items():
        res[f"rho_ad_{k}"] = float(spearmanr(v, amps_ad).statistic)
    res["rho_ad_shuffled"] = float(
        spearmanr(per["model"], rng.permutation(amps_ad)).statistic)
    if gpnmb_rhos:
        res["rho_ad_gpnmb_coupling"] = float(spearmanr(
            [g[0] for g in gpnmb_rhos], [g[1] for g in gpnmb_rhos]
        ).statistic)
    # Donor-level permutation interval for Delta (reported).
    all_c = np.array(coup_ad + coup_ctrl)
    labels = np.array([1] * len(coup_ad) + [0] * len(coup_ctrl))
    deltas = []
    for _ in range(200):
        p = rng.permutation(labels)
        deltas.append(all_c[p == 1].mean() - all_c[p == 0].mean())
    res["delta_perm_p95"] = float(np.percentile(np.abs(deltas), 95))
    return res


def test_seaad_lysodam_split_delta_readout():
    try:
        results = [run_split(seed=s) for s in (0, 1, 2)]
    except DataUnavailable as e:
        pytest.skip(str(e))
    for r in results:
        # The AD-arm readout: model deviation tracks within-AD
        # coupling amplification, clear of every control arm.
        assert r["rho_ad_model"] > 0.3, r
        assert r["rho_ad_model"] > r["rho_ad_gauss"], r
        assert r["rho_ad_model"] > r["rho_ad_lin"], r
        assert r["rho_ad_model"] > r["rho_ad_shuffled"], r
