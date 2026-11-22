"""Iteration 2: control-condition decomposition. Fit the harness
PGAN on the CONTROL (INT) train cells only, sample control, ED vs
held-out control — splits marginal-fit failure from conditioning
failure, using the driver's cached preps."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
from alztest.harness import ConditionalPGAN, energy_distance, pca_fit_transform  # noqa: E402

out = []
for seed in (0, 1, 2):
    z = np.load(f"data/neuroko_prep_seed{seed}.npz", allow_pickle=True)
    Xtr, ytr, Xte, yte = z["Xtr"], z["ytr"], z["Xte"], z["yte"]
    Ptr, (mu, comps) = pca_fit_transform(Xtr, n_comp=30)
    Pte = (Xte - mu) @ comps.T
    sd = Ptr.std(0) + 1e-8
    Ptr, Pte = Ptr / sd, Pte / sd
    ctrl_tr, ctrl_te = Ptr[ytr == 0], Pte[yte == 0]
    m = ConditionalPGAN(Ptr.shape[1], 1, seed=seed)
    m.fit(ctrl_tr, np.zeros(len(ctrl_tr), dtype=int), steps=300)
    gen = m.sample(0, 512, seed=seed)
    ed = energy_distance(gen, ctrl_te, seed=seed)
    out.append({"seed": seed, "ed_control_fit_vs_heldout_control": float(ed),
                "n_ctrl_train": int(len(ctrl_tr)), "n_ctrl_test": int(len(ctrl_te))})
    print(json.dumps(out[-1]), flush=True)
Path("data/neuroko_decomp_results.json").write_text(json.dumps(out, indent=1))
print("DECOMP COMPLETE")
