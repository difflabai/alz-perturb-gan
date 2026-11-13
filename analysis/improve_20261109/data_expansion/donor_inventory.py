"""Data-expansion inventory: donor-level METADATA tables for the local SEA-AD MTG cohort.

Reads only donor-level metadata (no count/expression arrays):
  - benchmark/rebuilt/mg_meta.json        (official SEA-AD MTG cell metadata, Micro-PVM rows;
                                           only donor-level columns are used)
  - benchmark/rebuilt/candidate_cohort.json (selected nuclei per donor)
  - benchmark/frozen_splits.json           (S0f0: 57 fit / 7 validation / 18 outer-test)
  - alz-audit/matched-benchmark-20261103/folds.json (3 folds over the 57)
  - improve_20261109/folds.json            (37 development donors, 3 inner folds)
Writes donor_inventory.json next to this script. CPU only, a few MB of RAM.
"""
import json
from collections import Counter, OrderedDict
from pathlib import Path

B = Path("/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark")
A = Path("/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/matched-benchmark-20261103")
S = Path("/ml2/alz/analysis/improve_20261109")
STAGES = ["Not AD", "Low", "Intermediate", "High"]
DONOR_COLS = ["Donor ID", "Neurotypical reference", "Sex", "Age at Death", "APOE Genotype", "PMI",
              "Overall AD neuropathological Change", "Thal", "Braak", "CERAD score", "LATE",
              "Cognitive Status", "Severely Affected Donor"]

meta = json.loads((B / "rebuilt/mg_meta.json").read_text())
cols = meta["cols"]
idx = {c: cols.index(c) for c in DONOR_COLS}
donor = {}
for row in meta["rows"].values():
    rec = {c: row[idx[c]] for c in DONOR_COLS}
    d = rec["Donor ID"]
    if d in donor and donor[d] != rec:
        raise ValueError(f"inconsistent donor-level metadata for {d}")
    donor[d] = rec
cells = json.loads((B / "rebuilt/candidate_cohort.json").read_text())["donors"]
assert set(cells) == set(donor) and len(donor) == 82

split = json.loads((B / "frozen_splits.json").read_text())["splits"]["S0f0"]
fit57, val7, test18 = split["fit_donors"], split["validation_donors"], split["test_donors"]
folds = json.loads((A / "folds.json").read_text())["folds"]
study = json.loads((S / "folds.json").read_text())
dev37 = study["development_donors"]
assert len(fit57) == 57 and len(val7) == 7 and len(test18) == 18
assert set(fit57) | set(val7) | set(test18) == set(donor)


def stage(d):
    return STAGES.index(donor[d]["Overall AD neuropathological Change"])


def apoe4(d):
    return donor[d]["APOE Genotype"].split("/").count("4")


def table(ds):
    st = Counter(stage(d) for d in ds)
    out = OrderedDict(n=len(ds), cells=sum(cells[d] for d in ds))
    for i, s in enumerate(STAGES):
        out[s] = st.get(i, 0)
    out["cells_by_stage"] = {s: sum(cells[d] for d in ds if stage(d) == i) for i, s in enumerate(STAGES)}
    out["Female"] = sum(donor[d]["Sex"] == "Female" for d in ds)
    out["Male"] = sum(donor[d]["Sex"] == "Male" for d in ds)
    out["APOE4_0/1/2"] = [sum(apoe4(d) == k for d in ds) for k in range(3)]
    out["neurotypical_reference"] = sum(donor[d]["Neurotypical reference"] == "True" for d in ds)
    out["severely_affected"] = sum(donor[d]["Severely Affected Donor"] == "Y" for d in ds)
    out["cognitive"] = dict(Counter(donor[d]["Cognitive Status"] for d in ds))
    return out


groups = OrderedDict()
groups["all82"] = sorted(donor)
groups["fit57"] = fit57
groups["other25"] = sorted(val7 + test18)
groups["validation7_dev"] = val7
groups["outer_test18_exposed_audit"] = test18
for f in folds:
    groups[f"matched_fold{f['fold']}_optimization"] = f["optimization_donors"]
    groups[f"matched_fold{f['fold']}_calibration"] = f["internal_calibration_donors"]
    groups[f"matched_fold{f['fold']}_evaluation"] = f["evaluation_donors"]
groups["study_dev37"] = dev37
for f in study["inner_folds"]:
    groups[f"study_inner{f['inner_fold']}_train"] = f["train_donors"]
    groups[f"study_inner{f['inner_fold']}_heldout"] = f["heldout_donors"]
groups["fit57_plus_val7"] = sorted(fit57 + val7)

tables = OrderedDict((k, table(v)) for k, v in groups.items())
role = {}
for d in donor:
    role[d] = "fit57" if d in fit57 else ("validation7" if d in val7 else "outer_test18")
per_donor = [dict(donor=d, role=role[d], dev37=d in dev37, n_cells=cells[d],
                  stage=donor[d]["Overall AD neuropathological Change"], sex=donor[d]["Sex"],
                  age=donor[d]["Age at Death"], apoe=donor[d]["APOE Genotype"], braak=donor[d]["Braak"],
                  thal=donor[d]["Thal"], cerad=donor[d]["CERAD score"], late=donor[d]["LATE"],
                  cognitive=donor[d]["Cognitive Status"],
                  neurotypical_reference=donor[d]["Neurotypical reference"],
                  severely_affected=donor[d]["Severely Affected Donor"])
             for d in sorted(donor)]
out = dict(sources=dict(meta=str(B / "rebuilt/mg_meta.json"), splits=str(B / "frozen_splits.json"),
                        matched_folds=str(A / "folds.json"), study_folds=str(S / "folds.json")),
           tables=tables, other25=[r for r in per_donor if r["role"] != "fit57"], all_donors=per_donor)
(Path(__file__).parent / "donor_inventory.json").write_text(json.dumps(out, indent=1))
hdr = ["n", "cells"] + STAGES + ["Female", "Male", "APOE4_0/1/2", "neurotypical_reference", "severely_affected"]
print("group".ljust(34), *[h[:12].rjust(12) for h in hdr])
for k, t in tables.items():
    print(k.ljust(34), *[str(t[h]).rjust(12) for h in hdr])
print()
for r in out["other25"]:
    print(r["role"].ljust(13), r["donor"], r["n_cells"], r["stage"].ljust(13), r["sex"].ljust(7), r["age"],
          r["apoe"], r["braak"], r["thal"], r["cerad"], r["cognitive"], "NR" if r["neurotypical_reference"] == "True" else "",
          "SAD" if r["severely_affected"] == "Y" else "")
