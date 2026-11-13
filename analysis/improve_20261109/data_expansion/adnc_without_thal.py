"""How well can SEA-AD ADNC be recovered WITHOUT Thal phase (as for ROSMAP)?
Metadata only: cross-tabulates ADNC against (Braak band B0-B3, CERAD C0-C3) over the 84 used aged
SEA-AD MTG donors in the official Micro-PVM candidate metadata (82 cohort donors + 2 below the 100-nuclei rule)."""
import json, collections
from pathlib import Path
B = Path("/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark/official_metadata/micro_pvm_candidates.json")
m = json.loads(B.read_text()); c = m["cols"]
ix = {k: c.index(k) for k in ["Donor ID", "Overall AD neuropathological Change", "Thal", "Braak", "CERAD score"]}
don = {}
for r in m["rows"].values():
    don[r[ix["Donor ID"]]] = (r[ix["Overall AD neuropathological Change"]], r[ix["Thal"]], r[ix["Braak"]], r[ix["CERAD score"]])
band = {"Braak 0": "B0", "Braak I": "B1", "Braak II": "B1", "Braak III": "B2", "Braak IV": "B2", "Braak V": "B3", "Braak VI": "B3"}
cer = {"Absent": "C0", "Sparse": "C1", "Moderate": "C2", "Frequent": "C3"}
tab = collections.defaultdict(collections.Counter)
for d, (a, t, b, ce) in don.items():
    tab[(band[b], cer[ce])][a] += 1
amb = 0
print(len(don), "donors")
for k in sorted(tab):
    print(k, dict(tab[k]))
    if len(tab[k]) > 1:
        amb += sum(tab[k].values())
print("donors in (Braak band, CERAD) cells with >1 ADNC level:", amb)
th = collections.defaultdict(collections.Counter)
for d, (a, t, b, ce) in don.items():
    th[a][t] += 1
print({k: dict(v) for k, v in th.items()})
