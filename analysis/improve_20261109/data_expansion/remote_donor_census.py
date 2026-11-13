"""Per-donor census of a remote CELLxGENE h5ad from obs metadata only (HTTPS range reads,
byte budget enforced by peek_h5ad_obs.RangeFile). Counts nuclei per donor, optionally restricted
to a set of Supertype/subclass values, and reports donor-level metadata columns.

Usage: python3 -I remote_donor_census.py URL donor_col filter_col filter_vals(comma|'') meta_cols(comma) [budget_MB]
Prints JSON {donor: {"n": cells, meta...}} plus totals. Writes nothing to disk.
"""
import io
import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from peek_h5ad_obs import RangeFile, col  # noqa: E402

url, dcol, fcol, fvals, mcols = sys.argv[1:6]
budget = int(float(sys.argv[6]) * 1e6) if len(sys.argv) > 6 else 40_000_000
fvals = set(fvals.split(",")) if fvals else None
mcols = [c for c in mcols.split(",") if c]
f = RangeFile(url, budget)
with h5py.File(io.BufferedReader(f, buffer_size=1 << 16), "r") as h:
    obs = h["obs"]
    donors = col(obs, dcol).astype(str)
    keep = np.ones(len(donors), bool)
    if fvals is not None:
        fv = col(obs, fcol).astype(str)
        keep = np.isin(fv, list(fvals))
        filt_counts = dict(zip(*[x.tolist() for x in np.unique(fv, return_counts=True)]))
    else:
        filt_counts = None
    meta = {c: col(obs, c).astype(str) for c in mcols}
out = {}
for i in np.flatnonzero(keep):
    d = donors[i]
    if d not in out:
        out[d] = {"n": 0, **{c: meta[c][i] for c in mcols}}
    out[d]["n"] += 1
print(json.dumps({"url": url, "n_cells_total": int(len(donors)), "n_cells_kept": int(keep.sum()),
                  "filter_counts": filt_counts, "n_donors_kept": len(out), "bytes_read": f.used,
                  "donors": out}, indent=1))
