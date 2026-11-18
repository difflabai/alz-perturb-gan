"""Stream-extract HypoAD compartments from the 51 GB GEO object.

Yu et al., Nature 2026 (doi 10.1038/s41586-026-11305-w); processed
data GEO GSE342635, file GSE342635_HypoAD.h5ad (51,107,823,722 B).
Only obs/var and the microglia CSR row-slices are pulled, via
RangeFile range requests. Output: data/hypoad_microglia.h5ad.
"""
import sys
sys.path.insert(0, "scripts")
import numpy as np, pandas as pd, anndata as ad
from rangefile import RangeFile

URL = ("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE342nnn/GSE342635/"
       "suppl/GSE342635_HypoAD.h5ad")
rf = RangeFile(URL)
print("remote size:", rf.size, flush=True)
adata = ad.read_h5ad(rf, backed="r")
print("shape:", adata.shape, flush=True)
print("obs cols:", list(adata.obs.columns), flush=True)
for c in adata.obs.columns:
    vc = adata.obs[c]
    try:
        nu = vc.nunique()
    except Exception:
        continue
    if nu < 40:
        print("COL", c, dict(vc.astype(str).value_counts().head(40)), flush=True)
adata.obs.to_parquet("data/hypoad_obs.parquet")
print("obs parquet saved", flush=True)
