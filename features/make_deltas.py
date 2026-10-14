# CPU only: pseudobulk deltas (log1p normalize_total 1e4, ALL cells) for the 150 knockdowns vs non-targeting.
import json, numpy as np, anndata as ad
from alz_perturb_gan.real_data import load_real_data
data, _ = load_real_data("/ml2/altgan/data/adata_Training.h5ad", split_seed=42, cell_seed=43, train_cells=1, eval_cells=1, input_scale="counts")
names = [p.name for p in data.perturbations]
D = np.stack([p.profile.numpy() for p in data.perturbations]) - data.control_mean.numpy()[None]
a = ad.read_h5ad("/ml2/altgan/data/adata_Training.h5ad", backed="r")
np.save("/ml2/altgan/features/deltas/deltas.npy", D.astype(np.float32))
json.dump({"names": names, "ensembl": [str(a.var.gene_id[n]) for n in names], "genes": list(map(str, data.gene_names)), "eval_names": data.metadata["eval_names"]}, open("/ml2/altgan/features/deltas/deltas_meta.json", "w"))
print("done", D.shape)
