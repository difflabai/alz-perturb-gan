"""ESM-2 35M (facebook/esm2_t12_35M_UR50D) mean-pooled last-layer embeddings, CPU, eval mode, no grad.
Proteins longer than 1022 residues: non-overlapping 1022-residue windows, residue-count-weighted mean."""
import hashlib, json
from pathlib import Path
import numpy as np, torch
from transformers import AutoTokenizer, AutoModel
COV = Path("/workspace/altgan/cov"); OUT = Path("/workspace/altgan/features/out")
MODEL = "facebook/esm2_t12_35M_UR50D"
torch.manual_seed(0); torch.set_num_threads(8)
tok = AutoTokenizer.from_pretrained(MODEL); model = AutoModel.from_pretrained(MODEL).eval()
rev = getattr(model.config, "_commit_hash", None)
rows = sorted((l.split("\t") for l in (COV / "genes150_ensembl.tsv").read_text().split("\n") if l), key=lambda r: r[1])
acc = json.load(open(COV / "uniprot_map.json"))
seqs = {}; cur = None
for l in open(COV / "uniprot150.fasta"):
    l = l.strip()
    if l.startswith(">"): cur = l.split("|")[1]; seqs[cur] = ""
    else: seqs[cur] += l
X = np.zeros((150, model.config.hidden_size), np.float32)
with torch.no_grad():
    for i, (s, e) in enumerate(rows):
        q = seqs[acc[s]]; parts = [q[j:j + 1022] for j in range(0, len(q), 1022)]
        acc_v = 0; tot = 0
        for p in parts:
            h = model(**tok(p, return_tensors="pt")).last_hidden_state[0, 1:-1]
            acc_v = acc_v + h.sum(0); tot += h.shape[0]
        X[i] = (acc_v / tot).numpy()
np.save(OUT / "esm2_35m_raw.npy", X)
json.dump({"model": MODEL, "hf_commit": rev, "uniprot": {s: acc[s] for s, _ in rows},
           "fasta_sha256": hashlib.sha256(open(COV / "uniprot150.fasta", "rb").read()).hexdigest()}, open(OUT / "esm2_meta.json", "w"), indent=1)
print("done", X.shape, rev)
