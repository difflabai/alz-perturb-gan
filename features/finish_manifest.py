import json, pickle, sys, hashlib
import numpy as np
sys.argv = ["x"]
exec(open("/workspace/altgan/features/build_features.py").read().split("manifest = {")[0])  # helpers, gene order (no rebuild)
m = json.load(open(OUT / "manifest_partial.json"))
# ESM-2
X = np.load(OUT / "esm2_35m_raw.npy"); meta = json.load(open(OUT / "esm2_meta.json"))
m["sources"]["esm2_35m"] = finish("esm2_35m", X, [], {
    "url": "https://huggingface.co/facebook/esm2_t12_35M_UR50D", "version": "HF commit 6fbf070e65b0b7291e7bbcd451118c216cff79d8",
    "license_model_or_data": "MIT (model card; fair-esm repo MIT)", "license_outputs": "MIT terms; no separate output restriction",
    "dim": 480, "sequences": "UniProt Swiss-Prot canonical, fetched 2026-10-14 UTC via rest.uniprot.org/uniprotkb/accessions",
    "fasta_sha256": meta["fasta_sha256"], "uniprot": meta["uniprot"],
    "processing": "eval mode, no grad, last hidden layer mean over residues (CLS/EOS dropped); >1022 aa: non-overlapping 1022-aa windows, residue-weighted mean",
    "calm_note": "CALM3 -> P0DP25; CALM1/2/3 encode identical proteins, so their ESM rows would be identical; only CALM3 is among the 150"})
(OUT / "esm2_35m_raw.npy").unlink()
# GenePT
d = pickle.load(open("/workspace/altgan/cov/GenePT_emebdding_v2/GenePT_gene_embedding_ada_text.pickle", "rb"))
X = np.zeros((150, 1536), np.float32); miss = []
for i, s in enumerate(sym_for_lookup):
    if s in d: X[i] = np.asarray(d[s], np.float32)
    else: miss.append(symbols[i])
m["sources"]["genept_ada"] = finish("genept_ada", X, miss, {
    "url": "https://zenodo.org/records/10833191 (GenePT_emebdding_v2.zip, md5 3f6ce4317e3a0091978ae5cb8fbf05a3)",
    "input_file": "GenePT_gene_embedding_ada_text.pickle", "input_sha256": sha("/workspace/altgan/cov/GenePT_emebdding_v2/GenePT_gene_embedding_ada_text.pickle"),
    "version": "Zenodo v2 (2024-03-18)", "license_model_or_data": "CC BY 4.0 (dataset)",
    "license_outputs": "CC BY 4.0 attribution; vectors are OpenAI text-embedding-ada-002 outputs (OpenAI terms assign outputs to the user)",
    "dim": 1536, "processing": "raw vectors",
    "key_note": "pickle key 'TAZ' is byte-identical to 'WWTR1' (wrong gene) and 'TAFAZZIN' is absent -> TAZ left missing; MBTPS1 absent; no alias fallback"})
m["excluded"] = {"arc_state": "excluded by decision (non-commercial license on weights and outputs)",
                 "scgpt": "not built: checkpoint is distributed via Google Drive links / unofficial HF mirrors with no license card; not needed for this pass",
                 "geneformer": "not built in this pass (Apache-2.0, ungated); token embeddings need the ~100M+ checkpoint",
                 "reactome": "not built in this pass (125/150 coverage)"}
m["gated_or_clickthrough_skipped"] = []
json.dump(m, open(OUT / "manifest.json", "w"), indent=1); (OUT / "manifest_partial.json").unlink()
