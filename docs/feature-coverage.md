# Feature-source coverage of the 150 Arc knockdown genes (step 0)

> **Learning-experiment audit.** No training, no evaluation, no GPU, no model code. Checked 2026-10-13 (America/Denver). This is step 0 of `docs/gene-feature-plan.md`. It is a lookup of vocabularies and metadata only, and no feature vectors were computed.

**Gene list.** The 150 perturbation symbols are `train_names + eval_names` from `runs/gpu-run-2/split.json` on PR #3 (120 train, 30 held-out). Their Ensembl IDs come from `var["gene_id"]` of `/ml2/altgan/data/adata_Training.h5ad`, read in backed mode with no expression loaded. All 150 symbols are present in the h5ad `var`.

**Mapping.** "Symbol" coverage means an exact match of the dataset symbol. "Ensembl" coverage means a match of the dataset's own Ensembl ID, which is the recommended key wherever the source supports it.

## Coverage table

| source | what was checked (small files only) | license (verified 2026-10-13) | dim | coverage by symbol | coverage via Ensembl / alias fix | misses and notes |
|---|---|---|---|---|---|---|
| **Gene2vec** *(checked earlier, in PR #4)* | `gene2vec_dim_200_iter_9.txt` (GitHub) | MIT (repo) | 200 | 149/150 | 149/150 (no Ensembl key) | **ZNF714** absent. Needs the missing-feature policy. |
| **scGPT gene tokens** (whole-human) | `vocab.json` (60,697 tokens) from HF `wanglab/scGPT-human`. The official checkpoint is distributed via the Google Drive links in the bowang-lab README. | code MIT (`bowang-lab/scGPT`). The checkpoint carries no separate license file and the HF mirror has no license card. Treat as MIT-by-repo, unconfirmed for the weights. | 512 | **150/150** | n/a (symbol vocab) | `TAZ` is present and `TAFAZZIN` is absent, so the vocab uses the old symbol. It presumably means tafazzin, given that it is a CELLxGENE-derived vocab. **Unconfirmed**: the vocab has no Ensembl IDs. |
| **Geneformer tokens** (V2, gc104M) | `token_dictionary_gc104M.pkl` (20,275 Ensembl tokens) and `gene_name_id_dict_gc104M.pkl` from HF `ctheodoris/Geneformer` | Apache-2.0 (model card) | from checkpoint (V2-104M hidden size; V2-316M reported 512) | 149/150 | **150/150** by Ensembl ID | The symbol dict lacks `TAZ` but maps `TAFAZZIN` to ENSG00000102125, which is the dataset's TAZ ID. **Use Ensembl IDs.** |
| **GenePT** (NCBI-summary text embeddings) | **Proxy only.** The 574 MB Zenodo zip was not downloaded, because only small files were allowed. Checked instead whether each gene has an NCBI gene summary (mygene.info `summary`), which is GenePT's input text. | CC BY 4.0 (Zenodo 10833191). Built from OpenAI embeddings, so cite GenePT. | 1536 (ada-002) / 3072 (3-large) | 149/150 have an NCBI summary | 150/150 counting `TAZ`→TAFAZZIN | **Unverified against the actual pickle keys.** GenePT keys are upper-case symbols plus HGNC aliases. The alias `TAZ` belongs to both TAFAZZIN and WWTR1, so a lookup of the `TAZ` key may return either. Verify by loading the pickle when downloads are allowed. |
| **ESM-2 protein embeddings** | UniProt Swiss-Prot accession per gene via mygene.info (sequences not fetched, no forward pass) | ESM-2 repo `facebookresearch/esm`: MIT, archived. Not the EvolutionaryScale ESM C/ESM3 Cambrian licenses. | 1280 (650M) / 480 (35M) | 149/150 have a reviewed (Swiss-Prot) protein | **150/150** (`TAFAZZIN` → Q16635) | All 150 are protein-coding. **CALM3** maps to three accessions, because CALM1/2/3 encode an identical protein (P0DP23/24/25). Use P0DP25 for CALM3. Its ESM embedding will equal CALM1/2's, which is harmless here because only CALM3 is in the list. |
| **GO annotations** | mygene.info `go` field (GOA via NCBI) | GO data CC BY 4.0 (geneontology.org). Pin a release date or DOI. | raw terms → SVD 64–256 | 149/150 have ≥1 GO term | **150/150** (`TAFAZZIN` has BP terms) | Sparse genes, by GO-term count: WSB2 (5), LAD1 (6), MED13L / NREP / PRR12 (7). They will have weak GO features. |
| **Reactome pathways** | Reactome AnalysisService identifier mapping (human, no interactors), 150 symbols + `TAFAZZIN` | **Data CC0** (reactome.org/license). Software Apache-2.0, illustrations CC BY 4.0. *This corrects PR #4, which guessed CC BY 4.0.* | ~10³ pathways → SVD 64 | **125/150** mapped to any Reactome entity | 125/150 | 25 not in Reactome: LAD1, SLC25A3, ZNF593, CDCA2, TXNDC9, MAST2, NREP, MKI67, TRAM2, CAMSAP2, SIX4, SEC62, TMSB10, HMGN1, METTL17, KLF10, SMAGP, EID2, EWSR1, DNAJA3, PHF14, TMBIM6, LZTR1, PRR12, NIT1. **`TAZ` was reported "found", probably through its WWTR1 alias, which would be the wrong gene.** Query by Ensembl or UniProt ID instead. |
| **STRING** *(checked earlier, in PR #4)* | STRING API `get_string_ids` | CC BY 4.0 | 64–128 (graph embedding) | 150/150 | — | `TAZ` resolution not inspected. Re-check it maps to TAFAZZIN (9606.ENSP for ENSG00000102125), not WWTR1. |
| **Arc State** | not checked (owned by Ember; license decision pending) | code CC BY-NC-SA 4.0; weights and outputs under the Arc non-commercial license | — | — | — | Blocked on the license and ownership decision (plan §7, Q1). |

## Symbol-mapping issues found

1. **TAZ is the only problematic symbol.** In the dataset, `TAZ` = **ENSG00000102125 = TAFAZZIN** (HGNC renamed it). `TAZ` is also a current alias of **WWTR1** (ENSG00000018408), which is a *different* gene and is itself in the h5ad gene space. Each source handles it differently:
   - Geneformer: missing by symbol, fine by Ensembl ID.
   - mygene, NCBI and UniProt: not found by symbol, fine as TAFAZZIN.
   - Reactome: likely mis-mapped to WWTR1.
   - GenePT: ambiguous alias key.
   - scGPT and Gene2vec: old symbol present, presumably tafazzin.

   **Rule: map via the dataset's Ensembl ID. Where a source only has symbols, map ENSG00000102125 → TAFAZZIN and never resolve `TAZ` through aliases.**
2. **ZNF714:** missing only in Gene2vec. Present in all other sources checked (Geneformer by Ensembl, NCBI summary, Swiss-Prot, GO, Reactome found).
3. **CAST:** mygene returns two records, an Ensembl-only stub and Entrez 831. Use the one whose Ensembl ID matches the dataset (ENSG00000153113).
4. **CALM3:** shares its protein sequence with CALM1/2 (see ESM-2 row).
5. Apart from TAZ, all 149 symbols resolve in mygene to the same Ensembl ID as the dataset.

## Blocking / not done

- **GenePT** coverage is a proxy until the 574 MB zip can be downloaded and its keys checked. The TAZ alias collision needs explicit handling.
- **scGPT** weight license is not separately stated. The vocab's `TAZ` meaning cannot be confirmed without Ensembl IDs.
- **State**: blocked on the license and ownership decision.
- **Model-free neighbour sanity check** (plan step 0, second half) needs the actual feature vectors, so it was **not done** here.

Scratch files: `/workspace/altgan/cov/` on the box (vocab and metadata files, mygene, and Reactome responses). Not committed.
