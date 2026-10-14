# Feature freeze packet (step 0 complete): 150 Arc knockdown genes

> **Learning-experiment artifact.** No training, no GPU, no model code. Built 2026-10-13 (America/Denver) on CPU. The frozen arrays live on pop-os at **`/ml2/altgan/features/`** and are **not committed**. The machine-readable record is [`features/manifest.json`](../features/manifest.json). The builder scripts are in `features/`. Arc State is **excluded**.

## 1. Shared gene order

All matrices share one row order: the 150 knockdown genes sorted by the **dataset's Ensembl ID** (`var["gene_id"]` of `adata_Training.h5ad`). The ordered list (`ensembl`, plus parallel `symbols` and `lookup_symbols`) is `manifest.json → gene_order`.
- Mapping rule: `TAZ` = ENSG00000102125 = **TAFAZZIN**. It is looked up as TAFAZZIN in symbol-keyed sources and never through the `TAZ` alias that WWTR1 shares.
- Missing genes get an **all-zero row**. They are listed and never imputed.

## 2. Per-source table

"Usable rows" counts rows that are non-missing, not all-zero, and not byte-duplicated with another row.

| source | file (on pop) | dtype / shape | sha256 | URL and version | license: model or data | license: outputs | usable rows | missing / degenerate |
|---|---|---|---|---|---|---|---|---|
| Gene2vec | `gene2vec.npy` | float32 / 150×200 | `5f2ace08d7daee9f404ba5cdf41003eb767a7bd2c7c3670e604219318a9ffc72` | github.com/jingcheng-du/Gene2vec, `gene2vec_dim_200_iter_9.txt` (2018) | MIT | MIT (data file in repo) | **149** | ZNF714 missing |
| GO (TF-IDF + SVD) | `go_svd.npy` | float32 / 150×64 | `9601e6935fb986b94161bddb38e7405daa3cfb763f17be0df2b545c5de54972b` | GOA via MyGene.info API, queried 2026-10-14 UTC (**GO release not pinned**) | GO data CC BY 4.0 | CC BY 4.0 attribution carries to derived features | **148** | ZNF426 and ZNF286A have identical GO term sets, so their rows are duplicates |
| STRING (spectral) | `string_spectral.npy` | float32 / 150×64 | `6a428429b6556f30bc690d252cb6d7910665cb52755d76a9d0795cfcf15c90f5` | stringdb-downloads.org, `9606.protein.links.v12.0` (STRING v12.0) | CC BY 4.0 | CC BY 4.0 attribution | **145** | ZNF426, NREP, ZNF714, SMAGP, ZNF286A have no edges ≥ 700 |
| ESM-2 35M | `esm2_35m.npy` | float32 / 150×480 | `60fe2011691dff0cb6639b689315e58a99ed08a6c1cc781e1327f5fd605727d6` | HF `facebook/esm2_t12_35M_UR50D` @ `6fbf070e65b0b7291e7bbcd451118c216cff79d8` (ungated) | MIT | MIT; no separate output restriction | **150** | CALM3 → P0DP25. CALM1/2/3 encode the identical protein, so their rows *would* be duplicates, but only CALM3 is among the 150. |
| GenePT (ada-002) | `genept_ada.npy` | float32 / 150×1536 | `1b32182c53b8bc5dddc26a78f9ab0e9a8e0221d2d1145240a55a12b5089db811` | Zenodo 10833191 v2 (zip md5 `3f6ce4317e3a0091978ae5cb8fbf05a3`, verified) | CC BY 4.0 (dataset) | CC BY 4.0 attribution; the vectors are OpenAI ada-002 outputs (OpenAI terms assign outputs to the user) | **148** | **TAZ excluded**: the pickle key `TAZ` is byte-identical to `WWTR1` (wrong gene) and `TAFAZZIN` is absent. MBTPS1 is absent. No alias fallback. |

Further detail in the manifest:
- **Inputs and processing:** the input-file sha256 for every source, the ESM-2 FASTA hash and UniProt accessions, and the processing notes.
- **ESM-2 pooling:** last hidden layer, mean over residues. Proteins longer than 1022 aa are split into 1022-aa windows and averaged weighted by residue count.

### SVD fitting provenance

| | GO | STRING |
|---|---|---|
| input | `features/go_annotations_150.json` on pop (per-gene GO IDs, BP+MF+CC, all evidence codes); sha256 in the manifest | `9606.protein.links.v12.0.txt.gz`, sha256 `manifest → sources.string_spectral.fit.input_sha256`; ID map `9606.protein.aliases.v12.0.txt.gz` (Ensembl_gene → STRING protein) |
| threshold | none (binary membership) | combined_score ≥ 700 |
| normalization | TF-IDF (smooth idf over the 150) then L2 row norm | symmetric, weights = score/1000, D^-1/2 A D^-1/2 |
| solver | `numpy.linalg.svd` (exact LAPACK), sign fixed by loading sum | `scipy.sparse.linalg.svds` (ARPACK), features = U·S, sign fixed by column sum |
| seed | n/a (deterministic) | 0 (`v0` and `random_state`) |
| components | 64 | 64 |
| gene universe | the 150 knockdown genes only (transductive on annotations; no expression used) | all human STRING proteins with ≥ 1 edge ≥ 700 |

## 3. Model-free sanity check (120 training perturbations only)

These are the protocol rules (full text in `features/sanity.py`), and they are part of the freeze:
- **Held-out:** the 30 held-out perturbations were **not used and are not reported**.
- **Delta:** pseudobulk (all cells, log1p normalize_total 1e4) minus non-targeting. Computed on pop CPU by `features/make_deltas.py`; sha256 in the manifest.
- **Columns:** the 150 knockdown-target genes are removed so that self-knockdown does not dominate.
- **Similarity:** cosine for both deltas and features.
- **Spearman:** over all usable training pairs.
- **NN hit@k:** inner 5-fold CV (seed 0). Each query's feature-nearest neighbour is searched only among the other folds' genes, and counts as a hit if it is among the query's top-k by delta similarity in that same reference set.
- **Shuffled control:** feature rows permuted among the usable training genes, 1000 permutations (seed 0). p-values are one-sided, (1 + #≥) / 1001.

| source | n | Spearman (shuffle mean; p) | NN hit@1 (shuffle; p) | hit@5 (shuffle; p) | hit@10 (shuffle; p) |
|---|---|---|---|---|---|
| Gene2vec | 119 | −0.005 (0.000; 0.59) | 0.008 (0.011; 0.72) | 0.092 (0.053; 0.076) | 0.126 (0.106; 0.30) |
| GO SVD | 118 | **0.052** (0.000; **0.004**) | 0.051 (0.010; 0.005) | 0.127 (0.053; 0.004) | 0.186 (0.106; 0.008) |
| STRING spectral | 116 | 0.025 (−0.001; 0.14) | **0.060** (0.011; **0.002**) | **0.190** (0.055; **0.001**) | **0.233** (0.109; **0.001**) |
| ESM-2 35M | 120 | 0.036 (−0.003; 0.13) | 0.058 (0.011; 0.001) | 0.117 (0.053; 0.008) | 0.167 (0.105; 0.039) |
| GenePT ada | 119 | 0.013 (0.000; 0.30) | 0.042 (0.010; 0.010) | 0.126 (0.053; 0.004) | 0.168 (0.105; 0.028) |

**Read:**
- **Effects are small.** Global similarity correlations are tiny: ρ ≤ 0.05 for every source.
- **Nearest-neighbour signal.** STRING, GO, ESM-2 and GenePT all place the nearest neighbour in the delta top-k more often than shuffled. Gene2vec does not.
- **Strongest:** STRING spectral has the largest NN enrichment (hit@5 0.19 vs 0.055). GO is the only source with a significant global correlation.
- **Multiple comparisons.** There are 5 sources × 4 metrics, so treat single p-values near 0.01–0.04 as weak.
- **Shortlist.** This screen suggests STRING and GO for the first feature arms, with ESM-2 as a third. It is a model-free screen on training genes only, not evidence that the GAN will generalize.

## 4. Locked evaluation rules (for the modeling phase)

1. **Selection only on training perturbations.**
   - Choose the feature source, the projection, the NN or other model, and any hyperparameters using **only the 120 training perturbations**, with an inner validation split (e.g. the 5-fold split above, or a fixed 96/24 split with seed recorded).
   - The 30 held-out targets are used **once**, to certify the pre-registered choice. They must not be used for both selection and certification.
   - This packet's sanity check follows that rule.
2. **Head-independent controls are mandatory.** Run the frozen-generator probe (generator trained, then frozen; a fresh head retrained on its profiles) and these generator-only metrics, which need no decision head:
   - expression MAE, Pearson-delta, DE overlap and PDS vs `mean_effect`
   - a feature-kNN baseline that predicts a held-out delta as the mean delta of its k feature-neighbours among training genes
   - the shuffled-feature control
3. **Paired deltas.** For every metric, report the paired difference per target and per seed against **both** the B0 one-hot baseline **and** the shuffled-feature control, with target-level bootstrap 95% CIs. "Arm beats baseline" alone is not sufficient.
4. **Success bar.** On 30 held-out targets, one AUROC measurement has an SE of roughly √(AUC(1−AUC)/30) ≈ 0.09 near 0.6, so "AUROC ≥ 0.65 in 2 of 3 seeds" alone is noisy and not paired. Proposed replacement:
   - **(primary)** the paired held-out AUROC difference vs **both** B0 and the shuffled control, pooled over 3 training seeds with targets as the bootstrap unit, has a 95% CI lower bound > 0
   - **(magnitude)** mean held-out AUROC ≥ 0.65
   - **(generator)** held-out MAE and DE overlap paired differences vs `mean_effect` are favourable, with CIs excluding 0

   All three are pre-registered before the held-out set is touched.
5. **Save exact scored tensors.**
   - Per target × candidate scores and labels (`scores.parquet` or `.pt`), the exact candidate profile tensors, and the target lists.
   - The perturbation dictionary: this gene order, feature file sha256 and missing policy.
   - An invocation manifest: argv, environment, git SHA, data, feature and checkpoint hashes, seeds.
6. **Zero-update evaluation.** The evaluator loads the checkpoint into freshly constructed modules with **no** training step. It wraps every optimizer's `step` and asserts **zero calls** during evaluation, and it asserts parameter hashes are identical before and after. This replaces PR #3's `like_for_like.py` pattern of one throwaway step.

## 5. Licenses and exclusions

- **Accepted:** only licenses already accepted here (MIT, Apache-2.0, CC BY 4.0, CC0). No click-through or gated download was used: the ESM-2 HF repo is ungated, and the Zenodo, STRING, GitHub and UniProt downloads are open.
- **Excluded:**
  - **Arc State**: non-commercial license on weights *and* outputs; excluded by decision.
  - **scGPT**: the checkpoint is served via Google Drive links and HF mirrors without a license card. Not built; listed rather than accepted.
  - **Geneformer**: Apache-2.0 and ungated, but not built in this pass.
  - **Reactome**: data CC0, but only 125/150 coverage. Not built.
- **Gated or click-through sources skipped:** none were needed for the five built sources.
- **Terms kept separate:** model-use terms and output terms are separate columns in §2 and in `manifest.json`.

## 6. Open items / blockers

- The GO release is not pinned, because MyGene.info serves current GOA. Rebuild from a dated `goa_human.gaf` for a strict freeze.
- GO SVD is fitted on the 150 genes, including the 30 held-out genes' **annotations**, but never their expression. Annotations are label-free, but they do define the projection. If Ember wants a strict train-only fit, refit on a larger annotation universe or on the 120 training genes.
- Identical-feature pairs must be handled by the missing or degenerate policy in modeling: ZNF426/ZNF286A in GO, the zero rows elsewhere.
- **Recovery:** the STRING inputs, the GenePT pickle and the UniProt FASTA are cached on the box under `/workspace/altgan/cov/`. Their hashes are in the manifest, so the arrays can be rebuilt by re-running `features/build_features.py`, `build_esm.py`, and `finish_manifest.py`.
