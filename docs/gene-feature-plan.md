# Gene-feature plan for alz-perturb-gan

> **Learning-experiment plan, not a result.** This document proposes work; it reports no new training or evaluation. It was written during a research hold (no training, no evaluation, no GPU, no code changes). The repository is an exploratory learning experiment on H1 hESC knockdowns from the public Arc Virtual Cell Challenge training release. It is preparation for the Prima Mente Alzheimer's challenge and makes no Alzheimer's or therapeutic claim.

## 1. Problem

PR #3 (`decision-label-fix`) made the decision head's label well posed: is candidate perturbation P the source of target pseudobulk T? It then evaluated the run-2 checkpoint like for like, with every candidate represented by the generator's 64-draw averaged profile, the same metric, and the same `decision_pair_report` protocol:

| targets | n | AUROC (all 150 candidates) | balanced accuracy |
|---|---|---|---|
| training perturbations | 120 | **0.895470** | 0.804 |
| held-out perturbations | 30 | **0.534325** | 0.517 |

**Narrowed claim (from PR #3):** this shows a generalization gap in the joint generator-plus-head pipeline. It does **not** isolate the cause. It does not show that the generator alone is responsible, and it does not rule out the head memorizing training identities.

A likely contributor is that perturbations are encoded as **one-hot identity vectors**. An unseen perturbation's column receives no gradient, so nothing in the model can relate a held-out gene to any training gene. Informative gene features are the obvious lever. The ablations in §4 are designed to show *which* component benefits, instead of assuming it.

Note on code state: on `main`, the decision head still takes the one-hot embedding. On PR #3's branch, the head takes pseudobulk profiles (control, candidate profile, target) and no embedding. This plan assumes PR #3's head as the baseline. Where the head gets features directly (§4, arms H and GH), they are an added input.

## 2. Feature options

Gene list: the 150 knockdown targets in `runs/gpu-run-2/split.json` on PR #3, split into 120 training and 30 held-out. "Coverage" is the number of those 150 symbols that the source maps directly.

Two coverage checks were actually run for this plan (2026-10-13):
- **Gene2vec: 149/150.** Exact symbol match against `gene2vec_dim_200_iter_9.txt`; the only miss is `ZNF714`.
- **STRING: 150/150.** All 150 resolve via the STRING API `get_string_ids` (species 9606).

All other coverage figures below are **unverified expectations**. Step 0 of the experiment (§4) computes them exactly.

Licenses were checked by a quick web search on 2026-10-13 and should be re-checked before any redistribution.

### (a) Pretrained gene embeddings

| feature | what it is | how to get it | license (checked) | dim | coverage of the 150 |
|---|---|---|---|---|---|
| **Gene2vec** | word2vec on gene co-expression pairs from 984 GEO datasets | `github.com/jingcheng-du/Gene2vec`, `pre_trained_emb/gene2vec_dim_200_iter_9.txt` | MIT (repo) | 200 | **149/150 verified** (missing ZNF714 → needs a fallback) |
| **GenePT** | OpenAI text embeddings of NCBI (+UniProt) gene summaries | Zenodo record 10833191 (`GenePT_emebdding_v2.zip`, 574 MB; pickles keyed by upper-case symbol) | CC BY 4.0 | 1536 (ada-002); 3072 (text-embedding-3-large) | expected near-complete (~33k genes + ~60k aliases); unverified |
| **scGPT gene tokens** | input gene-embedding table of the scGPT `whole-human` checkpoint | `github.com/bowang-lab/scGPT` checkpoint folder + `vocab.json` | MIT (repo); check the checkpoint terms | 512 | expected high (vocab ~60k symbols); unverified |
| **Geneformer gene tokens** | token-embedding table of Geneformer V2 (Ensembl-ID vocabulary) | `huggingface.co/ctheodoris/Geneformer` | Apache-2.0 | 768 (V2-104M) / 512 reported for V2-316M; read from the checkpoint | ~20k protein-coding vocabulary; needs symbol→Ensembl mapping; unverified |
| **ESM-2 protein embeddings** | mean-pooled per-residue embeddings of each gene's canonical protein | `facebookresearch/esm` (ESM-2, e.g. `esm2_t33_650M_UR50D`) + UniProt canonical sequences | ESM-2 repo is MIT. The newer EvolutionaryScale ESM C/ESM3 use Cambrian licenses; do not mix them up. **Re-verify.** | 1280 (650M) / 480 (35M) | protein-coding genes only; non-coding or ambiguous symbols miss; unverified. Computing it needs a forward pass over 150 sequences (cheap, but it is compute, so it waits for the hold to lift) |

### (b) GO / pathway / network features

| feature | what it is | how to get it | license (checked) | dim | coverage |
|---|---|---|---|---|---|
| **GO annotations** | binary gene × GO-term matrix (BP/MF/CC), filtered to terms with ≥ k genes among the 150 or a larger background, optionally SVD-reduced | geneontology.org downloads (`goa_human.gaf`); pin a release date / Zenodo DOI | CC BY 4.0 (data); attribution and release citation required | raw ~10³–10⁴ terms → SVD 64–256 | expected high; unverified |
| **Pathways (Reactome)** | binary gene × pathway membership | reactome.org downloads | check before use (believed CC BY 4.0; **not verified here**) | ~10³ → SVD 64 | unverified |
| **STRING network** | node2vec / spectral embedding of the STRING human PPI graph (combined score ≥ 700), or a row of adjacency to training genes | string-db.org downloads (`9606.protein.links.v12.0`) or API | CC BY 4.0 | 64–128 (embedding) | **150/150 verified** to map |

The network option has a useful property for this task: "a held-out gene's neighbours among the 120 training genes" is a direct, interpretable bridge between the splits.

### (c) State / hybrid

| feature | what it is | how to get it | license (checked) | dim | coverage |
|---|---|---|---|---|---|
| **Arc State embeddings** | perturbation or gene representations from Arc's State models: SE-600M embeddings, or ST transition models trained on Replogle | `github.com/ArcInstitute/state`; HF `arcinstitute/SE-600M`, `ST-SE-Replogle`, `ST-HVG-Replogle` | code CC BY-NC-SA 4.0; weights and **outputs** under the Arc State Model Non-Commercial License + Acceptable Use Policy (non-commercial entities only; commercial use needs a separate paid licence) | model-dependent (read from the checkpoint) | depends on the State vocabulary; unverified |

**Ownership caveat:** the earlier brief assigned State and pretrained gene embeddings to Ember. This plan only lays out options for Ember to take. The licence terms of (c) also need a decision from Mikkel (§7) before anything is produced.

## 3. Integration point (interface sketch only; no code in this PR)

Replace the one-hot perturbation embedding with a frozen feature vector passed through a small learned projection. One shared table is used everywhere a perturbation identity appears.

```python
class PerturbationFeatures(Protocol):
    names: list[str]              # library order, persisted as the perturbation dictionary
    source: str                   # e.g. "gene2vec_dim_200_iter_9", with version and checksum
    def raw(self, names: list[str]) -> Tensor: ...    # [N, F] frozen features
    def missing(self) -> list[str]: ...                # symbols with no feature (filled by policy)

class FeatureProjector(nn.Module):  # trainable, F -> emb_dim
    def __init__(self, in_dim: int, emb_dim: int, dropout: float = 0.1): ...
    def forward(self, raw: Tensor) -> Tensor: ...      # LayerNorm -> Linear -> (optional) dropout

# PerturbationSet keeps its fields; Perturbation.embedding becomes projector(raw[name]).
# Generator:     ConditionalGenerator(z, control, embedding=proj_G(raw))      # same signature
# Decision head: DecisionModel(control, profile, target, pert_embedding=None)
#                pert_embedding = proj_H(raw[candidate]) is optional; arms below toggle it
```

Design rules:
- Features are frozen, and only the projections train. This keeps "features help" separate from "fine-tuning a large model helps".
- Missing symbols use an explicit policy, either a zero vector plus a "missing" indicator bit, or exclusion. The policy is recorded in the manifest and never silently imputed from training labels.
- The one-hot arm stays available as `source="onehot"`, so every comparison is in-code and like for like.
- Features are standardized with statistics from **training perturbations only**.

## 4. Experiment design

**Step 0 (no training): feature audit.** For each candidate source, record exact coverage of the 150 genes, the missing list, dimension, file checksum and licence. Also compute a cheap, model-free sanity check: whether a held-out gene's nearest training neighbours in feature space have more similar true pseudobulk deltas than random training genes do. A source that fails this has little chance and can be dropped early.

**Main grid.** Fix one feature source per grid (start with Gene2vec or STRING, which are verified and cheap). Then run these arms on the same split (seed 42), the same cell caps and the same step budget:

| arm | generator input | head pert input | what it isolates |
|---|---|---|---|
| **B0** | one-hot | none (PR #3 head) | baseline; reproduces PR #3 numbers |
| **G** | features | none | does a better generator alone close the held-out gap? |
| **H** | one-hot | features | can the head use features directly, despite a weak generator? |
| **GH** | features | features | joint effect |
| **G-frozen probe** | features, trained, then **frozen** | none; head retrained from scratch on frozen-generator profiles | separates generator quality from joint co-adaptation |
| **head-on-truth probe** | none: the head scores *true* held-out pseudobulks as candidate profiles (oracle profiles) | none | upper bound for the head with perfect profiles. If this is also ~0.5, the head (not the generator) is the limit |
| **shuffled-feature control** | features permuted across genes | as in GH | rules out "more parameters / regularization" explanations |

The head-on-truth probe matters most for attribution. A high score there combined with low G/GH points to profile quality. A low score there points to the head or label design. That is exactly the question PR #3 could not answer.

**Metrics (per arm, train and held-out, same protocol):**
- decision head: AUROC over all candidates, balanced accuracy and AUROC, Brier, top-1, MRR, recall@3
- generator: held-out MAE, Pearson-delta, top-10 DE overlap, local PDS
- Arc cell-eval VCC profile for each generator arm

**Reproducibility and provenance (addresses Ember's PR #3 limits).** Every evaluation writes:
- `scores.parquet`: per target × candidate score, label and split
- `targets.json`: exact evaluated target lists for train and held-out, in order
- `perturbation_dictionary.json`: library order, name → index, feature source, checksum, missing policy
- `manifest.json`: full argv, environment variables, git SHA, package versions, data SHA-256, checkpoint SHA-256, seeds, device, start and end time
- **zero-update evaluation**: an evaluator that constructs modules and loads the checkpoint with **no optimizer step**, unlike `scripts/like_for_like.py`. A test asserts the parameter hashes before and after evaluation are identical, and the RNG is seeded only after loading.

**Statistics.** With only 30 held-out targets, one hit moves top-1 by 0.033. So: report bootstrap 95% CIs over targets; run ≥ 3 seeds per arm, varying training seed at a fixed split; optionally repeat with 2–3 alternative held-out splits (split seeds) for the most promising arm.

## 5. Success criteria

Pre-registered, judged on held-out targets with the like-for-like protocol:

1. **Primary:** in the best feature arm, held-out AUROC is ≥ 0.65 and its bootstrap CI excludes the B0 value (0.534). This must hold in ≥ 2 of 3 seeds.
2. **Generator:** held-out MAE and DE overlap beat `mean_effect` (MAE 0.0143, DE overlap 0.0289 in run 2), not just one-hot GAN.
3. **Ranking:** held-out MRR exceeds both random (≈0.037) and the best baseline (additive 0.046) with a CI that excludes them.
4. **Attribution:** the shuffled-feature control does **not** reproduce the gain, and the arm pattern (G vs H vs GH vs probes) supports a stated mechanism, worded no more strongly than the evidence.
5. **Provenance:** all files in §4 are present and the zero-update test passes.

A null result, with features not closing the gap, is also a valid outcome and should be written up as such.

## 6. Compute estimate

Reference: run 2 used 20,000 steps on 1 × RTX A6000. That took ~30 min of GAN training plus ~20 min of data loading, with a peak of 1.5 GiB allocated.
- Step 0 audit: CPU only, minutes, plus downloads (GenePT 574 MB, Gene2vec ~40 MB, GO/STRING files).
- Data loading: cache the pseudobulks and sampled cells once, e.g. to `.npz` under `/ml2/altgan/.cache`. This saves ~20 min per run.
- Main grid: 7 arms × 3 seeds × ~30–40 min ≈ **11–14 GPU-hours** for one feature source. Each run needs ≈ 2 GiB, so runs can share a GPU 2–4 at a time if free memory allows. Wall time is then ≈ 4–6 h.
- Additional feature sources: ≈ +4–5 GPU-hours each, if only B0/G/GH/shuffled are rerun.
- ESM-2 / scGPT / Geneformer / State feature extraction: well under 1 GPU-hour for 150 genes, but blocked by the hold and (for State) by the licence question.

## 7. Open questions

1. **Licence and ownership:** may State-derived features (non-commercial licence covering outputs) be committed to a public MIT-style repo or used for the Prima Mente challenge at all? Ember owns State and embeddings, so who implements which arm?
2. **Split policy:** keep the single seed-42 split (comparable to PR #3) or move to multiple splits now?
3. **Feature leakage:** some sources (e.g. scGPT and Geneformer pretrained on corpora that may include perturbation screens, GenePT built from literature) might encode knockdown outcomes for these very genes. Is that acceptable for a learning experiment? How do we report it?
4. **Missing-gene policy:** zero plus indicator, or exclude (e.g. ZNF714 for Gene2vec)?
5. **Projection capacity:** a linear projection or a small MLP? Is a frozen projection (no training) also needed as a probe?
6. **Head design:** should the head get features at all, or should it stay profile-only so the generator carries all identity information?
7. **Merge order:** does this depend on PR #3 merging first (it assumes PR #3's head and evaluator)?
8. **Compute window:** when does the research hold lift, and which GPU budget applies?
