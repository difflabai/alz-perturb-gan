# Data expansion inventory (improve_20261109)

Component: data-expansion inventory (read-only research). Date of all checks: **2026-11-09**.
Author: Claude Code worker. No count or expression matrix was opened for any donor; only
donor/cell metadata, public listings and small public metadata files were read (details in
"What was run").

Convention: **[V]** = verified on 2026-11-09 by a command or page fetch listed here;
**[I]** = inference or recommendation, not verified.

---

## 0. Bottom line

* Today's study trains on **2-3 Not AD donors per inner fold** (dev37 contains 4). The whole
  local SEA-AD MTG cohort contains only **9 Not AD donors** (82 donors). Even taking every local donor
  gives at most 9 Not AD; a 3-fold design over all 82 donors would train on about 6. [V]
* The 25 SEA-AD donors outside the fitting universe are **7 "development/validation" donors**
  (examined many times) plus **18 "historically exposed audit" donors** (they were scored on
  2026-10-27). Between them they hold **3 Not AD** (1 + 2). Releasing them breaks written commitments
  (section 1.3) and this study's hard rule, and it gains just 3 controls. [V]
* The SEA-AD DLPFC (A9/DFC) data is public and open. It covers essentially the same donors (80 of 84
  aged donors on S3, 78 of our 82 in the CELLxGENE microglia file) with about the same number of
  microglia again (39,435 extra nuclei for our 82 donors). This gives more cells per donor and a
  second region, but **not more donors and not more controls**. [V]
* Getting many more low-pathology donors means using an external cohort. **ROSMAP** (Mathys 2023 / Sun 2023:
  423-427 donors, 189 without NIA-Reagan pathologic AD) and **PsychAD** (1,469 donors in an open
  CELLxGENE myeloid object; 141 aged controls with at least 100 myeloid nuclei) are the large ones. Neither
  publishes Thal phase openly, so **neither can be mapped to the four ADNC levels exactly**. Their
  pathology variables need a controlled-access Data Use Certificate (DUC) that Mikkel's institution must sign. [V]/[I]

---

## 1. Local SEA-AD MTG cohort (82 donors, 35,765 microglia)

Sources: `benchmark/rebuilt/mg_meta.json`, `rebuilt/candidate_cohort.json`, `benchmark/frozen_splits.json`
(split `S0f0`), `alz-audit/matched-benchmark-20261103/folds.json`, this study's `folds.json`.
Script: `data_expansion/donor_inventory.py` writes `data_expansion/donor_inventory.json` with a per-donor
table. [V]

### 1.1 Who the 25 donors outside the 57 are [V]

`frozen_splits.json["splits"]["S0f0"]` (written by `benchmark/freeze_splits.py` before any fitting) defines
fit 57 / validation 7 / test 18. `recovery_screen_20261029/prepare.py` built the 57-donor universe from
`fit_donors`, used the 7 `validation_donors` for screening, and quarantined the 18 `test_donors`
(`SCREEN_REPORT.md`).

| Group | Donors |
|---|---|
| 7 validation ("seven development donors") | H20.33.004, H20.33.016, H21.33.004, H21.33.007, H21.33.010, H21.33.014, H21.33.032 |
| 18 outer-test ("historically exposed audit") | H19.33.004, H20.33.002, H20.33.017, H20.33.020, H20.33.024, H20.33.034, H20.33.036, H20.33.038, H20.33.040, H20.33.045, H20.33.046, H21.33.005, H21.33.008, H21.33.015, H21.33.019, H21.33.022, H21.33.028, H21.33.045 |

Two further aged SEA-AD donors have Micro-PVM nuclei but fail the at-least-100-nuclei rule: H20.33.037
(63 nuclei, High) and H21.33.020 (79 nuclei, High) (`official_metadata/micro_pvm_candidates.json`). They add
no controls.

Not AD donors in the 25: H21.33.004 (validation; M, 93, Braak II, Thal 0), H19.33.004 (exposed; F, 80,
Braak IV, Thal 0) and H20.33.002 (exposed; F, 97, Braak IV, Thal 0).

### 1.2 Donor counts by ADNC stage [V]

| Set | n | Not AD | Low | Intermediate | High | microglia nuclei | F/M | APOE4 dose 0/1/2 |
|---|---:|---:|---:|---:|---:|---:|---|---|
| All 82 | 82 | **9** | 12 | 21 | 40 | 35,765 | 50/32 | 57/19/6 |
| Fitting universe 57 | 57 | **6** | 8 | 14 | 29 | 24,676 | 37/20 | 36/17/4 |
| Other 25 | 25 | **3** | 4 | 7 | 11 | 11,089 | 13/12 | 21/2/2 |
| of which validation 7 | 7 | 1 | 1 | 2 | 3 | 2,830 | 4/3 | 6/1/0 |
| of which exposed audit 18 | 18 | 2 | 3 | 5 | 8 | 8,259 | 9/9 | 15/1/2 |
| Matched fold 0 optimization | 33 | **3** | 4 | 8 | 18 | 14,762 | 21/12 | 20/10/3 |
| Matched fold 1 optimization | 33 | **3** | 4 | 8 | 18 | 15,211 | 25/8 | 20/12/1 |
| Matched fold 2 optimization | 36 | **3** | 5 | 9 | 19 | 14,762 | 22/14 | 22/10/4 |
| This study dev37 (fold0 opt + 4 calib) | 37 | **4** | 5 | 9 | 19 | 17,032 | 22/15 | 24/10/3 |
| Study inner fold 0 train | 25 | 3 | 3 | 6 | 13 | 11,486 | | |
| Study inner fold 1 train | 24 | 3 | 3 | 6 | 12 | 10,697 | | |
| Study inner fold 2 train | 25 | **2** | 4 | 6 | 13 | 11,881 | | |

No donor in the 82 is a "Neurotypical reference". All 9 Not AD donors are Thal 0 / CERAD Absent / no
dementia, and **7 of the 9 have Braak IV-V** (SEA-AD "Not AD" means amyloid-negative, A0; it is not
tangle-free). This matters for harmonising external "controls" (section 4.4).

### 1.3 What releasing the 25 into training would break [V: quoted documents]

* **This study's PROTOCOL.md, hard rules:** "Never read counts or expression for the 20
  `sealed_evaluation_donors_fold0`, nor for the 25 SEA-AD donors outside the 57-donor fitting universe."
  Releasing them needs a written amendment from the coordinator/owner before any fitting.
* **7 validation/development donors:** `matched-benchmark-20261103/SCHEDULE.md`: "The seven development
  donors get one locked descriptive comparison, without triggering further tuning."
  `scientific-proposal-20261103/PROPOSAL.md` keeps them "unchanged". Training on them uses up that
  locked comparison.
* **18 exposed audit donors:** `alz-audit/current-status.md`: "They are an exposed audit set, excluded from
  later tuning ... No independent confirmation set exists inside this cohort by relabeling these donors."
  `comparison-checklist.md`: "Do not tune against them, select checkpoints with them or call them
  independent held-out confirmation." `SCHEDULE.md` reserves "a later separately gated exposed-donor audit"
  (18 GPU-h envelope). Training on them removes the only remaining (already weak) audit set of the
  benchmark097 line.
* **All 82** were scored or fitted in the October 27 five-fold benchmark (`FINAL_REPORT.md`, S0 covers each
  donor once). Nothing inside this cohort is fresh. The data-expansion question is therefore about training
  signal, not about creating a clean test set.
* [I] Because this study already says it "makes no SOTA or held-out claim" and is separate from benchmark097,
  the scientific cost of releasing the 7 is small. The 18 are the only thing that would let the
  benchmark097 line run its planned exposed audit, so they should be released only by an explicit owner decision.

### 1.4 Not AD donors available under each option [V counts; I for the "~2/3" fold rule]

| Option | Donor pool | Not AD in pool | Not AD per 3-fold training split (approx. 2/3) | Not AD in a final all-pool model |
|---|---:|---:|---:|---:|
| Current study (dev37) | 37 | 4 | 2-3 (actual: 3/3/2) | 4 |
| Current fold-0 optimisation set (reference) | 33 | 3 | n/a | 3 |
| Fitting universe 57 (after the study's sealed 20 are released) | 57 | 6 | 4 | 6 |
| 57 + 7 validation | 64 | 7 | 4-5 | 7 |
| All 82 (also releases the 18 exposed) | 82 | 9 | 6 | 9 |

## 2. Cost of the 3-fold structure itself [V counts; I interpretation]

| Stage | matched fold-0 optimisation (33) | study inner train (24-25) | final on dev37 | final on 57 | final on 82 |
|---|---:|---:|---:|---:|---:|
| Not AD | 3 | 2-3 | 4 | 6 | 9 |
| Low | 4 | 3-4 | 5 | 8 | 12 |
| Intermediate | 8 | 6 | 9 | 14 | 21 |
| High | 18 | 12-13 | 19 | 29 | 40 |

* Inner training keeps about 65-68% of dev37 per stage. A final refit on dev37, run after evaluation is
  frozen, has 4 Not AD rather than 2-3. A final refit on 57, once the sealed fold-0 evaluation is finished,
  has 6 rather than 3. That doubles the controls with no new data. [V]
* [I] Under a simple donor-sampling model, the variance of a Not AD stage mean scales as 1/n. Going from 3
  to 6 Not AD donors halves it, and going to 9 cuts it to one third. The High-minus-NotAD contrast is bound
  by the Not AD term because there are 18-29 High donors. Split-half reproducibility (~0.10) should improve
  correspondingly but cannot be measured honestly at final refit, because no held-out donors remain.
* [I] Recommended rule: report model selection only from inner-fold held-out donors. Then refit the
  chosen recipe on the largest released pool, and use that refit only for descriptive outputs such as
  generated contrasts, with no new metric claims.

## 3. SEA-AD DLPFC (A9 / "DFC") for the same donors

### 3.1 Location, format, size, licence [V unless marked]

* Bucket listing (`https://sea-ad-single-cell-profiling.s3.amazonaws.com/?list-type=2&delimiter=/`,
  unauthenticated). The former `DLPFC/RNAseq/` folder now holds only a README: "All files from this
  directory have been moved to ... PFC/RNAseq/".
* `PFC/RNAseq/` (2026-06-22 Multiregion release): `SEAAD_DFC_RNAseq_final-nuclei.2026-06-22.h5ad`
  **38.52 GB** (HEAD: Content-Length 38,520,614,071, Last-Modified 2026-06-23); `..._all-nuclei.2026-06-22.h5ad`
  38.41 GB; `..._final-nuclei_metadata.2026-06-22.csv` 1.33 GB; `donors_objects/` holds **80 per-donor
  DFC h5ads** (38.29 GB total); `previous_objects/SEAAD_A9_RNAseq_*.2024-02-13.*` (older A9 release);
  `public_datasets/*-cell-annotation*.csv` (annotation only for 10 public PFC datasets).
* `Multiregion_2026/subclass_objects/SEAAD_Immune_HIP_MEC_LEC_ITG_MTG_FI_STG_DFC_AnG_V1C_RNAseq_final-nuclei.2026-06-22.h5ad`
  **3.27 GB**: the immune/microglia subclass across all 10 regions, MTG and DFC included.
  `pseudobulk_objects/..._Immune_..._pseudobulked.2026-06-22.h5ad` is 0.124 GB.
* Format (bucket and Multiregion READMEs): AnnData; `X` is log-normalised CP10k, `layers` holds raw
  UMI counts, 36,601 genes from the 10x 2020-A reference (**the same gene axis as our MTG counts**),
  10x 3' v3.1 singleome plus 10x Multiome, cellranger 6.1.2 / cellranger-arc 2.0.0. "All 84 donors from
  the previous MTG release had their MTG, DFC, and MEC profiled."
  https://sea-ad-single-cell-profiling.s3.amazonaws.com/README.md,
  https://sea-ad-single-cell-profiling.s3.amazonaws.com/Multiregion_2026/README.md
* Licence: "Allen Institute terms of use" (https://alleninstitute.org/legal/terms-use/), and no AWS account
  is required (https://registry.opendata.aws/allen-sea-ad-atlas/). FASTQs are controlled at Sage,
  syn26223298 (Gabitto et al. 2024, Nat Neurosci, https://doi.org/10.1038/s41593-024-01774-5, CC BY 4.0 article;
  states the A9 dataset is "1.2-million nuclei ... in the same 84 donors").
* CELLxGENE copy (older taxonomy): "Microglia-PVM - DLPFC: SEA-AD", dataset 100c6145-7b0e-4ba6-81c1-ffebed0d1ac4,
  **42,486 cells, 83 donors, 0.715 GB H5AD**, CC BY 4.0
  (https://api.cellxgene.cziscience.com/curation/v1/collections/1ca90a2d-2943-483d-b678-b809bf464c30;
  licence: https://cellxgene.cziscience.com/docs/032__Contribute%20and%20Publish%20Data). The collection itself
  warns that CELLxGENE downloads have "dramatically reduced metadata and a reduced number of genes" and
  recommends the official S3 files.

### 3.2 Donor and microglia census for DLPFC [V: obs-only HTTPS range reads, about 1.6 MB transferred]

From the CELLxGENE DLPFC Micro-PVM obs (`data_expansion/remote_cxg_seaad_dlpfc_microglia_census.json`). The
microglia supertypes there are Micro-PVM_1, _1_1-SEAAD, _2, _2_1-SEAAD, _2_2-SEAAD and _3-SEAAD, giving
41,516 microglia nuclei over 83 donors (80 aged plus 3 neurotypical references aged under 65, ADNC "Reference").

| Set (our metadata roles) | donors with DLPFC microglia (all have at least 100) | Not AD / Low / Int / High donors | extra DLPFC microglia nuclei |
|---|---:|---|---:|
| All 82 | 78 (missing: H20.33.043, H21.33.009, H21.33.010, H21.33.039) | 9 / 12 / 20 / 37 | 39,435 |
| fit 57 | 54 | 6 / 8 / 13 / 27 | 26,478 |
| dev37 | 35 (missing H20.33.043, H21.33.009) | 4 / 5 / 8 / 18 | 16,673 (MTG dev37: 17,032) |
| other 25 | 24 | 3 / 4 / 7 / 10 | 12,957 |

Caveats: this census uses the 2024 CELLxGENE object. The 2026-06-22 official DFC object has a newer
taxonomy (its `donors_objects/` lists 80 donors, including H20.33.037 and H21.33.020 and excluding the 4 above),
and its exact microglia counts were not checked: the metadata CSV is 1.33 GB, over the 50 MB download limit.

### 3.3 What a second region buys [I]

* It does buy roughly **2x cells per donor** (dev37: ~17k MTG + ~16.7k DLPFC). That gives a better per-donor
  pseudobulk, better nonzero-fraction estimates (the PPARG excess-zero issue) and a within-donor replicate for
  the noise ceiling (MTG versus DLPFC of the same donor).
* It does **not** buy more donors or more Not AD donors (4 in dev37 either way). The donor-level contrast noise
  that limits split-half reproducibility is between-donor, so a second region mostly tightens per-donor
  estimates, not the stage contrast.
* Region is a real covariate. DLPFC is later in the cortical AD sequence than MTG, so stage effects differ by
  region. Both regions of one donor must sit in the same fold (donor-level splitting), and region must enter the
  condition as a flag. Treating DLPFC cells as extra MTG cells would bias the stage law.
* Rule risk: the Multiregion Immune object (3.27 GB) and the CELLxGENE DLPFC object contain expression for
  sealed and forbidden donors too. Any extraction must select dev37 rows from `obs` and read only those
  rows, and it should be authorised as a protocol amendment first. It is not covered by the current file
  interfaces.

## 4. External cohorts

### 4.1 Summary table [V unless marked]

| Cohort / paper | Donors (controls) | Microglia / myeloid nuclei | Region, tech | Pathology in data | Access | Size |
|---|---|---|---|---|---|---|
| **ROSMAP, Mathys et al. Cell 2023** (doi 10.1016/j.cell.2023.08.039) | 427; 238 NIA-Reagan AD / **189 not** | 83,889 immune (microglia+CAM+T) of 2.3M | PFC; 10x 3' v3 | open file: sex, age bin, PMI, race, `Pathologic_diagnosis_of_AD` only. Braak/CERAD/NIA-Reagan/amyloid/tangles need DUC (syn3157322 or RADC) | Raw (syn52293417) **DUC**. Open files on personal.broadinstitute.org (see 4.2) | `PFC427_raw_data.h5ad` 91.57 GB; `Immune_cells.rds` 645 MB; metadata TSV 12.6 KB |
| **ROSMAP, Sun et al. Cell 2023** (doi 10.1016/j.cell.2023.08.037) | 424 snRNA (443 total incl. ATAC); PFC: nonAD 218 / earlyAD 132 / lateAD 73 | 152,459 immune nuclei in released matrix; PFC 66,594 over 423 donors (median 147/donor; 280 donors with at least 100) | PFC + 5 regions (subset of 48+); 10x v2 or v3 per cell | `ADdiag3types` = **k-means on 7 pathology+cognition phenotypes, "does not conform to standard diagnostic criteria"** | Raw DUC (syn52293417); open counts/meta on personal.broadinstitute.org | counts .rds 631.6 MB; meta .txt 24.5 MB |
| **ROSMAP, Green et al. Nature 2024** (doi 10.1038/s41586-024-07871-6) | 437 (preprint: 465) | 1.65M nuclei total; microglia count not verified | DLPFC (BA9) | ROSMAP variables via DUC | Synapse syn31512863 (as cited by Gabitto 2024 data availability) **DUC** | not checked |
| **PsychAD myeloid, Nat Genet 2026** (doi 10.1038/s41588-026-02716-6) | 1,469 snRNA donors (M 1,022 / H 295 / R 152); `Control`=1: 470; `AD`=1: 560 | **289,279** frozen in-silico-sorted myeloid nuclei; median 167/donor; 1,035 donors with at least 100 | DLPFC; 10x 3' v3 | open obs: diagnosis flags only (AD, Control, Aging, ...); **no Braak/CERAD/Thal**. Full PsychAD neuropathology is controlled (syn52160016) | **Open on CELLxGENE, CC BY 4.0** (dataset 09642770-...). Pathology: controlled AD Knowledge Portal | 3.01 GB H5AD (raw counts in `raw`) |
| PsychAD full atlas (preprint doi 10.1101/2024.10.31.24316513) | 1,494 donors, 6.3M nuclei | (myeloid subset above) | DLPFC; 10x 3' v3 | controlled | CELLxGENE cohort objects open (RADC 6.3 GB, MSSM 36.1 GB, Aging 12.4 GB, HBCC 14.2 GB); S3 "Controlled Access" (https://registry.opendata.aws/mssm-psych-ad/) | as listed |
| **Prater et al. Nat Aging 2023** (doi 10.1038/s43587-023-00424-y) | 22: 12 AD (ADNC 2-3) / **10 controls (ADNC 0-1)** | 127,371 microglia-like nuclei (PU.1 FANS-enriched) | DLPFC, UW ADRC | **ADNC reported** (same scale as SEA-AD, same UW/Kaiser source region) | Synapse syn51272688, **data use agreement** | not checked |
| Gazestani et al. Cell 2023 (doi 10.1016/j.cell.2023.08.005) | 52 living NPH biopsies: 25 no pathology / 19 Abeta+ / 8 Abeta+Tau+ | 892,828 nuclei total; microglia count not verified | frontal BA8/9 biopsy; 10x v3 | biopsy histology (Abeta, pTau); no Braak/Thal | access not verified | not checked |
| Olah et al. Nat Commun 2020 (doi 10.1038/s41467-020-19737-2) | 17 (autopsy aged + young surgical) | live sorted microglia scRNA-seq (median 833 cells/subject) | DLPFC/temporal; scRNA | limited | Synapse syn21438358; CELLxGENE collection fcb3d1c1 open | not checked |
| SEA-AD neurotypical references | 3 with DLPFC/MTG microglia (H18.30.002, H19.30.001, H19.30.002), all under 65, male | 295-655 per donor | MTG/DLPFC, same tech | ADNC = "Reference" (not staged) | open | - |
| Other open CELLxGENE AD sets (not microglia-focused) | Rexach 2024: 40 donors (BA4/insula/V1); "Layer 4 resilience" 2026: 46 donors (BA9/7/17) | all cells, includes microglia | 10x v2/v3 (+Drop-seq) | coarse disease labels | open CC BY 4.0 | 5.0 GB; 6.5 GB |

Overlap [V]: `Study_Participant_Overlap_table.csv` (personal.broadinstitute.org/cboix/ad427_data/MIT_ROSMAP_Studies_Overlap/)
shows **Sun 2023's 424 donors are all in Mathys 2023's 427**, so these are one cohort, not two. PsychAD source
"R" (RADC = ROSMAP, 152 donors) and Green 2024 also draw from ROSMAP. Donors must be de-duplicated through the
ROSMAP ID mapping (controlled).

### 4.2 Access and terms [V]

* **AD Knowledge Portal controlled data** (ROSMAP raw data and clinical/pathology metadata, PsychAD pathology,
  Green 2024, Prater 2023) require: a certified Synapse user, a 1-3 paragraph Intended Data Use statement,
  and a **DUC signed by an institutional signing official who is not the requester**. One institution per DUC.
  **Newer DUCs require that "use of AI tools must be described"**. Additional ROSMAP phenotypes go
  through RADC directly. Sources: https://help.adknowledgeportal.org/apd/Data-Use-Certificates.2623373330.html,
  https://help.adknowledgeportal.org/apd/requesting-access.
  This is an account action for Mikkel and needs an institutional signatory. Whether he has one is unknown
  (a possible blocker).
* **ROSMAP "open" files** (Mathys 2023 `PFC427_raw_data.h5ad`, `Immune_cells.rds`, Sun 2023 counts) can be
  downloaded without login, but both project pages also say "The data and human metadata must be accessed
  through Synapse with a Data Use Certificate (DUC)" (https://compbio.mit.edu/ad_aging_brain/,
  http://compbio.mit.edu/microglia_states/). No licence is stated for the open files. [I] Using them without a
  DUC is a terms grey zone. I recommend not training on them until the DUC is in place, or until a sister or
  Mikkel explicitly accepts the open-file terms.
* **CELLxGENE** datasets: CC BY 4.0, no account (PsychAD myeloid, SEA-AD DLPFC/MTG microglia, Rexach, Layer 4).
* **SEA-AD S3**: Allen Institute terms of use, no account.

### 4.3 Expected batch differences [I unless marked]

* Chemistry: SEA-AD 10x 3' v3.1 plus Multiome [V]. ROSMAP Mathys 2023 v3 [V]. Sun 2023 mixes v2 and v3 per
  cell [V]. PsychAD v3 [V]. Prater FANS-sorted (very different capture and depth) [V].
* Region: all large external cohorts are **prefrontal** (DLPFC/BA9/BA10), not MTG. SEA-AD's own DLPFC is
  the natural bridge: same donors in two regions let a region effect be estimated without cohort confounding.
* Pipeline: SEA-AD uses cellranger 6.1.2 with the 2020-A reference (36,601 genes) [V]. ROSMAP/PsychAD use
  other references and pipelines, so gene IDs must be intersected (Ensembl IDs preferred) and intronic-read
  handling may differ.
* Cell typing: SEA-AD supertypes are not available in external sets. Microglia must be re-identified with a
  classifier trained on train donors only (P2RY12/CX3CR1/TMEM119 versus CAM/T cells).
* Depth and library size: different per cohort, hence per-cohort library pools in the generator.

### 4.4 Mapping external pathology to the four ADNC levels [V for SEA-AD analysis; I for ROSMAP derivation]

ADNC (NIA-AA 2012) combines A = Thal phase, B = Braak band and C = CERAD. In SEA-AD every Not AD donor has
Thal 0, and High is always Thal 4-5 [V, `data_expansion/adnc_without_thal.py` over the 84 aged donors].
ROSMAP's open and commonly used variables are Braak (`braaksc`), CERAD (`ceradsc`), NIA-Reagan (`niareagansc`),
quantitative amyloid and tangle density. A Thal phase variable was **not found** in any source checked
(https://mindds.github.io/apoe-glia/ROSMAP-variables lists only braaksc/ceradsc). Treat it as absent until
the DUC codebook confirms otherwise.

How ambiguous is ADNC without Thal? Within SEA-AD itself, cross-tabulating ADNC against (Braak band, CERAD) [V]:

| (Braak band, CERAD) | ADNC in SEA-AD |
|---|---|
| B2 (III-IV), C0 (absent) | **6 Not AD, 5 Low, 1 Intermediate** |
| B1 (I-II), C0 | 1 Not AD, 1 Low |
| B3 (V-VI), C0 | 1 Not AD |
| B2, C1 | 3 Low, 6 Intermediate |
| B3, C2 | 16 High, 3 Intermediate |
| B3, C3 | 26 High, 2 Intermediate |

Assigning each cell its majority label recovers ADNC for about 69/84 donors (82%). However, **"Not AD"
predicted from Braak+CERAD is right for only about 9 of 16 donors (~56% precision)**, because amyloid-negative
and amyloid-positive donors with Braak III-IV and no neuritic plaques look identical on B and C. Consequences [I]:

* NIA-Reagan "no AD" (ROSMAP's 189) is **not** SEA-AD "Not AD". It includes Low and some Intermediate.
* SEA-AD "Not AD" includes Braak IV-V donors. A Braak-only control rule (Braak 0-II) would drop most SEA-AD
  controls and pick a different population.
* What can be derived exactly: Braak band (B0-B3) and CERAD (C0-C3) for all ROSMAP donors (with DUC), and
  "High" is plausible only when B3 and C2-3 (it needs Thal 4-5, unverifiable). A defensible option is to
  condition on **(B band, C score) for every cohort plus an "A known" mask**, with SEA-AD supplying A. ADNC
  then stays the SEA-AD-only evaluation label. Alternatively, derive a proxy A0 from ROSMAP quantitative
  amyloid load = 0 (plus CERAD absent), and report the proxy's disagreement rate on SEA-AD, where
  quantitative 6E10 % area exists (`sea-ad-quantitative-neuropathology` bucket) and the truth is known.
* Prater 2023 reports ADNC directly but only as AD (2-3) versus control (0-1). It is small (10 controls) but
  is the only external source verified to use the same staging scale.

## 5. Recommended plan, in priority order

### Today, no account action (each item is a protocol amendment the coordinator must approve before any fitting)

1. **Use what the 3-fold design leaves out (0.5 day).** After inner-fold evaluation is frozen, refit the chosen
   arm on all dev37 (4 Not AD rather than 2-3) for descriptive outputs. Propose that once the
   benchmark's fold-0 sealed evaluation is complete, the owner may release the 20 sealed donors for a
   57-donor training pool (6 Not AD). [I]
2. **Add SEA-AD DLPFC microglia for dev37 donors (1-2 days, about 3.3 GB transient disk).** Download the
   Multiregion Immune subclass object (3.27 GB, raw counts, same 36,601-gene axis). Extract only dev37 DFC
   microglia rows, selected from `obs` before touching `X`, into a cache under `runs/improve_20261109/`, then
   delete the download. Add a `region` flag to the condition, and keep both regions of a donor in the same
   inner fold. Evaluate MTG held-out donors exactly as now. DLPFC must beat MTG-only training on MTG held-out
   pseudobulk to count. The expected gain is per-donor precision, not more controls. [I]
3. **PsychAD open myeloid pilot without pathology (2-3 days).** The CELLxGENE object (3.0 GB, CC BY 4.0) gives
   141 aged (at least 65) controls with at least 100 myeloid nuclei, and 404 aged AD donors. Labels are only
   diagnostic flags, so use it at most for (a) pre-training or regularising the generator's cell-state manifold
   with a `cohort` flag and **no stage**, or (b) a sanity check of the Not AD versus High direction. Do not
   feed it as Not AD. [I]

### Needs Mikkel's account action (and an institutional signing official)

4. **AD Knowledge Portal DUC for ROSMAP (syn52293417 + syn3157322, then RADC for extended phenotypes) and
   PsychAD (syn52160016).** Effort for Mikkel: Synapse certification plus the IDU (with AI-tool use
   described) plus institutional signature. Turnaround is not stated by the portal. This unlocks Braak/CERAD/
   NIA-Reagan/amyloid for about 427 ROSMAP and about 1,469 PsychAD donors, and the ROSMAP ID mapping for
   de-duplication. Engineering after access: 3-5 days.
5. **Prater 2023 (syn51272688, data use agreement).** It adds 10 ADNC 0-1 controls with deep microglia
   coverage on the same staging scale. Small, but the cleanest label match.

### How an external cohort should enter the model [I]

* Condition vector: add a one-hot `cohort` (SEA-AD MTG, SEA-AD DLPFC, ROSMAP, PsychAD) and a `region` flag.
  Replace the single ADNC slot by harmonised (Braak band, CERAD, A-known mask, A level), so that each cohort
  supplies only the variables it truly measures. Keep generated outputs for evaluation conditioned on
  cohort = SEA-AD MTG.
* Gene axis: intersect on Ensembl ID across cohorts before HVG selection. Fit HVG, scaling and PCA on training
  donors of all cohorts (or MTG-train only, as a variant), and report how many of the forced AD genes survive.
* Library handling: per-cohort library-size pools (and per-chemistry pools for Sun v2/v3), sampled
  depth-matched per target donor as Interface 2 already specifies.
* Microglia identity: one train-only classifier applied identically to every cohort; FANS-sorted Prater nuclei
  kept as a separate cohort.

### Keeping held-out evaluation honest [I]

* The primary held-out metric stays **SEA-AD MTG inner-fold held-out donors**, computed exactly as now. An
  external cohort may only enter training. Its own donors are never pooled into the SEA-AD evaluation.
* Donor-level splits across regions and cohorts: a donor's MTG and DLPFC cells are always on the same side.
* For external-cohort evaluation, hold out whole donors by cohort-stratified folds, and report
  transfer (train SEA-AD, test ROSMAP Braak/CERAD contrasts) separately. Do not call ROSMAP results ADNC results.
* Freeze harmonisation rules (proxy A0, gene intersection, microglia classifier) before looking at any
  evaluation metric. Write them into PROTOCOL.md as an amendment.
* The 18 exposed audit donors stay out of all of this unless the owner releases them, and they are never
  described as independent confirmation.

## 6. What was run (reproducible)

* `data_expansion/donor_inventory.py` (local metadata only) writes `donor_inventory.json` and the tables
  in sections 1.2, 1.4 and 2.
* `data_expansion/adnc_without_thal.py` (local official metadata) gives the section 4.4 cross-tab.
* `data_expansion/peek_h5ad_obs.py` and `remote_donor_census.py` do HTTPS range reads of `obs` only, with a
  hard byte budget of at most 45 MB. Bytes actually read: PsychAD myeloid 5.9-6.0 MB, SEA-AD DLPFC 1.2-1.6 MB,
  SEA-AD MTG 2.0 MB. Outputs: `remote_cxg_*_census.json`. No `X`, `raw` or `layers` were read.
* Unauthenticated S3 list/HEAD on `sea-ad-single-cell-profiling`; CELLxGENE curation API collection JSON.
* Small public files downloaded to a scratch directory (not the repo), all under 50 MB: ROSMAP
  `individual_metadata_deidentified.tsv` (12.6 KB), `batch_mapping_deidentified.tsv`, `Cell_types.csv`, the overlap
  tables, Sun 2023 `...selected.deidentified.metadata.txt` (24.5 MB, metadata only), and the Mathys 2023 / Sun 2023
  PDFs (19.7 / 21.8 MB) from compbio.mit.edu. Europe PMC full text: Prater 2023, Olah 2020, Gabitto 2024,
  and the Gazestani preprint.

### Not verified / open

* Exact microglia counts per donor in the **2026-06-22** official DFC object (only the 2024 CELLxGENE copy
  was censused).
* Green 2024 microglia counts and access ID beyond Gabitto's citation; Gazestani 2023 data access route.
* Whether ROSMAP releases a Thal phase variable to DUC holders (not found in public sources).
* PsychAD `Control` definition and its relation to CERAD/Braak (pathology is controlled).
* Whether Mikkel has an institutional signing official for a DUC.
* No model was trained. Every "expected gain" above is an inference.
