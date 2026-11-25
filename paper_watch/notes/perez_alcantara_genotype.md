# Perez-Alcantara genotype-conditioned microglia — iteration log

Find: Washer & Perez-Alcantara et al., Nature Genetics 2026
(briefed cycle 3); data access solved cycle 8 (2026-11-20).
Route: NEW DISTRIBUTION — expression conditioned on donor
genotype x stimulation state in pooled iPSC-derived microglia
(261 donor lines; states: Untreated / LPS / IFN).

Data: Zenodo concept doi 10.5281/zenodo.16684931, v3 doi
10.5281/zenodo.21471267 (2026-07-21), CC-BY 4.0, public.
Per-treatment h5ad + metadata zips (Untreated 5,727,317,598 B;
LPS 4,005,023,329 B; IFN 3,549,447,433 B); donor VCF
microglia_samples.GRCh38.filtered.with_X.vcf.gz (6,279,692,482 B);
nominal eQTL zip; CRISPR_raw_data.zip. Code:
github.com/TrynkaLab/Perez_Alcantara_pooled_iMGL.
Unverified until landing: whether the h5ad objects carry raw
counts or normalized values; donor demultiplexing fields in the
metadata CSVs.

Design (planned): state-split fits only (the paper's own lesson:
perturbation/genotype effects are context-signed — TREM2 in
microglia vs macrophages). Start Untreated: loader + sanity
(per-donor cell counts, donor assignment rate), then a small
conditional ParticleGAN fit (unmodified EMPF) conditioned on
donor genotype group at one AD locus vs shuffled-genotype arm +
mean/linear baseline; donor is the unit in every split.

## Iteration log
- Iteration 0 (data) — RUNNING 2026-11-20: Untreated zip
  downloading via resumable curl (log data/perez_download.log).
  LPS/IFN/VCF follow after the counts check on Untreated.
- Iteration 0 COMPLETE 2026-11-22 (evening check, cycle 10):
  untreated_h5ad_and_metadata.zip assembled at exact size
  5,727,317,598 B (all 16 parts verified).
- Iteration 1 (loader + sanity) — 2026-11-22, cycle 11:
  tests/test_perez_genotype.py + loaders in alztest/data.py.
  Metadata sanity PASSED (pytest, 2026-11-22): 862,274 cells;
  242 donors (of 261 lines; 19 absent or below reporting in the
  Untreated arm); per-donor cells min 1 / median 743 / max 67,622;
  195 donors >= 100 cells, 176 >= 200 — donor-unit conditioning is
  well powered. Demultiplexing: mean prob_max 0.9993,
  best_singlet agrees with donor_id on every row. Composition
  caveat: merged_clusters are 87% DAM-labelled (751,399), 10.9%
  Proliferating, 2.0% CAM, 39 IRM cells — the "untreated" cultured
  state is already DAM-dominated in this model system, so
  genotype-conditioned contrasts here are contrasts WITHIN a
  DAM-like state, not homeostatic microglia. 16 pools, 14
  sequencing batches (batch is a design factor in every split).
  Counts-vs-normalized check: PENDING — the h5ad inside the zip is
  31,365,866,293 B uncompressed; extraction was at ~26 GB when
  this was written and the test skips cleanly with the byte count
  until it completes. Iteration 2 (genotype-group conditional fit,
  Untreated only) waits on that check plus the VCF download.
- Counts check RESOLVED 2026-11-22 (pytest 2 passed): the
  extracted h5ad (862,274 x 36,497) holds RAW COUNTS
  (frac_integer 1.0 on a 2,000-row probe, max 2493) — iteration 2
  preprocessing log-normalizes from counts, as in the other
  conversions. obs carries donor_id / pool / treatment /
  sequencing_batch. Genotype VCF (6.28 GB) downloading under the
  resumable fetcher; iteration 2's locus grouping waits on it.
- 2026-11-23 (cycle 12): nominal_eQTL.zip LANDED byte-exact
  (835,253,235 B, assembled 2026-11-23 16:50 UTC, size-verified
  against the Zenodo record) — the eQTL layer is in hand for
  locus prioritization in iteration 2. The VCF fetcher was found
  DEAD again (VM-restart pattern; ~54% banked: 7 of 16 parts
  complete, 8 partial, part 15 absent) and was relaunched this
  cycle via scripts/parallel_download.py against the Zenodo API
  content URL (record 21471267, total 6,279,692,482 B verified
  via the API this cycle). Iteration 2 still waits on VCF
  assembly; on landing, the grouping uses the paper's own
  nominal-eQTL hits to pick the locus, not a fishing pass.
- 2026-11-23 (evening): VCF LANDED byte-exact (6,279,692,482 B,
  all 16 parts verified) — iteration 2 unblocked.
- 2026-11-25 (cycle 13): BIN1 dosages extracted from the VCF
  (scripts/perez_extract_bin1.py): variant 2_127135234_C_T,
  the strongest AD-locus eQTL in the untreated nominal table
  (p=7.05e-22, AF=0.392, slope +0.78); 261 samples, dosage
  0/1/2 = 90/134/37. Test written:
  tests/test_perez_genotype_fit.py (donor-split 80/20 stratified
  by dosage, cap 300 cells/donor, >=200 cells/donor, PCA-30,
  unmodified EMPF 300 steps; arms: shuffled-genotype,
  mean-shift, dosage-0-only).

## Iteration 2 — genotype-conditioned fit (BIN1, Untreated)
- Attempt 0 (2026-11-25): FAILED at the test's own gate —
  "only 14 genotyped donors with >=200 cells". Diagnosis
  (data handling only): VCF sample names are full HipSci ids
  (HPSI0114i-bezi_3); metadata/h5ad donor_id is the short
  suffix (bezi_3). Fix in alztest/data.py
  (perez_bin1_genotypes): suffix registered as an alias key
  after verifying 261 samples -> 261 unique suffixes, zero
  collisions, zero dosage conflicts; all 242 metadata donors
  match; 176 donors >= 200 cells (dosage 0/1/2 = 63/90/23).
- Attempts 1-2: prep died silently twice (empty log, process
  gone) — anndata's backed row slice over the 30 GB object,
  the cycle-12 NeuroKO-astro OOM seam (attempt 2 confirmed
  OOM-killed, exit 137, under the tracked driver). Fixed with
  the lean h5py CSR block reader (the NeuroKO pattern) in
  scripts/perez_fit_driver.py v2: same design, prep peak
  ~1.6 GB during loads vs death at ~3.1 GB on the anndata
  path; prep caches to npz per seed. One logged deviation:
  capped-cell draws iterate donors sorted (the test iterates
  a set — hash order, already cross-process nondeterministic;
  the donor split is identical).
- Seed 0 RESULT (train 41,573 / test 10,487 cells): model ED
  0.308 vs shuffled 0.318, mean-shift 0.128, dosage-0-only
  0.122; shift cosine model -0.078 vs shuffled -0.101,
  mean-shift +0.050. NOT PROVEN on seed 0: the generator is
  ~2.5x worse than sampling real dosage-0 cells — the
  marginal-fit failure shape of GSE289721/NeuroKO — and the
  BIN1 cis effect itself is small in 30-PC space (mean-shift
  barely beats dosage-0-only: 0.128 vs 0.122), expected for a
  single-gene cis-eQTL diluted across a 2,000-gene PCA.
  Seeds 1-2 running to complete the gauntlet; verdict and
  the iteration-3 candidate (a cis-target-subspace readout —
  score deviation where the eQTL actually acts, per the
  effect-size-stratification lesson of the Sci Adv benchmark,
  cycle 9) land on completion.
- Seed 1 RESULT (train 41,509 / test 10,551): model ED 0.374
  vs shuffled 0.394, mean-shift 0.176, dosage-0-only 0.127;
  cosine -0.063. Seed 2 RESULT (train 41,411 / test 10,649):
  model ED 0.397 vs shuffled 0.428, mean-shift 0.233,
  dosage-0-only 0.226; cosine +0.051.
- ITERATION 2 VERDICT (3-seed gauntlet): NOT PROVEN. The
  conditional model beats its shuffled-genotype twin on ED
  on every seed (0.308<0.318, 0.374<0.394, 0.397<0.428) —
  conditioning carries a small real signal — but loses
  decisively to the no-fit arms (dosage-0-only 0.122/0.127/
  0.226; mean-shift 0.128/0.176/0.233) and no arm recovers
  the shift direction (cosines ~0 for model AND mean-shift).
  Two separable findings: (1) marginal-fit failure, third
  dataset class (after GSE289721 + NeuroKO): at the company-
  standard 300-step budget the generator's samples sit
  farther from held-out real cells than real control cells
  do — handed to the working-GANs effort with numbers,
  not fixed here. (2) The BIN1 cis-eQTL effect is genuinely
  small in whole-transcriptome 30-PC space (mean-shift ~
  control-only on seeds 0-1): a single-gene cis effect is
  diluted by construction in this readout.
- Iteration 3 (named, one change — readout only): score the
  same fits on a cis-target subspace (BIN1 + its untreated
  nominal-eQTL target neighborhood) instead of the full
  30-PC space; arms unchanged. If the model cannot beat
  control-only even where the eQTL acts, the genotype
  conversion closes as an honest negative at iteration 3-4.
