# HypoAD microglia readout — iteration log

Find: Yu et al., Nature 2026 (doi 10.1038/s41586-026-11305-w),
published 2026-11-18 — the ALZ paper watch's cycle-4 top find.
Route: NEW DISTRIBUTION + READOUT — hypothalamic microglia, a
region/compartment no prior cycle has touched (SEA-AD covers ten
neo-/allocortical regions; the hypothalamus is not among them).

Data: GEO GSE342635 (public on 2026-10-05). Processed series object
GSE342635_HypoAD.h5ad is 51,107,823,722 bytes for 614,403 nuclei;
GEO carries NO per-sample supplementary files, and the only
per-cell-type subset is a neuron-only object (9,322,705,096 bytes).
Raw FASTQs: BioProject PRJNA1330312. Access on 2026-11-18 (iteration
0, below): plain GETs to ftp.ncbi.nlm.nih.gov returned 403 through that
run's proxy while Range requests succeeded, and aiohttp (the fsspec
HTTP backend) did not connect. Access recheck on 2026-11-20 from
pop-os: `curl -I` returns HTTP 200, a range read returns 206 with
`Content-Range: bytes 0-7/51107823722`, and the first bytes are the
HDF5 signature. Access is therefore not the blocker from this host;
the remaining cost is the transfer of the 51,107,823,722-byte object
(CPU/network only, roughly 1-6 h depending on throughput).

Design: donor is the unit (DREAM-Challenge leakage lesson). Per
seed: fit the unmodified company-standard ParticleGAN (EMPF recipe)
on microglia from a random half of the 7 control donors; score the
held-out control donors and the 6 sporadic-AD donors. The familial
APOE4/4 donor AD_44 (as in the paper) and the young reference donors
are removed before grouping (`alztest.hypoad.EXCLUDE_DONORS`; a
missing AD_44 is an error, not a silent pass). Gene selection (top
1,500 by variance plus the module genes), module-gene z-scoring, PCA
and component standardization are fitted on the fit-control cells
only and applied unchanged to all cells. Readouts: per-donor
energy-distance deviation AUROC (AD vs held-out control), DAM-program
score AUROC (paper's own module genes, DAM minus homeostatic,
per-donor mean), Gaussian and linear-centroid arms,
shuffled-diagnosis arm. Every model AUROC is reported with its donor
counts (fit controls, held-out controls, AD), the exact
label-permutation p over scored donors, and a stratified
donor-bootstrap 95% interval.
Assertions: model permutation p <= 0.05 and bootstrap lower bound
> 0.5 at every seed; mean > shuffled + 0.15; mean > DAM-program-only;
mean > Gaussian. With 3 fit controls, 4 held-out controls and 6 AD
donors there are C(10,6) = 210 label assignments, so AUROC moves in
steps of 1/24 and the smallest attainable p is 1/210. The earlier
"AUROC > 0.65 every seed" rule was dropped: at this donor count it
is not a calibrated bar.

Known structural limit, stated before any run: 13 usable donors is
the whole cohort. Every prior closed loop in this watch (SEA-AD
ordering, LL burden, resilience) was ultimately donor-bound; this
one starts there. A weak-but-stable signal is the realistic best
case at toy scale, and a donor-bound partial is an acceptable,
honest outcome — what would change the verdict is a larger
hypothalamic cohort, which does not exist yet.

## Iteration log

- Iteration 0 (access) — RESULT: data verified open, extraction
  seam found, bulk transfer is the binding constraint.
  GEO series verified live (public 2026-10-05; supplementary files
  enumerated; sizes probed by Content-Range: full object
  51,107,823,722 B, neuron-only 9,322,705,096 B, RDS 11.3 GB).
  Three extraction routes tried on 2026-11-18:
  (1) fsspec/aiohttp HTTP random access — aiohttp does not connect
  through this environment's proxy at all (connection timeout);
  (2) requests-based range-caching file object (scripts/rangefile.py)
  + anndata backed read — opened the file and began reading, then
  died on a proxy-dropped block while reading layers/counts;
  retries patched in, but an h5py metadata walk of the 51 GB object
  over range requests proved too slow to complete in the run
  (>200 s without finishing the X group header walk);
  (3) chunked ranged download (plain GETs 403 on this proxy; range
  GETs succeed) — sustained ~3.5 MB/s aggregate even with 8 parallel
  streams, so the 9.3 GB neuron object (the structure-learning
  subset; same annotation table) is a ~45-minute pull and the full
  object ~4 hours at that rate. The neuron download was left
  running in the background of the watch's local repo
  (scripts/parallel_download.py, resumable by part files).
  Status: test implemented and skipping cleanly with the precise
  DataUnavailable reason; iteration 1 (first fit) waits on the
  bulk transfer landing. This is an access-format finding, not a
  data-availability blocker: the data are public, the packaging
  (one 51 GB object, no per-sample files) is what gates toy-scale
  use. Next cycle: check the neuron download, learn the obs map
  from it, extract microglia row-slices from the full object with
  the same ranged tooling (or finish the full download), run
  iteration 1.
- Review fixes (2026-11-20, before any fit). (1) AD_44 was excluded
  only in the text: the code removed young donors and left AD_44 in
  the AD group. It is now removed before grouping. (2) Gene variance,
  PCA and the PCA standardization were computed on all cells,
  including the held-out control and AD cells being scored; they are
  now fitted on fit-control cells only (a unit test perturbs held-out
  cells and checks the fitted basis is unchanged). (3)
  scripts/extract_hypoad_microglia.py printed obs and wrote an obs
  parquet; it now reads a local copy of the GEO object and writes
  data/hypoad_microglia.h5ad with the microglia rows, raw counts and
  donor/condition/cell_type, with paths resolved from the script
  location. (4) parallel_download.py now runs curl with --fail and
  --retry and deletes failed parts, so an HTTP error body cannot be
  kept as data; RangeFile raises a clear error without a 206 and a
  Content-Range total. (5) The 2026-11-18 neuron transfer is not
  assumed to exist: the 2026-11-20 review found no partial HypoAD
  file on /ml2, /mnt/nvme, /home or /tmp.
  Status: no fit has been run. Next step, separate from this code
  change: download GSE342635_HypoAD.h5ad into a gitignored data
  directory with more than 60 GB free, extract microglia, run
  iteration 1 and record the numbers here.
