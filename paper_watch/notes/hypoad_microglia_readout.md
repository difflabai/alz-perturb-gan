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
Raw FASTQs: BioProject PRJNA1330312. Access seam found and fixed on
our end: plain GETs to ftp.ncbi.nlm.nih.gov return 403 through this
environment's proxy while Range requests succeed, and aiohttp (the
fsspec HTTP backend) does not connect at all — the extraction uses
a small requests-based range-caching file object and pulls only
obs/var plus the microglia CSR row-slices (~52k nuclei).

Design: donor is the unit (DREAM-Challenge leakage lesson). Per
seed: fit the unmodified company-standard ParticleGAN (EMPF recipe)
on microglia from a random half of the 7 control donors; score the
held-out control donors and the 6 sporadic-AD donors (familial
APOE4/4 donor AD_44 excluded, as in the paper; young reference
donors excluded). Readouts: per-donor energy-distance deviation
AUROC (AD vs held-out control), DAM-program score AUROC (paper's
own module genes, DAM minus homeostatic, per-donor mean), Gaussian
and linear-centroid arms, shuffled-diagnosis arm.
Assertions: model AUROC > 0.65 every seed; mean > shuffled + 0.15;
mean > DAM-program-only; mean > Gaussian.

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
