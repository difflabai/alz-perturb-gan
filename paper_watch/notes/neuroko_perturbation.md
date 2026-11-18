# NeuroKO perturbation test — iteration log

Find: Dominicus et al., bioRxiv 2026.10.13.758778. Route: new distribution + new target.
Test: `tests/test_neuroko_perturbation.py`. Arms: conditional ParticleGAN vs
shuffled-KO ParticleGAN vs mean-shift moment baseline vs control-only baseline.
Metric: energy distance to held-out KO cells + shift-direction cosine, 3 seeds.

## Iteration 0 — 2026-11-17 (access verification, no fit run)
- Test implemented end to end (loader, KO/cell-type introspection, arms, assertions).
- Data access checked four ways: (1) Zenodo API search by title/authors — no record;
  (2) the DOI printed in the preprint's data availability (10.5281/zenodo.2326038861)
  does not resolve — the identifier is corrupt as printed (11-digit suffix);
  (3) Europe PMC record — no data links, no full-text XML; (4) authors' GitHub
  (Lotfollahi-lab/NeuroID-reproducibility) — README carries no data link.
- Raw sequencing exists at ENA ERP134820 (unprocessed FASTQ) — processing a pooled
  Perturb-seq screen from raw reads is a project, not a toy test; not attempted.
- Status: BLOCKED on the annotated object. Next iteration runs unchanged the day
  data/neuroko_annotated.h5ad (or NEUROKO_H5AD) exists. Re-check the Zenodo record
  each watch cycle.
