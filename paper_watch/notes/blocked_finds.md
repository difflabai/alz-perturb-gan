# Blocked paper-watch finds

Three paper-watch conversions were opened as stacked pull requests on
2026-11-18 and closed without merging, as the owner decided on
2026-11-18. None of them ran a fit, so none produced a result. This
note keeps a record of each access blocker, what was rechecked, and
what a revival would need. The test code stays on its unmerged branch
for reference. It is not a template to merge as written.

| Find | Closed PR | Design branch | Blocker | Recheck after |
|---|---|---|---|---|
| NeuroKO Perturb-seq | [#10](https://github.com/difflabai/alz-perturb-gan/pull/10) | `paper-watch/neuroko` | Annotated objects have no resolvable public location | 2026-12-20 |
| panAD MONET subtypes | [#12](https://github.com/difflabai/alz-perturb-gan/pull/12) | `paper-watch/panad` | No open release of the matrix or donor subtype labels | 2026-12-20 |
| GNPC plasma proteomics | [#14](https://github.com/difflabai/alz-perturb-gan/pull/14) | `paper-watch/gnpc` | Consortium access only, and outside the microglia scope | 2026-12-20 |

HypoAD ([#17](https://github.com/difflabai/alz-perturb-gan/pull/17))
is still open and is not covered here.

The dataset loaders for these three finds were removed from
`alztest/data.py` on main. Their access facts are recorded below. A
revival should add a new loader with its own access check.

## NeuroKO

- **Find.** Dominicus et al., bioRxiv 2026.10.13.758778 (DOI prefix
  10.64898). It is a pooled Perturb-seq screen of Alzheimer's risk genes
  in iPSC-derived neurons and astrocytes, not microglia.
- **Blocker.** The data availability statement gives a Zenodo DOI,
  10.5281/zenodo.2326038861. The identifier does not resolve. Raw reads
  exist at ENA ERP134820 as unprocessed FASTQ. Processing a pooled
  screen from raw reads is a separate project.
- **Rechecked 2026-11-20.**
  - `https://doi.org/10.5281/zenodo.2326038861` returns HTTP 404.
  - The Zenodo API returns "The persistent identifier does not exist."
  - A Zenodo search for the preprint title returns no records.
  - The bioRxiv API lists v1 (2026-10-14) and v2 (2026-10-16). The data
    statement of v2 was not re-read.
  - The README of `Lotfollahi-lab/NeuroID-reproducibility` (last push
    2026-10-21) has no data link.
- **Defects in the closed design** (`tests/test_neuroko_perturbation.py`
  on the branch).
  - Lines 82–84 fit one PCA on the training cells and a second PCA on
    the held-out cells, then standardize each. Generated samples and
    held-out cells are therefore in different coordinates, and the
    energy distances and shift cosines would not be comparable.
  - The docstring requires at least 100 held-out cells per knockout,
    but the code skips only knockouts with fewer than 40.
  - The PR description lists an expression-matched control arm that is
    not in the code.
- **Revival needs.**
  - An annotated object from a resolvable, versioned source.
  - One transform (gene selection, PCA, scaling) fitted on training
    cells only and applied unchanged to held-out cells.
  - Holdouts by guide or replicate, not by random cell.
  - The held-out cell threshold that the docstring states.
  - A note on the cell-type mismatch with the microglia model.

## panAD

- **Find.** panAD integrated atlas, bioRxiv 2026.09.03.747935 (DOI
  prefix 10.64898). The MONET framework assigns donor-level molecular
  subtypes.
- **Blocker.** No download of the integrated matrix or of the MONET
  donor subtype labels was located. The authors' atlas interface is a
  browser, not a bulk download. The full text is not archived in PMC.
- **Rechecked 2026-11-20.**
  - The bioRxiv API lists only v1 (2026-09-04), with no journal
    publication recorded.
  - No collection in the CELLxGENE curation API mentions panAD or MONET.
- **Defects in the closed design** (`tests/test_panad_subtype_contrast.py`
  on the branch).
  - Line 53 drops any (subtype, diagnosis) condition with fewer than 300
    cells. Line 79 then assumes control is index `2i` and AD is index
    `2i+1`. If one arm of a subtype is dropped, every later pair
    contrasts AD against another subtype's controls.
  - Line 47 labels diagnosis with
    `str.contains("AD|Alzheimer", case=False)`. That pattern also
    matches labels such as "non-AD" and "Head trauma".
  - Line 72 fits the PCA on training and held-out cells together.
  - The split is per cell within condition, although the subtype is a
    donor-level label. The same donors appear in both splits.
- **Revival needs.**
  - A versioned export of the matrix and the subtype labels.
  - Splits by donor.
  - A transform fitted on training donors only, as in
    `alztest/psychad.py` `project_profiles`.
  - AD/control pairs keyed by subtype name, and skipped when an arm is
    missing.
  - Diagnosis taken from an explicit table of exact source labels.

## GNPC

- **Find.** Longitudinal plasma proteomics, medRxiv 2026.07.01.26356385
  (DOI prefix 10.64898). It reports a progression-linked protein
  signature from Global Neurodegeneration Proteomics Consortium (GNPC)
  SomaScan data.
- **Blocker.** The GNPC SomaScan v4.1 resource is distributed under
  consortium access terms. There is no public download. The design
  would also model plasma protein deltas, not microglial expression, so
  it tests a different modality from the current model.
- **Rechecked 2026-11-20.** The medRxiv API lists only v1 (2026-07-10),
  with no journal publication recorded. No public release was found.
- **Defects in the closed design** (`tests/test_gnpc_progression.py`
  on the branch). Line 72 fits the PCA on the deltas of all
  participants, test participants included, before the model is fitted
  on the training rows.
- **Revival needs.**
  - Approved consortium access.
  - A study question about plasma progression, scoped separately from
    the microglia model.
  - Splits by participant.
  - Preprocessing fitted on training participants only.

## Recheck procedure

At each recheck, run the commands below and update the dates in this
note. Reopen a find only in a new pull request against `main` that fixes
the defects listed for it.

```sh
curl -sI https://doi.org/10.5281/zenodo.2326038861 | head -1
curl -s https://api.biorxiv.org/details/biorxiv/10.64898/2026.10.13.758778
curl -s https://api.biorxiv.org/details/biorxiv/10.64898/2026.09.03.747935
curl -s https://api.biorxiv.org/details/medrxiv/10.64898/2026.07.01.26356385
```
