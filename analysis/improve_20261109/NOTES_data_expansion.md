# NOTES_data_expansion

## data_expansion (inventory) - interpretation choices

* "Do not download files larger than 50 MB": I did not download any file of that size. For remote h5ad files
  (CELLxGENE PsychAD myeloid 3.0 GB, SEA-AD DLPFC microglia 0.7 GB, SEA-AD MTG microglia 0.5 GB) I made
  HTTPS range reads of the `obs` group only. A hard byte budget was enforced in code
  (`data_expansion/peek_h5ad_obs.py`), and at most 6.0 MB were transferred per file. `X`/`raw`/`layers` were
  never read, so no expression data was touched for any donor. I judged this the most faithful reading of
  the rule, since it serves the donor-metadata inventory the task requires.
* Donor-level metadata for SEA-AD donors outside the 37 (the sealed 20, the 25 others and the neurotypical
  references) was read, which the protocol allows only for this task. Per-cell nuclei counts per donor are
  treated as metadata (they come from `obs`/candidate lists, not from count matrices).
* Small public metadata files and two paper PDFs (all under 50 MB) were saved to the session scratch
  directory, not to the repo. The repo holds only scripts and small JSON census outputs under
  `data_expansion/`.
* Concern for the coordinator: every recommended SEA-AD DLPFC source object (Multiregion Immune 3.27 GB,
  CELLxGENE DLPFC microglia) also contains expression for the sealed and forbidden donors. Extracting dev37
  rows from them needs an explicit protocol amendment that defines an obs-first row filter. I did not do this.
* Concern for the coordinator: the ROSMAP "open" files on personal.broadinstitute.org carry project-page wording
  that the data must be accessed via Synapse with a DUC. Using them before a DUC is a terms question for
  the owner, not an engineering one.
