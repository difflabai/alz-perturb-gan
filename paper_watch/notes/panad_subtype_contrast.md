# panAD subtype contrast — iteration log

Find: bioRxiv 2026.09.03.747935. Route: new target.
Test: `tests/test_panad_subtype_contrast.py`.

## Iteration 0 — 2026-11-17 (access verification, no fit run)
- Test implemented against the expected obs schema (donor MONET subtype,
  diagnosis, cell type; microglia stratum; shuffled-subtype arm).
- Access: the preprint's license bars PMC archiving, so no full text there;
  no integrated-object download or MONET label table located on the preprint
  server, CELLxGENE, or the authors' channels checked 2026-11-17. The atlas
  is served through the authors' "Stellar Atlas" interface, which is an
  explorer, not a bulk download.
- Status: BLOCKED on data release. Re-check each cycle; the test runs
  unchanged once data/panad.h5ad with subtype labels exists.
