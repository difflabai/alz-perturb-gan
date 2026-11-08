# Build and inspect the public research report

The report is a static site in `docs/`, published from `main` by GitHub Pages. It has no build-time GPU dependency, external runtime service, remote JavaScript, or tracking. The committed evidence is anonymous aggregate output, not raw cells or donor-level metadata.

## Rebuild from committed results

Python 3.10+ is required; Python 3.11.17 with the pinned dependencies was used for this report.

```bash
python3 -m pip install -r scripts/report/requirements.txt
python3 scripts/report/build_report.py
python3 scripts/report/build_report.py --check
python3 -m http.server 8765 --directory docs
```

Open `http://localhost:8765`. Modify `report.template.html` rather than the generated `docs/index.html`. Styles and explorer behavior live in `docs/report.css` and `docs/report.js`.

The builder reads nine committed analysis JSON files and `docs/evidence/stage-profiles.json`. It exports all 2,000 gene results to CSV/JSON, embeds the same data in the page, and renders four scientific figures to SVG and PNG with Matplotlib. It checks every gene's four-stage profile against its independently aggregated endpoint contrast, screen counts, hashes, duplicate HTML IDs, internal anchors, local asset paths, and public exports for machine paths/known donor-ID prefixes. Exact figure bytes may vary with plotting dependencies/fonts; the manifest records the artifacts produced by the current build.

## Verify the explorer

Node 24.15+ (or another Node version supported by the pinned jsdom dependency):

```bash
npm ci --prefix scripts/report
npm test --prefix scripts/report
node --check docs/report.js
```

The DOM-engine verification exercises the actual exported data, evidence filters, gene selection, accessible profile descriptions, numerical values, absent genes, sorting, and pagination. It is not a visual browser test. No browser preview was available in the publishing session; scientific chart PNGs were visually inspected and the HTML/assets were served locally for HTTP checks. Responsive CSS is included, but desktop/mobile rendering still merits a browser review.

## Original stage-profile export

`export_stage_profiles.py` is only needed to reconstruct the committed anonymous profiles from the original completed inference artifacts:

```bash
python3 scripts/report/export_stage_profiles.py --state artifacts/primamente-t005-gene-inference-recovery-v1-20261108
```

Those retained artifacts and raw datasets are not bundled with the public repository. The export contains per-gene real stage means and per-model stage means, never individual donor records or raw generated cells. Source hashes and all endpoint reconstructions are checked by the report build. The 2026-11-08 study snapshot is rooted at research commit `89fa5ea`; the public repository also preserves the earlier experiment history.
