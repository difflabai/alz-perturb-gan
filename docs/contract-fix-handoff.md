# Prepared-contract repair and Tau supervisor review — 2026-10-14

The bundle is now present and verified; it is not an access or user-input blocker. CPU-only byte/metadata validation passed for bundle `c50072314cdc12272da083def3322413ab01880a9363fd2d6fbad778ad35ad7b`, protocol `0e6287d226368408a8228f14da91d77be7fee7fcaa9993fa281f159e6232ce1e`, 9,415 candidate/9,420 query cells, >=33 cells per identity. No bundle, source data, feature packet, supplemental receipt or Tau script was rewritten.

## Implemented fixes

`load_study_bundle` compares the protocol to the grant before partition construction. Real bundles bind declared expression-gene IDs/symbols to existing raw H5ad metadata, perturbation names/stable IDs to that namespace, selected cell IDs to raw perturbation labels, controls to raw non-targeting IDs, and library order/groups to the hash-checked original split. Duplicate/empty partition IDs and missing library partitions are rejected. The original bundle SHA binds tensors; the builder already rehashed raw expression. Runtime metadata verification does not reread raw X or rehash 15 GB. It does not independently recompute every sampled expression value.

The Tau STRING adapter now requires and consumes the frozen row mapping and license supplement. It checks packet/matrix applicability, source order/stable IDs, source graph/alias hashes, protein graph-membership/missing status, reviewed source/asset/license metadata, and hashes of the existing primary license/page copies. All are saved in feature provenance. Neither original packet nor matrix was rewritten.

New required Tau adapter arguments:

```text
--string-row-mapping /ml2/altgan/features/addons/string_row_mapping.json
--string-row-mapping-sha256 044b2756e940d31519c85d81073a3dc17ac7f3115a8fc4e04b6a539f4b994a74
--string-license-receipt /ml2/altgan/features/addons/string_license_receipt.json
--string-license-receipt-sha256 1a7666849ddabcfedf7547b8eac844514eb8038be12a591743221b6e3711da86
```

Grant fields `string_row_mapping_sha256` and `string_license_receipt_sha256` must equal those command values. Existing `bundle_sha256`, when declared, must equal the invocation. Add `cuda_allocator_cap_gib: 4` and `--cuda-allocator-cap-gib 4` for the approved memory-cap arrangement; the CLI applies the grant-bound Torch allocator limit before model transfer. This does not cap non-Torch overhead or guarantee fit. Set `ALZ_LEASE_FDS` to comma-separated actual inherited lock FD numbers. The evaluator propagates them to its Git provenance children via `pass_fds`; it neither opens nor unlocks leases. Trainer setup is included in its cooperative wall timer. There is no objective change.

75 tests passed (one existing AnnData warning), CPU-only, 4.16s pytest/5.12s process time. New regressions cover grant protocol, raw gene order/stable IDs, source labels, duplicate/empty partitions, supplemental packet/row/license/hash faults, mocked allocator cap and inherited FDs. Receipts: `.review/contract-fix-20261014/pytest-receipt.json`, `prepared-contract-validation.json`. Torch2.6.0+cu124/NumPy2.4.6 versions are captured there.

## Saved diagnostic verification

Both saved legacy-run2 CPU evaluations have eight artifact hashes checked without mismatches. They share bundle/protocol/checkpoint, exploratory target hash and negatives, report zero optimizer calls and identical before/after module hashes, and are selection-ineligible (`selection_split=none`). Macro AUROC generated/oracle: 0.5903803/0.9729306; L2: 0.4941834/0.9979866. Micro head AUROC is 0.5388143/0.9747726. These compare the same queries but different generated versus independent-real candidate tensors. They do not establish a unique causal explanation, new training quality, inner independence or AD relevance. Old all-cell metrics are a different query construction; do not splice them into this cohort.

## Tau-owned supervisor review; do not launch this version

Reviewed `supervise_seed42.py` SHA256 `10b9f5718389d9e0603237cf8c50bdcb003e8b7d29e8ea69dfcad4693d817c8d`, template SHA256 `842a33595766d090168251d528945c05dde3c5331c2a3165ad0bd01e9cb5db04`. Correct: explicit October14 08-09 MDT/14-15 UTC, <=3600 constant, UUID, both nonblocking flocks, recorded inodes, direct arm `pass_fds`, own sessions, >=9GiB initial admission, runtime <4GiB floor, no explicit unlock, source hashes and serial G-sourced probe.

Blocking corrections for Tau:

1. Replace old harness commit pin with the repair commit, add both supplement flags/hashes and cap flag/grant field. Export `ALZ_LEASE_FDS` from actual FDs. Prefer passing feature metadata to B0 too so missing-feature strata are recorded on the matched baseline. Validate full frozen grant identifiers and aggregate3600, not merely step/per-arm values.
2. Remove `arm_end = now + budget + 30`; the entire arm process including import/load/evaluation and termination must stay <=600. Reserve termination/probe time within the assigned arm and aggregate deadline. Poll sleeps and blocking GPU/PS calls must be bounded by remaining time. The current 20s GPU queries, unbounded PS and `p.wait`, plus up to30s cleanup, do not enforce the claimed hard deadline. Cleanup triggered at deadline-15 can overrun the window.
3. Install SIGTERM/SIGINT handling that routes through owned-group cleanup, with an independent outer deadline/watchdog if necessary. Default SIGTERM does not run Python finally. Signal/timeout paths must preserve the FD references until owned descendants actually exit. Avoid revisiting cleared/recycled historical PGIDs when cleaning up; keep active owned groups or check their recorded process identity.
4. For real canonical locks, open existing files without O_CREAT; missing paths must abort. Preserve inodes. No explicit LOCK_UN. Pass both FDs through supervisor subprocesses after acquisition and through arm/provenance children.
5. Separate static preflight from actual acquisition. The current preflight requires finalized lease/capacity references before taking the locks, conflicting with issuance after actual acquisition. Acquire actual FDs, record live lease/inodes and fresh capacity, then materialize and hash the grant. Recheck window/expiry before every spawn. Never fill future lease references with proposal or stale receipts. Hard deadline must also respect actual lease/capacity expiry.
6. Retain remaining-time<120 abort/skip and no expansion. Record incomplete arms honestly. Verify mid-arm low-memory termination and supervisor signal cleanup with Tau's fake CPU tests, not GPU stress. Existing lowmem receipt tests a pre-arm stop; the normal receipt for this exact hash ends in G_frozen timeout rather than six successes.

The script was not edited or executed here, and no canonical lock or GPU was acquired. Tau is the sole supervisor writer/executor. The 08-09 window is conditional, not a recorded lease. No new user approval is needed for these already-approved local limits; actual lease/capacity evidence remains required.

## Metadata classification

Source SHA/generation, exact bundle hash, raw namespace/cell metadata and original split provide essential identity binding. GCS CRC32C exists in `adata_Training.gcs.json` (`/Z0row==`); MD5 is absent and is redundant to the recorded/rechecked source SHA. Per-control-cell batch labels are recoverable from raw metadata using the already saved control IDs; bundle batch composition is present. These do not require rewriting the bundle. Runtime versions are captured in validation receipts. The repository MIT license is code licensing, not Arc dataset terms. No Arc dataset-terms receipt was found in these existing project artifacts; retain an explicit provenance note and no redistribution claim. This is not a new gate on the already-authorized local research pilot.
