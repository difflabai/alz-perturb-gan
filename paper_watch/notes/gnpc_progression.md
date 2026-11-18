# GNPC progression deltas — iteration log

Find: medRxiv 2026.07.01.26356385. Route: new distribution + new target.
Test: `tests/test_gnpc_progression.py`.

## Iteration 0 — 2026-11-17 (access verification, no fit run)
- Test implemented against the expected long-format export schema
  (participant_id, visit, diagnosis, APT_* aptamer columns; per-person
  delta = last − baseline; converter vs stable-CN conditions; mean-shift
  and shuffled-status arms).
- Access: the GNPC SomaScan v4.1 resource is distributed under consortium
  access terms; no public download exists (checked 2026-11-17).
- Status: BLOCKED on consortium access. Re-check each cycle.
