# Calibration-fix plan v4 (wording fixes to v3 `d5f772db…`, accepted scientifically by Ember) — plan only, no execution approval

Base: branch `calibration-fix` @ `a624aeec`. v2 and v3 are unchanged. Hold: no runs, no receipt edits, read-only inspection, everything scoped to `/ml2/altgan`.
Standing result for the 42/43/44 cohort (`cal_selection_frozen.json` `538e63ae…`): **"selection criterion unmet; evidence inconclusive."**

## 1. Prospective amendments (not clarifications)
- **A1 (prospective amendment):** calibration status no longer vetoes selection. It is reported only.
- **A2 (prospective amendment):** G_frozen is a diagnostic probe. It is reported but never selectable.
- Both apply only to cohorts run after this plan is approved. They are not applied retroactively.
- The previous result stands as recorded. Under the old rule (calibration veto, G_frozen eligible) nothing was selected. Every pooled inner interval crosses zero vs B0 and vs shuffle: G, H, GH and G_frozen lower bounds are −0.021 to −0.094. That alone blocks selection under either rule, so the amendments cannot change the past verdict.

## 2. Ranks, temperature and the FIT change
- All rank metrics are computed from **raw head logits only** (`gene_evaluation.py:206` passes logits as `scores`).
- A single positive temperature applied to raw logits leaves every rank unchanged, both within rows and pooled cross-row (micro) AUROC on logits.
- Micro-AUROC changes only when:
  - probabilities are normalized row-wise (softmax), which shifts each row differently, or
  - rows scored with **different** temperatures (e.g. per-fold T in cross-fitting) are pooled.
- Finite-precision ties can also appear (seen at a624aeec: float32 softmax at T≈1e4).
- So micro-AUROC is always reported on raw logits, never on calibrated probabilities.
- **Precision on what is new:** a624aeec already excluded the 16 calibration identities from G, D, J, the gate and the learned prior at runtime (`gene_training.py:76-77`). The only new changes in v4 are:
  - the **explicit protocol encoding** (FIT 80 + `calibration` 16, new protocol and bundle hashes)
  - **FIT-80 feature standardization**: at a624aeec, `FeatureTable` still standardized on all 96 FIT identities.
- The standardization change is a real input change for the feature arms. Results are not comparable cell-for-cell with the a624aeec cohort, and comparisons are labelled as such.
- The existing acceptance test is kept: during training, the indices fed to the prior, G, D, J and the gate never contain a calibration identity.
- **Bundle v2 keeps the existing data intact:** cell IDs, the candidate/query pools (`partition` with seed 43), control, library order and gene order. Only the protocol's group assignment changes.
- Acceptance: per-identity cell-ID lists, pool IDs and `gene_ids` are byte-equal between `study_bundle.pt` and `study_bundle_v2.pt`, and the protocol diff touches only `fit` and `calibration`.

## 3. Calibration evaluation: what 4-fold OOF does and doesn't show
- 4-fold out-of-fold (OOF) evaluation (seed 1729, 4×4) **evaluates the procedure** of fitting T on 12 identities. It does **not** independently validate the all-16 temperature that is then applied to inner and exploratory. The plan states this.
- The unit of evidence is **16 dependent targets**, not 2,400 independent pairs. Only descriptive uncertainty is reported: per-fold values and the range across folds. No p-values, and no pair-level CIs.
- **Report, for every arm and seed:**
  - per-fold T, plus the all-16 T
  - cap hits at both the **lower** bound (0.05) and the **upper** bound (1e4)
  - OOF and in-sample Brier, ECE and NLL
  - base Brier (1/150)(1−1/150) = 0.0066222222
- **Informative only if all of these hold:**
  - no fold hits either cap
  - not near-uniform
  - OOF Brier ≤ base × 0.995 = **0.0065891111**
  - OOF NLL ≤ log 150 − 0.01
- **Near-uniform is defined directly, not via the cap**, since uniform logits needn't hit 1e4. It applies when the median over target rows of `(max − min)/T` (logit spread after temperature) is < 0.05, i.e. calibrated row probabilities stay within about ±5% of 1/150.
- **Wording:** anything else "did not meet prespecified informativeness thresholds" (never "no probability information"). No ECE or Brier claim is made for it.

## 4. Exploratory sealing and selection binding
- **The a624aeec exploratory-30 numbers have already been seen.** They stay non-confirmatory permanently, and no future analysis may treat any exploratory result as confirmation.
- **Seal everything exploratory:** summary metrics, the `pairs/matrices.pt` exploratory blocks and the exploratory rows of `pairs/scores.jsonl`.
  - They go to `sealed/exploratory.{json,pt,jsonl}`. Unsealed outputs carry only the sealed files' sha256.
  - Code: `gene_evaluation.py:198-206` and `_write_groups` `:244-262`.
- **The selection file binds:** run IDs, supervisor receipt sha, protocol sha, bundle sha, rule-text sha, scoring mode (`raw_logit_ranks`), and per-arm calibrator and calibration-split sha.
- `gene_analysis.py explore` refuses to run unless every bound hash matches the current artifacts and the selection sha is supplied.
- **Stated limit:** the writable ledger (`ledger.jsonl`) and file mtimes are **not independent proof of chronology**. They are an audit aid only. Real chronology needs an external timestamp, e.g. posting the selection sha to the PR before running `explore`, and that needs separate approval.

## 5. One global monotonic deadline (proposal pending approval)
- **Arithmetic:** the last cohort took 1333.5 + 1288.2 + 1308.1 = 3929.8 s. 3930 × 1.15 = **4519.5 s**. A 4500 s cap is ≈ **14.5%** headroom (4500/3930 = 1.145).
- 4500 s exceeds the approved 3600 s per invocation, so the budget **remains a proposal pending approval.** Without approval, the cap stays at 3600 s and the cohort is expected to be incomplete.
- **One monotonic deadline:** `deadline = t0 + min(cap, lease_expiry − now, window_end − now)` is set once at start and **never reset** between seeds or arms.
- It covers everything: bundle and feature loading, training, calibration, evaluation, checkpoint saving, GPU and lock probes, the post-run probe and cleanup.
- A **termination reserve** of 60 s is held back from the arms. Each arm's slot is min(600, deadline − now − reserve), and an arm is skipped if under 120 s.
- **Cleanup overrun:** if teardown runs past the 60 s reserve, the run is reported as **incomplete/overrun** with the actual end time. Protection is not dropped early: the lock FDs stay held while any of our children are alive. The post-exit probes (§6) share the **same** deadline; there is no separate budget.
- **On overrun:** the remaining arms and seeds are marked `skipped_global_budget`. **An incomplete cohort stays incomplete:** no rerun or extension, and the §7 rule is reported as unmet (cohort incomplete).

## 6. Post-teardown lock and state evidence
- **Not read-only, stated plainly:** the post-run `flock(LOCK_EX|LOCK_NB)` **briefly acquires** the slot. It is a momentary acquisition followed by close. The plan says this, and it needs the same approval as lock use.
- **Verify the opened file, not just the path:** after `open`, `fstat` the handle and check its device:inode equals the recorded value (inode 4006/4007, device 47) before flocking.
- **Busy later ≠ ours:** a busy result after our supervisor exited is given exactly one of three labels, never retried:
  - **`ours`:** the `/proc/locks` holder matches the device:inode, and its PID plus `/proc/<pid>/stat` start time match a recorded supervisor or child identity. This is our failure.
  - **`other_owner`:** recorded only with positive evidence: a live holder PID and start time that is not one of ours.
  - **`unknown_owner`:** anything else, e.g. the holder exited or `/proc/locks` was unreadable. Failing to match us does not prove another owner.
- **Bounds:** every probe (`nvidia-smi`, `/proc` scans, flock) is time-bounded (≤ 5 s each) and fits inside the global deadline (§5).
- **Recorded state is scoped to:**
  - GPU UUIDs with their compute-app list and free/total memory
  - our owned pgids (empty)
  - the lock device:inode values
  - llama-server PID and start time
- No other process detail is recorded.
- **Limit stated:** an unchanged llama-server PID and start time shows the process wasn't restarted. It **does not prove it stayed responsive.** No health probe is made, since that would be outside our scope.

## 7. Path containment
- Raw prefix checks (`[[ $D == /ml2/altgan/* ]]`, `startswith`) allow `..` traversal and symlink escape, so they are not used.
- **Runtime helper `alz_perturb_gan/paths.py: scoped(p)`:**
  - rejects empty or None input
  - `Path(p).resolve(strict=…)` canonicalizes
  - containment is by path components: `ROOT.resolve() in [r, *r.parents]`
  - the exact allowlist is the 2 lock files, `/proc/locks`, `/proc/<pid>/{stat,status}` and nvidia-smi
  - it raises on anything else
- `O_NOFOLLOW` only protects the **final** component. Directory components are canonicalized via `scoped()` first, then opened with `O_NOFOLLOW`.
- **New output files:** `realpath -e` / `resolve(strict=True)` can't resolve a file that doesn't exist yet. So:
  - canonicalize and contain-check the **existing parent directory**
  - validate the file name as one component (no `/`, not `.`/`..`, non-empty)
  - where it matters (receipts, sealed files), anchor on a directory handle: open the parent with `O_DIRECTORY|O_NOFOLLOW`, then `openat(dirfd, name, O_CREAT|O_EXCL|O_NOFOLLOW)`.
- **Directory assumptions and race limits:** we assume that the directories under `/ml2/altgan/studies/…` and `/ml2/altgan/.cache/…` are owned by `mikkel`, are not writable by other users, and that no one else renames them. A path-based check outside this handle anchoring has a time-of-check/time-of-use window; we accept it only for non-receipt outputs.
- **Shell:** `set -euo pipefail`, plus a glob-array check: `shopt -s nullglob; a=(…); (( ${#a[@]} )) || exit 3`. With nullglob an empty array would otherwise turn into a bare `ls` of the current directory.
  - Every path goes through `realpath -e --` and a component check by a Python one-liner calling `scoped`.
  - Commands use `--` before path arguments.
- **Enforcement is both runtime and lint:**
  - runtime: `scoped()` on every open, glob, listdir and subprocess path argument
  - lint: `tests/test_path_hygiene.py` flags out-of-scope literals, unguarded `ls`/`cd`/`rm`, and missing `--`
  - unit tests cover `..`, symlink escape, `~`, empty input and an empty glob

## Pre-registered success rule (prospective; applies to new cohorts only)
- **Precondition:**
  - all 3 seeds complete
  - gate gap |B0−G| ≤ 0.10 at steps 1000 and 2000
  - both gate values in [0.3, 0.7]
- **Rule, ranks from raw logits:** an arm in {G, H, GH} is selected only if its inner-24 head-AUROC paired delta is > 0 vs **both** B0 and shuffle in all 3 seeds **and** the pooled hierarchical bootstrap (`all_seed_deltas`, 1000 draws) lower bound is > 0 vs both.
- A1 and A2 apply. No extra seeds, steps or arms after results are seen.

## Wording rules
- If the rule is unmet: "selection criterion unmet; evidence inconclusive". Never "no effect".
- If the rule is met: "selected on inner; not confirmed".
- Exploratory: "post-selection, non-confirmatory". The a624aeec exploratory numbers are permanently non-confirmatory.
- Calibration that fails the thresholds: "did not meet prespecified informativeness thresholds".
- Name the seeds, the target n (16 calibration, 24 inner, 30 exploratory) and the single-cohort, single-dataset scope. Make no AD-relevance claims.
