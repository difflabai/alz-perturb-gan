# Confirmation queue: anchored scorer and explicit job ledger

This note covers two additions to the EMPF confirmation pass ([PROTOCOL.md](PROTOCOL.md)).
Neither one changes a file that the running queue reads.

- **Part A, anchored scorer.** `score.py`, `analyze.py`, `test_confirm.py` and
  `source_manifest.json` are tracked byte-identical to the copies the live
  queue hash-checks. They were committed upstream in c59c2fa while this branch
  was being built, and the branch merges that commit rather than adding
  second copies. `gate_freeze.json` records the analysis pins and the GUARDS
  thresholds as of the first score.
- **Part B, job ledger.** [`queue_state.py`](queue_state.py) turns the queue's
  implicit job state into an explicit per-job state machine. It reads the
  legacy layout without writing to it and provides a Python driver for use
  after the drain.

`train_queue.sh` and `train_all.sh` stay byte-identical. The live bash process
reads its script lazily, so editing either file under a running queue would
change what it executes.

## Part A: gate freeze

PROTOCOL.md says: "No read-out, threshold, fold or metric may be added after
the first new score exists". `score.pin_analysis()` rewrites
`source_manifest.json` with whatever `analyze.py` and `test_confirm.py`
contain when it runs, so a post-result edit followed by a re-pin would leave
no trace. `gate_freeze.json` records the following:

- the `analysis_sources` sha256 of `analyze.py` and `test_confirm.py`;
- the `GUARDS` dict from both `score.py` and `analyze.py`, extracted by `ast`
  (value and source text);
- `first_score_cutoff`: the mtime of the pilot's `A_raw_score.json` under the
  current scoring pins (2026-11-20T06:41:10Z). The superseded first pilot score
  from 04:02:16Z is also recorded. Both analysis files were last modified
  before the cutoff (06:26 and 06:27); `pin_analysis()` ran at 06:41:31.

`tests/test_confirm_gate_freeze.py` checks the committed files against
`source_manifest.json` and `gate_freeze.json`. Changing `GUARDS`, `analyze.py`
or `test_confirm.py`, or re-pinning `analysis_sources`, fails the test and
appears as a git diff. To regenerate the freeze deliberately (for example for
an exploratory, labelled analysis), call `build_freeze()` from that test
module and commit the diff with the reason.

### Deferred to a post-drain follow-up

`score.py` is pinned, and the live queue checks its hash, so these fixes wait
until the drain:

1. **Scorer two-step write.** `score.py` writes `<label>_raw_score.json`,
   the crossfitted NPZs and `score.json` as separate files. A crash between
   them leaves a partial score directory. `analyze.py` treats a fit without
   `score.json` as missing and checks the raw-score hashes recorded in it, so
   a partial directory is not misread. It still has to be cleaned up by hand
   before a rerun. The fix is to write everything into
   `scores/foldK/seedS.partial.<rand>/`, fsync it, and then `os.replace` it to
   `scores/foldK/seedS/`. The ledger's SCORED state then only needs one atomic
   marker.
2. **Per-fit lock.** Two `score.py` processes for the same fit can currently
   interleave writes. The fix is a `durable_state.FileLock` on
   `scores/foldK/seedS` for the whole scoring run. Activation already has
   `ACTIVATION_LOCK`; the per-fit lock is separate.

After the drain, both changes need a deliberate `pin_sources` refresh, which
must not happen while any score exists (`pin_sources` refuses), or a new
confirmation directory. They are not part of this branch.

## Part B: the job machine

```mermaid
stateDiagram-v2
    [*] --> UNCLAIMED
    UNCLAIMED --> CLAIMED: claim (owner pid + /proc starttime)
    CLAIMED --> RUNNING: start (try n, launch id)
    RETRY_WAIT --> RUNNING: start
    CLAIMED --> UNCLAIMED: owner_dead
    RETRY_WAIT --> UNCLAIMED: owner_dead
    RUNNING --> UNCLAIMED: owner_dead (driver dead, child never started)
    RUNNING --> RUNNING: reown (driver dead, child alive)
    RUNNING --> EXITED_OK: exit_ok (rc 0)
    RUNNING --> FAILED: exit_failed (classify)
    FAILED --> RETRY_WAIT: retry [retryable and tries < 3]
    EXITED_OK --> TRAINED: verify_fit (RECEIPTED or AUDITED)
    TRAINED --> CLI_GENERATED: verify_cli_a (readout-A NPZ + CLI receipt)
    CLI_GENERATED --> SCORED: scored (scores/foldK/seedS/score.json)
    UNCLAIMED --> ABANDONED: abandon (manual, reason)
    FAILED --> ABANDONED: abandon
    SCORED --> [*]
    ABANDONED --> [*]
```

ABANDONED can also be reached from RETRY_WAIT, EXITED_OK, TRAINED and
CLI_GENERATED. It cannot be reached from CLAIMED or RUNNING: the owner has to
be gone first.

Each job's state lives in `<ledger>/fold<F>_seed<S>.json`, a
`durable_state/1` document. Next to it are its journal and its `.lock`. Every
transition takes the job's flock, checks the edge and the guard, and writes
`version+1` atomically.

**Failure reasons** come from `classify(log tail, rc, fit dir)`, applied in
this order:

| reason | evidence | retryable |
|---|---|---|
| HOST_DIFFERS | `existing derived EMPF host differs` | yes |
| HOST_PARTIAL_IMPORT | SyntaxError/IndentationError/ImportError/ModuleNotFoundError plus the derived host path (`alz-host-empf-native`) | yes |
| KILLED | rc missing, 137 or 9 | yes |
| OOM | `CUDA out of memory` / `OutOfMemoryError` | no |
| OUTPUT_EXISTS | `FileExistsError`, or rc 2 with an exists/outside message | no |
| SOURCE_CHANGED | the runtime's or the confirmation's source/manifest-changed messages | no |
| GENERATION_FAILED | fit has `final.pt` + `fit_receipt.json` but the CLI failed | yes (the CLI resumes a complete fit straight to generation) |
| TRAIN_ERROR | anything else | no |

A job gets at most 3 classified attempts (`tries <= 3`). If a driver crashes,
its jobs are re-queued (`requeues`, which is capped at 5). A crash does not
use up an attempt.

**Ownership.** A claim belongs to the driver's pid and its `/proc` starttime,
so a reused pid counts as dead. A running attempt belongs to its child. The
`_child` wrapper holds the flock on `<ledger>/children/fold<F>_seed<S>.try<n>.run`
for its whole life and passes the lock fd on to the job command. A free lock
therefore means no live child. If a child starts late, it blocks on that lock
and then sees that the ledger has moved on, so it never runs. When the
wrapper's child exits, the wrapper writes `.rc` atomically. A restarted driver
handles a RUNNING job with a dead owner like this:

- lock held: `reown` and wait for the child;
- lock free, `.rc` present: `exit` with that rc;
- `.child` present but no `.rc`: `exit(missing)`, which gives KILLED;
- the child never started: `owner_dead`, which re-queues the job.

This replaces the positional `queue.next` cursor. Jobs that were claimed but
not finished are re-queued after a crash. Under the old cursor they were lost:
this is how `8 0` and `1 1` were dropped.

**Queue state** is derived and never stored. It is `DRAINED` iff every job is
SCORED, ABANDONED or FAILED with no retry left (not retryable, or 3 tries).
`training_drained` is the weaker condition: nothing is unclaimed, claimed,
running, waiting to retry or retryable. `queue.json` records the job list and
the jobfile's sha256. `run` refuses a different jobfile unless
`--replace-jobfile` is given. States are keyed by (fold, seed), not by
position, so a new jobfile cannot inherit a stale cursor.

## Commands

```bash
PY=/mnt/nvme/ai/conda/anaconda3/bin/python3
# read-only; never writes into the run root
$PY -B analysis/coverage_empf_confirm_20261120/queue_state.py status \
    --run-root runs/coverage_empf_confirm_20261120 [--json] [--deep]
# adopt the legacy state into a ledger directory the operator chooses
$PY -B analysis/coverage_empf_confirm_20261120/queue_state.py status \
    --run-root runs/coverage_empf_confirm_20261120 --write-ledger --ledger-dir /some/dir
# AFTER the drain only: the Python replacement for train_queue.sh
$PY -B analysis/coverage_empf_confirm_20261120/queue_state.py run \
    --jobfile JOBS --ledger-dir LEDGER [--workers 8 --start-spacing 25 --poll 30]
$PY -B analysis/coverage_empf_confirm_20261120/queue_state.py abandon \
    --ledger-dir LEDGER --fold K --seed S --reason "..."
```

`status --write-ledger` writes to `<run root>/logs/queue_state/` when no
`--ledger-dir` is given. Nothing in this branch writes into
`runs/coverage_empf_confirm_20261120` unless the operator asks for that.

`run` keeps the following behaviour from `train_queue.sh`:

- worker `w` runs its jobs under `taskset -c 3w..3w+2` with
  `CUDA_VISIBLE_DEVICES=w % 2`;
- starts are spaced 25 s apart under a start lock;
- every job runs `scripts/microglia.py train --fold K --seed S --device cuda`
  from the repository root.

It differs in two ways. Retries follow `classify` rather than only the
activation-race grep. Claims are taken under `durable_state.FileLock`. Fault
injection for tests: `QUEUE_STATE_FAULT=after_claim|after_start|while_running|after_exit`.

## Legacy adoption (read-only)

The reader works over the live layout:

- **`queue.log`, generation 1** (`train_all.sh`): `start foldK seedS gpuG`,
  `end foldK seedS rc=X` and `queue finished`.
- **`queue.log`, generation 2** (`train_queue.sh`):
  `start ... coresA-B tryN`, `retry ... (activation race)`,
  `end ... rc=X|missing` and `queue2 finished`. The finished lines are recorded
  as markers and are never treated as terminal.
- **`queue.next` with `jobs_queue2.txt`**: the cursor and the jobfile sha256.
  If the cursor has passed a row that never started and the queue process is
  dead, that job is reported as a dropped job.
- **`pid_<tag>` and `rc_<tag>`**: a pid counts as live only when its
  `/proc/<pid>/cmdline` names microglia and the job's fold and seed, and the
  process started no earlier than the job's start line. Generation-1 jobs have
  no pid files; the reader finds them by the same cmdline match. A
  generation-1 job with no end line and a live process is RUNNING (gen 1, no
  retry).
- **The fit directory** (`runs/microglia_empf/fits/pbulk/hvg2000/foldK/seedS`):
  read with the fit-lifecycle legacy rules (FRESH, TRAINING, INVALID,
  FINAL_UNRECEIPTED, RECEIPTED, AUDITED; an explicit `fit_state.json` wins).
  These rules are duplicated here as the pure function `fit_state()` because
  this branch is on the calibration line.
- **The CLI readout-A NPZ and receipt** under `runs/microglia_empf/generated/`.
- **`scores/foldK/seedS/score.json`**, plus `score_all.sh`'s `score_queue.log`.
  The log is used only to explain why a generated fit has no score yet: a live
  `score.py` pid, one failed attempt, or a scorer that gave up after two.

## Merge status

Do not merge until the confirmation queue drains. This branch changes no
file that the queue or the scorer hash-checks. The scorer files it carries
come byte-identical from `research/calibration-optimization`, so a merge
cannot break a hash check. The hold is a policy, not a technical
requirement: nothing new lands in the live checkout while the queue and
`score_all.sh` run. The `run` driver is meant for after the drain.
