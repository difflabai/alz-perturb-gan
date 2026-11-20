"""Tests for alz_perturb_gan.durable_state (CPU only, stdlib only).

(a) transition-table property test (hypothesis when installed, otherwise
    seeded random event sequences over three toy machines)
(b) crash injection at every fault point and every nth occurrence
(c) multiprocess concurrency on one StateFile and on atomic_write_json
(d) FileLock busy / killed holder / stale owner record
(e) adopt() never mutates the legacy directory
(f) one regression test per adversarial-review finding (post-commit errors,
    sweep races and pid parsing, nested locks, non-blocking open, corrupt
    inputs, non-dict data, symlinks, sink states, long names)
"""

from __future__ import annotations

import enum
import hashlib
import json
import multiprocessing
import os
import random
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from alz_perturb_gan import durable_state as ds  # noqa: E402
from alz_perturb_gan.durable_state import (  # noqa: E402
    Busy,
    FileLock,
    GuardFailed,
    IllegalTransition,
    InvalidMachine,
    Machine,
    StateFile,
)

try:  # pragma: no cover - depends on the environment
    import hypothesis  # noqa: F401
    HAVE_HYPOTHESIS = True
except ImportError:
    HAVE_HYPOTHESIS = False


# --------------------------------------------------------------------------
# Toy machines (module level so spawn workers can import them)


class Lin(enum.Enum):
    A = 1
    B = 2
    C = 3
    D = 4


LINEAR = Machine(
    "linear", Lin, Lin.A, {Lin.D},
    {(Lin.A, "go"): Lin.B, (Lin.B, "go"): Lin.C, (Lin.C, "done"): Lin.D},
)


class Dia(enum.Enum):
    START = 1
    LEFT = 2
    RIGHT = 3
    JOIN = 4


DIAMOND = Machine(
    "diamond", Dia, Dia.START, {Dia.JOIN},
    {
        (Dia.START, "left"): Dia.LEFT,
        (Dia.START, "right"): Dia.RIGHT,
        (Dia.LEFT, "join"): Dia.JOIN,
        (Dia.RIGHT, "join"): Dia.JOIN,
    },
)


class Ret(enum.Enum):
    IDLE = 1
    RUN = 2
    FAILED = 3
    DONE = 4
    ABANDONED = 5


RETRY = Machine(
    "retry", Ret, Ret.IDLE, {Ret.DONE, Ret.ABANDONED},
    {
        (Ret.IDLE, "start"): Ret.RUN,
        (Ret.RUN, "fail"): Ret.FAILED,
        (Ret.FAILED, "retry"): Ret.RUN,
        (Ret.RUN, "ok"): Ret.DONE,
        (Ret.FAILED, "abandon"): Ret.ABANDONED,
    },
)

TOY_MACHINES = {"linear": LINEAR, "diamond": DIAMOND, "retry": RETRY}

# The counter machine is shared, as source, with the crash-injection child.
COUNTER_SRC = '''
import enum
from alz_perturb_gan.durable_state import Machine
class Ctr(enum.Enum):
    RUN = 1
    DONE = 2
COUNTER = Machine("counter", Ctr, Ctr.RUN, {Ctr.DONE},
                  {(Ctr.RUN, "inc"): Ctr.RUN, (Ctr.RUN, "finish"): Ctr.DONE})
def bump(data):
    data["n"] = data.get("n", 0) + 1
    return data
'''
exec(COUNTER_SRC)  # defines Ctr, COUNTER, bump in this module


def child_env(**extra):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(SRC), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    env.pop(ds.FAULT_ENV, None)
    env.update(extra)
    return env


# --------------------------------------------------------------------------
# (a) transition-table property test


class Model:
    """Pure-Python reference for one StateFile."""

    def __init__(self, machine):
        self.machine = machine
        self.state = machine.initial
        self.version = 1
        self.n = 0


def _step(sf, model, event, guard_ok):
    machine = model.machine
    before = sf.read().version
    target = machine.transitions.get((model.state, event))
    if target is None:
        with pytest.raises(IllegalTransition):
            sf.transition(event, update=bump)
        assert sf.read().version == before
        return
    if not guard_ok:
        with pytest.raises(GuardFailed):
            sf.transition(event, guard=lambda snap: False, update=bump)
        assert sf.read().version == before
        return
    seen = []
    snap = sf.transition(event, guard=lambda s: seen.append(s.state) or True, update=bump)
    assert seen == [model.state]
    model.state, model.version, model.n = target, model.version + 1, model.n + 1
    assert snap.state is target and snap.version == model.version


def _check(sf, model):
    snap = sf.read()
    assert snap.state is model.state
    assert snap.version == model.version
    assert snap.data.get("n", 0) == model.n
    state, seq = ds.replay_journal(sf.journal_path)
    assert (state, seq) == (model.state.name, model.version)
    assert [e["seq"] for e in sf.journal()] == list(range(1, model.version + 1))


def _random_sequences(machine, root, count, seed):
    rng = random.Random(seed)
    events = sorted(machine.events) + ["undeclared"]
    for i in range(count):
        sf = StateFile(root / f"{machine.name}-{i}.json", machine)
        sf.create()
        model = Model(machine)
        for _ in range(rng.randint(1, 8)):
            # Bias towards legal events so walks reach deep states.
            legal = machine.allowed_events(model.state)
            event = rng.choice(legal) if legal and rng.random() < 0.6 else rng.choice(events)
            _step(sf, model, event, rng.random() >= 0.1)
            _check(sf, model)


@pytest.mark.parametrize("name", sorted(TOY_MACHINES))
def test_transition_table_property(name, tmp_path):
    machine = TOY_MACHINES[name]
    if not HAVE_HYPOTHESIS:
        # 2000 seeded sequences in total over the three machines.
        count = {"linear": 667, "diamond": 667, "retry": 666}[name]
        _random_sequences(machine, tmp_path, count, seed=sum(map(ord, name)))
        return
    from hypothesis import settings  # pragma: no cover
    from hypothesis import strategies as st
    from hypothesis.stateful import (RuleBasedStateMachine, invariant, rule,
                                     run_state_machine_as_test)

    class Walk(RuleBasedStateMachine):
        def __init__(self):
            super().__init__()
            self.dir = tempfile.mkdtemp(dir=tmp_path)
            self.sf = StateFile(os.path.join(self.dir, "s.json"), machine)
            self.sf.create()
            self.model = Model(machine)

        @rule(event=st.sampled_from(sorted(machine.events) + ["undeclared"]),
              guard_ok=st.booleans())
        def step(self, event, guard_ok):
            _step(self.sf, self.model, event, guard_ok)

        @invariant()
        def agrees(self):
            _check(self.sf, self.model)

        def teardown(self):
            shutil.rmtree(self.dir, ignore_errors=True)

    run_state_machine_as_test(Walk, settings=settings(max_examples=200, deadline=None))


def test_validate_rejects_malformed_tables():
    class S(enum.Enum):
        A = 1
        B = 2
        C = 3

    good = {(S.A, "go"): S.B, (S.B, "go"): S.C}
    Machine("ok", S, S.A, {S.C}, good)
    with pytest.raises(InvalidMachine, match="unreachable"):
        Machine("unreach", S, S.A, {S.C}, {(S.A, "go"): S.B})
    # The same state is fine once it is declared as an adoption entry.
    Machine("adopt", S, S.A, {S.C}, {(S.A, "go"): S.B, (S.B, "back"): S.A},
            adopt_entries={S.C})
    with pytest.raises(InvalidMachine, match="terminal"):
        Machine("term", S, S.A, {S.C}, {**good, (S.C, "back"): S.A})
    with pytest.raises(InvalidMachine, match="not declared"):
        Machine("undecl", S, S.A, {S.C}, good, events={"go_typo"})
    with pytest.raises(InvalidMachine, match="reserved"):
        Machine("reserved", S, S.A, {S.C}, {**good, (S.A, ds.ADOPT_EVENT): S.B})
    with pytest.raises(InvalidMachine, match="initial"):
        Machine("init", S, "A", {S.C}, good)


def test_expect_version_and_terminal(tmp_path):
    sf = StateFile(tmp_path / "s.json", LINEAR)
    sf.create({"k": 1})
    with pytest.raises(ds.VersionConflict):
        sf.transition("go", expect_version=7)
    assert sf.transition("go", expect_version=1).version == 2
    sf.transition("go")
    sf.transition("done")
    for event in sorted(LINEAR.events):
        with pytest.raises(IllegalTransition):
            sf.transition(event)
    assert sf.read().version == 4 and sf.read().data == {"k": 1}
    with pytest.raises(ds.AlreadyExists):
        sf.create()
    with pytest.raises(ds.NoState):
        StateFile(tmp_path / "missing.json", LINEAR).transition("go")


def test_journal_ahead_and_torn_tail(tmp_path):
    sf = StateFile(tmp_path / "s.json", LINEAR)
    sf.create()
    sf.transition("go")
    # A torn final line (a crash inside the append) is cut on recovery.
    with open(sf.journal_path, "ab") as fh:
        fh.write(b'{"seq": 3, "fr')
    StateFile(sf.path, LINEAR)
    assert [e["seq"] for e in sf.journal()] == [1, 2]
    assert Path(sf.journal_path).read_bytes().endswith(b"\n")
    # A journal ahead of the state cannot come from this library's write order.
    with open(sf.journal_path, "ab") as fh:
        fh.write(json.dumps({"seq": 9, "to": "C"}).encode() + b"\n")
    with pytest.raises(ds.CorruptState, match="ahead"):
        StateFile(sf.path, LINEAR)


# --------------------------------------------------------------------------
# (b) crash injection

CRASH_CHILD = COUNTER_SRC + '''
import sys
from alz_perturb_gan.durable_state import StateFile
sf = StateFile(sys.argv[1], COUNTER)
if sys.argv[3] == "create":
    sf.create({"n": 0})
for _ in range(int(sys.argv[2])):
    sf.transition("inc", update=bump)
'''

FAULT_POINTS = ["after_tmp_write", "after_fsync", "after_replace", "after_journal"]
TRANSITIONS = 3


def _temps(directory):
    return sorted(p.name for p in Path(directory).iterdir() if p.name.endswith(".tmp"))


@pytest.mark.parametrize("nth", range(1, TRANSITIONS + 1))
@pytest.mark.parametrize("point", FAULT_POINTS)
def test_crash_injection(point, nth, tmp_path):
    path = tmp_path / "state.json"
    sf = StateFile(path, COUNTER)
    sf.create({"n": 0})
    sf.transition("inc", update=bump)  # v2, n=1
    v0 = sf.read().version

    proc = subprocess.run(
        [sys.executable, "-B", "-c", CRASH_CHILD, str(path), str(TRANSITIONS), "no"],
        env=child_env(**{ds.FAULT_ENV: f"{point}:{nth}"}), capture_output=True, text=True,
    )
    assert proc.returncode == 137, proc.stderr
    child_pid = None
    leftovers = _temps(tmp_path)

    # Lock-free read right after the crash: always parses, pre or post state.
    raw = sf.read()
    pre, post = v0 + nth - 1, v0 + nth
    committed = point in ("after_replace", "after_journal")
    assert raw.version == (post if committed else pre)
    assert raw.data["n"] == raw.version - 1  # data and version written together

    # The killed holder released the kernel lock; its owner record is stale.
    owner = FileLock(path).read_owner()
    assert owner is not None and owner["status"] == "stale"
    child_pid = owner["pid"]

    # Reopen: the journal reconciles to the state version (I2).
    snap = StateFile(path, COUNTER).read()
    assert snap == raw
    journal = sf.journal()
    assert journal[-1]["seq"] == snap.version
    assert ds.replay_journal(sf.journal_path) == ("RUN", snap.version)
    if point == "after_replace":
        assert journal[-1]["event"] == ds.RECONCILE_EVENT
        assert journal[-1]["reconciled_from_seq"] == snap.version - 1
    else:
        assert all(e["event"] != ds.RECONCILE_EVENT for e in journal)

    # Only a crash before os.replace leaves a temp file, and it is the child's.
    if point in ("after_tmp_write", "after_fsync"):
        assert len(leftovers) == 1 and leftovers[0].startswith(".state.json.")
        assert leftovers[0].endswith(f".{child_pid}.tmp")
    else:
        assert leftovers == []

    # Decoys that sweep_temps must keep: a live writer's temp, normal files.
    live = tmp_path / f".state.json.abcd1234.{os.getpid()}.tmp"
    live.write_text("live")
    (tmp_path / "notes.txt").write_text("keep")
    keep = set(os.listdir(tmp_path)) - set(leftovers)
    assert ds.sweep_temps(tmp_path, dry_run=True) == [str(tmp_path / n) for n in leftovers]
    removed = ds.sweep_temps(tmp_path)
    assert removed == [str(tmp_path / n) for n in leftovers]
    assert set(os.listdir(tmp_path)) == keep
    assert (tmp_path / "state.json.lock").exists()

    # The machine keeps working after recovery.
    after = StateFile(path, COUNTER).transition("inc", update=bump)
    assert after.version == snap.version + 1
    assert [e["seq"] for e in sf.journal()][-1] == after.version


@pytest.mark.parametrize("point", FAULT_POINTS)
def test_crash_during_create(point, tmp_path):
    path = tmp_path / "state.json"
    proc = subprocess.run(
        [sys.executable, "-B", "-c", CRASH_CHILD, str(path), "0", "create"],
        env=child_env(**{ds.FAULT_ENV: point}), capture_output=True, text=True,
    )
    assert proc.returncode == 137, proc.stderr
    sf = StateFile(path, COUNTER)
    if point in ("after_tmp_write", "after_fsync"):
        assert sf.read() is None and sf.journal() == []
        assert len(ds.sweep_temps(tmp_path)) == 1
        assert sf.create({"n": 0}).version == 1
    else:
        assert sf.read().version == 1
        assert sf.journal()[-1]["seq"] == 1
    assert _temps(tmp_path) == []


def test_fault_env_is_inert_without_match(tmp_path, monkeypatch):
    monkeypatch.setenv(ds.FAULT_ENV, "no_such_point")
    ds.atomic_write_json(tmp_path / "x.json", {"a": 1})
    assert json.loads((tmp_path / "x.json").read_text()) == {"a": 1}


# --------------------------------------------------------------------------
# (c) multiprocess concurrency

WRITERS = 8
INCREMENTS = 200
READS = 5000


def _inc_worker(path, count):
    sf = StateFile(path, COUNTER, recover=False)
    for _ in range(count):
        sf.transition("inc", update=bump)


def _state_reader(path, reads, queue):
    sf = StateFile(path, COUNTER, recover=False)
    invalid = missing = regress = 0
    last = 0
    deadline = time.monotonic() + 120
    while sf.read().version < 2 and time.monotonic() < deadline:
        pass  # start counting once the writers are running
    first = sf.read().version
    for _ in range(reads):
        try:
            with open(path, "rb") as fh:
                doc = json.loads(fh.read())
            snap = sf._parse(json.dumps(doc).encode())
            if snap.version < last or snap.data.get("n") != snap.version - 1:
                regress += 1
            last = snap.version
        except FileNotFoundError:
            missing += 1
        except (ValueError, ds.CorruptState):
            invalid += 1
    queue.put((invalid, missing, regress, (first, last)))


def test_concurrent_transitions(tmp_path):
    path = str(tmp_path / "state.json")
    StateFile(path, COUNTER).create({"n": 0})
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    reader = ctx.Process(target=_state_reader, args=(path, READS, queue))
    writers = [ctx.Process(target=_inc_worker, args=(path, INCREMENTS)) for _ in range(WRITERS)]
    for p in writers:
        p.start()
    reader.start()
    invalid, missing, regress, (first, last) = queue.get(timeout=600)
    for p in writers + [reader]:
        p.join(timeout=600)
        assert p.exitcode == 0
    assert (invalid, missing, regress) == (0, 0, 0)
    assert 1 < first <= last  # the reads overlapped the writers

    snap = StateFile(path, COUNTER).read()
    total = WRITERS * INCREMENTS
    assert snap.version == 1 + total  # create is version 1
    assert snap.data["n"] == total
    seqs = [e["seq"] for e in read_all(path)]
    assert seqs == list(range(1, total + 2))
    assert len({e["pid"] for e in read_all(path)[1:]}) == WRITERS
    assert _temps(tmp_path) == []


def read_all(path):
    return ds.read_journal(path + ds.JOURNAL_SUFFIX)


def _json_writer(path, wid, count, queue):
    errors = 0
    for i in range(count):
        try:
            ds.atomic_write_json(path, {"writer": wid, "i": i, "pad": "x" * 4096})
        except OSError:
            errors += 1
    queue.put(("w", errors))


def _json_reader(path, reads, queue):
    missing = invalid = 0
    for _ in range(reads):
        try:
            with open(path, "rb") as fh:
                doc = json.loads(fh.read())
            if set(doc) != {"writer", "i", "pad"} or len(doc["pad"]) != 4096:
                invalid += 1
        except FileNotFoundError:
            missing += 1
        except ValueError:
            invalid += 1
    queue.put(("r", (missing, invalid)))


def test_concurrent_atomic_write_json(tmp_path):
    """Regression for repro_tmp.py: writers sharing one '<path>.tmp' name."""
    path = str(tmp_path / "out.json")
    ds.atomic_write_json(path, {"writer": -1, "i": -1, "pad": "x" * 4096})
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    procs = [ctx.Process(target=_json_writer, args=(path, w, 150, queue)) for w in range(WRITERS)]
    procs.append(ctx.Process(target=_json_reader, args=(path, READS, queue)))
    for p in procs:
        p.start()
    results = [queue.get(timeout=600) for _ in procs]
    for p in procs:
        p.join(timeout=600)
        assert p.exitcode == 0
    assert [r for kind, r in results if kind == "w"] == [0] * WRITERS
    assert [r for kind, r in results if kind == "r"] == [(0, 0)]
    assert _temps(tmp_path) == []


def test_atomic_write_keeps_mode_and_cleans_up(tmp_path):
    target = tmp_path / "f.bin"
    ds.atomic_write_bytes(target, b"one")
    os.chmod(target, 0o640)
    ds.atomic_write_bytes(target, b"two")
    assert target.read_bytes() == b"two"
    assert target.stat().st_mode & 0o777 == 0o640
    with pytest.raises(ValueError):
        ds.atomic_write_json(target, {"bad": float("nan")})
    assert target.read_bytes() == b"two"
    assert _temps(tmp_path) == []


# --------------------------------------------------------------------------
# (d) FileLock

HOLDER = '''
import sys, time
from alz_perturb_gan.durable_state import FileLock
lock = FileLock(sys.argv[1], purpose="holder").acquire()
print("locked", flush=True)
time.sleep(120)
'''


def test_filelock_busy_and_killed_holder(tmp_path):
    path = tmp_path / "res"
    proc = subprocess.Popen([sys.executable, "-B", "-c", HOLDER, str(path)],
                            env=child_env(), stdout=subprocess.PIPE, text=True)
    try:
        assert proc.stdout.readline().strip() == "locked"
        with pytest.raises(Busy) as info:
            FileLock(path, blocking=False).acquire()
        owner = info.value.owner
        assert owner["pid"] == proc.pid and owner["purpose"] == "holder"
        assert owner["status"] == "alive"
        assert owner["pid_starttime"] == ds.pid_starttime(proc.pid)
        start = time.monotonic()
        with pytest.raises(Busy):
            FileLock(path, timeout=0.2).acquire()
        assert time.monotonic() - start >= 0.19
        with pytest.raises(Busy):
            FileLock(path, exclusive=False, blocking=False).acquire()
    finally:
        proc.send_signal(signal.SIGKILL)
        proc.wait()
        proc.stdout.close()
    # The kernel released the lock; the leftover owner record is stale.
    lock = FileLock(path, blocking=False)
    assert lock.read_owner()["status"] == "stale"
    with lock:
        assert lock.read_owner()["pid"] == os.getpid()
        assert lock.read_owner()["status"] == "alive"
    assert lock.read_owner() is None  # clean release removes the record
    assert (tmp_path / "res.lock").exists()  # lock files are never deleted


def test_shared_locks_coexist(tmp_path):
    path = tmp_path / "res"
    with FileLock(path, exclusive=False), FileLock(path, exclusive=False, blocking=False):
        with pytest.raises(Busy) as info:
            FileLock(path, blocking=False).acquire()
        assert info.value.owner is None


def test_owner_status_staleness():
    me = os.getpid()
    start = ds.pid_starttime(me)
    host = ds.socket.gethostname()
    assert isinstance(start, int)
    assert ds.owner_status({"pid": me, "pid_starttime": start, "host": host}) == "alive"
    # Same (live) pid but a different start time: the pid was reused.
    assert ds.owner_status({"pid": me, "pid_starttime": start + 1, "host": host}) == "stale"
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    assert ds.owner_status({"pid": dead.pid, "pid_starttime": 1, "host": host}) == "stale"
    assert ds.owner_status({"pid": me, "host": "elsewhere.invalid"}) == "unknown"


# --------------------------------------------------------------------------
# (e) adopt()


def _tree_digest(root, names):
    h = hashlib.sha256()
    for rel in sorted(names):
        p = Path(root) / rel
        st = p.lstat()
        h.update(rel.encode() + b"\0" + oct(st.st_mode).encode() + str(st.st_mtime_ns).encode())
        if p.is_file():
            h.update(p.read_bytes())
    return h.hexdigest()


def _listing(root):
    out = set()
    for dirpath, dirnames, filenames in os.walk(root):
        for name in dirnames + filenames:
            out.add(os.path.relpath(os.path.join(dirpath, name), root))
    return out


def test_adopt_is_read_only_on_legacy_files(tmp_path):
    legacy = tmp_path / "run"
    (legacy / "checkpoints").mkdir(parents=True)
    (legacy / "checkpoints" / "step_100.pt").write_bytes(os.urandom(256))
    (legacy / "progress.json").write_text(json.dumps({"step": 100}))
    (legacy / "CAL_GPU0_done").write_text("")
    os.chmod(legacy / "progress.json", 0o444)
    before_names = _listing(legacy)
    before = _tree_digest(legacy, before_names)

    calls = []

    def infer(directory):
        calls.append(directory)
        progress = json.loads((Path(directory) / "progress.json").read_text())
        state = Ret.DONE if (Path(directory) / "CAL_GPU0_done").exists() else Ret.RUN
        return state, {"step": progress["step"], "marker": "CAL_GPU0_done"}

    sf = StateFile(legacy / "fit.state.json", RETRY)
    snap = sf.adopt(infer)
    assert calls == [str(legacy)]
    assert snap.state is Ret.DONE and snap.version == 1
    assert snap.data["adopted"]["evidence"] == {"step": 100, "marker": "CAL_GPU0_done"}
    entry, = sf.journal()
    assert entry["event"] == ds.ADOPT_EVENT and entry["from"] is None
    assert entry["to"] == "DONE" and entry["evidence"]["step"] == 100

    assert _tree_digest(legacy, before_names) == before
    added = _listing(legacy) - before_names
    assert added == {"fit.state.json", "fit.state.json.journal.jsonl", "fit.state.json.lock"}

    # Idempotent: an existing state file wins and inference is not repeated.
    assert StateFile(legacy / "fit.state.json", RETRY).adopt(infer) == sf.read()
    assert len(calls) == 1
    # Adopted terminal state accepts no further events.
    with pytest.raises(IllegalTransition):
        sf.transition("retry")


def test_adopt_rejects_unknown_state(tmp_path):
    sf = StateFile(tmp_path / "s.json", RETRY)
    with pytest.raises(ds.CorruptState):
        sf.adopt(lambda d: ("NOT_A_STATE", {}))
    assert not sf.exists() and sf.journal() == []


# --------------------------------------------------------------------------
# sweep_temps details


def test_sweep_temps_rules(tmp_path):
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    names = {
        f".a.json.zzzz1111.{dead.pid}.tmp": True,        # dead writer
        f".a.json.zzzz2222.{os.getpid()}.tmp": False,    # live writer
        f"model.pt.partial.{dead.pid}": False,           # digits are not a pid: age only
        "model.pt.partial": False,                       # no pid, young
        "a.json": False,
        "a.json.lock": False,
    }
    for name in names:
        (tmp_path / name).write_text("x")
    expect = sorted(str(tmp_path / n) for n, gone in names.items() if gone)
    assert ds.sweep_temps(tmp_path, dry_run=True) == expect
    # A held target lock protects even a dead writer's temp file.
    with FileLock(tmp_path / "a.json"):
        assert ds.sweep_temps(tmp_path, dry_run=True) == []
    # Old enough: age alone qualifies files without a trusted pid ...
    old = time.time() - 7200
    for name in ("model.pt.partial", f"model.pt.partial.{dead.pid}"):
        os.utime(tmp_path / name, (old, old))
    # The live writer's temp is as old as it can be (written as it started).
    started = ds._process_start_epoch(os.getpid())
    os.utime(tmp_path / f".a.json.zzzz2222.{os.getpid()}.tmp", (started, started))
    assert ds.sweep_temps(tmp_path, min_age_s=0.5, dry_run=True) == sorted(
        expect + [str(tmp_path / "model.pt.partial"),
                  str(tmp_path / f"model.pt.partial.{dead.pid}")])
    removed = ds.sweep_temps(tmp_path, min_age_s=3600)
    assert removed == sorted(expect + [str(tmp_path / "model.pt.partial"),
                                       str(tmp_path / f"model.pt.partial.{dead.pid}")])
    # ... but never a live writer's own-format temp, however old.
    assert sorted(os.listdir(tmp_path)) == sorted(
        ["a.json", "a.json.lock", f".a.json.zzzz2222.{os.getpid()}.tmp"])


@pytest.mark.parametrize("name,pattern,ok", [
    (".a.json.x1.12.tmp", ".*.*.tmp", True),
    ("a.json.tmp", ".*.*.tmp", False),
    ("m.partial.3", "*.partial*", True),
    ("m.part", "*.partial*", False),
    ("abc", "a?c", True),
    ("", "*", True),
])
def test_glob_match(name, pattern, ok):
    assert ds._glob_match(name, pattern) is ok


# --------------------------------------------------------------------------
# Regressions from the adversarial review (one per finding)


def _cycle_machine():
    class C(enum.Enum):
        A = 1
        B = 2
        DONE = 3
    return C, Machine("cyc", C, C.A, {C.DONE},
                      {(C.A, "go"): C.B, (C.B, "back"): C.A, (C.B, "finish"): C.DONE})


def test_post_commit_failure_is_reported_as_committed(tmp_path, monkeypatch):
    """Finding 1: a journal/fsync error after os.replace must not look 'not applied'."""
    C, m = _cycle_machine()
    sf = StateFile(tmp_path / "s.json", m)
    sf.create({"n": 0})

    def no_space(self, entry, opened=None):
        if opened is not None:
            os.close(opened[0])
        raise OSError(28, "No space left on device (simulated)")

    monkeypatch.setattr(StateFile, "_append_journal", no_space)
    with pytest.raises(ds.CommittedWithError) as info:
        sf.transition("go", update=lambda d: d.update(n=1))
    assert isinstance(info.value.__cause__, OSError)
    assert info.value.snapshot.version == 2 and info.value.snapshot.state is C.B
    assert (sf.read().state, sf.read().version) == (C.B, 2)
    monkeypatch.undo()
    # A retry keyed on the version the caller saw cannot double-apply.
    with pytest.raises(ds.VersionConflict):
        sf.transition("back", expect_version=1)
    assert sf.read().version == 2
    # The next write reconciles the missing journal line.
    sf.transition("back", expect_version=2)
    journal = sf.journal()
    assert [e["seq"] for e in journal] == [1, 2, 3]
    assert journal[1]["event"] == ds.RECONCILE_EVENT

    # A directory fsync failing after the replace is also a commit.
    real = ds._fsync_dir
    calls = []

    def flaky_fsync(path):
        calls.append(path)
        raise OSError(5, "EIO (simulated)")

    monkeypatch.setattr(ds, "_fsync_dir", flaky_fsync)
    with pytest.raises(ds.CommittedWithError):
        sf.transition("go")
    assert sf.read().version == 4 and calls
    monkeypatch.setattr(ds, "_fsync_dir", real)

    # A failure before the replace is a plain error and nothing changes.
    def broken_replace(src, dst):
        raise OSError(28, "No space left on device (simulated)")

    monkeypatch.setattr(ds.os, "replace", broken_replace)
    with pytest.raises(OSError) as info:
        sf.transition("back")
    assert not isinstance(info.value, ds.CommittedWithError)
    monkeypatch.undo()
    assert sf.read().version == 4
    assert _temps(tmp_path) == []


def test_unwritable_journal_fails_before_commit(tmp_path):
    """Finding 1/5: a read-only journal fails the write with nothing changed."""
    C, m = _cycle_machine()
    sf = StateFile(tmp_path / "s.json", m)
    sf.create()
    os.chmod(sf.journal_path, 0o444)
    if os.access(sf.journal_path, os.W_OK):  # pragma: no cover - running as root
        pytest.skip("root ignores file modes")
    with pytest.raises(PermissionError):
        sf.transition("go")
    assert sf.read().version == 1
    # A read-only journal that needs no repair does not stop opening/reading.
    assert StateFile(sf.path, m).read().version == 1


SWEEP_WORKER = COUNTER_SRC + '''
import sys, os, time, collections
from alz_perturb_gan import durable_state as ds
kind, path, plain, seconds = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
errors = collections.Counter()
ok = 0
end = time.monotonic() + seconds
sf = None
while time.monotonic() < end:
    try:
        if kind == "sf":
            if sf is None:
                sf = ds.StateFile(path, COUNTER)
            sf.transition("inc", update=bump)
        else:
            ds.atomic_write_json(plain, {"pid": os.getpid(), "pad": "x" * 100000})
        ok += 1
    except Exception as exc:
        errors[type(exc).__name__] += 1
print(kind, ok, sum(errors.values()), dict(errors), flush=True)
'''


def test_sweep_never_races_live_writers(tmp_path):
    """Finding 2: a min_age_s=0 sweep loop must not delete live writers' temps."""
    path = tmp_path / "s.json"
    StateFile(path, COUNTER).create({"n": 0})
    procs = [
        subprocess.Popen([sys.executable, "-B", "-c", SWEEP_WORKER, kind, str(path),
                          str(tmp_path / "plain.json"), "3"],
                         env=child_env(), stdout=subprocess.PIPE, text=True)
        for kind in ["sf"] * 3 + ["plain"] * 3
    ]
    removed = 0
    while any(p.poll() is None for p in procs):
        removed += len(ds.sweep_temps(tmp_path, min_age_s=0))
    results = [p.communicate()[0].split() for p in procs]
    for p in procs:
        assert p.returncode == 0
    assert all(int(r[2]) == 0 for r in results), results
    assert all(int(r[1]) > 0 for r in results), results
    assert _temps(tmp_path) == []
    snap = StateFile(path, COUNTER).read()
    assert snap.data["n"] == snap.version - 1
    assert [e["seq"] for e in ds.read_journal(str(path) + ds.JOURNAL_SUFFIX)] == \
        list(range(1, snap.version + 1))


def test_sweep_keeps_lock_owner_temp_of_live_holder(tmp_path):
    """Finding 2: '<x>.lock.owner' temps are guarded by '<x>.lock', and live pids win."""
    assert ds._lock_file_for("s.json.lock.owner") == "s.json.lock"
    assert ds._lock_file_for("s.json") == "s.json.lock"
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        old = time.time() - 7200
        with FileLock(tmp_path / "s.json"):
            mine = tmp_path / f".s.json.lock.owner.abcd1234.{os.getpid()}.tmp"
            other = tmp_path / f".plain.json.efgh5678.{sleeper.pid}.tmp"
            for f in (mine, other):
                f.write_text("x")
            started = ds._process_start_epoch(os.getpid())
            os.utime(mine, (started, started))  # as old as a live holder's temp can be
            # 'other' keeps a fresh mtime: its live writer started before it.
            assert ds.sweep_temps(tmp_path, min_age_s=0, dry_run=True) == []
        # Released lock: the own-format temp still has a live pid, so it stays.
        assert ds.sweep_temps(tmp_path, min_age_s=0, dry_run=True) == []
        # A pid that started after the temp was written was reused: sweepable.
        os.utime(other, (old, old))
        assert ds.sweep_temps(tmp_path, min_age_s=0, dry_run=True) == [str(other)]
    finally:
        sleeper.kill()
        sleeper.wait()


def test_sweep_does_not_trust_arbitrary_digits(tmp_path):
    """Finding 3: dates/shard numbers are not pids; huge numbers must not crash."""
    names = [".shard.json.abcdefgh.4000000000.tmp", "gen.partial.3", "gen_2026.partial",
             "out.partial.20261120", ".x.partial.npz", "z.partial.99999999999999999999",
             ".y.json.tmp", ".q.json.short.12.tmp"]
    for n in names:
        (tmp_path / n).write_text("live writer")
    assert ds.sweep_temps(tmp_path, dry_run=True) == []  # all fresh, min_age 1h
    assert ds._temp_owner(".x.partial.npz") == ("x.npz", None)
    assert ds._temp_owner("out.partial.20261120") == ("out", None)
    assert ds._temp_owner("model.pt.partial") == ("model.pt", None)
    assert ds._temp_owner(".shard.json.abcdefgh.4000000000.tmp") == (None, None)
    assert ds._temp_owner(".a.json.ab_cd123.42.tmp") == ("a.json", 42)
    assert ds._pid_alive(99999999999999999999) is False
    assert ds._pid_alive(0) is False and ds._pid_alive(-1) is False
    # PR #19's hidden partial is guarded by its real target's lock.
    old = time.time() - 7200
    os.utime(tmp_path / ".x.partial.npz", (old, old))
    with FileLock(tmp_path / "x.npz"):
        assert str(tmp_path / ".x.partial.npz") not in ds.sweep_temps(tmp_path, dry_run=True)
    assert str(tmp_path / ".x.partial.npz") in ds.sweep_temps(tmp_path, dry_run=True)


def _run_with_deadline(fn, seconds=20):
    import threading
    out = {}

    def target():
        try:
            out["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - reported below
            out["error"] = exc

    th = threading.Thread(target=target, daemon=True)
    th.start()
    th.join(seconds)
    assert not th.is_alive(), "deadlocked"
    if "error" in out:
        raise out["error"]
    return out.get("value")


def test_nested_lock_reuses_or_refuses_instead_of_deadlocking(tmp_path):
    """Finding 4: sf.lock() + transition()/StateFile() in one thread."""
    C, m = _cycle_machine()
    sf = StateFile(tmp_path / "s.json", m)
    sf.create()

    def job():
        with sf.lock(purpose="job"):
            assert ds.held_mode(sf.path) == "exclusive"
            StateFile(sf.path, m)              # construction inside the lock
            sf.transition("go")                # reuses the caller's lock
            sf.transition("back", expect_version=2)
            with pytest.raises(RuntimeError, match="deadlock"):
                FileLock(sf.path).acquire()    # explicit nesting is refused
            with pytest.raises(Busy):
                FileLock(sf.path, blocking=False).acquire()
            # The owner record is the outer holder's.
            assert FileLock(sf.path).read_owner()["purpose"] == "job"
        assert ds.held_mode(sf.path) is None
        with FileLock(sf.path, exclusive=False):
            with pytest.raises(RuntimeError, match="shared"):
                sf.transition("go")
        return sf.read().version

    assert _run_with_deadline(job) == 3

    # Another thread still blocks normally (and is not confused with ours).
    import threading
    order = []
    with sf.lock():
        th = threading.Thread(target=lambda: order.append(sf.transition("go").version))
        th.start()
        time.sleep(0.2)
        assert order == []
    th.join(10)
    assert order == [4]


LONG_HOLDER = '''
import sys, time
from alz_perturb_gan.durable_state import FileLock
FileLock(sys.argv[1], purpose="long job").acquire()
print("held", flush=True)
time.sleep(60)
'''


def test_constructor_never_blocks_or_needs_write_access(tmp_path):
    """Finding 5: StateFile() is a cheap, non-blocking open for status readers."""
    C, m = _cycle_machine()
    path = tmp_path / "s.json"
    StateFile(path, m).create()
    holder = subprocess.Popen([sys.executable, "-B", "-c", LONG_HOLDER, str(path)],
                              env=child_env(), stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "held"
        start = time.monotonic()
        assert _run_with_deadline(lambda: StateFile(path, m).read().version, 5) == 1
        assert time.monotonic() - start < 2
    finally:
        holder.kill()
        holder.wait()
        holder.stdout.close()

    # Read-only directory without a lock file: open and read still work.
    os.unlink(str(path) + ds.LOCK_SUFFIX)
    os.chmod(tmp_path, 0o555)
    try:
        if os.access(tmp_path, os.W_OK):  # pragma: no cover - running as root
            pytest.skip("root ignores directory modes")
        assert StateFile(path, m).read().version == 1
        with pytest.raises(PermissionError):
            StateFile(path, m).transition("go")
    finally:
        os.chmod(tmp_path, 0o755)


def _state_doc(**over):
    doc = {"schema": ds.SCHEMA, "machine": "cyc", "state": "A", "version": 1}
    doc.update(over)
    return json.dumps(doc).encode()


CORRUPT_STATES = {
    "empty": b"",
    "garbage": b"\x00\x00\x00",
    "non_utf8": b"\xff\xfe{",
    "list": b"[1,2]",
    "legacy_plain": b'{"status":"completed"}',
    "wrong_machine": _state_doc(machine="other"),
    "bad_state_name": _state_doc(state="ZZ"),
    "state_list": _state_doc(state=["A"]),
    "state_dict": _state_doc(state={"a": 1}),
    "version_float": _state_doc(version=1.0),
    "version_bool": _state_doc(version=True),
    "data_list": _state_doc(data=[1]),
    "data_str": _state_doc(data="x"),
    "data_zero": _state_doc(data=0),
    "owner_str": _state_doc(owner="x"),
    "updated_at_str": _state_doc(updated_at="now"),
    "nan": b'{"schema":"durable_state/1","machine":"cyc","state":"A","version":1,"data":{"x":NaN}}',
    "truncated": _state_doc()[:30],
}


@pytest.mark.parametrize("case", sorted(CORRUPT_STATES))
def test_corrupt_state_file_raises_corruptstate(case, tmp_path):
    """Finding 6: every malformed state document is CorruptState."""
    C, m = _cycle_machine()
    path = tmp_path / "s.json"
    path.write_bytes(CORRUPT_STATES[case])
    with pytest.raises(ds.CorruptState):
        StateFile(path, m).read()
    with pytest.raises(ds.CorruptState):
        StateFile(path, m, recover=False).transition("go")


def _good_state(version=3):
    return _state_doc(version=version, state="B", data={"n": version - 1})


CORRUPT_JOURNALS = {
    "garbage_last_complete": b'{"seq":1,"to":"A"}\nnot json\n',
    "nul_line_newline": b'{"seq":1,"to":"A"}\n' + b"\x00" * 50 + b"\n",
    "non_utf8_line": b'{"seq":1,"to":"A"}\n\xff\xfe\n',
    "missing_seq": b'{"to":"A"}\n',
    "seq_str": b'{"seq":"2","to":"A"}\n',
    "seq_bool": b'{"seq":true,"to":"A"}\n',
    "missing_to": b'{"seq":1}\n',
    "non_dict_line": b"[1]\n",
    "nan_line": b'{"seq":1,"to":"A","x":NaN}\n',
    "mid_garbage": b'{"seq":1,"to":"A"}\nxx\n{"seq":2,"to":"B"}\n',
    "decreasing_mid": b'{"seq":2,"to":"A"}\n{"seq":1,"to":"A"}\n',
    "ahead": b'{"seq":1,"to":"A"}\n{"seq":9,"to":"A"}\n',
}


@pytest.mark.parametrize("case", sorted(CORRUPT_JOURNALS))
def test_corrupt_journal_raises_corruptstate(case, tmp_path):
    """Finding 6: malformed journals are CorruptState, never reconciled over."""
    C, m = _cycle_machine()
    path = tmp_path / "s.json"
    path.write_bytes(_good_state())
    journal = Path(str(path) + ds.JOURNAL_SUFFIX)
    journal.write_bytes(CORRUPT_JOURNALS[case])
    before = journal.read_bytes()
    with pytest.raises(ds.CorruptState):
        StateFile(path, m)
    with pytest.raises(ds.CorruptState):
        StateFile(path, m, recover=False).transition("back")
    if case != "ahead":  # 'ahead' is a valid journal, just ahead of the state
        with pytest.raises(ds.CorruptState):
            ds.replay_journal(journal)
    assert journal.read_bytes() == before  # no reconcile line appended


@pytest.mark.parametrize("raw,tail", [
    (b'{"seq":1,"to":"A"}\n{"seq":2,"to":"B"}\n{"seq":3,"t', 2),
    (b'{"seq":1,"to":"A"}\n' + b"\x00" * 50, 1),
    (b'{"seq":1,"to":"A"}\n{"seq":2,"pad":"' + b"x" * 20000, 1),
    (b'{"seq":1,"to":"A"}\n\n\n', 1),
    (b'{"seq":1,"to":"A"}\r\n{"seq":2,"to":"B"}\r\n', 2),
])
def test_benign_journal_tails_reconcile(raw, tail, tmp_path):
    C, m = _cycle_machine()
    path = tmp_path / "s.json"
    path.write_bytes(_good_state())
    Path(str(path) + ds.JOURNAL_SUFFIX).write_bytes(raw)
    sf = StateFile(path, m)
    entries = sf.journal()
    assert entries[-1]["seq"] == 3 and entries[-1]["reconciled_from_seq"] == tail
    assert ds.replay_journal(sf.journal_path) == ("B", 3)


def test_data_and_update_results_must_be_dicts(tmp_path):
    """Finding 7: non-dict data is refused before anything is written."""
    C, m = _cycle_machine()
    sf = StateFile(tmp_path / "s.json", m)
    with pytest.raises(TypeError):
        sf.create([1])
    assert not sf.exists()
    with pytest.raises(TypeError):
        sf.adopt(lambda d: ("A", ["not", "a", "dict"]))
    assert not sf.exists()
    sf.create({"n": 0})
    for bad in (0, [1], "x", 1.5):
        with pytest.raises(TypeError):
            sf.transition("go", update=lambda d, bad=bad: bad)
        assert sf.read().version == 1 and sf.read().data == {"n": 0}
    snap = sf.transition("go", update=lambda d: {})
    assert snap.data == {} and sf.read().data == {}
    assert sf.journal()[-1]["data_sha256"] == ds._data_sha256(sf.read().data)
    with pytest.raises(ValueError):
        sf.transition("back", update=lambda d: d.update(v=float("nan")))
    assert sf.read().version == 2


def test_atomic_write_through_symlink(tmp_path):
    """Finding 8: a symlinked target is written through, the link is kept."""
    real = tmp_path / "real.json"
    real.write_text("{}")
    link = tmp_path / "link.json"
    link.symlink_to("real.json")
    ds.atomic_write_json(link, {"x": 1})
    assert link.is_symlink()
    assert json.loads(real.read_text()) == {"x": 1}
    dangling = tmp_path / "dangling.json"
    dangling.symlink_to("created.json")
    ds.atomic_write_json(dangling, {"y": 2})
    assert dangling.is_symlink() and json.loads((tmp_path / "created.json").read_text()) == {"y": 2}
    assert _temps(tmp_path) == []


def test_machine_rejects_sinks_and_is_frozen(tmp_path):
    """Finding 9: non-terminal sinks are rejected; the table cannot be mutated."""
    class S(enum.Enum):
        X = 1
        STUCK = 2
        END = 3

    with pytest.raises(InvalidMachine, match="no outgoing edge"):
        Machine("m", S, S.X, {S.END}, {(S.X, "a"): S.STUCK, (S.X, "b"): S.END})
    with pytest.raises(InvalidMachine, match="no outgoing edge"):
        Machine("m", S, S.X, set(), {(S.X, "a"): S.STUCK, (S.STUCK, "b"): S.END})
    m = Machine("mut", S, S.X, {S.END},
                {(S.X, "b"): S.END, (S.X, "a"): S.STUCK, (S.STUCK, "c"): S.END})
    with pytest.raises(TypeError):
        m.transitions[(S.END, "zombie")] = S.X
    sf = StateFile(tmp_path / "s.json", m)
    sf.create()
    sf.transition("b")
    with pytest.raises(IllegalTransition):
        sf.transition("zombie")
    # A purely cyclic machine with no terminal state is still allowed.
    class T(enum.Enum):
        ON = 1
        OFF = 2
    Machine("toggle", T, T.OFF, set(), {(T.OFF, "on"): T.ON, (T.ON, "off"): T.OFF})


def test_long_target_names(tmp_path):
    """Finding 10: the temp name stays within NAME_MAX for any valid target."""
    for length in (200, 236, 240, 255):
        target = tmp_path / ("a" * length)
        ds.atomic_write_json(target, {"n": length})
        assert json.loads(target.read_text()) == {"n": length}
    assert _temps(tmp_path) == []
    prefix = ds._temp_prefix(str(tmp_path), "a" * 255, ".123.tmp")
    temp = prefix + "abcd1234.123.tmp"
    assert len(temp.encode()) <= 255 and "~" in prefix
    target, pid = ds._temp_owner(temp)
    assert pid == 123 and target.endswith("~")
