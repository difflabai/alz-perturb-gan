"""Durable, crash-safe state files with an explicit transition table.

This module is the shared base for making the repository's implicit state
machines explicit (fit lifecycle, output promotion, host activation, queue job
state and so on). It is stdlib-only on purpose, so research scripts can import
it without numpy or torch. It needs Python 3.11 or newer and Linux
(``fcntl.flock`` and ``/proc/<pid>/stat``).

Pieces
------
``atomic_write_bytes`` / ``atomic_write_json``
    Write to a temp file that is unique to this writer, fsync it,
    ``os.replace`` it over the target and fsync the parent directory. A reader
    sees either the old file or the new one, never a torn or missing file.
``FileLock``
    ``fcntl.flock`` on the sidecar ``<path>.lock``. Lock files are never
    deleted. The holder of an exclusive lock records itself in
    ``<path>.lock.owner``; that file is informational only and the kernel lock
    is the only liveness signal.
``Machine``
    A validated transition table over an ``Enum`` of states.
``StateFile``
    The JSON state document plus an append-only journal. Every transition runs
    under the file's lock, checks the edge and an optional guard, writes the
    new state atomically with ``version + 1`` and only then appends the
    journal line. The state file is authoritative; a journal that is behind
    after a crash is reconciled by the next write (or, opportunistically, on
    open when the lock is free).
``sweep_temps``
    Removes temp files left by dead writers. A temp in this module's own
    ``.<name>.<random>.<pid>.tmp`` format is never removed while its pid is
    alive on this host, however old it is. Other patterns (``*.partial*``)
    carry no trustworthy pid and are removed by age only.

Error contract of ``StateFile`` writes: every exception except
``CommittedWithError`` means nothing was written. ``CommittedWithError``
means the new state *was* committed (``os.replace`` ran) and a later step
(journal append, directory fsync) failed; its ``snapshot`` is the committed
state and the journal is reconciled by the next write. Retry loops should pass
``expect_version`` so a retry can never apply an event twice.

Locks are not reentrant. Re-acquiring an exclusive lock the same thread
already holds raises ``RuntimeError`` instead of deadlocking, and
``StateFile`` operations run inside ``with sf.lock():`` reuse the caller's
lock.

Invariants every ``StateFile`` keeps:

I1  ``version`` increases strictly by one per transition.
I2  After recovery the journal's last ``seq`` equals ``version``.
I3  The state file on disk always parses.
I4  One writer at a time per state file.
I5  No temp file is shared between processes (or between writers).

Fault injection: setting ``DURABLE_STATE_FAULT=<point>[:<nth>]`` makes the
process call ``os._exit(137)`` the nth time it reaches ``<point>``. The points
are ``after_tmp_write``, ``after_fsync``, ``after_replace`` (all three inside
atomic writes of state files) and ``after_journal``. Lock-owner records never
trigger a fault, so the count follows state writes only.
"""

from __future__ import annotations

import contextlib
import dataclasses
import enum
import errno
import fcntl
import hashlib
import json
import os
import socket
import tempfile
import threading
import time
import types

__all__ = [
    "ADOPT_EVENT",
    "CREATE_EVENT",
    "RECONCILE_EVENT",
    "SCHEMA",
    "AlreadyExists",
    "Busy",
    "CommittedWithError",
    "CorruptState",
    "DurableStateError",
    "FileLock",
    "GuardFailed",
    "IllegalTransition",
    "InvalidMachine",
    "Machine",
    "NoState",
    "Snapshot",
    "StateFile",
    "VersionConflict",
    "atomic_write_bytes",
    "atomic_write_json",
    "held_mode",
    "owner_status",
    "pid_starttime",
    "read_journal",
    "replay_journal",
    "sweep_temps",
]

SCHEMA = "durable_state/1"
ADOPT_EVENT = "__adopt__"
CREATE_EVENT = "__create__"
RECONCILE_EVENT = "reconciled"
RESERVED_EVENTS = frozenset({ADOPT_EVENT, CREATE_EVENT, RECONCILE_EVENT})
FAULT_ENV = "DURABLE_STATE_FAULT"
FAULT_EXIT_CODE = 137
JOURNAL_SUFFIX = ".journal.jsonl"
LOCK_SUFFIX = ".lock"
OWNER_SUFFIX = ".lock.owner"
TEMP_SUFFIX = ".tmp"
DEFAULT_SWEEP_PATTERNS = (".*.*" + TEMP_SUFFIX, "*.partial*")


# --------------------------------------------------------------------------
# Errors


class DurableStateError(Exception):
    """Base class for every error this module raises."""


class Busy(DurableStateError):
    """A non-blocking or timed lock attempt found the lock held.

    ``owner`` is the parsed ``<path>.lock.owner`` record with a ``status`` of
    ``alive``, ``stale`` or ``unknown`` added, or ``None`` when no exclusive
    holder recorded itself (for example a shared holder).
    """

    def __init__(self, path, owner):
        self.path = os.fspath(path)
        self.owner = owner
        super().__init__(f"lock busy: {self.path} (owner={owner!r})")


class IllegalTransition(DurableStateError):
    """The (state, event) pair is not in the machine's transition table."""


class GuardFailed(DurableStateError):
    """The guard rejected the transition; nothing was written."""


class VersionConflict(DurableStateError):
    """``expect_version`` did not match the version on disk."""


class InvalidMachine(DurableStateError, ValueError):
    """The transition table failed validation."""


class CorruptState(DurableStateError):
    """The state file or journal does not match this library's format."""


class NoState(DurableStateError):
    """The state file does not exist yet; call create() or adopt() first."""


class AlreadyExists(DurableStateError):
    """create() found an existing state file."""


class CommittedWithError(DurableStateError):
    """The write WAS committed, but a step after ``os.replace`` failed.

    ``snapshot`` is the committed state; ``__cause__`` is the original error
    (for example ``OSError(ENOSPC)`` from the journal append). Do not retry
    the event as if it had not happened. The journal line that is missing is
    written as a ``reconciled`` entry by the next write.
    """

    def __init__(self, snapshot, cause):
        self.snapshot = snapshot
        super().__init__(
            f"committed {snapshot.machine} v{snapshot.version} ({snapshot.state.name}) "
            f"but a post-commit step failed: {cause!r}"
        )


# --------------------------------------------------------------------------
# Fault injection

_fault_counts: dict[str, int] = {}


def _fault(name: str) -> None:
    """Exit with 137 at the configured occurrence of ``name``; else no-op."""
    spec = os.environ.get(FAULT_ENV)
    if not spec:
        return
    point, _, nth = spec.partition(":")
    if point != name:
        return
    count = _fault_counts.get(name, 0) + 1
    _fault_counts[name] = count
    if count == (int(nth) if nth else 1):
        os._exit(FAULT_EXIT_CODE)


# --------------------------------------------------------------------------
# Atomic writes


def _fsync_dir(path: str) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _default_mode() -> int:
    """0o666 under the process umask, read without changing the umask."""
    try:
        with open("/proc/self/status", "rb") as fh:
            for line in fh:
                if line.startswith(b"Umask:"):
                    return 0o666 & ~int(line.split()[1], 8)
    except (OSError, ValueError, IndexError):
        pass
    return 0o644


def _temp_prefix(parent: str, name: str, suffix: str) -> str:
    """``.<name>.``, with ``name`` shortened when the temp name would pass NAME_MAX.

    A shortened name ends in ``~`` so it cannot be mistaken for a real
    target; sweep_temps still recognises the temp by its pid.
    """
    try:
        name_max = os.pathconf(parent, "PC_NAME_MAX")
    except (OSError, ValueError):
        name_max = 255
    overhead = 2 + 8 + len(suffix.encode())  # '.' + name + '.' + 8 random + suffix
    raw = name.encode()
    if len(raw) + overhead <= name_max:
        return "." + name + "."
    keep = raw[: max(1, name_max - overhead - 1)].decode(errors="ignore")
    return "." + keep + "~."


def _atomic_write(path, data: bytes, *, durable: bool = True, faults: bool = True,
                  committed: list | None = None) -> None:
    # Write through symlinks: replace the link's final target, keep the link.
    path = os.path.realpath(os.fspath(path))
    parent = os.path.dirname(path)
    name = os.path.basename(path)
    try:
        mode = os.stat(path).st_mode & 0o7777
    except FileNotFoundError:
        mode = _default_mode()
    suffix = f".{os.getpid()}{TEMP_SUFFIX}"
    fd, tmp = tempfile.mkstemp(dir=parent, prefix=_temp_prefix(parent, name, suffix), suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as fh:
            os.fchmod(fh.fileno(), mode)
            fh.write(data)
            fh.flush()
            if faults:
                _fault("after_tmp_write")
            if durable:
                os.fsync(fh.fileno())
        if faults:
            _fault("after_fsync")
        os.replace(tmp, path)
        if committed is not None:
            committed.append(True)
        if faults:
            _fault("after_replace")
        if durable:
            _fsync_dir(parent)
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass


def atomic_write_bytes(path, data: bytes) -> None:
    """Replace ``path`` with ``data`` so readers never see a partial file.

    The temp file is ``.<name>.<random>.<pid>.tmp`` in the same directory, so
    concurrent writers never share one. The target keeps its mode, or gets
    0o666 under the umask when it is new. A symlinked ``path`` is written
    through: the link's final target is replaced and the link is kept. An
    ``OSError`` from the parent-directory fsync can surface after the new
    content is already in place.
    """
    _atomic_write(path, bytes(data))


def _dumps(obj) -> bytes:
    return (json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def atomic_write_json(path, obj) -> None:
    """``atomic_write_bytes`` of ``obj`` as indented, key-sorted JSON."""
    _atomic_write(path, _dumps(obj))


# --------------------------------------------------------------------------
# Process identity


def pid_starttime(pid: int) -> int | None:
    """Field 22 of ``/proc/<pid>/stat`` (start time in clock ticks), or None."""
    try:
        with open(f"/proc/{int(pid)}/stat", "rb") as fh:
            raw = fh.read()
    except OSError:
        return None
    # The command name (field 2) can contain spaces and parentheses; the
    # fields after the last ')' start at field 3.
    rest = raw[raw.rfind(b")") + 2 :].split()
    try:
        return int(rest[22 - 3])
    except (IndexError, ValueError):
        return None


def _pid_alive(pid) -> bool:
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False  # 0 and negatives address process groups, not a process
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (OverflowError, ValueError):
        return False
    return True


def _pid_max() -> int:
    try:
        with open("/proc/sys/kernel/pid_max", "rb") as fh:
            return int(fh.read())
    except (OSError, ValueError):
        return 4194304


def _process_start_epoch(pid: int) -> float | None:
    """Wall-clock start time of ``pid`` (about 1 s resolution), or None."""
    ticks = pid_starttime(pid)
    if ticks is None:
        return None
    try:
        with open("/proc/stat", "rb") as fh:
            for line in fh:
                if line.startswith(b"btime "):
                    return int(line.split()[1]) + ticks / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, IndexError):
        pass
    return None


def owner_status(info: dict | None) -> str:
    """``alive``, ``stale`` or ``unknown`` for a lock-owner record.

    A record is stale when its pid is gone, or when the pid exists but its
    start time differs (the pid was reused). A record from another host is
    ``unknown``. The kernel lock, not this status, decides who holds a lock.
    """
    if not info or "pid" not in info:
        return "unknown"
    if not isinstance(info["pid"], int) or isinstance(info["pid"], bool):
        return "unknown"
    if info.get("host") not in (None, socket.gethostname()):
        return "unknown"
    pid = info["pid"]
    if not _pid_alive(pid):
        return "stale"
    recorded = info.get("pid_starttime")
    current = pid_starttime(pid)
    if recorded is not None and current is not None and recorded != current:
        return "stale"
    return "alive"


def _identity() -> dict:
    return {"pid": os.getpid(), "host": socket.gethostname()}


# --------------------------------------------------------------------------
# Locks

# Locks this process holds, per thread: {(pid, thread id, realpath): [FileLock]}.
# Keyed by pid too, so a forked child does not inherit its parent's entries.
_held: dict[tuple, list] = {}
_held_mutex = threading.Lock()


def _held_key(lock_path: str) -> tuple:
    return (os.getpid(), threading.get_ident(), os.path.realpath(lock_path))


def held_mode(path) -> str | None:
    """``exclusive``, ``shared`` or None: this thread's FileLock on ``path``."""
    with _held_mutex:
        locks = _held.get(_held_key(os.fspath(path) + LOCK_SUFFIX), [])
        if any(lock.exclusive for lock in locks):
            return "exclusive"
        return "shared" if locks else None


class FileLock:
    """``fcntl.flock`` on ``<path>.lock``; usable as a context manager.

    ``exclusive=False`` takes a shared lock. ``blocking=False`` raises
    ``Busy`` at once when the lock is held; ``timeout`` (seconds) polls and
    raises ``Busy`` when it runs out. The lock is tied to this object's open
    file description, so two FileLock objects exclude each other even inside
    one process. Locks are not reentrant: a blocking, untimed acquire of a
    path for which this thread already holds an exclusive lock (or of an
    exclusive lock while it holds a shared one) would deadlock, so it raises
    ``RuntimeError`` instead. Non-blocking and timed attempts raise ``Busy``
    as usual, and other threads block or get ``Busy`` as usual.

    The lock file itself is never deleted (deleting it would let two
    processes lock different inodes). An exclusive holder writes
    ``{pid, pid_starttime, host, acquired_at, purpose}`` to
    ``<path>.lock.owner`` and removes it again on a clean release.
    """

    def __init__(self, path, exclusive=True, blocking=True, timeout=None, purpose=None,
                 record_owner=True):
        self.path = os.fspath(path)
        self.lock_path = self.path + LOCK_SUFFIX
        self.owner_path = self.path + OWNER_SUFFIX
        self.exclusive = exclusive
        self.blocking = blocking
        self.timeout = timeout
        self.purpose = purpose
        self.record_owner = record_owner and exclusive
        self._fd: int | None = None
        self._key: tuple | None = None

    @property
    def held(self) -> bool:
        return self._fd is not None

    def read_owner(self) -> dict | None:
        """The owner record with ``status`` added, or None if there is none."""
        try:
            with open(self.owner_path, "rb") as fh:
                info = json.loads(fh.read())
        except (FileNotFoundError, ValueError):
            return None
        if not isinstance(info, dict):
            return None
        info["status"] = owner_status(info)
        return info

    def acquire(self) -> "FileLock":
        if self._fd is not None:
            raise RuntimeError(f"FileLock already held: {self.lock_path}")
        key = _held_key(self.lock_path)
        with _held_mutex:
            mine = _held.get(key, [])
            would_wait_forever = self.blocking and self.timeout is None
            if would_wait_forever and mine and (
                    self.exclusive or any(lock.exclusive for lock in mine)):
                raise RuntimeError(
                    f"nested lock on {self.lock_path} in one thread would deadlock "
                    f"(held: {[('exclusive' if l.exclusive else 'shared') for l in mine]}, "
                    f"wanted: {'exclusive' if self.exclusive else 'shared'})"
                )
        try:
            fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, _default_mode())
        except PermissionError as denied:
            # A read-only consumer can still flock an existing lock file.
            try:
                fd = os.open(self.lock_path, os.O_RDONLY | os.O_CLOEXEC)
            except FileNotFoundError:
                raise denied from None
        op = fcntl.LOCK_EX if self.exclusive else fcntl.LOCK_SH
        try:
            if self.blocking and self.timeout is None:
                fcntl.flock(fd, op)
            else:
                deadline = None if not self.blocking else time.monotonic() + self.timeout
                delay = 0.001
                while True:
                    try:
                        fcntl.flock(fd, op | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        if deadline is None or time.monotonic() >= deadline:
                            raise Busy(self.path, self.read_owner()) from None
                        time.sleep(min(delay, max(0.0, deadline - time.monotonic())))
                        delay = min(delay * 2, 0.05)
        except BaseException:
            os.close(fd)
            raise
        self._fd = fd
        with _held_mutex:
            _held.setdefault(key, []).append(self)
        self._key = key
        if self.record_owner:
            info = _identity()
            info.update(
                pid_starttime=pid_starttime(info["pid"]),
                acquired_at=time.time(),
                purpose=self.purpose,
            )
            try:
                _atomic_write(self.owner_path, _dumps(info), durable=False, faults=False)
            except BaseException:
                self.release()
                raise
        return self

    def release(self) -> None:
        fd, self._fd = self._fd, None
        if fd is None:
            return
        with _held_mutex:
            locks = _held.get(self._key, [])
            if self in locks:
                locks.remove(self)
            if not locks:
                _held.pop(self._key, None)
        try:
            if self.record_owner:
                # Still holding the lock, so no other holder's record is lost.
                try:
                    os.unlink(self.owner_path)
                except FileNotFoundError:
                    pass
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    def __enter__(self) -> "FileLock":
        return self.acquire()

    def __exit__(self, *exc) -> None:
        self.release()


# --------------------------------------------------------------------------
# Machines


class Machine:
    """A validated transition table.

    ``transitions`` maps ``(from_state, event) -> to_state``. ``events`` is
    the declared event vocabulary (defaults to the events the table uses).
    ``adopt_entries`` lists states that only legacy adoption can reach; they
    count as roots for the reachability check. Validation runs at
    construction and raises ``InvalidMachine``: unknown states, undeclared or
    reserved events, edges out of a terminal state, unreachable states, and
    non-terminal sinks (a state with no way out must be declared terminal).
    The table is frozen after validation (a read-only mapping), so it cannot
    drift from what was validated.
    """

    def __init__(self, name, states, initial, terminal, transitions, events=None,
                 adopt_entries=()):
        self.name = str(name)
        self.states = states
        self.initial = initial
        self.terminal = frozenset(terminal)
        self.transitions = types.MappingProxyType(dict(transitions))
        self.events = (
            frozenset(events)
            if events is not None
            else frozenset(event for _, event in self.transitions)
        )
        self.adopt_entries = frozenset(adopt_entries)
        self.validate()

    def validate(self) -> None:
        if not (isinstance(self.states, type) and issubclass(self.states, enum.Enum)):
            raise InvalidMachine(f"{self.name}: states must be an Enum class")
        members = set(self.states)
        if not members:
            raise InvalidMachine(f"{self.name}: no states")
        if self.initial not in members:
            raise InvalidMachine(f"{self.name}: initial {self.initial!r} is not a state")
        for group, label in ((self.terminal, "terminal"), (self.adopt_entries, "adopt entry")):
            for state in group:
                if state not in members:
                    raise InvalidMachine(f"{self.name}: {label} {state!r} is not a state")
        bad = sorted(self.events & RESERVED_EVENTS)
        if bad:
            raise InvalidMachine(f"{self.name}: reserved event names {bad}")
        for (src, event), dst in self.transitions.items():
            if src not in members or dst not in members:
                raise InvalidMachine(f"{self.name}: edge {src!r} -{event}-> {dst!r} uses an unknown state")
            if not isinstance(event, str) or event not in self.events:
                raise InvalidMachine(f"{self.name}: event {event!r} is not declared")
            if src in self.terminal:
                raise InvalidMachine(f"{self.name}: terminal state {src.name} has outgoing edge {event!r}")
        seen = {self.initial} | set(self.adopt_entries)
        frontier = list(seen)
        while frontier:
            src = frontier.pop()
            for (s, _), dst in self.transitions.items():
                if s == src and dst not in seen:
                    seen.add(dst)
                    frontier.append(dst)
        unreachable = sorted(s.name for s in members - seen)
        if unreachable:
            raise InvalidMachine(f"{self.name}: unreachable states {unreachable}")
        sources = {src for (src, _) in self.transitions}
        sinks = sorted(s.name for s in members - self.terminal - sources)
        if sinks:
            raise InvalidMachine(
                f"{self.name}: non-terminal states with no outgoing edge {sinks} "
                "(declare them terminal)"
            )

    def state(self, value) -> enum.Enum:
        """Coerce a member or a member name to a state of this machine."""
        if isinstance(value, self.states):
            return value
        try:
            return self.states[value]
        except KeyError:
            raise CorruptState(f"{self.name}: unknown state {value!r}") from None

    def next_state(self, state, event):
        """The target of ``(state, event)``, or None when there is no edge."""
        return self.transitions.get((self.state(state), event))

    def is_terminal(self, state) -> bool:
        return self.state(state) in self.terminal

    def allowed_events(self, state) -> list[str]:
        state = self.state(state)
        return sorted(event for (s, event) in self.transitions if s == state)


# --------------------------------------------------------------------------
# State files


@dataclasses.dataclass(frozen=True)
class Snapshot:
    """One parsed, validated state document."""

    machine: str
    state: enum.Enum
    version: int
    updated_at: float
    owner: dict
    data: dict


def _data_sha256(data) -> str:
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _reject_constant(name):
    raise ValueError(f"non-finite number {name} is not allowed")


def _loads(raw: bytes, where: str):
    """Strict JSON (no NaN/Infinity); any parse failure is ``CorruptState``."""
    try:
        return json.loads(raw, parse_constant=_reject_constant)
    except (ValueError, TypeError, RecursionError) as exc:  # incl. JSON/Unicode errors
        raise CorruptState(f"{where}: invalid JSON: {exc}") from None


def _journal_entry(line: bytes, where: str) -> dict:
    entry = _loads(line, where)
    if not isinstance(entry, dict):
        raise CorruptState(f"{where}: journal line is not an object")
    seq = entry.get("seq")
    if not isinstance(seq, int) or isinstance(seq, bool) or seq < 1:
        raise CorruptState(f"{where}: journal line has bad seq {seq!r}")
    if not isinstance(entry.get("to"), str):
        raise CorruptState(f"{where}: journal line has bad 'to' {entry.get('to')!r}")
    return entry


def read_journal(journal_path) -> list[dict]:
    """Every complete journal line, parsed. A torn final line is skipped.

    A complete line that is not a valid entry raises ``CorruptState``.
    """
    journal_path = os.fspath(journal_path)
    try:
        with open(journal_path, "rb") as fh:
            raw = fh.read()
    except FileNotFoundError:
        return []
    lines = raw.split(b"\n")
    if lines and lines[-1] != b"":
        lines = lines[:-1]  # incomplete tail written by a crashed appender
    return [_journal_entry(line, f"{journal_path}:{i + 1}")
            for i, line in enumerate(lines) if line.strip()]


def replay_journal(journal_path) -> tuple[str | None, int]:
    """(state name, last seq) obtained by replaying the journal.

    Raises ``CorruptState`` for a malformed line or a seq that does not
    increase.
    """
    state, seq = None, 0
    for entry in read_journal(journal_path):
        if entry["seq"] <= seq:
            raise CorruptState(f"{journal_path}: seq {entry['seq']} after {seq}")
        state, seq = entry["to"], entry["seq"]
    return state, seq


def _as_dict(value, what: str) -> dict:
    """A deep, JSON-clean copy of ``value``; it must be a dict (None means {})."""
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise TypeError(f"{what} must be a dict, got {type(value).__name__}")
    return json.loads(json.dumps(value, allow_nan=False))


class StateFile:
    """A JSON state document with a transition table and a journal.

    The file holds ``{schema, machine, state, version, updated_at,
    owner: {pid, host}, data}``. ``<path>.journal.jsonl`` gets one line per
    accepted transition: ``{seq, from, event, to, at, pid, host,
    data_sha256}``.

    Construction never blocks and never fails on a read-only directory. With
    ``recover=True`` (the default) it reconciles a lagging journal only when
    the lock can be taken at once and the files are writable; otherwise the
    next write reconciles. Use ``recover()`` to reconcile explicitly.

    Every write (``create``, ``adopt``, ``transition``) takes the exclusive
    lock, unless this thread already holds it through ``sf.lock()``, in which
    case it runs inside the caller's lock. See the module docstring for the
    error contract (``CommittedWithError``).
    """

    def __init__(self, path, machine: Machine, *, recover: bool = True):
        self.path = os.fspath(path)
        self.machine = machine
        self.journal_path = self.path + JOURNAL_SUFFIX
        if recover and os.path.exists(self.path):
            self._try_recover()

    # -- reading ------------------------------------------------------------

    def _parse(self, raw: bytes) -> Snapshot:
        doc = _loads(raw, self.path)
        if not isinstance(doc, dict) or doc.get("schema") != SCHEMA:
            raise CorruptState(f"{self.path}: schema is not {SCHEMA}")
        if doc.get("machine") != self.machine.name:
            raise CorruptState(f"{self.path}: machine {doc.get('machine')!r} != {self.machine.name!r}")
        version = doc.get("version")
        if not isinstance(version, int) or isinstance(version, bool) or version < 1:
            raise CorruptState(f"{self.path}: bad version {version!r}")
        state = doc.get("state")
        if not isinstance(state, str):
            raise CorruptState(f"{self.path}: bad state {state!r}")
        updated_at = doc.get("updated_at")
        if updated_at is not None and (not isinstance(updated_at, (int, float))
                                       or isinstance(updated_at, bool)):
            raise CorruptState(f"{self.path}: bad updated_at {updated_at!r}")
        owner = doc.get("owner", {})
        data = doc.get("data", {})
        for value, label in ((owner, "owner"), (data, "data")):
            if not isinstance(value, dict):
                raise CorruptState(f"{self.path}: {label} is not an object")
        return Snapshot(
            machine=doc["machine"],
            state=self.machine.state(state),
            version=version,
            updated_at=updated_at,
            owner=owner,
            data=data,
        )

    def read(self) -> Snapshot | None:
        """The current snapshot without taking the lock; None if absent.

        The file is only ever replaced atomically, so this never sees a torn
        document.
        """
        try:
            with open(self.path, "rb") as fh:
                raw = fh.read()
        except FileNotFoundError:
            return None
        return self._parse(raw)

    def exists(self) -> bool:
        return os.path.exists(self.path)

    def journal(self) -> list[dict]:
        return read_journal(self.journal_path)

    def lock(self, purpose=None, **kwargs) -> FileLock:
        return FileLock(self.path, purpose=purpose or self.machine.name, **kwargs)

    @contextlib.contextmanager
    def _write_lock(self, purpose: str):
        """The exclusive lock, or the caller's when this thread already holds it."""
        mode = held_mode(self.path)
        if mode == "exclusive":
            yield
            return
        if mode == "shared":
            raise RuntimeError(f"{self.path}: cannot write while this thread holds a shared lock")
        with self.lock(purpose=purpose):
            yield

    # -- journal ------------------------------------------------------------

    def _journal_tail_seq(self) -> int:
        """Last complete seq; truncates an incomplete final line. Lock held.

        Opens the journal read-only unless a torn tail has to be cut, so a
        read-only journal that needs no repair does not block anything.
        """
        try:
            fd = os.open(self.journal_path, os.O_RDONLY | os.O_CLOEXEC)
        except FileNotFoundError:
            return 0
        try:
            size = os.fstat(fd).st_size
            if size == 0:
                return 0
            chunk = 4096
            while True:
                start = max(0, size - chunk)
                buf = os.pread(fd, size - start, start)
                if not buf.endswith(b"\n"):
                    cut = buf.rfind(b"\n")
                    if cut < 0 and start > 0:
                        chunk *= 2
                        continue
                    new_size = start + cut + 1
                    self._truncate_journal(new_size)
                    size = new_size
                    if size == 0:
                        return 0
                    continue
                lines = buf.rstrip(b"\n").split(b"\n")
                if len(lines) >= 2 or start == 0:
                    return _journal_entry(lines[-1], f"{self.journal_path}:last")["seq"]
                chunk *= 2
        finally:
            os.close(fd)

    def _truncate_journal(self, size: int) -> None:
        fd = os.open(self.journal_path, os.O_WRONLY | os.O_CLOEXEC)
        try:
            os.ftruncate(fd, size)
            os.fsync(fd)
        finally:
            os.close(fd)

    def _open_journal(self) -> tuple[int, bool]:
        """(append fd, created). Opened before a commit so a journal that
        cannot be written fails the write while nothing has changed yet."""
        created = not os.path.exists(self.journal_path)
        fd = os.open(self.journal_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_CLOEXEC,
                     _default_mode())
        return fd, created

    def _append_journal(self, entry: dict, opened: tuple[int, bool] | None = None) -> None:
        fd, created = opened if opened is not None else self._open_journal()
        try:
            line = (json.dumps(entry, sort_keys=True, allow_nan=False) + "\n").encode()
            view = memoryview(line)
            while view:
                view = view[os.write(fd, view):]
            os.fsync(fd)
        finally:
            os.close(fd)
        if created:
            _fsync_dir(os.path.dirname(os.path.abspath(self.journal_path)))
        _fault("after_journal")

    def _reconcile_locked(self, snap: Snapshot) -> None:
        tail = self._journal_tail_seq()
        if tail > snap.version:
            raise CorruptState(
                f"{self.journal_path}: journal seq {tail} is ahead of state version {snap.version}"
            )
        if tail < snap.version:
            # Rare path: validate the whole journal before extending it, so
            # mid-journal corruption is reported instead of papered over.
            replay_journal(self.journal_path)
            ident = _identity()
            self._append_journal({
                "seq": snap.version,
                "from": None,
                "event": RECONCILE_EVENT,
                "to": snap.state.name,
                "at": time.time(),
                "pid": ident["pid"],
                "host": ident["host"],
                "data_sha256": _data_sha256(snap.data),
                "reconciled_from_seq": tail,
            })

    def recover(self) -> Snapshot | None:
        """Take the lock (blocking) and bring the journal level with the state (I2)."""
        with self._write_lock(f"{self.machine.name}:recover"):
            snap = self.read()
            if snap is not None:
                self._reconcile_locked(snap)
            return snap

    def _try_recover(self) -> None:
        """recover() if it can run without blocking and the files are writable."""
        if held_mode(self.path) is not None:
            if held_mode(self.path) == "exclusive":
                self.recover()
            return
        try:
            with self.lock(purpose=f"{self.machine.name}:recover", blocking=False):
                snap = self.read()
                if snap is not None:
                    self._reconcile_locked(snap)
        except Busy:
            return  # someone is working on it; their write reconciles
        except PermissionError:
            return  # read-only consumer
        except OSError as exc:
            if exc.errno == errno.EROFS:
                return
            raise

    # -- writing ------------------------------------------------------------

    def _write_locked(self, state, version: int, data: dict, event: str, from_state,
                      extra: dict | None = None) -> Snapshot:
        ident = _identity()
        now = time.time()
        doc = {
            "schema": SCHEMA,
            "machine": self.machine.name,
            "state": state.name,
            "version": version,
            "updated_at": now,
            "owner": ident,
            "data": data,
        }
        payload = _dumps(doc)  # serialise before touching the disk
        entry = {
            "seq": version,
            "from": from_state.name if from_state is not None else None,
            "event": event,
            "to": state.name,
            "at": now,
            "pid": ident["pid"],
            "host": ident["host"],
            "data_sha256": _data_sha256(data),
        }
        if extra:
            entry.update(extra)
        snap = Snapshot(self.machine.name, state, version, now, ident, data)
        opened = self._open_journal()  # fails here, before the commit, if read-only
        committed: list = []
        try:
            _atomic_write(self.path, payload, committed=committed)
        except Exception as exc:
            os.close(opened[0])
            if committed:  # the post-replace directory fsync failed
                raise CommittedWithError(snap, exc) from exc
            raise
        except BaseException:
            os.close(opened[0])
            raise
        try:
            self._append_journal(entry, opened)
        except Exception as exc:
            raise CommittedWithError(snap, exc) from exc
        return snap

    def create(self, data=None, *, exist_ok: bool = False) -> Snapshot:
        """NO_FILE -create-> initial, at version 1. ``data`` must be a dict."""
        data = _as_dict(data, "create data")
        with self._write_lock(f"{self.machine.name}:{CREATE_EVENT}"):
            snap = self.read()
            if snap is not None:
                if exist_ok:
                    self._reconcile_locked(snap)
                    return snap
                raise AlreadyExists(self.path)
            self._check_fresh_journal()
            return self._write_locked(self.machine.initial, 1, data, CREATE_EVENT, None)

    def adopt(self, infer_legacy, legacy_dir=None, data=None) -> Snapshot:
        """NO_FILE -adopt-> S0, where S0 is inferred from legacy files.

        ``infer_legacy(legacy_dir)`` must only read; it returns
        ``(state, evidence_dict)``. The library writes the new state file and
        journal next to the legacy files and never touches them. When the
        state file already exists this returns it unchanged, so callers can
        adopt unconditionally at start-up.
        """
        legacy_dir = os.fspath(legacy_dir) if legacy_dir is not None else (
            os.path.dirname(os.path.abspath(self.path)))
        body = _as_dict(data, "adopt data")
        with self._write_lock(f"{self.machine.name}:{ADOPT_EVENT}"):
            snap = self.read()
            if snap is not None:
                self._reconcile_locked(snap)
                return snap
            self._check_fresh_journal()
            state, evidence = infer_legacy(legacy_dir)
            state = self.machine.state(state)
            evidence = _as_dict(evidence, "adopt evidence")
            body["adopted"] = {"evidence": evidence, "legacy_dir": legacy_dir}
            return self._write_locked(state, 1, body, ADOPT_EVENT, None,
                                      extra={"evidence": evidence})

    def _check_fresh_journal(self) -> None:
        if self._journal_tail_seq() != 0:
            raise CorruptState(
                f"{self.journal_path}: journal exists but the state file is missing"
            )

    def transition(self, event, guard=None, update=None, expect_version=None) -> Snapshot:
        """Apply ``event`` under the lock and return the new snapshot.

        ``guard(snapshot)`` must return truthy or ``GuardFailed`` is raised.
        ``update(data_copy)`` returns the new data dict (or mutates the copy
        and returns None); any other return value raises ``TypeError``.
        Raises ``IllegalTransition`` when the current state has no ``event``
        edge and ``VersionConflict`` when ``expect_version`` is given and
        differs. Nothing is written when any of these raise.

        ``CommittedWithError`` is the one exception raised *after* the commit:
        the new state is on disk (see its ``snapshot``). Retry loops should
        pass ``expect_version`` so a retry cannot apply the event twice.
        """
        with self._write_lock(f"{self.machine.name}:{event}"):
            snap = self.read()
            if snap is None:
                raise NoState(self.path)
            self._reconcile_locked(snap)
            if expect_version is not None and expect_version != snap.version:
                raise VersionConflict(
                    f"{self.path}: expected version {expect_version}, found {snap.version}"
                )
            target = self.machine.transitions.get((snap.state, event))
            if target is None:
                raise IllegalTransition(
                    f"{self.machine.name}: no edge {snap.state.name} -{event}-> "
                    f"(allowed: {self.machine.allowed_events(snap.state)})"
                )
            if guard is not None and not guard(snap):
                raise GuardFailed(f"{self.machine.name}: guard rejected {snap.state.name} -{event}->")
            data = _as_dict(snap.data, "state data")
            if update is not None:
                result = update(data)
                data = _as_dict(result if result is not None else data, "update() result")
            return self._write_locked(target, snap.version + 1, data, event, snap.state)


# --------------------------------------------------------------------------
# Temp sweeping


def _glob_match(name: str, pattern: str) -> bool:
    """fnmatch-style matching of '*' and '?' (no character classes)."""
    n, p = 0, 0
    star_p, star_n = -1, 0
    while n < len(name):
        if p < len(pattern) and pattern[p] in ("?", name[n]) and pattern[p] != "*":
            n += 1
            p += 1
        elif p < len(pattern) and pattern[p] == "*":
            star_p, star_n = p, n
            p += 1
        elif star_p >= 0:
            p = star_p + 1
            star_n += 1
            n = star_n
        else:
            return False
    while p < len(pattern) and pattern[p] == "*":
        p += 1
    return p == len(pattern)


_RANDOM_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789_")


def _own_temp(name: str) -> tuple[str, int] | None:
    """(target name, writer pid) for this module's ``.<target>.<rand8>.<pid>.tmp``.

    Only this exact format carries a pid that sweep_temps trusts: the random
    field must be 8 ``mkstemp`` characters and the pid must lie in
    ``1..pid_max``. Anything else returns None.
    """
    if not (name.startswith(".") and name.endswith(TEMP_SUFFIX)):
        return None
    parts = name[1:-len(TEMP_SUFFIX)].split(".")
    if len(parts) < 3:
        return None
    target, rand, pid = ".".join(parts[:-2]), parts[-2], parts[-1]
    if not target or len(rand) != 8 or not set(rand) <= _RANDOM_CHARS:
        return None
    if not (pid.isascii() and pid.isdigit()) or len(pid) > 10:
        return None
    pid_n = int(pid)
    if not 1 <= pid_n <= _pid_max():
        return None
    return target, pid_n


def _partial_target(name: str) -> str | None:
    """Target of a ``*.partial*`` temp, for the lock check.

    ``.<stem>.partial.<ext>`` (the hidden form runtime/hardening writes) maps
    to ``<stem>.<ext>``; ``<name>.partial`` and ``<name>.partial.<suffix>`` map
    to ``<name>``.
    """
    head, sep, tail = name.partition(".partial")
    if not sep or not head:
        return None
    if head.startswith(".") and tail.startswith(".") and len(head) > 1:
        return head[1:] + tail
    return head


def _temp_owner(name: str) -> tuple[str | None, int | None]:
    """(target file name, trusted writer pid or None) for a temp-file name."""
    own = _own_temp(name)
    if own is not None:
        return own
    if ".partial" in name:
        return _partial_target(name), None
    return None, None


def _lock_file_for(target: str) -> str:
    """The lock file guarding ``target``.

    A ``<x>.lock.owner`` record is written while ``<x>.lock`` is held, so its
    temp is guarded by ``<x>.lock``; every other target by ``<target>.lock``.
    """
    if target.endswith(OWNER_SUFFIX):
        return target[: -len(OWNER_SUFFIX)] + LOCK_SUFFIX
    return target + LOCK_SUFFIX


def _target_lock_free(directory: str, target: str | None) -> bool:
    if not target:
        return True
    try:
        fd = os.open(os.path.join(directory, _lock_file_for(target)), os.O_RDONLY | os.O_CLOEXEC)
    except FileNotFoundError:
        return True  # nobody has ever locked the target
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        fcntl.flock(fd, fcntl.LOCK_UN)
        return True
    finally:
        os.close(fd)


# A process that started more than this long after a temp's mtime cannot have
# written it (start time is known to about a second), so its pid was reused.
_PID_REUSE_SLACK_S = 2.0


def _writer_alive(pid: int, mtime: float) -> bool:
    if not _pid_alive(pid):
        return False
    started = _process_start_epoch(pid)
    return started is None or started <= mtime + _PID_REUSE_SLACK_S


def sweep_temps(directory, patterns=DEFAULT_SWEEP_PATTERNS, min_age_s=3600.0, dry_run=False):
    """Delete orphaned temp files in ``directory`` (not recursive).

    A file in this module's own ``.<target>.<rand8>.<pid>.tmp`` format is
    removed only when its writer pid is dead on this host (or was reused by
    a process that started after the file was written). It is never removed
    while that writer is alive, however old the file is. A file matching
    another pattern carries no trustworthy pid (digits in ``out.partial.20261120``
    are not a pid) and is removed when it is older than ``min_age_s``. Either
    way the file is kept while the lock on its target is held. Lock files
    and lock-owner records are never touched. Returns the sorted list of
    paths removed (or that would be removed when ``dry_run`` is true).

    The target-lock probe takes the lock for an instant, so a concurrent
    non-blocking acquirer can see a spurious ``Busy``.
    """
    directory = os.fspath(directory)
    now = time.time()
    victims = []
    for entry in os.scandir(directory):
        name = entry.name
        if name.endswith(LOCK_SUFFIX) or name.endswith(OWNER_SUFFIX):
            continue
        if not any(_glob_match(name, pattern) for pattern in patterns):
            continue
        try:
            if not entry.is_file(follow_symlinks=False):
                continue
            mtime = entry.stat(follow_symlinks=False).st_mtime
        except FileNotFoundError:
            continue
        target, pid = _temp_owner(name)
        if pid is not None:
            if _writer_alive(pid, mtime):
                continue
        elif now - mtime < min_age_s:
            continue
        if not _target_lock_free(directory, target):
            continue
        victims.append(entry.path)
    victims.sort()
    if dry_run:
        return victims
    removed = []
    for path in victims:
        try:
            os.unlink(path)
            removed.append(path)
        except FileNotFoundError:
            pass
    return removed
