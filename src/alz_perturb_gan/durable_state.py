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
    after a crash is reconciled on open.
``sweep_temps``
    Removes temp files left by dead writers, never ones a live writer owns.

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

import dataclasses
import enum
import fcntl
import hashlib
import json
import os
import socket
import tempfile
import time

__all__ = [
    "ADOPT_EVENT",
    "CREATE_EVENT",
    "RECONCILE_EVENT",
    "SCHEMA",
    "AlreadyExists",
    "Busy",
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


def _atomic_write(path, data: bytes, *, durable: bool = True, faults: bool = True) -> None:
    path = os.fspath(path)
    parent = os.path.dirname(os.path.abspath(path))
    name = os.path.basename(path)
    try:
        mode = os.stat(path).st_mode & 0o7777
    except FileNotFoundError:
        mode = _default_mode()
    fd, tmp = tempfile.mkstemp(
        dir=parent, prefix="." + name + ".", suffix=f".{os.getpid()}{TEMP_SUFFIX}"
    )
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
    0o666 under the umask when it is new.
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


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def owner_status(info: dict | None) -> str:
    """``alive``, ``stale`` or ``unknown`` for a lock-owner record.

    A record is stale when its pid is gone, or when the pid exists but its
    start time differs (the pid was reused). A record from another host is
    ``unknown``. The kernel lock, not this status, decides who holds a lock.
    """
    if not info or "pid" not in info:
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


class FileLock:
    """``fcntl.flock`` on ``<path>.lock``; usable as a context manager.

    ``exclusive=False`` takes a shared lock. ``blocking=False`` raises
    ``Busy`` at once when the lock is held; ``timeout`` (seconds) polls and
    raises ``Busy`` when it runs out. The lock is tied to this object's open
    file description, so two FileLock objects exclude each other even inside
    one process: do not nest two locks on the same path.

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
        fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, _default_mode())
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
    construction and raises ``InvalidMachine``.
    """

    def __init__(self, name, states, initial, terminal, transitions, events=None,
                 adopt_entries=()):
        self.name = str(name)
        self.states = states
        self.initial = initial
        self.terminal = frozenset(terminal)
        self.transitions = dict(transitions)
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


def read_journal(journal_path) -> list[dict]:
    """Every complete journal line, parsed. A torn final line is skipped."""
    try:
        with open(journal_path, "rb") as fh:
            raw = fh.read()
    except FileNotFoundError:
        return []
    lines = raw.split(b"\n")
    if lines and lines[-1] != b"":
        lines = lines[:-1]  # incomplete tail written by a crashed appender
    return [json.loads(line) for line in lines if line.strip()]


def replay_journal(journal_path) -> tuple[str | None, int]:
    """(state name, last seq) obtained by replaying the journal."""
    state, seq = None, 0
    for entry in read_journal(journal_path):
        if entry["seq"] <= seq:
            raise CorruptState(f"{journal_path}: seq {entry['seq']} after {seq}")
        state, seq = entry["to"], entry["seq"]
    return state, seq


class StateFile:
    """A JSON state document with a transition table and a journal.

    The file holds ``{schema, machine, state, version, updated_at,
    owner: {pid, host}, data}``. ``<path>.journal.jsonl`` gets one line per
    accepted transition: ``{seq, from, event, to, at, pid, host,
    data_sha256}``. Opening an existing state file reconciles a journal that
    a crash left one line behind.
    """

    def __init__(self, path, machine: Machine, *, recover: bool = True):
        self.path = os.fspath(path)
        self.machine = machine
        self.journal_path = self.path + JOURNAL_SUFFIX
        if recover and os.path.exists(self.path):
            self.recover()

    # -- reading ------------------------------------------------------------

    def _parse(self, raw: bytes) -> Snapshot:
        try:
            doc = json.loads(raw)
        except ValueError as exc:
            raise CorruptState(f"{self.path}: invalid JSON: {exc}") from None
        if not isinstance(doc, dict) or doc.get("schema") != SCHEMA:
            raise CorruptState(f"{self.path}: schema is not {SCHEMA}")
        if doc.get("machine") != self.machine.name:
            raise CorruptState(f"{self.path}: machine {doc.get('machine')!r} != {self.machine.name!r}")
        version = doc.get("version")
        if not isinstance(version, int) or isinstance(version, bool) or version < 1:
            raise CorruptState(f"{self.path}: bad version {version!r}")
        return Snapshot(
            machine=doc["machine"],
            state=self.machine.state(doc.get("state")),
            version=version,
            updated_at=doc.get("updated_at"),
            owner=doc.get("owner") or {},
            data=doc.get("data") or {},
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

    # -- journal ------------------------------------------------------------

    def _journal_tail_seq(self) -> int:
        """Last complete seq; truncates an incomplete final line. Lock held."""
        try:
            fd = os.open(self.journal_path, os.O_RDWR | os.O_CLOEXEC)
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
                    os.ftruncate(fd, new_size)
                    os.fsync(fd)
                    size = new_size
                    if size == 0:
                        return 0
                    continue
                lines = buf.rstrip(b"\n").split(b"\n")
                if len(lines) >= 2 or start == 0:
                    return int(json.loads(lines[-1])["seq"])
                chunk *= 2
        finally:
            os.close(fd)

    def _append_journal(self, entry: dict) -> None:
        line = (json.dumps(entry, sort_keys=True, allow_nan=False) + "\n").encode()
        created = not os.path.exists(self.journal_path)
        fd = os.open(self.journal_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_CLOEXEC,
                     _default_mode())
        try:
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
        """Take the lock and bring the journal level with the state (I2)."""
        with self.lock(purpose=f"{self.machine.name}:recover"):
            snap = self.read()
            if snap is not None:
                self._reconcile_locked(snap)
            return snap

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
        atomic_write_bytes(self.path, payload)
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
        self._append_journal(entry)
        return Snapshot(self.machine.name, state, version, now, ident, data)

    @staticmethod
    def _copy(data) -> dict:
        return json.loads(json.dumps(data if data is not None else {}, allow_nan=False))

    def create(self, data=None, *, exist_ok: bool = False) -> Snapshot:
        """NO_FILE -create-> initial, at version 1."""
        with self.lock(purpose=f"{self.machine.name}:{CREATE_EVENT}"):
            snap = self.read()
            if snap is not None:
                if exist_ok:
                    self._reconcile_locked(snap)
                    return snap
                raise AlreadyExists(self.path)
            self._check_fresh_journal()
            return self._write_locked(self.machine.initial, 1, self._copy(data), CREATE_EVENT, None)

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
        with self.lock(purpose=f"{self.machine.name}:{ADOPT_EVENT}"):
            snap = self.read()
            if snap is not None:
                self._reconcile_locked(snap)
                return snap
            self._check_fresh_journal()
            state, evidence = infer_legacy(legacy_dir)
            state = self.machine.state(state)
            evidence = self._copy(evidence)
            body = self._copy(data)
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
        and returns None). Raises ``IllegalTransition`` when the current state
        has no ``event`` edge and ``VersionConflict`` when ``expect_version``
        is given and differs. Nothing is written when any of these raise.
        """
        with self.lock(purpose=f"{self.machine.name}:{event}"):
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
            data = self._copy(snap.data)
            if update is not None:
                result = update(data)
                data = self._copy(result if result is not None else data)
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


def _temp_owner(name: str) -> tuple[str | None, int | None]:
    """(target file name, embedded writer pid) for a temp-file name."""
    parts = name.split(".")
    if name.startswith(".") and name.endswith(TEMP_SUFFIX) and len(parts) >= 5 and parts[-2].isdigit():
        # .<target>.<random>.<pid>.tmp as written by atomic_write_bytes
        return ".".join(parts[1:-3]), int(parts[-2])
    if ".partial" in name:
        target = name[: name.index(".partial")]
        digits = [p for p in name[len(target):].split(".") if p.isdigit()]
        return target or None, int(digits[-1]) if digits else None
    digits = [p for p in parts if p.isdigit()]
    return None, int(digits[-1]) if digits else None


def _target_lock_free(directory: str, target: str | None) -> bool:
    if not target:
        return True
    try:
        fd = os.open(os.path.join(directory, target + LOCK_SUFFIX), os.O_RDONLY | os.O_CLOEXEC)
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


def sweep_temps(directory, patterns=DEFAULT_SWEEP_PATTERNS, min_age_s=3600.0, dry_run=False):
    """Delete orphaned temp files in ``directory`` (not recursive).

    A file matching one of ``patterns`` is removed when its embedded writer
    pid is dead on this host, or when it is older than ``min_age_s`` (which
    also covers names without a pid and reused pids), and only if the lock on
    its target file can be taken without blocking. Lock files are never
    touched. Returns the sorted list of paths removed (or that would be
    removed when ``dry_run`` is true).
    """
    directory = os.fspath(directory)
    now = time.time()
    me = os.getpid()
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
            age = now - entry.stat(follow_symlinks=False).st_mtime
        except FileNotFoundError:
            continue
        target, pid = _temp_owner(name)
        dead = pid is not None and pid != me and not _pid_alive(pid)
        if not (dead or age >= min_age_s):
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
