"""State-machine tests for scripts/parallel_download.py and scripts/rangefile.py.

CPU only and offline: a fake curl (a small script written to tmp_path), an
in-process fake transport, and a local http.server on 127.0.0.1 with Range,
ETag/If-Range and fault modes.

(a) random fetch / stall / kill / ENOSPC / assemble sequences through the
    property shim (hypothesis when installed, else seeded random walks)
(b) crash injection at every assembly step (mid-append, after the append
    before the state write, after the state write before the part unlink)
    and ENOSPC from a write wrapper; legacy-layout migration
(c) two concurrent processes on one OUT, 10 times: one exits 75, the
    output is byte-identical to the source
(d) the stall counter counts CONSECUTIVE zero-byte responses only
(e) a remote change between or inside sessions gives REMOTE_CHANGED and no
    mixed bytes; RangeFile raises RemoteChanged
(f) a rerun of a DONE download exits 0 without touching the network
(i) the vendored durable_state library is byte-identical to src/
(r) regressions from the adversarial review: orphan parts, legacy bytes and
    servers without validators need --sha256 or --allow-unverified, the
    expected sha256 is persisted, weak ETags, corrupt inputs, VERIFY_FAILED
    recovery, OUT replaced after DONE, sweep_temps and the assembly file
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

PW = Path(__file__).resolve().parent.parent
SCRIPTS = PW / "scripts"
if str(PW) not in sys.path:
    sys.path.insert(0, str(PW))

from alztest import _durable_state as ds  # noqa: E402

try:  # pragma: no cover - depends on the environment
    from hypothesis import given, settings
    from hypothesis import strategies as st
    HAVE_HYPOTHESIS = True
except ImportError:
    HAVE_HYPOTHESIS = False


def _script(name):
    spec = importlib.util.spec_from_file_location(f"_pw_{name}", SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


dl = _script("parallel_download")
S = dl.S


class Killed(BaseException):
    """Stands in for SIGKILL inside one process (nothing catches it)."""


# ------------------------------------------------------------------ fakes

class FakeTransport:
    """In-process transport over ``payload`` with stall and size faults."""

    def __init__(self, payload: bytes, etag='"v1"', *, max_bytes=None, zero=None):
        self.payload = payload
        self.etag = etag
        self.max_bytes = max_bytes
        self.zero = zero  # callable(start) -> bool: answer this request with 0 bytes
        self.served = 0
        self.requests = 0
        self.probes = 0
        self._lock = threading.Lock()

    def probe(self, url, hdr):
        self.probes += 1
        return dl.Probe(200, len(self.payload), self.etag, "Fri, 20 Nov 2026 00:00:00 GMT")

    def get_range(self, url, start, hi, if_range, chunk, hdr):
        with self._lock:
            self.requests += 1
        if if_range is not None and if_range != self.etag:
            Path(chunk).write_bytes(b"")
            return dl.Response(63, 200, None, self.etag)
        n = hi - start + 1
        if self.max_bytes:
            n = min(n, self.max_bytes)
        if self.zero is not None and self.zero(start):
            n = 0
        Path(chunk).write_bytes(self.payload[start:start + n])
        with self._lock:
            self.served += n
        return dl.Response(0 if n == hi - start + 1 else 18, 206,
                           f"bytes {start}-{start + max(n, 1) - 1}/{len(self.payload)}", self.etag)


class NoNetwork:
    def probe(self, *a):
        raise AssertionError("no network access expected")

    get_range = probe


FAKE_CURL = r'''
import os, sys, time
a = sys.argv[1:]
def opt(k):
    return a[a.index(k) + 1] if k in a else None
src = open(os.environ["FAKE_SRC"], "rb").read()
etag_file = os.environ.get("FAKE_ETAG_FILE")
etag = open(etag_file).read().strip() if etag_file else '"v1"'
hdr, out = opt("-D"), opt("-o")
if "--head" in a:
    open(hdr, "w").write(f"HTTP/1.1 200 OK\r\nContent-Length: {len(src)}\r\nETag: {etag}\r\n\r\n")
    sys.exit(0)
if_range = None
for i, x in enumerate(a):
    if x == "-H" and a[i + 1].lower().startswith("if-range:"):
        if_range = a[i + 1].split(":", 1)[1].strip()
lo, hi = map(int, opt("-r").split("-"))
time.sleep(float(os.environ.get("FAKE_SLEEP", 0)))
if if_range is not None and if_range != etag:
    open(hdr, "w").write(f"HTTP/1.1 200 OK\r\nContent-Length: {len(src)}\r\nETag: {etag}\r\n\r\n")
    open(out, "wb").close()
    sys.exit(63)
n = min(hi - lo + 1, int(os.environ.get("FAKE_MAX", hi - lo + 1)))
open(hdr, "w").write(f"HTTP/1.1 206 Partial Content\r\n"
                     f"Content-Range: bytes {lo}-{lo + max(n, 1) - 1}/{len(src)}\r\nETag: {etag}\r\n\r\n")
with open(out, "wb") as f:
    f.write(src[lo:lo + n])
log = os.environ.get("FAKE_LOG")
if log:
    fd = os.open(log, os.O_WRONLY | os.O_APPEND | os.O_CREAT)
    os.write(fd, f"{lo} {n}\n".encode())
    os.close(fd)
sys.exit(0 if n == hi - lo + 1 else 28)
'''


@pytest.fixture
def fake_curl(tmp_path, monkeypatch):
    """(curl argv, payload, env) for a fake curl serving a random payload."""
    tools = tmp_path / "tools"
    tools.mkdir()
    script = tools / "fakecurl.py"
    script.write_text(FAKE_CURL)
    payload = os.urandom(40_000)
    src = tools / "src.bin"
    src.write_bytes(payload)
    env = {"FAKE_SRC": str(src), "FAKE_LOG": str(tools / "served.log")}
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return [sys.executable, "-I", str(script)], payload, env


def _served(env) -> int:
    log = Path(env["FAKE_LOG"])
    return sum(int(line.split()[1]) for line in log.read_text().splitlines()) if log.exists() else 0


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _run_cli(out, payload_len, nparts, curl, env, extra_env=None, workers=4, wait=True):
    code = (f"import sys; sys.path.insert(0, {str(SCRIPTS)!r}); import parallel_download as d; "
            f"sys.exit(d.main(['http://fake/obj', {str(out)!r}, '{payload_len}', '{nparts}'], "
            f"curl={curl!r}, workers={workers}))")
    proc = subprocess.Popen([sys.executable, "-B", "-c", code],
                            env={**os.environ, **env, **(extra_env or {})},
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if not wait:
        return proc
    out_, err = proc.communicate(timeout=120)
    return proc.returncode, out_, err


def _state(out):
    return ds.StateFile(f"{out}.state.json", dl.MACHINE, recover=False).read()


def _check_journal(out):
    """Every journaled transition is an edge of the table, seq is contiguous (I1/I2)."""
    sf = ds.StateFile(f"{out}.state.json", dl.MACHINE, recover=False)
    entries = sf.journal()
    assert [e["seq"] for e in entries] == list(range(1, len(entries) + 1))
    for e in entries:
        if e["from"] is not None and e["event"] != ds.RECONCILE_EVENT:
            assert dl.MACHINE.transitions[(S[e["from"]], e["event"])] is S[e["to"]], e
    snap = sf.read()
    assert ds.replay_journal(sf.journal_path) == (snap.state.name, snap.version)
    return entries


def _check_invariants(out, payload):
    """Bytes on disk are always slices of the one source; OUT only when verified."""
    snap = _state(out)
    if snap is None:
        return None
    parts = [tuple(j) for j in snap.data["parts"]]
    for k, lo, hi in parts:
        p = Path(dl.part_path(str(out), k))
        if p.exists():
            b = p.read_bytes()
            assert len(b) <= hi - lo + 1 and b == payload[lo:lo + len(b)], (k, len(b))
    partial = Path(f"{out}.assembling")
    if partial.exists():
        b = partial.read_bytes()
        assert b == payload[:len(b)]
        if snap.state is S.ASSEMBLING:
            assert len(b) >= snap.data["out_size_at_part_start"]
    if Path(out).exists():
        assert snap.state in (S.VERIFIED, S.DONE)
        assert Path(out).read_bytes() == payload
    if snap.state is S.DONE:
        receipt = json.loads(Path(f"{out}.receipt.json").read_text())
        assert receipt["sha256"] == _sha(payload) == snap.data["sha256"]
        assert receipt["size"] == len(payload)
        assert not partial.exists() and not list(Path(out).parent.glob(Path(out).name + ".part*"))
    _check_journal(out)
    return snap


# ------------------------------------------------------------------ (i) vendored library

def test_vendored_durable_state_is_byte_identical_to_src():
    src = PW.parent / "src" / "alz_perturb_gan" / "durable_state.py"
    if not src.exists():
        pytest.skip("paper_watch is used outside the repository; nothing to compare")
    vendored = PW / "alztest" / "_durable_state.py"
    assert vendored.read_bytes() == src.read_bytes(), (
        "paper_watch/alztest/_durable_state.py must be a byte-identical copy of "
        "src/alz_perturb_gan/durable_state.py; copy it again")


# ------------------------------------------------------------------ the table

def test_machine_matches_the_documented_table():
    assert dl.MACHINE.terminal == {S.DONE, S.REMOTE_CHANGED}
    assert dl.MACHINE.allowed_events(S.FETCHING) == [
        "adopt_prefix", "expect", "part_done", "parts_complete", "remote_changed", "stall"]
    assert dl.MACHINE.allowed_events(S.STALLED) == ["expect", "remote_changed", "resume"]
    assert dl.MACHINE.allowed_events(S.ASSEMBLING) == [
        "adopt_prefix", "append", "assembled", "expect"]
    # a wrong --sha256 is recoverable: VERIFY_FAILED -reverify-> ASSEMBLED
    assert dl.MACHINE.allowed_events(S.VERIFY_FAILED) == ["reverify"]
    assert dl.MACHINE.next_state(S.VERIFY_FAILED, "reverify") is S.ASSEMBLED
    # nothing records a new expected sha256 once the bytes are verified
    assert dl.MACHINE.next_state(S.VERIFIED, "expect") is None
    assert dl.MACHINE.next_state(S.VERIFIED, "finish") is S.DONE
    # no way back from assembly to fetching, and nothing leaves DONE
    assert dl.MACHINE.next_state(S.ASSEMBLING, "fetch") is None
    assert dl.MACHINE.allowed_events(S.DONE) == []


# ------------------------------------------------------------------ (a) property walk

FAULT_POINTS = ("part_appended", "mid_append", "after_append", "after_state",
                "after_rename", "after_receipt")


def _one_walk(seed: int, root: Path, monkeypatch) -> None:
    rng = random.Random(seed)
    payload = rng.randbytes(rng.randint(1, 3000))
    nparts = rng.randint(1, 5)
    out = root / f"w{seed}" / "obj.bin"
    out.parent.mkdir(parents=True)
    argv = ["http://fake/obj", str(out), str(len(payload)), str(nparts),
            "--attempts", str(rng.randint(1, 3)), "--no-space-check"]
    total_served = 0
    finished = False
    for session in range(12):
        clean = session >= 8  # the walk always ends with fault-free sessions
        zero_p = 0.0 if clean else rng.choice([0.0, 0.3, 0.7])
        t = FakeTransport(payload, max_bytes=None if clean else rng.choice([None, 97, 700]),
                          zero=(lambda start, p=zero_p: rng.random() < p))
        monkeypatch.setattr(dl, "COPY_BLOCK", rng.choice([7, 64, 8192]))
        kill = None if clean or rng.random() < 0.3 else (rng.choice(FAULT_POINTS), rng.randint(1, 3))
        counts: dict = {}

        def fault(point, kill=kill, counts=counts):
            counts[point] = counts.get(point, 0) + 1
            if kill and kill == (point, counts[point]):
                raise Killed(point)
        monkeypatch.setattr(dl, "_fault", fault)
        enospc_after = None if clean or rng.random() < 0.7 else rng.randint(0, len(payload))
        written = [0]
        real_write = dl._write

        def write(fd, data, limit=enospc_after):
            if limit is not None and os.readlink(f"/proc/self/fd/{fd}").endswith(".assembling"):
                room = limit - written[0]
                if len(data) > room:
                    real_write(fd, bytes(data)[:max(room, 0)])
                    written[0] = limit
                    raise OSError(28, "No space left on device (simulated)")
                written[0] += len(data)
            real_write(fd, data)
        monkeypatch.setattr(dl, "_write", write)
        try:
            rc = dl.main(argv, transport=t, workers=rng.randint(1, 3))
        except Killed:
            rc = None
        monkeypatch.setattr(dl, "_write", real_write)
        total_served += t.served
        snap = _check_invariants(out, payload)
        if rc == 0:
            assert snap.state is S.DONE
            finished = True
            break
        if snap is not None:
            assert snap.state is not S.REMOTE_CHANGED and snap.state is not S.VERIFY_FAILED
    assert finished, f"seed {seed}: no DONE after the clean sessions"
    assert out.read_bytes() == payload
    # Zero parts discarded: every byte was fetched exactly once.
    assert total_served == len(payload), (seed, total_served, len(payload))
    # A DONE rerun is a no-op.
    assert dl.main(argv, transport=NoNetwork()) == 0


if HAVE_HYPOTHESIS:  # pragma: no cover - hypothesis is optional
    @settings(max_examples=60, deadline=None)
    @given(seed=st.integers(0, 2**32 - 1))
    def test_random_fetch_stall_kill_assemble_sequences(seed):
        with tempfile.TemporaryDirectory() as root, pytest.MonkeyPatch.context() as mp:
            _one_walk(seed, Path(root), mp)
else:
    @pytest.mark.parametrize("seed", range(120))
    def test_random_fetch_stall_kill_assemble_sequences(seed, tmp_path, monkeypatch):
        _one_walk(seed, tmp_path, monkeypatch)


# ------------------------------------------------------------------ (b) crash injection

@pytest.mark.parametrize("fault", ["mid_append:1", "mid_append:3", "after_append:1",
                                   "after_append:3", "after_state:1", "after_state:4",
                                   "after_rename:1", "after_receipt:1", "part_appended:2"])
def test_kill_at_each_step_then_resume(tmp_path, fake_curl, fault):
    curl, payload, env = fake_curl
    out = tmp_path / "d" / "obj.h5ad"
    out.parent.mkdir()
    rc, _, err = _run_cli(out, len(payload), 4, curl, env,
                          {"PARALLEL_DOWNLOAD_FAULT": fault, "PYTHONDONTWRITEBYTECODE": "1"},
                          workers=1)
    assert rc == 137, err
    snap = _check_invariants(out, payload)
    point = fault.split(":")[0]
    if point in ("mid_append", "after_append", "after_state"):
        assert snap.state is S.ASSEMBLING
    served_before = _served(env)
    rc, stdout, err = _run_cli(out, len(payload), 4, curl, env)
    assert rc == 0, err
    snap = _check_invariants(out, payload)
    assert snap.state is S.DONE and _sha(out.read_bytes()) == _sha(payload)
    if point != "part_appended":
        assert _served(env) == served_before == len(payload)  # no part was refetched
    assert _served(env) == len(payload)
    events = [e["event"] for e in _check_journal(out)]
    appends = events.count("append")
    assert appends == 4  # one per part, the re-done one included only once
    assert not list(out.parent.glob("obj.h5ad.part*"))


def test_enospc_mid_append_resumes_without_losing_parts(tmp_path, monkeypatch):
    payload = os.urandom(40_000)
    out = tmp_path / "obj.bin"
    argv = ["u", str(out), str(len(payload)), "4", "--no-space-check"]
    real_write = dl._write
    state = {"written": 0}

    def short_write(fd, data):
        if os.readlink(f"/proc/self/fd/{fd}").endswith(".assembling"):
            if state["written"] + len(data) > 15_000:  # inside part 01
                real_write(fd, bytes(data)[:15_000 - state["written"]])
                state["written"] = 15_000
                raise OSError(28, "No space left on device (simulated)")
            state["written"] += len(data)
        real_write(fd, data)

    t = FakeTransport(payload)
    monkeypatch.setattr(dl, "_write", short_write)
    assert dl.main(argv, transport=t) == 1
    snap = _check_invariants(out, payload)
    assert snap.state is S.ASSEMBLING and snap.data["next_part"] == 1
    assert Path(f"{out}.assembling").stat().st_size == 15_000  # ends inside part 01
    assert all(Path(dl.part_path(str(out), k)).exists() for k in (1, 2, 3))
    monkeypatch.setattr(dl, "_write", real_write)
    assert dl.main(argv, transport=t) == 0
    assert out.read_bytes() == payload and t.served == len(payload)


def test_free_space_preflight_refuses_with_the_numbers(tmp_path, monkeypatch, capsys):
    payload = os.urandom(4000)
    out = tmp_path / "obj.bin"
    usage = shutil.disk_usage(tmp_path)
    monkeypatch.setattr(dl.shutil, "disk_usage", lambda p: usage._replace(free=4999))
    assert dl.main(["u", str(out), "4000", "4"], transport=FakeTransport(payload)) == 1
    err = capsys.readouterr().err
    assert "need 5000 bytes (total 4000 + largest part 1000 - 0 already on disk), have 4999" in err
    assert not Path(f"{out}.state.json").exists() and not out.exists()


def test_busy_lock_exits_75(tmp_path, capsys):
    out = tmp_path / "obj.bin"
    with ds.FileLock(str(out), purpose="test holder"):
        assert dl.main(["u", str(out), "10"], transport=NoNetwork()) == dl.EX_BUSY
    assert "exiting 75" in capsys.readouterr().err


# ------------------------------------------------------------------ migration

def _legacy_layout(out: Path, payload: bytes, have: int, nparts: int):
    jobs = dl.plan(have, len(payload), nparts)
    Path(f"{out}.plan").write_text(json.dumps(
        {"url": "u", "total": len(payload), "jobs": [list(j) for j in jobs]}))
    for k, lo, hi in jobs:
        Path(dl.part_path(str(out), k)).write_bytes(payload[lo:hi + 1])
    return jobs


@pytest.mark.parametrize("case", ["crashed_mid_append", "appended_not_removed",
                                  "prefix_only", "partial_parts"])
def test_legacy_plan_is_adopted_and_finished(tmp_path, case):
    payload = os.urandom(40_000)
    out = tmp_path / "obj.bin"
    jobs = _legacy_layout(out, payload, 1000, 4)
    if case == "crashed_mid_append":  # old bug A2: OUT ends inside part 01
        out.write_bytes(payload[:jobs[1][1] + 2500])
        os.remove(dl.part_path(str(out), 0))
        expect = S.ASSEMBLING
    elif case == "appended_not_removed":  # old bug A: part 01 appended, file still there
        out.write_bytes(payload[:jobs[1][2] + 1])
        os.remove(dl.part_path(str(out), 0))
        expect = S.ASSEMBLING
    elif case == "prefix_only":
        out.write_bytes(payload[:1000])
        expect = S.FETCHING
    else:
        out.write_bytes(payload[:1000])
        Path(dl.part_path(str(out), 2)).write_bytes(payload[jobs[2][1]:jobs[2][1] + 10])
        os.remove(dl.part_path(str(out), 3))
        expect = S.FETCHING
    t = FakeTransport(payload)
    argv = ["u", str(out), str(len(payload)), "4"]
    # Legacy bytes were fetched without If-Range: refused without --sha256
    # (or --allow-unverified), before anything is written.
    assert dl.main(argv, transport=NoNetwork()) == 1
    assert not Path(f"{out}.state.json").exists() and Path(f"{out}.plan").exists()
    assert dl.main(argv + ["--sha256", _sha(payload)], transport=t) == 0
    assert out.read_bytes() == payload
    receipt = json.loads(Path(f"{out}.receipt.json").read_text())
    assert receipt["verified_by"] == "sha256" and receipt["unvalidated"] == ["legacy_bytes"]
    entries = _check_journal(out)
    assert entries[0]["event"] == ds.ADOPT_EVENT and entries[0]["to"] == expect.name
    assert entries[1]["event"] == "adopt_prefix"
    assert not Path(f"{out}.plan").exists() and Path(f"{out}.plan.legacy").exists()
    if case in ("crashed_mid_append", "appended_not_removed"):
        assert t.requests == 0  # nothing re-downloaded
    if case == "partial_parts":
        assert t.served == (jobs[2][2] - jobs[2][1] + 1 - 10) + (jobs[3][2] - jobs[3][1] + 1)


def test_full_size_out_without_plan_is_verified(tmp_path):
    payload = os.urandom(5000)
    out = tmp_path / "obj.bin"
    out.write_bytes(payload)
    argv = ["u", str(out), "5000", "--sha256", _sha(payload)]
    assert dl.main(argv, transport=NoNetwork()) == 0
    assert out.read_bytes() == payload
    events = [e["event"] for e in _check_journal(out)]
    assert events == [ds.ADOPT_EVENT, "adopt_prefix", "verify", "finish"]


def test_full_size_out_with_the_wrong_sha_is_kept_not_promoted(tmp_path):
    payload = os.urandom(5000)
    out = tmp_path / "obj.bin"
    out.write_bytes(payload)
    assert dl.main(["u", str(out), "5000", "--sha256", "0" * 64], transport=NoNetwork()) == 1
    assert _state(out).state is S.VERIFY_FAILED
    assert Path(f"{out}.assembling").read_bytes() == payload  # renamed once, never deleted
    assert dl.main(["u", str(out), "5000"], transport=NoNetwork()) == 1  # stays VERIFY_FAILED
    assert _state(out).state is S.VERIFY_FAILED
    # review probe 9: the right sha256 re-verifies the kept bytes and finishes
    assert dl.main(["u", str(out), "5000", "--sha256", _sha(payload)], transport=NoNetwork()) == 0
    assert out.read_bytes() == payload
    assert [e["event"] for e in _check_journal(out)][-3:] == ["reverify", "verify", "finish"]


def test_changed_layout_and_orphan_parts_are_refused(tmp_path):
    out = tmp_path / "obj.bin"
    t = FakeTransport(os.urandom(100), max_bytes=1, zero=lambda s: True)
    assert dl.main(["u", str(out), "100", "2", "--attempts", "1"], transport=t) == 1
    assert _state(out).state is S.STALLED
    assert dl.main(["other", str(out), "100", "2"], transport=t) == 1
    out2 = tmp_path / "orphan.bin"
    Path(dl.part_path(str(out2), 0)).write_bytes(b"x")
    assert dl.main(["u", str(out2), "100", "2"], transport=t) == 1
    assert not Path(f"{out2}.state.json").exists()


# ------------------------------------------------------------------ (c) concurrency

def test_two_processes_on_one_out_never_corrupt_it(tmp_path, fake_curl):
    curl, payload, env = fake_curl
    for run in range(10):
        out = tmp_path / f"r{run}" / "obj.h5ad"
        out.parent.mkdir()
        slow = {"FAKE_SLEEP": "0.15", "FAKE_MAX": "7000"}
        procs = [_run_cli(out, len(payload), 2, curl, env, slow, workers=2, wait=False)
                 for _ in range(2)]
        rcs = sorted(p.wait(timeout=120) for p in procs)
        for p in procs:
            p.stdout.close()
            p.stderr.close()
        assert rcs == [0, dl.EX_BUSY], (run, rcs)
        assert _sha(out.read_bytes()) == _sha(payload)
        _check_invariants(out, payload)


# ------------------------------------------------------------------ (d) stall reset

def test_stall_counter_resets_on_progress(tmp_path):
    payload = os.urandom(10_000)
    flip = {"n": 0}

    def alternate(start):
        flip["n"] += 1
        return flip["n"] % 2 == 1  # zero, progress, zero, progress, ...

    t = FakeTransport(payload, max_bytes=500, zero=alternate)
    res = dl.fetch_part(t, "u", str(tmp_path / "o"), (0, 0, 9999), attempts=2, total=10_000,
                        etag='"v1"', if_range='"v1"', stop=threading.Event())
    assert res == "ok" and t.requests == 40  # 20 progress + 20 zero, never 2 zeros in a row
    assert (tmp_path / "o.part00").read_bytes() == payload


@pytest.mark.parametrize("attempts", [1, 2, 4])
def test_stall_after_exactly_attempts_consecutive_zero_responses(tmp_path, attempts):
    payload = os.urandom(1000)
    seq = iter([False] + [True] * (attempts - 1) + [False] + [True] * attempts + [False] * 99)
    t = FakeTransport(payload, max_bytes=100, zero=lambda s: next(seq))
    res = dl.fetch_part(t, "u", str(tmp_path / "o"), (0, 0, 999), attempts=attempts, total=1000,
                        etag=None, if_range=None, stop=threading.Event())
    assert res == "stalled"
    assert t.requests == 1 + (attempts - 1) + 1 + attempts
    assert (tmp_path / "o.part00").stat().st_size == 200


# ------------------------------------------------------------------ (e) remote change

class _Remote:
    payload = b""
    etag = '"v1"'
    weak = False  # serve the ETag as W/"..." (never matches If-Range)
    send_etag = True
    lm = None  # Last-Modified, when set
    head_etag = None  # a stale HEAD (CDN) still reporting the old ETag
    flip_after = None  # flip to payload2/etag2 after this many GETs
    payload2 = b""
    etag2 = '"v2"'
    gets = 0


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    @staticmethod
    def _served_etag(etag):
        return ("W/" + etag) if _Remote.weak else etag

    def _validators(self, etag):
        if _Remote.send_etag:
            self.send_header("ETag", self._served_etag(etag))
        if _Remote.lm:
            self.send_header("Last-Modified", _Remote.lm)

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Length", str(len(_Remote.payload)))
        self._validators(_Remote.head_etag or _Remote.etag)
        self.end_headers()

    def do_GET(self):
        _Remote.gets += 1
        if _Remote.flip_after is not None and _Remote.gets > _Remote.flip_after:
            _Remote.payload, _Remote.etag, _Remote.flip_after = _Remote.payload2, _Remote.etag2, None
        body, etag = _Remote.payload, _Remote.etag
        rng = self.headers.get("Range")
        if_range = self.headers.get("If-Range")
        matches = if_range is None or (if_range == etag and not _Remote.weak) or (
            _Remote.lm is not None and if_range == _Remote.lm)
        if rng and matches:
            lo, hi = map(int, rng.split("=")[1].split("-"))
            hi = min(hi, len(body) - 1)
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {lo}-{hi}/{len(body)}")
            body = body[lo:hi + 1]
        else:
            self.send_response(200)
        self._validators(etag)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def remote(monkeypatch):
    for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    _Remote.payload, _Remote.payload2 = os.urandom(20_000), os.urandom(20_000)
    _Remote.etag, _Remote.head_etag, _Remote.flip_after, _Remote.gets = '"v1"', None, None, 0
    _Remote.weak, _Remote.send_etag, _Remote.lm = False, True, None
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/obj"
    srv.shutdown()


CURL_FAST = ["curl", "--fail", "--silent", "--show-error", "--location"]
needs_curl = pytest.mark.skipif(shutil.which("curl") is None, reason="curl not installed")


def _kill_after_first_part(monkeypatch):
    def fault(point):
        if point == "part_appended":
            raise Killed(point)
    monkeypatch.setattr(dl, "_fault", fault)


def _no_new_bytes(out, old: bytes, new: bytes):
    for p in Path(out).parent.glob(Path(out).name + ".part*"):
        k = int(p.name.rsplit("part", 1)[1])
        lo = k * 5000
        b = p.read_bytes()
        assert b == old[lo:lo + len(b)] and (not b or b != new[lo:lo + len(b)])


@needs_curl
@pytest.mark.parametrize("how", ["head", "if_range", "mid_session"])
def test_remote_change_stops_without_mixing(tmp_path, remote, monkeypatch, how):
    out = tmp_path / "obj.bin"
    argv = [remote, str(out), "20000", "4"]
    old = _Remote.payload
    if how == "mid_session":
        _Remote.flip_after = 2  # the third range request sees the new object
        assert dl.main(argv, curl=CURL_FAST, workers=1) == 1
    else:
        _kill_after_first_part(monkeypatch)
        with pytest.raises(Killed):
            dl.main(argv, curl=CURL_FAST, workers=1)
        monkeypatch.setattr(dl, "_fault", lambda point: None)
        assert _state(out).state is S.FETCHING
        _Remote.payload, _Remote.etag = _Remote.payload2, '"v2"'
        if how == "if_range":
            _Remote.head_etag = '"v1"'  # HEAD is stale; only If-Range can tell
        assert dl.main(argv, curl=CURL_FAST, workers=1) == 1
    snap = _state(out)
    assert snap.state is S.REMOTE_CHANGED, snap
    _no_new_bytes(out, old, _Remote.payload2)
    assert not out.exists() and not Path(f"{out}.assembling").exists()
    assert dl.main(argv, curl=CURL_FAST) == 1  # terminal until the user starts over
    _check_journal(out)


@needs_curl
def test_download_over_http_then_done_rerun(tmp_path, remote, capsys):
    out = tmp_path / "obj.bin"
    argv = [remote, str(out), "20000", "3", "--sha256", _sha(_Remote.payload)]
    assert dl.main(argv, curl=CURL_FAST) == 0
    assert out.read_bytes() == _Remote.payload
    receipt = json.loads(Path(f"{out}.receipt.json").read_text())
    assert receipt["etag"] == '"v1"' and receipt["sha256"] == _sha(_Remote.payload)
    gets = _Remote.gets
    # (f) DONE -rerun-> DONE: exit 0, nothing fetched, even if the remote moved on.
    _Remote.etag = '"v9"'
    capsys.readouterr()
    assert dl.main(argv, curl=CURL_FAST) == 0
    assert "already complete" in capsys.readouterr().out
    assert _Remote.gets == gets


def test_rangefile_raises_remote_changed(remote):
    pytest.importorskip("requests")
    rf_mod = _script("rangefile")
    with pytest.raises(rf_mod.RemoteChanged):
        rf_mod.RangeFile(remote, etag='"v0"')
    rf = rf_mod.RangeFile(remote, block=1000)
    assert rf.etag == '"v1"'
    assert rf.read(1500) == _Remote.payload[:1500]
    _Remote.payload, _Remote.etag = _Remote.payload2, '"v2"'
    rf.seek(5000)
    with pytest.raises(rf_mod.RemoteChanged):
        rf.read(10)
    rf.close()


def test_rangefile_without_a_strong_etag_uses_last_modified(remote):
    """Review: weak ETags broke every block; NCBI sends Last-Modified only."""
    pytest.importorskip("requests")
    rf_mod = _script("rangefile")
    _Remote.weak, _Remote.lm = True, "Fri, 20 Nov 2026 00:00:00 GMT"
    rf = rf_mod.RangeFile(remote, block=1000)
    assert rf.etag == 'W/"v1"' and rf.if_range == _Remote.lm
    assert rf.read(2500) == _Remote.payload[:2500]  # used to raise RemoteChanged
    rf.close()
    _Remote.send_etag = False  # the real NCBI URL: no ETag at all
    rf = rf_mod.RangeFile(remote, block=1000)
    assert rf.etag is None and rf.if_range == _Remote.lm
    assert rf.read(1500) == _Remote.payload[:1500]
    _Remote.payload, _Remote.lm = _Remote.payload2, "Sat, 21 Nov 2026 00:00:00 GMT"
    rf.seek(5000)
    with pytest.raises(rf_mod.RemoteChanged):
        rf.read(10)
    rf.close()


# ------------------------------------------------------------------ (r) review regressions

LM = "Fri, 20 Nov 2026 00:00:00 GMT"


class ValidatorTransport(FakeTransport):
    """Honours If-Range like RFC 9110: a STRONG ETag or Last-Modified must match.

    ``flip_after``: requests after this many are served from ``payload2``
    (with ``etag2``/``lm2``), i.e. the remote object changes mid-download.
    """

    def __init__(self, payload, etag='"v1"', lm=LM, *, payload2=None, flip_after=None,
                 etag2='"v2"', lm2="Sat, 21 Nov 2026 00:00:00 GMT", **kw):
        super().__init__(payload, etag, **kw)
        self.lm, self.if_ranges = lm, []
        self.payload2, self.flip_after, self.etag2, self.lm2 = payload2, flip_after, etag2, lm2

    def _maybe_flip(self):
        if self.flip_after is not None and self.requests >= self.flip_after:
            self.payload = self.payload2
            self.etag = self.etag2 if self.etag is not None else None
            self.lm = self.lm2 if self.lm is not None else None
            self.flip_after = None

    def probe(self, url, hdr):
        self.probes += 1
        return dl.Probe(200, len(self.payload), self.etag, self.lm)

    def get_range(self, url, start, hi, if_range, chunk, hdr):
        with self._lock:
            self._maybe_flip()
            self.if_ranges.append(if_range)
        strong = bool(self.etag) and not self.etag.startswith("W/")
        if if_range is not None and not ((strong and if_range == self.etag)
                                         or (self.lm and if_range == self.lm)):
            with self._lock:
                self.requests += 1
            Path(chunk).write_bytes(b"")
            return dl.Response(63, 200, None, self.etag)
        return FakeTransport.get_range(self, url, start, hi, None, chunk, hdr)


def _argv(out, payload, nparts=4, *extra):
    return ["u", str(out), str(len(payload)), str(nparts), *extra]


def test_weak_etag_falls_back_to_last_modified(tmp_path):
    """Probe 1: If-Range: W/"..." never matches, so it used to go REMOTE_CHANGED at once."""
    assert dl.if_range_validator('W/"x"', None) is None
    assert dl.if_range_validator('W/"x"', LM) == LM
    assert dl.if_range_validator('"x"', LM) == '"x"'
    payload = os.urandom(20_000)
    out = tmp_path / "obj.bin"
    t = ValidatorTransport(payload, etag='W/"abc"')
    assert dl.main(_argv(out, payload), transport=t) == 0
    assert out.read_bytes() == payload and set(t.if_ranges) == {LM}
    receipt = json.loads(Path(f"{out}.receipt.json").read_text())
    assert receipt["verified_by"] == "validator" and receipt["unvalidated"] == []


def test_done_rerun_with_a_different_sha256_fails(tmp_path, capsys):
    """Probe 2: a DONE rerun with another --sha256 used to print 'already complete', rc 0."""
    payload = os.urandom(20_000)
    out = tmp_path / "obj.bin"
    assert dl.main(_argv(out, payload), transport=FakeTransport(payload)) == 0
    capsys.readouterr()
    assert dl.main(_argv(out, payload, 4, "--sha256", "0" * 64), transport=NoNetwork()) == 1
    assert "not the given --sha256" in capsys.readouterr().err
    assert dl.main(_argv(out, payload, 4, "--sha256", _sha(payload).upper()),
                   transport=NoNetwork()) == 0


def _killed_first_session(out, payload, monkeypatch, *extra):
    _kill_after_first_part(monkeypatch)
    with pytest.raises(Killed):
        dl.main(_argv(out, payload, 4, *extra), transport=FakeTransport(payload), workers=1)
    monkeypatch.setattr(dl, "_fault", lambda point: None)
    assert _state(out).state is S.FETCHING


def test_expected_sha256_survives_a_resume_without_the_flag(tmp_path, monkeypatch):
    """Probe 3: --sha256 on the killed first run, dropped on resume, used to verify nothing."""
    payload = os.urandom(20_000)
    out = tmp_path / "obj.bin"
    _killed_first_session(out, payload, monkeypatch, "--sha256", "0" * 64)
    assert _state(out).data["expected_sha256"] == "0" * 64
    assert dl.main(_argv(out, payload), transport=FakeTransport(payload)) == 1
    assert _state(out).state is S.VERIFY_FAILED and not out.exists()
    # The right one re-verifies the kept bytes without fetching anything.
    assert dl.main(_argv(out, payload, 4, "--sha256", _sha(payload)), transport=NoNetwork()) == 0
    assert out.read_bytes() == payload
    assert json.loads(Path(f"{out}.receipt.json").read_text())["verified_by"] == "sha256"


def test_a_different_sha256_on_resume_is_refused(tmp_path, monkeypatch, capsys):
    payload = os.urandom(20_000)
    out = tmp_path / "obj.bin"
    _killed_first_session(out, payload, monkeypatch, "--sha256", _sha(payload))
    capsys.readouterr()
    assert dl.main(_argv(out, payload, 4, "--sha256", "1" * 64), transport=NoNetwork()) == 1
    assert "records expected sha256" in capsys.readouterr().err
    assert _state(out).state is S.FETCHING
    assert dl.main(_argv(out, payload), transport=FakeTransport(payload)) == 0
    assert out.read_bytes() == payload


def test_sha256_first_given_on_resume_is_recorded(tmp_path, monkeypatch):
    payload = os.urandom(20_000)
    out = tmp_path / "obj.bin"
    _killed_first_session(out, payload, monkeypatch)
    assert dl.main(_argv(out, payload, 4, "--sha256", _sha(payload)),
                   transport=FakeTransport(payload)) == 0
    events = [e["event"] for e in _check_journal(out)]
    assert "expect" in events and events.index("expect") < events.index("verify")
    assert _state(out).data["expected_sha256"] == _sha(payload)


@pytest.mark.parametrize("orphan", ["part00", "part03.chunk", "part17"])
def test_orphan_parts_next_to_a_legacy_out_are_refused(tmp_path, orphan, capsys):
    """Probe 4: junk OUT.part00 + legacy OUT prefix, no plan, used to be promoted (rc 0)."""
    payload = os.urandom(50_000)
    out = tmp_path / "obj.bin"
    out.write_bytes(payload[:1000])
    Path(f"{out}.{orphan}").write_bytes(b"X" * 500)
    t = FakeTransport(payload)
    assert dl.main(_argv(out, payload, 4, "--allow-unverified"), transport=t) == 1
    assert "never reinterpreted" in capsys.readouterr().err
    assert out.read_bytes() == payload[:1000] and Path(f"{out}.{orphan}").read_bytes() == b"X" * 500
    assert not Path(f"{out}.state.json").exists() and t.requests == 0


def test_parts_outside_the_legacy_plan_are_refused(tmp_path):
    payload = os.urandom(20_000)
    out = tmp_path / "obj.bin"
    _legacy_layout(out, payload, 0, 2)
    Path(dl.part_path(str(out), 5)).write_bytes(b"x")
    assert dl.main(_argv(out, payload, 2, "--sha256", _sha(payload)),
                   transport=FakeTransport(payload)) == 1
    assert not Path(f"{out}.state.json").exists() and Path(f"{out}.plan").exists()


def test_legacy_parts_from_an_older_remote_version_are_never_promoted(tmp_path, capsys):
    """Probe 11: old-tool parts of version B + a server now serving A used to mix (rc 0)."""
    new, old = os.urandom(20_000), os.urandom(20_000)
    out = tmp_path / "obj.bin"
    jobs = dl.plan(0, len(new), 4)
    Path(f"{out}.plan").write_text(json.dumps(
        {"url": "u", "total": len(new), "jobs": [list(j) for j in jobs]}))
    k, lo, hi = jobs[0]
    Path(dl.part_path(str(out), k)).write_bytes(old[lo:hi + 1])
    t = FakeTransport(new, etag='"v2"')
    assert dl.main(_argv(out, new), transport=t) == 1  # refused before anything is written
    assert "--allow-unverified" in capsys.readouterr().err
    assert t.probes == t.requests == 0 and not Path(f"{out}.state.json").exists()
    assert dl.main(_argv(out, new, 4, "--sha256", _sha(new)), transport=t) == 1
    assert _state(out).state is S.VERIFY_FAILED and not out.exists()


def test_allow_unverified_is_recorded_as_size_only(tmp_path):
    payload = os.urandom(20_000)
    out = tmp_path / "obj.bin"
    _legacy_layout(out, payload, 0, 4)
    assert dl.main(_argv(out, payload, 4, "--allow-unverified"),
                   transport=FakeTransport(payload)) == 0
    receipt = json.loads(Path(f"{out}.receipt.json").read_text())
    assert receipt["verified_by"] == "size-only" and receipt["unvalidated"] == ["legacy_bytes"]
    assert receipt["expected_sha256"] is None and _state(out).data["allow_unverified"] is True


def test_server_without_validators_needs_sha256_or_allow_unverified(tmp_path, capsys):
    """Probe 10: no ETag and no Last-Modified + a change mid-download used to promote mixed bytes."""
    a, b = os.urandom(20_000), os.urandom(20_000)
    out = tmp_path / "obj.bin"
    t = ValidatorTransport(a, etag=None, lm=None, payload2=b, flip_after=2)
    assert dl.main(_argv(out, a), transport=t, workers=1) == 1
    assert "neither ETag nor Last-Modified" in capsys.readouterr().err
    assert t.requests == 0 and not Path(f"{out}.state.json").exists()
    assert dl.main(_argv(out, a, 4, "--sha256", _sha(a)), transport=t, workers=1) == 1
    assert _state(out).state is S.VERIFY_FAILED and not out.exists()
    out2 = tmp_path / "steady.bin"
    t2 = ValidatorTransport(a, etag=None, lm=None)
    assert dl.main(_argv(out2, a, 4, "--allow-unverified"), transport=t2) == 0
    receipt = json.loads(Path(f"{out2}.receipt.json").read_text())
    assert receipt["verified_by"] == "size-only" and receipt["unvalidated"] == ["no_validator"]


def test_last_modified_change_mid_download_is_remote_changed(tmp_path):
    a, b = os.urandom(20_000), os.urandom(20_000)
    out = tmp_path / "obj.bin"
    t = ValidatorTransport(a, etag=None, payload2=b, flip_after=2)
    assert dl.main(_argv(out, a), transport=t, workers=1) == 1
    assert _state(out).state is S.REMOTE_CHANGED and not out.exists()
    _check_invariants(out, a)


@pytest.mark.parametrize("case", ["plan_out_too_big", "plan_garbage", "plan_not_object",
                                  "plan_bad_jobs", "state_truncated"])
def test_corrupt_inputs_are_refused_not_tracebacks(tmp_path, capsys, case):
    """Probes 5-7: StopIteration, JSONDecodeError and CorruptState used to escape main()."""
    payload = os.urandom(20_000)
    out = tmp_path / "obj.bin"
    argv = _argv(out, payload, 4, "--sha256", _sha(payload))
    if case == "state_truncated":
        assert dl.main(argv, transport=FakeTransport(payload)) == 0
        Path(f"{out}.state.json").write_text('{"schema": 1')
    else:
        plan = {"plan_out_too_big": json.dumps({"url": "u", "total": len(payload),
                                                "jobs": [list(j) for j in dl.plan(0, len(payload), 4)]}),
                "plan_garbage": "{not json",
                "plan_not_object": "[1, 2]",
                "plan_bad_jobs": json.dumps({"url": "u", "total": len(payload), "jobs": [[0, 1]]})}[case]
        Path(f"{out}.plan").write_text(plan)
        out.write_bytes(payload + b"extra" if case == "plan_out_too_big" else payload[:10])
    before = sorted(p.name for p in tmp_path.iterdir())
    assert dl.main(argv, transport=FakeTransport(payload)) == 1
    assert "refused:" in capsys.readouterr().err
    assert sorted(p.name for p in tmp_path.iterdir() if not p.name.endswith(".lock")) == \
        [n for n in before if not n.endswith(".lock")]  # nothing deleted or created


@pytest.mark.parametrize("bad", [["0"], ["-5"], ["10", "0"], ["10", "--attempts", "0"],
                                 ["10", "--sha256", "abc"]])
def test_bad_arguments_are_usage_errors(tmp_path, bad):
    with pytest.raises(SystemExit) as e:
        dl.main(["u", str(tmp_path / "x"), *bad], transport=NoNetwork())
    assert e.value.code == 2


def test_out_replaced_after_done_is_detected(tmp_path, capsys):
    """Probe 12: same-size bytes swapped in after DONE used to stay SOURCE_DONE and rc 0."""
    from alztest import data as adata
    payload = os.urandom(20_000)
    out = tmp_path / "obj.bin"
    assert dl.main(_argv(out, payload), transport=FakeTransport(payload)) == 0
    assert adata.hypoad_source_state(out)[0] == "SOURCE_DONE"
    receipt = json.loads(Path(f"{out}.receipt.json").read_text())
    assert receipt["stamp"] == dl.file_stamp(str(out))
    mtime = out.stat().st_mtime_ns
    out.write_bytes(os.urandom(len(payload)))
    os.utime(out, ns=(mtime, mtime + 1_000_000_000))
    capsys.readouterr()
    assert dl.main(_argv(out, payload), transport=NoNetwork()) == 1
    assert "not the file recorded at DONE" in capsys.readouterr().err
    assert adata.hypoad_source_state(out) == ("SOURCE_CHANGED", None)
    assert adata.download_receipt(out) is None


def test_sweep_temps_never_touches_a_resumable_assembly(tmp_path, monkeypatch):
    """Review: the assembly used to be OUT.partial, which sweep_temps deletes when aged."""
    payload = os.urandom(20_000)
    out = tmp_path / "obj.bin"

    def fault(point, n={"k": 0}):
        if point == "after_state":
            n["k"] += 1
            if n["k"] == 2:
                raise Killed(point)
    monkeypatch.setattr(dl, "_fault", fault)
    with pytest.raises(Killed):
        dl.main(_argv(out, payload, 4, "--sha256", _sha(payload)),
                transport=FakeTransport(payload), workers=1)
    monkeypatch.setattr(dl, "_fault", lambda point: None)
    assert _state(out).state is S.ASSEMBLING and Path(f"{out}{dl.PARTIAL_SUFFIX}").exists()
    old = os.path.getmtime(out.parent) - 7200
    for p in tmp_path.iterdir():
        os.utime(p, (old, old))
    assert ds.sweep_temps(tmp_path, dry_run=True) == []
    assert ds.sweep_temps(tmp_path) == []
    t = FakeTransport(payload)
    assert dl.main(_argv(out, payload), transport=t) == 0
    assert out.read_bytes() == payload and t.requests == 0
