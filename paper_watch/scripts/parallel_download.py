"""Parallel ranged download as an explicit, journaled state machine.

Usage: python parallel_download.py URL OUT TOTAL_BYTES [NPARTS]
                                   [--sha256 HEX] [--allow-unverified]
                                   [--attempts N] [--no-space-check]

For GSE342635_HypoAD.h5ad, TOTAL_BYTES is 51,107,823,722; write OUT to a
gitignored directory with more than 60 GB free.

State lives in ``OUT.state.json`` (plus ``OUT.state.json.journal.jsonl``),
written through ``alztest._durable_state``. It replaces the old
``OUT.plan``, which is still read once for migration. Files:

    OUT.partNN       bytes of part NN, always a prefix of that part's range
    OUT.assembling   the assembled prefix; never written at OUT's final name.
                     Deliberately NOT named *.partial: durable_state.sweep_temps
                     deletes aged *.partial* files, and during assembly this
                     file is the only copy of the parts already appended
    OUT              appears only by renaming a verified OUT.assembling
    OUT.receipt.json {url, etag, size, sha256, expected_sha256, verified_by,
                     unvalidated, stamp {size, mtime_ns}, finished_at},
                     written at DONE

States and transitions (``MACHINE`` below is the authoritative table):

    ABSENT -plan-> PLANNED            HEAD records total, ETag, Last-Modified;
                                      free-space preflight before anything
                                      is written
    PLANNED -fetch-> FETCHING         every range request sends If-Range
                                      (a strong ETag, else Last-Modified; a
                                      weak ETag can never match If-Range)
    FETCHING -part_done-> FETCHING    a part reached its exact size
    FETCHING -parts_complete-> PARTS_COMPLETE
    FETCHING -stall-> STALLED         a part saw `attempts` CONSECUTIVE
                                      requests that added no bytes; any
                                      progress resets the count
    STALLED -resume-> FETCHING        the next run
    PLANNED|FETCHING|STALLED -remote_changed-> REMOTE_CHANGED
                                      a 200 instead of 206, or a changed
                                      ETag (HEAD or response); terminal
    PARTS_COMPLETE -assemble-> ASSEMBLING(next_part, out_size_at_part_start)
    ASSEMBLING -append-> ASSEMBLING   truncate OUT.assembling to
                                      out_size_at_part_start, append part k,
                                      fsync, record next_part=k+1, and only
                                      then unlink part k; so a kill or ENOSPC
                                      at any point is resumed by redoing part
                                      k and no part is ever lost
    ASSEMBLING -assembled-> ASSEMBLED
    ASSEMBLED -verify-> VERIFIED {sha256, verified_by}
                                      the recorded expected_sha256 must match
    ASSEMBLED -verify_failed-> VERIFY_FAILED  the bytes stay at OUT.assembling
    VERIFY_FAILED -reverify-> ASSEMBLED       only when a DIFFERENT --sha256 is
                                      given (a typo in the first one)
    VERIFIED -finish-> DONE           rename OUT.assembling -> OUT, write receipt
    *-adopt_prefix-> same state       journaled rename of a legacy OUT prefix
                                      to OUT.assembling (migration only)
    *-expect-> same state             record a --sha256 first given on a
                                      resumed run (before VERIFIED)

The expected sha256 is part of the state: it is recorded by the run that
first passes --sha256, a later run that passes a different one is refused
(except from VERIFY_FAILED, see reverify), and a resume without the flag
still verifies against the recorded one.

Every byte must be tied to ONE remote version. Bytes fetched with If-Range
against a recorded validator are; two cases are not, and both need either
--sha256 (verified_by='sha256') or an explicit --allow-unverified
(verified_by='size-only', recorded in the state and the receipt as
``unvalidated``): the server sends neither ETag nor Last-Modified
('no_validator'), or bytes were adopted from the old tool, which fetched
without If-Range ('legacy_bytes'). Without one of the two flags such a run
is refused before anything is written.

A rerun of a DONE download prints 'already complete' and exits 0 when OUT
still matches the receipt's size+mtime stamp (and --sha256, if given,
matches the recorded sha256); otherwise it exits 1. The whole run holds an
exclusive lock on OUT (``OUT.lock``); a second instance exits 75
(EX_TEMPFAIL) at once instead of interleaving with the first.

Migration (no state file yet): an ``OUT.plan`` written by the old tool is
adopted read-only: its part layout is reused, a prefix at OUT is renamed
once to OUT.assembling (journaled ``adopt_prefix``; it is never deleted),
and a legacy assembly that crashed half-way is resumed at the part it
reached. An OUT of exactly TOTAL_BYTES with no plan is adopted as ASSEMBLED
and verified. The old plan is renamed to ``OUT.plan.legacy`` afterwards.
Part files whose layout is unknown (no state file and no plan, or indices
the plan does not have) are refused, never reinterpreted.

Exit codes: 0 done, 1 failed or refused (the message says which state and
what to do), 75 another instance holds the lock.

Fault injection for tests: PARALLEL_DOWNLOAD_FAULT=<point>[:<nth>] makes
the process exit with 137 the nth time it reaches <point> (see _fault).
"""
from __future__ import annotations

import argparse
import enum
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import NamedTuple

PW = Path(__file__).resolve().parent.parent
if str(PW) not in sys.path:
    sys.path.insert(0, str(PW))

from alztest import _durable_state as ds  # noqa: E402

CURL = ["curl", "--fail", "--silent", "--show-error", "--location",
        "--retry", "5", "--retry-delay", "5", "--retry-connrefused",
        "--connect-timeout", "60", "--speed-limit", "10000", "--speed-time", "300"]

STATE_SUFFIX = ".state.json"
RECEIPT_SUFFIX = ".receipt.json"
PARTIAL_SUFFIX = ".assembling"  # not *.partial*: see the module docstring
LEGACY_PLAN_SUFFIX = ".plan"
EX_BUSY = 75
COPY_BLOCK = 8 * 1024 * 1024
FAULT_ENV = "PARALLEL_DOWNLOAD_FAULT"
FAULT_POINTS = ("part_appended", "mid_append", "after_append", "after_state",
                "after_rename", "after_receipt")


class S(enum.Enum):
    PLANNED = "PLANNED"
    FETCHING = "FETCHING"
    PARTS_COMPLETE = "PARTS_COMPLETE"
    ASSEMBLING = "ASSEMBLING"
    ASSEMBLED = "ASSEMBLED"
    VERIFIED = "VERIFIED"
    DONE = "DONE"
    STALLED = "STALLED"
    REMOTE_CHANGED = "REMOTE_CHANGED"
    VERIFY_FAILED = "VERIFY_FAILED"


TRANSITIONS = {
    (S.PLANNED, "fetch"): S.FETCHING,
    (S.PLANNED, "remote_changed"): S.REMOTE_CHANGED,
    (S.FETCHING, "part_done"): S.FETCHING,
    (S.FETCHING, "parts_complete"): S.PARTS_COMPLETE,
    (S.FETCHING, "stall"): S.STALLED,
    (S.FETCHING, "remote_changed"): S.REMOTE_CHANGED,
    (S.STALLED, "resume"): S.FETCHING,
    (S.STALLED, "remote_changed"): S.REMOTE_CHANGED,
    (S.PARTS_COMPLETE, "assemble"): S.ASSEMBLING,
    (S.ASSEMBLING, "append"): S.ASSEMBLING,
    (S.ASSEMBLING, "assembled"): S.ASSEMBLED,
    (S.ASSEMBLED, "verify"): S.VERIFIED,
    (S.ASSEMBLED, "verify_failed"): S.VERIFY_FAILED,
    (S.VERIFIED, "finish"): S.DONE,
    (S.VERIFY_FAILED, "reverify"): S.ASSEMBLED,
    # migration only: the journaled rename of a legacy prefix
    (S.PLANNED, "adopt_prefix"): S.PLANNED,
    (S.FETCHING, "adopt_prefix"): S.FETCHING,
    (S.ASSEMBLING, "adopt_prefix"): S.ASSEMBLING,
    (S.ASSEMBLED, "adopt_prefix"): S.ASSEMBLED,
}
# A --sha256 first given on a resumed run is recorded before verification.
for _s in (S.PLANNED, S.FETCHING, S.STALLED, S.PARTS_COMPLETE, S.ASSEMBLING, S.ASSEMBLED):
    TRANSITIONS[(_s, "expect")] = _s

MACHINE = ds.Machine("parallel_download", S, S.PLANNED, {S.DONE, S.REMOTE_CHANGED}, TRANSITIONS)


class Refused(Exception):
    """A precondition failed; nothing was changed. The message says why."""


def _fault(point: str) -> None:
    """os._exit(137) at the configured occurrence of ``point`` (tests patch this)."""
    spec = os.environ.get(FAULT_ENV)
    if not spec:
        return
    name, _, nth = spec.partition(":")
    if name != point:
        return
    with _fault_lock:
        _fault_counts[point] = _fault_counts.get(point, 0) + 1
        if _fault_counts[point] == (int(nth) if nth else 1):
            os._exit(137)


_fault_lock = threading.Lock()
_fault_counts: dict[str, int] = {}


def _write(fd: int, data) -> None:
    """Write all of ``data`` to ``fd`` (tests wrap this to inject ENOSPC)."""
    view = memoryview(data)
    while view:
        view = view[os.write(fd, view):]


# ------------------------------------------------------------------ paths

def part_path(out: str, k: int) -> str:
    return f"{out}.part{k:02d}"


def _size(path: str) -> int:
    try:
        return os.path.getsize(path)
    except FileNotFoundError:
        return 0


def _remove(*paths: str) -> None:
    for path in paths:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


def _fsync_dir(path: str) -> None:
    fd = os.open(os.path.dirname(os.path.abspath(path)) or ".", os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def file_stamp(path: str) -> dict:
    """Size and mtime in ns; os.replace keeps both, so a stamp taken at DONE
    tells a later reader whether OUT is still the file that was verified."""
    st = os.stat(path)
    return {"size": st.st_size, "mtime_ns": st.st_mtime_ns}


def if_range_validator(etag: str | None, last_modified: str | None) -> str | None:
    """The If-Range value: a strong ETag, else Last-Modified, else None.

    RFC 9110 compares If-Range entity tags strongly, so a weak ETag (W/...)
    never matches and a compliant server would answer every range with 200.
    """
    if etag and not etag.startswith("W/"):
        return etag
    return last_modified or None


_PART_RX = re.compile(r"\.part(\d+)(\.chunk|\.hdr)?$")


def part_files(out: str) -> dict[str, int]:
    """Every OUT.partNN[.chunk|.hdr] on disk, mapped to its part index."""
    d = os.path.dirname(os.path.abspath(out))
    base = os.path.basename(out)
    found = {}
    try:
        names = os.listdir(d)
    except FileNotFoundError:
        return found
    for name in names:
        if name.startswith(base):
            m = _PART_RX.fullmatch(name[len(base):])
            if m:
                found[os.path.join(d, name)] = int(m.group(1))
    return found


def plan(have: int, total: int, nparts: int) -> list[tuple[int, int, int]]:
    if have >= total:
        raise ValueError(f"nothing to fetch: have {have} of {total} bytes")
    part = (total - have + nparts - 1) // nparts
    jobs = []
    for k in range(nparts):
        lo = have + k * part
        hi = min(total - 1, lo + part - 1)
        if lo > hi:
            break
        jobs.append((k, lo, hi))
    return jobs


# ------------------------------------------------------------------ HTTP

def parse_headers(header_file: str) -> dict:
    """Status and headers (lower-case names) of the LAST response in a curl -D dump."""
    resp: dict = {"status": None}
    try:
        with open(header_file, "rb") as fh:
            text = fh.read().decode("latin-1")
    except FileNotFoundError:
        return resp
    for line in text.splitlines():
        if line.startswith("HTTP/"):
            parts = line.split()
            resp = {"status": int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None}
        elif ":" in line:
            name, value = line.split(":", 1)
            resp[name.strip().lower()] = value.strip()
    return resp


def final_response(header_file: str) -> tuple[int | None, str | None]:
    """Status and Content-Range of the last response in a curl -D dump."""
    h = parse_headers(header_file)
    return h["status"], h.get("content-range")


def range_ok(status, crange, start: int, total: int | None) -> bool:
    if status != 206 or not crange:
        return False
    m = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+|\*)", crange)
    if not m or int(m.group(1)) != start:
        return False
    return total is None or m.group(3) == str(total)


class Probe(NamedTuple):
    status: int | None
    total: int | None
    etag: str | None
    last_modified: str | None


class Response(NamedTuple):
    rc: int
    status: int | None
    crange: str | None
    etag: str | None


class CurlTransport:
    """HEAD probe and ranged GETs with curl (honours the proxy environment)."""

    def __init__(self, curl=CURL):
        self.curl = list(curl)

    def probe(self, url: str, hdr: str) -> Probe:
        _remove(hdr)
        subprocess.run([*self.curl, "--head", "-D", hdr, "-o", os.devnull, url],
                       check=False, stdout=subprocess.DEVNULL)
        h = parse_headers(hdr)
        length = h.get("content-length", "")
        if h["status"] == 200 and length.isdigit():
            _remove(hdr)
            return Probe(200, int(length), h.get("etag"), h.get("last-modified"))
        # Some proxies refuse HEAD; a one-byte range read carries the same facts.
        _remove(hdr)
        subprocess.run([*self.curl, "-D", hdr, "-r", "0-0", "-o", os.devnull, url],
                       check=False, stdout=subprocess.DEVNULL)
        h = parse_headers(hdr)
        _remove(hdr)
        m = re.fullmatch(r"bytes \d+-\d+/(\d+)", h.get("content-range", ""))
        return Probe(h["status"], int(m.group(1)) if m else None, h.get("etag"),
                     h.get("last-modified"))

    def get_range(self, url: str, start: int, hi: int, if_range: str | None,
                  chunk: str, hdr: str) -> Response:
        extra = ["-H", f"If-Range: {if_range}"] if if_range else []
        rc = subprocess.run(
            [*self.curl, *extra, "--max-filesize", str(hi - start + 1), "-D", hdr,
             "-r", f"{start}-{hi}", "-o", chunk, url], check=False).returncode
        h = parse_headers(hdr)
        return Response(rc, h["status"], h.get("content-range"), h.get("etag"))


# ------------------------------------------------------------------ fetching

def _append_file(src: str, dst: str) -> int:
    """Append src to dst durably; returns the bytes appended."""
    n = 0
    fd = os.open(dst, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        with open(src, "rb") as fin:
            while True:
                b = fin.read(COPY_BLOCK)
                if not b:
                    break
                _write(fd, b)
                n += len(b)
        os.fsync(fd)
    finally:
        os.close(fd)
    return n


def fetch_part(transport, url: str, out: str, job, *, attempts: int, total: int | None,
               etag: str | None, if_range: str | None, stop: threading.Event,
               log=None) -> str:
    """Fetch one part until it has its exact size: 'ok', 'stalled',
    'remote_changed' or 'stopped'. ``attempts`` CONSECUTIVE requests that
    add no bytes give up; any progress resets the count."""
    k, lo, hi = job
    p = part_path(out, k)
    chunk, hdr = f"{p}.chunk", f"{p}.hdr"
    want = hi - lo + 1
    stalls = 0
    try:
        while True:
            have = _size(p)
            if have > want:  # cannot come from this tool (appends are capped)
                raise Refused(f"{p} holds {have} bytes, more than its {want}-byte range")
            if have == want:
                return "ok"
            if stalls >= attempts:
                return "stalled"
            if stop.is_set():
                return "stopped"
            start = lo + have
            _remove(chunk, hdr)
            resp = transport.get_range(url, start, hi, if_range, chunk, hdr)
            if resp.status == 200 or (etag and resp.etag and resp.etag != etag):
                # If-Range fell through (or the validator moved): the object
                # changed, or the server ignores ranges. Keep nothing.
                if log:
                    log(f"part {k:02d}: status={resp.status} etag={resp.etag!r} "
                        f"(recorded {etag!r}): remote object changed")
                stop.set()
                return "remote_changed"
            got = _size(chunk)
            if not (range_ok(resp.status, resp.crange, start, total) and 0 < got <= hi - start + 1):
                got = 0
            if got:
                _append_file(chunk, p)
                stalls = 0
                _fault("part_appended")
            else:
                stalls += 1
            if have + got != want and log:
                log(f"part {k:02d}: rc={resp.rc} status={resp.status} range={resp.crange!r} "
                    f"kept={got} have={have + got} want={want} stalls={stalls}/{attempts}")
    finally:
        _remove(chunk, hdr)


# ------------------------------------------------------------------ the run

class Args(NamedTuple):
    url: str
    out: str
    total: int
    nparts: int = 16
    sha256: str | None = None
    attempts: int = 3
    space_check: bool = True
    allow_unverified: bool = False


class Download:
    def __init__(self, args: Args, transport, workers: int = 8, log=None):
        self.a = args
        self.out = args.out
        self.partial = args.out + PARTIAL_SUFFIX
        self.receipt = args.out + RECEIPT_SUFFIX
        self.legacy_plan = args.out + LEGACY_PLAN_SUFFIX
        self.transport = transport
        self.workers = workers
        self.log = log or (lambda msg: print(msg, file=sys.stderr, flush=True))
        self.sf = ds.StateFile(args.out + STATE_SUFFIX, MACHINE)
        self._tlock = threading.Lock()
        self._fetched = False

    # -- helpers ------------------------------------------------------------

    def _t(self, event, update=None):
        with self._tlock:
            return self.sf.transition(event, update=update)

    def _jobs(self, data) -> list[tuple[int, int, int]]:
        return [tuple(j) for j in data["parts"]]

    def _probe(self) -> Probe:
        pr = self.transport.probe(self.a.url, self.out + ".probe.hdr")
        if pr.total is None:
            raise Refused(f"{self.a.url}: probe got HTTP {pr.status} with no size; "
                          "the server must report Content-Length or Content-Range")
        return pr

    def _check_space(self, jobs) -> None:
        if not self.a.space_check:
            return
        total = self.a.total
        prefix = _size(self.partial) if os.path.exists(self.partial) else _size(self.out)
        on_disk = prefix + sum(_size(part_path(self.out, k)) for k, _, _ in jobs)
        biggest = max((hi - lo + 1 for _, lo, hi in jobs), default=0)
        # Peak use is the assembled file plus the part being appended: parts
        # are unlinked as they are appended, so parts and assembly never
        # both hold the whole object.
        need = total + biggest - on_disk
        free = shutil.disk_usage(os.path.dirname(os.path.abspath(self.out))).free
        if free < need:
            raise Refused(
                f"not enough free space in {os.path.dirname(os.path.abspath(self.out))}: "
                f"need {need} bytes (total {total} + largest part {biggest} - "
                f"{on_disk} already on disk), have {free}; pass --no-space-check to override")

    def _check_layout(self, data) -> None:
        if data.get("url") != self.a.url or data.get("total") != self.a.total:
            raise Refused(
                f"{self.sf.path} records url={data.get('url')!r} total={data.get('total')!r}, "
                f"not url={self.a.url!r} total={self.a.total}; remove {self.sf.path}, "
                f"its journal, {self.out}.part* and {self.partial} to start over")
        if data.get("nparts") != self.a.nparts:
            self.log(f"note: using the recorded layout of {len(data['parts'])} parts")

    def _remote_check(self, data) -> str | None:
        """Probe again before fetching: None if unchanged, else the reason."""
        pr = self._probe()
        if pr.total != self.a.total:
            return f"remote size is {pr.total}, recorded {self.a.total}"
        if data.get("etag") and pr.etag and pr.etag != data["etag"]:
            return f"remote ETag is {pr.etag!r}, recorded {data['etag']!r}"
        if (not data.get("etag") and data.get("last_modified") and pr.last_modified
                and pr.last_modified != data["last_modified"]):
            return f"remote Last-Modified is {pr.last_modified!r}, recorded {data['last_modified']!r}"
        return None

    # -- start / adoption ---------------------------------------------------

    def start(self):
        snap = self.sf.read()
        if snap is not None:
            self._check_layout(snap.data)
            self._retire_legacy_plan(snap)
            return snap
        if os.path.exists(self.partial):
            raise Refused(f"{self.partial} exists without {self.sf.path}; its origin is "
                          "unknown. Move it away to start over")
        legacy = os.path.exists(self.legacy_plan) or os.path.exists(self.out)
        if legacy:
            return self._adopt()
        self._refuse_orphans(set())
        pr = self._probe()
        if pr.total != self.a.total:
            raise Refused(f"server reports {pr.total} bytes, expected {self.a.total}")
        unvalidated = [] if (pr.etag or pr.last_modified) else ["no_validator"]
        self._require_consent(unvalidated)
        jobs = plan(0, self.a.total, self.a.nparts)
        self._check_space(jobs)
        data = {"url": self.a.url, "total": self.a.total, "nparts": self.a.nparts,
                "parts": [list(j) for j in jobs], "prefix": 0, "etag": pr.etag,
                "last_modified": pr.last_modified, "parts_done": [],
                **self._consent_data(unvalidated)}
        return self.sf.create(data)

    def _given_sha(self) -> str | None:
        return self.a.sha256.lower() if self.a.sha256 else None

    def _require_consent(self, unvalidated: list[str]) -> None:
        """Bytes not tied to one remote version need --sha256 or --allow-unverified."""
        if unvalidated and not self.a.sha256 and not self.a.allow_unverified:
            why = {"no_validator": "the server sends neither ETag nor Last-Modified, so a "
                                   "change of the remote object between requests cannot be seen",
                   "legacy_bytes": "bytes on disk were fetched by the old tool without "
                                   "If-Range, possibly from an earlier version of the object"}
            raise Refused("; ".join(why[u] for u in unvalidated) + ". Pass --sha256 HEX to "
                          "verify the result, or --allow-unverified to accept a size-only "
                          "check (recorded as verified_by='size-only' in the receipt). "
                          "Nothing was written")

    def _consent_data(self, unvalidated: list[str]) -> dict:
        return {"expected_sha256": self._given_sha(), "unvalidated": list(unvalidated),
                "allow_unverified": bool(self.a.allow_unverified)}

    def _refuse_orphans(self, known: set[int]) -> None:
        """Part files (and their .chunk/.hdr) whose index is not in ``known``
        have an unknown layout; with a plan, those of a planned part are fine."""
        orphans = sorted(p for p, k in part_files(self.out).items() if k not in known)
        if orphans:
            raise Refused(f"{orphans} exist without {self.sf.path}"
                          f"{' and are not in ' + self.legacy_plan if known else ''}; their "
                          "layout is unknown and they are never reinterpreted. Move them away "
                          "to start over")

    def infer_legacy(self) -> tuple[S, dict, dict]:
        """Read-only: (state, data, evidence) for a pre-state-machine layout."""
        total = self.a.total
        out_size = _size(self.out) if os.path.isfile(self.out) else None
        if os.path.exists(self.out) and not os.path.isfile(self.out):
            raise Refused(f"{self.out} exists and is not a regular file")
        if os.path.exists(self.partial):
            raise Refused(f"{self.partial} and a legacy {self.out} layout both exist; resolve by hand")
        evidence = {"out_size": out_size, "legacy_plan": None}
        if os.path.exists(self.legacy_plan):
            try:
                with open(self.legacy_plan) as fh:
                    layout = json.load(fh)
                if not isinstance(layout, dict):
                    raise ValueError("not a JSON object")
                url, ltotal = layout.get("url"), layout.get("total")
                jobs = [(int(k), int(lo), int(hi)) for k, lo, hi in layout.get("jobs") or []]
            except (ValueError, TypeError) as e:
                raise Refused(f"{self.legacy_plan} is not a readable plan ({e}); move it "
                              "away and resolve by hand") from e
            evidence["legacy_plan"] = layout
            if url != self.a.url or ltotal != total:
                raise Refused(
                    f"{self.legacy_plan} records a different part layout (url "
                    f"{url!r}, total {ltotal!r}); remove it and "
                    f"{self.out}.part* to start over")
            if not jobs or any(jobs[i + 1][1] != jobs[i][2] + 1 for i in range(len(jobs) - 1)) \
                    or jobs[-1][2] != total - 1 or jobs[0][1] < 0 \
                    or any(lo > hi for _, lo, hi in jobs) \
                    or len({k for k, _, _ in jobs}) != len(jobs):
                raise Refused(f"{self.legacy_plan}: parts do not tile [prefix, {total})")
            self._refuse_orphans({k for k, _, _ in jobs})
            have = jobs[0][1]
            s = out_size or 0
            if s > total:
                raise Refused(f"{self.out} has {s} bytes, more than {total}; resolve by hand")
            parts = {k: _size(part_path(self.out, k)) for k, _, _ in jobs
                     if os.path.exists(part_path(self.out, k))}
            evidence["part_sizes"] = {str(k): v for k, v in parts.items()}
            data = {"url": self.a.url, "total": total, "nparts": self.a.nparts,
                    "parts": [list(j) for j in jobs], "prefix": have, "etag": None,
                    "last_modified": None, "parts_done": [],
                    "prefix_pending": out_size is not None}
            if s < have:
                raise Refused(f"{self.out} has {s} bytes but {self.legacy_plan} was planned "
                              f"from a {have}-byte prefix")
            if s == have:
                return S.FETCHING, data, evidence
            # s > have: the old assembler crashed after appending some parts.
            boundaries = {hi + 1: i for i, (_, _, hi) in enumerate(jobs)}
            if s in boundaries:
                nxt, start = boundaries[s] + 1, s
            else:
                idx = next(i for i, (_, lo, hi) in enumerate(jobs) if lo < s <= hi)
                nxt, start = idx, jobs[idx][1]
            for i, (k, lo, hi) in enumerate(jobs[nxt:], nxt):
                if parts.get(k) != hi - lo + 1:
                    raise Refused(f"{self.out} ends inside the assembly but "
                                  f"{part_path(self.out, k)} is missing or short; resolve by hand")
            data.update(next_part=nxt, out_size_at_part_start=start,
                        parts_done=[k for k, _, _ in jobs])
            if nxt == len(jobs):
                return S.ASSEMBLED, data, evidence
            return S.ASSEMBLING, data, evidence
        # No plan: OUT is a prefix (the old tool resumed from it) or the whole object.
        # Part files here would come from a layout nobody recorded; refuse them
        # rather than read them as prefixes of a freshly planned layout.
        self._refuse_orphans(set())
        if out_size > total:
            raise Refused(f"{self.out} has {out_size} bytes, more than {total}")
        data = {"url": self.a.url, "total": total, "nparts": self.a.nparts,
                "prefix": out_size, "etag": None, "last_modified": None,
                "parts_done": [], "prefix_pending": True}
        if out_size == total:
            data.update(parts=[], next_part=0, out_size_at_part_start=total)
            return S.ASSEMBLED, data, evidence
        data["parts"] = [list(j) for j in plan(out_size, total, self.a.nparts)]
        return S.PLANNED, data, evidence

    def _adopt(self):
        state, data, evidence = self.infer_legacy()
        # Everything the old tool fetched was fetched without If-Range.
        unvalidated = ["legacy_bytes"]
        self._require_consent(unvalidated)  # before any network access
        if state in (S.PLANNED, S.FETCHING):
            pr = self._probe()
            if pr.total != self.a.total:
                raise Refused(f"server reports {pr.total} bytes, expected {self.a.total}")
            data["etag"], data["last_modified"] = pr.etag, pr.last_modified
            if not (pr.etag or pr.last_modified):
                unvalidated.append("no_validator")
        self._require_consent(unvalidated)
        data.update(self._consent_data(unvalidated))
        if state in (S.PLANNED, S.FETCHING):
            self._check_space(self._jobs(data))
        snap = self.sf.adopt(lambda _dir: (state, evidence), data=data)
        self.log(f"adopted legacy layout as {snap.state.name}")
        self._retire_legacy_plan(snap)
        return snap

    def _retire_legacy_plan(self, snap) -> None:
        if snap.data.get("adopted") and os.path.exists(self.legacy_plan):
            os.replace(self.legacy_plan, self.legacy_plan + ".legacy")

    def adopt_prefix(self, snap):
        """Journaled, idempotent rename of a legacy OUT prefix to OUT.assembling."""
        if not snap.data.get("prefix_pending"):
            return snap
        if os.path.exists(self.out):
            if os.path.exists(self.partial):
                raise Refused(f"both {self.out} and {self.partial} exist; resolve by hand")
            os.replace(self.out, self.partial)
            _fsync_dir(self.out)
        elif not os.path.exists(self.partial) and snap.data.get("prefix"):
            raise Refused(f"the adopted prefix is neither at {self.out} nor at {self.partial}")
        size = _size(self.partial)
        if snap.state in (S.PLANNED, S.FETCHING) and size != snap.data["prefix"]:
            raise Refused(f"{self.partial} has {size} bytes, adopted prefix was "
                          f"{snap.data['prefix']}")

        def upd(d):
            d["prefix_pending"] = False
            d["prefix_adopted"] = {"from": self.out, "to": self.partial, "size": size}
        return self._t("adopt_prefix", upd)

    # -- phases -------------------------------------------------------------

    def fetch(self, snap):
        data = snap.data
        jobs = self._jobs(data)
        reason = self._remote_check(data)
        if reason:
            self.log(f"REMOTE_CHANGED: {reason}")
            return self._t("remote_changed", lambda d: d.update(reason=reason))
        self._check_space(jobs)
        if snap.state is S.PLANNED:
            snap = self._t("fetch")
        elif snap.state is S.STALLED:
            snap = self._t("resume", lambda d: {k: v for k, v in d.items() if k != "stalled_parts"})
        etag = data.get("etag")
        if_range = if_range_validator(etag, data.get("last_modified"))
        stop = threading.Event()
        done = set(data.get("parts_done", []))

        def one(job):
            res = fetch_part(self.transport, self.a.url, self.out, job, attempts=self.a.attempts,
                             total=self.a.total, etag=etag, if_range=if_range, stop=stop,
                             log=self.log)
            if res == "ok" and job[0] not in done:
                self._t("part_done", lambda d: d.update(
                    parts_done=sorted(set(d.get("parts_done", [])) | {job[0]})))
            return job[0], res

        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            results = list(ex.map(one, jobs))
        self.log(f"results: {results}")
        if any(r == "remote_changed" for _, r in results):
            return self._t("remote_changed", lambda d: d.update(
                reason="a ranged response came back as 200 or with a different ETag"))
        bad = [k for k, r in results if r != "ok"]
        if bad:
            return self._t("stall", lambda d: d.update(stalled_parts=bad))
        return self._t("parts_complete")

    def assemble(self, snap):
        jobs = self._jobs(snap.data)
        if snap.state is S.PARTS_COMPLETE:
            prefix = snap.data["prefix"]
            if _size(self.partial) < prefix:
                raise Refused(f"{self.partial} has {_size(self.partial)} bytes, prefix is {prefix}")
            snap = self._t("assemble", lambda d: d.update(next_part=0, out_size_at_part_start=prefix))
        while True:
            nxt, start = snap.data["next_part"], snap.data["out_size_at_part_start"]
            for k, _, _ in jobs[:nxt]:
                _remove(part_path(self.out, k))  # appended and recorded before a crash
            if nxt == len(jobs):
                break
            k, lo, hi = jobs[nxt]
            want = hi - lo + 1
            p = part_path(self.out, k)
            if start != lo or _size(p) != want:
                raise Refused(f"cannot append {p}: it has {_size(p)} of {want} bytes and "
                              f"the assembly is at byte {start} (part starts at {lo})")
            fd = os.open(self.partial, os.O_RDWR | os.O_CREAT, 0o644)
            try:
                if os.fstat(fd).st_size < start:
                    raise Refused(f"{self.partial} is shorter than the recorded {start} bytes")
                os.ftruncate(fd, start)  # idempotent: drops a half-appended part k
                os.lseek(fd, start, os.SEEK_SET)
                with open(p, "rb") as fin:
                    while True:
                        b = fin.read(COPY_BLOCK)
                        if not b:
                            break
                        half = len(b) // 2
                        _write(fd, b[:half])
                        _fault("mid_append")
                        _write(fd, b[half:])
                os.fsync(fd)
                if os.fstat(fd).st_size != start + want:
                    raise Refused(f"{self.partial} is {os.fstat(fd).st_size} bytes after "
                                  f"appending {p}, expected {start + want}")
            finally:
                os.close(fd)
            _fault("after_append")
            snap = self._t("append", lambda d: d.update(next_part=nxt + 1,
                                                        out_size_at_part_start=start + want))
            _fault("after_state")
            _remove(p)
        size = _size(self.partial)
        if size != self.a.total:
            raise Refused(f"assembled {size} bytes, expected {self.a.total}")
        return self._t("assembled", lambda d: d.update(assembled_size=size))

    def verify(self, snap):
        if (snap.data.get("unvalidated") and not snap.data.get("expected_sha256")
                and not snap.data.get("allow_unverified") and not self.a.allow_unverified):
            raise Refused(f"{self.out}: unvalidated bytes ({snap.data['unvalidated']}) and no "
                          "expected sha256; pass --sha256 HEX or --allow-unverified")
        for k, _, _ in self._jobs(snap.data):
            _remove(part_path(self.out, k))  # every part is in OUT.assembling by now
        if _size(self.partial) != self.a.total:
            raise Refused(f"{self.partial} has {_size(self.partial)} bytes, expected {self.a.total}")
        h = hashlib.sha256()
        with open(self.partial, "rb") as fh:
            for b in iter(lambda: fh.read(COPY_BLOCK), b""):
                h.update(b)
        sha = h.hexdigest()
        expected = snap.data.get("expected_sha256")
        if expected and sha != expected:
            self.log(f"VERIFY_FAILED: sha256 {sha} != expected {expected}")
            return self._t("verify_failed", lambda d: d.update(sha256=sha))
        unvalidated = snap.data.get("unvalidated") or []
        if expected:
            verified_by = "sha256"
        elif unvalidated:
            verified_by = "size-only"
            self.log(f"WARNING: {self.out} is verified by size only ({', '.join(unvalidated)}); "
                     "the receipt records verified_by='size-only'")
        else:
            verified_by = "validator"  # every byte was fetched under If-Range
        return self._t("verify", lambda d: d.update(sha256=sha, verified_by=verified_by))

    def finish(self, snap):
        d = snap.data
        if os.path.exists(self.partial):
            if os.path.exists(self.out):
                raise Refused(f"{self.out} appeared while {self.partial} awaits promotion")
            os.replace(self.partial, self.out)
            _fsync_dir(self.out)
        elif _size(self.out) != self.a.total:
            raise Refused(f"VERIFIED but neither {self.partial} nor a complete {self.out} exists")
        _fault("after_rename")
        finished_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        receipt = {"url": self.a.url, "etag": d.get("etag"),
                   "last_modified": d.get("last_modified"), "size": self.a.total,
                   "sha256": d["sha256"], "expected_sha256": d.get("expected_sha256"),
                   "verified_by": d.get("verified_by"), "unvalidated": d.get("unvalidated") or [],
                   "stamp": file_stamp(self.out), "finished_at": finished_at,
                   "state_file": os.path.basename(self.sf.path)}
        ds.atomic_write_json(self.receipt, receipt)
        _fault("after_receipt")
        return self._t("finish", lambda dd: dd.update(finished_at=finished_at))

    def apply_sha_arg(self, snap):
        """Reconcile --sha256 with the expected sha256 recorded in the state."""
        given = self._given_sha()
        if given is None:
            return snap
        st, recorded = snap.state, snap.data.get("expected_sha256")
        if st in (S.DONE, S.REMOTE_CHANGED):
            return snap  # DONE compares in run(); REMOTE_CHANGED has nothing to verify
        if st is S.VERIFY_FAILED:
            if given != recorded:
                self.log(f"re-verifying {self.partial} against the new --sha256 {given}")
                return self._t("reverify", lambda d: d.update(expected_sha256=given))
            return snap
        if st is S.VERIFIED:
            if given != snap.data.get("sha256"):
                raise Refused(f"{self.out} was verified with sha256 {snap.data.get('sha256')}, "
                              f"not the given {given}; nothing was promoted")
            return snap
        if recorded is not None and recorded != given:
            raise Refused(f"{self.sf.path} records expected sha256 {recorded}; this run "
                          f"passes {given}. Rerun with the recorded one (or none)")
        if recorded is None:
            return self._t("expect", lambda d: d.update(expected_sha256=given))
        return snap

    def _check_done(self, snap) -> int:
        try:
            with open(self.receipt) as fh:
                receipt = json.load(fh)
        except (FileNotFoundError, ValueError):
            receipt = None
        if not isinstance(receipt, dict) or _size(self.out) != self.a.total or not os.path.isfile(self.out):
            self.log(f"{self.sf.path} says DONE but {self.out} or its receipt is missing")
            return 1
        stamp = receipt.get("stamp")
        if receipt.get("sha256") != snap.data.get("sha256") or (
                stamp is not None and stamp != file_stamp(self.out)):
            self.log(f"{self.out} is not the file recorded at DONE (its size/mtime stamp or "
                     f"the receipt's sha256 differs from {self.receipt}); it was replaced or "
                     "modified afterwards. Move it away with its receipt, state and journal "
                     "to download again")
            return 1
        given = self._given_sha()
        if given is not None and given != snap.data.get("sha256"):
            self.log(f"{self.out}: DONE with sha256 {snap.data.get('sha256')}, which is not "
                     f"the given --sha256 {given}")
            return 1
        print(f"{self.out}: already complete (sha256 {snap.data.get('sha256')})", flush=True)
        return 0

    def run(self) -> int:
        snap = self.apply_sha_arg(self.start())
        while True:
            st = snap.state
            if st is S.DONE:
                return self._check_done(snap)
            if st is S.REMOTE_CHANGED:
                self.log(f"{self.out}: REMOTE_CHANGED ({snap.data.get('reason')}); the parts on "
                         f"disk belong to the old version. Remove {self.sf.path}, its journal, "
                         f"{self.out}.part* and {self.partial} to start over")
                return 1
            if st is S.VERIFY_FAILED:
                self.log(f"{self.out}: VERIFY_FAILED (sha256 {snap.data.get('sha256')} != "
                         f"{snap.data.get('expected_sha256')}); the bytes are kept at "
                         f"{self.partial}. If the expected sha256 was wrong, rerun with the "
                         f"right --sha256 to verify again; otherwise remove {self.sf.path}, its "
                         f"journal and {self.partial} to start over")
                return 1
            if snap.data.get("prefix_pending"):
                snap = self.adopt_prefix(snap)
                continue
            if st in (S.PLANNED, S.FETCHING, S.STALLED):
                if st is S.STALLED and self._fetched:
                    print("INCOMPLETE", flush=True)
                    self.log(f"STALLED parts {snap.data.get('stalled_parts')}; rerun to resume")
                    return 1
                self._fetched = True
                snap = self.fetch(snap)
            elif st in (S.PARTS_COMPLETE, S.ASSEMBLING):
                snap = self.assemble(snap)
                print("assembled size:", _size(self.partial), "expected:", self.a.total, flush=True)
            elif st is S.ASSEMBLED:
                snap = self.verify(snap)
            elif st is S.VERIFIED:
                snap = self.finish(snap)
            else:  # pragma: no cover - the table has no other states
                raise AssertionError(st)


def parse_args(argv) -> Args:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("url")
    ap.add_argument("out")
    ap.add_argument("total", type=int)
    ap.add_argument("nparts", type=int, nargs="?", default=16)
    ap.add_argument("--sha256", help="expected sha256 of the whole object (recorded in the state)")
    ap.add_argument("--allow-unverified", action="store_true",
                    help="accept a size-only check when bytes cannot be tied to one remote "
                         "version (no ETag/Last-Modified, or adopted from the old tool)")
    ap.add_argument("--attempts", type=int, default=3,
                    help="consecutive zero-byte requests before a part stalls")
    ap.add_argument("--no-space-check", action="store_true")
    a = ap.parse_args(argv)
    if a.total <= 0:
        ap.error(f"TOTAL_BYTES must be positive, got {a.total}")
    if a.nparts < 1:
        ap.error(f"NPARTS must be at least 1, got {a.nparts}")
    if a.attempts < 1:
        ap.error(f"--attempts must be at least 1, got {a.attempts}")
    if a.sha256 is not None and not re.fullmatch(r"[0-9a-fA-F]{64}", a.sha256):
        ap.error("--sha256 must be 64 hex digits")
    return Args(a.url, a.out, a.total, a.nparts, a.sha256, a.attempts, not a.no_space_check,
                a.allow_unverified)


def main(argv=None, curl=CURL, workers: int = 8, transport=None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    transport = transport if transport is not None else CurlTransport(curl)
    try:
        with ds.FileLock(args.out, blocking=False, purpose="parallel_download"):
            return Download(args, transport, workers).run()
    except ds.Busy as busy:
        print(f"{args.out} is being downloaded by another process "
              f"(owner {busy.owner}); exiting {EX_BUSY}", file=sys.stderr, flush=True)
        return EX_BUSY
    except Refused as why:
        print(f"refused: {why}", file=sys.stderr, flush=True)
        return 1
    except ds.DurableStateError as err:
        print(f"refused: {args.out}{STATE_SUFFIX}: {type(err).__name__}: {err}; nothing was "
              "deleted. Inspect the state file and its journal", file=sys.stderr, flush=True)
        return 1
    except OSError as err:
        print(f"error: {err}; nothing recorded is lost, rerun to resume",
              file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
