"""Parallel ranged download of the bytes of a file not yet on disk.

Usage: python parallel_download.py URL OUT TOTAL_BYTES [NPARTS]

Parts land as <OUT>.partNN and are appended to OUT in order once every
part has exactly the expected size. Each request's bytes are kept only
when the final response is 206 with a Content-Range that starts at the
requested offset and names TOTAL_BYTES as the object size; an HTTP error
(curl --fail), a 200 that ignores Range, or a wrong range is discarded.
--max-filesize caps each request at the bytes still wanted, so a server
that ignores Range is cut off at the Content-Length check rather than
streaming the whole object into every part.

There is no wall-clock limit. curl aborts a stalled transfer (below
10 kB/s for 5 min) and the part resumes from the bytes it already holds,
so slow but steady connections finish. A part gives up after `attempts`
consecutive requests that add no bytes. Rerunning resumes: finished and
partial parts are kept, OUT's existing prefix is kept. <OUT>.plan records
the part layout; a rerun with a different layout stops instead of mixing
parts from two layouts.

For GSE342635_HypoAD.h5ad, TOTAL_BYTES is 51,107,823,722; write OUT to a
gitignored directory with more than 60 GB free (parts plus the assembled file).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

CURL = ["curl", "--fail", "--silent", "--show-error", "--location",
        "--retry", "5", "--retry-delay", "5", "--retry-connrefused",
        "--connect-timeout", "60", "--speed-limit", "10000", "--speed-time", "300"]


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


def _size(path: str) -> int:
    return os.path.getsize(path) if os.path.exists(path) else 0


def _remove(*paths: str) -> None:
    for path in paths:
        if os.path.exists(path):
            os.remove(path)


def final_response(header_file: str) -> tuple[int | None, str | None]:
    """Status and Content-Range of the last response in a curl -D dump."""
    status, crange = None, None
    if not os.path.exists(header_file):
        return status, crange
    with open(header_file, "rb") as fh:
        for line in fh.read().decode("latin-1").splitlines():
            if line.startswith("HTTP/"):
                parts = line.split()
                status = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
                crange = None
            elif line.lower().startswith("content-range:"):
                crange = line.split(":", 1)[1].strip()
    return status, crange


def range_ok(status, crange, start: int, total: int | None) -> bool:
    if status != 206 or not crange:
        return False
    m = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+|\*)", crange)
    if not m or int(m.group(1)) != start:
        return False
    return total is None or m.group(3) == str(total)


def fetch(url: str, out: str, job, attempts: int = 3, curl=CURL,
          total: int | None = None) -> tuple[int, bool]:
    k, lo, hi = job
    p = f"{out}.part{k:02d}"
    chunk, hdr = f"{p}.chunk", f"{p}.hdr"
    want = hi - lo + 1
    stalls = 0
    while True:
        have = _size(p)
        if have > want:
            _remove(p)
            have = 0
        if have == want:
            _remove(chunk, hdr)
            return k, True
        if stalls >= attempts:
            _remove(chunk, hdr)
            return k, False
        start = lo + have
        _remove(chunk, hdr)
        rc = subprocess.run(
            [*curl, "--max-filesize", str(hi - start + 1), "-D", hdr,
             "-r", f"{start}-{hi}", "-o", chunk, url], check=False).returncode
        status, crange = final_response(hdr)
        got = _size(chunk)
        if range_ok(status, crange, start, total) and 0 < got <= hi - start + 1:
            with open(p, "ab") as fout, open(chunk, "rb") as fin:
                shutil.copyfileobj(fin, fout, 8 * 1024 * 1024)
        else:
            got = 0
        if got == 0:
            stalls += 1
        if have + got != want:
            print(f"part {k:02d}: curl rc={rc} status={status} range={crange!r} "
                  f"kept={got} have={have + got} want={want}", file=sys.stderr, flush=True)


def assemble(out: str, jobs) -> None:
    with open(out, "ab") as fout:
        for k, _, _ in jobs:
            p = f"{out}.part{k:02d}"
            with open(p, "rb") as fin:
                while True:
                    b = fin.read(8 * 1024 * 1024)
                    if not b:
                        break
                    fout.write(b)
            os.remove(p)


def main(argv=None, curl=CURL, workers: int = 8) -> int:
    argv = sys.argv[1:] if argv is None else argv
    url, out, total = argv[0], argv[1], int(argv[2])
    nparts = int(argv[3]) if len(argv) > 3 else 16
    have = os.path.getsize(out) if os.path.exists(out) else 0
    jobs = plan(have, total, nparts)
    layout = {"url": url, "total": total, "jobs": [list(j) for j in jobs]}
    plan_file = f"{out}.plan"
    if os.path.exists(plan_file):
        with open(plan_file) as fh:
            if json.load(fh) != layout:
                print(f"{plan_file} records a different part layout; remove it and "
                      f"{out}.part* to start over", file=sys.stderr, flush=True)
                return 1
    else:
        with open(plan_file, "w") as fh:
            json.dump(layout, fh)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(lambda j: fetch(url, out, j, curl=curl, total=total), jobs))
    print("results:", results, flush=True)
    if not all(ok for _, ok in results):
        print("INCOMPLETE", flush=True)
        return 1
    assemble(out, jobs)
    size = os.path.getsize(out)
    print("assembled size:", size, "expected:", total, flush=True)
    if size != total:
        return 1
    _remove(plan_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
