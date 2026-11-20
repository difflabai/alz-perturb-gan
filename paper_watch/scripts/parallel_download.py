"""Parallel ranged download of the bytes of a file not yet on disk.

Usage: python parallel_download.py URL OUT TOTAL_BYTES [NPARTS]

Parts land as <OUT>.partNN and are appended to OUT in order once every
part has exactly the expected size. curl runs with --fail, so an HTTP
error response is never saved as data, and with --retry for transient
errors. A failed or wrong-sized part is deleted before it is retried.
Rerunning resumes: finished parts are kept, OUT's existing prefix is kept.

For GSE342635_HypoAD.h5ad, TOTAL_BYTES is 51,107,823,722; write OUT to a
gitignored directory with more than 60 GB free (parts plus the assembled file).
"""
from __future__ import annotations

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

CURL = ["curl", "--fail", "--silent", "--show-error", "--location",
        "--retry", "5", "--retry-delay", "5", "--retry-connrefused"]


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


def fetch(url: str, out: str, job, attempts: int = 3, curl=CURL,
          timeout: int = 3600) -> tuple[int, bool]:
    k, lo, hi = job
    p = f"{out}.part{k:02d}"
    want = hi - lo + 1
    for _ in range(attempts):
        if os.path.exists(p) and os.path.getsize(p) == want:
            return k, True
        if os.path.exists(p):
            os.remove(p)
        try:
            rc = subprocess.run([*curl, "-r", f"{lo}-{hi}", "-o", p, url],
                                check=False, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            rc = -1
        if rc == 0 and os.path.exists(p) and os.path.getsize(p) == want:
            return k, True
        print(f"part {k:02d}: curl rc={rc}, size="
              f"{os.path.getsize(p) if os.path.exists(p) else 0} want={want}",
              file=sys.stderr, flush=True)
        if os.path.exists(p):
            os.remove(p)
    return k, False


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
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(lambda j: fetch(url, out, j, curl=curl), jobs))
    print("results:", results, flush=True)
    if not all(ok for _, ok in results):
        print("INCOMPLETE", flush=True)
        return 1
    assemble(out, jobs)
    size = os.path.getsize(out)
    print("assembled size:", size, "expected:", total, flush=True)
    return 0 if size == total else 1


if __name__ == "__main__":
    sys.exit(main())
