"""Parallel ranged download of the remaining bytes of a file whose
prefix is already downloaded. Parts land as <out>.partNN and are
appended in order on success."""
import subprocess, sys, os
from concurrent.futures import ThreadPoolExecutor

URL = sys.argv[1]
OUT = sys.argv[2]
TOTAL = int(sys.argv[3])
NPARTS = int(sys.argv[4]) if len(sys.argv) > 4 else 16

have = os.path.getsize(OUT) if os.path.exists(OUT) else 0
assert have < TOTAL, (have, TOTAL)
span = TOTAL - have
part = (span + NPARTS - 1) // NPARTS
jobs = []
for k in range(NPARTS):
    lo = have + k * part
    hi = min(TOTAL - 1, lo + part - 1)
    if lo > hi:
        break
    jobs.append((k, lo, hi))

def fetch(job):
    k, lo, hi = job
    p = f"{OUT}.part{k:02d}"
    want = hi - lo + 1
    for attempt in range(8):
        if os.path.exists(p) and os.path.getsize(p) == want:
            return (k, True)
        subprocess.run(["curl", "-sL", "-r", f"{lo}-{hi}", "-o", p, URL],
                       check=False, timeout=3600)
        if os.path.exists(p) and os.path.getsize(p) == want:
            return (k, True)
    return (k, False)

with ThreadPoolExecutor(max_workers=8) as ex:
    results = list(ex.map(fetch, jobs))
print("results:", results, flush=True)
if all(ok for _, ok in results):
    with open(OUT, "ab") as fout:
        for k, _, _ in [(j[0], 0, 0) for j in jobs]:
            p = f"{OUT}.part{k:02d}"
            with open(p, "rb") as fin:
                while True:
                    b = fin.read(8 * 1024 * 1024)
                    if not b:
                        break
                    fout.write(b)
            os.remove(p)
    print("assembled size:", os.path.getsize(OUT), "expected:", TOTAL, flush=True)
else:
    print("INCOMPLETE", flush=True)
    sys.exit(1)
