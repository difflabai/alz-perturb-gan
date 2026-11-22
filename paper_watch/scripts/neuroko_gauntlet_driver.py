"""Restart-resilient NeuroKO gauntlet driver.

_prepare is the expensive step (backed read of the 6.3 GB object);
cache its output per seed to npz, then call the test module's OWN
run_once with _prepare monkeypatched to the cache. Results append
to a JSONL as each seed completes, so a VM restart loses at most
one seed's fits. The test file itself is untouched.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import test_neuroko_perturbation as T  # noqa: E402

CACHE = Path(__file__).resolve().parent.parent / "data"
OUT = CACHE / "neuroko_gauntlet_results.jsonl"

_real_prepare = T._prepare


def cached_prepare(seed, **kw):
    f = CACHE / f"neuroko_prep_seed{seed}.npz"
    if f.exists():
        z = np.load(f, allow_pickle=True)
        return z["Xtr"], z["ytr"], z["Xte"], z["yte"], list(z["conds"])
    out = _real_prepare(seed, **kw)
    np.savez_compressed(f, Xtr=out[0], ytr=out[1], Xte=out[2],
                        yte=out[3], conds=np.array(out[4]))
    return out


T._prepare = cached_prepare

done = set()
if OUT.exists():
    for line in OUT.read_text().splitlines():
        if line.strip():
            done.add(json.loads(line)["seed"])
print(f"already done: {sorted(done)}", flush=True)
for seed in (0, 1, 2):
    if seed in done:
        continue
    r = T.run_once(seed=seed, steps=300)
    with OUT.open("a") as fh:
        fh.write(json.dumps(r) + "\n")
    print("SEED DONE", json.dumps(r), flush=True)
print("GAUNTLET COMPLETE", flush=True)
