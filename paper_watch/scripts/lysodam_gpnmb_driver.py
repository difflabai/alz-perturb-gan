import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import test_seaad_lysodam_coupling_readout as T
OUT = Path("data/lysodam_gpnmb_results.jsonl")
done = set()
if OUT.exists():
    for line in OUT.read_text().splitlines():
        if line.strip(): done.add(json.loads(line)["seed"])
for seed in (0, 1, 2):
    if seed in done: continue
    r = T.run_gpnmb(seed=seed, steps=300)
    with OUT.open("a") as fh: fh.write(json.dumps(r) + "\n")
    print("SEED DONE", json.dumps(r), flush=True)
print("GPNMB GAUNTLET COMPLETE", flush=True)
