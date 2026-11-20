#!/usr/bin/env python3
"""Write empirical-resampling reference archives in the generated.npz contract.

Uses the repository baseline generator unchanged (analysis/improve_20261109/r5
baselines.generate: 'resample_blind' = stage-blind, 'resample' = stage-matched;
M=32 donors x K=256 depth-thinned training-donor cells per target, 64 stored
as PCs; per-target stream stable_seed('baseline-gen', panel, fold, donor)).
The retained results_r3 baseline NPZs hold only scores, not cells.
"""
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(k, '3')
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', required=True); p.add_argument('--out-dir', required=True)
    p.add_argument('--r5', default=str(HERE.parent / 'improve_20261109/r5'))
    a = p.parse_args()
    sys.path.insert(0, str(Path(a.r5).resolve()))
    import numpy as np
    import evaluate as E
    import baselines as B
    real = E.load_real(a.cache)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    for name in ('resample_blind', 'resample'):
        path = out / f'{name}.npz'
        if path.exists():
            raise SystemExit(f'{path} exists; inspect before regenerating')
        t = time.time()
        arrays, info = B.generate(name, real)
        E.save_npz(path, **arrays)
        receipt = dict(baseline=name, info=info, generated_SHA256=sha(path), seconds=time.time() - t,
                       cache_manifest_sha256=sha(Path(a.cache) / 'manifest.json'),
                       baselines_sha256=sha(Path(a.r5) / 'baselines.py'), evaluate_sha256=sha(Path(a.r5) / 'evaluate.py'),
                       source='training donors only; target excluded; repository baseline code unchanged')
        path.with_name(path.stem + '.receipt.json').write_text(json.dumps(receipt, indent=1, default=str) + '\n')
        print(name, receipt['generated_SHA256'], round(receipt['seconds'], 1), flush=True)


if __name__ == '__main__':
    main()
