"""Accuracy of the frozen fold0 TRAIN-only subtype classifier on observed cells (context for generated labels)."""
from __future__ import annotations

import os
os.environ.update(OMP_NUM_THREADS='3', CUDA_VISIBLE_DEVICES='')
import json
from pathlib import Path
import sys

import numpy as np
import scipy.sparse as sp
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402


def main(out):
    torch.set_num_threads(3)
    space = np.load(C.CACHE / 'space.npz', allow_pickle=False)
    state = torch.load(C.CACHE / 'type_classifier.pt', map_location='cpu', weights_only=True)
    w = {k: v.double().numpy() for k, v in state.items()}
    result = {}
    for split in ('train', 'heldout'):
        X = sp.load_npz(C.CACHE / split / 'counts.npz').toarray().astype(np.float64)
        meta = np.load(C.CACHE / split / 'metadata.npz', allow_pickle=False)
        Z = ((C.log_cp10k(X) - space['mean']) / space['sd']) @ space['W'] / space['pc_sd']
        h = np.maximum(Z @ w['net.0.weight'].T + w['net.0.bias'], 0)
        logits = h @ w['net.2.weight'].T + w['net.2.bias']
        pred = logits.argmax(1); y = meta['type'].astype(int)
        prob = np.exp(logits - logits.max(1, keepdims=True)); prob /= prob.sum(1, keepdims=True)
        recall = {C.TYPES[t]: (float((pred[y == t] == t).mean()) if (y == t).any() else None) for t in range(6)}
        result[split] = dict(cells=int(len(y)), accuracy=float((pred == y).mean()),
                             balanced_accuracy=float(np.mean([v for v in recall.values() if v is not None])),
                             recall_by_type=recall, mean_confidence=float(prob.max(1).mean()),
                             observed_fraction=np.bincount(y, minlength=6).astype(float).__truediv__(len(y)).tolist(),
                             predicted_fraction=np.bincount(pred, minlength=6).astype(float).__truediv__(len(y)).tolist())
    result['note'] = 'train = apparent (in-sample) accuracy; heldout = donor-held-out accuracy'
    result['classifier_sha256'] = C.sha(C.CACHE / 'type_classifier.pt')
    C.write_json(out, result)
    print(json.dumps({k: {kk: v[kk] for kk in ('accuracy', 'balanced_accuracy', 'mean_confidence')}
                      for k, v in result.items() if isinstance(v, dict) and 'accuracy' in v}, indent=1))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
