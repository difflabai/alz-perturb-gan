"""Choose the offpc gene set from TRAINING donors only (fold 0).

Input: raw ``snapens`` readout of the retained native EMPF fold0/seed0 fit (trained on the
25 fold-0 training donors) and the fold-0 TRAINING counts.  Held-out generated rows and the
held-out cache directory are never read.

For each training stage s:  gap_s = mean_{train donors d of stage s}(gen mean_log - real mean_log) / sd_g
(= mean gap of the standardized per-cell log expression z that the critic's PCs project).
Off-PC part r_s = gap_s - Wo Wo^T gap_s.  score_g = sqrt(mean_s r_{s,g}^2).
Selected: PPARG (declared a priori from the README diagnostic) + the K-1 highest-score other genes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import scipy.sparse as sp

K = 16
FORCED = 'PPARG'


def mean_log_by_donor(counts, labels, donors):
    x = counts.tocsr().astype(np.float64)
    lib = np.asarray(x.sum(1)).ravel()
    x = sp.diags(1e4 / np.maximum(lib, 1)) @ x
    x.data = np.log1p(x.data)
    return np.stack([np.asarray(x[np.flatnonzero(labels == d)].mean(0)).ravel() for d in donors])


def select(gen_path, cache, k=K):
    cache = Path(cache)
    space = np.load(cache/'space.npz')
    genes = space['genes'].astype(str)
    meta = np.load(cache/'train/metadata.npz')
    donors = meta['donors'].astype(str)
    real = mean_log_by_donor(sp.load_npz(cache/'train/counts.npz'), meta['donor'].astype(str), donors)
    g = np.load(gen_path)
    train_rows = np.flatnonzero(~g['is_heldout'])
    if list(g['targets'][train_rows]) != list(donors) or not np.array_equal(g['genes'].astype(str), genes):
        raise ValueError('generated train targets / genes differ from the training cache')
    gen = g['mean_log'][train_rows].astype(np.float64).mean(1)
    gstage = g['stage'][train_rows]
    W = space['W'].astype(np.float64)
    gap_z = (gen - real) / space['sd'].astype(np.float64)
    stages = sorted(set(gstage.tolist()))
    gap = np.stack([gap_z[gstage == s].mean(0) for s in stages])
    off = gap - (gap @ W) @ W.T
    score = np.sqrt((off ** 2).mean(0))
    order = [int(i) for i in np.argsort(-score) if genes[i] != FORCED]
    chosen = [FORCED] + [str(genes[i]) for i in order[:k-1]]
    j = int(np.flatnonzero(genes == FORCED)[0])
    pparg = {f'stage{s}': dict(n_train_donors=int((gstage == s).sum()),
                               real_mean_log=float(real[gstage == s, j].mean()), gen_mean_log=float(gen[gstage == s, j].mean()),
                               gap_z=float(gap[i, j]), offpc_gap_z=float(off[i, j]),
                               offpc_share=float(off[i, j] / gap[i, j]) if gap[i, j] != 0 else None)
             for i, s in enumerate(stages)}
    return dict(genes=chosen, k=k, forced=FORCED, rule=__doc__.strip(),
                scores={str(genes[i]): float(score[i]) for i in [j] + order[:k-1]},
                score_rank_of_PPARG=int((score > score[j]).sum()) + 1,
                median_score_all_genes=float(np.median(score)),
                pparg_training_gap=pparg,
                inputs=dict(generated=str(gen_path), generated_sha256=hashlib.sha256(Path(gen_path).read_bytes()).hexdigest(),
                            train_counts_sha256=hashlib.sha256((cache/'train/counts.npz').read_bytes()).hexdigest(),
                            space_sha256=hashlib.sha256((cache/'space.npz').read_bytes()).hexdigest()),
                heldout_data_read=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--generated', required=True)
    p.add_argument('--cache', default='/ml2/alz/runs/improve_20261109/caches/hvg2000/fold0')
    p.add_argument('--out', default=str(Path(__file__).with_name('selected_genes.json')))
    a = p.parse_args()
    v = select(a.generated, a.cache)
    Path(a.out).write_text(json.dumps(v, indent=2) + '\n')
    print(json.dumps({k: v[k] for k in ('genes', 'score_rank_of_PPARG', 'pparg_training_gap')}, indent=1))
