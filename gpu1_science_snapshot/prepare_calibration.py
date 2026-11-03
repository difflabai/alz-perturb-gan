"""CPU-only calibration projection using the immutable optimization-fitted space."""
import datetime
import json
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path('/ml2/alz')
SCIENCE = Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/core-comparison-20261103')
INTEGRATION = Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-11/gpu0-coexistence/integration')


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    started = time.monotonic()
    sys.path.insert(0, str(INTEGRATION))
    import binding as b
    deployment = b.read(INTEGRATION.parent.parent / 'alz-gpu0-one-use-deployment.json')
    auth, _ = b.verify(deployment['authorization_path'])
    assert auth['cell'] == {'fold': 0, 'method': 'pg', 'seed': 0}
    sys.path.extend(map(str, b.native_site_roots(auth['cell'], auth['files_SHA256'])))
    sys.path.insert(0, str(SCIENCE))
    from contracts import B, FOLDS, FOLDS_SHA256, fold, sha
    import numpy as np
    import scipy.sparse as sp
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from fold_data import read_donors, project_chunks
    from bench_core import load_space
    f = fold(0)
    donors = f['internal_calibration_donors']
    assert len(donors) == 4
    assert not set(donors) & set(f['optimization_donors'] + f['evaluation_donors'])
    cache = SCIENCE / 'caches/fold0'
    space = load_space(cache)
    raw = read_donors(donors, allowed=donors,
                      meta=json.loads((B / 'rebuilt/mg_meta.json').read_text()))
    counts = raw['counts'][:, space.hvg].tocsr()
    pcs = project_chunks(space, counts)
    stages = np.rint(raw['raw_c'][:, 3] * 3).astype(int)
    assert set(stages.tolist()) == set(range(4))
    summaries = np.stack([space.summary(pcs[raw['donor'] == d]) for d in donors])
    out = ROOT / 'data/calibration_fold0'
    out.mkdir(parents=True, exist_ok=False)
    sp.save_npz(out / 'counts_hvg.npz', counts)
    np.save(out / 'real_pcs.npy', pcs)
    np.savez_compressed(out / 'metadata.npz',
                        **{k: raw[k] for k in ['cell', 'donor', 'type', 'donors', 'raw_c']},
                        stage=stages)
    np.savez_compressed(out / 'real_summaries.npz', donors=raw['donors'],
                        stage=stages, summary=summaries)
    # Exactly375 independently selected real cells per stage. Replacement only
    # when that calibration donor has fewer than375available cells, recorded.
    rng = np.random.default_rng(20261103)
    reservoirs = []
    sampling = []
    for stage in range(4):
        index = int(np.flatnonzero(stages == stage)[0])
        pool = np.flatnonzero(raw['donor'] == donors[index])
        replacement = len(pool) < 375
        selected = rng.choice(pool, size=375, replace=replacement)
        reservoirs.append(pcs[selected])
        sampling.append({'stage': stage, 'donor': donors[index],
                         'available_cells': len(pool), 'selected_cells': 375,
                         'replacement': replacement})
    np.savez_compressed(out / 'cell_reservoir.npz',
                        by_stage=np.stack(reservoirs),
                        equal_stage=np.concatenate(reservoirs))
    assert not torch.cuda.is_initialized()
    manifest = {
        'status': 'CPU_CALIBRATION_PROJECTED_WITHOUT_REFITTING',
        'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'fold': 0, 'donors': donors, 'cells': len(pcs), 'stages': stages.tolist(),
        'source_counts': raw['sources'],
        'source_fold_cache': str(cache), 'source_fold_cache_SHA256': sha(cache / 'manifest.json'),
        'source_folds_SHA256': FOLDS_SHA256,
        'all_transforms_and_classifier_fit_only_on_optimization_donors': True,
        'new_model_or_transform_fit': False,
        'evaluation_counts_read': False, 'audit_or_development_counts_read': False,
        'calibration_zero_HVG_rows_retained': int((np.asarray(counts.sum(1)).ravel() == 0).sum()),
        'cell_reservoir_sampling': sampling,
        'calibration_cell_metric': 'Equal375cells per stage; fixed real reservoir shared by every candidate.',
        'files_SHA256': {str(p.relative_to(out)): sha(p) for p in out.iterdir() if p.is_file()},
        'source_SHA256': {str(p): sha(p) for p in [Path(__file__), SCIENCE / 'fold_data.py', SCIENCE / 'contracts.py', B / 'bench_core.py', FOLDS]},
        'seconds': time.monotonic() - started,
        'peak_RSS_KiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'CUDA_context_initialized': False,
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({k: v for k, v in manifest.items()
                      if k not in ['source_counts', 'files_SHA256', 'source_SHA256']}, indent=2))


if __name__ == '__main__':
    main()
