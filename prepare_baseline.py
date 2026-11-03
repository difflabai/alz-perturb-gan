"""Generate the matched hierarchical-resampling calibration baseline on CPU."""
import json
import os
from pathlib import Path
import sys

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    root = Path('/ml2/alz')
    integration = Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-11/gpu0-coexistence/integration')
    sys.path.insert(0, str(integration))
    import binding as b
    deployment = b.read(integration.parent.parent / 'alz-gpu0-one-use-deployment.json')
    auth, _ = b.verify(deployment['authorization_path'])
    sys.path.extend(map(str, b.native_site_roots(auth['cell'], auth['files_SHA256'])))
    sys.path.insert(0, str(b.SCIENCE))
    sys.path.insert(0, str(root))
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from pg_adapter import Context
    from generation import iter_resampling
    from calibration import evaluate, average, dump
    ctx = Context(b.SCIENCE / 'caches/fold0', 'cpu')
    out = root / 'artifacts/calibration_resampling_seed0'
    records = [evaluate(ctx, 'hierarchical_resampling', 0, rep,
                        lambda schedule, c, s: iter_resampling(ctx, schedule, c, s),
                        out / f'r{rep}') for rep in [0, 1]]
    result = average(records)
    result.update(status='CPU_BASELINE_COMPLETED', model='hierarchical_resampling',
                  new_generative_model_fitted=False, CUDA_context_initialized=torch.cuda.is_initialized())
    assert not torch.cuda.is_initialized()
    dump(root / 'baseline_calibration.json', result)
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
