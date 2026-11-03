"""Exercise actual synthetic updates across the native20k schedule horizon."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

ROOT = Path('/ml2/alz')
B = Path('/ml2/codex-storage/future-workspaces/2026-10-27/task-2/benchmark')
SCIENCE = Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/core-comparison-20261103')


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    out = ROOT / 'artifacts/cpu_extension_qualification_isolated'
    out.mkdir(exist_ok=False)
    source = out / 'source'
    (source / 'model').mkdir(parents=True)
    copied = [ROOT / 'test_variant.py', *sorted((ROOT / 'model').glob('*.py'))]
    origins = {}
    for path in copied:
        target = source / path.relative_to(ROOT)
        shutil.copyfile(path, target)
        origins[str(target)] = {'original_path': str(path), 'SHA256': hashlib.sha256(target.read_bytes()).hexdigest()}
        assert origins[str(target)]['SHA256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    (out / 'copied-source.json').write_text(json.dumps(origins, indent=2) + '\n')
    sys.path[:0] = list(map(str, [source, source / 'model', SCIENCE, B,
                               B / 'current_base_20261029/runtime/lib/python3.12/site-packages',
                               B / '.venv312/lib/python3.12/site-packages']))
    import torch
    import test_variant as tests
    import variant_pg_adapter as variant
    import variant_pg_update as update
    assert Path(variant.__file__).parent == source / 'model'
    assert Path(tests.__file__).parent == source
    native = B / 'current_base_20261029/runtime/lib/python3.12/site-packages/particlegan/policy.py'
    immutable = [Path(__file__), native, *map(Path, origins)]
    source_pins = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in immutable}
    config = dict(z_dim=8, stage_balance=False, cell_z_dim=8, theta_init_scale=1., fourier_feature_scale=1.)
    update.configure(config)
    start = time.monotonic()
    with tempfile.TemporaryDirectory(dir=out) as directory:
        cache = Path(directory)
        (cache / 'manifest.json').write_text('{}')
        ctx = tests.fixture(cache)
        bundle, generators, rng, usage, streams = variant.build(ctx.mu, ctx.theta, 'cpu', 0, 0)
        assert bundle[0].total_steps == 20000
        for _ in range(20000):
            metrics = variant.advance(ctx, bundle, generators, rng, usage, execution_limit=40000)
            if metrics['step'] % 1000 == 0:
                progress = {'scope': 'SYNTHETIC_CPU_ONLY_NOT_ALZ_FIT', 'completed_updates': metrics['step'],
                            'elapsed_seconds': time.monotonic() - start, 'metrics': metrics}
                temporary = out / 'progress.tmp'
                temporary.write_text(json.dumps(progress, indent=2, allow_nan=False) + '\n')
                temporary.replace(out / 'progress.json')
        checkpoint = variant.snapshot(ctx, bundle, generators, rng, usage, 0, streams, 'synthetic')
        assert checkpoint['completed_updates'] == checkpoint['recipe']['total_steps'] == 20000
        path = out / 'synthetic_20000.pt'
        torch.save(checkpoint, path)
        expected_metrics = variant.advance(ctx, bundle, generators, rng, usage, execution_limit=40000)
        expected = variant.snapshot(ctx, bundle, generators, rng, usage, 0, streams, 'synthetic')
        rb, rg, rr, ru, rs = variant.restore(ctx, torch.load(path, map_location='cpu', weights_only=False))
        actual_metrics = variant.advance(ctx, rb, rg, rr, ru, execution_limit=40000)
        tests.same(expected_metrics, actual_metrics)
        tests.same(expected, variant.snapshot(ctx, rb, rg, rr, ru, 0, rs, 'synthetic'))
        assert actual_metrics['step'] == 20001 and not torch.cuda.is_initialized()
    assert source_pins == {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in immutable}
    result = {'status': 'SYNTHETIC_CPU_20000_TO_20001_CONTINUATION_BITWISE_VERIFIED',
              'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'synthetic_actual_updates': 20001,
              'native_recipe_schedule_horizon': 20000, 'caller_execution_limit': 40000,
              'complete_state_and_next_update_bitwise_equal_after_restore': True,
              'elapsed_seconds': time.monotonic() - start, 'variant_config': config,
              'synthetic_checkpoint_SHA256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'source_SHA256': source_pins, 'copied_source_origins': origins,
              'GPU_used': False, 'real_ALZ_fitting_updates': 0,
              'scope': 'Synthetic6donors/24cells/8genes only. Demonstrates native horizon and full-state continuation, not ALZ quality or GPU admission. No real calibration/evaluation data used.'}
    (ROOT / 'analysis/extension_cpu_qualification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
