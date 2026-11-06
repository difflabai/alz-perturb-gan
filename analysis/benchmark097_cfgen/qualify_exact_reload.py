"""Read-only historical full-model proof for the future fresh-fit verifier."""
import datetime
import json
import os
import sys
from pathlib import Path

assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
ROOT = Path('/ml2/alz')
HERE = Path(__file__).resolve().parent
SCIENCE = Path('/ml2/codex-storage/future-workspaces/2026-11-03/task-2/alz-audit/core-comparison-20261103')
sys.path.extend([str(SCIENCE.parent / 'isolated/cfgen-cu124-py311/lib/python3.11/site-packages')])
sys.path[:0] = [str(HERE), str(ROOT / 'cfgen_gpu_qualification_runtime'), str(SCIENCE)]
import fit_adapter as adapter
from native_reload import restore_checkpoint, load_complete

def main():
    fit = ROOT / 'runs/CAL_GPU0_V1_matched_cfgen_f0_cfgen_s0/fit'
    receipt = adapter.read(fit / 'fit_receipt.json')
    model, proof = restore_checkpoint(fit / 'FM/final.ckpt', 'cpu', receipt['FM_checkpoint_SHA256'])
    assert proof['native_global_step'] == receipt['FM_updates'] == 85500
    assert proof['native_epoch'] == receipt['FM_epochs'] == 1500
    assert not adapter.torch.cuda.is_initialized()
    # Historical receipt has its own old protocol pin and no generalized
    # updates_per_epoch field. It must not pass new-model deployment validation.
    try:
        load_complete(SCIENCE / 'caches/fold0', fit, 0, 'cpu')
    except (AssertionError, KeyError):
        pass
    else:
        raise AssertionError('Historical receipt accepted as a new generalized fit')
    result = dict(status='NATIVE_CFGEN_REGISTERED_AND_EXTERNAL_FEATURE_EXACT_CPU_RESTORE_PASS',
                  UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  historical_checkpoint=str(fit / 'FM/final.ckpt'), **proof,
                  original_historical_receipt_rejected_as_new_benchmark_fit=True,
                  fresh_training_performed=False, historical_weights_used_for_training=False,
                  evaluation_targets_read=False, CUDA_initialized=False,
                  GPU_allocation_created=False, runtime_deployment_qualified=False,
                  source_SHA256={str(p): adapter.sha(p) for p in [Path(__file__), HERE / 'native_reload.py', HERE / 'fit_adapter.py', HERE / 'protocol.json']},
                  limitation='Read-only historical checkpoint verifies strict registered state, unregistered categorical embeddings, frozen encoder, explicit CPU device and actual85500-step/1500-epoch native clocks. New complete fresh-fit receipts and guarded CUDA runtime remain to be exercised by the benchmark; this is not a new training run.')
    out = HERE / 'CPU_exact_reload_qualification.json'
    assert not out.exists()
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'source_SHA256'}))

if __name__ == '__main__':
    main()
