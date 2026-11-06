"""Read-only CFGen checkpoint/count adapter qualification; no new training."""
import datetime
import hashlib
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
from fit_adapter import torch, np
from generation import trace
from sampling_context import Context
from state_fingerprint import fingerprint

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    path = ROOT / 'runs/CAL_GPU0_V1_matched_cfgen_f0_cfgen_s0/fit/FM/final.ckpt'
    declared = adapter.read(ROOT / 'analysis/CAL_GPU0_V1_matched_cfgen_f0_cfgen_s0-result.json')
    assert declared['candidate']['checkpoint_path'] == str(path)
    assert declared['candidate']['checkpoint_SHA256'] == sha(path)
    model = adapter.restore(path, 'cpu')
    state = fingerprint(model.state_dict())
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    features = fingerprint({k: f.state_dict() for k, f in model.feature_embeddings.items()})
    ctx = Context(SCIENCE / 'caches/fold0', 'cpu')
    cases = []
    with torch.no_grad():
        for stage in range(4):
            seed, condition, schedule = trace(ctx, 0, 0, stage)
            first = next(adapter.iter_counts(ctx, model, schedule, condition, seed))
            second = next(adapter.iter_counts(ctx, model, schedule, condition, seed))
            assert first[0] == second[0] == 0
            for x, y in zip(first[1:], second[1:]):
                np.testing.assert_array_equal(x, y)
            pcs, counts, totals = first[1:]
            assert pcs.shape == (5, 400, 50) and counts.shape == (1, 400, 2000)
            assert totals.shape == (2000,)
            assert np.isfinite(pcs).all() and np.isfinite(totals).all()
            assert np.isfinite(counts).all() and (counts >= 0).all()
            assert np.equal(counts, np.floor(counts)).all()
            cases.append(dict(stage=stage, PC_shape=list(pcs.shape), count_shape=list(counts.shape),
                              exact_repeated_native_counts_and_PCs=True,
                              count_SHA256=hashlib.sha256(counts.tobytes()).hexdigest()))
    after = model.state_dict()
    # Native FM.sample assigns self.node=NeuralODE(...self.denoising_model).
    # This registers additional names for the SAME tensors. Require every
    # original tensor and the exact alias set/storage to remain unchanged.
    assert set(before).issubset(after)
    assert fingerprint({k: after[k] for k in before}) == state
    aliases = {f'node.vf.vf.model.{k[len("denoising_model."):]}': k
               for k in before if k.startswith('denoising_model.')}
    assert set(after) - set(before) == set(aliases)
    for alias, original in aliases.items():
        assert after[alias].data_ptr() == after[original].data_ptr()
        assert torch.equal(after[alias], before[original])
    assert fingerprint({k: f.state_dict() for k, f in model.feature_embeddings.items()}) == features
    assert not torch.cuda.is_initialized() and not adapter.NETWORK
    protocol = adapter.read(HERE / 'protocol.json')
    sources = [Path(__file__), HERE / 'fit_adapter.py', HERE / 'protocol.json',
               *[Path(p) for p in protocol['source_SHA256']]]
    proof = dict(status='READ_ONLY_NATIVE_CFGEN_FOUR_STAGE_FULLWIDTH_COUNT_ADAPTER_REPLAY_PASS',
                 UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 historical_checkpoint=str(path), historical_checkpoint_SHA256=sha(path),
                 cases=cases, native_model_fingerprint=state, external_feature_fingerprint=features,
                 native_model_and_external_features_unchanged=True,
                 native_sampler_added_registered_aliases=len(aliases),
                 sampler_aliases_share_original_storage_exactly=True,
                 original_state_keys_removed_or_changed=False,
                 fresh_training_performed=False, historical_weights_used_for_training=False,
                 optimizer_updates=0, evaluation_target_values_read=False,
                 GPU_allocation_created=False, CUDA_initialized=False,
                 runtime_deployment_qualified=False,
                 source_SHA256={str(p): sha(p) for p in sources},
                 limitation='Read-only historical checkpoint diagnostic only. This verifies the generalized native count adapter, stage queries, common training-derived exposure traces, counts-to-PCA projection and deterministic replay; it does not replace fresh full-endpoint guarded benchmark training or qualify the resource runtime.')
    out = HERE / 'CPU_saved_sampling_qualification.json'
    assert not out.exists()
    out.write_text(json.dumps(proof, indent=2) + '\n')
    print(json.dumps({k: v for k, v in proof.items() if k != 'source_SHA256'}))

if __name__ == '__main__':
    main()
