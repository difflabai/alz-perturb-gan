"""Audit a plotted gene against raw training counts and frozen inference summaries.

Read-only CPU analysis. No model loading, generation, fitting, or evaluation data.
Original retained research artifacts are required; public output is anonymous.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.sparse import load_npz

ROOT = Path(__file__).resolve().parents[2]
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def audit(gene, state, output):
    plan_path = ROOT / 'analysis/primamente_inference_hypothesis_plan_20261108.json'
    plan = json.loads(plan_path.read_bytes())
    cache = Path(plan['fitting_cache'])
    for name in ['manifest.json', 'train/counts_hvg.npz', 'train/metadata.npz']:
        assert sha(cache / name) == plan['input_SHA256'][str(cache / name)]
    with np.load(cache / 'train/metadata.npz', allow_pickle=False) as z:
        j = list(z['genes']).index(gene)
        donor, ids, types = z['donor'], z['donors'], z['type']
        stages = np.rint(z['raw_c'][:, 3] * 3).astype(int)
    counts = load_npz(cache / 'train/counts_hvg.npz')
    values = counts[:, j].toarray().ravel()
    log = np.log1p(10000 * values.astype(np.float64) / np.maximum(np.asarray(counts.sum(1)).ravel(), 1))
    groups = [0, 2, 4]  # pooled, then the two prespecified eligible subtypes
    real = np.zeros((4, 3)); detected = np.zeros_like(real)
    for stage in range(4):
        for k, group in enumerate(groups):
            means, fractions = [], []
            for d in np.flatnonzero(stages == stage):
                keep = (donor == ids[d]) & ((types == group - 1) if group else True)
                if keep.sum() >= (20 if group else 1):
                    means.append(log[keep].mean())
                    fractions.append((values[keep] > 0).mean())
            assert len(means) >= 3
            real[stage, k], detected[stage, k] = np.mean(means), np.mean(fractions)
    completion = json.loads((state / 'completed.json').read_bytes())
    model = np.full((5, 8, 4, 3), np.nan); model_detected = np.full_like(model, np.nan)
    pins = []
    for row in completion['stage_replicates']:
        path = Path(row['path']); pin = sha(path)
        assert pin == row['SHA256']; pins.append(pin)
        with np.load(path, allow_pickle=False) as z:
            key = row['seed'], row['replicate'], row['stage']
            model[key] = z['sums'][groups, j] / z['n'][groups]
            model_detected[key] = z['detection_counts'][groups, j] / z['n'][groups]
    assert len(pins) == 160 and np.isfinite(model).all()
    profiles = json.loads((ROOT / 'docs/evidence/stage-profiles.json').read_bytes())['profiles'][gene]
    np.testing.assert_allclose(real[:, 1:].mean(1), profiles['real'], atol=3e-7, rtol=0)
    np.testing.assert_allclose(model.mean(1)[:, :, 1:].mean(2), profiles['model_by_seed'], atol=3e-8, rtol=0)
    model_mean, detection_mean = model.mean((0, 1)), model_detected.mean((0, 1))
    def result(real_log, model_log, real_detection, model_detection):
        return dict(observed_mean_log_expression=real_log.tolist(), model_mean_log_expression=model_log.tolist(),
                    observed_nonzero_fraction=real_detection.tolist(), model_nonzero_fraction=model_detection.tolist(),
                    observed_conditional_nonzero_mean_log_expression=(real_log / real_detection).tolist(),
                    model_conditional_nonzero_mean_log_expression=(model_log / model_detection).tolist())
    out = dict(gene=gene, stage_order=['Not AD', 'Low', 'Intermediate', 'High'],
               units='Per-cell log1p(panel CP10k); equal observed-donor weights.',
               pooled=result(real[:, 0], model_mean[:, 0], detected[:, 0], detection_mean[:, 0]),
               within_subtype=result(real[:, 1:].mean(1), model_mean[:, 1:].mean(1), detected[:, 1:].mean(1), detection_mean[:, 1:].mean(1)),
               frozen_case_hashes_verified=len(pins), frozen_case_SHA256=pins,
               plotted_profiles_reconstructed=True, original_training_counts_used=True,
               no_evaluation_donors_read=True, new_training_fits=0, new_optimizer_updates=0,
               limitations=['Excess zero counts describe a model discrepancy, not its optimization or biological cause.',
                            'Generated subtype assignments use a classifier; pooled results do not depend on those assignments.',
                            'Common generated depth templates differ from stage-specific observed depths.'],
               input_SHA256={name:sha(cache / name) for name in ['manifest.json', 'train/counts_hvg.npz', 'train/metadata.npz']},
               source_SHA256=sha(__file__))
    output.write_text(json.dumps(out, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'gene':gene, 'status':'RAW_COUNTS_AND_ALL_160_FROZEN_CASES_AUDIT_PASS', 'within_subtype':out['within_subtype']}))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gene', default='PPARG')
    parser.add_argument('--state', type=Path, default=ROOT / 'artifacts/primamente-t005-gene-inference-recovery-v1-20261108')
    parser.add_argument('--output', type=Path, default=ROOT / 'docs/evidence/pparg-fit-audit.json')
    args = parser.parse_args()
    audit(args.gene, args.state, args.output)
