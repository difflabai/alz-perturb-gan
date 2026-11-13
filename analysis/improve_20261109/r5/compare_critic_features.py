"""Apply the predeclared initial critic-feature gate, without layer search."""
import argparse
import json
from pathlib import Path
import numpy as np
import trainer as T

ROOT = T.RUNS/'smoke_r5/architecture_pass/critic_features'


def remove_train_stage_effect(prediction, stage, train):
    """Each predictor gets its OWN train-fitted stage effect.

    Subtracting one common target-derived stage reference from both models
    could manufacture cross-model agreement even for constant predictors.
    """
    return prediction-stage @ np.linalg.lstsq(stage[train], prediction[train], rcond=None)[0]


def compare(root=ROOT):
    results = [json.loads((root/f'fold0/seed{s}/evaluation.json').read_text()) for s in (0, 1)]
    controls = ('train_mean', 'conditions', 'untrained')
    improvement = {name: [1-r['aggregate']['heldout']['rmse']['learned']/
                            r['aggregate']['heldout']['rmse'][name] for r in results] for name in controls}
    repeat = [r['feature_diagnostics']['learned']['heldout']['stage_residual_repeatability'] for r in results]
    passes = bool(all(min(v) > 0 for v in improvement.values())
                  and all(np.mean(improvement[name]) >= .01 for name in ('train_mean', 'untrained'))
                  and min(repeat) >= .8)
    rows = [{row['donor']: row for row in r['records']} for r in results]
    donors = sorted(set(rows[0]) & set(rows[1]))
    if set(rows[0]) != set(rows[1]):
        raise ValueError('cross-seed donor mismatch')
    held = np.array([rows[0][d]['heldout'] for d in donors])
    predictions = [np.array([r[d]['learned_predicted_profile'] for d in donors]) for r in rows]
    deviations = [p-p[~held].mean(0) for p in predictions]
    # Existing records store each known stage's train-fitted gene reference.
    # Identical references identify the categorical stage without reading any
    # held-out target counts. Only membership is used, not reference values.
    reference = np.array([rows[0][d]['stage_gene_profile'] for d in donors])
    _, labels = np.unique(reference, axis=0, return_inverse=True)
    stage = np.eye(int(labels.max())+1)[labels]
    stage_residual = [remove_train_stage_effect(p, stage, ~held) for p in predictions]
    def correlation(pair):
        if min(np.var(value[held]) for value in pair) < 1e-20:
            return None
        return float(np.corrcoef(pair[0][held].ravel(), pair[1][held].ravel())[0, 1])
    compact = [{key: r[key] for key in ('fold', 'seed', 'aggregate', 'readouts', 'feature_diagnostics',
               'depth_standardized_feature_mse', 'GAN_weights_unchanged', 'fit_final_sha256',
               'source_sha256', 'protocol_sha256', 'wall_seconds')} for r in results]
    return {'seeds': compact, 'comparator_sha256': T.sha(__file__),
            'relative_rmse_improvement': improvement,
            'stage_residual_repeatability': repeat, 'passes_initial_gate': passes,
            'heldout_prediction_deviation_cross_seed_correlation': correlation(deviations),
            'heldout_stage_residual_prediction_cross_seed_correlation': correlation(stage_residual),
            'decision': ('continue fixed probe to folds1/2, both seeds' if passes else
                         'stop this fixed critic-feature probe; do not promote a patient encoder'),
            'scope': 'two initial fits on same development split; discriminator fingerprint, not biological axes'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--out', type=Path, default=T.RUNS/'results_r5/architecture_pass/critic_features_comparison.json')
    a = p.parse_args()
    value = compare(a.root)
    T.dump(a.out, value)
    print(json.dumps(value, indent=2))
