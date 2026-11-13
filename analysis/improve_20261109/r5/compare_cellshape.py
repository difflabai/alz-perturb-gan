"""Apply the fixed adoption rule to the six paired count-shape read-outs."""
from pathlib import Path
import json
import sys
sys.dont_write_bytecode = True
import numpy as np
import evaluate as E
from pzero_compare import paired

RUNS = E.RUNS
OUT = RUNS / 'results_r5/default_pass'
METRICS = ['pb_rmse', 'meanlog_rmse_gene', 'nonzero_rmse_gene', 'logvar_rmse_gene',
           'pc_energy_rep', 'cell_energy_rep', 'pc_spread_rep', 'pc_coverage80_rep']


def loss(metric, value):
    if metric == 'pc_spread_rep':
        return abs(np.log(value))
    if metric == 'pc_coverage80_rep':
        return abs(value - .8)
    return value


def main():
    candidate, reference, biascal = {}, {}, {}
    neutral_diffs = []
    for fold in range(3):
        for seed in (0, 1):
            key = (fold, seed)
            p = OUT / 'fits/pbulk~r3#cellshape@0.5:biascal:stagehalf/hvg2000' / f'fold{fold}/seed{seed}.json'
            candidate[key] = E.headline(json.loads(p.read_text()))
            p = RUNS / 'smoke_r5/shape/zeval' / f'ctrl_f{fold}_s{seed}_zcal@biascal:stagehalf.json'
            historical = json.loads(p.read_text())['heldout']
            p = OUT / 'fits/pbulk~r3#cellshape@0:biascal:stagehalf/hvg2000' / f'fold{fold}/seed{seed}.json'
            reference[key] = E.headline(json.loads(p.read_text()))
            neutral_diffs.extend(abs(reference[key][m] - historical[m]) for m in historical if m in reference[key])
            p = RUNS / 'results_r3/fits/pbulk~r3#biascal@stagehalf/hvg2000' / f'fold{fold}/seed{seed}.json'
            biascal[key] = E.headline(json.loads(p.read_text()))
    neutral_maxdiff = max(neutral_diffs)
    assert neutral_maxdiff < 1e-12, neutral_maxdiff
    rep = {'protocol': 'PROTOCOL_DEFAULT_PASS_20261113.txt', 'neutral_maxdiff': neutral_maxdiff,
           'comparison': {}, 'means': {}, 'targets': {}, 'guards': {}, 'n_pairs': 6}
    for label, data in [('candidate', candidate), ('zcal_reference', reference), ('biascal_reference', biascal)]:
        rep['means'][label] = {m: float(np.mean([v[m] for v in data.values()])) for m in METRICS}
    rep['means']['resample'] = {m: float(np.mean([E.headline(json.loads((RUNS / 'results_r3/baselines/hvg2000' /
                     f'fold{k}/resample.json').read_text()))[m] for k in range(3)])) for m in METRICS}
    floors = [json.loads((RUNS / 'smoke_r5/shape' / f'real_fold{k}.json').read_text())['floor_route_A'] for k in range(3)]
    rep['means']['train_statistic_predictor'] = {
        'meanlog_rmse_gene': float(np.mean([f['meanlog']['A_raw_train_mean'] for f in floors])),
        'nonzero_rmse_gene': float(np.mean([f['nonzero']['A_raw_train_mean'] for f in floors]))}
    for m in METRICS:
        differences = {k: [loss(m, candidate[(k, s)][m]) - loss(m, reference[(k, s)][m]) for s in (0, 1)]
                       for k in range(3)}
        p = paired(differences)
        rep['comparison'][m] = p
        if m in ('meanlog_rmse_gene', 'nonzero_rmse_gene'):
            rep['targets'][m] = p['n_folds_lower'] == 3 and p['mean_diff'] < -2 * p['se_seed_noise']
        else:
            rep['guards'][m] = p['mean_diff'] <= 2 * p['se_seed_noise']
    rep['passes_initial_rule'] = all(rep['targets'].values()) and all(rep['guards'].values())
    rep['decision'] = 'requires_extra_fold_confirmation' if rep['passes_initial_rule'] else 'retain_existing_default'
    rep['scope'] = 'development read-out calibration; no new training or sealed donors; no independent final test'
    (OUT / 'comparison.json').write_text(json.dumps(rep, indent=2) + '\n')
    print(json.dumps(rep, indent=2))


if __name__ == '__main__':
    main()
