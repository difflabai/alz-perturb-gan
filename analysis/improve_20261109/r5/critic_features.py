"""Frozen final-critic donor fingerprints, with grouped readout controls.

This measures recoverable information, not biological identification. No GAN
weights are updated, and outer held-out donors cannot select a readout.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '3')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
import trainer as T
import generate as G
import arms
import evaluate as E
from inverse_donor import log_profile
from donor_encoder import repeatability

ALPHAS = (.01, .1, 1., 10., 100.)


def normalize(train, test):
    mean = train.mean(0)
    sd = train.std(0)
    sd = np.where(sd > 1e-8, sd, 1.)
    return (train-mean)/sd, (test-mean)/sd


def ridge_grid(train_x, train_y, test_x, alphas=ALPHAS):
    """Dual ridge: ||Y-XB/sqrt(p)||^2/n + alpha*||B||^2.

    All centering/scaling comes from this training subset. Eigen decomposition
    is shared over the fixed alpha grid, with no rank selection.
    """
    x, test = normalize(np.asarray(train_x, float), np.asarray(test_x, float))
    mean = train_y.mean(0)
    y = train_y-mean
    p = max(1, x.shape[1])
    eigenvalues, vectors = np.linalg.eigh(x @ x.T / p)
    right = vectors.T @ y
    left = (test @ x.T / p) @ vectors
    return np.stack([mean + (left/(np.maximum(eigenvalues, 0.) + len(x)*alpha)) @ right
                     for alpha in alphas])


def fit_readout(x, y, groups, train_mask):
    """The mask guards every normalization, target mean and model selection.

    Both cell directions of a donor leave together. Predictions for all rows
    are made only after selecting alpha from TRAIN-donor out-of-fold errors.
    """
    train_mask = np.asarray(train_mask, bool)
    groups = np.asarray(groups)
    if len(set(groups[train_mask])) < 3:
        raise ValueError('readout needs at least three training donors')
    errors = []
    for donor in sorted(set(groups[train_mask])):
        validation = train_mask & (groups == donor)
        inner_train = train_mask & (groups != donor)
        predictions = ridge_grid(x[inner_train], y[inner_train], x[validation])
        errors.append(np.sqrt(np.mean((predictions-y[validation])**2, axis=2)).mean(1))
    cv = np.mean(errors, axis=0)
    chosen = int(np.argmin(cv))
    prediction = ridge_grid(x[train_mask], y[train_mask], x, (ALPHAS[chosen],))[0]
    return prediction, {'alpha': ALPHAS[chosen], 'alphas': list(ALPHAS),
                        'inner_train_donor_rmse': cv.tolist(),
                        'n_train_donors': len(errors)}


def feature_diagnostics(x, stage, train, held):
    """Train-normalized, paired A/B features; no mean-activation correlation."""
    train = np.asarray(train, bool)
    stage_effect = np.linalg.lstsq(stage[train], x[train], rcond=None)[0]
    residual = x-stage @ stage_effect
    _, centered = normalize(x[train], x)
    _, centered_residual = normalize(residual[train], residual)
    values = {}
    for label, h in [('train', False), ('heldout', True)]:
        rows = np.flatnonzero(np.asarray(held) == h)
        if len(rows) % 2 or not np.array_equal(rows[::2]+1, rows[1::2]):
            raise ValueError('expected adjacent paired directions')
        values[label] = {
            'split_cell_repeatability': repeatability(centered[rows[::2]], centered[rows[1::2]]),
            'stage_residual_repeatability': repeatability(centered_residual[rows[::2]], centered_residual[rows[1::2]]),
            'stage_residual_variance': float(np.var(residual[rows], axis=0).mean())}
    return values


@torch.no_grad()
def collect(ctx, real, learned, untrained, cells=64, depth=False):
    output = {key: [] for key in ('learned', 'untrained', 'full_summary', 'pseudobulk', 'conditions')}
    targets, inputs, groups, held, counts_per_input = [], [], [], [], []
    for i, donor in enumerate(real.donors):
        rows = np.random.default_rng(E.MX.stable_seed('encoder-cell-halves', donor)).permutation(len(real.libs[i]))
        halves = np.array_split(rows, 2)
        c = torch.tensor(ctx.condition_matrix(real.raw_c[i:i+1]), dtype=torch.float32)
        categorical_stage = np.eye(4)[int(round(float(c[0, 3])*3))]
        for direction, half in enumerate(halves):
            key = ('encoder-depth', donor, cells) if depth else ('encoder-input', donor)
            n = cells if depth else min(cells, len(half))
            chosen = np.random.default_rng(E.MX.stable_seed(*key)).choice(half, n, replace=n > len(half))
            counts = torch.tensor(real.cells[i][chosen].toarray())[None]
            features = ctx.features.fake(T.standardize_counts(counts, ctx.mean, ctx.sd), counts, counts)
            combined = torch.cat([features, c], 1)
            output['learned'].append(learned.phi(combined)[0].numpy())
            output['untrained'].append(untrained.phi(combined)[0].numpy())
            output['full_summary'].append(features[0].numpy())
            output['pseudobulk'].append(log_profile(real.cells[i][chosen]))
            output['conditions'].append(categorical_stage)
            targets.append(log_profile(real.cells[i][halves[1-direction]]))
            inputs.append(log_profile(real.cells[i][chosen]))
            groups.append(i)
            held.append(bool(real.is_heldout[i]))
            counts_per_input.append(n)
    return ({key: np.array(value, dtype=float) for key, value in output.items()},
            np.array(targets), np.array(inputs), np.array(groups), np.array(held), counts_per_input)


def run(fold, seed, out_root):
    T.check_out_root(out_root)
    out = Path(out_root)/f'fold{fold}/seed{seed}'
    if (out/'evaluation.json').exists():
        raise FileExistsError(out)
    fit = T.RUNS/f'fits_r3/pbulk/hvg2000/fold{fold}/seed{seed}'
    cache = T.RUNS/f'caches/hvg2000/fold{fold}'
    cfg = json.loads((fit/'config.json').read_text())
    if cfg['flags'] != ['pbulk']:
        raise ValueError('fixed probe expects pbulk only')
    started = time.monotonic()
    ctx = T.Context(cache, 'cpu', tuple(cfg['flags']))
    final = torch.load(fit/'final.pt', map_location='cpu', weights_only=False)
    G.check_final(ctx, cfg, final)
    torch.manual_seed(T.training_streams(fold, seed, root=cfg['stream_root'])['critic_init'])
    untrained = arms.WideVectorCritic(ctx.features.n_features, h=128).eval().requires_grad_(False)
    learned = arms.WideVectorCritic(ctx.features.n_features, h=128).eval().requires_grad_(False)
    learned.load_state_dict(final['policy']['models']['critic'])
    original = {k: v.clone() for k, v in learned.state_dict().items()}
    del final
    real = E.load_real(cache)
    x, y, inputs, groups, held, input_cells = collect(ctx, real, learned, untrained)
    train = ~held
    stage = x['conditions']
    predictions = {'train_mean': np.broadcast_to(y[train].mean(0), y.shape), 'input_pseudobulk': inputs}
    readouts = {}
    for name in ('conditions', 'pseudobulk', 'full_summary', 'untrained', 'learned'):
        design = x[name] if name == 'conditions' else np.concatenate([x[name], stage], axis=1)
        predictions[name], readouts[name] = fit_readout(design, y, groups, train)
        print(json.dumps({'readout': name, **readouts[name]}), flush=True)
    diagnostics = {name: feature_diagnostics(x[name], stage, train, held) for name in ('learned', 'untrained')}
    depth = {}
    for k in (32, 128):
        dx, _, _, _, _, _ = collect(ctx, real, learned, untrained, cells=k, depth=True)
        depth[str(k)] = {}
        for name in ('learned', 'untrained'):
            sd = x[name][train].std(0)
            sd = np.where(sd > 1e-8, sd, 1.)
            differences = np.mean(((dx[name]-x[name])/sd)**2, axis=1)
            depth[str(k)][name] = {label: float(differences[held == value].mean())
                                 for label, value in [('train', False), ('heldout', True)]}
    errors = {name: np.sqrt(np.mean((p-y)**2, axis=1)) for name, p in predictions.items()}
    stage_gene = stage @ np.linalg.lstsq(stage[train], y[train], rcond=None)[0]
    records = []
    for i, donor in enumerate(real.donors):
        pair = slice(2*i, 2*i+2)
        records.append({'donor': str(donor), 'heldout': bool(real.is_heldout[i]),
                        'rmse': {name: float(value[pair].mean()) for name, value in errors.items()},
                        'input_cells': input_cells[2*i:2*i+2],
                        'learned_predicted_profile': predictions['learned'][pair].mean(0).tolist(),
                        'stage_gene_profile': stage_gene[pair].mean(0).tolist(),
                        'learned_phi_a': x['learned'][2*i].tolist(),
                        'learned_phi_b': x['learned'][2*i+1].tolist()})
    aggregate = {}
    for label, value in [('train', False), ('heldout', True)]:
        mask = held == value
        aggregate[label] = {'rmse': {name: float(error[mask].mean()) for name, error in errors.items()},
                            'n_donors': int(mask.sum()//2)}
        target_residual = (y-stage_gene)[mask]
        pred_residual = (predictions['learned']-stage_gene)[mask]
        aggregate[label]['stage_residual_gene_correlation'] = float(np.corrcoef(
            target_residual.ravel(), pred_residual.ravel())[0, 1])
        aggregate[label]['stage_residual_gene_r2'] = float(1.-np.sum((target_residual-pred_residual)**2)/
                                                        max(1e-12, np.sum(target_residual**2)))
    assert all(torch.equal(value, original[key]) for key, value in learned.state_dict().items())
    assert all(p.grad is None for p in learned.parameters())
    result = {'fold': fold, 'seed': seed, 'aggregate': aggregate, 'readouts': readouts,
              'feature_diagnostics': diagnostics, 'depth_standardized_feature_mse': depth,
              'records': records, 'GAN_weights_unchanged': True,
              'fit_final_sha256': T.sha(fit/'final.pt'), 'source_sha256': T.sha(__file__),
              'protocol_sha256': T.sha(Path(__file__).parents[1]/'PROTOCOL_CRITIC_FEATURES_20261113.txt'),
              'wall_seconds': time.monotonic()-started,
              'scope': 'frozen final-critic reconstruction audit; no identified biology or native generator latent'}
    T.dump(out/'evaluation.json', result)
    print(json.dumps({key: result[key] for key in ('aggregate', 'feature_diagnostics', 'wall_seconds')}, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fold', type=int, choices=range(3), required=True)
    p.add_argument('--seed', type=int, choices=(0, 1), required=True)
    p.add_argument('--out-root', type=Path, default=T.RUNS/'smoke_r5/architecture_pass/critic_features')
    a = p.parse_args()
    torch.set_num_threads(3)
    run(a.fold, a.seed, a.out_root)
